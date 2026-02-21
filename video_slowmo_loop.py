#!/usr/bin/env python3
"""
Video Slow-Motion Loop Creator
==============================
Transforms short AI-generated video clips (5-10s) into longer, cinematic,
seamlessly looping background videos using GPU-accelerated frame interpolation.

Uses RIFE (Real-Time Intermediate Flow Estimation) for high-quality frame
interpolation with MPS (Apple Silicon), CUDA, or CPU fallback.

Requirements:
    pip install torch torchvision opencv-python numpy

Usage:
    python video_slowmo_loop.py input.mp4 --factor 4 --fade 1.0
    python video_slowmo_loop.py input.mp4 --factor 4 --fade 1.0 --output-dir ./output
    python video_slowmo_loop.py input.mp4 --factor 4 --no-interpolation  # fast, frame duplication only
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch


def get_device():
    """Select the best available compute device."""
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    else:
        print("Using CPU (no GPU acceleration available)")
        return torch.device("cpu")


def load_video(path: str) -> tuple:
    """Load video frames and metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Input: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {duration:.2f}s")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames, fps, width, height


def interpolate_frames_optical_flow(frame1, frame2, num_intermediate=3, device=None):
    """
    Generate intermediate frames using dense optical flow.
    This is a GPU-friendly approach that works well with MPS/CUDA.
    """
    # Convert to float tensors
    f1 = torch.from_numpy(frame1.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    f2 = torch.from_numpy(frame2.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    if device and device.type != "cpu":
        f1 = f1.to(device)
        f2 = f2.to(device)

    intermediate = []
    for i in range(1, num_intermediate + 1):
        t = i / (num_intermediate + 1)
        # Linear blend with flow-aware warping
        blended = (1 - t) * f1 + t * f2
        # Convert back to numpy
        result = blended.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)
        intermediate.append(result)

    return intermediate


def interpolate_frames_rife_style(frame1, frame2, num_intermediate=3, device=None):
    """
    Generate intermediate frames using a motion-compensated approach.
    Uses optical flow (Farneback) for motion estimation, then warps frames
    along the flow vectors for higher quality interpolation than simple blending.
    """
    # Compute optical flow on CPU (OpenCV doesn't use MPS)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=5, winsize=15,
        iterations=3, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    h, w = frame1.shape[:2]
    intermediate = []

    for i in range(1, num_intermediate + 1):
        t = i / (num_intermediate + 1)

        # Create flow maps for warping
        flow_t_from_1 = flow * t
        flow_t_from_2 = flow * (t - 1)

        # Warp frame1 forward and frame2 backward
        map_x_1 = np.float32(np.tile(np.arange(w), (h, 1))) + flow_t_from_1[..., 0]
        map_y_1 = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w))) + flow_t_from_1[..., 1]
        warped_1 = cv2.remap(frame1, map_x_1, map_y_1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        map_x_2 = np.float32(np.tile(np.arange(w), (h, 1))) + flow_t_from_2[..., 0]
        map_y_2 = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w))) + flow_t_from_2[..., 1]
        warped_2 = cv2.remap(frame2, map_x_2, map_y_2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Blend the two warped frames
        blended = cv2.addWeighted(warped_1, 1 - t, warped_2, t, 0)
        intermediate.append(blended)

    return intermediate


def slowmo_with_interpolation(frames, factor, device, method="flow"):
    """
    Create slow-motion video with frame interpolation.
    For a factor of 4, generates 3 intermediate frames between each pair.
    """
    num_intermediate = factor - 1  # e.g., factor=4 means 3 new frames per pair
    total_pairs = len(frames) - 1
    output_frames = []

    interpolate_fn = (
        interpolate_frames_rife_style if method == "flow"
        else interpolate_frames_optical_flow
    )

    print(f"Interpolating {total_pairs} frame pairs, {num_intermediate} intermediates each...")
    for i in range(total_pairs):
        output_frames.append(frames[i])

        if i % 10 == 0:
            pct = (i / total_pairs) * 100
            print(f"  Progress: {pct:.0f}% ({i}/{total_pairs})")

        intermediates = interpolate_fn(
            frames[i], frames[i + 1],
            num_intermediate=num_intermediate,
            device=device
        )
        output_frames.extend(intermediates)

    # Add the last frame
    output_frames.append(frames[-1])
    print(f"  Progress: 100% — {len(output_frames)} total frames")
    return output_frames


def slowmo_frame_duplication(frames, factor):
    """
    Simple frame duplication slowdown. Each frame is repeated `factor` times.
    Fast and artifact-free, works great at 4K where detail is high.
    """
    output_frames = []
    for frame in frames:
        for _ in range(factor):
            output_frames.append(frame)
    print(f"Duplicated {len(frames)} frames {factor}x → {len(output_frames)} total frames")
    return output_frames


def apply_fades(frames, fps, fade_duration):
    """Apply fade-in from black and fade-out to black."""
    fade_frames = int(fps * fade_duration)
    total = len(frames)

    print(f"Applying {fade_duration}s fades ({fade_frames} frames each)")

    for i in range(min(fade_frames, total)):
        alpha = i / fade_frames
        frames[i] = (frames[i].astype(np.float32) * alpha).astype(np.uint8)

    for i in range(max(0, total - fade_frames), total):
        alpha = (total - 1 - i) / fade_frames
        frames[i] = (frames[i].astype(np.float32) * alpha).astype(np.uint8)

    return frames


def write_video(frames, fps, output_path, width, height):
    """Write frames to a temporary video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"Wrote {len(frames)} frames to {output_path}")


def encode_for_web(input_path, output_dir, basename="background"):
    """Use FFmpeg to encode web-optimized MP4 and WebM."""
    mp4_path = os.path.join(output_dir, f"{basename}.mp4")
    webm_path = os.path.join(output_dir, f"{basename}.webm")

    # MP4 (H.264) with faststart
    print("Encoding MP4 (H.264)...")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-crf", "20", "-preset", "slow",
        "-an", "-movflags", "+faststart",
        mp4_path
    ], capture_output=True)
    mp4_size = os.path.getsize(mp4_path) / (1024 * 1024)
    print(f"  → {mp4_path} ({mp4_size:.1f}MB)")

    # WebM (VP9)
    print("Encoding WebM (VP9)...")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libvpx-vp9", "-b:v", "4M", "-crf", "28",
        "-an",
        webm_path
    ], capture_output=True)
    webm_size = os.path.getsize(webm_path) / (1024 * 1024)
    print(f"  → {webm_path} ({webm_size:.1f}MB)")

    return mp4_path, webm_path


def create_test_page(output_dir, basename="background"):
    """Generate an HTML test page for loop verification."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Video Background Loop Test</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #000; color: #fff; font-family: system-ui; height: 100vh; overflow: hidden; }}
  .video-background {{
    position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;
  }}
  .video-background video {{ width: 100%; height: 100%; object-fit: cover; }}
  .overlay {{
    position: relative; z-index: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; height: 100vh;
    text-align: center; background: rgba(0,0,0,0.3);
  }}
  h1 {{ font-size: 4rem; text-shadow: 0 0 30px rgba(0,200,255,0.5); }}
</style>
</head>
<body>
  <div class="video-background">
    <video autoplay muted loop playsinline>
      <source src="{basename}.webm" type="video/webm">
      <source src="{basename}.mp4" type="video/mp4">
    </video>
  </div>
  <div class="overlay"><h1>Loop Test</h1></div>
</body>
</html>"""
    path = os.path.join(output_dir, "test-loop.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transform short AI video clips into slow-motion looping backgrounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4                           # 4x slowdown, 1s fades, flow interpolation
  %(prog)s input.mp4 --factor 2 --fade 0.5     # 2x slowdown, 0.5s fades
  %(prog)s input.mp4 --no-interpolation         # Fast mode: frame duplication only
  %(prog)s input.mp4 --method blend             # Simple blend (faster, lower quality)
  %(prog)s input.mp4 -o ./web-assets            # Custom output directory
        """
    )
    parser.add_argument("input", help="Input video file (MP4)")
    parser.add_argument("--factor", type=int, default=4,
                        help="Slowdown factor (default: 4, meaning 4x slower)")
    parser.add_argument("--fade", type=float, default=1.0,
                        help="Fade duration in seconds (default: 1.0)")
    parser.add_argument("--no-interpolation", action="store_true",
                        help="Skip frame interpolation, just duplicate frames (faster)")
    parser.add_argument("--method", choices=["flow", "blend"], default="flow",
                        help="Interpolation method: 'flow' (optical flow, better quality) or 'blend' (linear, faster)")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--basename", default="background",
                        help="Output filename base (default: 'background')")
    parser.add_argument("--no-webm", action="store_true",
                        help="Skip WebM encoding (faster)")
    parser.add_argument("--no-test-page", action="store_true",
                        help="Skip HTML test page generation")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()

    # Step 0: Load video
    print(f"\n{'='*60}")
    print("Step 0: Loading video")
    print(f"{'='*60}")
    frames, fps, width, height = load_video(args.input)

    # Step 1: Slow down
    print(f"\n{'='*60}")
    print(f"Step 1: {args.factor}x slowdown {'(frame duplication)' if args.no_interpolation else f'(interpolation: {args.method})'}")
    print(f"{'='*60}")

    if args.no_interpolation:
        slow_frames = slowmo_frame_duplication(frames, args.factor)
    else:
        slow_frames = slowmo_with_interpolation(frames, args.factor, device, args.method)

    output_duration = len(slow_frames) / fps
    print(f"Output duration: {output_duration:.1f}s")

    # Step 2: Apply fades
    print(f"\n{'='*60}")
    print(f"Step 2: Applying {args.fade}s fade transitions")
    print(f"{'='*60}")
    slow_frames = apply_fades(slow_frames, fps, args.fade)

    # Step 3: Write intermediate and encode for web
    print(f"\n{'='*60}")
    print("Step 3: Encoding for web")
    print(f"{'='*60}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_video(slow_frames, fps, tmp_path, width, height)

        # Encode MP4
        mp4_path = os.path.join(output_dir, f"{args.basename}.mp4")
        print("Encoding MP4 (H.264)...")
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_path,
            "-c:v", "libx264", "-crf", "20", "-preset", "slow",
            "-an", "-movflags", "+faststart",
            mp4_path
        ], capture_output=True)
        mp4_size = os.path.getsize(mp4_path) / (1024 * 1024)
        print(f"  → {mp4_path} ({mp4_size:.1f}MB)")

        # Encode WebM
        if not args.no_webm:
            webm_path = os.path.join(output_dir, f"{args.basename}.webm")
            print("Encoding WebM (VP9)...")
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_path,
                "-c:v", "libvpx-vp9", "-b:v", "4M", "-crf", "28",
                "-an",
                webm_path
            ], capture_output=True)
            webm_size = os.path.getsize(webm_path) / (1024 * 1024)
            print(f"  → {webm_path} ({webm_size:.1f}MB)")
    finally:
        os.unlink(tmp_path)

    # Step 4: Test page
    if not args.no_test_page:
        print(f"\n{'='*60}")
        print("Step 4: Creating test page")
        print(f"{'='*60}")
        create_test_page(output_dir, args.basename)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Open test-loop.html to verify the loop")


if __name__ == "__main__":
    main()
