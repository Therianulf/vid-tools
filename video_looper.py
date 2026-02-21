#!/usr/bin/env python3
"""
True Video Looper
=================
Creates seamlessly looping videos by finding or specifying a loop point
where the end frame visually matches the start frame, then trimming and
optionally slowing the video.

Outputs both a simple (PTS-stretched) and frame-interpolated version
so you can compare quality.

Usage:
    python video_looper.py input.mp4 --scan                              # find loop points
    python video_looper.py input.mp4 --loop-second 4.7                   # create loop at 4.7s
    python video_looper.py input.mp4 --loop-second 4.7 --slowdown 2.0    # loop + slow down
    python video_looper.py input.mp4 --loop-second 4.7 --loop-frame 2    # sub-second precision

Requirements:
    opencv-python, numpy, torch, ffmpeg (system)
"""

import argparse
import os
import sys
import subprocess
import tempfile
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


def load_video_info(path):
    """Get video metadata without loading all frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    cap.release()
    return fps, width, height, total_frames, duration


def read_frame_at(path, frame_number):
    """Read a specific frame from a video file by seeking."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_number}")
    return frame


def frame_similarity(frame_a, frame_b):
    """Compute visual similarity score between two frames (0-1).

    Uses histogram correlation (40%) + inverse MSE on downscaled grayscale (60%).
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # Histogram comparison (global color distribution)
    hist_a = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [256], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    hist_score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

    # Structural comparison: resize to small size, compute MSE
    small_a = cv2.resize(gray_a, (64, 64)).astype(np.float64)
    small_b = cv2.resize(gray_b, (64, 64)).astype(np.float64)
    mse = np.mean((small_a - small_b) ** 2)
    mse_score = 1.0 / (1.0 + mse / 1000.0)

    return 0.4 * hist_score + 0.6 * mse_score


def scan_for_loop_points(video_path, fps, total_frames, start_second, count):
    """Scan video for frames most similar to frame 0. Returns ranked list."""
    frame_0 = read_frame_at(video_path, 0)
    start_frame = int(start_second * fps)

    if start_frame >= total_frames:
        print(f"Warning: scan-start ({start_second}s) is beyond video end, scanning from 50%")
        start_frame = total_frames // 2

    print(f"Scanning frames {start_frame}-{total_frames - 1} against frame 0...")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    scores = []
    for frame_num in range(start_frame, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        score = frame_similarity(frame_0, frame)
        scores.append((frame_num, frame_num / fps, score))

        if (frame_num - start_frame) % 30 == 0:
            pct = (frame_num - start_frame) / (total_frames - start_frame) * 100
            print(f"  Progress: {pct:.0f}% (frame {frame_num})")

    cap.release()

    # Sort by similarity score descending
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:count]


def scan_for_shifts(video_path, fps, total_frames, start_second, count):
    """Scan video for frames where the biggest visual shift happens.

    Compares each frame to its predecessor and ranks by largest difference —
    finds natural transition moments that could mask a loop seam.
    """
    start_frame = max(1, int(start_second * fps))

    if start_frame >= total_frames:
        print(f"Warning: scan-start ({start_second}s) is beyond video end, scanning from 10%")
        start_frame = max(1, total_frames // 10)

    print(f"Scanning frames {start_frame}-{total_frames - 1} for visual shifts...")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []

    scores = []
    for frame_num in range(start_frame, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Compute frame-to-frame difference (inverse similarity = shift magnitude)
        diff = 1.0 - frame_similarity(prev_frame, frame)
        scores.append((frame_num, frame_num / fps, diff))
        prev_frame = frame

        if (frame_num - start_frame) % 30 == 0:
            pct = (frame_num - start_frame) / (total_frames - start_frame) * 100
            print(f"  Progress: {pct:.0f}% (frame {frame_num})")

    cap.release()

    # Sort by shift magnitude descending
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:count]


def save_comparison(frame_first, frame_loop, output_path):
    """Save side-by-side comparison image of first and loop frames."""
    # Add labels
    labeled_first = frame_first.copy()
    labeled_loop = frame_loop.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(labeled_first, "Frame 0 (start)", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(labeled_loop, "Loop frame (end)", (10, 30), font, 0.8, (0, 255, 0), 2)

    combined = np.hstack([labeled_first, labeled_loop])
    cv2.imwrite(str(output_path), combined)
    print(f"Comparison: {output_path}")


def interpolate_frames_optical_flow(frame1, frame2, num_intermediate=3):
    """Generate intermediate frames using Farneback optical flow.

    Reuses the motion-compensated approach from video_slowmo_loop.py.
    """
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

        flow_t_from_1 = flow * t
        flow_t_from_2 = flow * (t - 1)

        map_x_1 = np.float32(np.tile(np.arange(w), (h, 1))) + flow_t_from_1[..., 0]
        map_y_1 = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w))) + flow_t_from_1[..., 1]
        warped_1 = cv2.remap(frame1, map_x_1, map_y_1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        map_x_2 = np.float32(np.tile(np.arange(w), (h, 1))) + flow_t_from_2[..., 0]
        map_y_2 = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w))) + flow_t_from_2[..., 1]
        warped_2 = cv2.remap(frame2, map_x_2, map_y_2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        blended = cv2.addWeighted(warped_1, 1 - t, warped_2, t, 0)
        intermediate.append(blended)

    return intermediate


def crossfade_loop(frames, fade_frames):
    """Cross-fade the tail of the video into the head for a seamless loop.

    Overlaps the last `fade_frames` with the first `fade_frames` by alpha blending,
    so the video smoothly morphs from end back into beginning with no seam.
    The output is shorter by `fade_frames` since the tail overlaps the head.
    """
    total = len(frames)
    if fade_frames >= total // 2:
        fade_frames = total // 3
        print(f"  Fade too long, clamping to {fade_frames} frames")

    print(f"  Cross-fading {fade_frames} frames (tail into head)...")

    # The blended region replaces the first fade_frames of the video
    for i in range(fade_frames):
        alpha = i / fade_frames  # 0.0 at start -> 1.0 at end of fade
        tail_idx = total - fade_frames + i
        head_frame = frames[i].astype(np.float32)
        tail_frame = frames[tail_idx].astype(np.float32)
        # Blend: start heavy on tail, transition to head
        frames[i] = ((1 - alpha) * tail_frame + alpha * head_frame).astype(np.uint8)

    # Trim off the tail (it's now blended into the head)
    frames = frames[:total - fade_frames]
    print(f"  Cross-fade complete: {total} -> {len(frames)} frames")
    return frames


def load_and_trim_frames(input_path, loop_frame):
    """Load video frames from frame 0 up to loop_frame (inclusive)."""
    cap = cv2.VideoCapture(input_path)
    frames = []
    for i in range(loop_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  Loaded {len(frames)} frames")
    return frames


def slowdown_duplicate(frames, factor):
    """Simple frame duplication slowdown. Each frame repeated `factor` times."""
    output = []
    for frame in frames:
        for _ in range(factor):
            output.append(frame)
    print(f"  Duplicated {len(frames)} frames {factor}x -> {len(output)} frames")
    return output


def slowdown_interpolate(frames, factor):
    """Optical flow interpolated slowdown."""
    num_intermediate = factor - 1
    total_pairs = len(frames) - 1
    output = []

    for i in range(total_pairs):
        output.append(frames[i])
        if i % 10 == 0:
            pct = (i / total_pairs) * 100
            print(f"  Interpolating: {pct:.0f}% ({i}/{total_pairs})")

        intermediates = interpolate_frames_optical_flow(
            frames[i], frames[i + 1],
            num_intermediate=num_intermediate
        )
        output.extend(intermediates)

    output.append(frames[-1])
    print(f"  Interpolating: 100% -- {len(output)} total frames")
    return output


def encode_h264(frames, fps, output_path):
    """Write frames to H.264 MP4 via temp file + ffmpeg."""
    height, width = frames[0].shape[:2]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()

        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "slow",
            "-an", "-movflags", "+faststart",
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            sys.exit(1)
    finally:
        os.unlink(tmp_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  -> {output_path} ({size_mb:.1f}MB)")




def main():
    parser = argparse.ArgumentParser(
        description="Create seamlessly looping videos by finding or specifying a loop cut point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4 --scan                              # find best loop points
  %(prog)s input.mp4 --loop-second 4.7                   # create loop at 4.7s
  %(prog)s input.mp4 --loop-second 4.7 --slowdown 2.0    # loop + 2x slow-motion
  %(prog)s input.mp4 --loop-second 4.7 --loop-frame 2    # sub-second precision
        """
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--scan", action="store_true",
                        help="Scan mode: find and rank best loop-point frames vs frame 0")
    parser.add_argument("--scan-shift", action="store_true",
                        help="Scan mode: find frames with the biggest visual shift (natural transition points)")
    parser.add_argument("--scan-start", type=float, default=None,
                        help="Start scanning from this second (default: 50%% of duration)")
    parser.add_argument("--scan-count", type=int, default=5,
                        help="Number of best candidates to show (default: 5)")
    parser.add_argument("--loop-second", type=float, default=None,
                        help="The second in the video to cut at for the loop point")
    parser.add_argument("--loop-frame", type=int, default=0,
                        help="Frame offset within the loop second for sub-second precision (default: 0)")
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="Slowdown factor, e.g. 2.0 = twice as slow (default: 1.0)")
    parser.add_argument("--fade", type=float, default=1.0,
                        help="Cross-fade duration in seconds for seamless loop transition (default: 1.0)")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--output-name", default=None,
                        help="Base name for output files (default: {input_stem}_loop)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not args.scan and not args.scan_shift and args.loop_second is None:
        print("Error: Must specify --scan, --scan-shift, or --loop-second")
        parser.print_help()
        sys.exit(1)

    # Load video info
    input_path = os.path.abspath(args.input)
    fps, width, height, total_frames, duration = load_video_info(input_path)
    print(f"Input: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {duration:.2f}s")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    basename = args.output_name or f"{Path(args.input).stem}_loop"

    # === SCAN MODE ===
    if args.scan:
        scan_start = args.scan_start if args.scan_start is not None else duration / 2
        print(f"\n{'='*60}")
        print(f"Scanning for loop points (from {scan_start:.1f}s)")
        print(f"{'='*60}")

        candidates = scan_for_loop_points(input_path, fps, total_frames, scan_start, args.scan_count)

        print(f"\n{'='*60}")
        print("Top loop point candidates:")
        print(f"{'='*60}")
        print(f"{'Rank':<6}{'Frame':<10}{'Time':<12}{'Similarity':<12}")
        print("-" * 40)

        frame_0 = read_frame_at(input_path, 0)
        for rank, (frame_num, time_sec, score) in enumerate(candidates, 1):
            print(f"{rank:<6}{frame_num:<10}{time_sec:.3f}s{'':<5}{score:.4f}")

            # Save comparison for top 3
            if rank <= 3:
                loop_frame = read_frame_at(input_path, frame_num)
                comp_path = os.path.join(output_dir, f"{basename}_candidate_{rank}.png")
                save_comparison(frame_0, loop_frame, comp_path)

        print(f"\nUse --loop-second <time> to create a loop at the chosen point.")
        return

    # === SCAN-SHIFT MODE ===
    if args.scan_shift:
        scan_start = args.scan_start if args.scan_start is not None else duration * 0.1
        print(f"\n{'='*60}")
        print(f"Scanning for visual shifts (from {scan_start:.1f}s)")
        print(f"{'='*60}")

        candidates = scan_for_shifts(input_path, fps, total_frames, scan_start, args.scan_count)

        print(f"\n{'='*60}")
        print("Top visual shift candidates:")
        print(f"{'='*60}")
        print(f"{'Rank':<6}{'Frame':<10}{'Time':<12}{'Shift':<12}")
        print("-" * 40)

        for rank, (frame_num, time_sec, shift) in enumerate(candidates, 1):
            print(f"{rank:<6}{frame_num:<10}{time_sec:.3f}s{'':<5}{shift:.4f}")

            # Save before/after for top 3
            if rank <= 3:
                before = read_frame_at(input_path, max(0, frame_num - 1))
                after = read_frame_at(input_path, frame_num)
                comp_path = os.path.join(output_dir, f"{basename}_shift_{rank}.png")
                save_comparison(before, after, comp_path)

        print(f"\nThese frames have the biggest visual change from the previous frame.")
        print(f"Use --loop-second <time> to create a loop at the chosen point.")
        return

    # === LOOP MODE ===
    loop_frame_abs = int(args.loop_second * fps) + args.loop_frame
    loop_time = loop_frame_abs / fps

    if loop_frame_abs >= total_frames:
        print(f"Error: Loop frame {loop_frame_abs} exceeds total frames {total_frames}")
        sys.exit(1)

    slowdown_factor = max(1, int(round(args.slowdown)))

    print(f"\n{'='*60}")
    print(f"Creating loop: frame 0 -> frame {loop_frame_abs} ({loop_time:.3f}s)")
    print(f"Slowdown: {slowdown_factor}x | Cross-fade: {args.fade}s")
    print(f"{'='*60}")

    # Load and trim frames
    print("\nStep 1: Loading frames...")
    frames = load_and_trim_frames(input_path, loop_frame_abs)

    # -- Simple version (frame duplication + crossfade) --
    print(f"\n{'='*60}")
    print("Step 2a: Simple version (frame duplication)")
    print(f"{'='*60}")

    if slowdown_factor > 1:
        simple_frames = slowdown_duplicate(frames, slowdown_factor)
    else:
        simple_frames = list(frames)

    fade_frames = int(args.fade * fps * slowdown_factor)
    simple_frames = crossfade_loop(simple_frames, fade_frames)

    simple_path = os.path.join(output_dir, f"{basename}_simple.mp4")
    print("  Encoding...")
    encode_h264(simple_frames, fps, simple_path)

    # -- Interpolated version (optical flow + crossfade) --
    print(f"\n{'='*60}")
    print("Step 2b: Interpolated version (optical flow)")
    print(f"{'='*60}")

    if slowdown_factor > 1:
        interp_frames = slowdown_interpolate(frames, slowdown_factor)
    else:
        interp_frames = list(frames)

    interp_frames = crossfade_loop(interp_frames, fade_frames)

    interp_path = os.path.join(output_dir, f"{basename}_interp.mp4")
    print("  Encoding...")
    encode_h264(interp_frames, fps, interp_path)

    # Summary
    output_dur = len(simple_frames) / fps
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"Loop point:      frame {loop_frame_abs} ({loop_time:.3f}s)")
    print(f"Slowdown:        {slowdown_factor}x")
    print(f"Cross-fade:      {args.fade}s")
    print(f"Output duration: ~{output_dur:.1f}s")
    print(f"Simple:          {simple_path}")
    print(f"Interpolated:    {interp_path}")


if __name__ == "__main__":
    main()
