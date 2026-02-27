#!/usr/bin/env python3
"""
Video Background Removal
========================
Frame-by-frame AI background removal from video using rembg.
Streams frames one at a time to manage memory at high resolutions.
Optionally applies temporal alpha smoothing to reduce inter-frame flickering.

Usage:
    python video_remove_bg.py input.mp4
    python video_remove_bg.py input.mp4 --model birefnet-general
    python video_remove_bg.py input.mp4 --model birefnet-general --temporal-smooth 5
    python video_remove_bg.py input.mp4 --no-temporal-smooth

Requirements:
    rembg, opencv-python, numpy, Pillow
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove


def get_video_info(path):
    """Get video metadata without loading frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, total_frames


def remove_bg_streaming(input_path, session, output_dir, total_frames):
    """Remove background from each frame, streaming one at a time."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: could not read frame {i}, stopping at {i} frames")
            break

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Remove background (returns RGBA PIL Image)
        result = remove(pil_img, session=session)

        # Save as RGBA PNG
        out_path = output_dir / f"frame_{i:04d}.png"
        result.save(out_path)

        print(f"  Frame {i + 1}/{total_frames}", end="\r", flush=True)

    cap.release()
    print()  # newline after \r progress


def temporal_smooth_alpha(frame_dir, window):
    """Smooth alpha channels across adjacent frames to reduce flicker.

    Pre-loads all alpha channels, applies Gaussian-weighted averaging,
    then writes smoothed alphas back. Only the alpha channel is smoothed;
    RGB is preserved.
    """
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    n = len(frame_paths)
    if n == 0 or window <= 1:
        return

    half = window // 2
    # Gaussian kernel
    kernel = np.exp(-0.5 * ((np.arange(window) - half) / max(window / 4, 1)) ** 2)
    kernel /= kernel.sum()

    print(f"  Temporal smoothing (window={window}) across {n} frames...")

    # Pre-load all alpha channels
    print("  Loading alpha channels...")
    alphas = []
    for fp in frame_paths:
        img = np.array(Image.open(fp))
        alphas.append(img[:, :, 3].astype(np.float32))

    # Compute smoothed alphas
    smoothed_alphas = []
    for i in range(n):
        win_start = max(0, i - half)
        win_end = min(n - 1, i + half)

        smoothed = np.zeros_like(alphas[0])
        weight_sum = 0.0
        for j in range(win_start, win_end + 1):
            k_idx = j - (i - half)
            if 0 <= k_idx < window:
                w = kernel[k_idx]
                smoothed += w * alphas[j]
                weight_sum += w

        if weight_sum > 0:
            smoothed /= weight_sum
        smoothed_alphas.append(np.clip(smoothed, 0, 255).astype(np.uint8))

    # Write smoothed alphas back
    for i, fp in enumerate(frame_paths):
        img = np.array(Image.open(fp))
        img[:, :, 3] = smoothed_alphas[i]
        Image.fromarray(img).save(fp)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  Smoothed {i + 1}/{n}", end="\r", flush=True)

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Frame-by-frame video background removal using rembg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4                                    # default model (u2net)
  %(prog)s input.mp4 --model birefnet-general           # higher quality model
  %(prog)s input.mp4 --model birefnet-general --temporal-smooth 7
  %(prog)s input.mp4 --no-temporal-smooth               # skip alpha smoothing
        """,
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--model", default="birefnet-general", help="rembg model name (default: birefnet-general)")
    parser.add_argument(
        "--temporal-smooth",
        type=int,
        default=5,
        help="Alpha channel temporal smoothing window size, must be odd (default: 5, use 1 to disable)",
    )
    parser.add_argument("--no-temporal-smooth", action="store_true", help="Disable temporal alpha smoothing")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--output-name", default=None, help="Base name for output frame directory")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Validate temporal smooth
    smooth_window = 1 if args.no_temporal_smooth else args.temporal_smooth
    if smooth_window > 1 and smooth_window % 2 == 0:
        smooth_window += 1
        print(f"Note: temporal-smooth adjusted to odd number: {smooth_window}")

    # Setup output
    output_base = Path(args.output_dir)
    dir_name = args.output_name or f"{input_path.stem}_frames"
    frame_dir = output_base / dir_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Get video info
    fps, width, height, total_frames = get_video_info(str(input_path))
    print(f"Input: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Model: {args.model}")
    print(f"Output: {frame_dir}")

    # Create rembg session (loaded once, reused for all frames)
    print(f"\n{'=' * 60}")
    print(f"Loading model: {args.model}")
    print(f"{'=' * 60}")
    session = new_session(args.model)

    # Process frames
    print(f"\n{'=' * 60}")
    print(f"Removing backgrounds ({total_frames} frames)")
    print(f"{'=' * 60}")
    remove_bg_streaming(str(input_path), session, frame_dir, total_frames)

    # Temporal smoothing
    if smooth_window > 1:
        print(f"\n{'=' * 60}")
        print(f"Temporal alpha smoothing (window={smooth_window})")
        print(f"{'=' * 60}")
        temporal_smooth_alpha(frame_dir, smooth_window)

    # Save metadata
    metadata = {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": total_frames,
        "model": args.model,
        "temporal_smooth": smooth_window,
        "source": str(input_path),
    }
    meta_path = frame_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")
    print(f"Frames: {frame_dir}")
    print(f"Metadata: {meta_path}")
    frame_count = len(list(frame_dir.glob("frame_*.png")))
    print(f"Total frames saved: {frame_count}")


if __name__ == "__main__":
    main()
