#!/usr/bin/env python3
"""
Transparency Test
=================
Composites a transparent video (or RGBA PNG) over a checkerboard background
to visually verify that the alpha channel is working correctly.

Extracts a frame from the video, draws a colored checkerboard behind it,
and saves the result as a PNG. Can also generate a short composited video.

Usage:
    python test_transparency.py output/chimpAnimation_loop_alpha_hevc.mov
    python test_transparency.py output/chimpAnimation_loop_alpha_hevc.mov --frame 10
    python test_transparency.py output/chimpAnimation_loop_alpha_hevc.mov --video
    python test_transparency.py output/some_frame.png

Requirements:
    ffmpeg, opencv-python, numpy, Pillow
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def make_checkerboard(width, height, square_size=64, color1=(255, 105, 180), color2=(50, 50, 50)):
    """Create a colored checkerboard pattern."""
    board = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            gy = y // square_size
            gx = x // square_size
            color = color1 if (gx + gy) % 2 == 0 else color2
            y_end = min(y + square_size, height)
            x_end = min(x + square_size, width)
            board[y:y_end, x:x_end] = color
    return board


def composite_over_checkerboard(rgba, square_size=64):
    """Composite an RGBA image over a checkerboard."""
    h, w = rgba.shape[:2]
    checker = make_checkerboard(w, h, square_size)

    # Alpha blend: out = fg * alpha + bg * (1 - alpha)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = rgba[:, :, :3].astype(np.float32)
    bg = checker.astype(np.float32)

    composited = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    return composited


def extract_rgba_frame(video_path, frame_num):
    """Extract a single RGBA frame from a video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_num})",
        "-vframes", "1",
        "-pix_fmt", "rgba",
        "-f", "rawvideo",
        "-"
    ]
    # First get dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True
    )
    if probe.returncode != 0:
        print(f"Error probing video: {probe.stderr}")
        sys.exit(1)

    dims = probe.stdout.strip().split(",")
    width, height = int(dims[0]), int(dims[1])

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"Error extracting frame: {result.stderr.decode()[-500:]}")
        sys.exit(1)

    raw = result.stdout
    expected = width * height * 4
    if len(raw) < expected:
        print(f"Error: got {len(raw)} bytes, expected {expected}")
        sys.exit(1)

    frame = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 4)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Test transparency by compositing over a checkerboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s output/video_hevc.mov                    # test frame 0
  %(prog)s output/video_hevc.mov --frame 10         # test frame 10
  %(prog)s output/video_hevc.mov --video             # render full composited video
  %(prog)s output/some_frame.png                     # test a single RGBA PNG
        """,
    )
    parser.add_argument("input", help="Transparent video (MOV) or RGBA PNG file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to extract (default: 0)")
    parser.add_argument("--square-size", type=int, default=64, help="Checkerboard square size (default: 64)")
    parser.add_argument("--video", action="store_true", help="Render a full composited video (MP4)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory (default: ./output)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    if input_path.suffix.lower() == ".png":
        # Direct RGBA PNG
        rgba = np.array(Image.open(input_path))
        if rgba.shape[2] != 4:
            print(f"Error: Image is not RGBA (has {rgba.shape[2]} channels)")
            sys.exit(1)
        print(f"Input: {input_path} ({rgba.shape[1]}x{rgba.shape[0]}, RGBA PNG)")
    else:
        # Video — extract frame
        print(f"Extracting frame {args.frame} from {input_path}...")
        rgba = extract_rgba_frame(input_path, args.frame)
        print(f"  Frame: {rgba.shape[1]}x{rgba.shape[0]}, RGBA")

    # Report alpha stats
    alpha = rgba[:, :, 3]
    opaque = np.count_nonzero(alpha == 255)
    transparent = np.count_nonzero(alpha == 0)
    semi = alpha.size - opaque - transparent
    print(f"  Alpha stats: {opaque} opaque, {transparent} transparent, {semi} semi-transparent")
    opaque_pct = opaque / alpha.size * 100
    transparent_pct = transparent / alpha.size * 100
    print(f"  Alpha coverage: {opaque_pct:.1f}% opaque, {transparent_pct:.1f}% transparent")

    # Composite single frame
    composited = composite_over_checkerboard(rgba, args.square_size)
    out_png = output_dir / f"{stem}_transparency_test.png"
    Image.fromarray(composited).save(out_png)
    print(f"  Saved: {out_png}")

    # Optionally render full composited video
    if args.video and input_path.suffix.lower() != ".png":
        print("\nRendering full composited video...")
        out_mp4 = output_dir / f"{stem}_transparency_test.mp4"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Generate checkerboard image
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(input_path)],
                capture_output=True, text=True
            )
            dims = probe.stdout.strip().split(",")
            width, height = int(dims[0]), int(dims[1])

            checker_path = tmpdir / "checker.png"
            checker = make_checkerboard(width, height, args.square_size)
            Image.fromarray(checker).save(checker_path)

            # Use ffmpeg overlay filter to composite
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(checker_path),
                "-i", str(input_path),
                "-filter_complex",
                "[0:v][1:v]overlay=0:0:shortest=1,format=yuv420p",
                "-c:v", "libx264", "-crf", "18",
                "-an", str(out_mp4),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Error: {result.stderr[-500:]}")
                sys.exit(1)

            import os
            size_mb = os.path.getsize(out_mp4) / (1024 * 1024)
            print(f"  Saved: {out_mp4} ({size_mb:.1f}MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
