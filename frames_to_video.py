#!/usr/bin/env python3
"""
Frames to Transparent Video
============================
Assembles RGBA PNG frames into transparent video files.
Supports HEVC with alpha (MOV, macOS/iOS native) and ProRes 4444 with alpha (MOV).

Note: VP9 WebM alpha encoding is broken in ffmpeg 8.x (alpha_mode metadata is set
but alpha data is not actually encoded). HEVC via videotoolbox is used instead for
the lightweight web-friendly format.

Usage:
    python frames_to_video.py output/chimp_cleaned/
    python frames_to_video.py output/chimp_cleaned/ --format hevc
    python frames_to_video.py output/chimp_cleaned/ --format prores
    python frames_to_video.py output/chimp_cleaned/ --format both --fps 24

Requirements:
    ffmpeg (system, with videotoolbox support on macOS), Pillow
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


def load_metadata(frame_dir):
    """Attempt to load metadata.json from the frame directory."""
    meta_path = frame_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def discover_frames(frame_dir):
    """Find and validate frame PNG files in the directory."""
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        print(f"Error: No frame_*.png files found in {frame_dir}")
        sys.exit(1)

    # Check first frame for RGBA
    test_img = Image.open(frames[0])
    if test_img.mode != "RGBA":
        print(f"Warning: Frames are {test_img.mode}, expected RGBA. Transparency may not work.")

    width, height = test_img.size
    print(f"  Found {len(frames)} frames at {width}x{height}")
    return frames


def encode_hevc_alpha(frame_pattern, output_path, fps):
    """Encode RGBA PNGs to HEVC MOV with alpha via macOS videotoolbox."""
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-vf",
        "premultiply=inplace=1",
        "-c:v",
        "hevc_videotoolbox",
        "-q:v",
        "55",
        "-alpha_quality",
        "0.75",
        "-tag:v",
        "hvc1",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr[-500:]}")
        return False

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  -> {output_path} ({size_mb:.1f}MB)")
    return True


def encode_prores_4444(frame_pattern, output_path, fps):
    """Encode RGBA PNGs to ProRes 4444 MOV with alpha channel."""
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4",
        "-pix_fmt",
        "yuva444p10le",
        "-vendor",
        "apl0",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr[-500:]}")
        return False

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  -> {output_path} ({size_mb:.1f}MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Assemble RGBA PNG frames into transparent video (HEVC alpha and/or ProRes 4444)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s output/chimp_cleaned/                        # both formats, auto fps
  %(prog)s output/chimp_cleaned/ --format hevc          # HEVC alpha MOV only
  %(prog)s output/chimp_cleaned/ --format prores        # ProRes 4444 only
  %(prog)s output/chimp_cleaned/ --fps 24               # explicit fps
        """,
    )
    parser.add_argument("input", help="Directory of RGBA PNG frames")
    parser.add_argument(
        "--format",
        choices=["hevc", "prores", "both"],
        default="both",
        help="Output format: hevc (HEVC+alpha MOV), prores (ProRes 4444 MOV), both (default: both)",
    )
    parser.add_argument("--fps", type=float, default=None, help="Frame rate (auto-detected from metadata.json)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--output-name", default=None, help="Base name for output files")

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}")
        sys.exit(1)

    # Check ffmpeg
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Install via: brew install ffmpeg")
        sys.exit(1)

    # Discover frames
    print(f"\n{'=' * 60}")
    print("Discovering frames")
    print(f"{'=' * 60}")
    discover_frames(input_dir)

    # Load metadata for fps
    metadata = load_metadata(input_dir)
    fps = args.fps
    if fps is None:
        if metadata and "fps" in metadata:
            fps = metadata["fps"]
            print(f"  FPS from metadata: {fps}")
        else:
            print("Error: Cannot determine fps. Use --fps or provide metadata.json")
            sys.exit(1)

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output name
    base_name = args.output_name
    if not base_name:
        dir_stem = input_dir.name
        for suffix in ("_cleaned", "_frames"):
            if dir_stem.endswith(suffix):
                dir_stem = dir_stem[: -len(suffix)]
        base_name = f"{dir_stem}_alpha"

    # Frame pattern for ffmpeg
    frame_pattern = str(input_dir / "frame_%04d.png")

    # Encode
    if args.format in ("hevc", "both"):
        print(f"\n{'=' * 60}")
        print("Encoding HEVC with alpha (MOV)")
        print(f"{'=' * 60}")
        hevc_path = output_dir / f"{base_name}_hevc.mov"
        encode_hevc_alpha(frame_pattern, hevc_path, fps)

    if args.format in ("prores", "both"):
        print(f"\n{'=' * 60}")
        print("Encoding ProRes 4444 (MOV with alpha)")
        print(f"{'=' * 60}")
        mov_path = output_dir / f"{base_name}_prores.mov"
        encode_prores_4444(frame_pattern, mov_path, fps)

    # Summary
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    if args.format in ("hevc", "both"):
        print(f"  HEVC:   {output_dir / f'{base_name}_hevc.mov'}")
    if args.format in ("prores", "both"):
        print(f"  ProRes: {output_dir / f'{base_name}_prores.mov'}")


if __name__ == "__main__":
    main()
