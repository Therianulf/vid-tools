#!/usr/bin/env python3
"""
Speech Bubble Cleanup (Contour-Based)
======================================
Extracts a speech bubble from a video using brightness-based contour detection.
The white border of the speech bubble is detected via thresholding, the largest
external contour is filled to create an alpha mask, and all frames are masked.

No AI/rembg involved — purely geometric, deterministic, and fast.

Usage:
    python bubble_cleanup.py input.mp4
    python bubble_cleanup.py input.mp4 --border-width 8 --smooth-radius 15
    python bubble_cleanup.py input.mp4 --threshold 200

Requirements:
    opencv-python, numpy, Pillow
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def detect_bubble_mask(frame_bgr, threshold, min_area_ratio):
    """Detect the speech bubble contour from the white border in a BGR frame.

    Converts to grayscale, thresholds at the given brightness to find the
    white border, then finds the largest external contour as the bubble boundary.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold to find bright pixels (the white border)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find external contours only (outermost boundary = bubble)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  Error: No contours found at brightness threshold")
        sys.exit(1)

    # Filter by area
    h, w = gray.shape
    total_area = h * w
    min_area = min_area_ratio * total_area
    large_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not large_contours:
        print(f"  Error: No contour meets minimum area ({min_area_ratio * 100:.1f}% of image)")
        largest = max(cv2.contourArea(c) for c in contours)
        print(f"  Largest contour area: {largest / total_area * 100:.1f}%")
        print("  Try lowering --threshold or --min-area-ratio")
        sys.exit(1)

    # Use the largest contour
    bubble_contour = max(large_contours, key=cv2.contourArea)
    area_pct = cv2.contourArea(bubble_contour) / total_area * 100
    print(f"  Found bubble contour: {area_pct:.1f}% of image area")

    if len(large_contours) > 1:
        print(f"  Note: {len(large_contours)} large contours found, using largest")

    # Create filled mask from the contour
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [bubble_contour], -1, 255, -1)

    return mask


def smooth_mask(mask, smooth_radius):
    """Smooth jagged edges of the bubble mask using morphological ops + blur."""
    # Morphological close to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_radius, smooth_radius))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Gaussian blur to soften edges
    blur_size = smooth_radius * 2 + 1
    blurred = cv2.GaussianBlur(closed, (blur_size, blur_size), 0)

    # Re-threshold for crisp edges
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return smoothed


def create_border_mask(smoothed_mask, border_width):
    """Create a ring-shaped border mask around the bubble."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width * 2 + 1, border_width * 2 + 1))
    dilated = cv2.dilate(smoothed_mask, kernel)
    border = dilated & ~smoothed_mask
    return border


def apply_mask_to_frame(frame_bgr, smoothed_mask, border_mask, border_color):
    """Apply the computed masks to a single BGR frame, returning RGBA."""
    h, w = frame_bgr.shape[:2]

    # BGR -> RGBA
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = frame_bgr[:, :, 2]  # R
    rgba[:, :, 1] = frame_bgr[:, :, 1]  # G
    rgba[:, :, 2] = frame_bgr[:, :, 0]  # B

    # Alpha: 255 inside bubble, 0 outside
    rgba[:, :, 3] = smoothed_mask

    # Paint border pixels with the border color at full opacity
    border_pixels = border_mask > 0
    rgba[border_pixels, 0] = border_color[0]
    rgba[border_pixels, 1] = border_color[1]
    rgba[border_pixels, 2] = border_color[2]
    rgba[border_pixels, 3] = 255

    return rgba


def main():
    parser = argparse.ArgumentParser(
        description="Extract speech bubble from video using contour-based masking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4                                  # defaults
  %(prog)s input.mp4 --border-width 8                 # thicker border
  %(prog)s input.mp4 --smooth-radius 15               # smoother edges
  %(prog)s input.mp4 --threshold 180                  # lower brightness threshold
  %(prog)s input.mp4 --output-name my_bubble          # custom output name
        """,
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--border-width", type=int, default=8, help="Black border width in pixels (default: 8)")
    parser.add_argument("--smooth-radius", type=int, default=15, help="Morphological smoothing radius (default: 15)")
    parser.add_argument("--border-color", default="0,0,0", help="Border color as R,G,B (default: 0,0,0 = black)")
    parser.add_argument(
        "--threshold", type=int, default=200, help="Brightness threshold for white border (default: 200)"
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.05,
        help="Minimum contour area as fraction of image (default: 0.05)",
    )
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--output-name", default=None, help="Base name for output frame directory")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Parse border color
    try:
        border_color = tuple(int(c) for c in args.border_color.split(","))
        if len(border_color) != 3:
            raise ValueError
    except ValueError:
        print("Error: --border-color must be R,G,B (e.g., 0,0,0)")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input: {input_path} ({width}x{height} @ {fps:.1f}fps, {total_frames} frames)")

    # Setup output
    output_base = Path(args.output_dir)
    dir_name = args.output_name or f"{input_path.stem}_cleaned"
    out_dir = output_base / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Read frame 0 and detect bubble contour
    print(f"\n{'=' * 60}")
    print(f"Step 1: Detecting speech bubble from frame 0 (threshold={args.threshold})")
    print(f"{'=' * 60}")

    ret, frame0 = cap.read()
    if not ret:
        print("Error: Cannot read frame 0")
        sys.exit(1)

    raw_mask = detect_bubble_mask(frame0, args.threshold, args.min_area_ratio)

    # Step 2: Smooth the mask
    print(f"\n{'=' * 60}")
    print(f"Step 2: Smoothing outline (radius={args.smooth_radius})")
    print(f"{'=' * 60}")
    smoothed = smooth_mask(raw_mask, args.smooth_radius)
    print(f"  Raw mask pixels: {np.count_nonzero(raw_mask)}")
    print(f"  Smoothed mask pixels: {np.count_nonzero(smoothed)}")

    # Step 3: Create border
    print(f"\n{'=' * 60}")
    print(f"Step 3: Creating border (width={args.border_width}, color={border_color})")
    print(f"{'=' * 60}")
    border = create_border_mask(smoothed, args.border_width)
    print(f"  Border pixels: {np.count_nonzero(border)}")

    # Save diagnostic masks
    diag_mask_path = out_dir / "bubble_mask.png"
    Image.fromarray(smoothed).save(diag_mask_path)
    print(f"  Saved diagnostic mask: {diag_mask_path}")

    diag_border_path = out_dir / "bubble_border.png"
    Image.fromarray(border).save(diag_border_path)
    print(f"  Saved diagnostic border: {diag_border_path}")

    # Step 4: Apply mask to all frames
    print(f"\n{'=' * 60}")
    print(f"Step 4: Applying mask to {total_frames} frames")
    print(f"{'=' * 60}")

    # Process frame 0 (already read)
    rgba = apply_mask_to_frame(frame0, smoothed, border, border_color)
    Image.fromarray(rgba).save(out_dir / "frame_0000.png")

    # Process remaining frames
    for i in range(1, total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"\n  Warning: could not read frame {i}, stopping at {i} frames")
            total_frames = i
            break

        rgba = apply_mask_to_frame(frame, smoothed, border, border_color)
        Image.fromarray(rgba).save(out_dir / f"frame_{i:04d}.png")

        if (i + 1) % 10 == 0 or i == total_frames - 1:
            print(f"  Processed {i + 1}/{total_frames}", end="\r", flush=True)

    cap.release()
    print()

    # Save metadata
    metadata = {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": total_frames,
        "source": str(input_path),
        "bubble_threshold": args.threshold,
        "bubble_border_width": args.border_width,
        "bubble_smooth_radius": args.smooth_radius,
        "bubble_border_color": list(border_color),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")
    print(f"Output: {out_dir}")
    print(f"Frames processed: {total_frames}")
    print(f"Diagnostic mask: {diag_mask_path}")


if __name__ == "__main__":
    main()
