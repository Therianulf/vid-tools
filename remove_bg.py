#!/usr/bin/env python3
"""Remove background from an image using rembg (U2-Net model)."""

import sys
from pathlib import Path

from PIL import Image
from rembg import remove


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_file> [output_file]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_nobg.png"

    print(f"Processing: {input_path}")
    input_img = Image.open(input_path)
    output_img = remove(input_img)
    output_img.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
