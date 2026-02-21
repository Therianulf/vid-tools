#!/usr/bin/env python3
"""Remove remaining green artifacts from a transparent PNG."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


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
        output_path = input_path.with_stem(f"{input_path.stem}_cleaned")

    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    green_mask = (g > 100) & (g > r * 1.5) & (g > b * 1.5)
    data[green_mask] = [0, 0, 0, 0]

    count = np.count_nonzero(green_mask)
    print(f"Removed {count} green pixels")

    output_img = Image.fromarray(data)
    output_img.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
