#!/usr/bin/env python3
"""Remove green spill/fringe from edges of a transparent PNG."""

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
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_despilled{input_path.suffix}"

    img = Image.open(input_path).convert("RGBA")
    data = np.array(img, dtype=np.float32)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # cap green channel to the average of red and blue
    # this is standard green despill used in compositing
    max_green = (r + b) / 2.0
    spill_mask = g > max_green
    visible_mask = a > 0

    apply = spill_mask & visible_mask
    data[apply, 1] = max_green[apply]

    count = np.count_nonzero(apply)
    print(f"Despilled {count} pixels")

    output_img = Image.fromarray(data.astype(np.uint8))
    output_img.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
