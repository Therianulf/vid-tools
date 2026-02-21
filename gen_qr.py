#!/usr/bin/env python3
"""Generate QR codes from a URL. Creates web and print versions."""

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import segno


def make_filename(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")

    # extract domain without TLD and www
    domain = parsed.hostname or ""
    domain = re.sub(r"^www\.", "", domain)
    domain = domain.split(".")[0]

    # grab the last meaningful path segment
    path = parsed.path.strip("/")
    segment = ""
    if path:
        last = path.split("/")[-1]
        segment = Path(last).stem  # strip .html, .php, etc.

    # PascalCase both parts
    domain = domain.capitalize()
    segment = segment.capitalize() if segment else ""

    return f"{domain}{segment}"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <url>")
        sys.exit(1)

    url = sys.argv[1]
    if "://" not in url:
        url = f"https://{url}"

    base = make_filename(url)
    qr = segno.make(url, error="H")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # web: scale 8 (~250px, fast loading, sharp on screens)
    web_path = output_dir / f"{base}_web.png"
    qr.save(str(web_path), scale=8, border=2)
    print(f"Web:   {web_path}")

    # print: scale 30 (~900px, crisp on flyers at 300dpi)
    print_path = output_dir / f"{base}_print.png"
    qr.save(str(print_path), scale=30, border=4)
    print(f"Print: {print_path}")

    # bonus: SVG for infinite scaling on print materials
    svg_path = output_dir / f"{base}.svg"
    qr.save(str(svg_path), scale=1, border=4)
    print(f"SVG:   {svg_path}")


if __name__ == "__main__":
    main()
