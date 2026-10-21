#!/usr/bin/env python3
"""
Trailer Cards
=============
Renders a trailer's on-screen lines (lower thirds) and end cards as PNGs with Pillow.
Used by trailer_cut.py; this ffmpeg has no drawtext, so text is drawn here and laid over the picture.

Usage:
    python trailer_cards.py --spec trailer-specs/coop-comms-v1.json \\
        --tagline-font brand/fonts/Orbitron-wght.ttf --wordmark wordmark.png
    python trailer_cards.py --spec spec.json --tagline-font font.ttf --wordmark wordmark.png -o output/cards

Requirements:
    Pillow
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

BAND_PAD_X = 44
BAND_PAD_Y = 26
BAND_BOTTOM = 96
BAND_ALPHA = 215
MARK_GAP = 30  # wordmark bottom sits this far above the vertical middle
TAGLINE_GAP = 40  # tagline top sits this far below the vertical middle

FRIBIDI_DIRS = ("/opt/homebrew/lib", "/usr/local/lib")


def ensure_raqm():
    """Pillow kerns text only through raqm, which needs libfribidi; re-run where this Python can find it, or stop."""
    if features.check("raqm"):
        return
    if sys.platform == "darwin" and "VIDTOOLS_RAQM_RETRY" not in os.environ:
        dirs = [d for d in FRIBIDI_DIRS if Path(d, "libfribidi.0.dylib").exists()]
        if dirs:
            fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
            env = dict(os.environ, VIDTOOLS_RAQM_RETRY="1", DYLD_FALLBACK_LIBRARY_PATH=":".join([*dirs, fallback]))
            os.execve(sys.executable, [sys.executable, *sys.argv], env)
    sys.exit("Pillow cannot load raqm, so the text would be set without kerning. Install fribidi: brew install fribidi")


def hex_rgb(value):
    """Turn "#121519" into (0x12, 0x15, 0x19)."""
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def font(path, size, weight):
    """Load a variable font at a size and weight."""
    f = ImageFont.truetype(str(path), size)
    f.set_variation_by_axes([weight])
    return f


def resolve_line_font(spec, override):
    """The line font: the override, else the spec's path (relative paths resolve against the vid-tools root)."""
    if override is not None:
        return Path(override)
    path = Path(spec["line_font"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def lower_third(text, face, size, background, color):
    """A transparent frame with the text on a rounded band near the bottom."""
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=face)
    tw, th = right - left, bottom - top
    box_w, box_h = tw + 2 * BAND_PAD_X, th + 2 * BAND_PAD_Y
    x0 = (w - box_w) // 2
    y0 = h - BAND_BOTTOM - box_h
    draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=box_h // 2, fill=background + (BAND_ALPHA,))
    draw.text((x0 + BAND_PAD_X - left, y0 + BAND_PAD_Y - top), text, font=face, fill=color + (255,))
    return img


def end_card(wordmark_png, tagline, face, size, background, color):
    """The wordmark centred above the middle, with an optional tagline below it."""
    w, h = size
    img = Image.new("RGB", (w, h), background)
    mark = Image.open(wordmark_png).convert("RGBA")
    mx = (w - mark.width) // 2
    my = h // 2 - mark.height - MARK_GAP
    img.paste(mark, (mx, my), mark)
    if tagline:
        draw = ImageDraw.Draw(img)
        left, top, right, _ = draw.textbbox((0, 0), tagline, font=face)
        draw.text(((w - (right - left)) // 2 - left, h // 2 + TAGLINE_GAP - top), tagline, font=face, fill=color)
    return img


def render_cards(spec, line_font, tagline_font, wordmark_png, out_dir):
    """Write text-N.png for each beat with a line and end-N.png for each end card. Returns (texts, ends)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = spec["cards"]
    size = tuple(spec["size"])
    background = hex_rgb(cards["background"])
    color = hex_rgb(cards["text_color"])
    line_face = font(line_font, cards["line_size"], cards["line_weight"])
    tagline_face = font(tagline_font, cards["tagline_size"], cards["tagline_weight"])
    texts = {}
    ends = {}
    for n, beat in enumerate(spec["beats"], start=1):
        if "line" in beat:
            path = out_dir / f"text-{n}.png"
            lower_third(beat["line"], line_face, size, background, color).save(path)
            texts[n] = path
        if beat["kind"] == "end_card":
            path = out_dir / f"end-{n}.png"
            end_card(wordmark_png, beat.get("tagline"), tagline_face, size, background, color).save(path)
            ends[n] = path
    return texts, ends


def main():
    ensure_raqm()
    parser = argparse.ArgumentParser(description="Render a trailer's text and end card PNGs.")
    parser.add_argument("--spec", required=True, type=Path, help="the trailer JSON spec")
    parser.add_argument("--line-font", type=Path, help="font for the on-screen lines (default: the spec's line_font)")
    parser.add_argument("--tagline-font", required=True, type=Path, help="font for the end card tagline")
    parser.add_argument("--wordmark", required=True, type=Path, help="the wordmark rasterised to PNG")
    parser.add_argument("-o", "--out", type=Path, default=Path("output/trailer-cards"), help="output folder")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    line_font = resolve_line_font(spec, args.line_font)
    texts, ends = render_cards(spec, line_font, args.tagline_font, args.wordmark, args.out)
    for path in [*texts.values(), *ends.values()]:
        print(path)


if __name__ == "__main__":
    main()
