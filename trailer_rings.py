#!/usr/bin/env python3
"""
Trailer Rings
=============
Renders the website's broadcast rings (coop-comms-website js/main.js, initRings) as PNG frames for a
trailer beat: full circles growing from a centre, the Host half teal (left by default) and the
Guest half blue, born in phrases like speech. Used by trailer_cut.py; also runs alone to preview a beat's rings.

Usage:
    python trailer_rings.py --spec trailer-specs/coop-comms-v1.json
    python trailer_rings.py --spec spec.json --beat 1 -o output/trailer-rings

Requirements:
    numpy, Pillow
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from trailer_cards import hex_rgb

DEFAULTS = {
    "host_side": "left",
    "colors": ["#5bd6b0", "#93b4ff"],
    "line_width": 3,
    "opacity": 0.55,
    "falloff": 1.5,
    "start_radius": 48,
    "speed": 108,
    "max_radius": None,
    "target": None,
    "reach": 1.0,
    "fade_ratio": 1.2,
    "phrase": {"rings": [2, 5], "gap": [0.18, 0.34], "pause": [0.9, 1.8]},
    "seed": 1,
    "first": 0.0,
    "start": 0.0,
    "end": None,
    "fade": 0.0,
    "glow": {"width": 20, "opacity": 0.3},
    "blend": "screen",
}


def ring_settings(block):
    """The rings block merged over the defaults (phrase and glow key by key)."""
    if "center" not in block:
        raise ValueError("rings: center is required")
    if block.get("host_side", "left") not in ("left", "right"):
        raise ValueError('rings: host_side must be "left" or "right"')
    if block.get("target") is not None and ("speed" in block or "max_radius" in block):
        raise ValueError("rings: give target or speed/max_radius, not both")
    settings = copy.deepcopy(DEFAULTS)
    for key, value in block.items():
        if key == "phrase":
            settings["phrase"].update(value)
        elif key == "glow" and isinstance(value, dict):
            settings["glow"].update(value)
        else:
            settings[key] = copy.deepcopy(value)
    if settings["target"] is not None:
        d = math.dist(settings["center"], settings["target"])
        settings["speed"] = d / settings["reach"]
        settings["max_radius"] = settings["fade_ratio"] * d
    return settings


def ring_births(settings, t1):
    """Birth times of the rings before t1: phrases of rings gap-spaced, then a pause; the first is at "first"."""
    rng = random.Random(settings["seed"])
    phrase = settings["phrase"]
    rings_lo, rings_hi = phrase["rings"]
    gap_lo, gap_hi = phrase["gap"]
    pause_lo, pause_hi = phrase["pause"]
    births = []
    t = settings["first"]
    while t < t1:
        k = rings_lo + int(rng.random() * (rings_hi - rings_lo + 1))
        for i in range(k):
            if t >= t1:
                break
            births.append(t)
            if i < k - 1:
                t += gap_lo + rng.random() * (gap_hi - gap_lo)
        t += pause_lo + rng.random() * (pause_hi - pause_lo)
    return births


class RingLayer:
    """Draws the rings for one beat; the per-pixel geometry is computed once."""

    def __init__(self, settings, size, seconds=None):
        self.settings = settings
        w, h = size
        self.w, self.h = w, h
        self.cx, self.cy = settings["center"]
        xs = np.arange(w, dtype=np.float32) + 0.5
        ys = np.arange(h, dtype=np.float32) + 0.5
        self.dist = np.hypot(xs[None, :] - self.cx, ys[:, None] - self.cy).astype(np.float32)
        self.left = np.broadcast_to(xs[None, :] < self.cx, (h, w)).copy()
        host = np.array(hex_rgb(settings["colors"][0]), dtype=np.float32) / 255
        guest = np.array(hex_rgb(settings["colors"][1]), dtype=np.float32) / 255
        left_rgb, right_rgb = (host, guest) if settings["host_side"] == "left" else (guest, host)
        self.colour = np.where(self.left[..., None], left_rgb, right_rgb).astype(np.float32)
        if settings["max_radius"] is not None:
            self.max_r = float(settings["max_radius"])
        else:
            self.max_r = math.hypot(max(self.cx, w - self.cx), max(self.cy, h - self.cy))
        self.seconds = seconds

    def frame(self, t, births):
        """An (h, w, 3) float32 image in 0..1 for beat-local time t."""
        s = self.settings
        start = s["start"]
        end = s["end"] if s["end"] is not None else self.seconds
        if t < start or (end is not None and t >= end):
            return np.zeros((self.h, self.w, 3), dtype=np.float32)
        env = 1.0
        if s["fade"] > 0:
            env = min(1.0, (t - start) / s["fade"])
            if end is not None:
                env = min(env, (end - t) / s["fade"])
        if env <= 0:
            return np.zeros((self.h, self.w, 3), dtype=np.float32)

        layer = np.zeros((self.h, self.w, 3), dtype=np.float32)
        lw = s["line_width"]
        for b in births:
            if b > t:
                continue
            r = s["start_radius"] + s["speed"] * (t - b)
            if r >= self.max_r:
                continue
            fade_factor = max(0.0, 1 - r / self.max_r) ** s["falloff"]
            glow = s["glow"]
            has_glow = bool(glow) and glow["opacity"] > 0
            sigma = glow["width"] / 2.355 if has_glow else 0.0
            pad = r + lw + 3 * sigma + 1
            y0 = max(0, int(math.floor(self.cy - pad)))
            y1 = min(self.h, int(math.ceil(self.cy + pad)) + 1)
            x0 = max(0, int(math.floor(self.cx - pad)))
            x1 = min(self.w, int(math.ceil(self.cx + pad)) + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            region = (slice(y0, y1), slice(x0, x1))
            delta = self.dist[region] - r
            if has_glow:
                g = np.exp(-(delta**2) / (2 * sigma**2))
                k = (g * glow["opacity"] * fade_factor)[..., None]
                layer[region] = layer[region] * (1 - k) + self.colour[region] * k
            coverage = np.clip(lw / 2 + 0.5 - np.abs(delta), 0, 1)
            k = (coverage * s["opacity"] * fade_factor)[..., None]
            layer[region] = layer[region] * (1 - k) + self.colour[region] * k
        return (layer * env).astype(np.float32)


def render_rings(block, size, fps, seconds, out_dir, name):
    """Write the beat's ring frames as PNGs and return the ffmpeg input pattern."""
    settings = ring_settings(block)
    births = ring_births(settings, seconds)
    layer = RingLayer(settings, size, seconds)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(round(seconds * fps)):
        pixels = np.round(layer.frame(i / fps, births) * 255).astype(np.uint8)
        Image.fromarray(pixels, "RGB").save(out_dir / f"{name}-{i:04d}.png")
    return str(out_dir / f"{name}-%04d.png")


def main():
    parser = argparse.ArgumentParser(description="Render a trailer beat's broadcast rings as PNG frames.")
    parser.add_argument("--spec", required=True, type=Path, help="the trailer JSON spec")
    parser.add_argument("--beat", type=int, help="beat number (default: the first beat with rings)")
    parser.add_argument("-o", "--out", type=Path, default=Path("output/trailer-rings"), help="folder for the PNGs")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    beats = spec["beats"]
    if args.beat is None:
        numbers = [n for n, b in enumerate(beats, start=1) if "rings" in b]
        if not numbers:
            parser.error("no beat in the spec has a rings block")
        args.beat = numbers[0]
    if not 1 <= args.beat <= len(beats) or "rings" not in beats[args.beat - 1]:
        parser.error(f"beat {args.beat} has no rings block")
    beat = beats[args.beat - 1]
    pattern = render_rings(beat["rings"], spec["size"], spec["fps"], beat["seconds"], args.out, f"rings-{args.beat}")
    print(pattern)
    print(f"{round(beat['seconds'] * spec['fps'])} frames")


if __name__ == "__main__":
    main()
