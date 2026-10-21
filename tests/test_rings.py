"""Tests for trailer_rings.py and its place in trailer_cut.py (no ffmpeg needed)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from trailer_cut import build_ffmpeg_args
from trailer_rings import RingLayer, ring_births, ring_settings

ROOT = Path(__file__).resolve().parent.parent
SPEC = json.loads((ROOT / "trailer-specs" / "coop-comms-v1.json").read_text())
SIZE = (320, 180)


def small(**extra):
    block = {"center": [160, 90], "line_width": 3, "glow": None, "start_radius": 40, "speed": 0, "opacity": 0.28}
    block.update(extra)
    return ring_settings(block)


def in_ranges(d, ranges):
    return any(lo - 1e-9 <= d <= hi + 1e-9 for lo, hi in ranges)


def test_births_defaults():
    s = ring_settings({"center": [0, 0], "first": 0.2})
    births = ring_births(s, 3.0)
    assert births == ring_births(s, 3.0)
    assert births[0] == 0.2
    assert all(in_ranges(b - a, [(0.18, 0.34), (0.9, 1.8)]) for a, b in zip(births, births[1:], strict=False))
    other = ring_settings({"center": [0, 0], "first": 0.2, "seed": 2})
    assert ring_births(other, 3.0) != births


def test_phrase_structure():
    s = ring_settings({"center": [0, 0], "phrase": {"rings": [2, 2]}})
    births = ring_births(s, 20.0)
    diffs = [b - a for a, b in zip(births, births[1:], strict=False)]
    for i, d in enumerate(diffs):
        assert in_ranges(d, [(0.18, 0.34)] if i % 2 == 0 else [(0.9, 1.8)])


def test_ring_pixels():
    layer = RingLayer(small(), SIZE, 3.0)
    img = layer.frame(0.5, [0.0])
    left, right = img[90, 120], img[90, 200]
    assert left.max() > 0 and left.argmax() == 1
    assert right.max() > 0 and right.argmax() == 2
    assert not img[90, 160].any()
    assert img.max() <= 0.28 + 1e-6


def test_host_side_right():
    img = RingLayer(small(host_side="right"), SIZE, 3.0).frame(0.5, [0.0])
    assert img[90, 120].argmax() == 2
    assert img[90, 200].argmax() == 1
    with pytest.raises(ValueError):
        ring_settings({"center": [0, 0], "host_side": "up"})


def test_start_end_fade():
    layer = RingLayer(small(start=1.0, end=2.0), SIZE, 3.0)
    assert not layer.frame(0.5, [0.0]).any()
    assert not layer.frame(2.5, [0.0]).any()
    faded = RingLayer(small(start=1.0, end=2.0, fade=0.5), SIZE, 3.0)
    ratio = faded.frame(1.25, [0.0]).max() / faded.frame(1.5, [0.0]).max()
    assert 0.45 <= ratio <= 0.55


def test_determinism():
    a = RingLayer(small(), SIZE, 3.0).frame(0.5, [0.0])
    b = RingLayer(small(), SIZE, 3.0).frame(0.5, [0.0])
    assert np.array_equal(a, b)


def test_target():
    s = ring_settings({"center": [160, 90], "target": [260, 90], "reach": 0.5, "fade_ratio": 1.2})
    assert s["speed"] == pytest.approx(200)
    assert s["max_radius"] == pytest.approx(120)
    with pytest.raises(ValueError):
        ring_settings({"center": [160, 90], "target": [260, 90], "speed": 100})


def test_glow():
    on = RingLayer(small(glow={"width": 20, "opacity": 0.3}), SIZE, 3.0).frame(0.5, [0.0])
    off = RingLayer(small(glow=None), SIZE, 3.0).frame(0.5, [0.0])
    assert on[90, 206].any()
    assert not off[90, 206].any()


def test_ffmpeg_args_with_rings():
    spec = copy.deepcopy(SPEC)
    spec["beats"][0]["rings"] = {"center": [190, 525]}
    texts = {n: Path(f"/fake/work/text-{n}.png") for n in (1, 2, 3, 4)}
    ends = {5: Path("/fake/work/end-5.png")}
    voice = [(n, Path(f"/fake/vo/vo-{n}.wav")) for n in range(1, 6)]
    rings = {1: "/fake/work/rings-1-%04d.png"}

    def build(v, m):
        return build_ffmpeg_args(spec, Path("/fake/take.mp4"), Path("/fake/shots"), texts, ends, v, m, 0.0,
                                 Path("/fake/out.mp4"), rings)

    args = build(voice, Path("/fake/music.wav"))
    inputs = [i for i, a in enumerate(args) if a == "-i"]
    assert len(inputs) == 17
    last = inputs[-1]
    assert args[last + 1] == rings[1]
    assert args[last - 4 : last] == ["-framerate", "30", "-start_number", "0"]
    graph = args[args.index("-filter_complex") + 1]
    for text in (
        "[0:v]trim=0:3,setpts=PTS-STARTPTS,fps=30,scale=1920:1080:flags=lanczos,setsar=1[c1]",
        "[16:v]format=gbrp[r1]",
        "[c1]format=gbrp[g1]",
        "[g1][r1]blend=all_mode=screen:shortest=1,format=yuv420p,fade=t=in:st=0:d=0.4[b1]",
        "[15:a]",
    ):
        assert text in graph
    assert "[15:a]aresample" in graph and "afade" in graph
    args = build([], None)
    assert "[10:v]format=gbrp[r1]" in args[args.index("-filter_complex") + 1]


def test_settings_need_center():
    with pytest.raises(ValueError):
        ring_settings({})
