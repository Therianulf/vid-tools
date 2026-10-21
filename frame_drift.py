#!/usr/bin/env python3
"""
Frame Drift
===========
Scores an AI video take, or one still, for drift: SSIM and the loop finder's similarity score against an
approved reference still and against the previous sample, over the whole frame and named crops (faces, props).
Also finds changed regions on its own and writes side-by-side evidence images of each suspected area.
Built on video_looper.py --scan: the same frame reader and frame_similarity score. Agent-facing: stdout is one
JSON document (also written to <out>/drift.json), progress goes to stderr, and the exit code is the verdict.

Usage:
    python frame_drift.py take.mp4 --reference still.png --size 1920x1080 \\
        --crop face_left=400,240,240,260 --crop mic=900,620,130,160 -o output/take-drift
    python frame_drift.py take.mp4 --crop face_left=400,240,240,260    # frame to frame only
    python frame_drift.py new-still.png --reference approved-still.png

Exit codes: 0 clean, 2 something to review (a threshold crossed or a changed region found), 1 usage or runtime
error. Changed regions and flagged crops each get an evidence image for a judge; see review[] in the JSON.

Requirements:
    opencv-python, numpy, torch (imported by video_looper.py)
"""

import argparse
import contextlib
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from video_looper import frame_similarity, iter_frames, load_video_info

TOOL_VERSION = "1.2"  # bump when the JSON document changes shape
DEFAULT_EVERY = 6  # every 6th frame: 4 samples a second at 24 fps, 21 samples on a 5 s take
DEFAULT_THRESHOLD = 0.6  # SSIM vs reference; on take-04 static parts stayed above 0.78, drifting faces fell below 0.6
DEFAULT_PREV_THRESHOLD = 0.5  # SSIM vs previous sample; half the structure gone in one step (0.25 s at the defaults)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}  # inputs read as one still, by suffix
SSIM_WINDOW = 11  # Gaussian window of Wang et al. (2004)
SSIM_SIGMA = 1.5  # Gaussian sigma of Wang et al.
SSIM_K1 = 0.01  # C1 = (K1 * 255) ** 2
SSIM_K2 = 0.03  # C2 = (K2 * 255) ** 2
WORST_COUNT = 5  # samples listed in summary.worst
SHEET_COLUMNS = 6  # thumbnails per contact-sheet row
SHEET_THUMB_W = 320  # thumbnail width in pixels; height keeps the analysis aspect
SHEET_LABEL_H = 54  # label strip under each thumbnail: three lines 17 px apart
SHEET_GAP = 8  # margin around and between cells
SHEET_BORDER = 4  # width of the red border on a flagged thumbnail
SHEET_BG = (32, 32, 32)  # BGR
SHEET_TEXT = (235, 235, 235)  # BGR
SHEET_FLAG = (0, 0, 255)  # BGR red: flagged border and first label line
DEFAULT_SUSPECTS = 3  # changed regions kept per sample; 0 turns the search off
SUSPECT_SSIM = 0.5  # a pixel is "changed" where the local SSIM map falls below this
SUSPECT_MIN_AREA = 0.001  # smallest changed region, as a fraction of the analysis frame's area
SUSPECT_PAD = 24  # pixels added around a changed region's box
EVIDENCE_PANEL_H = 360  # height of each evidence panel in pixels
EVIDENCE_HEADER_H = 46  # two header lines above the panels
EVIDENCE_CAPTION_H = 24  # one caption line under each panel
SUSPECT_COLOR = (0, 215, 255)  # BGR amber: suspect boxes on the contact sheet
EXIT_CLEAN = 0
EXIT_ERROR = 1  # usage or runtime error; stdout stays empty
EXIT_FLAGGED = 2  # something to review: a sample crossed a threshold or a changed region was found
CROP_RE = re.compile(r"^([A-Za-z0-9_-]+)=(\d+),(\d+),(\d+),(\d+)$")
SIZE_RE = re.compile(r"^(\d+)[xX](\d+)$")
SSIM_TEXT = (
    "Wang et al. 2004 on luma (OpenCV BGR2GRAY), Gaussian 11x11 sigma 1.5, K1 0.01, K2 0.03, edges reflected; "
    "mean of the SSIM map over the whole frame or inside each crop"
)
LEGEND = {
    "ssim_ref": (
        "SSIM of the sample against the reference still (1.0 identical, near 0 unrelated); "
        "below settings.threshold is flagged"
    ),
    "ssim_prev": (
        "SSIM of the sample against the previous sample; a drop is a jump between samples; "
        "below settings.prev_threshold is flagged"
    ),
    "ssim_ref_lm": (
        "with --match-light: SSIM against the reference after the sample's grey levels in the region are "
        "histogram-matched to the reference's (CDF matching); a light change that keeps the shapes scores near 1, "
        "shape drift still scores low; flagged instead of ssim_ref when --match-light is on; null otherwise"
    ),
    "ssim_prev_lm": (
        "the same against the previous sample; flagged instead of ssim_prev when --match-light is on; "
        "null otherwise"
    ),
    "loop_score_ref": (
        "video_looper.py frame_similarity against the reference "
        "(40% grey histogram correlation, 60% inverse MSE at 64x64); reported, not thresholded"
    ),
    "loop_score_prev": "the same loop-finder score against the previous sample; reported, not thresholded",
    "whole": "metrics over the whole analysis frame",
    "crops": "metrics inside each named crop; boxes are [x, y, w, h] in analysis pixels (settings.analysis_size)",
    "flags": "checks the sample failed, written region:metric<threshold",
    "time": "seconds from the start of the input (frame / fps)",
    "crop_to_aspect": (
        "[x, y, w, h] centred box kept before resizing to the analysis size so the aspect matches; "
        "null when the whole picture was used"
    ),
    "suspects": (
        "changed regions found automatically: where the SSIM map vs the previous sample (a still: vs the "
        "reference) falls below 0.5, joined, at least 0.1% of the frame, padded 24 px; biggest first, up to "
        "--suspects per sample; box [x, y, w, h] in analysis pixels"
    ),
    "evidence": (
        "side-by-side images of a suspected area: the previous sample, the current sample and the reference, "
        "cropped to the box and enlarged, with the time and the scores burned in; one per flagged region and "
        "per suspect"
    ),
    "review": (
        "the judge's queue: every evidence image in time order with its reason (flag:<checks> or suspect) and "
        "scores; a judge looks at each image and answers natural or unnatural (see the rubric in frame-drift.md)"
    ),
    "null": "a metric is null when it does not apply: no reference, or the first sample has no previous one",
}
EPILOG = """example:
  python frame_drift.py take.mp4 --reference still.png --size 1920x1080 \\
    --crop face_left=400,240,240,260 --crop mic=900,620,130,160 -o out/take-drift
stdout: one JSON document (verdict, flagged samples, paths, per-sample scores),
        also written to <out>/drift.json. Progress and errors go to stderr.
exit: 0 clean, 2 something to review (threshold crossed or changed region), 1 error.
"""
METRIC_NAMES = ("ssim_ref", "ssim_prev", "ssim_ref_lm", "ssim_prev_lm", "loop_score_ref", "loop_score_prev")


class DriftError(Exception):
    """A runtime problem reported on stderr with exit code 1."""


class AgentParser(argparse.ArgumentParser):
    """argparse whose usage errors exit 1, so exit 2 always means something to review."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def log(message):
    print(message, file=sys.stderr)


def parse_crop(text):
    """argparse type for --crop: NAME=X,Y,W,H -> (name, (x, y, w, h))."""
    m = CROP_RE.match(text)
    if not m:
        raise argparse.ArgumentTypeError(f"crop must be NAME=X,Y,W,H, got {text!r}")
    name = m.group(1)
    if name == "whole":
        raise argparse.ArgumentTypeError("crop name 'whole' is reserved for the whole frame")
    x, y, w, h = (int(m.group(i)) for i in range(2, 6))
    if w == 0 or h == 0:
        raise argparse.ArgumentTypeError("crop width and height must be above 0")
    return name, (x, y, w, h)


def parse_size(text):
    """argparse type for --size: WxH -> (w, h)."""
    m = SIZE_RE.match(text)
    if not m or int(m.group(1)) == 0 or int(m.group(2)) == 0:
        raise argparse.ArgumentTypeError(f"size must be WxH with both above 0, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def is_still(path):
    """True when the path's suffix says it is a still image."""
    return path.suffix.lower() in IMAGE_SUFFIXES


def center_crop_box(src_w, src_h, dst_w, dst_h):
    """The largest centred box (x, y, w, h) in the source with the destination's aspect."""
    if src_w * dst_h > src_h * dst_w:  # source is wider
        w = round(src_h * dst_w / dst_h)
        return ((src_w - w) // 2, 0, w, src_h)
    if src_w * dst_h < src_h * dst_w:  # source is taller
        h = round(src_w * dst_h / dst_w)
        return (0, (src_h - h) // 2, src_w, h)
    return (0, 0, src_w, src_h)


def fit_to_size(image, size):
    """Centre-crop to the size's aspect, then resize; box is the crop kept, or None when the whole image was used."""
    ih, iw = image.shape[:2]
    x, y, w, h = center_crop_box(iw, ih, *size)
    crop = image[y:y + h, x:x + w]
    if (crop.shape[1], crop.shape[0]) == tuple(size):
        fitted = crop
    else:
        smaller = size[0] <= crop.shape[1] and size[1] <= crop.shape[0]
        fitted = cv2.resize(crop, tuple(size), interpolation=cv2.INTER_AREA if smaller else cv2.INTER_CUBIC)
    box = None if (x, y, w, h) == (0, 0, iw, ih) else [x, y, w, h]
    return fitted, box


def blur(img):
    return cv2.GaussianBlur(img, (SSIM_WINDOW, SSIM_WINDOW), SSIM_SIGMA)


def ssim_map(gray_a, gray_b):
    """Per-pixel SSIM of two same-size 8-bit greyscale images (Wang et al. 2004, Gaussian 11x11, sigma 1.5)."""
    a = gray_a.astype(np.float64)
    b = gray_b.astype(np.float64)
    mu_a = blur(a)
    mu_b = blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    c1 = (SSIM_K1 * 255) ** 2
    c2 = (SSIM_K2 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def ssim(gray_a, gray_b):
    """Mean SSIM of two same-size 8-bit greyscale images."""
    return float(ssim_map(gray_a, gray_b).mean())


def match_light(src, ref):
    """src's grey levels remapped so its histogram matches ref's (CDF matching); both uint8 arrays."""
    s_cdf = np.cumsum(np.bincount(src.ravel(), minlength=256)) / src.size
    r_cdf = np.cumsum(np.bincount(ref.ravel(), minlength=256)) / ref.size
    lut = np.searchsorted(r_cdf, s_cdf, side="left").clip(0, 255).astype(np.uint8)
    return lut[src]


def sample_frames(total_frames, fps, every=None, sample_fps=None, start=0.0, end=None):
    """Frame numbers to sample, in order."""
    first = round(start * fps)
    stop = total_frames if end is None else min(total_frames, round(end * fps))
    if first >= stop:
        return []
    if sample_fps is None:
        return list(range(first, stop, every or DEFAULT_EVERY))
    step = fps / sample_fps
    frames = []
    k = 0
    while True:
        n = first + round(k * step)
        if n >= stop:
            break
        if not frames or n != frames[-1]:
            frames.append(n)
        k += 1
    return frames


def check_crops(crops, size):
    """Raise DriftError for the first crop that does not fit inside the analysis frame."""
    aw, ah = size
    for name, (x, y, w, h) in crops:
        if x + w > aw or y + h > ah:
            raise DriftError(f"crop {name} {x},{y},{w},{h} is outside the {aw}x{ah} analysis frame")


def region_metrics(frame, ref, prev, ref_map, prev_map, box, gray=None, ref_gray=None, prev_gray=None, light=False):
    """The six metrics inside box (x, y, w, h), rounded to 4 places; None where there is nothing to compare.

    The light-matched pair is scored only when light is True."""
    x, y, w, h = box
    sl = (slice(y, y + h), slice(x, x + w))

    def lm(other):
        if not light or other is None:
            return None
        g = gray[sl]
        r = other[sl]
        return round(ssim(match_light(g, r), r), 4)

    return {
        "ssim_ref": None if ref_map is None else round(float(ref_map[sl].mean()), 4),
        "ssim_prev": None if prev_map is None else round(float(prev_map[sl].mean()), 4),
        "ssim_ref_lm": lm(ref_gray),
        "ssim_prev_lm": lm(prev_gray),
        "loop_score_ref": None if ref is None else round(float(frame_similarity(frame[sl], ref[sl])), 4),
        "loop_score_prev": None if prev is None else round(float(frame_similarity(frame[sl], prev[sl])), 4),
    }


def check_names(crop_names, has_reference, light=False):
    """Thresholded checks in report order: whole first, then crops in CLI order; ssim_ref before ssim_prev."""
    suffix = "_lm" if light else ""
    metrics = ([f"ssim_ref{suffix}"] if has_reference else []) + [f"ssim_prev{suffix}"]
    return [f"{region}:{metric}" for region in ["whole", *crop_names] for metric in metrics]


def check_limit(check, threshold, prev_threshold):
    """The threshold a check is held to."""
    return threshold if ":ssim_ref" in check else prev_threshold


def check_value(record, check):
    """The stored value a check reads from a sample record."""
    region, metric = check.split(":")
    source = record["whole"] if region == "whole" else record["crops"][region]
    return source[metric]


def sample_flags(record, checks, threshold, prev_threshold):
    """Checks the sample fails, written region:metric<limit."""
    flags = []
    for check in checks:
        limit = check_limit(check, threshold, prev_threshold)
        value = check_value(record, check)
        if value is not None and value < limit:
            flags.append(f"{check}<{limit:g}")
    return flags


def summarize(samples, crop_names, checks, threshold, prev_threshold, has_reference, light=False):
    """Counts, per-check failures, per-region metric ranges and the worst samples."""
    check_info = {}
    for check in checks:
        limit = check_limit(check, threshold, prev_threshold)
        failing = [s for s in samples if (v := check_value(s, check)) is not None and v < limit]
        first = failing[0] if failing else None
        check_info[check] = {
            "threshold": limit,
            "failed": len(failing),
            "first_below": None
            if first is None
            else {"index": first["index"], "frame": first["frame"], "time": first["time"]},
        }

    metric_info = {}
    for region in ["whole", *crop_names]:
        metric_info[region] = {}
        for metric in METRIC_NAMES:
            values = [
                (s["whole"] if region == "whole" else s["crops"][region])[metric]
                for s in samples
            ]
            pairs = [(i, v) for i, v in enumerate(values) if v is not None]
            if pairs:
                low = min(v for _, v in pairs)
                metric_info[region][metric] = {
                    "min": round(low, 4),
                    "mean": round(sum(v for _, v in pairs) / len(pairs), 4),
                    "min_index": next(i for i, v in pairs if v == low),
                }
            else:
                metric_info[region][metric] = {"min": None, "mean": None, "min_index": None}

    metric = ("ssim_ref" if has_reference else "ssim_prev") + ("_lm" if light else "")
    candidates = []
    for s in samples:
        best = None
        for region in ["whole", *crop_names]:
            value = (s["whole"] if region == "whole" else s["crops"][region])[metric]
            if value is not None and (best is None or value < best[1]):
                best = (region, value)
        if best is not None:
            candidates.append((best[1], s["index"], s, f"{best[0]}:{metric}"))
    candidates.sort(key=lambda c: (c[0], c[1]))
    worst = [
        {"index": s["index"], "frame": s["frame"], "time": s["time"], "file": s["file"], "check": check, "value": value}
        for value, _, s, check in candidates[:WORST_COUNT]
    ]

    return {
        "samples": len(samples),
        "flagged_samples": sum(1 for s in samples if s["flagged"]),
        "checks": check_info,
        "metrics": metric_info,
        "worst": worst,
    }


def verdict_line(summary):
    """One line for the verdict: CLEAN, REVIEW (changed regions only), or FLAGGED with the first crossing."""
    n = summary["samples"]
    noun = "sample" if n == 1 else "samples"
    if not summary["flagged_samples"]:
        if summary["suspects"]:
            regions = summary["review"]
            return (
                f"REVIEW: {n} {noun} above thresholds; "
                f"{regions} changed {'region' if regions == 1 else 'regions'} to review"
            )
        return f"CLEAN: {n} {noun}, all above thresholds"
    fired = [c for c, info in summary["checks"].items() if info["failed"] > 0]
    first = min(fired, key=lambda c: summary["checks"][c]["first_below"]["index"])
    region, metric = first.split(":")
    label = "whole frame" if region == "whole" else region
    against = "reference" if metric.startswith("ssim_ref") else "previous sample"
    word = "light-matched SSIM" if metric.endswith("_lm") else "SSIM"
    info = summary["checks"][first]
    line = (
        f"FLAGGED: {label} below {info['threshold']:g} {word} vs {against} "
        f"from {info['first_below']['time']:.2f} s ({summary['flagged_samples']} of {n} {noun})"
    )
    others = [c for c in fired if c != first]
    if others:
        line += "; also " + ", ".join(others)
    regions = summary["review"]
    return line + f"; {regions} {'region' if regions == 1 else 'regions'} to review"


def suspect_regions(sim_map, max_regions, min_area, pad):
    """Boxes [x, y, w, h] around the largest changed regions of an SSIM map, biggest first."""
    if max_regions <= 0:
        return []
    mask = (sim_map < SUSPECT_SSIM).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    map_h, map_w = sim_map.shape[:2]
    found = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = max(0, int(stats[i, cv2.CC_STAT_LEFT]) - pad)
        y0 = max(0, int(stats[i, cv2.CC_STAT_TOP]) - pad)
        x1 = min(map_w, int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) + pad)
        y1 = min(map_h, int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) + pad)
        found.append({"box": [x0, y0, x1 - x0, y1 - y0], "area": area})

    merged = True
    while merged:
        merged = False
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                ax, ay, aw, ah = found[i]["box"]
                bx, by, bw, bh = found[j]["box"]
                if min(ax + aw, bx + bw) > max(ax, bx) and min(ay + ah, by + bh) > max(ay, by):
                    x0, y0 = min(ax, bx), min(ay, by)
                    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    found[i] = {"box": [x0, y0, x1 - x0, y1 - y0], "area": found[i]["area"] + found[j]["area"]}
                    del found[j]
                    merged = True
                    break
            if merged:
                break

    found.sort(key=lambda r: (-r["area"], r["box"][1], r["box"][0]))
    return found[:max_regions]


def evidence_image(prev_img, cur_img, ref_img, box, header_lines, flagged):
    """One judge image: the box from the previous sample, the current sample and the reference, with a header."""
    x, y, w, h = box
    panel_w = round(w * EVIDENCE_PANEL_H / h)
    gap = SHEET_GAP
    width = gap + 3 * (panel_w + gap)
    height = gap + EVIDENCE_HEADER_H + gap + EVIDENCE_PANEL_H + EVIDENCE_CAPTION_H + gap
    image = np.full((height, width, 3), SHEET_BG, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for j, line in enumerate(header_lines[:2]):
        color = SHEET_FLAG if j == 0 and flagged else SHEET_TEXT
        cv2.putText(image, line, (gap, gap + 16 + 20 * j), font, 0.55, color, 1, cv2.LINE_AA)
    top = gap + EVIDENCE_HEADER_H + gap
    for i, (source, caption) in enumerate(((prev_img, "previous"), (cur_img, "current"), (ref_img, "reference"))):
        x0 = gap + i * (panel_w + gap)
        if source is None:
            (tw, th), _ = cv2.getTextSize("none", font, 0.55, 1)
            cv2.putText(
                image, "none", (x0 + (panel_w - tw) // 2, top + (EVIDENCE_PANEL_H + th) // 2),
                font, 0.55, SHEET_TEXT, 1, cv2.LINE_AA,
            )
        else:
            crop = source[y:y + h, x:x + w]
            enlarging = EVIDENCE_PANEL_H > h
            image[top:top + EVIDENCE_PANEL_H, x0:x0 + panel_w] = cv2.resize(
                crop, (panel_w, EVIDENCE_PANEL_H), interpolation=cv2.INTER_CUBIC if enlarging else cv2.INTER_AREA
            )
        cv2.putText(
            image, caption, (x0, top + EVIDENCE_PANEL_H + 17), font, 0.55, SHEET_TEXT, 1, cv2.LINE_AA
        )
    return image


def box_scores(box, ref_map, prev_map, gray, ref_gray, prev_gray, light=False):
    """SSIM vs previous and reference inside box (plus the light-matched pair with light), rounded to 4 places."""
    x, y, w, h = box
    sl = (slice(y, y + h), slice(x, x + w))

    def mean(m):
        return None if m is None else round(float(m[sl].mean()), 4)

    def lm(other):
        if other is None:
            return None
        return round(ssim(match_light(gray[sl], other[sl]), other[sl]), 4)

    scores = {"ssim_prev": mean(prev_map), "ssim_ref": mean(ref_map)}
    if light:
        scores["ssim_prev_lm"] = lm(prev_gray)
        scores["ssim_ref_lm"] = lm(ref_gray)
    return scores


def evidence_header(record, region, box, scores, light=False):
    """The two header lines of an evidence image."""
    x, y, w, h = box
    line1 = f"#{record['index']} f{record['frame']} {record['time']:.2f}s  {region}  box {x},{y},{w},{h}"
    line2 = f"ssim prev {fmt2(scores['ssim_prev'])} ref {fmt2(scores['ssim_ref'])}"
    if light:
        line2 += f"  lm prev {fmt2(scores['ssim_prev_lm'])} ref {fmt2(scores['ssim_ref_lm'])}"
    return [line1, line2]


def sheet_size(cells, frame_w, frame_h):
    """Contact-sheet grid: (cols, rows, thumb_h, width, height)."""
    cols = min(SHEET_COLUMNS, cells)
    rows = math.ceil(cells / cols)
    thumb_h = round(SHEET_THUMB_W * frame_h / frame_w)
    width = SHEET_GAP + cols * (SHEET_THUMB_W + SHEET_GAP)
    height = SHEET_GAP + rows * (thumb_h + SHEET_LABEL_H + SHEET_GAP)
    return cols, rows, thumb_h, width, height


def contact_sheet(cells, frame_w, frame_h):
    """Draw the contact sheet; cells is a list of (image, lines, flagged, suspect boxes, flagged crop boxes)."""
    cols, _, thumb_h, width, height = sheet_size(len(cells), frame_w, frame_h)
    sheet = np.full((height, width, 3), SHEET_BG, dtype=np.uint8)
    sx = SHEET_THUMB_W / frame_w
    sy = thumb_h / frame_h
    for i, (image, lines, flagged, suspect_boxes, flag_boxes) in enumerate(cells):
        x0 = SHEET_GAP + (i % cols) * (SHEET_THUMB_W + SHEET_GAP)
        y0 = SHEET_GAP + (i // cols) * (thumb_h + SHEET_LABEL_H + SHEET_GAP)
        sheet[y0:y0 + thumb_h, x0:x0 + SHEET_THUMB_W] = cv2.resize(
            image, (SHEET_THUMB_W, thumb_h), interpolation=cv2.INTER_AREA
        )
        if flagged:
            b = SHEET_BORDER
            sheet[y0:y0 + b, x0:x0 + SHEET_THUMB_W] = SHEET_FLAG
            sheet[y0 + thumb_h - b:y0 + thumb_h, x0:x0 + SHEET_THUMB_W] = SHEET_FLAG
            sheet[y0:y0 + thumb_h, x0:x0 + b] = SHEET_FLAG
            sheet[y0:y0 + thumb_h, x0 + SHEET_THUMB_W - b:x0 + SHEET_THUMB_W] = SHEET_FLAG
        for boxes, color in ((suspect_boxes, SUSPECT_COLOR), (flag_boxes, SHEET_FLAG)):
            for bx, by, bw, bh in boxes:
                cv2.rectangle(
                    sheet,
                    (x0 + round(bx * sx), y0 + round(by * sy)),
                    (x0 + round((bx + bw) * sx), y0 + round((by + bh) * sy)),
                    color, 2,
                )
        for j, line in enumerate(lines):
            color = SHEET_FLAG if j == 0 and flagged else SHEET_TEXT
            cv2.putText(
                sheet, line, (x0 + 4, y0 + thumb_h + 16 + 17 * j),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
    return sheet


def fmt2(v):
    """A score to two places for a contact-sheet label, or -- when it does not apply."""
    return "--" if v is None else f"{v:.2f}"


def sample_label(record, has_reference, light=False):
    """The three contact-sheet label lines for a sample."""
    whole = record["whole"]
    line1 = f"#{record['index']} f{record['frame']} {record['time']:.2f}s" + (" FLAG" if record["flagged"] else "")
    if light:
        line2 = f"lm ref {fmt2(whole['ssim_ref_lm'])}  prev {fmt2(whole['ssim_prev_lm'])}"
    else:
        line2 = f"ref {fmt2(whole['ssim_ref'])}  prev {fmt2(whole['ssim_prev'])}"
    metric = ("ssim_ref" if has_reference else "ssim_prev") + ("_lm" if light else "")
    best = None
    for name, values in record["crops"].items():
        value = values[metric]
        if value is not None and (best is None or value < best[1]):
            best = (name, value)
    if best is not None:
        line3 = f"{best[0][:14]} {best[1]:.2f}"
    else:
        line3 = f"loop {fmt2(whole['loop_score_ref'] if has_reference else whole['loop_score_prev'])}"
    return [line1, line2, line3]


def build_parser():
    parser = AgentParser(
        prog="frame_drift.py",
        description="Score a video take or still for drift vs a reference still and frame to frame.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, metavar="INPUT", help="the take (.mp4, .mov) or one still (.png, .jpg)")
    parser.add_argument(
        "--reference", type=Path, metavar="PATH", help="approved still; adds SSIM vs reference checks"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--every", type=int, default=None, metavar="N", help="sample every Nth frame (default 6)")
    group.add_argument(
        "--fps", dest="sample_fps", type=float, metavar="F", help="sample F frames a second instead of --every"
    )
    parser.add_argument("--start", type=float, default=0.0, metavar="S", help="first second to sample (default 0)")
    parser.add_argument("--end", type=float, metavar="S", help="stop before this second (default: the end)")
    parser.add_argument(
        "--size", type=parse_size, metavar="WxH", help="analysis size, e.g. 1920x1080 (default: native)"
    )
    parser.add_argument(
        "--crop", type=parse_crop, action="append", default=[], metavar="NAME=X,Y,W,H",
        help="named region in analysis pixels; repeatable",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, metavar="SSIM",
        help="flag SSIM vs reference below this (default 0.6)",
    )
    parser.add_argument(
        "--prev-threshold", type=float, default=DEFAULT_PREV_THRESHOLD, metavar="SSIM",
        help="flag SSIM vs previous sample below this (default 0.5)",
    )
    parser.add_argument(
        "--suspects", type=int, default=DEFAULT_SUSPECTS, metavar="N",
        help="changed regions to find per sample (default 3; 0 off)",
    )
    parser.add_argument(
        "--match-light", action="store_true", help="also score light-matched SSIM and flag on it instead"
    )
    parser.add_argument(
        "-o", "--out", type=Path, metavar="DIR", help="output folder (default output/<input stem>-drift)"
    )
    return parser


def read_image(path):
    """Read an image as BGR or raise DriftError."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DriftError(f"cannot read image: {path}")
    return image


def run(args):
    """Do the scoring; returns (json text, exit code)."""
    path = args.input
    if not path.is_file():
        raise DriftError(f"input not found: {path}")
    if args.reference is not None and not args.reference.is_file():
        raise DriftError(f"reference not found: {args.reference}")
    has_reference = args.reference is not None

    still = is_still(path)
    if still:
        still_image = read_image(path)
        native_h, native_w = still_image.shape[:2]
        planned = [0]
        fps = total = duration = None
        if args.every is not None or args.sample_fps is not None or args.start != 0 or args.end is not None:
            log("frame_drift: warning: --every, --fps, --start and --end do not apply to a still; ignored")
    else:
        try:
            fps, native_w, native_h, total, duration = load_video_info(str(path))
        except (RuntimeError, ZeroDivisionError):
            raise DriftError(f"cannot read video: {path}") from None
        planned = sample_frames(total, fps, args.every, args.sample_fps, args.start, args.end)
        if not planned:
            raise DriftError("no frames to sample between --start and --end")
        if len(planned) == 1 and not has_reference:
            raise DriftError("only one sample and no --reference: nothing to compare")

    size = args.size or (native_w, native_h)
    aw, ah = size
    check_crops(args.crop, size)
    ref_fit = ref_gray = ref_box = ref_native = None
    if has_reference:
        ref_img = read_image(args.reference)
        ref_native = [ref_img.shape[1], ref_img.shape[0]]
        ref_fit, ref_box = fit_to_size(ref_img, size)
        ref_gray = cv2.cvtColor(ref_fit, cv2.COLOR_BGR2GRAY)

    out = args.out if args.out is not None else Path("output") / f"{path.stem}-drift"
    if out.exists() and not out.is_dir():
        raise DriftError(f"output is not a folder: {out}")
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for old in (out / "frames").glob("s*-f*.png"):
        old.unlink()
    (out / "evidence").mkdir(exist_ok=True)
    for old in (out / "evidence").glob("s*.png"):
        old.unlink()
    out = out.resolve()

    kind = "still" if still else "video"
    log(
        f"frame_drift: {path} ({kind}, {native_w}x{native_h}"
        + (f", {fps:g} fps, {total} frames" if not still else "")
        + ")"
    )
    log(f"frame_drift: sampling {len(planned)} frames at {aw}x{ah}")

    crop_names = [name for name, _ in args.crop]
    checks = check_names(crop_names, has_reference, args.match_light)
    if still:
        source = [(0, still_image)]
    else:
        wanted = set(planned)
        source = ((n, f) for n, f in iter_frames(str(path), planned[0], planned[-1] + 1) if n in wanted)

    min_area = max(1, round(SUSPECT_MIN_AREA * aw * ah))
    boxes_by_region = {"whole": (0, 0, aw, ah), **dict(args.crop)}
    samples = []
    cells = []
    review = []
    prev = prev_gray = input_box = None
    for frame_num, frame in source:
        index = len(samples)
        fit, box = fit_to_size(frame, size)
        gray = cv2.cvtColor(fit, cv2.COLOR_BGR2GRAY)
        file = out / "frames" / f"s{index:04d}-f{frame_num:06d}.png"
        if not cv2.imwrite(str(file), fit):
            raise DriftError(f"cannot write {file}")
        ref_map = ssim_map(gray, ref_gray) if has_reference else None
        prev_map = ssim_map(gray, prev_gray) if prev is not None else None
        time = 0.0 if still else round(frame_num / fps, 4)
        record = {
            "index": index,
            "frame": frame_num,
            "time": time,
            "file": str(file),
            "flagged": False,
            "flags": [],
            "whole": region_metrics(
                fit, ref_fit, prev, ref_map, prev_map, (0, 0, aw, ah), gray, ref_gray, prev_gray, args.match_light
            ),
            "crops": {
                name: region_metrics(
                    fit, ref_fit, prev, ref_map, prev_map, box_, gray, ref_gray, prev_gray, args.match_light
                )
                for name, box_ in args.crop
            },
            "suspects": [],
            "evidence": [],
        }
        record["flags"] = sample_flags(record, checks, args.threshold, args.prev_threshold)
        record["flagged"] = bool(record["flags"])

        if prev_map is not None:
            found = suspect_regions(prev_map, args.suspects, min_area, SUSPECT_PAD)
        elif still and ref_map is not None:
            found = suspect_regions(ref_map, args.suspects, min_area, SUSPECT_PAD)
        else:
            found = []
        jobs = []  # (region, box, flagged, reason, suspect record or None)
        for region in ["whole", *crop_names]:
            failed = [f for f in record["flags"] if f.split(":")[0] == region]
            if failed:
                jobs.append((region, boxes_by_region[region], True, "flag:" + ",".join(failed), None))
        for k, found_one in enumerate(found, start=1):
            bx, by, bw, bh = found_one["box"]
            sl = (slice(by, by + bh), slice(bx, bx + bw))
            suspect = {
                "box": found_one["box"],
                "area": found_one["area"],
                "ssim_prev": None if prev_map is None else round(float(prev_map[sl].mean()), 4),
                "ssim_ref": None if ref_map is None else round(float(ref_map[sl].mean()), 4),
                "evidence": None,
            }
            record["suspects"].append(suspect)
            jobs.append((f"suspect{k}", found_one["box"], False, "suspect", suspect))
        for region, box_, flagged, reason, suspect in jobs:
            scores = box_scores(box_, ref_map, prev_map, gray, ref_gray, prev_gray, args.match_light)
            evidence_file = out / "evidence" / f"s{index:04d}-{region}.png"
            image = evidence_image(
                prev, fit, ref_fit, box_, evidence_header(record, region, box_, scores, args.match_light), flagged
            )
            if not cv2.imwrite(str(evidence_file), image):
                raise DriftError(f"cannot write {evidence_file}")
            if suspect is not None:
                suspect["evidence"] = str(evidence_file)
            record["evidence"].append({"region": region, "box": list(box_), "file": str(evidence_file)})
            review.append({
                "index": index,
                "frame": frame_num,
                "time": time,
                "region": region,
                "box": list(box_),
                "reason": reason,
                "scores": scores,
                "evidence": str(evidence_file),
            })
        flag_text = "  FLAG " + ", ".join(record["flags"]) if record["flags"] else ""
        log(f"  #{index} f{frame_num} {time:.2f}s" + flag_text)
        if index == 0:
            input_box = box
        prev, prev_gray = fit, gray
        samples.append(record)
        flag_boxes = [
            boxes_by_region[r] for r in crop_names if any(f.split(":")[0] == r for f in record["flags"])
        ]
        cells.append((
            fit, sample_label(record, has_reference, args.match_light), record["flagged"],
            [s["box"] for s in record["suspects"]], flag_boxes,
        ))

    if not samples:
        raise DriftError(f"no frames could be read from {path}")
    if len(samples) < len(planned):
        log(f"frame_drift: warning: read {len(samples)} of {len(planned)} planned samples; the stream ended early")

    summary = summarize(
        samples, crop_names, checks, args.threshold, args.prev_threshold, has_reference, args.match_light
    )
    suspect_total = sum(len(s["suspects"]) for s in samples)
    summary["review"] = len(review)
    summary["suspects"] = suspect_total
    verdict = verdict_line(summary)
    code = EXIT_FLAGGED if summary["flagged_samples"] or suspect_total else EXIT_CLEAN

    if has_reference:
        cells.insert(0, (ref_fit, ["REFERENCE", args.reference.name[:34], f"{aw}x{ah}"], False, [], []))
    sheet_path = out / "contact-sheet.png"
    cv2.imwrite(str(sheet_path), contact_sheet(cells, aw, ah))

    doc = {
        "tool": "frame_drift",
        "version": TOOL_VERSION,
        "verdict": verdict,
        "exit_code": code,
        "output_dir": str(out),
        "drift_json": str(out / "drift.json"),
        "contact_sheet": str(sheet_path),
        "legend": LEGEND,
        "input": {
            "path": str(path),
            "kind": kind,
            "native_size": [native_w, native_h],
            "fps": None if still else round(fps, 4),
            "total_frames": total,
            "duration": None if still else round(duration, 4),
            "crop_to_aspect": input_box,
        },
        "reference": None if not has_reference else {
            "path": str(args.reference),
            "native_size": ref_native,
            "crop_to_aspect": ref_box,
        },
        "settings": {
            "analysis_size": [aw, ah],
            "every": (args.every or DEFAULT_EVERY) if not still and args.sample_fps is None else None,
            "sample_fps": None if still else args.sample_fps,
            "start": None if still else args.start,
            "end": None if still else args.end,
            "threshold": args.threshold,
            "prev_threshold": args.prev_threshold,
            "match_light": args.match_light,
            "suspects": args.suspects,
            "ssim": SSIM_TEXT,
        },
        "crops": [{"name": name, "box": list(box_)} for name, box_ in args.crop],
        "summary": summary,
        "flagged": [
            {
                **{k: s[k] for k in ("index", "frame", "time", "file", "flags")},
                "evidence": [e["file"] for e in s["evidence"]],
            }
            for s in samples
            if s["flagged"]
        ],
        "review": review,
        "samples": samples,
    }
    text = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    (out / "drift.json").write_text(text)
    log(f"frame_drift: {verdict}")
    return text, code


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.every is not None and args.every < 1:
        parser.error("--every must be 1 or more")
    if args.sample_fps is not None and args.sample_fps <= 0:
        parser.error("--fps must be above 0")
    if args.suspects < 0:
        parser.error("--suspects must be 0 or more")
    if args.start < 0:
        parser.error("--start must be 0 or more")
    if args.end is not None and args.end <= args.start:
        parser.error("--end must be after --start")
    names = [name for name, _ in args.crop]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        parser.error(f"crop names must be unique: {', '.join(sorted(dupes))}")
    if is_still(args.input) and args.reference is None:
        parser.error("a still needs --reference: it has no previous frame to compare")

    real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            text, code = run(args)
    except DriftError as e:
        log(f"frame_drift: error: {e}")
        return EXIT_ERROR
    real_stdout.write(text)
    real_stdout.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())
