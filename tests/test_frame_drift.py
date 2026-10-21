"""Tests for frame_drift.py and the shared frame reader in video_looper.py (clip tests need ffmpeg)."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_drift import (
    DriftError,
    center_crop_box,
    check_crops,
    fit_to_size,
    main,
    match_light,
    parse_crop,
    parse_size,
    sample_flags,
    sample_frames,
    sheet_size,
    ssim,
    verdict_line,
)
from video_looper import frame_similarity, load_video_info, read_frame_at, scan_for_loop_points, scan_for_shifts

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS, FRAMES = 192, 108, 24, 48
STEADY = (8, 20, 40, 40)  # a crop that never changes
CHANGE = (140, 20, 40, 40)  # a crop replaced with other texture from frame 24
TOP_KEYS = [
    "tool", "version", "verdict", "exit_code", "output_dir", "drift_json", "contact_sheet", "legend",
    "input", "reference", "settings", "crops", "summary", "flagged", "samples",
]

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _old_scan_for_loop_points(video_path, fps, total_frames, start_second, count):
    """Scan video for frames most similar to frame 0. Returns ranked list."""
    frame_0 = read_frame_at(video_path, 0)
    start_frame = int(start_second * fps)

    if start_frame >= total_frames:
        print(f"Warning: scan-start ({start_second}s) is beyond video end, scanning from 50%")
        start_frame = total_frames // 2

    print(f"Scanning frames {start_frame}-{total_frames - 1} against frame 0...")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    scores = []
    for frame_num in range(start_frame, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        score = frame_similarity(frame_0, frame)
        scores.append((frame_num, frame_num / fps, score))

        if (frame_num - start_frame) % 30 == 0:
            pct = (frame_num - start_frame) / (total_frames - start_frame) * 100
            print(f"  Progress: {pct:.0f}% (frame {frame_num})")

    cap.release()

    # Sort by similarity score descending
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:count]


def _old_scan_for_shifts(video_path, fps, total_frames, start_second, count):
    """Scan video for frames where the biggest visual shift happens.

    Compares each frame to its predecessor and ranks by largest difference —
    finds natural transition moments that could mask a loop seam.
    """
    start_frame = max(1, int(start_second * fps))

    if start_frame >= total_frames:
        print(f"Warning: scan-start ({start_second}s) is beyond video end, scanning from 10%")
        start_frame = max(1, total_frames // 10)

    print(f"Scanning frames {start_frame}-{total_frames - 1} for visual shifts...")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []

    scores = []
    for frame_num in range(start_frame, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Compute frame-to-frame difference (inverse similarity = shift magnitude)
        diff = 1.0 - frame_similarity(prev_frame, frame)
        scores.append((frame_num, frame_num / fps, diff))
        prev_frame = frame

        if (frame_num - start_frame) % 30 == 0:
            pct = (frame_num - start_frame) / (total_frames - start_frame) * 100
            print(f"  Progress: {pct:.0f}% (frame {frame_num})")

    cap.release()

    # Sort by shift magnitude descending
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:count]


def texture(seed):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (H, W), dtype=np.uint8)
    grey = cv2.GaussianBlur(noise, (5, 5), 0)
    return np.repeat(grey[:, :, None], 3, axis=2)


def write_clip(path, frames):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clips")
    base = texture(1)
    x, y, w, h = CHANGE
    changed = base.copy()
    changed[y:y + h, x:x + w] = texture(2)[y:y + h, x:x + w]
    if shutil.which("ffmpeg") is not None:
        write_clip(folder / "clean.mp4", [base] * FRAMES)
        write_clip(folder / "change.mp4", [base] * (FRAMES // 2) + [changed] * (FRAMES // 2))
    padded = np.vstack([np.zeros((4, W, 3), np.uint8), base, np.zeros((4, W, 3), np.uint8)])
    cv2.imwrite(str(folder / "reference.png"), padded)
    cv2.imwrite(str(folder / "still.png"), base)
    return folder


@pytest.fixture(scope="module")
def runs(clips, tmp_path_factory):
    result = {}
    for name in ("clean", "change"):
        out = tmp_path_factory.mktemp(f"out-{name}")
        cmd = [
            sys.executable, str(ROOT / "frame_drift.py"), str(clips / f"{name}.mp4"),
            "--reference", str(clips / "reference.png"),
            "--crop", "steady=8,20,40,40", "--crop", "change=140,20,40,40", "-o", str(out),
        ]
        result[name] = (subprocess.run(cmd, capture_output=True, text=True), out)
    return result


def gray_of(seed):
    return cv2.cvtColor(texture(seed), cv2.COLOR_BGR2GRAY)


def test_ssim_identical_is_one():
    img = gray_of(1)
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-9)


def test_ssim_constant_images_follow_c1():
    a = np.full((20, 20), 100, np.uint8)
    b = np.full((20, 20), 110, np.uint8)
    assert ssim(a, b) == pytest.approx(22006.5025 / 22106.5025, rel=1e-9)


def test_ssim_drops_with_noise():
    img = gray_of(1)
    rng = np.random.default_rng(7)

    def noisy(sigma):
        return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

    s10 = ssim(img, noisy(10))
    s40 = ssim(img, noisy(40))
    assert 1 > s10 > s40
    assert s10 < 0.999


def test_center_crop_box():
    assert center_crop_box(1920, 1088, 1280, 720) == (0, 4, 1920, 1080)
    assert center_crop_box(1920, 1088, 1920, 1080) == (0, 4, 1920, 1080)
    assert center_crop_box(1280, 720, 1920, 1080) == (0, 0, 1280, 720)
    assert center_crop_box(2000, 1080, 1920, 1080) == (40, 0, 1920, 1080)
    assert center_crop_box(1200, 1200, 1920, 1080) == (0, 262, 1200, 675)


def test_fit_to_size_crops_before_resizing():
    image = np.zeros((1088, 1920, 3), np.uint8)
    image[0:4] = 255
    image[1084:1088] = 255
    fitted, box = fit_to_size(image, (1280, 720))
    assert box == [0, 4, 1920, 1080]
    assert fitted.shape == (720, 1280, 3)
    assert fitted.max() == 0
    fitted, box = fit_to_size(np.zeros((720, 1280, 3), np.uint8), (1920, 1080))
    assert box is None
    assert fitted.shape == (1080, 1920, 3)


def test_parse_crop():
    assert parse_crop("face_left=400,240,240,260") == ("face_left", (400, 240, 240, 260))
    for bad in ("face=1,2,3", "face=1,2,0,4", "whole=0,0,1,1", "bad name=1,2,3,4"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_crop(bad)


def test_parse_size():
    assert parse_size("1920x1080") == (1920, 1080)
    assert parse_size("1920X1080") == (1920, 1080)
    for bad in ("0x10", "1920*1080"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_size(bad)


def test_sample_frames():
    assert sample_frames(48, 24, every=6) == [0, 6, 12, 18, 24, 30, 36, 42]
    assert len(sample_frames(48, 24, every=5)) == 10
    assert sample_frames(48, 24, every=6, start=0.5, end=1.5) == [12, 18, 24, 30]
    assert sample_frames(48, 24, sample_fps=4) == sample_frames(48, 24, every=6)
    assert sample_frames(48, 24, sample_fps=5) == [0, 5, 10, 14, 19, 24, 29, 34, 38, 43]
    assert sample_frames(48, 24, sample_fps=48) == list(range(48))
    assert sample_frames(48, 24, start=3.0) == []


def test_check_crops():
    check_crops([("a", (152, 68, 40, 40))], (192, 108))
    with pytest.raises(DriftError):
        check_crops([("a", (153, 68, 40, 40))], (192, 108))


def test_sample_flags():
    record = {
        "whole": {"ssim_ref": 0.9, "ssim_prev": None},
        "crops": {"face": {"ssim_ref": 0.59, "ssim_prev": None}},
    }
    checks = ["whole:ssim_ref", "whole:ssim_prev", "face:ssim_ref", "face:ssim_prev"]
    assert sample_flags(record, checks, 0.6, 0.5) == ["face:ssim_ref<0.6"]
    record["crops"]["face"]["ssim_ref"] = 0.6
    assert sample_flags(record, checks, 0.6, 0.5) == []


def make_summary(flagged, failed, first_below):
    checks = {}
    for name in ("whole:ssim_ref", "whole:ssim_prev", "face_left:ssim_ref", "face_left:ssim_prev"):
        checks[name] = {"threshold": 0.5 if name.endswith("prev") else 0.6, "failed": 0, "first_below": None}
    for name, count in failed.items():
        checks[name]["failed"] = count
        checks[name]["first_below"] = first_below[name]
    return {"samples": 21, "flagged_samples": flagged, "checks": checks}


def test_verdict_line():
    assert verdict_line(make_summary(0, {}, {})) == "CLEAN: 21 samples, all above thresholds"
    summary = make_summary(
        7,
        {"face_left:ssim_ref": 7, "face_left:ssim_prev": 2},
        {
            "face_left:ssim_ref": {"index": 7, "frame": 42, "time": 1.75},
            "face_left:ssim_prev": {"index": 9, "frame": 54, "time": 2.25},
        },
    )
    assert verdict_line(summary) == (
        "FLAGGED: face_left below 0.6 SSIM vs reference from 1.75 s (7 of 21 samples); also face_left:ssim_prev"
    )


def test_sheet_size():
    assert sheet_size(9, 192, 108) == (6, 2, 180, 1976, 492)
    assert sheet_size(1, 1920, 1080) == (1, 1, 180, 336, 250)


def test_help_fits_one_screen(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "80")
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) <= 40
    assert "stdout: one JSON document" in out
    assert "exit: 0 clean, 2" in out


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["x.mp4", "--every", "2", "--fps", "3"],
        ["x.mp4", "--crop", "bad"],
        ["x.mp4", "--crop", "a=0,0,4,4", "--crop", "a=1,1,4,4"],
        ["x.png"],
    ],
)
def test_usage_errors_exit_1_without_stdout(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 1
    assert capsys.readouterr().out == ""


def test_missing_input_exit_1(tmp_path, capsys):
    assert main([str(tmp_path / "none.mp4")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "input not found" in captured.err


@needs_ffmpeg
def test_scan_for_loop_points_unchanged(clips, capsys):
    path = str(clips / "change.mp4")
    fps, _, _, total, _ = load_video_info(path)
    new = scan_for_loop_points(path, fps, total, 0.5, 5)
    new_out = capsys.readouterr().out
    old = _old_scan_for_loop_points(path, fps, total, 0.5, 5)
    old_out = capsys.readouterr().out
    assert new == old
    assert new_out == old_out


@needs_ffmpeg
def test_scan_for_shifts_unchanged(clips, capsys):
    path = str(clips / "change.mp4")
    fps, _, _, total, _ = load_video_info(path)
    new = scan_for_shifts(path, fps, total, 0.25, 5)
    new_out = capsys.readouterr().out
    old = _old_scan_for_shifts(path, fps, total, 0.25, 5)
    old_out = capsys.readouterr().out
    assert new == old
    assert new_out == old_out


@needs_ffmpeg
def test_stdout_is_one_json_document(runs):
    for name in ("clean", "change"):
        proc, out = runs[name]
        json.loads(proc.stdout)
        assert proc.stdout == (out / "drift.json").read_text()
        assert "frame_drift:" in proc.stderr


@needs_ffmpeg
def test_exit_codes(runs):
    proc, _ = runs["clean"]
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["verdict"].startswith("CLEAN: 8 samples")
    proc, _ = runs["change"]
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["verdict"] == (
        "FLAGGED: change below 0.6 SSIM vs reference from 1.00 s (4 of 8 samples); also change:ssim_prev"
    )


@needs_ffmpeg
def test_changed_crop_flagged_from_right_sample(runs):
    doc = json.loads(runs["change"][0].stdout)
    ref_flagged = [s["index"] for s in doc["samples"] if "change:ssim_ref<0.6" in s["flags"]]
    assert ref_flagged == [4, 5, 6, 7]
    prev_flagged = [s["index"] for s in doc["samples"] if "change:ssim_prev<0.5" in s["flags"]]
    assert prev_flagged == [4]
    assert not [f for s in doc["samples"] for f in s["flags"] if f.startswith(("steady:", "whole:"))]
    assert doc["summary"]["checks"]["change:ssim_ref"]["first_below"] == {"index": 4, "frame": 24, "time": 1.0}


@needs_ffmpeg
def test_json_shape(runs):
    doc = json.loads(runs["change"][0].stdout)
    assert list(doc) == TOP_KEYS
    assert doc["reference"]["crop_to_aspect"] == [0, 4, 192, 108]
    assert doc["settings"]["analysis_size"] == [192, 108]
    assert [s["frame"] for s in doc["samples"]] == [0, 6, 12, 18, 24, 30, 36, 42]
    assert doc["crops"] == [{"name": "steady", "box": [8, 20, 40, 40]}, {"name": "change", "box": [140, 20, 40, 40]}]
    assert doc["samples"][0]["whole"]["ssim_prev"] is None
    assert all(Path(s["file"]).is_file() for s in doc["samples"])
    assert len(doc["flagged"]) == 4
    assert set(doc["legend"]) >= {"ssim_ref", "ssim_prev", "loop_score_ref", "loop_score_prev"}


@needs_ffmpeg
def test_contact_sheet_marks_flagged(runs):
    doc = json.loads(runs["change"][0].stdout)
    sheet = cv2.imread(doc["contact_sheet"])
    assert sheet.shape[:2] == (492, 1976)
    assert sheet[9, 1649].tolist() == [0, 0, 255]
    assert sheet[9, 337].tolist() != [0, 0, 255]


def test_still_input(clips, tmp_path, capsys):
    code = main([str(clips / "still.png"), "--reference", str(clips / "reference.png"), "-o", str(tmp_path)])
    assert code == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["input"]["kind"] == "still"
    assert doc["summary"]["samples"] == 1
    assert doc["samples"][0]["whole"]["ssim_prev"] is None
    assert doc["samples"][0]["whole"]["ssim_ref"] > 0.99
    assert doc["input"]["fps"] is None


@needs_ffmpeg
def test_crop_outside_frame_exit_1(clips, tmp_path, capsys):
    code = main([str(clips / "change.mp4"), "--crop", "a=180,100,40,40", "-o", str(tmp_path)])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "outside the 192x108" in captured.err


def light_texture():
    """A textured 200x200 uint8 image, stretched to 0..255."""
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, (200, 200), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (7, 7), 0).astype(np.float64)
    return ((blurred - blurred.min()) / (blurred.max() - blurred.min()) * 255).round().astype(np.uint8)


def change_light(img):
    """Gamma 2.2, then a contrast and brightness change."""
    lit = (img / 255.0) ** 2.2 * 255 * 0.7 + 30
    return np.clip(lit, 0, 255).astype(np.uint8)


def test_match_light_identity_and_monotone_lut():
    img = light_texture()
    assert np.array_equal(match_light(img, img), img)
    ref = change_light(img)
    mapped = match_light(img, ref)
    order = np.argsort(img.ravel(), kind="stable")
    assert (np.diff(mapped.ravel()[order].astype(int)) >= 0).all()


def test_match_light_forgives_light_not_shape():
    ref = light_texture()
    light = change_light(ref)
    shape = np.roll(ref, 15, axis=1)
    plain = ssim(light, ref)
    matched = ssim(match_light(light, ref), ref)
    shifted = ssim(match_light(shape, ref), ref)
    print(f"plain {plain:.4f} matched {matched:.4f} shifted {shifted:.4f}")
    assert matched > 0.9
    assert matched - plain >= 0.15
    assert shifted < 0.6


@pytest.fixture(scope="module")
def light_clip(tmp_path_factory):
    folder = tmp_path_factory.mktemp("light")
    base = texture(1)
    x, y, w, h = CHANGE
    lit = base.copy()
    lit[y:y + h, x:x + w] = change_light(base[y:y + h, x:x + w])
    cv2.imwrite(str(folder / "reference.png"), base)
    if shutil.which("ffmpeg") is not None:
        write_clip(folder / "light.mp4", [base] * (FRAMES // 2) + [lit] * (FRAMES // 2))
    return folder, ssim(gray_of(1)[y:y + h, x:x + w], cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w])


def run_light(folder, tmp_path, threshold, *extra):
    cmd = [
        sys.executable, str(ROOT / "frame_drift.py"), str(folder / "light.mp4"),
        "--reference", str(folder / "reference.png"), "--crop", "lit=140,20,40,40",
        "--threshold", str(threshold), "-o", str(tmp_path), *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


@needs_ffmpeg
def test_match_light_cli(light_clip, tmp_path):
    folder, plain = light_clip
    print(f"plain crop SSIM {plain:.4f}")
    threshold = round(plain + 0.05, 2)
    off = run_light(folder, tmp_path / "off", threshold)
    assert off.returncode == 2
    on = run_light(folder, tmp_path / "on", threshold, "--match-light")
    assert on.returncode == 0, on.stdout
    doc = json.loads(on.stdout)
    assert doc["settings"]["match_light"] is True
    assert all(key.endswith("_lm") for key in (k.split(":")[1] for k in doc["summary"]["checks"]))
    assert doc["samples"][-1]["crops"]["lit"]["ssim_ref"] < threshold
    assert doc["samples"][-1]["crops"]["lit"]["ssim_ref_lm"] >= threshold


@needs_ffmpeg
def test_default_run_has_null_light_metrics(runs):
    doc = json.loads(runs["change"][0].stdout)
    assert doc["version"] == "1.1"
    assert doc["settings"]["match_light"] is False
    for sample in doc["samples"]:
        for region in (sample["whole"], *sample["crops"].values()):
            assert region["ssim_ref_lm"] is None
            assert region["ssim_prev_lm"] is None
