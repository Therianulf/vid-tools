# CLAUDE.md

## Rules

- Add to this file one line at a time: the thing and its rule or symptom, linked to the doc that
  holds the detail (a tool's own `.md` or its README section). Never a paragraph here.
- Every tool is a standalone Python script in the project root with a `main()` and an
  `if __name__ == "__main__"` guard, and the shebang `#!/usr/bin/env python3`.
- Simple tools parse `sys.argv`; tools with several flags or options use `argparse`.
- File paths use `pathlib.Path`; status and progress messages go to stdout.
- Output goes to `output/` (created automatically, gitignored) unless the user gives an explicit
  output path, which is used as given. Source and input media go in `src/` (gitignored).
- Video tools write both a simple and an interpolated variant, for side-by-side comparison.

## Contents

- `README.md` — install steps and usage for each tool that has a README section.
- `video_looper.py` — seamless loop by cross-fading the tail into the head; `--scan`,
  `--scan-shift`, `--loop-second`/`--loop-frame`, `--slowdown`, `--fade`; simple and
  interpolated outputs. [README](README.md#video_looperpy--true-video-looper).
- `video_slowmo_loop.py` — slow-motion loop with fade-to-black transitions and GPU frame
  interpolation, for clips that fade to black and restart.
  [README](README.md#video_slowmo_looppy--slow-motion-loop-creator).
- `gen_qr.py` — QR code from a URL as web PNG, print PNG and SVG.
  [README](README.md#gen_qrpy--qr-code-generator).
- `remove_bg.py` — AI background removal from an image (rembg/U2-Net).
  [README](README.md#remove_bgpy--background-removal).
- `despill_green.py` — green spill removal from transparent PNGs.
  [README](README.md#despill_greenpy--green-spill-removal).
- `clean_green.py` — aggressive green artifact removal from transparent PNGs.
  [README](README.md#clean_greenpy--green-artifact-removal).
- `video_remove_bg.py` — frame-by-frame rembg background removal from video, with temporal alpha
  smoothing; usage in its module docstring.
- `bubble_cleanup.py` — extracts a speech bubble from video by brightness contour, no AI; usage in
  its module docstring.
- `frames_to_video.py` — assembles RGBA PNG frames into transparent HEVC or ProRes 4444 MOV; usage
  in its module docstring.
- `test_transparency.py` — composites a transparent video or PNG over a checkerboard to check the
  alpha; usage in its module docstring.
- `trailer_cut.py` — assembles a short trailer from a JSON spec in `trailer-specs/` (beats,
  on-screen lines, voice-over, ducked music bed). [trailer-cut.md](trailer-cut.md).
- `trailer_cards.py` — renders the trailer's lower-third and end card PNGs with Pillow; used by
  `trailer_cut.py`. [trailer-cut.md](trailer-cut.md).
- `trailer_rings.py` — renders the broadcast rings overlay for a trailer beat;
  see [trailer-cut.md](trailer-cut.md) (Rings).

## Context

- vid-tools is a collection of standalone Python CLI tools for video and image editing.
- Layout: tools in the root; `tests/` (pytest); `trailer-specs/` (trailer JSON specs); `fonts/`
  (Atkinson Hyperlegible Next and its OFL, the trailer line font).
- Run: `source .venv/bin/activate`, then `python <tool>.py <args>`. The venv is Python 3.11.
- Test: `.venv/bin/python -m pytest -q` (pytest, `tests/`); from a worktree, use the main
  checkout's `.venv`. Lint: Ruff, target Python 3.10, line length 120, rules E/F/I/N/W/UP (config
  in `pyproject.toml`). Both come with the dev extra: `pip install -e '.[dev]'`.
- `pyproject.toml` manages dependencies (torch, torchvision, opencv-python, numpy, rembg, Pillow,
  segno; dev: pytest, ruff) and the console entry points.
- System dependencies: `ffmpeg` (`brew install ffmpeg`); for `trailer_cut.py`, `rsvg-convert`
  (`brew install librsvg`) and `libfribidi` (`brew install fribidi`).
