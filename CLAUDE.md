# CLAUDE.md

## Project Overview

vid-tools is a collection of standalone Python CLI tools for video and image editing.

## Project Structure

- All tools are standalone Python scripts in the project root
- Each script has a `main()` function and `if __name__ == "__main__"` guard
- Tools output to the `output/` directory by default (created automatically, gitignored)
- Source/input media goes in `src/` (gitignored)
- `pyproject.toml` manages dependencies and entry points

## Tools

- `video_looper.py` — True video looper. Cross-fades tail into head for seamless loops. Has `--scan` (find frames similar to frame 0), `--scan-shift` (find natural visual transition points), `--loop-second`/`--loop-frame` (set cut point), `--slowdown` (slow down), `--fade` (cross-fade duration). Outputs both a simple (frame duplication) and interpolated (optical flow) version.
- `video_slowmo_loop.py` — Slow-motion loop with fade-to-black transitions and GPU-accelerated frame interpolation. For clips that fade to black and restart.
- `gen_qr.py` — QR code generator (web PNG, print PNG, SVG from a URL)
- `remove_bg.py` — AI background removal (rembg/U2-Net)
- `despill_green.py` — Green spill removal from transparent PNGs
- `clean_green.py` — Aggressive green artifact removal from transparent PNGs

## Conventions

- Shebang: `#!/usr/bin/env python3`
- Simple tools use `sys.argv` for argument parsing
- Complex tools (multiple flags/options) use `argparse`
- File paths use `pathlib.Path`
- Status/progress messages go to stdout
- When user provides an explicit output path, respect it; otherwise default to `output/`
- Video tools produce both simple and interpolated variants for human comparison

## Running Tools

```bash
source .venv/bin/activate
python <tool>.py <args>
```

Venv is Python 3.11. System dependency: `ffmpeg` (install via `brew install ffmpeg`)

## Dependencies

Managed in `pyproject.toml`: torch, torchvision, opencv-python, numpy, rembg, Pillow, segno

## Linting

Ruff: target Python 3.10, line length 120, rules E/F/I/N/W/UP
