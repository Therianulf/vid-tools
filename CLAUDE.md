# CLAUDE.md

## Project Overview

vid-tools is a collection of standalone Python CLI tools for video and image editing.

## Project Structure

- All tools are standalone Python scripts in the project root
- Each script has a `main()` function and `if __name__ == "__main__"` guard
- Tools output to the `output/` directory by default (created automatically, gitignored)
- `pyproject.toml` manages dependencies and entry points

## Conventions

- Shebang: `#!/usr/bin/env python3`
- Simple tools use `sys.argv` for argument parsing
- Complex tools (multiple flags/options) use `argparse`
- File paths use `pathlib.Path`
- Status/progress messages go to stdout
- When user provides an explicit output path, respect it; otherwise default to `output/`

## Running Tools

```bash
source .venv/bin/activate
python <tool>.py <args>
```

System dependency: `ffmpeg` (install via `brew install ffmpeg`)

## Dependencies

Managed in `pyproject.toml`: torch, torchvision, opencv-python, numpy, rembg, Pillow, segno

## Linting

Ruff: target Python 3.10, line length 120, rules E/F/I/N/W/UP
