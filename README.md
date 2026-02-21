# vid-tools

A collection of CLI tools for video and image editing.

## Tools

### video_looper.py — True Video Looper

Creates seamlessly looping videos by finding a loop cut point and trimming. Outputs both a simple (PTS-stretched) and frame-interpolated version for comparison.

```bash
# Scan for the best loop points (compares frames against frame 0)
python video_looper.py input.mp4 --scan

# Create a loop at a specific time
python video_looper.py input.mp4 --loop-second 4.7

# Loop + slow down 2x
python video_looper.py input.mp4 --loop-second 4.7 --slowdown 2.0

# Sub-second precision (frame offset within that second)
python video_looper.py input.mp4 --loop-second 4.7 --loop-frame 2
```

### video_slowmo_loop.py — Slow-Motion Loop Creator

Transforms short AI-generated video clips into slow-motion, seamlessly looping backgrounds with fade transitions and GPU-accelerated frame interpolation.

```bash
python video_slowmo_loop.py input.mp4 --factor 4 --fade 1.0
python video_slowmo_loop.py input.mp4 --no-interpolation   # fast, frame duplication only
```

### gen_qr.py — QR Code Generator

Generates QR codes in multiple formats (web PNG, print PNG, SVG) from a URL.

```bash
python gen_qr.py example.com/page
```

### remove_bg.py — Background Removal

Removes image backgrounds using AI (U2-Net via rembg).

```bash
python remove_bg.py photo.jpg
python remove_bg.py photo.jpg custom_output.png
```

### despill_green.py — Green Spill Removal

Removes green spill/fringe from edges of transparent PNGs (compositing cleanup).

```bash
python despill_green.py keyed_image.png
```

### clean_green.py — Green Artifact Removal

Aggressively removes remaining green artifacts from transparent PNGs.

```bash
python clean_green.py keyed_image.png
```

## Installation

```bash
git clone <repo-url>
cd vid-tools
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

System dependency:

```bash
brew install ffmpeg
```

## Output

All tools write to `./output/` by default. You can override this with an explicit output path argument where supported.

## License

MIT
