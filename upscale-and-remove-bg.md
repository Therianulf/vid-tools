# Upscale + Remove Background from AI-Generated Images

AI image generators often render "transparency" as a baked-in checkerboard pattern (alternating gray and white pixels). This isn't real transparency — it's just pixel data. Removing it cleanly requires a multi-step process.

## The Problem

Running `remove_bg.py` directly on the checkerboard image doesn't work well — rembg can't distinguish the checkerboard from the actual subject, so it either clips edges or produces bad alpha.

## The Solution: Green Screen + Mask Transplant

### Step 1: Regenerate with a solid green background

Go back to your image generator and regenerate the same image with a **solid green background** instead of the checkerboard. This gives the background removal AI a clear, high-contrast edge to cut against.

### Step 2: Upscale both versions with Real-ESRGAN

Upscale the original (checkerboard) and the green screen version at 4x:

```python
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path="weights/RealESRGAN_x4plus.pth",
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
)

for name in ["original.jpeg", "greenscreen.jpeg"]:
    img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, outscale=4)
    cv2.imwrite(f"{name.split('.')[0]}_4x.png", output)
```

Or install the `realesrgan` package and run them individually:

```bash
pip install realesrgan
```

> **Note:** If you hit `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`, patch the import in `.venv/.../basicsr/data/degradations.py` from `functional_tensor` to `functional`.

### Step 3: Remove background from the green screen version

```bash
python remove_bg.py greenscreen_4x.png greenscreen_4x_nobg.png
```

This produces a clean alpha mask because rembg can easily distinguish the subject from the solid green.

### Step 4: Transplant the alpha mask onto the original

The green screen version may have artifacts from regeneration. The original has the best artwork. Combine them — pixels from the original, alpha from the green screen:

```python
from PIL import Image
import numpy as np

# Get clean alpha mask from green screen removal
gs = Image.open("greenscreen_4x_nobg.png").convert("RGBA")
mask = np.array(gs)[:, :, 3]

# Apply mask to original upscaled image
original = Image.open("original_4x.png").convert("RGBA")
orig_arr = np.array(original)
orig_arr[:, :, 3] = mask

Image.fromarray(orig_arr).save("final_nobg.png")
```

## Result

You get a 4x upscaled image with clean, real transparency — best quality artwork from the original, clean edges from the green screen mask.

## Dependencies

```bash
pip install realesrgan rembg Pillow numpy opencv-python
```

System: `brew install ffmpeg`

## Why Not Just...

| Approach | Problem |
|---|---|
| `rembg` on the checkerboard directly | Can't distinguish checkerboard from subject; clips edges |
| Color-detect the checkerboard and delete it | Gray/white tones overlap with the subject; creates holes or demon artifacts |
| Replace checkerboard with solid color via code | Hard to detect checkerboard perfectly; color bleeds into subject edges |
| `rembg` on the green screen only | Regenerated image may have artifacts vs. the original |
| **Green screen mask + original pixels** | **Best of both worlds** |
