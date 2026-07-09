# NAFNet Restoration Studio

A Flask website built around the official **NAFNet** (Nonlinear Activation
Free Network) image-restoration model, extended with a couple of bonus
classical tools, for a mini-project style demo.

## What's inside

| Feature | How it works |
|---|---|
| **Denoise** | Real NAFNet inference (SIDD-trained weights), width 32 or 64 |
| **Deblur** | Real NAFNet inference (GoPro-trained weights, TLC/local variant), width 32 or 64 |
| **Low-light enhance** | Classical CLAHE (LAB L-channel) + gamma correction — NAFNet's public repo has no official low-light checkpoint, so this fills the gap |
| **Upscale & sharpen** | Classical Lanczos resize + unsharp mask — NAFNet's only SR model (NAFSSR) needs a stereo left/right pair, so this is a lightweight single-image stand-in |
| **Combine tools** | Optionally pre-brighten before denoise/deblur, or upscale the result afterwards |
| **Batch mode** | Upload many images at once, process them all with the same settings, download everything as one `.zip` |
| **History** | Session gallery of everything you've processed |
| **Weight status panel** | Shows exactly which `.pth` checkpoints are present/missing and what to do about it |

The site is a plain Flask app + vanilla HTML/CSS/JS — no build step, no
frontend framework required.

## 1. Install

```bash
cd nafnet_webapp
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

If you're on a machine without a GPU, the CPU-only PyTorch wheel is smaller
and installs faster:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. Add pretrained weights (optional but needed for Denoise/Deblur)

See [`weights/README.md`](weights/README.md) — short version: download the
official `.pth` files from the NAFNet model zoo (linked in
`reference/NAFNet-original-readme.md`) and drop them into `weights/`.

The **Low-light enhance** and **Upscale & sharpen** tools need no weights at
all and work immediately.

## 3. Run

```bash
python app.py
```

Then open **http://localhost:5000**.

## Project layout

```
app.py                  Flask routes (single image, batch, history, zip download)
inference.py             NAFNet model builder/loader + the two classical tools
models/                  Vendored NAFNet architecture (from the original repo)
weights/                 Drop your .pth checkpoints here
templates/index.html     Single-page UI
static/css/style.css     Styling
static/js/main.js        Upload, before/after slider, batch, history logic
static/uploads /results  Runtime-generated images (gitignored)
demo/                    Two sample images (noisy.png, blurry.jpg) to try instantly
reference/               Original NAFNet readme, kept for the weight download links
```

## Notes & honesty about scope

- Denoise and Deblur run the **real NAFNet architecture** with whatever
  official checkpoint you supply — no shortcuts there.
- Low-light and Upscale are clearly labeled in the UI as classical,
  non-NAFNet additions, since the original project doesn't ship models for
  those exact tasks (single-image SR isn't in scope for NAFNet; only the
  stereo NAFSSR variant is, and low-light isn't covered at all).
- Everything runs on CPU by default (`device='cpu'` in `inference.py`); if
  you have a CUDA GPU available, change that string to `'cuda'` for a large
  speed-up.
- History is stored in memory per server process — restart clears it. Swap
  in a real database if you need persistence across restarts.
