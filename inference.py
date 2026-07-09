# ------------------------------------------------------------------------
# inference.py
# Thin wrapper around the vendored NAFNet architecture (models/NAFNet_arch.py)
# that:
#   1. builds the correct NAFNet / NAFNetLocal network for a given task+width
#   2. loads a pretrained .pth checkpoint from ./weights if present
#   3. runs single-image inference (denoise / deblur)
#   4. implements two extra, non-NAFNet "bonus" features with classical
#      image processing so the site is useful even before weights are
#      downloaded: low-light enhancement and quick upscale/sharpen.
# ------------------------------------------------------------------------
import os
import time
import numpy as np
import cv2
import torch

from models.NAFNet_arch import NAFNet, NAFNetLocal

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')

# Network definitions taken directly from the official test configs
# (options/test/SIDD/*.yml and options/test/GoPro/*.yml in the original repo)
TASKS = {
    'denoise': {
        'label': 'Denoise (SIDD)',
        'arch': 'NAFNet',
        'variants': {
            32: {'enc_blk_nums': [2, 2, 4, 8], 'middle_blk_num': 12, 'dec_blk_nums': [2, 2, 2, 2],
                 'weight_file': 'NAFNet-SIDD-width32.pth'},
            64: {'enc_blk_nums': [2, 2, 4, 8], 'middle_blk_num': 12, 'dec_blk_nums': [2, 2, 2, 2],
                 'weight_file': 'NAFNet-SIDD-width64.pth'},
        }
    },
    'deblur': {
        'label': 'Deblur (GoPro)',
        'arch': 'NAFNetLocal',
        'variants': {
            32: {'enc_blk_nums': [1, 1, 1, 28], 'middle_blk_num': 1, 'dec_blk_nums': [1, 1, 1, 1],
                 'weight_file': 'NAFNet-GoPro-width32.pth'},
            64: {'enc_blk_nums': [1, 1, 1, 28], 'middle_blk_num': 1, 'dec_blk_nums': [1, 1, 1, 1],
                 'weight_file': 'NAFNet-GoPro-width64.pth'},
        }
    },
}

_MODEL_CACHE = {}


def weight_status():
    """Report which pretrained checkpoints are present so the UI can show
    the user exactly what still needs to be downloaded."""
    status = {}
    for task, cfg in TASKS.items():
        status[task] = {}
        for width, variant in cfg['variants'].items():
            path = os.path.join(WEIGHTS_DIR, variant['weight_file'])
            status[task][width] = {
                'file': variant['weight_file'],
                'present': os.path.isfile(path),
            }
    return status


def _build_network(task, width):
    cfg = TASKS[task]
    variant = cfg['variants'][width]
    common = dict(img_channel=3, width=width,
                  middle_blk_num=variant['middle_blk_num'],
                  enc_blk_nums=variant['enc_blk_nums'],
                  dec_blk_nums=variant['dec_blk_nums'])
    if cfg['arch'] == 'NAFNet':
        return NAFNet(**common)
    else:
        return NAFNetLocal(**common, train_size=(1, 3, 256, 256), fast_imp=True)


def load_model(task, width, device='cpu'):
    key = (task, width, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if task not in TASKS or width not in TASKS[task]['variants']:
        raise ValueError(f'Unknown task/width combination: {task}/{width}')

    variant = TASKS[task]['variants'][width]
    weight_path = os.path.join(WEIGHTS_DIR, variant['weight_file'])
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Pretrained weight '{variant['weight_file']}' was not found in ./weights. "
            f"Download it from the official NAFNet model zoo and place it there."
        )

    net = _build_network(task, width)
    state = torch.load(weight_path, map_location=device)
    state = state.get('params', state)
    net.load_state_dict(state, strict=True)
    net.eval()
    net.to(device)

    _MODEL_CACHE[key] = net
    return net


@torch.no_grad()
def run_nafnet(bgr_img, task, width, device='cpu'):
    """Run real NAFNet inference on a BGR uint8 numpy image, return BGR uint8."""
    model = load_model(task, width, device=device)

    img = bgr_img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, 0..1
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    start = time.time()
    output = model(tensor)
    elapsed = time.time() - start

    output = output.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    output = (output * 255.0).round().astype(np.uint8)[:, :, ::-1]  # RGB->BGR
    return output, elapsed


# ---------------------------------------------------------------------------
# Bonus, non-NAFNet classical features. NAFNet's public repo/weights only
# cover denoising (SIDD) and deblurring (GoPro/REDS) - there is no official
# low-light or single-image super-resolution checkpoint (NAFSSR is a
# *stereo* super-resolution model that needs a left+right image pair). These
# two features fill those gaps with fast, dependency-light OpenCV pipelines
# so the "mini project" covers 4 tasks end-to-end without requiring extra
# multi-gigabyte model downloads.
# ---------------------------------------------------------------------------

def enhance_low_light(bgr_img, clip_limit=2.5, gamma=0.8):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # gamma correction to lift shadows further
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    out = cv2.LUT(out, table)
    return out


def upscale_and_sharpen(bgr_img, scale=2):
    h, w = bgr_img.shape[:2]
    upscaled = cv2.resize(bgr_img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # unsharp mask
    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    return sharpened
