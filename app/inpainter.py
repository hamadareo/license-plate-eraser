"""High-quality AI inpainting for license plate regions using a pretrained LaMa model.

Runs on small crops around each detected box (not the whole frame) so it stays fast
enough for video, while still giving the network enough surrounding context to
paint in a plausible continuation of the background.
"""
from __future__ import annotations

import os

import certifi
import cv2
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from simple_lama_inpainting import SimpleLama  # noqa: E402

from .masker import _rounded_alpha_mask

_MODEL_CACHE: SimpleLama | None = None

# Extra visible (unmasked) pixels around each plate fed to the model as context.
CONTEXT_PADDING = 80


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lama() -> SimpleLama:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = SimpleLama(device=_pick_device())
    return _MODEL_CACHE


def preload():
    """Load (and download, on first run) the LaMa model ahead of time."""
    get_lama()


def inpaint_regions(
    frame_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int, float]],
) -> np.ndarray:
    if not boxes:
        return frame_bgr

    lama = get_lama()
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for x1, y1, x2, y2, _ in boxes:
        pad = max(CONTEXT_PADDING, x2 - x1, y2 - y1)
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = out[cy1:cy2, cx1:cx2]
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        # Mask only the exact detected plate box: nothing outside it is touched.
        mx1, my1 = x1 - cx1, y1 - cy1
        mx2, my2 = x2 - cx1, y2 - cy1
        mask[max(0, my1) : my2, max(0, mx1) : mx2] = 255

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = lama(Image.fromarray(crop_rgb), Image.fromarray(mask))
        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result_bgr = result_bgr[: crop.shape[0], : crop.shape[1]]

        # LaMa returns a full reconstructed crop, but its autoencoder can
        # very slightly alter pixels outside the masked box too (a faint
        # blur/tone shift). Only blend in the plate box itself -- with the
        # same rounded, feathered mask used for mosaic/fill -- so everything
        # else in the crop (the surrounding bumper/frame) stays byte-for-byte
        # original, and tilted plates don't leave AI-smudged corners poking
        # past their real edges.
        box_h, box_w = my2 - my1, mx2 - mx1
        if box_h > 0 and box_w > 0:
            alpha = np.zeros(crop.shape[:2], dtype=np.float32)
            feather = max(1.5, min(box_h, box_w) * 0.03)
            alpha[my1:my2, mx1:mx2] = _rounded_alpha_mask(box_h, box_w, feather_px=feather)
            alpha = alpha[..., None]
            blended = result_bgr.astype(np.float32) * alpha + crop.astype(np.float32) * (1 - alpha)
            out[cy1:cy2, cx1:cx2] = blended.astype(np.uint8)

    return out
