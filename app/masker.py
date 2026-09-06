"""Apply a masking effect (mosaic / white fill) over detected regions.

High-quality AI inpainting lives in inpainter.py (it needs a separate, heavier
model) and is dispatched to from media.py instead of going through here.
"""
from __future__ import annotations

import cv2
import numpy as np

MODE_MOSAIC = "mosaic"
MODE_FILL = "fill"
MODE_INPAINT = "inpaint"

FILL_COLOR = (255, 255, 255)  # BGR white


def _expand_box(x1, y1, x2, y2, w, h, margin_ratio: float):
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin_ratio), int(bh * margin_ratio)
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(w, x2 + mx),
        min(h, y2 + my),
    )


def _mosaic_region(region: np.ndarray, cells: int) -> np.ndarray:
    """Pixelate so that the region's shorter side is split into `cells` tiles.

    Sizing the tile in absolute pixels doesn't scale with the plate's size in
    the frame (a close-up plate stays readable at a "fine" block size that
    would be plenty coarse on a distant plate), so we derive it from the
    region itself instead.
    """
    h, w = region.shape[:2]
    if h == 0 or w == 0:
        return region
    block = max(4, min(h, w) // cells)
    small_w = max(1, w // block)
    small_h = max(1, h // block)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _rounded_alpha_mask(h: int, w: int, radius_ratio: float = 0.14, feather_px: float = 0.0) -> np.ndarray:
    """Alpha mask (0..1) that is fully opaque everywhere except the four
    corners, which are cut on a rounded-rect curve.

    A YOLO box is axis-aligned, but a plate photographed at an angle is a
    tilted quadrilateral, so its tight bounding box has a small triangular
    sliver of background poking past each corner. Rounding the corners trims
    exactly those slivers (real plates are rounded-rect anyway) without ever
    removing coverage from the flat edges, where the actual plate digits are.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = max(1, int(min(h, w) * radius_ratio))
    cv2.rectangle(mask, (radius, 0), (max(radius, w - radius), h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, max(radius, h - radius)), 255, -1)
    for cx, cy in (
        (radius, radius),
        (w - radius, radius),
        (radius, h - radius),
        (w - radius, h - radius),
    ):
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    if feather_px > 0:
        blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_px)
        # max() keeps the solid interior at full opacity (never re-reveals
        # plate pixels) and only softens the outer edge/corner transition.
        mask = np.maximum(mask, blurred)

    return mask.astype(np.float32) / 255.0


def _blend_effect(original: np.ndarray, effect: np.ndarray) -> np.ndarray:
    h, w = original.shape[:2]
    feather = max(1.5, min(h, w) * 0.03)
    alpha = _rounded_alpha_mask(h, w, feather_px=feather)[..., None]
    blended = effect.astype(np.float32) * alpha + original.astype(np.float32) * (1 - alpha)
    return blended.astype(np.uint8)


def mask_regions(
    frame_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int, float]],
    mode: str = MODE_MOSAIC,
    margin_ratio: float = 0.0,
    mosaic_cells: int = 8,
) -> np.ndarray:
    """Mask only the pixels inside each detected plate box (no bleed onto the
    surrounding frame/bumper) unless margin_ratio explicitly widens it."""
    if not boxes:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    for x1, y1, x2, y2, _ in boxes:
        ex1, ey1, ex2, ey2 = _expand_box(x1, y1, x2, y2, w, h, margin_ratio)
        if ex2 <= ex1 or ey2 <= ey1:
            continue
        original = out[ey1:ey2, ex1:ex2]
        if mode == MODE_FILL:
            effect = np.full_like(original, FILL_COLOR)
        else:  # MODE_MOSAIC
            effect = _mosaic_region(original, mosaic_cells)
        out[ey1:ey2, ex1:ex2] = _blend_effect(original, effect)

    return out
