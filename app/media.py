"""Image / video I/O and the per-file processing pipeline."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .detector import PlateDetector
from .inpainter import inpaint_regions
from .masker import MODE_INPAINT, mask_regions
from .tracker import PlateTracker

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

DOWNLOADS_DIR = Path.home() / "Downloads"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def is_supported(path: Path) -> bool:
    return is_image(path) or is_video(path)


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: Path, img: np.ndarray) -> None:
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    buf.tofile(str(path))


def output_path_for(src: Path, suffix: str, out_ext: Optional[str] = None) -> Path:
    """Pick a free filename directly inside ~/Downloads (no subfolder)."""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ext = out_ext if out_ext else src.suffix
    base = f"{src.stem}{suffix}"
    candidate = DOWNLOADS_DIR / f"{base}{ext}"
    i = 1
    while candidate.exists():
        candidate = DOWNLOADS_DIR / f"{base}_{i}{ext}"
        i += 1
    return candidate


def _ffmpeg_exe() -> Optional[str]:
    """Prefer a system ffmpeg if present, otherwise fall back to the static
    binary bundled via the imageio-ffmpeg package (also what ships inside the
    packaged .app, so audio remux works even on a Mac with no ffmpeg/Homebrew
    installed at all)."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return exe
    except Exception:
        return None


def _apply_mask(frame_bgr, boxes, mode: str, margin_ratio: float, mosaic_cells: int):
    if mode == MODE_INPAINT:
        return inpaint_regions(frame_bgr, boxes)
    return mask_regions(frame_bgr, boxes, mode=mode, margin_ratio=margin_ratio, mosaic_cells=mosaic_cells)


def process_image(
    src: Path,
    detector: PlateDetector,
    mode: str,
    margin_ratio: float = 0.0,
    mosaic_cells: int = 8,
) -> Path:
    img = _imread_unicode(src)
    if img is None:
        raise RuntimeError(f"画像を読み込めませんでした: {src}")
    boxes = detector.detect(img)
    out = _apply_mask(img, boxes, mode, margin_ratio, mosaic_cells)
    dst = output_path_for(src, "_masked")
    _imwrite_unicode(dst, out)
    return dst


def process_video(
    src: Path,
    detector: PlateDetector,
    mode: str,
    margin_ratio: float = 0.0,
    mosaic_cells: int = 8,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"動画を読み込めませんでした: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    dst = output_path_for(src, "_masked", out_ext=".mp4")

    tmp_dir = Path(tempfile.mkdtemp(prefix="lpmask_"))
    tmp_video = tmp_dir / "video_noaudio.mp4"

    writer = cv2.VideoWriter(
        str(tmp_video), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    # Bridges brief single/few-frame detection dropouts (motion blur, a plate
    # angled awkwardly for a frame or two) so the mask doesn't flicker off.
    # ~1/3 second of persistence at this video's own frame rate.
    tracker = PlateTracker(max_misses=max(4, int(fps * 0.35)))

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = detector.detect(frame)
            tracked_boxes = tracker.update(boxes)
            out_frame = _apply_mask(frame, tracked_boxes, mode, margin_ratio, mosaic_cells)
            writer.write(out_frame)
            frame_idx += 1
            if progress_cb is not None:
                progress_cb(frame_idx, total)
    finally:
        cap.release()
        writer.release()

    ffmpeg = _ffmpeg_exe()
    if ffmpeg:
        cmd = [
            ffmpeg, "-y",
            "-i", str(tmp_video),
            "-i", str(src),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not dst.exists():
            shutil.copy(tmp_video, dst)
    else:
        shutil.copy(tmp_video, dst)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return dst
