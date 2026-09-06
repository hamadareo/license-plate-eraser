"""License plate detection backed by a YOLO model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # PyInstaller bundle
    return Path(__file__).resolve().parent.parent


MODEL_PATH = _base_dir() / "models" / "license-plate-finetune-v1s.pt"


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PlateDetector:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        conf: float = 0.25,
        imgsz: int = 960,
        augment: bool = True,
    ):
        self.device = _pick_device()
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.conf = conf
        # Larger inference size (vs. the ultralytics default of 640) picks up
        # small/distant plates better, and test-time augmentation (flip +
        # multi-scale) trades extra inference time for fewer missed frames.
        # Both were requested explicitly ("精度重視、時間はかかってよい").
        self.imgsz = imgsz
        self.augment = augment

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Return a list of (x1, y1, x2, y2, confidence) boxes in pixel coords."""
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            device=self.device,
            imgsz=self.imgsz,
            augment=self.augment,
            verbose=False,
        )
        boxes: list[tuple[int, int, int, int, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes
