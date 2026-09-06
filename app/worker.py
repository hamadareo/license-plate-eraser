"""Background threads: model loading and batch file processing."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from . import inpainter
from .detector import PlateDetector
from .media import is_image, is_video, process_image, process_video


class ModelLoaderThread(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            detector = PlateDetector()
            self.loaded.emit(detector)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class LamaPreloadThread(QThread):
    loaded = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self):
        try:
            inpainter.preload()
            self.loaded.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class BatchWorker(QThread):
    file_started = pyqtSignal(str)
    file_progress = pyqtSignal(str, int)
    file_done = pyqtSignal(str, str)
    file_error = pyqtSignal(str, str)
    batch_progress = pyqtSignal(int, int)  # (finished_count, total_count)
    all_done = pyqtSignal(bool)  # was_cancelled

    def __init__(
        self,
        detector: PlateDetector,
        files: list[str],
        mode: str,
        margin_ratio: float = 0.0,
        mosaic_cells: int = 8,
    ):
        super().__init__()
        self.detector = detector
        self.files = files
        self.mode = mode
        self.margin_ratio = margin_ratio
        self.mosaic_cells = mosaic_cells
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total = len(self.files)
        finished = 0
        cancelled = False
        for f in self.files:
            if self._cancel:
                cancelled = True
                break
            src = Path(f)
            self.file_started.emit(f)
            try:
                if is_image(src):
                    dst = process_image(
                        src, self.detector, self.mode, self.margin_ratio, self.mosaic_cells
                    )
                elif is_video(src):
                    def cb(cur, total, _f=f):
                        pct = int(cur * 100 / total) if total else 0
                        self.file_progress.emit(_f, pct)

                    dst = process_video(
                        src,
                        self.detector,
                        self.mode,
                        self.margin_ratio,
                        self.mosaic_cells,
                        progress_cb=cb,
                    )
                else:
                    self.file_error.emit(f, "対応していないファイル形式です")
                    finished += 1
                    self.batch_progress.emit(finished, total)
                    continue
                self.file_done.emit(f, str(dst))
            except Exception as e:  # noqa: BLE001
                self.file_error.emit(f, str(e))
            finished += 1
            self.batch_progress.emit(finished, total)
        self.all_done.emit(cancelled)
