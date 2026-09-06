"""Lightweight IOU-based box tracker for video.

Per-frame YOLO detection can miss a plate for a frame or two (motion blur,
brief angle change, partial occlusion) even though it was clearly detected
just before and after. Left alone this shows up as a flicker where the plate
briefly isn't masked. This tracker bridges those gaps by holding a track's
last known box for a short window of frames when detection drops out,
instead of only ever trusting the current frame in isolation.
"""
from __future__ import annotations

Box = tuple[int, int, int, int, float]


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter)


class _Track:
    __slots__ = ("box", "misses")

    def __init__(self, box: Box):
        self.box = box
        self.misses = 0


class PlateTracker:
    def __init__(self, max_misses: int = 10, iou_threshold: float = 0.2):
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self._tracks: list[_Track] = []

    def update(self, detections: list[Box]) -> list[Box]:
        unmatched = list(range(len(detections)))
        for track in self._tracks:
            best_iou, best_j = 0.0, -1
            for j in unmatched:
                iou = _iou(track.box, detections[j])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_threshold:
                track.box = detections[best_j]
                track.misses = 0
                unmatched.remove(best_j)
            else:
                track.misses += 1

        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        for j in unmatched:
            self._tracks.append(_Track(detections[j]))

        return [t.box for t in self._tracks]
