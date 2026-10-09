"""Find faces in a single frame.

Returns raw detections (box, confidence, landmarks) with no idea of identity;
the tracker is responsible for linking detections across frames.

Detection runs on a downscaled copy of the frame for speed, and results are
mapped back to full-frame pixels. Boxes are rounded outwards when converted
to whole pixels, so a box never ends up smaller than the face it covers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from npcify.models import ensure_model
from npcify.types import BoundingBox, Detection, Frame


class FaceDetector(Protocol):
    """Anything that can find faces in a BGR frame."""

    def detect(self, frame: Frame) -> list[Detection]: ...


def detection_size(frame_w: int, frame_h: int, det_width: int) -> tuple[int, int]:
    """Size of the downscaled image to run detection on. Never upscales."""
    if frame_w <= det_width:
        return frame_w, frame_h
    return det_width, max(1, round(frame_h * det_width / frame_w))


def to_detections(
    raw: NDArray[np.float32] | None,
    scale_x: float,
    scale_y: float,
    frame_w: int,
    frame_h: int,
) -> list[Detection]:
    """Convert YuNet's raw output rows into full-frame Detections.

    Each raw row is: x, y, w, h, then 5 landmark (x, y) pairs, then the score,
    all in detection-image pixels. Boxes are scaled, clipped to the frame,
    rounded outwards, and dropped if they end up with no area.
    """
    if raw is None:
        return []
    detections = []
    for row in raw:
        x, y, w, h = (float(v) for v in row[:4])
        x1 = max(0, math.floor(x * scale_x))
        y1 = max(0, math.floor(y * scale_y))
        x2 = min(frame_w, math.ceil((x + w) * scale_x))
        y2 = min(frame_h, math.ceil((y + h) * scale_y))
        if x2 <= x1 or y2 <= y1:
            continue
        landmarks = tuple(
            (float(row[4 + 2 * i]) * scale_x, float(row[5 + 2 * i]) * scale_y) for i in range(5)
        )
        detections.append(
            Detection(
                box=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                confidence=float(row[14]),
                landmarks=landmarks,
            )
        )
    return detections


class YuNetDetector:
    """OpenCV's YuNet face detector (cv2.FaceDetectorYN).

    Settings:
        score_threshold: minimum confidence to report a face (0-1).
        nms_threshold: overlap above which duplicate boxes are merged (0-1).
        max_faces: maximum candidate boxes considered before NMS.
        det_width: width of the downscaled image detection runs on. Larger finds
            smaller (further away) faces but is slower.
    """

    def __init__(
        self,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        max_faces: int = 5000,
        det_width: int = 640,
        model_path: str | Path | None = None,
    ) -> None:
        self.det_width = det_width
        path = Path(model_path) if model_path else ensure_model("yunet")
        # OpenCV 5 warns "Targets are not supported by the new graph engine" here
        # even though the default CPU target is fine; silence just this call.
        log = cv2.utils.logging
        previous_level = log.getLogLevel()
        log.setLogLevel(log.LOG_LEVEL_ERROR)
        try:
            self._net = cv2.FaceDetectorYN.create(
                str(path), "", (det_width, det_width), score_threshold, nms_threshold, max_faces
            )
        finally:
            log.setLogLevel(previous_level)
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame: Frame) -> list[Detection]:
        frame_h, frame_w = frame.shape[:2]
        det_w, det_h = detection_size(frame_w, frame_h, self.det_width)
        small = (
            frame
            if (det_w, det_h) == (frame_w, frame_h)
            else cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        )
        if self._input_size != (det_w, det_h):
            self._net.setInputSize((det_w, det_h))
            self._input_size = (det_w, det_h)
        _, raw = self._net.detect(small)
        return to_detections(raw, frame_w / det_w, frame_h / det_h, frame_w, frame_h)
