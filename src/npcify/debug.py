"""Development-only drawing helpers.

These draw what the pipeline sees (boxes, scores, landmarks) so we can check
it by eye. They are not part of the real output path and must never be
relied on to hide a face.
"""

from __future__ import annotations

import cv2

from npcify.types import Detection, Frame

_BOX_COLOUR = (0, 255, 0)  # BGR green
# Right eye, left eye, nose tip, right mouth corner, left mouth corner.
_LANDMARK_COLOURS = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def draw_detections(frame: Frame, detections: list[Detection]) -> Frame:
    """Return a copy of the frame with boxes, confidence scores and landmark dots."""
    out = frame.copy()
    for det in detections:
        b = det.box
        cv2.rectangle(out, (b.x, b.y), (b.x + b.width - 1, b.y + b.height - 1), _BOX_COLOUR, 2)
        label = f"{det.confidence:.2f}"
        origin = (b.x, max(12, b.y - 6))
        cv2.putText(out, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOUR, 1, cv2.LINE_AA)
        for (x, y), colour in zip(det.landmarks or (), _LANDMARK_COLOURS, strict=False):
            cv2.circle(out, (round(x), round(y)), 3, colour, -1, cv2.LINE_AA)
    return out
