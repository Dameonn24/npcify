"""Find faces in a single frame.

Returns raw detections (box + confidence) with no idea of identity; the
tracker is responsible for linking detections across frames.
"""

from __future__ import annotations

from npcify.types import Detection, Frame


def detect_faces(frame: Frame) -> list[Detection]:
    """Return every face found in the frame."""
    raise NotImplementedError
