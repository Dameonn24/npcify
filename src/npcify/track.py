"""Give each person a stable ID across frames.

Matches this frame's detections to the people we saw in previous frames, so
decisions (real vs avatar) stick to a person instead of flickering.
"""

from __future__ import annotations

from npcify.types import Detection, Track


class Tracker:
    """Keeps track IDs stable from frame to frame."""

    def update(self, detections: list[Detection]) -> list[Track]:
        """Match new detections to existing tracks and return the current tracks."""
        raise NotImplementedError
