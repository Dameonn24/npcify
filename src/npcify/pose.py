"""Measure head angle and facial expression for each face.

These values drive the avatar so it turns and emotes like the real person.
"""

from __future__ import annotations

from npcify.types import FaceState, Frame, Track


def estimate_face_states(frame: Frame, tracks: list[Track]) -> list[FaceState]:
    """Return head angles and expression values for each track."""
    raise NotImplementedError
