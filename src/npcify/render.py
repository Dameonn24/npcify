"""Draw cartoon avatars over faces.

Takes the frame plus each face's position and pose, and paints an avatar on
top so the real face is fully covered.
"""

from __future__ import annotations

from npcify.types import FaceState, Frame, Track


def draw_avatars(frame: Frame, tracks: list[Track], states: list[FaceState]) -> Frame:
    """Return a copy of the frame with avatars drawn over the given faces."""
    raise NotImplementedError
