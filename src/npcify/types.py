"""Shared data shapes used across every stage of the pipeline.

Keeping these in one place means each module agrees on what a "detection",
"track" or "decision" looks like, so stages can be swapped out independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

# A single video frame: height x width x 3 (BGR, as OpenCV delivers it), uint8.
Frame = NDArray[np.uint8]

# An (x, y) position in full-frame pixel coordinates.
Point = tuple[float, float]


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned box in pixel coordinates. (x, y) is the top-left corner."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    """One face found in one frame, before we know who it belongs to.

    `landmarks`, when the detector provides them, are five (x, y) points in
    full-frame pixels, in YuNet's order: right eye, left eye, nose tip, right
    mouth corner, left mouth corner ("right"/"left" from the person's point of
    view). They are not clipped, so they may fall outside a face cut off by
    the frame edge.
    """

    box: BoundingBox
    confidence: float
    landmarks: tuple[Point, ...] | None = None


class TrackState(Enum):
    """Lifecycle of a track.

    Every state except REMOVED is a real face that must be covered. Confirmation
    only matters for things like stable avatar assignment, never for deciding
    whether to hide someone.
    """

    TENTATIVE = "tentative"  # newly seen, not yet confirmed
    CONFIRMED = "confirmed"  # reliably tracked
    LOST = "lost"  # no matching detection right now; position is predicted
    REMOVED = "removed"  # gone for good


@dataclass
class Track:
    """A person followed across frames, with a stable ID.

    `box` is the smoothed (or, while lost, predicted and enlarged) face box in
    full-frame pixels. `velocity` is the box centre's movement in pixels per
    frame. `landmarks` come from this frame's matched detection, and are None
    when the track was not matched this frame.
    """

    track_id: int
    state: TrackState
    box: BoundingBox
    velocity: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    hits: int = 0
    frames_since_seen: int = 0
    age: int = 0
    landmarks: tuple[Point, ...] | None = None


@dataclass(frozen=True)
class Decision:
    """Whether a tracked person stays real (e.g. the creator) or becomes an avatar."""

    track_id: int
    keep_real: bool


@dataclass
class FaceState:
    """Head angles (degrees) and expression values for one tracked face.

    Expressions map a name (e.g. "smile", "jaw_open") to a strength in [0, 1].
    """

    track_id: int
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    expressions: dict[str, float] = field(default_factory=dict)
