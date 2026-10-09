"""Shared data shapes used across every stage of the pipeline.

Keeping these in one place means each module agrees on what a "detection",
"track" or "decision" looks like, so stages can be swapped out independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# A single video frame: height x width x 3 (BGR, as OpenCV delivers it), uint8.
Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned box in pixel coordinates. (x, y) is the top-left corner."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    """One face found in one frame, before we know who it belongs to."""

    box: BoundingBox
    confidence: float


@dataclass
class Track:
    """A person followed across frames, with a stable ID."""

    track_id: int
    box: BoundingBox
    confidence: float
    frames_since_seen: int = 0


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
