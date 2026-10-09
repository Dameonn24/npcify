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


@dataclass(frozen=True)
class CoverRegion:
    """A face area that must be hidden this frame.

    `box` is the face box (padding is added by the anonymiser). `track_id` is
    None for a raw detection that no track accounts for. `reason` says why it
    is covered (e.g. "track:lost", "detection"), for debugging and leak checks.
    """

    box: BoundingBox
    track_id: int | None
    reason: str


@dataclass(frozen=True)
class AvatarExpression:
    """The small, stable set of face values that drive avatars.

    This is the contract between pose estimation and avatar rendering: avatar
    styles should only need these. Left/right are in IMAGE terms (as the face
    appears in the frame being drawn on), so they are correct whether or not
    the video is mirrored.

    blink_left / blink_right: eye closed amount, 0 open .. 1 closed, for the eye
        on the image's left / right.
    mouth_open: 0 closed .. 1 wide open.
    smile: 0 neutral .. 1 big smile.
    brows_up / brows_down: eyebrows raised (surprise) / lowered (frown), 0 .. 1.
    look_x: eye direction, -1 toward image left .. 1 toward image right.
    look_y: eye direction, -1 down .. 1 up.
    """

    blink_left: float = 0.0
    blink_right: float = 0.0
    mouth_open: float = 0.0
    smile: float = 0.0
    brows_up: float = 0.0
    brows_down: float = 0.0
    look_x: float = 0.0
    look_y: float = 0.0


@dataclass(frozen=True)
class HeadPose:
    """Head angles in degrees, in image terms (see FaceState)."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class FaceState:
    """Head pose and expression for one hidden (bystander) face.

    Angles are in degrees, in IMAGE terms:
        yaw: + face turned toward the image's right, - toward its left.
        pitch: + looking up, - looking down.
        roll: + head tilted clockwise as seen in the image.
    `expressions` holds the raw blendshape scores (52 for the landmarker,
    empty for the coarse estimate). `source` is "landmarker", "coarse" or
    "none"; `confidence` is 0..1. `mesh` is the full-frame landmark mesh when
    available (debug drawing only). All of this lives in memory only, for live
    tracks; it is never logged or saved.
    """

    track_id: int
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    expressions: dict[str, float] = field(default_factory=dict)
    avatar: AvatarExpression = field(default_factory=AvatarExpression)
    source: str = "none"
    confidence: float = 0.0
    mesh: NDArray[np.float32] | None = None
