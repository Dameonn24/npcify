"""Development-only drawing helpers.

These draw what the pipeline sees (boxes, scores, landmarks, track IDs) so we
can check it by eye. They are not part of the real output path and must
never be relied on to hide a face.
"""

from __future__ import annotations

import colorsys
from collections import deque
from collections.abc import Mapping

import cv2

from npcify.output import put_outlined_text
from npcify.types import BoundingBox, Detection, Frame, Point, Track, TrackState

_BOX_COLOUR = (0, 255, 0)  # BGR green
_RAW_COLOUR = (160, 160, 160)  # grey, for raw detections under tracks
# Right eye, left eye, nose tip, right mouth corner, left mouth corner.
_LANDMARK_COLOURS = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def _landmarks(img: Frame, landmarks: tuple[Point, ...] | None) -> None:
    for (x, y), colour in zip(landmarks or (), _LANDMARK_COLOURS, strict=False):
        cv2.circle(img, (round(x), round(y)), 3, colour, -1, cv2.LINE_AA)


def _corners(b: BoundingBox) -> tuple[tuple[int, int], tuple[int, int]]:
    return (b.x, b.y), (b.x + b.width - 1, b.y + b.height - 1)


def draw_detections(frame: Frame, detections: list[Detection]) -> Frame:
    """Return a copy of the frame with boxes, confidence scores and landmark dots."""
    out = frame.copy()
    for det in detections:
        cv2.rectangle(out, *_corners(det.box), _BOX_COLOUR, 2)
        put_outlined_text(
            out, f"{det.confidence:.2f}", (det.box.x, max(12, det.box.y - 6)), _BOX_COLOUR
        )
        _landmarks(out, det.landmarks)
    return out


def track_colour(track_id: int) -> tuple[int, int, int]:
    """A stable, bright BGR colour for a track ID (golden-ratio hue spacing)."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return round(b * 255), round(g * 255), round(r * 255)


def _dashed_rect(img: Frame, b: BoundingBox, colour: tuple[int, int, int], dash: int = 8) -> None:
    (x1, y1), (x2, y2) = _corners(b)
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), colour, 2)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), colour, 2)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), colour, 2)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), colour, 2)


class TrackTrails:
    """Remembers the last few centre points of each track, for drawing trails."""

    def __init__(self, length: int = 15) -> None:
        self.length = length
        self._trails: dict[int, deque[tuple[int, int]]] = {}

    def update(self, tracks: list[Track]) -> None:
        live = {t.track_id for t in tracks}
        self._trails = {k: v for k, v in self._trails.items() if k in live}
        for t in tracks:
            c = (t.box.x + t.box.width // 2, t.box.y + t.box.height // 2)
            self._trails.setdefault(t.track_id, deque(maxlen=self.length)).append(c)

    def get(self, track_id: int) -> list[tuple[int, int]]:
        return list(self._trails.get(track_id, ()))


def draw_tracks(
    frame: Frame,
    tracks: list[Track],
    trails: TrackTrails | None = None,
    detections: list[Detection] | None = None,
    notes: Mapping[int, str] | None = None,
) -> Frame:
    """Return a copy of the frame with each track in its own colour.

    Lost tracks are dashed and show how many frames since they were last seen.
    Raw detections, if given, are drawn as thin grey boxes underneath.
    `notes` adds a second label line per track ID (e.g. keep/hide status).
    """
    out = frame.copy()
    for det in detections or ():
        cv2.rectangle(out, *_corners(det.box), _RAW_COLOUR, 1)
    for t in tracks:
        colour = track_colour(t.track_id)
        if trails is not None:
            points = trails.get(t.track_id)
            for a, b in zip(points, points[1:], strict=False):
                cv2.line(out, a, b, colour, 2, cv2.LINE_AA)
        if t.state is TrackState.LOST:
            _dashed_rect(out, t.box, colour)
            label = f"#{t.track_id} lost {t.frames_since_seen}"
        else:
            thickness = 2 if t.state is TrackState.CONFIRMED else 1
            cv2.rectangle(out, *_corners(t.box), colour, thickness)
            label = f"#{t.track_id} {t.state.value} {t.confidence:.2f}"
        top = max(30 if notes else 12, t.box.y - 6)
        put_outlined_text(out, label, (t.box.x, top), colour)
        if notes and t.track_id in notes:
            put_outlined_text(out, notes[t.track_id], (t.box.x, top - 16), colour)
        _landmarks(out, t.landmarks)
    return out
