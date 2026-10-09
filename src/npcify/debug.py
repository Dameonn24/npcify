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
from npcify.pose import angles_to_rotation
from npcify.types import BoundingBox, Detection, FaceState, Frame, Point, Track, TrackState

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


_BARS = [
    ("blinkL", "blink_left"),
    ("blinkR", "blink_right"),
    ("mouth", "mouth_open"),
    ("smile", "smile"),
    ("brow+", "brows_up"),
    ("brow-", "brows_down"),
]


def draw_face_states(
    frame: Frame,
    tracks: list[Track],
    states: Mapping[int, FaceState],
    mesh: bool = False,
) -> Frame:
    """Return a copy with head axes, angles and expression bars for each hidden face.

    Axes: red = the face's right-left axis, green = up, blue = where the nose points.
    """
    out = frame.copy()
    for t in tracks:
        s = states.get(t.track_id)
        if s is None or s.source == "none":
            continue
        b = t.box
        if mesh and s.mesh is not None:
            for x, y in s.mesh:
                out[
                    min(max(int(y), 0), out.shape[0] - 1), min(max(int(x), 0), out.shape[1] - 1)
                ] = (
                    0,
                    255,
                    255,
                )

        # Head axes, projected straight onto the image (camera y is up, image y is down).
        cx, cy = b.x + b.width / 2, b.y + b.height / 2
        if s.mesh is not None and len(s.mesh) > 1:
            cx, cy = float(s.mesh[1][0]), float(s.mesh[1][1])  # nose tip
        r = angles_to_rotation(s.yaw, s.pitch, s.roll)
        length = 0.5 * b.width
        for axis, colour in ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0))):
            vx, vy = r[0, axis], -r[1, axis]
            end = (round(cx + length * vx), round(cy + length * vy))
            cv2.line(out, (round(cx), round(cy)), end, colour, 2, cv2.LINE_AA)

        # Numbers and bars to the right of the face.
        x0, y0 = b.x + b.width + 6, b.y + 12
        src = "lm" if s.source == "landmarker" else s.source
        put_outlined_text(out, f"y{s.yaw:+.0f} p{s.pitch:+.0f} r{s.roll:+.0f} {src}", (x0, y0))
        if s.source != "landmarker":
            continue
        for i, (name, attr) in enumerate(_BARS):
            y = y0 + 16 + i * 13
            put_outlined_text(out, name, (x0, y), scale=0.38)
            value = getattr(s.avatar, attr)
            cv2.rectangle(out, (x0 + 46, y - 8), (x0 + 96, y - 1), (60, 60, 60), -1)
            cv2.rectangle(
                out, (x0 + 46, y - 8), (x0 + 46 + round(50 * value), y - 1), (0, 220, 255), -1
            )
        # Gaze: a dot inside a small box.
        gy = y0 + 16 + len(_BARS) * 13
        cv2.rectangle(out, (x0, gy - 8), (x0 + 30, gy + 14), (60, 60, 60), 1)
        gx = x0 + 15 + round(13 * s.avatar.look_x)
        gyy = gy + 3 - round(9 * s.avatar.look_y)
        cv2.circle(out, (gx, gyy), 3, (0, 220, 255), -1)
    return out
