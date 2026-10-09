"""Hide faces: anonymise every region, then draw cartoon avatars on top.

The Anonymizer covers each face with a padded, soft-edged ellipse. The core of
the ellipse is fully opaque and always contains the whole face box; feathering
only happens outside that core, so no original face pixel is ever blended
into the output. Only the face regions are processed, not the whole frame.
"""

from __future__ import annotations

import math
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from npcify.avatar import look_seed
from npcify.smoothing import OneEuroFilter
from npcify.types import (
    AvatarExpression,
    BoundingBox,
    FaceState,
    Frame,
    HeadPose,
    Track,
    TrackState,
)

STYLES = ("pixelate-blur", "pixelate", "blur", "solid")


@dataclass(frozen=True)
class Padding:
    """How much bigger than the detected face box the covered area is.

    width/height: extra size as a fraction of the face box (0.3 = 30% wider).
    shift_up: move the area up by this fraction of face height, to cover more
        hair and forehead (detector boxes start around the eyebrows).
    """

    width: float = 0.3
    height: float = 0.4
    shift_up: float = 0.1


@dataclass(frozen=True)
class _Ellipse:
    """The fully-opaque core ellipse for one face, in frame coordinates."""

    cx: float
    cy: float
    a: float  # half-width
    b: float  # half-height


def core_ellipse(box: BoundingBox, padding: Padding) -> _Ellipse:
    """The padded ellipse, enlarged if needed so it fully contains the face box.

    An ellipse drawn inside a padded box does not reach that box's corners, and
    with modest padding it may not reach the face box's corners either. So
    scale the ellipse up just enough that the face box's farthest corner lies
    inside it, plus a pixel of margin for rounding.
    """
    w, h = box.width, box.height
    cx = box.x + w / 2
    cy = box.y + h / 2 - padding.shift_up * h
    a = w * (1 + padding.width) / 2
    b = h * (1 + padding.height) / 2
    # Farthest face-box corner from the ellipse centre.
    dx = w / 2
    dy = h / 2 + abs(padding.shift_up) * h
    k = math.hypot(dx / a, dy / b)
    if k > 1:
        a, b = a * k, b * k
    return _Ellipse(cx, cy, a + 1, b + 1)


def _region_mask(
    box: BoundingBox, frame_w: int, frame_h: int, padding: Padding, feather: float
) -> tuple[tuple[int, int, int, int], NDArray[np.float32]] | None:
    """Region of interest (x1, y1, x2, y2), clipped to the frame, and its alpha mask.

    Alpha is 1 inside the core ellipse and falls linearly to 0 over `feather`
    (as a fraction of the ellipse size) outside it. The face box itself is also
    forced to 1, as a second guarantee against rounding.
    """
    e = core_ellipse(box, padding)
    outer = 1 + feather
    x1 = max(0, math.floor(e.cx - e.a * outer))
    y1 = max(0, math.floor(e.cy - e.b * outer))
    x2 = min(frame_w, math.ceil(e.cx + e.a * outer))
    y2 = min(frame_h, math.ceil(e.cy + e.b * outer))
    if x2 <= x1 or y2 <= y1:
        return None

    # Pixel centres, relative to the ellipse centre, in units of the axes.
    xs = (np.arange(x1, x2, dtype=np.float32) + 0.5 - e.cx) / e.a
    ys = (np.arange(y1, y2, dtype=np.float32) + 0.5 - e.cy) / e.b
    r = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
    if feather > 0:
        alpha = np.clip((outer - r) / feather, 0.0, 1.0).astype(np.float32)
    else:
        alpha = (r <= 1.0).astype(np.float32)

    fx1, fy1 = max(box.x, x1) - x1, max(box.y, y1) - y1
    fx2, fy2 = min(box.x + box.width, x2) - x1, min(box.y + box.height, y2) - y1
    if fx2 > fx1 and fy2 > fy1:
        alpha[fy1:fy2, fx1:fx2] = 1.0
    return (x1, y1, x2, y2), alpha


class Anonymizer:
    """Hides face regions in a frame.

    Styles:
        "pixelate-blur" (default): a mosaic, then a light blur to soften the
            blocks. Blur on its own can sometimes be partly reversed
            (deconvolution, or recognition models trained on blurred faces),
            and it's easy to make it too weak. Pixelating first throws the
            detail away outright, so it is the stronger default.
        "pixelate": the mosaic only.
        "blur": a very strong blur (the face is shrunk to ~16 px wide, blurred,
            and scaled back up).
        "solid": a flat fill. The strongest: nothing of the face remains.

    blocks: mosaic blocks across the face width. Small faces get fewer, larger
        blocks (never smaller than `min_block` pixels) so they are still hidden.
    feather: soft-edge width outside the opaque core, as a fraction of its size.
    """

    def __init__(
        self,
        style: str = "pixelate-blur",
        padding: Padding | None = None,
        blocks: int = 8,
        min_block: int = 3,
        feather: float = 0.15,
        colour: tuple[int, int, int] = (90, 90, 90),
    ) -> None:
        if style not in STYLES:
            raise ValueError(f"Unknown style {style!r}; choose from {', '.join(STYLES)}")
        self.style = style
        self.padding = padding or Padding()
        self.blocks = blocks
        self.min_block = min_block
        self.feather = feather
        self.colour = colour

    def apply(self, frame: Frame, boxes: list[BoundingBox]) -> Frame:
        """Return a copy of the frame with every box hidden."""
        out = frame.copy()
        h, w = frame.shape[:2]
        for box in boxes:
            region = _region_mask(box, w, h, self.padding, self.feather)
            if region is None:
                continue
            (x1, y1, x2, y2), alpha = region
            roi = out[y1:y2, x1:x2]
            hidden = self._hide(roi, box.width)
            a = alpha[:, :, None]
            # Where alpha == 1 this is exactly `hidden`: no original pixel survives.
            out[y1:y2, x1:x2] = (hidden * a + roi * (1.0 - a)).astype(np.uint8)
        return out

    def coverage_mask(self, shape: tuple[int, ...], boxes: list[BoundingBox]) -> NDArray[np.bool_]:
        """Boolean mask of pixels that are fully hidden (alpha == 1) by `apply`."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        for box in boxes:
            region = _region_mask(box, w, h, self.padding, self.feather)
            if region is not None:
                (x1, y1, x2, y2), alpha = region
                mask[y1:y2, x1:x2] |= alpha >= 1.0
        return mask

    def _hide(self, roi: Frame, face_width: int) -> NDArray[np.float32]:
        rh, rw = roi.shape[:2]
        if self.style == "solid":
            return np.broadcast_to(np.array(self.colour, np.float32), roi.shape)
        if self.style == "blur":
            scale = min(1.0, 16 / max(1, face_width))
            small = cv2.resize(
                roi,
                (max(1, round(rw * scale)), max(1, round(rh * scale))),
                interpolation=cv2.INTER_AREA,
            )
            small = cv2.GaussianBlur(small, (0, 0), 3)
            return cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        block = max(self.min_block, face_width / self.blocks)
        small = cv2.resize(
            roi,
            (max(1, math.ceil(rw / block)), max(1, math.ceil(rh / block))),
            interpolation=cv2.INTER_AREA,
        )
        mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        if self.style == "pixelate-blur":
            k = max(3, int(block) // 2 * 2 + 1)  # odd kernel about one block wide
            mosaic = cv2.GaussianBlur(mosaic, (k, k), 0)
        return mosaic.astype(np.float32)


class AvatarRenderer(Protocol):
    """A drawable avatar style. Other styles (3D heads, avatar packs) implement this.

    identity(seed): a random look for one person (same seed, same look).
    render(identity, pose, expression, size): a size x size BGRA image with
        PREMULTIPLIED alpha (colour channels already multiplied by alpha).
    cover_centre / cover_radius: a circle, as fractions of `size`, inside which
        the rendered image is fully opaque for every pose and expression. The
        compositor scales the avatar so this circle contains the face box, so
        a style that breaks this promise could leave part of a face showing
        (the anonymisation underneath still hides it).
    """

    name: str
    cover_centre: tuple[float, float]
    cover_radius: float

    def identity(self, seed: int) -> Any: ...

    def render(
        self, identity: Any, pose: HeadPose, expression: AvatarExpression, size: int
    ) -> NDArray[np.uint8]: ...


@dataclass
class _AvatarTrack:
    identity: Any
    placement: OneEuroFilter
    fade: float = 0.0


class AvatarLayer:
    """Draws avatars over faces that have already been anonymised.

    Who gets an avatar: confirmed hidden tracks with a FaceState from the
    landmarker, or a coarse FaceState for faces at least `min_coarse_width`
    wide (drawn with a neutral expression). Everyone else - tentative and lost
    tracks, raw detections, tracks without a FaceState - keeps just the blur.

    An avatar fades in over `fade_frames` frames once a track qualifies and
    vanishes instantly when it stops qualifying. Both are safe, because the
    blur is always underneath.

    Size and position are smoothed, but the radius is then forced back up if
    needed so that the avatar's opaque circle always contains the face box.
    """

    def __init__(
        self,
        renderer: AvatarRenderer,
        session_seed: int | None = None,
        fade_frames: int = 5,
        min_coarse_width: int = 40,
        render_size: int = 256,
    ) -> None:
        self.renderer = renderer
        self.session_seed = secrets.randbits(32) if session_seed is None else session_seed
        self.fade_frames = fade_frames
        self.min_coarse_width = min_coarse_width
        self.render_size = render_size
        self._tracks: dict[int, _AvatarTrack] = {}

    def reroll(self, session_seed: int | None = None) -> None:
        """Give everyone a new random look."""
        self.session_seed = secrets.randbits(32) if session_seed is None else session_seed
        self._tracks = {}

    def qualifies(self, track: Track, face: FaceState | None) -> bool:
        if face is None or track.state is not TrackState.CONFIRMED:
            return False
        if face.source == "landmarker":
            return True
        return face.source == "coarse" and track.box.width >= self.min_coarse_width

    def apply(
        self,
        frame: Frame,
        tracks: list[Track],
        faces: Mapping[int, FaceState],
        now: float | None = None,
        advance: bool = True,
    ) -> Frame:
        """Return a copy of an already-anonymised frame with avatars drawn on top.

        `faces` must only contain hidden tracks (kept people never get one).
        With advance=False (e.g. re-drawing a paused frame) fades don't move on.
        """
        now = time.perf_counter() if now is None else now
        out = frame.copy()
        live = {t.track_id for t in tracks}
        self._tracks = {k: v for k, v in self._tracks.items() if k in live}
        for t in tracks:
            face = faces.get(t.track_id)
            if not self.qualifies(t, face):
                self._tracks.pop(t.track_id, None)  # back to blur instantly
                continue
            assert face is not None
            at = self._tracks.get(t.track_id)
            if at is None:
                seed = look_seed(t.track_id, self.session_seed)
                at = _AvatarTrack(self.renderer.identity(seed), OneEuroFilter(1.5, 0.02))
                self._tracks[t.track_id] = at
            if advance:
                at.fade = min(1.0, at.fade + 1.0 / max(1, self.fade_frames))
            if at.fade <= 0:
                continue
            expression = face.avatar if face.source == "landmarker" else AvatarExpression()
            pose = HeadPose(face.yaw, face.pitch, face.roll)
            cx, cy, radius = self._placement(at, t.box, now, advance)
            self._draw(out, at, pose, expression, cx, cy, radius)
        return out

    def _placement(
        self, at: _AvatarTrack, box: BoundingBox, now: float, advance: bool
    ) -> tuple[float, float, float]:
        bx, by = box.x + box.width / 2, box.y + box.height / 2
        needed = math.hypot(box.width, box.height) / 2
        if advance:
            cx, cy, r = at.placement([bx, by, needed], now)
        else:
            cx, cy, r = bx, by, needed
        # Never smaller than what covers the actual box from the smoothed centre.
        corners = max(
            math.hypot(x - cx, y - cy)
            for x in (box.x, box.x + box.width)
            for y in (box.y, box.y + box.height)
        )
        return float(cx), float(cy), max(float(r), corners) + 2.0

    def _draw(
        self,
        out: Frame,
        at: _AvatarTrack,
        pose: HeadPose,
        expression: AvatarExpression,
        cx: float,
        cy: float,
        radius: float,
    ) -> None:
        r = self.renderer
        size = max(8, round(radius / r.cover_radius))
        rgba = r.render(at.identity, pose, expression, self.render_size)
        interp = cv2.INTER_AREA if size < self.render_size else cv2.INTER_LINEAR
        rgba = cv2.resize(rgba, (size, size), interpolation=interp)
        x0 = round(cx - r.cover_centre[0] * size)
        y0 = round(cy - r.cover_centre[1] * size)
        h, w = out.shape[:2]
        fx1, fy1, fx2, fy2 = max(0, x0), max(0, y0), min(w, x0 + size), min(h, y0 + size)
        if fx2 <= fx1 or fy2 <= fy1:
            return
        patch = rgba[fy1 - y0 : fy2 - y0, fx1 - x0 : fx2 - x0].astype(np.float32) * at.fade
        alpha = patch[:, :, 3:4] / 255.0
        roi = out[fy1:fy2, fx1:fx2].astype(np.float32)
        out[fy1:fy2, fx1:fx2] = np.clip(patch[:, :, :3] + roi * (1.0 - alpha), 0, 255).astype(
            np.uint8
        )
