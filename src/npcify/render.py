"""Hide faces: anonymise regions now, draw cartoon avatars later (step 8).

The Anonymizer covers each face with a padded, soft-edged ellipse. The core of
the ellipse is fully opaque and always contains the whole face box; feathering
only happens outside that core, so no original face pixel is ever blended
into the output. Only the face regions are processed, not the whole frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from npcify.types import BoundingBox, FaceState, Frame, Track

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


def draw_avatars(frame: Frame, tracks: list[Track], states: list[FaceState]) -> Frame:
    """Return a copy of the frame with avatars drawn over the given faces (step 8)."""
    raise NotImplementedError
