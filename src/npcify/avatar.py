"""The "npc" cartoon head style.

An original, simple, friendly style: a round head, flat colours, clean dark
outlines, big simple eyes. Loosely inspired by the rounded feel of Wii-era
avatars, but its own design (no Mii shapes, proportions or assets).

Looks are random, never based on the real person (their skin, hair or
anything else): a random look gives away nothing about who is underneath.
A look is seeded from the track ID plus a per-session random value, so it
stays fixed for a whole track but doesn't repeat across sessions.

Drawing works on a 512x512 canvas (2x) and is scaled down to 256x256 for
smooth edges. The canvas is BGRA with *premultiplied* alpha: OpenCV's
anti-aliased drawing onto a transparent (all-zero) canvas produces exactly
that, and premultiplied images scale and rotate without dark fringes.

Pseudo-3D: facial features sit on an imaginary sphere. Turning the head
(yaw) slides them sideways and foreshortens them, and features that go round
the side are hidden, so at big turns you mostly see hair. Nodding (pitch)
does the same vertically. Tilting (roll) rotates the whole drawing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from npcify.types import AvatarExpression, HeadPose

BGR = tuple[int, int, int]


def _rgb(r: int, g: int, b: int) -> BGR:
    return (b, g, r)


SKIN_TONES: list[BGR] = [
    _rgb(255, 228, 206),
    _rgb(247, 207, 172),
    _rgb(236, 188, 150),
    _rgb(222, 168, 125),
    _rgb(203, 145, 104),
    _rgb(178, 120, 82),
    _rgb(150, 98, 64),
    _rgb(122, 78, 50),
    _rgb(96, 62, 40),
    _rgb(74, 48, 32),
]
HAIR_COLOURS: list[BGR] = [
    _rgb(32, 28, 28),
    _rgb(62, 42, 30),
    _rgb(104, 68, 42),
    _rgb(140, 66, 38),
    _rgb(214, 176, 108),
    _rgb(196, 104, 46),
    _rgb(160, 160, 160),
    _rgb(228, 226, 220),
]
FUN_HAIR_COLOURS: list[BGR] = [_rgb(70, 120, 210), _rgb(220, 110, 170), _rgb(90, 180, 140)]
HAT_COLOURS: list[BGR] = [
    _rgb(214, 64, 58),
    _rgb(52, 110, 200),
    _rgb(60, 160, 90),
    _rgb(240, 190, 60),
    _rgb(120, 80, 170),
    _rgb(40, 40, 48),
]
HAIR_STYLES = ("bald", "short", "long", "bun", "curly")
EYE_SHAPES = ("round", "oval", "sleepy", "wide")


@dataclass(frozen=True)
class NpcLook:
    """One person's random avatar look."""

    skin: BGR
    hair_style: str
    hair: BGR
    eye_shape: str
    face_width: float  # 0.92 .. 1.04, relative
    glasses: bool
    hat: BGR | None


def make_look(seed: int) -> NpcLook:
    """A random look from a seed (same seed, same look)."""
    rng = np.random.default_rng(seed)
    pick = rng.integers
    hair_colours = HAIR_COLOURS if rng.random() > 0.08 else FUN_HAIR_COLOURS
    return NpcLook(
        skin=SKIN_TONES[pick(len(SKIN_TONES))],
        hair_style=HAIR_STYLES[pick(len(HAIR_STYLES))],
        hair=hair_colours[pick(len(hair_colours))],
        eye_shape=EYE_SHAPES[pick(len(EYE_SHAPES))],
        face_width=float(rng.uniform(0.92, 1.04)),
        glasses=bool(rng.random() < 0.2),
        hat=HAT_COLOURS[pick(len(HAT_COLOURS))] if rng.random() < 0.15 else None,
    )


def look_seed(track_id: int, session_seed: int) -> int:
    return int(np.random.SeedSequence([session_seed, track_id]).generate_state(1)[0])


# --- drawing ------------------------------------------------------------------

S = 512  # working canvas (2x)
OUT = 256  # output size
C = (256.0, 276.0)  # head centre on the working canvas
R = 165.0  # radius of the always-opaque head circle on the working canvas

# The renderer contract: within this circle (fractions of the output size)
# every pixel is fully opaque, for any pose and expression. The compositor
# relies on it to guarantee the face box is covered.
COVER_CENTRE = (C[0] / S, C[1] / S)
COVER_RADIUS = (R - 4) / S  # a little inside the drawn circle, for anti-aliasing

_OUTLINE = 4
_LINE = cv2.LINE_AA
_WHITE: BGR = (250, 250, 250)


def _darker(c: BGR, f: float = 0.55) -> BGR:
    return (int(c[0] * f), int(c[1] * f), int(c[2] * f))


def _bgra(c: BGR) -> tuple[int, int, int, int]:
    return (c[0], c[1], c[2], 255)


def _ellipse(img, centre, axes, colour: BGR, outline: BGR | None, angle: float = 0.0) -> None:
    ctr = (round(centre[0] * 4), round(centre[1] * 4))  # 2 bits of sub-pixel precision
    ax = (max(1, round(axes[0] * 4)), max(1, round(axes[1] * 4)))
    cv2.ellipse(img, ctr, ax, angle, 0, 360, _bgra(colour), -1, _LINE, 2)
    if outline is not None:
        cv2.ellipse(img, ctr, ax, angle, 0, 360, _bgra(outline), _OUTLINE, _LINE, 2)


def _poly(img, pts, colour: BGR, outline: BGR | None, closed: bool = True) -> None:
    p = np.round(np.asarray(pts) * 4).astype(np.int32).reshape(-1, 1, 2)
    if closed:
        cv2.fillPoly(img, [p], _bgra(colour), _LINE, 2)
    if outline is not None:
        cv2.polylines(img, [p], closed, _bgra(outline), _OUTLINE, _LINE, 2)


class NpcRenderer:
    """Draws npc-style heads. Implements the AvatarRenderer interface."""

    name = "npc"
    cover_centre = COVER_CENTRE
    cover_radius = COVER_RADIUS

    def identity(self, seed: int) -> NpcLook:
        return make_look(seed)

    def render(
        self, identity: NpcLook, pose: HeadPose, expression: AvatarExpression, size: int = OUT
    ) -> NDArray[np.uint8]:
        """A size x size BGRA image (premultiplied alpha) of the head."""
        img = np.zeros((S, S, 4), np.uint8)
        look, e = identity, expression
        yaw = math.radians(max(-80.0, min(80.0, pose.yaw)))
        pitch = math.radians(max(-45.0, min(45.0, pose.pitch)))
        hair_out = _darker(look.hair, 0.5)
        skin_out = _darker(look.skin, 0.6)
        bald = look.hair_style == "bald"
        turn = math.sin(yaw)

        # 1. Hair behind the head.
        back_x = C[0] - 0.18 * R * turn
        if look.hair_style == "long":
            # Flowing sides: a tall rounded shape behind the head, flared at the ends.
            pts = []
            for t in np.linspace(-0.12 * math.pi, 1.12 * math.pi, 31):
                flare = 1.0 + 0.12 * max(0.0, math.sin(t)) ** 8
                pts.append((back_x + 1.06 * R * flare * math.cos(t), C[1] + 1.1 * R * math.sin(t)))
            _poly(img, pts, look.hair, hair_out)
        elif look.hair_style == "bun":
            _ellipse(img, (back_x, C[1] - 1.02 * R), (0.36 * R, 0.32 * R), look.hair, hair_out)
        elif look.hair_style == "curly":
            for k in range(11):
                a = math.pi * (0.95 + 1.1 * k / 10)
                _ellipse(
                    img,
                    (back_x + 1.0 * R * math.cos(a), C[1] + 0.98 * R * math.sin(a)),
                    (0.27 * R, 0.27 * R),
                    look.hair,
                    hair_out,
                )

        # 2. The always-opaque head circle (hair colour, or skin when bald).
        _ellipse(img, C, (R, R), look.skin if bald else look.hair, skin_out if bald else hair_out)

        # 3. Ears (round shapes or long hair cover them).
        # Turning narrows the face and slides it toward the turn, showing more hair behind.
        fw = 0.8 * R * look.face_width * (1 - 0.28 * turn * turn)
        fh = 0.86 * R * (1 - 0.06 * abs(math.sin(pitch)))
        fcx = C[0] + 0.2 * R * turn
        fcy = C[1] + 0.12 * R - 0.06 * R * math.sin(pitch)
        if look.hair_style not in ("long", "curly"):
            for side in (-1, 1):
                a = side * math.pi / 2 + yaw
                if math.cos(a) > -0.25:
                    ex = fcx + 1.02 * fw * math.sin(a)
                    _ellipse(
                        img,
                        (ex, fcy - 0.05 * R),
                        (0.16 * R * max(0.35, math.cos(a) + 0.3), 0.22 * R),
                        look.skin,
                        skin_out,
                    )

        # 4. Face.
        _ellipse(img, (fcx, fcy), (fw, fh), look.skin, skin_out)

        def project(theta: float, phi: float) -> tuple[float, float, float, float]:
            """Screen x, y and foreshortening (sx, sy) of a point on the face sphere."""
            a, b = theta + yaw, phi - pitch
            return (fcx + fw * math.sin(a), fcy + fh * 0.95 * math.sin(b), math.cos(a), math.cos(b))

        # 5. Fringe (front hair) over the top of the face.
        if not bald:
            fringe_y = (
                fcy - fh * (0.62 if look.hair_style != "bun" else 0.7) - 0.12 * R * math.sin(pitch)
            )
            pts = [
                (C[0] + R * math.cos(t), C[1] + R * math.sin(t))
                for t in np.linspace(math.pi * 1.06, math.pi * 1.94, 24)
            ]
            n = 9
            for k in range(n + 1):
                u = 1 - 2 * k / n
                wave = 0.07 * R * (math.sin(k * 2.1) if look.hair_style == "curly" else 0.4 * u)
                pts.append((fcx + u * fw * 1.02, fringe_y + wave + 0.12 * R * u * u))
            _poly(img, pts, look.hair, hair_out)

        # 6. Eyes and pupils.
        shape = {
            "round": (1.0, 1.0),
            "oval": (1.15, 0.85),
            "sleepy": (1.1, 0.6),
            "wide": (1.2, 1.15),
        }[look.eye_shape]
        eyes = []
        for side, blink in ((-1, e.blink_left), (1, e.blink_right)):
            x, y, sx, sy = project(side * 0.42, -0.1)
            if sx < 0.2:
                continue  # round the side of the head: hidden
            eyes.append((x, y, sx))
            ew = 0.15 * R * shape[0] * sx
            eh = 0.17 * R * shape[1] * sy * (1 - max(0.0, min(1.0, blink)))
            if eh < 0.03 * R:
                _poly(
                    img,
                    [(x - ew, y), (x, y + 0.035 * R), (x + ew, y)],
                    look.skin,
                    skin_out,
                    closed=False,
                )
                continue
            _ellipse(img, (x, y), (ew, eh), _WHITE, _darker(look.skin, 0.35))
            px = x + max(-1.0, min(1.0, e.look_x)) * 0.45 * ew
            py = y - max(-1.0, min(1.0, e.look_y)) * 0.35 * eh
            pr = min(0.075 * R, eh * 0.85)
            _ellipse(img, (px, py), (pr * sx, pr), (40, 34, 30), None)
            _ellipse(img, (px - 0.3 * pr, py - 0.35 * pr), (0.3 * pr, 0.3 * pr), _WHITE, None)

        # 7. Eyebrows.
        brow_col = _darker(look.hair, 0.7) if not bald else _darker(look.skin, 0.45)
        lift = 0.12 * e.brows_up - 0.06 * e.brows_down
        for side in (-1, 1):
            xo, yo, sxo, _ = project(side * 0.62, -0.38 - lift)
            xi, yi, sxi, _ = project(side * 0.22, -0.36 - lift + 0.1 * e.brows_down)
            if min(sxo, sxi) < 0.2:
                continue
            p = np.round(np.array([[xo, yo], [xi, yi]]) * 4).astype(np.int32)
            cv2.line(img, tuple(p[0]), tuple(p[1]), _bgra(brow_col), round(0.06 * R), _LINE, 2)

        # 8. Nose: a small soft curve.
        nx, ny, nsx, _ = project(0.0, 0.16)
        if nsx > 0.2:
            _poly(
                img,
                [
                    (nx - 0.06 * R * nsx, ny),
                    (nx + 0.02 * R, ny + 0.05 * R),
                    (nx + 0.08 * R * nsx, ny),
                ],
                look.skin,
                skin_out,
                closed=False,
            )

        # 9. Mouth: a curve that smiles, opening into a filled shape when talking.
        mx, my, msx, msy = project(0.0, 0.45)
        if msx > 0.2:
            hw = 0.24 * R * msx * (1 + 0.15 * e.smile)
            curve = 0.09 * R * e.smile
            opening = 0.26 * R * max(0.0, min(1.0, e.mouth_open)) * msy
            us = np.linspace(-1, 1, 15)
            upper = [(mx + u * hw, my - curve * (u * u - 0.4)) for u in us]
            if opening > 0.02 * R:
                lower = [
                    (mx + u * hw, my - curve * (u * u - 0.4) + opening * math.sqrt(1 - u * u))
                    for u in us[::-1]
                ]
                _poly(img, upper + lower, _rgb(120, 36, 48), skin_out)
            else:
                _poly(img, upper, look.skin, skin_out, closed=False)
        if e.smile > 0.3 and eyes:  # rosy cheeks when smiling
            for x, y, sx in eyes:
                _ellipse(
                    img, (x, y + 0.22 * R), (0.08 * R * sx, 0.05 * R), _rgb(240, 140, 140), None
                )

        # 10. Glasses and hat.
        if look.glasses:
            for x, y, sx in eyes:
                cv2.ellipse(
                    img,
                    (round(x * 4), round(y * 4)),
                    (round(0.19 * R * sx * 4), round(0.17 * R * 4)),
                    0,
                    0,
                    360,
                    _bgra((40, 40, 40)),
                    5,
                    _LINE,
                    2,
                )
            if len(eyes) == 2:
                (x1, y1, s1), (x2, y2, s2) = eyes
                a = (round((x1 + 0.19 * R * s1) * 4), round(y1 * 4))
                b = (round((x2 - 0.19 * R * s2) * 4), round(y2 * 4))
                cv2.line(img, a, b, _bgra((40, 40, 40)), 5, _LINE, 2)
        if look.hat is not None:
            top = [
                (C[0] + 1.04 * R * math.cos(t), C[1] - 0.25 * R + 0.95 * R * math.sin(t))
                for t in np.linspace(math.pi, 2 * math.pi, 20)
            ]
            _poly(img, top, look.hat, _darker(look.hat))
            brim_y = C[1] - 0.25 * R
            _ellipse(
                img,
                (C[0] + 0.15 * R * turn, brim_y),
                (1.12 * R, 0.13 * R),
                look.hat,
                _darker(look.hat),
            )

        # 11. Tilt, then scale down for smooth edges.
        if abs(pose.roll) > 0.1:
            m = cv2.getRotationMatrix2D(C, -max(-60.0, min(60.0, pose.roll)), 1.0)
            img = cv2.warpAffine(img, m, (S, S), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
