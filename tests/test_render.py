import numpy as np
import pytest

from npcify.render import STYLES, Anonymizer, Padding, _region_mask, core_ellipse
from npcify.types import BoundingBox


def noisy_frame(h: int = 360, w: int = 640, seed: int = 0) -> np.ndarray:
    """Random texture, so any pixel that is processed is almost certain to change."""
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("style", STYLES)
def test_face_region_changes_and_far_pixels_do_not(style: str) -> None:
    frame = noisy_frame()
    box = BoundingBox(280, 140, 80, 100)
    out = Anonymizer(style=style).apply(frame, [box])
    face = (slice(box.y, box.y + box.height), slice(box.x, box.x + box.width))
    changed = (out[face] != frame[face]).any(axis=2)
    assert changed.mean() > 0.95
    # Well outside the padded, feathered ellipse: untouched.
    assert (out[:, :150] == frame[:, :150]).all()
    assert (out[:, 500:] == frame[:, 500:]).all()
    assert (out[:40] == frame[:40]).all() and (out[320:] == frame[320:]).all()


def test_input_frame_is_not_modified() -> None:
    frame = noisy_frame()
    before = frame.copy()
    Anonymizer().apply(frame, [BoundingBox(100, 100, 80, 80)])
    assert (frame == before).all()


@pytest.mark.parametrize(
    "padding",
    [Padding(), Padding(0, 0, 0), Padding(0.1, 0.1, 0.3), Padding(1.0, 0.2, -0.1)],
)
@pytest.mark.parametrize("size", [(12, 12), (80, 100), (200, 140), (37, 91)])
def test_face_box_is_inside_fully_covered_core(padding: Padding, size: tuple[int, int]) -> None:
    w, h = size
    box = BoundingBox(300, 200, w, h)
    # Geometrically: all four corners inside the core ellipse.
    e = core_ellipse(box, padding)
    for x in (box.x, box.x + w):
        for y in (box.y, box.y + h):
            assert ((x - e.cx) / e.a) ** 2 + ((y - e.cy) / e.b) ** 2 <= 1.0
    # And in the actual mask: every face-box pixel has alpha exactly 1.
    (x1, y1, _, _), alpha = _region_mask(box, 1280, 720, padding, feather=0.15)
    face_alpha = alpha[box.y - y1 : box.y - y1 + h, box.x - x1 : box.x - x1 + w]
    assert face_alpha.shape == (h, w)
    assert (face_alpha == 1.0).all()
    # Feathering exists, but only outside the face box.
    assert ((alpha > 0) & (alpha < 1)).any()


def test_tiny_face_is_meaningfully_pixelated() -> None:
    frame = noisy_frame(seed=3)
    box = BoundingBox(100, 100, 12, 12)
    out = Anonymizer(style="pixelate").apply(frame, [box])
    face_in = frame[100:112, 100:112].reshape(-1, 3)
    face_out = out[100:112, 100:112].reshape(-1, 3)
    # 144 distinct random pixels collapse into a handful of blocks.
    assert len(np.unique(face_in, axis=0)) > 100
    assert len(np.unique(face_out, axis=0)) <= 25
    # And the default style removes most of the detail too.
    out = Anonymizer().apply(frame, [box])
    assert out[100:112, 100:112].std() < 0.5 * frame[100:112, 100:112].std()


@pytest.mark.parametrize(
    "box",
    [
        BoundingBox(-40, 100, 100, 100),  # off the left edge
        BoundingBox(600, 300, 100, 100),  # off the bottom-right corner
        BoundingBox(-50, -50, 70, 70),  # mostly off the top-left
        BoundingBox(2000, 2000, 50, 50),  # entirely off-screen
    ],
)
def test_faces_partly_off_screen_are_covered_to_the_edge(box: BoundingBox) -> None:
    frame = noisy_frame()
    anonymizer = Anonymizer()
    out = anonymizer.apply(frame, [box])
    assert out.shape == frame.shape
    mask = anonymizer.coverage_mask(frame.shape, [box])
    x1, y1 = max(0, box.x), max(0, box.y)
    x2, y2 = min(640, box.x + box.width), min(360, box.y + box.height)
    if x2 > x1 and y2 > y1:
        assert mask[y1:y2, x1:x2].all()
        assert (out[y1:y2, x1:x2] != frame[y1:y2, x1:x2]).any(axis=2).mean() > 0.9


def test_coverage_mask_matches_what_apply_changes() -> None:
    frame = noisy_frame()
    boxes = [BoundingBox(100, 80, 90, 110), BoundingBox(400, 150, 60, 60)]
    anonymizer = Anonymizer(style="solid", colour=(1, 2, 3))
    out = anonymizer.apply(frame, boxes)
    mask = anonymizer.coverage_mask(frame.shape, boxes)
    assert (out[mask] == (1, 2, 3)).all()


@pytest.mark.parametrize("style", STYLES)
def test_deterministic(style: str) -> None:
    frame = noisy_frame(seed=7)
    boxes = [BoundingBox(50, 60, 70, 90), BoundingBox(300, 100, 150, 180)]
    a = Anonymizer(style=style).apply(frame, boxes)
    b = Anonymizer(style=style).apply(frame.copy(), list(boxes))
    assert (a == b).all()


def test_unknown_style_rejected() -> None:
    with pytest.raises(ValueError):
        Anonymizer(style="sparkles")
