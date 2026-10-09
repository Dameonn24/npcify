from __future__ import annotations

import math

import numpy as np
import pytest

from npcify.avatar import NpcRenderer, look_seed, make_look
from npcify.pipeline import Pipeline
from npcify.render import AvatarLayer
from npcify.types import (
    AvatarExpression,
    BoundingBox,
    Detection,
    FaceState,
    Frame,
    HeadPose,
    Track,
    TrackState,
)

RENDERER = NpcRenderer()
LOOK = make_look(3)
EYE_WHITE = np.array([250, 250, 250])
MOUTH = np.array([48, 36, 120])  # BGR of the open-mouth fill


NEUTRAL_POSE = HeadPose()
NEUTRAL = AvatarExpression()


def render(pose: HeadPose = NEUTRAL_POSE, expression: AvatarExpression = NEUTRAL, look=LOOK):
    return RENDERER.render(look, pose, expression, 256)


def colour_mask(img: np.ndarray, colour: np.ndarray, tol: int = 12) -> np.ndarray:
    return (np.abs(img[:, :, :3].astype(int) - colour).max(axis=2) <= tol) & (img[:, :, 3] == 255)


# --- looks ---


def test_same_track_and_session_give_the_same_look() -> None:
    assert make_look(look_seed(5, 1234)) == make_look(look_seed(5, 1234))


def test_different_tracks_or_sessions_usually_differ() -> None:
    looks = {make_look(look_seed(t, 1234)) for t in range(30)}
    assert len(looks) >= 25
    assert make_look(look_seed(5, 1)) != make_look(look_seed(5, 2))


def test_looks_cover_the_variety() -> None:
    looks = [make_look(s) for s in range(400)]
    assert {lk.hair_style for lk in looks} == {"bald", "short", "long", "bun", "curly"}
    assert len({lk.skin for lk in looks}) >= 8
    assert any(lk.glasses for lk in looks) and any(lk.hat for lk in looks)


# --- renderer contract and animation ---


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize(
    "pose",
    [
        HeadPose(),
        HeadPose(yaw=80),
        HeadPose(yaw=-80),
        HeadPose(pitch=45, roll=40),
        HeadPose(pitch=-45, roll=-60),
        HeadPose(30, -20, 15),
    ],
)
def test_cover_circle_is_always_fully_opaque(seed: int, pose: HeadPose) -> None:
    expr = AvatarExpression(blink_left=1, mouth_open=1, smile=1, brows_up=1, look_x=1)
    img = RENDERER.render(make_look(seed), pose, expr, 256)
    cx, cy = (c * 256 for c in RENDERER.cover_centre)
    r = RENDERER.cover_radius * 256
    yy, xx = np.mgrid[0:256, 0:256]
    inside = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r**2
    assert (img[:, :, 3][inside] == 255).all()


def test_blink_closes_the_eyes() -> None:
    open_eyes = colour_mask(render(), EYE_WHITE).sum()
    closed = colour_mask(
        render(expression=AvatarExpression(blink_left=1, blink_right=1)), EYE_WHITE
    ).sum()
    one = colour_mask(render(expression=AvatarExpression(blink_left=1)), EYE_WHITE).sum()
    assert open_eyes > 200
    assert closed < 0.1 * open_eyes
    assert 0.3 * open_eyes < one < 0.7 * open_eyes  # one eye closed, one open


def test_mouth_open_grows_the_mouth() -> None:
    areas = [
        colour_mask(render(expression=AvatarExpression(mouth_open=m)), MOUTH).sum()
        for m in (0, 0.5, 1)
    ]
    assert areas[0] == 0 and 0 < areas[1] < areas[2]


def test_roll_rotates_the_head() -> None:
    def eye_angle(img: np.ndarray) -> float:
        ys, xs = np.nonzero(colour_mask(img, EYE_WHITE))
        left = xs < np.median(xs)
        dy = ys[~left].mean() - ys[left].mean()
        dx = xs[~left].mean() - xs[left].mean()
        return math.degrees(math.atan2(dy, dx))

    assert eye_angle(render()) == pytest.approx(0, abs=2)
    assert eye_angle(render(HeadPose(roll=25))) == pytest.approx(25, abs=4)  # clockwise


def test_yaw_slides_the_features_toward_the_turn() -> None:
    def eye_centre_x(img: np.ndarray) -> float:
        return float(np.nonzero(colour_mask(img, EYE_WHITE))[1].mean())

    assert eye_centre_x(render(HeadPose(yaw=40))) > eye_centre_x(render()) + 15
    assert eye_centre_x(render(HeadPose(yaw=-40))) < eye_centre_x(render()) - 15


# --- compositing ---

FACE_BOX = BoundingBox(300, 200, 120, 150)


def face_state(tid: int, source: str = "landmarker") -> FaceState:
    return FaceState(tid, yaw=10, source=source, avatar=AvatarExpression(smile=0.5))


def confirmed(tid: int = 1, box: BoundingBox = FACE_BOX, state=TrackState.CONFIRMED) -> Track:
    return Track(tid, state, box)


def opaque_mask(layer: AvatarLayer, tracks, faces, shape=(720, 1280, 3)) -> np.ndarray:
    """Pixels the avatar fully covers: identical output on black and white backgrounds."""
    black = layer.apply(np.zeros(shape, np.uint8), tracks, faces, now=0.0, advance=False)
    white = layer.apply(np.full(shape, 255, np.uint8), tracks, faces, now=0.0, advance=False)
    return (black == white).all(axis=2)


def faded_in_layer(tracks, faces) -> AvatarLayer:
    layer = AvatarLayer(RENDERER, session_seed=1, fade_frames=5)
    for i in range(6):
        layer.apply(np.zeros((720, 1280, 3), np.uint8), tracks, faces, now=i / 30)
    return layer


@pytest.mark.parametrize("roll", [0, 30, -55])
def test_avatar_fully_covers_the_face_box(roll: float) -> None:
    tracks = [confirmed()]
    faces = {1: FaceState(1, yaw=40, roll=roll, source="landmarker")}
    layer = faded_in_layer(tracks, faces)
    mask = opaque_mask(layer, tracks, faces)
    b = FACE_BOX
    assert mask[b.y : b.y + b.height, b.x : b.x + b.width].all()


def test_anonymisation_is_still_underneath_the_avatar() -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)

    class Fixed:
        def detect(self, f: Frame) -> list[Detection]:
            return [Detection(FACE_BOX, 0.95)]

    pipeline = Pipeline(detector=Fixed(), avatars=AvatarLayer(RENDERER, session_seed=1))
    pipeline.pose = None
    for _ in range(10):
        pipeline.process_frame(frame)
        pipeline.faces = {t.track_id: face_state(t.track_id) for t in pipeline.tracks}
    out = pipeline.process_frame(frame)
    anonymised = pipeline.anonymise(frame)
    avatar_px = (out != anonymised).any(axis=2)
    hidden = pipeline.anonymizer.coverage_mask(frame.shape, [r.box for r in pipeline.regions])
    edge = hidden & ~avatar_px  # inside the padded region, but not drawn on by the avatar
    assert edge.sum() > 500
    assert (out[edge] == anonymised[edge]).all()
    assert (out[edge] != frame[edge]).any(axis=1).mean() > 0.95


@pytest.mark.parametrize(
    "track, face",
    [
        (confirmed(state=TrackState.TENTATIVE), face_state(1)),
        (confirmed(state=TrackState.LOST), face_state(1)),
        (confirmed(), None),  # no FaceState
        (confirmed(), face_state(1, source="none")),
        (confirmed(box=BoundingBox(300, 200, 30, 36)), face_state(1, source="coarse")),
    ],
)
def test_uncertain_faces_get_blur_never_an_avatar(track: Track, face: FaceState | None) -> None:
    layer = AvatarLayer(RENDERER, session_seed=1)
    frame = np.full((720, 1280, 3), 77, np.uint8)
    faces = {} if face is None else {1: face}
    for i in range(10):
        out = layer.apply(frame, [track], faces, now=i / 30)
        assert (out == frame).all()


def test_big_coarse_face_gets_a_neutral_avatar() -> None:
    layer = AvatarLayer(RENDERER, session_seed=1)
    frame = np.full((720, 1280, 3), 77, np.uint8)
    out = layer.apply(frame, [confirmed()], {1: face_state(1, source="coarse")}, now=0.0)
    assert (out != frame).any()


def test_fades_in_and_switches_off_instantly() -> None:
    layer = AvatarLayer(RENDERER, session_seed=1, fade_frames=5)
    frame = np.zeros((720, 1280, 3), np.uint8)
    faces = {1: face_state(1)}
    centre = (FACE_BOX.y + FACE_BOX.height // 2, FACE_BOX.x + FACE_BOX.width // 2 + 30)
    levels = [layer.apply(frame, [confirmed()], faces, now=i / 30)[centre].sum() for i in range(6)]
    assert levels[0] < levels[2] < levels[4] == levels[5]  # gradual, then full
    out = layer.apply(frame, [confirmed(state=TrackState.LOST)], faces, now=0.3)
    assert (out == frame).all()  # instantly gone


def test_unmatched_detections_only_get_blur() -> None:
    class Fixed:
        def detect(self, f: Frame) -> list[Detection]:
            return [Detection(FACE_BOX, 0.4)]  # low confidence: never starts a track

    pipeline = Pipeline(detector=Fixed(), avatars=AvatarLayer(RENDERER, session_seed=1))
    frame = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    for _ in range(5):
        out = pipeline.process_frame(frame)
    assert pipeline.tracks == []
    assert (out == pipeline.anonymise(frame)).all()


@pytest.mark.parametrize(
    "box",
    [
        BoundingBox(-60, 300, 120, 150),
        BoundingBox(1200, 650, 120, 150),
        BoundingBox(-100, -120, 150, 160),
    ],
)
def test_avatars_partly_off_screen_are_clipped(box: BoundingBox) -> None:
    layer = AvatarLayer(RENDERER, session_seed=1)
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = layer.apply(frame, [confirmed(box=box)], {1: face_state(1)}, now=0.0)
    assert out.shape == frame.shape


def test_kept_people_never_get_an_avatar() -> None:
    # Kept people have no FaceState (pose only covers hidden tracks), so: no avatar.
    layer = AvatarLayer(RENDERER, session_seed=1)
    frame = np.zeros((720, 1280, 3), np.uint8)
    for i in range(6):
        out = layer.apply(frame, [confirmed()], {}, now=i / 30)
    assert (out == frame).all()


def test_reroll_changes_looks() -> None:
    layer = AvatarLayer(RENDERER, session_seed=1)
    frame = np.zeros((720, 1280, 3), np.uint8)
    faces = {1: face_state(1)}
    for i in range(6):
        first = layer.apply(frame, [confirmed()], faces, now=i / 30)
    layer.reroll(session_seed=99)
    for i in range(6):
        second = layer.apply(frame, [confirmed()], faces, now=(6 + i) / 30)
    assert (first != second).any()


def test_pipeline_blur_mode_skips_avatars() -> None:
    pipeline = Pipeline(
        detector=type("D", (), {"detect": lambda self, f: []})(), avatars=AvatarLayer(RENDERER)
    )
    pipeline.show_avatars = False
    frame = np.zeros((100, 100, 3), np.uint8)
    assert (pipeline.process_frame(frame) == frame).all()
    assert "avatar" not in pipeline.timer.averages()
