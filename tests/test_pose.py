from __future__ import annotations

import numpy as np
import pytest

from npcify.pose import (
    LandmarkResult,
    PoseConfig,
    PoseEstimator,
    angles_to_rotation,
    blendshapes_to_avatar,
    coarse_pose,
    mirror_rotation,
    rotation_to_angles,
)
from npcify.types import AvatarExpression, BoundingBox, Decision, Frame, Track, TrackState

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)

# --- rotation maths ---


@pytest.mark.parametrize(
    "yaw, pitch, roll",
    [(0, 0, 0), (30, 0, 0), (-45, 0, 0), (0, 20, 0), (0, -35, 0), (0, 0, 15), (25, -10, 8)],
)
def test_round_trip_known_rotations(yaw: float, pitch: float, roll: float) -> None:
    assert rotation_to_angles(angles_to_rotation(yaw, pitch, roll)) == pytest.approx(
        (yaw, pitch, roll), abs=1e-6
    )


def test_angle_directions_in_camera_space() -> None:
    # Camera space: x right, y up, z toward the viewer; front of face = +z when facing us.
    front = np.array([0, 0, 1.0])
    up = np.array([0, 1.0, 0])
    assert (angles_to_rotation(30, 0, 0) @ front)[0] > 0  # yaw +: turned toward image right
    assert (angles_to_rotation(0, 20, 0) @ front)[1] > 0  # pitch +: looking up
    assert (angles_to_rotation(0, 0, 15) @ up)[0] > 0  # roll +: top of head toward right


def test_matrix_with_scale_still_gives_angles() -> None:
    r = 1.7 * angles_to_rotation(20, -5, 3)
    assert rotation_to_angles(r) == pytest.approx((20, -5, 3), abs=1e-6)


@pytest.mark.parametrize("yaw, pitch, roll", [(30, 10, 5), (-20, -15, -12)])
def test_mirroring_flips_yaw_and_roll_not_pitch(yaw: float, pitch: float, roll: float) -> None:
    mirrored = rotation_to_angles(mirror_rotation(angles_to_rotation(yaw, pitch, roll)))
    assert mirrored == pytest.approx((-yaw, pitch, -roll), abs=1e-6)


# --- expressions ---


def test_blendshape_mapping_uses_image_sides() -> None:
    e = blendshapes_to_avatar({"eyeBlinkRight": 0.9, "eyeBlinkLeft": 0.1})
    assert e.blink_left == pytest.approx(0.9)  # MediaPipe "Right" eye is on the image's left
    assert e.blink_right == pytest.approx(0.1)


def test_blendshape_mapping_known_values() -> None:
    e = blendshapes_to_avatar(
        {
            "jawOpen": 0.3,
            "mouthSmileLeft": 0.6,
            "mouthSmileRight": 0.4,
            "browInnerUp": 0.2,
            "browOuterUpLeft": 0.8,
            "browOuterUpRight": 0.6,
            "browDownLeft": 0.5,
            "browDownRight": 0.3,
        }
    )
    assert e.mouth_open == pytest.approx(0.5)  # 0.3 / 0.6
    assert e.smile == pytest.approx(0.5)
    assert e.brows_up == pytest.approx(0.7)  # max(inner 0.2, outer mean 0.7)
    assert e.brows_down == pytest.approx(0.4)


def test_look_direction() -> None:
    right = blendshapes_to_avatar({"eyeLookOutLeft": 0.8, "eyeLookInRight": 0.8})
    left = blendshapes_to_avatar({"eyeLookInLeft": 0.8, "eyeLookOutRight": 0.8})
    up = blendshapes_to_avatar({"eyeLookUpLeft": 0.6, "eyeLookUpRight": 0.6})
    assert right.look_x == pytest.approx(0.8) and left.look_x == pytest.approx(-0.8)
    assert up.look_y == pytest.approx(0.6)


def test_values_are_clipped_and_missing_ones_neutral() -> None:
    e = blendshapes_to_avatar({"jawOpen": 0.95})
    assert e.mouth_open == 1.0
    assert blendshapes_to_avatar({}) == AvatarExpression()


# --- coarse fallback ---


def five(eye_l, eye_r, nose, mouth_y=150.0):
    return (eye_r, eye_l, nose, (eye_l[0] + 5, mouth_y), (eye_r[0] - 5, mouth_y))


def test_coarse_frontal_is_near_zero() -> None:
    yaw, pitch, roll = coarse_pose(five((100, 100), (140, 100), (120, 128)))
    assert abs(yaw) < 3 and abs(roll) < 1e-6 and abs(pitch) < 10


def test_coarse_roll_from_eye_angle() -> None:
    # Image-right eye 20 px lower over 40 px: clockwise tilt of ~26.6 degrees.
    _, _, roll = coarse_pose(five((100, 100), (140, 120), (122, 135), mouth_y=165))
    assert roll == pytest.approx(26.57, abs=0.1)


def test_coarse_yaw_direction_from_nose_position() -> None:
    right, _, _ = coarse_pose(five((100, 100), (140, 100), (135, 128)))  # nose toward image right
    left, _, _ = coarse_pose(five((100, 100), (140, 100), (105, 128)))
    assert right > 15 and left < -15


def test_coarse_ignores_eye_order_so_mirroring_is_safe() -> None:
    pts = five((100, 100), (140, 100), (135, 128))
    swapped = (pts[1], pts[0], *pts[2:])
    assert coarse_pose(pts) == pytest.approx(coarse_pose(swapped))


# --- estimator ---


class FakeLandmarker:
    def __init__(self, result: LandmarkResult | None) -> None:
        self.result = result
        self.calls = 0

    def run(self, crop: Frame) -> LandmarkResult | None:
        self.calls += 1
        return self.result


def frontal(box: BoundingBox):
    x, y, w, h = box.x, box.y, box.width, box.height
    return (
        (x + 0.3 * w, y + 0.4 * h),
        (x + 0.7 * w, y + 0.4 * h),
        (x + 0.5 * w, y + 0.58 * h),
        (x + 0.35 * w, y + 0.75 * h),
        (x + 0.65 * w, y + 0.75 * h),
    )


def track(tid: int, x: int = 100, size: int = 120, seen: bool = True) -> Track:
    box = BoundingBox(x, 200, size, size)
    return Track(
        tid,
        TrackState.CONFIRMED if seen else TrackState.LOST,
        box,
        frames_since_seen=0 if seen else 2,
        landmarks=frontal(box) if seen else None,
    )


RESULT = LandmarkResult(
    angles_to_rotation(20, -5, 3),
    {"jawOpen": 0.6, "eyeBlinkRight": 1.0},
    np.full((478, 2), 0.5, np.float32),
)


def test_landmarker_result_becomes_face_state_in_full_frame_coords() -> None:
    est = PoseEstimator(FakeLandmarker(RESULT))
    states = est.update(FRAME, [track(1)], [], now=0.0)
    s = states[1]
    assert s.source == "landmarker" and s.confidence > 0.5
    assert (s.yaw, s.pitch, s.roll) == pytest.approx((20, -5, 3), abs=1e-6)
    assert s.avatar.mouth_open == pytest.approx(1.0) and s.avatar.blink_left == pytest.approx(1.0)
    assert len(s.expressions) == 2
    # Mesh point 0.5, 0.5 of the crop = the face box centre.
    assert s.mesh is not None and s.mesh[0] == pytest.approx((160, 260), abs=1)


def test_small_faces_use_coarse_pose_without_running_the_landmarker() -> None:
    model = FakeLandmarker(RESULT)
    est = PoseEstimator(model, PoseConfig(min_landmarker_width=50))
    s = est.update(FRAME, [track(1, size=40)], [], now=0.0)[1]
    assert s.source == "coarse" and model.calls == 0
    assert s.avatar == AvatarExpression()


def test_landmarker_failure_falls_back_to_coarse() -> None:
    est = PoseEstimator(FakeLandmarker(None))
    assert est.update(FRAME, [track(1)], [], now=0.0)[1].source == "coarse"


def test_no_landmarks_and_no_history_gives_source_none() -> None:
    est = PoseEstimator(FakeLandmarker(None))
    assert est.update(FRAME, [track(1, seen=False)], [], now=0.0)[1].source == "none"


def test_round_robin_spreads_updates_fairly() -> None:
    model = FakeLandmarker(RESULT)
    est = PoseEstimator(model, PoseConfig(max_per_frame=2))
    tracks = [track(i, x=100 + 150 * i) for i in range(1, 6)]
    counts = dict.fromkeys(range(1, 6), 0)
    for frame_no in range(10):
        before = dict(est._last_run)
        est.update(FRAME, tracks, [], now=frame_no / 30)
        assert model.calls == 2 * (frame_no + 1)
        for tid, last in est._last_run.items():
            if before.get(tid) != last:
                counts[tid] += 1
    assert set(counts.values()) == {4}  # 20 runs shared equally by 5 tracks


def test_values_held_between_updates() -> None:
    model = FakeLandmarker(RESULT)
    est = PoseEstimator(model, PoseConfig(max_per_frame=1))
    tracks = [track(1), track(2, x=400)]
    est.update(FRAME, tracks, [], now=0.0)
    first = est.states[1]
    est.update(FRAME, tracks, [], now=1 / 30)  # track 2's turn
    assert est.states[1] is first


def test_kept_people_never_get_a_face_state_and_removed_tracks_are_cleaned_up() -> None:
    est = PoseEstimator(FakeLandmarker(RESULT))
    tracks = [track(1), track(2, x=400)]
    states = est.update(FRAME, tracks, [Decision(1, keep_real=True)], now=0.0)
    assert set(states) == {2}
    states = est.update(FRAME, tracks, [Decision(2, keep_real=True)], now=0.1)
    assert set(states) == {1}  # track 2 became kept: its state is dropped
    states = est.update(FRAME, [], [], now=0.2)
    assert states == {} and est._smoothers == {} and est._last_run == {}


def test_smoothing_is_applied_per_track() -> None:
    model = FakeLandmarker(RESULT)
    est = PoseEstimator(model)
    est.update(FRAME, [track(1)], [], now=0.0)
    model.result = LandmarkResult(angles_to_rotation(-20, -5, 3), {}, RESULT.points)
    s = est.update(FRAME, [track(1)], [], now=1 / 30)[1]
    assert -20 < s.yaw < 20  # moving toward -20, not jumping straight there


def test_all_avatar_fields_are_smoothed_in_order() -> None:
    # Guards the field order used when packing AvatarExpression into the filter.
    fields = list(vars(AvatarExpression()).keys())
    assert fields == [
        "blink_left",
        "blink_right",
        "mouth_open",
        "smile",
        "brows_up",
        "brows_down",
        "look_x",
        "look_y",
    ]
