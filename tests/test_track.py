import pytest

from npcify.track import Tracker, TrackerConfig
from npcify.types import BoundingBox, Detection, Track, TrackState

FRAME = (1280, 720)


def face(cx: float, cy: float, size: float = 80, conf: float = 0.9) -> Detection:
    return Detection(
        BoundingBox(round(cx - size / 2), round(cy - size / 2), round(size), round(size)), conf
    )


def centre(t: Track) -> tuple[float, float]:
    return t.box.x + t.box.width / 2, t.box.y + t.box.height / 2


def by_id(tracks: list[Track]) -> dict[int, Track]:
    return {t.track_id: t for t in tracks}


def test_straight_line_keeps_one_id_for_100_frames() -> None:
    tracker = Tracker()
    ids = set()
    for i in range(100):
        tracks = tracker.update([face(100 + 8 * i, 300)], FRAME)
        assert len(tracks) == 1
        ids.add(tracks[0].track_id)
    assert ids == {1}
    assert tracks[0].state is TrackState.CONFIRMED
    assert tracks[0].velocity[0] == pytest.approx(8, abs=0.5)
    assert abs(centre(tracks[0])[0] - (100 + 8 * 99)) < 3  # no lag behind the face


def test_new_track_is_confirmed_after_min_hits() -> None:
    tracker = Tracker()
    states = [tracker.update([face(300, 300)], FRAME)[0].state for _ in range(4)]
    assert states == [
        TrackState.TENTATIVE,
        TrackState.TENTATIVE,
        TrackState.CONFIRMED,
        TrackState.CONFIRMED,
    ]


def test_short_gap_keeps_id_and_predicts_position() -> None:
    tracker = Tracker()
    for i in range(30):
        tracker.update([face(100 + 6 * i, 300)], FRAME)
    for i in range(30, 40):  # 10 missed frames
        [t] = tracker.update([], FRAME)
        assert t.state is TrackState.LOST
        # The predicted box keeps moving with the face and still covers it.
        assert t.box.x <= 100 + 6 * i - 40 and t.box.x + t.box.width >= 100 + 6 * i + 40
    [t] = tracker.update([face(100 + 6 * 40, 300)], FRAME)
    assert t.track_id == 1
    assert t.state is TrackState.CONFIRMED
    assert tracker.stats.lost_events == 1 and tracker.stats.recovered == 1


def test_long_gap_removes_track_and_new_face_gets_new_id() -> None:
    cfg = TrackerConfig(max_age=15)
    tracker = Tracker(cfg)
    for _ in range(10):
        tracker.update([face(300, 300)], FRAME)
    for i in range(cfg.max_age):
        assert len(tracker.update([], FRAME)) == 1, f"removed too early at gap frame {i + 1}"
    assert tracker.update([], FRAME) == []
    [t] = tracker.update([face(300, 300)], FRAME)
    assert t.track_id == 2
    assert t.state is TrackState.TENTATIVE


def test_crossing_faces_keep_their_ids() -> None:
    tracker = Tracker()
    first_ids = None
    for i in range(80):
        a = face(200 + 10 * i, 300)  # moving right
        b = face(1000 - 10 * i, 330)  # moving left, slightly lower, passes in front
        # While they overlap, the one behind (a) is hidden from the detector.
        overlap = abs(a.box.x - b.box.x) < 70
        tracks = tracker.update([b] if overlap else [a, b], FRAME)
        if i == 10:
            first_ids = {round(centre(t)[0]) < 640: t.track_id for t in tracks}
    ids = {round(centre(t)[0]) < 640: t.track_id for t in tracks}
    # The face that started on the left is now on the right, with the same ID.
    assert ids[False] == first_ids[True]
    assert ids[True] == first_ids[False]
    assert tracker.stats.ids_created == 2


def test_one_frame_false_detection_never_confirmed() -> None:
    tracker = Tracker()
    [t] = tracker.update([face(500, 300)], FRAME)
    assert t.state is TrackState.TENTATIVE
    for _ in range(5):
        assert tracker.update([], FRAME) == []
    assert tracker.stats.ids_confirmed == 0


def test_lost_box_grows_each_frame_up_to_cap() -> None:
    cfg = TrackerConfig(lost_growth=0.05, max_lost_growth=1.5, max_age=30)
    tracker = Tracker(cfg)
    for _ in range(20):
        [t] = tracker.update([face(600, 360, size=100)], FRAME)
    seen_width = t.box.width
    widths = [tracker.update([], FRAME)[0].box.width for _ in range(15)]
    assert all(b >= a for a, b in zip([seen_width, *widths], widths, strict=False))
    assert widths[0] > seen_width  # grows from the first missed frame
    assert widths[-1] == pytest.approx(seen_width * 1.5, abs=3)  # capped at 150%
    assert widths[-1] == widths[-5]  # no further growth once capped


def test_low_confidence_keeps_track_alive() -> None:
    tracker = Tracker()
    for _ in range(5):
        tracker.update([face(300, 300, conf=0.9)], FRAME)
    for _ in range(30):  # e.g. head turned: detector is much less sure
        [t] = tracker.update([face(300, 300, conf=0.4)], FRAME)
        assert t.state is TrackState.CONFIRMED
    assert t.track_id == 1


def test_low_confidence_alone_never_starts_a_track() -> None:
    tracker = Tracker()
    for _ in range(30):
        assert tracker.update([face(300, 300, conf=0.4)], FRAME) == []
    assert tracker.stats.ids_created == 0


def test_below_low_threshold_is_ignored() -> None:
    tracker = Tracker()
    for _ in range(5):
        tracker.update([face(300, 300)], FRAME)
    tracker.update([face(300, 300, conf=0.1)], FRAME)
    [t] = tracker.update([], FRAME)
    assert t.state is TrackState.LOST


def test_skipped_frames_only_predict() -> None:
    tracker = Tracker()
    for i in range(20):
        tracker.update([face(100 + 5 * i, 300)], FRAME)
    for _ in range(3):  # detection not run on these frames
        [t] = tracker.update(None, FRAME)
        assert t.state is TrackState.CONFIRMED
    assert centre(t)[0] == pytest.approx(100 + 5 * 22, abs=2)


def test_track_predicted_off_screen_is_removed() -> None:
    tracker = Tracker()
    for i in range(20):
        tracker.update([face(1100 + 8 * i, 300)], FRAME)  # walking out to the right
    for _ in range(14):
        tracks = tracker.update([], FRAME)
    assert tracks == []


def test_boxes_are_clipped_to_frame() -> None:
    tracker = Tracker()
    for _ in range(5):
        [t] = tracker.update([face(10, 10, size=80)], FRAME)
    assert t.box.x == 0 and t.box.y == 0


def confirmed_ids(tracker: Tracker, frames: list[list[Detection]]) -> set[int]:
    ids: set[int] = set()
    for dets in frames:
        ids |= {t.track_id for t in tracker.update(dets, FRAME) if t.state is TrackState.CONFIRMED}
    return ids


def test_hands_over_face_keeps_id() -> None:
    # Hands coming up push the box down and shrink it before the face vanishes for
    # 1.5 s, then it reappears a little away from where it was predicted.
    frames = [[face(640, 360, 180)] for _ in range(60)]
    frames += [[face(640, 360 + 8 * k, 180 - 10 * k, conf=0.8)] for k in range(5)]
    frames += [[] for _ in range(45)]
    frames += [[face(625, 345, 170)] for _ in range(20)]
    assert confirmed_ids(Tracker(), frames) == {1}


def test_turning_away_and_back_keeps_id() -> None:
    # Spinning on a chair: the face narrows and slides sideways, disappears for
    # 1.5 s, then comes back narrow from the other side and widens again.
    frames = [[face(640, 360, 180)] for _ in range(60)]
    frames += [[face(640 + 12 * k, 360, 180 - 20 * k)] for k in range(6)]
    frames += [[] for _ in range(45)]
    frames += [[face(560 + 14 * k, 365, 80 + 18 * k)] for k in range(6)]
    frames += [[face(645, 365, 180)] for _ in range(20)]
    assert confirmed_ids(Tracker(), frames) == {1}


def test_someone_else_far_away_does_not_inherit_a_lost_id() -> None:
    tracker = Tracker()
    for _ in range(10):
        tracker.update([face(300, 360, 120)], FRAME)
    tracker.update([], FRAME)
    tracks = [tracker.update([face(1000, 360, 120)], FRAME) for _ in range(4)][-1]
    states = {t.track_id: t.state for t in tracks}
    assert states[1] is TrackState.LOST
    assert states[2] is TrackState.CONFIRMED


def test_lost_prediction_settles_instead_of_drifting() -> None:
    tracker = Tracker()
    for i in range(30):
        tracker.update([face(400 + 6 * i, 360)], FRAME)
    for _ in range(40):
        [t] = tracker.update([], FRAME)
    # Without decay it would have drifted ~240 px; it should settle within ~60 px.
    assert abs(centre(t)[0] - (400 + 6 * 29)) < 80
