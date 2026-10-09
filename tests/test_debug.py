import numpy as np

from npcify.debug import TrackTrails, draw_detections, draw_tracks, track_colour
from npcify.types import BoundingBox, Detection, Track, TrackState


def test_draw_detections_draws_on_a_copy() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    lm = ((80.0, 80.0), (120.0, 80.0), (100.0, 100.0), (85.0, 120.0), (115.0, 120.0))
    det = Detection(BoundingBox(50, 50, 100, 100), 0.9, lm)
    out = draw_detections(frame, [det])
    assert not frame.any()  # input untouched
    assert out[50, 100].any()  # box edge drawn
    assert out[100, 100].any()  # nose landmark drawn


def test_draw_detections_without_landmarks() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(frame, [Detection(BoundingBox(0, 0, 100, 100), 0.5)])
    assert out.shape == frame.shape


def test_track_colour_is_stable_and_distinct() -> None:
    assert track_colour(3) == track_colour(3)
    assert len({track_colour(i) for i in range(1, 11)}) == 10


def test_draw_tracks_all_states_with_trails() -> None:
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    tracks = [
        Track(1, TrackState.CONFIRMED, BoundingBox(10, 10, 60, 60), confidence=0.9),
        Track(2, TrackState.TENTATIVE, BoundingBox(100, 10, 60, 60), confidence=0.7),
        Track(3, TrackState.LOST, BoundingBox(10, 150, 80, 80), frames_since_seen=4),
    ]
    trails = TrackTrails(length=3)
    for _ in range(5):
        trails.update(tracks)
    assert len(trails.get(1)) == 3
    out = draw_tracks(frame, tracks, trails, [Detection(BoundingBox(200, 200, 40, 40), 0.4)])
    assert not frame.any()
    assert out[10, 40].any() and out[10, 130].any()  # confirmed and tentative edges
    assert out[220, 200].any()  # raw detection drawn
    trails.update(tracks[:1])
    assert trails.get(2) == []  # trails dropped for tracks that are gone
