from npcify.decide import HideEveryone, regions_to_cover
from npcify.types import BoundingBox, Decision, Detection, Track, TrackState

FRAME = None  # policies don't look at pixels yet


def track(tid: int, state: TrackState, box: BoundingBox) -> Track:
    return Track(tid, state, box)


TRACKS = [
    track(1, TrackState.CONFIRMED, BoundingBox(100, 100, 80, 80)),
    track(2, TrackState.TENTATIVE, BoundingBox(300, 100, 80, 80)),
    track(3, TrackState.LOST, BoundingBox(500, 100, 120, 120)),
]


def test_hide_everyone_hides_every_track() -> None:
    decisions = HideEveryone().decide(FRAME, TRACKS)
    assert {d.track_id for d in decisions} == {1, 2, 3}
    assert not any(d.keep_real for d in decisions)


def test_every_track_in_every_state_is_covered() -> None:
    regions = regions_to_cover(TRACKS, [], HideEveryone().decide(FRAME, TRACKS))
    assert {r.track_id: r.box for r in regions} == {t.track_id: t.box for t in TRACKS}
    assert {r.reason for r in regions} == {"track:confirmed", "track:tentative", "track:lost"}


def test_tracks_without_a_decision_are_still_covered() -> None:
    regions = regions_to_cover(TRACKS, [], [])
    assert len(regions) == 3


def test_unmatched_detection_is_covered() -> None:
    new_face = Detection(BoundingBox(800, 300, 60, 60), 0.35)
    regions = regions_to_cover(TRACKS, [new_face], [])
    assert [r for r in regions if r.reason == "detection"] == [regions[-1]] and regions[
        -1
    ].box == new_face.box


def test_detection_inside_hidden_track_is_not_duplicated() -> None:
    inside = Detection(BoundingBox(105, 105, 70, 70), 0.9)
    regions = regions_to_cover(TRACKS, [inside], [])
    assert len(regions) == 3


def test_detection_sticking_out_of_its_track_is_covered_too() -> None:
    # The smoothed track box lags the face; the raw detection must still be covered.
    shifted = Detection(BoundingBox(130, 100, 80, 80), 0.9)
    regions = regions_to_cover(TRACKS, [shifted], [])
    assert shifted.box in [r.box for r in regions]


def test_kept_track_is_not_covered_but_everyone_else_is() -> None:
    decisions = [Decision(1, keep_real=True), Decision(2, False), Decision(3, False)]
    dets = [
        Detection(BoundingBox(102, 98, 80, 82), 0.9),  # the kept person's own detection
        Detection(BoundingBox(800, 300, 60, 60), 0.9),  # a stranger
    ]
    regions = regions_to_cover(TRACKS, dets, decisions)
    boxes = [r.box for r in regions]
    assert TRACKS[0].box not in boxes
    assert dets[0].box not in boxes
    assert dets[1].box in boxes
    assert len(regions) == 3  # tracks 2 and 3, plus the stranger
