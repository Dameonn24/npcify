"""Keep-list policy tests, with synthetic faceprints instead of a real model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from npcify.decide import KeepConfig, KeepListPolicy, regions_to_cover
from npcify.recognise import Enrolment, Faceprint, KeepList, QualityGate, normalise
from npcify.types import BoundingBox, Detection, Frame, Point, Track, TrackState

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
DIM = 128


def identity(n: int) -> Faceprint:
    """A distinct, fixed 'person' vector."""
    return normalise(np.random.default_rng(1_000_000 + n).normal(size=DIM))


def near(v: Faceprint, noise: float, seed: int) -> Faceprint:
    return normalise(v + noise * np.random.default_rng(seed).normal(size=DIM) / np.sqrt(DIM))


ALICE, BOB, CAROL = identity(1), identity(2), identity(3)


def frontal(box: BoundingBox) -> tuple[Point, ...]:
    x, y, w, h = box.x, box.y, box.width, box.height
    return (
        (x + 0.3 * w, y + 0.4 * h),
        (x + 0.7 * w, y + 0.4 * h),
        (x + 0.5 * w, y + 0.58 * h),
        (x + 0.35 * w, y + 0.75 * h),
        (x + 0.65 * w, y + 0.75 * h),
    )


def side_on(box: BoundingBox) -> tuple[Point, ...]:
    x, y, w, h = box.x, box.y, box.width, box.height
    return (
        (x + 0.55 * w, y + 0.4 * h),
        (x + 0.75 * w, y + 0.4 * h),
        (x + 0.95 * w, y + 0.58 * h),
        (x + 0.6 * w, y + 0.75 * h),
        (x + 0.8 * w, y + 0.75 * h),
    )


def track(
    tid: int,
    x: int = 100,
    size: int = 120,
    state: TrackState = TrackState.CONFIRMED,
    seen: bool = True,
    landmarks=frontal,
) -> Track:
    box = BoundingBox(x, 200, size, size)
    return Track(
        tid,
        state,
        box,
        frames_since_seen=0 if seen else 3,
        landmarks=landmarks(box) if seen and landmarks else None,
    )


class ScriptedModel:
    """Returns the faceprint the test assigned to each track (looked up by box x)."""

    def __init__(self) -> None:
        self.faces: dict[int, Faceprint] = {}
        self.calls = 0

    def faceprint(self, frame: Frame, box: BoundingBox, landmarks: Sequence[Point]) -> Faceprint:
        self.calls += 1
        return self.faces[box.x]


def keep_list(*people: tuple[str, Faceprint]) -> KeepList:
    kl = KeepList()
    for label, v in people:
        kl.add(label, [near(v, 0.3, 10_000 + s) for s in range(10)])  # own noise seeds
    return kl


def policy(kl: KeepList, **cfg) -> tuple[KeepListPolicy, ScriptedModel]:
    model = ScriptedModel()
    defaults = {"recheck_every": 1, "max_checks_per_frame": 10}
    return KeepListPolicy(model, kl, KeepConfig(**{**defaults, **cfg})), model


def keep_of(p: KeepListPolicy, tracks: list[Track]) -> dict[int, bool]:
    return {d.track_id: d.keep_real for d in p.decide(FRAME, tracks)}


# --- KeepList ---


def test_matching_finds_the_right_person_and_rejects_strangers() -> None:
    kl = keep_list(("alice", ALICE), ("bob", BOB))
    assert kl.best_match(near(ALICE, 0.5, 99))[0] == "alice"
    assert kl.best_match(near(BOB, 0.5, 98))[0] == "bob"
    label, score = kl.best_match(CAROL)
    assert label is None and score < kl.threshold


def test_threshold_is_configurable() -> None:
    kl = keep_list(("alice", ALICE))
    probe = near(ALICE, 1.2, 7)
    score = kl.best_match(probe)[1]
    kl.threshold = score - 0.01
    assert kl.best_match(probe)[0] == "alice"
    kl.threshold = score + 0.01
    assert kl.best_match(probe)[0] is None


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    kl = keep_list(("creator", ALICE), ("friend 1", BOB))
    path = tmp_path / "keep.npz"
    kl.save(path)
    loaded = KeepList.load(path)
    assert loaded.labels == ["creator", "friend 1"]
    for a, b in zip(kl.people, loaded.people, strict=True):
        assert np.allclose(np.stack(a.faceprints), np.stack(b.faceprints))
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"labels", "counts", "faceprints"}  # no images


# --- reveal / hide rules ---


def test_new_tracks_always_start_hidden() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    model.faces[100] = near(ALICE, 0.2, 1)
    assert keep_of(p, [track(1)]) == {1: False}


def test_reveal_needs_three_of_last_five_matches() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    seq = [ALICE, CAROL, ALICE, CAROL, ALICE]  # match, miss, match, miss, match
    results = []
    for i, v in enumerate(seq):
        model.faces[100] = near(v, 0.2, i)
        results.append(keep_of(p, [track(1)])[1])
    assert results == [False, False, False, False, True]


def test_two_matches_out_of_five_is_not_enough() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    for i, v in enumerate([ALICE, CAROL, CAROL, ALICE, CAROL, CAROL]):
        model.faces[100] = near(v, 0.2, i)
        assert keep_of(p, [track(1)])[1] is False


def test_hidden_after_two_failed_checks_in_a_row() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    model.faces[100] = near(ALICE, 0.2, 0)
    for _ in range(3):
        keep_of(p, [track(1)])
    assert keep_of(p, [track(1)])[1] is True
    model.faces[100] = CAROL
    assert keep_of(p, [track(1)])[1] is True  # one failure: still kept
    model.faces[100] = near(ALICE, 0.2, 5)
    assert keep_of(p, [track(1)])[1] is True  # failure streak reset
    model.faces[100] = CAROL
    keep_of(p, [track(1)])
    assert keep_of(p, [track(1)])[1] is False  # two in a row: hidden


def test_poor_quality_views_keep_current_status() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    model.faces[100] = near(ALICE, 0.2, 0)
    for _ in range(3):
        keep_of(p, [track(1)])
    calls = model.calls
    model.faces[100] = CAROL  # would fail, if it were checked
    for bad in (track(1, landmarks=side_on), track(1, size=40), track(1, landmarks=None)):
        assert keep_of(p, [bad])[1] is True
    assert model.calls == calls  # no faceprint computed for bad views

    # And a hidden track isn't revealed by bad views either.
    p2, model2 = policy(keep_list(("alice", ALICE)))
    model2.faces[100] = ALICE
    for _ in range(5):
        assert keep_of(p2, [track(1, landmarks=side_on)])[1] is False
    assert model2.calls == 0


def test_quality_gate_rejects_side_on_small_and_tilted() -> None:
    gate = QualityGate()
    box = BoundingBox(0, 0, 120, 120)
    assert gate.passes(box, frontal(box))
    assert not gate.passes(box, side_on(box))
    assert not gate.passes(BoundingBox(0, 0, 40, 40), frontal(BoundingBox(0, 0, 40, 40)))
    tilted = list(frontal(box))
    tilted[1] = (tilted[1][0], tilted[1][1] + 40)  # one eye much lower: big roll
    assert not gate.passes(box, tilted)


def test_two_tracks_matching_the_same_person_only_one_stays_real() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    model.faces[100] = near(ALICE, 1.2, 1)  # weaker match
    model.faces[600] = near(ALICE, 0.1, 2)  # stronger match
    tracks = [track(1, x=100), track(2, x=600)]
    for _ in range(3):
        result = keep_of(p, tracks)
    assert result == {1: False, 2: True}


def test_kept_track_coasting_while_lost_stays_real() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    model.faces[100] = near(ALICE, 0.2, 0)
    for _ in range(3):
        keep_of(p, [track(1)])
    lost = track(1, state=TrackState.LOST, seen=False)
    assert keep_of(p, [lost])[1] is True


def test_recovered_track_is_hidden_until_reverified() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    model.faces[100] = near(ALICE, 0.2, 0)
    for _ in range(3):
        keep_of(p, [track(1)])
    keep_of(p, [track(1, state=TrackState.LOST, seen=False)])
    # Found again: hidden until it matches twice in a row.
    assert keep_of(p, [track(1)])[1] is False
    assert keep_of(p, [track(1)])[1] is True


def test_recovered_track_taken_over_by_a_stranger_stays_hidden() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    model.faces[100] = near(ALICE, 0.2, 0)
    for _ in range(3):
        keep_of(p, [track(1)])
    keep_of(p, [track(1, state=TrackState.LOST, seen=False)])
    model.faces[100] = CAROL  # the tracker gave alice's ID to a stranger
    for _ in range(10):
        assert keep_of(p, [track(1)])[1] is False


def test_detection_near_a_lost_kept_track_is_covered() -> None:
    lost_kept = track(1, state=TrackState.LOST, seen=False)
    det = Detection(BoundingBox(105, 205, 110, 110), 0.9)
    from npcify.types import Decision

    regions = regions_to_cover([lost_kept], [det], [Decision(1, keep_real=True)])
    assert det.box in [r.box for r in regions]


def test_crowded_kept_track_is_hidden_after_one_failure() -> None:
    p, model = policy(keep_list(("alice", ALICE)))
    model.faces[100] = near(ALICE, 0.2, 0)
    model.faces[200] = CAROL  # someone right next to alice
    tracks = [track(1, x=100), track(2, x=200)]
    for _ in range(3):
        keep_of(p, tracks)
    assert keep_of(p, tracks)[1] is True
    model.faces[100] = CAROL  # IDs swapped as they crossed
    assert keep_of(p, tracks)[1] is False


def test_checks_are_limited_per_frame_and_shared_round_robin() -> None:
    p, model = policy(keep_list(("alice", ALICE)), max_checks_per_frame=2)
    tracks = [track(i, x=100 + 150 * i) for i in range(5)]
    for t in tracks:
        model.faces[t.box.x] = CAROL
    checked: set[int] = set()
    for _ in range(3):
        before = model.calls
        p.decide(FRAME, tracks)
        assert model.calls - before <= 2
        checked |= {tid for tid, st in p.statuses.items() if st.score is not None}
    assert checked == {t.track_id for t in tracks}


def test_no_faceprints_computed_with_empty_keep_list() -> None:
    p, model = policy(KeepList())
    for _ in range(5):
        p.decide(FRAME, [track(1)])
    assert model.calls == 0


def test_bystander_faceprints_are_never_stored(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    kl = keep_list(("alice", ALICE))
    stored_before = [np.stack(p.faceprints).copy() for p in kl.people]
    p, model = policy(kl)
    strangers = [track(i, x=100 + 200 * i) for i in range(4)]
    for t in strangers:
        model.faces[t.box.x] = identity(100 + t.track_id)
    for _ in range(20):
        p.decide(FRAME, strangers)
    assert model.calls > 0
    # Keep list unchanged; no arrays anywhere in the per-track state.
    assert kl.labels == ["alice"]
    for a, b in zip(stored_before, kl.people, strict=True):
        assert np.array_equal(a, np.stack(b.faceprints))
    for st in p.statuses.values():
        assert not any(isinstance(v, np.ndarray) for v in vars(st).values())
        assert all(not isinstance(h, np.ndarray) for h in st.history)
    assert p.enrolment is None
    assert list(tmp_path.iterdir()) == []  # nothing written to disk


# --- enrolment ---


def test_enrolment_collects_then_reveals() -> None:
    kl = KeepList()
    p, model = policy(kl)
    label = p.enrol(1)
    assert label == "person 1"
    frames = 0
    while p.enrolment is not None:
        model.faces[100] = near(ALICE, 0.3, frames)
        assert keep_of(p, [track(1)])[1] is False  # covered while enrolling
        frames += 1
        assert frames < 40
    assert kl.labels == ["person 1"] and len(kl.people[0].faceprints) == 10
    results = []
    for i in range(3):
        model.faces[100] = near(ALICE, 0.3, 100 + i)
        results.append(keep_of(p, [track(1)])[1])
    # The check that completes enrolment already counts as the first match.
    assert results == [False, True, True]


def test_enrolment_cancelled_when_track_leaves() -> None:
    p, model = policy(KeepList())
    p.enrol(1)
    model.faces[100] = ALICE
    keep_of(p, [track(1)])
    keep_of(p, [])
    assert p.enrolment is None
    assert len(p.keep_list) == 0


def test_enrolment_ignores_a_different_face_mid_way() -> None:
    e = Enrolment(1, "x", target=5, interval=1)
    for i in range(3):
        e.offer(near(ALICE, 0.3, i))
    e.offer(BOB)  # track switched to someone else
    assert len(e.faceprints) == 3


def test_removing_and_clearing_hide_people_again() -> None:
    p, model = policy(keep_list(("alice", ALICE), ("bob", BOB)))
    model.faces[100] = near(ALICE, 0.2, 0)
    model.faces[600] = near(BOB, 0.2, 0)
    tracks = [track(1, x=100), track(2, x=600)]
    for _ in range(3):
        keep_of(p, tracks)
    assert keep_of(p, tracks) == {1: True, 2: True}
    assert p.forget_track_person(1) == "alice"
    assert keep_of(p, tracks) == {1: False, 2: True}
    p.clear()
    assert keep_of(p, tracks) == {1: False, 2: False}


@pytest.mark.parametrize("bad", [[], None])
def test_cannot_add_person_without_faceprints(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        KeepList().add("x", bad)


def test_enrolling_with_empty_keep_list_only_checks_the_enrolling_face() -> None:
    p, model = policy(KeepList())
    tracks = [track(1, x=100), track(2, x=600)]
    model.faces[100], model.faces[600] = ALICE, BOB
    p.enrol(1)
    p.decide(FRAME, tracks)
    assert model.calls == 1


def test_enrolment_hint_explains_a_face_that_is_too_small() -> None:
    p, model = policy(KeepList())
    p.enrol(1)
    p.decide(FRAME, [track(1, size=40)])
    assert p.enrolment_hint == "too small: move closer"
    assert model.calls == 0


def test_clear_strangers_are_checked_less_often() -> None:
    p, model = policy(keep_list(("alice", ALICE)), stranger_recheck_every=5)
    model.faces[100] = CAROL
    for _ in range(5):
        p.decide(FRAME, [track(1)])
    assert model.calls == 5  # checked every frame at first
    for _ in range(10):
        p.decide(FRAME, [track(1)])
    assert model.calls == 7  # then every 5th frame
