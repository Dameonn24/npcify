"""Decide who stays real and who gets hidden, and list every region to cover.

A CoveragePolicy says, per track, whether that person may stay real.
HideEveryone hides everyone; KeepListPolicy reveals only people on the keep
list, once their faceprints have matched enough times.

`regions_to_cover` then turns tracks, raw detections and decisions into the
list of face regions to hide. It is deliberately one-sided: a track is shown
only with an explicit keep_real=True decision, and a raw detection is shown
only if it clearly belongs to a kept track. Everything else is covered.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Protocol

from npcify.recognise import Enrolment, FaceprintModel, KeepList, QualityGate
from npcify.timing import StageTimer
from npcify.types import BoundingBox, CoverRegion, Decision, Detection, Frame, Track, TrackState

# A raw detection counts as belonging to a kept track only with at least this
# overlap, and only if no hidden track overlaps it more.
KEEP_MATCH_IOU = 0.5


class CoveragePolicy(Protocol):
    """Decides, per track, whether that person may stay real."""

    def decide(self, frame: Frame, tracks: list[Track]) -> list[Decision]: ...


class HideEveryone:
    """Hide every face. The default, and the fallback whenever unsure."""

    def decide(self, frame: Frame, tracks: list[Track]) -> list[Decision]:
        return [Decision(t.track_id, keep_real=False) for t in tracks]


def iou(a: BoundingBox, b: BoundingBox) -> float:
    ix = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    iy = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and outer.x + outer.width >= inner.x + inner.width
        and outer.y + outer.height >= inner.y + inner.height
    )


def regions_to_cover(
    tracks: list[Track],
    detections: list[Detection],
    decisions: list[Decision],
) -> list[CoverRegion]:
    """Every region that must be hidden this frame.

    - Every track (tentative, confirmed or lost) unless a decision explicitly
      says keep_real=True for it. A track with no decision is hidden.
    - Every raw detection, unless it lies entirely inside a hidden track's box
      (already covered), or it clearly belongs to a kept track that the
      tracker matched this frame. A kept track that is lost (coasting) never
      exempts a detection: a face appearing near it starts covered.
    """
    keep_ids = {d.track_id for d in decisions if d.keep_real}
    hidden = [t for t in tracks if t.track_id not in keep_ids]
    kept = [t for t in tracks if t.track_id in keep_ids and t.frames_since_seen == 0]

    regions = [CoverRegion(t.box, t.track_id, f"track:{t.state.value}") for t in hidden]
    for det in detections:
        if any(contains(t.box, det.box) for t in hidden):
            continue
        best_kept = max((iou(t.box, det.box) for t in kept), default=0.0)
        best_hidden = max((iou(t.box, det.box) for t in hidden), default=0.0)
        if best_kept >= KEEP_MATCH_IOU and best_kept > best_hidden:
            continue
        regions.append(CoverRegion(det.box, None, "detection"))
    return regions


@dataclass(frozen=True)
class KeepConfig:
    """How cautiously KeepListPolicy reveals and hides people.

    Slow to reveal, quick to hide:
        reveal_matches / reveal_window: a hidden track is revealed after matching
            the same kept person in this many of its last `reveal_window` checks.
        hide_after_fails: a revealed track is hidden after this many failed checks
            in a row (1 if another face is close by, where ID swaps happen).
        reverify_matches: a kept track that was lost and then found again starts
            hidden, and needs this many matches in a row to be revealed again
            (the tracker may have handed its ID to someone else).
    Speed:
        recheck_every: frames between checks of a revealed track.
        stranger_recheck_every: frames between checks of a hidden track that has
            failed `reveal_window` checks in a row (clearly not a kept person).
            New and recently-matching tracks are still checked every frame.
        max_checks_per_frame: faceprints computed per frame at most (~4-5 ms each).
    crowd_distance: another track whose centre is within this many face-widths
        counts as close by.
    """

    reveal_matches: int = 3
    reveal_window: int = 5
    hide_after_fails: int = 2
    reverify_matches: int = 2
    recheck_every: int = 15
    stranger_recheck_every: int = 5
    max_checks_per_frame: int = 2
    crowd_distance: float = 1.5


@dataclass
class TrackStatus:
    """What the policy knows about one track. Holds no faceprints."""

    keep: bool = False
    kept_as: str | None = None  # label it is revealed as
    label: str | None = None  # kept person matched at the latest check, if any
    score: float | None = None  # similarity at the latest check
    history: deque[str | None] = field(default_factory=deque)
    fails_in_row: int = 0  # failed checks in a row while revealed
    misses_in_row: int = 0  # failed checks in a row while hidden
    last_check: int = -(10**9)
    reverify: bool = False
    prev_state: TrackState | None = None


class KeepListPolicy:
    """Reveals people on the keep list; hides everyone else.

    Every track starts hidden. Each frame, up to `max_checks_per_frame` tracks
    with a good-quality view get a faceprint, which is compared against the
    keep list and then discarded (only the label and score are kept). Tracks
    without a good view this frame keep their current status.
    """

    def __init__(
        self,
        model: FaceprintModel,
        keep_list: KeepList | None = None,
        config: KeepConfig | None = None,
        gate: QualityGate | None = None,
        timer: StageTimer | None = None,
    ) -> None:
        self.model = model
        self.keep_list = keep_list if keep_list is not None else KeepList()
        self.config = config or KeepConfig()
        self.gate = gate or QualityGate()
        self.timer = timer
        self.enrolment: Enrolment | None = None
        self.enrolment_hint: str | None = None  # why the enrolling face can't be used now
        self.statuses: dict[int, TrackStatus] = {}
        self._frame = 0

    # --- keep-list management (e.g. from mouse clicks) ---

    def enrol(self, track_id: int, label: str | None = None) -> str:
        """Start collecting faceprints from a track to add them to the keep list."""
        label = label or self.keep_list.next_label()
        self.enrolment = Enrolment(track_id, label)
        return label

    def forget_track_person(self, track_id: int) -> str | None:
        """Remove the kept person this track is revealed as (or enrolling). Returns the label."""
        if self.enrolment is not None and self.enrolment.track_id == track_id:
            label = self.enrolment.label
            self.enrolment = None
            return label
        st = self.statuses.get(track_id)
        label = st.kept_as if st else None
        if label is not None:
            self.forget(label)
        return label

    def forget(self, label: str) -> None:
        self.keep_list.remove(label)
        for st in self.statuses.values():
            if st.kept_as == label:
                self._hide(st)

    def clear(self) -> None:
        self.keep_list.clear()
        self.enrolment = None
        for st in self.statuses.values():
            self._hide(st)

    def status_text(self, track_id: int) -> str:
        """Short description for debug drawing, e.g. "keep: creator 0.62"."""
        st = self.statuses.get(track_id)
        if self.enrolment is not None and self.enrolment.track_id == track_id:
            e = self.enrolment
            return f"enrolling {e.label} {len(e.faceprints)}/{e.target}"
        if st is None:
            return "hide"
        score = f" {st.score:.2f}" if st.score is not None else ""
        if st.keep:
            return f"keep: {st.kept_as}{score}"
        return f"hide{score}"

    # --- per-frame decision ---

    def decide(self, frame: Frame, tracks: list[Track]) -> list[Decision]:
        self._frame += 1
        live = {t.track_id for t in tracks}
        self.statuses = {k: v for k, v in self.statuses.items() if k in live}
        if self.enrolment is not None and self.enrolment.track_id not in live:
            self.enrolment = None  # the person left before enrolment finished
        self.enrolment_hint = None
        if self.enrolment is not None:
            t = next(t for t in tracks if t.track_id == self.enrolment.track_id)
            self.enrolment_hint = self.gate.problem(t.box, t.landmarks)

        crowded = {t.track_id for t in tracks if self._crowded(t, tracks)}
        for t in tracks:
            st = self.statuses.setdefault(
                t.track_id, TrackStatus(history=deque(maxlen=self.config.reveal_window))
            )
            # Found again after being lost: it may be a different face now.
            if st.prev_state is TrackState.LOST and t.state is not TrackState.LOST and st.keep:
                self._hide(st)
                st.reverify = True
            st.prev_state = t.state

        start = time.perf_counter()
        if self.enrolment is not None or len(self.keep_list) > 0:
            for t in self._pick_checks(tracks, crowded):
                self._check(frame, t, self.statuses[t.track_id], t.track_id in crowded)
        if self.timer is not None:
            self.timer.record("recognise", (time.perf_counter() - start) * 1000.0)

        return self._decisions(tracks)

    def _pick_checks(self, tracks: list[Track], crowded: set[int]) -> list[Track]:
        cfg = self.config
        due = []
        for t in tracks:
            st = self.statuses[t.track_id]
            if t.frames_since_seen != 0 or not self.gate.passes(t.box, t.landmarks):
                continue  # no good view this frame: keep current status
            enrolling = self.enrolment is not None and self.enrolment.track_id == t.track_id
            if not enrolling and len(self.keep_list) == 0:
                continue  # nobody to compare against yet
            since = self._frame - st.last_check
            if st.keep and not enrolling and t.track_id not in crowded:
                if since < cfg.recheck_every:
                    continue
            if not st.keep and not enrolling and st.misses_in_row >= cfg.reveal_window:
                if since < cfg.stranger_recheck_every:
                    continue
            if enrolling:
                priority = 0
            elif st.keep and t.track_id in crowded:
                priority = 1
            elif st.keep:
                priority = 2
            else:
                priority = 3
            due.append((priority, st.last_check, t.track_id, t))
        due.sort(key=lambda x: x[:3])
        return [t for *_, t in due[: cfg.max_checks_per_frame]]

    def _check(self, frame: Frame, t: Track, st: TrackStatus, crowded: bool) -> None:
        assert t.landmarks is not None
        faceprint = self.model.faceprint(frame, t.box, t.landmarks)
        if self.enrolment is not None and self.enrolment.track_id == t.track_id:
            self.enrolment.offer(faceprint)
            if self.enrolment.done:
                self.keep_list.add(self.enrolment.label, self.enrolment.faceprints)
                self.enrolment = None
        if len(self.keep_list):
            label, score = self.keep_list.best_match(faceprint)
        else:
            label, score = None, None
        del faceprint  # bystander faceprints are never kept

        st.last_check = self._frame
        st.label, st.score = label, score
        st.history.append(label)
        st.misses_in_row = 0 if label is not None else st.misses_in_row + 1
        cfg = self.config

        if st.keep:
            if label == st.kept_as:
                st.fails_in_row = 0
                return
            st.fails_in_row += 1
            if st.fails_in_row >= (1 if crowded else cfg.hide_after_fails):
                self._hide(st)
            return

        if label is None:
            return
        if st.reverify:
            recent = list(st.history)[-cfg.reverify_matches :]
            if len(recent) == cfg.reverify_matches and all(x == label for x in recent):
                self._reveal(st, label)
            return
        counts = Counter(x for x in st.history if x is not None)
        best, n = counts.most_common(1)[0]
        if n >= cfg.reveal_matches:
            self._reveal(st, best)

    def _decisions(self, tracks: list[Track]) -> list[Decision]:
        # If several tracks are revealed as the same person, only the best match stays real.
        best: dict[str, tuple[float, int]] = {}
        for t in tracks:
            st = self.statuses[t.track_id]
            if st.keep and st.kept_as is not None:
                score = st.score if st.score is not None else -1.0
                if st.kept_as not in best or score > best[st.kept_as][0]:
                    best[st.kept_as] = (score, t.track_id)
        winners = {tid for _, tid in best.values()}
        return [Decision(t.track_id, keep_real=t.track_id in winners) for t in tracks]

    def _crowded(self, t: Track, tracks: list[Track]) -> bool:
        cx, cy = t.box.x + t.box.width / 2, t.box.y + t.box.height / 2
        limit = self.config.crowd_distance * max(t.box.width, t.box.height)
        for o in tracks:
            if o.track_id == t.track_id:
                continue
            ox, oy = o.box.x + o.box.width / 2, o.box.y + o.box.height / 2
            if (ox - cx) ** 2 + (oy - cy) ** 2 < limit**2:
                return True
        return False

    @staticmethod
    def _reveal(st: TrackStatus, label: str) -> None:
        st.keep, st.kept_as = True, label
        st.fails_in_row = 0
        st.reverify = False

    @staticmethod
    def _hide(st: TrackStatus) -> None:
        st.keep, st.kept_as = False, None
        st.fails_in_row = 0
        st.history.clear()
