"""Decide who stays real and who gets hidden, and list every region to cover.

A CoveragePolicy says, per track, whether that person may stay real. For now
the only policy is HideEveryone. Step 6 adds a keep-list policy that marks
some tracks keep_real=True; nothing else in the pipeline needs to change.

`regions_to_cover` then turns tracks, raw detections and decisions into the
list of face regions to hide. It is deliberately one-sided: a track is shown
only with an explicit keep_real=True decision, and a raw detection is shown
only if it clearly belongs to a kept track. Everything else is covered.
"""

from __future__ import annotations

from typing import Protocol

from npcify.types import BoundingBox, CoverRegion, Decision, Detection, Frame, Track

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
      (already covered), or it clearly belongs to a kept track.
    """
    keep_ids = {d.track_id for d in decisions if d.keep_real}
    hidden = [t for t in tracks if t.track_id not in keep_ids]
    kept = [t for t in tracks if t.track_id in keep_ids]

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
