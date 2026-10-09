"""Give each person a stable ID across frames.

A small ByteTrack-style tracker:

- Each track has a Kalman filter (constant velocity on box centre and size)
  that predicts where the face will be next and smooths out detector jitter.
- Detections are matched to predicted boxes by overlap (IoU) with optimal
  assignment, in passes: high-confidence detections first; then
  low-confidence detections, which can only rescue existing tracks; then
  remaining high-confidence detections against tentative tracks.
- Unmatched high-confidence detections start new tentative tracks, which are
  confirmed after `min_hits` matches in a row.
- A confirmed track with no match becomes lost: its position keeps being
  predicted and its box grows (we're less sure where it is) for up to
  `max_age` frames before it is removed. Its speed decays while lost, because
  someone who is covered or turned away usually stays roughly where they were.
- A lost track that no longer overlaps its face when the face reappears (e.g.
  the person shifted while hidden) can still be recovered by a nearby
  confident detection of similar size (see `recover_distance`).

Safety: tentative and lost tracks are still real faces. Callers must cover
every track this returns, plus every raw detection; confirmation is never a
reason to show a face.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from npcify.types import BoundingBox, Detection, Point, Track, TrackState

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrackerConfig:
    """Tracker settings. Defaults assume ~30 FPS video.

    high_threshold: detections at or above this are matched first and can start
        new tracks.
    low_threshold: detections between this and high_threshold can only keep an
        existing track alive. The detector must be run at this threshold (or
        lower) for these to arrive at all.
    match_iou: minimum overlap between a track's predicted box and a detection
        to count as the same face. Lower tolerates faster movement but risks
        swapping IDs between nearby faces.
    low_match_iou: stricter overlap needed for a low-confidence rescue match.
    min_hits: consecutive matches needed to confirm a new track.
    max_age: frames a lost track is kept (and its position predicted) before
        removal. 60 frames is about 2 seconds at 30 FPS: long enough to cover
        your face with your hands or turn round on a chair.
    lost_velocity_decay: a lost track's speed is multiplied by this each missed
        frame, so its prediction settles near where the face was last seen
        instead of drifting away. Not applied to faces last seen at the frame
        edge, which are most likely leaving.
    recover_distance: a lost track can reclaim a confident detection whose centre
        is within this many face-widths of its predicted centre, if the sizes
        are within `recover_size_ratio` of each other. Larger survives bigger
        movements while hidden, but risks a different person who appears nearby
        inheriting the ID.
    recover_size_ratio: maximum size ratio (bigger / smaller) for that recovery.
    lost_growth: how much a lost track's box grows per missed frame (0.05 = 5%).
    max_lost_growth: cap on that growth, as a size multiplier (1.5 = 150%).
    position_noise: expected detector jitter as a fraction of face size. Higher
        smooths more but reacts more slowly.
    velocity_noise: how quickly velocity may change, as a fraction of face size
        per frame. Higher follows sudden starts and turns faster but smooths less.
    """

    high_threshold: float = 0.6
    low_threshold: float = 0.3
    match_iou: float = 0.2
    low_match_iou: float = 0.5
    min_hits: int = 3
    max_age: int = 60
    lost_velocity_decay: float = 0.9
    recover_distance: float = 1.5
    recover_size_ratio: float = 2.5
    lost_growth: float = 0.05
    max_lost_growth: float = 1.5
    position_noise: float = 1 / 20
    velocity_noise: float = 1 / 80


@dataclass
class TrackerStats:
    """Running counts, for evaluating how well tracking holds together."""

    ids_created: int = 0
    ids_confirmed: int = 0
    lost_events: int = 0
    recovered: int = 0


class _KalmanBox:
    """Constant-velocity Kalman filter on [cx, cy, w, h] and their speeds.

    Noise is scaled by the face size, so a big close-up face is allowed to
    move more pixels per frame than a small far-away one.
    """

    _F = np.eye(8) + np.eye(8, k=4)  # x' = x + v
    _H = np.eye(4, 8)

    def __init__(self, z: FloatArray, position_noise: float, velocity_noise: float) -> None:
        self._pos = position_noise
        self._vel = velocity_noise
        self.x = np.concatenate([z, np.zeros(4)])
        s = max(z[2], z[3])
        self.P = np.diag(np.square([2 * self._pos * s] * 4 + [10 * self._vel * s] * 4))

    def predict(self) -> None:
        s = max(self.x[2], self.x[3])
        Q = np.diag(np.square([self._pos * s] * 4 + [self._vel * s] * 4))
        self.x = self._F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = self._F @ self.P @ self._F.T + Q

    def update(self, z: FloatArray) -> None:
        s = max(self.x[2], self.x[3])
        R = np.diag(np.square([self._pos * s] * 4))
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self._H @ self.x)
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = (np.eye(8) - K @ self._H) @ self.P


@dataclass
class _Record:
    track_id: int
    kf: _KalmanBox
    state: TrackState
    confidence: float
    hits: int = 1
    frames_since_seen: int = 0
    age: int = 1
    landmarks: tuple[Point, ...] | None = field(default=None)
    # Where the face was last actually seen, and its recent sizes (max of w, h).
    # Used to recover lost tracks: a face that turns away shrinks before it
    # disappears, so the recent maximum is a better guide to its real size.
    last_centre: tuple[float, float] = (0.0, 0.0)
    recent_sizes: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    at_edge: bool = False  # last seen touching the frame edge (probably leaving)

    def saw(self, z: FloatArray, frame_size: tuple[int, int] | None) -> None:
        self.last_centre = (float(z[0]), float(z[1]))
        self.recent_sizes.append(float(max(z[2], z[3])))
        if frame_size is not None:
            margin = 2
            self.at_edge = (
                z[0] - z[2] / 2 <= margin
                or z[1] - z[3] / 2 <= margin
                or z[0] + z[2] / 2 >= frame_size[0] - margin
                or z[1] + z[3] / 2 >= frame_size[1] - margin
            )

    def ref_size(self) -> float:
        return max(self.recent_sizes, default=float(max(self.kf.x[2], self.kf.x[3])))

    def xyxy(self) -> FloatArray:
        """The (ungrown) predicted box as [x1, y1, x2, y2]."""
        cx, cy, w, h = self.kf.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _box_to_z(box: BoundingBox) -> FloatArray:
    return np.array(
        [box.x + box.width / 2, box.y + box.height / 2, box.width, box.height], dtype=float
    )


def _iou_matrix(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise IoU between boxes a (N x 4) and b (M x 4), both as x1, y1, x2, y2."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _match(
    tracks: list[_Record], dets: list[Detection], min_iou: float
) -> tuple[list[tuple[_Record, Detection]], list[_Record], list[Detection]]:
    """Optimal one-to-one matching by IoU. Returns matches, unmatched tracks, unmatched dets."""
    if not tracks or not dets:
        return [], list(tracks), list(dets)
    track_boxes = np.array([t.xyxy() for t in tracks])
    det_boxes = np.array(
        [[d.box.x, d.box.y, d.box.x + d.box.width, d.box.y + d.box.height] for d in dets],
        dtype=float,
    )
    iou = _iou_matrix(track_boxes, det_boxes)
    rows, cols = linear_sum_assignment(1.0 - iou)
    matches = [
        (tracks[r], dets[c]) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou
    ]
    matched_t = {id(t) for t, _ in matches}
    matched_d = {id(d) for _, d in matches}
    return (
        matches,
        [t for t in tracks if id(t) not in matched_t],
        [d for d in dets if id(d) not in matched_d],
    )


def _match_lost_by_distance(
    tracks: list[_Record], dets: list[Detection], max_dist: float, max_ratio: float
) -> tuple[list[tuple[_Record, Detection]], list[_Record], list[Detection]]:
    """Match lost tracks to detections by centre distance (in face-widths) and size.

    Distance is measured to whichever is closer: the predicted centre, or where
    the face was last seen (people who turn away usually come back there).
    """
    if not tracks or not dets:
        return [], list(tracks), list(dets)
    big = 1e6
    cost = np.full((len(tracks), len(dets)), big)
    for i, t in enumerate(tracks):
        px, py = t.kf.x[:2]
        lx, ly = t.last_centre
        size = t.ref_size()
        for j, d in enumerate(dets):
            z = _box_to_z(d.box)
            det_size = max(z[2], z[3])
            dist = min(math.hypot(z[0] - px, z[1] - py), math.hypot(z[0] - lx, z[1] - ly)) / size
            ratio = max(size, det_size) / max(1.0, min(size, det_size))
            if dist <= max_dist and ratio <= max_ratio:
                cost[i, j] = dist
    rows, cols = linear_sum_assignment(cost)
    matches = [(tracks[r], dets[c]) for r, c in zip(rows, cols, strict=True) if cost[r, c] < big]
    matched_t = {id(t) for t, _ in matches}
    matched_d = {id(d) for _, d in matches}
    return (
        matches,
        [t for t in tracks if id(t) not in matched_t],
        [d for d in dets if id(d) not in matched_d],
    )


class Tracker:
    """Keeps track IDs stable from frame to frame."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.stats = TrackerStats()
        self._tracks: list[_Record] = []
        self._next_id = 1

    def update(
        self,
        detections: list[Detection] | None,
        frame_size: tuple[int, int] | None = None,
    ) -> list[Track]:
        """Advance one frame and return every live (non-removed) track.

        Pass `detections=None` on frames where detection was skipped: tracks are
        only predicted forward, and nobody is marked lost for not being seen.
        `frame_size` is (width, height); when given, boxes are clipped to the
        frame and lost tracks predicted entirely off-screen are removed.
        """
        cfg = self.config
        for t in self._tracks:
            if t.state is TrackState.LOST:
                t.kf.x[6:8] = 0.0  # don't let the size keep drifting while unseen
                if not t.at_edge:
                    t.kf.x[4:6] *= cfg.lost_velocity_decay
            t.kf.predict()
            t.age += 1
            t.frames_since_seen += 1
            t.landmarks = None

        if detections is not None:
            self._associate(detections, frame_size)

        for t in self._tracks:
            unseen = t.frames_since_seen > 0
            if t.state is TrackState.LOST and t.frames_since_seen > cfg.max_age:
                t.state = TrackState.REMOVED
            elif unseen and frame_size is not None and self._box(t, frame_size) is None:
                t.state = TrackState.REMOVED  # predicted to have left the frame
        self._tracks = [t for t in self._tracks if t.state is not TrackState.REMOVED]

        tracks = []
        for t in self._tracks:
            box = self._box(t, frame_size)
            if box is not None:
                tracks.append(self._snapshot(t, box))
        return tracks

    def _associate(self, detections: list[Detection], frame_size: tuple[int, int] | None) -> None:
        cfg = self.config
        high = [d for d in detections if d.confidence >= cfg.high_threshold]
        low = [d for d in detections if cfg.low_threshold <= d.confidence < cfg.high_threshold]
        active = [t for t in self._tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self._tracks if t.state is TrackState.TENTATIVE]

        # 1. Established tracks vs high-confidence detections.
        matches, unmatched, high_left = _match(active, high, cfg.match_iou)
        # 2. Low-confidence detections may only rescue established tracks.
        rescued, unmatched, _ = _match(unmatched, low, cfg.low_match_iou)
        # 3. Lost tracks whose prediction no longer overlaps the face: nearby,
        #    similar-sized confident detections. Runs before new tracks can start.
        lost = [t for t in unmatched if t.state is TrackState.LOST]
        recovered, _, high_left = _match_lost_by_distance(
            lost, high_left, cfg.recover_distance, cfg.recover_size_ratio
        )
        unmatched = [t for t in unmatched if all(t is not r for r, _ in recovered)]
        # 4. Tentative tracks vs remaining high-confidence detections.
        tent_matches, tent_unmatched, high_left = _match(tentative, high_left, cfg.match_iou)

        for t, d in matches + rescued + recovered + tent_matches:
            self._hit(t, d, frame_size)
        for t in tent_unmatched:
            t.state = TrackState.REMOVED  # a tentative track must be seen every frame
        for t in unmatched:
            if t.state is TrackState.CONFIRMED:
                t.state = TrackState.LOST
                self.stats.lost_events += 1
        for d in high_left:
            self._start(d, frame_size)

    def _hit(self, t: _Record, d: Detection, frame_size: tuple[int, int] | None) -> None:
        t.saw(_box_to_z(d.box), frame_size)
        if t.state is TrackState.LOST:
            # Restart the motion estimate: the face may have moved while hidden, and
            # treating that jump as one frame of movement would make the box overshoot.
            t.kf = _KalmanBox(
                _box_to_z(d.box), self.config.position_noise, self.config.velocity_noise
            )
        else:
            t.kf.update(_box_to_z(d.box))
        t.hits += 1
        t.frames_since_seen = 0
        t.confidence = d.confidence
        t.landmarks = d.landmarks
        if t.state is TrackState.LOST:
            t.state = TrackState.CONFIRMED
            self.stats.recovered += 1
        elif t.state is TrackState.TENTATIVE and t.hits >= self.config.min_hits:
            t.state = TrackState.CONFIRMED
            self.stats.ids_confirmed += 1

    def _start(self, d: Detection, frame_size: tuple[int, int] | None) -> None:
        cfg = self.config
        kf = _KalmanBox(_box_to_z(d.box), cfg.position_noise, cfg.velocity_noise)
        t = _Record(self._next_id, kf, TrackState.TENTATIVE, d.confidence, landmarks=d.landmarks)
        t.saw(_box_to_z(d.box), frame_size)
        self._next_id += 1
        self.stats.ids_created += 1
        if cfg.min_hits <= 1:
            t.state = TrackState.CONFIRMED
            self.stats.ids_confirmed += 1
        self._tracks.append(t)

    def _box(self, t: _Record, frame_size: tuple[int, int] | None) -> BoundingBox | None:
        """Output box: grown while lost, rounded outwards, clipped. None if empty."""
        cx, cy, w, h = t.kf.x[:4]
        if t.state is TrackState.LOST:
            cfg = self.config
            scale = min(1.0 + cfg.lost_growth * t.frames_since_seen, cfg.max_lost_growth)
            w, h = w * scale, h * scale
        x1, y1 = math.floor(cx - w / 2), math.floor(cy - h / 2)
        x2, y2 = math.ceil(cx + w / 2), math.ceil(cy + h / 2)
        if frame_size is not None:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_size[0], x2), min(frame_size[1], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _snapshot(t: _Record, box: BoundingBox) -> Track:
        return Track(
            track_id=t.track_id,
            state=t.state,
            box=box,
            velocity=(float(t.kf.x[4]), float(t.kf.x[5])),
            confidence=t.confidence,
            hits=t.hits,
            frames_since_seen=t.frames_since_seen,
            age=t.age,
            landmarks=t.landmarks,
        )
