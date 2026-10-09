"""Faceprints and the keep list: recognising the people who stay real.

A faceprint is a 128-number summary of a face (from OpenCV's SFace model),
built so that two photos of the same person give similar faceprints. We use
them only to answer one question: "is this one of the people on the keep
list?"

Privacy: faceprints are biometric data.
- Only kept people's faceprints are stored, and only in memory by default.
- A bystander's faceprint is computed, compared, and immediately discarded.
  It is never stored on a track, logged, or written anywhere.
- The keep list is written to disk only by KeepList.save, which callers must
  only use on an explicit user request (--save-keep-list). It saves
  faceprints and labels only, never face images.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from npcify.models import ensure_model
from npcify.types import BoundingBox, Frame, Point

Faceprint = NDArray[np.float32]  # shape (128,), unit length

# Cosine similarity at or above which a faceprint counts as the same person.
#
# OpenCV suggests 0.363 for SFace (its operating point on the LFW benchmark).
# We match each face against every faceprint of every kept person and take
# the best, so a stranger gets ~10 chances per kept person to score high;
# that is why the default is a little stricter than 0.363.
#   Higher (e.g. 0.5): strangers almost never revealed, but the kept person
#     may stay covered at odd angles or in poor light.
#   Lower (e.g. 0.3): the kept person is revealed faster and more reliably,
#     but a lookalike is more likely to be revealed.
# Being too strict only costs the creator some blur; being too loose exposes
# a bystander. When unsure, raise it.
DEFAULT_MATCH_THRESHOLD = 0.40


class FaceprintModel(Protocol):
    """Anything that can turn a face (box + 5 landmarks) into a faceprint."""

    def faceprint(
        self, frame: Frame, box: BoundingBox, landmarks: Sequence[Point]
    ) -> Faceprint: ...


class SFaceRecognizer:
    """OpenCV's SFace model (cv2.FaceRecognizerSF), via the model cache.

    It aligns the face to a 112x112 crop using the 5 YuNet landmarks, then
    computes a faceprint. About 4-5 ms per face on an M-series Mac.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else ensure_model("sface")
        log = cv2.utils.logging
        previous_level = log.getLogLevel()
        log.setLogLevel(log.LOG_LEVEL_ERROR)  # silence OpenCV 5's harmless target warning
        try:
            self._net = cv2.FaceRecognizerSF.create(str(path), "")
        finally:
            log.setLogLevel(previous_level)

    def faceprint(self, frame: Frame, box: BoundingBox, landmarks: Sequence[Point]) -> Faceprint:
        row = np.array(
            [[box.x, box.y, box.width, box.height, *(v for p in landmarks for v in p), 1.0]],
            dtype=np.float32,
        )
        crop = self._net.alignCrop(frame, row)
        return normalise(self._net.feature(crop).reshape(-1))


def normalise(v: NDArray[np.floating]) -> Faceprint:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass(frozen=True)
class QualityGate:
    """Only good views produce faceprints: big enough and roughly facing the camera.

    Uses the 5 landmarks:
        yaw: how far the nose sits from the midpoint between the eyes, in
            eye-distances (0 = facing the camera, ~0.5+ = side-on).
        roll: tilt of the line between the eyes, in degrees.
        pitch: where the nose sits between the eye line (0) and the mouth (1);
            about 0.55 when level, smaller looking up, larger looking down.
    """

    min_width: int = 60
    max_yaw: float = 0.35
    max_roll_deg: float = 25.0
    pitch_range: tuple[float, float] = (0.3, 0.8)

    def passes(self, box: BoundingBox, landmarks: Sequence[Point] | None) -> bool:
        return self.problem(box, landmarks) is None

    def problem(self, box: BoundingBox, landmarks: Sequence[Point] | None) -> str | None:
        """Why this view is not good enough (a short hint for the user), or None if it is."""
        if landmarks is None or len(landmarks) != 5:
            return "face not visible"
        if box.width < self.min_width:
            return "too small: move closer"
        (rx, ry), (lx, ly), (nx, ny), (mrx, mry), (mlx, mly) = landmarks
        dx, dy = lx - rx, ly - ry
        eye_dist = math.hypot(dx, dy)
        if eye_dist < 0.2 * box.width:  # eyes collapsed together: side-on
            return "face the camera"
        if abs((nx - (rx + lx) / 2) / eye_dist) > self.max_yaw:
            return "face the camera"
        if dx == 0 or abs(math.degrees(math.atan(dy / dx))) > self.max_roll_deg:
            return "keep your head level"
        eye_y, mouth_y = (ry + ly) / 2, (mry + mly) / 2
        if mouth_y <= eye_y:
            return "face the camera"
        pitch = (ny - eye_y) / (mouth_y - eye_y)
        if not self.pitch_range[0] <= pitch <= self.pitch_range[1]:
            return "look straight at the camera"
        return None


@dataclass
class KeptPerson:
    label: str
    faceprints: list[Faceprint] = field(default_factory=list)


class KeepList:
    """The people who stay real, each with several faceprints.

    In memory only, unless `save` is called on an explicit user request.
    """

    def __init__(self, threshold: float = DEFAULT_MATCH_THRESHOLD) -> None:
        self.threshold = threshold
        self.people: list[KeptPerson] = []

    def __len__(self) -> int:
        return len(self.people)

    @property
    def labels(self) -> list[str]:
        return [p.label for p in self.people]

    def next_label(self) -> str:
        n = 1
        while f"person {n}" in self.labels:
            n += 1
        return f"person {n}"

    def add(self, label: str, faceprints: Sequence[Faceprint]) -> None:
        if not faceprints:
            raise ValueError("a kept person needs at least one faceprint")
        if label in self.labels:
            raise ValueError(f"{label!r} is already on the keep list")
        self.people.append(KeptPerson(label, [normalise(f) for f in faceprints]))

    def remove(self, label: str) -> None:
        self.people = [p for p in self.people if p.label != label]

    def clear(self) -> None:
        self.people = []

    def best_match(self, faceprint: Faceprint) -> tuple[str | None, float]:
        """The kept person this faceprint is most similar to, and the similarity.

        Similarity is the best cosine similarity against any of that person's
        faceprints. The label is None if nobody reaches the threshold.
        """
        best_label, best_score = None, -1.0
        for person in self.people:
            score = float(np.max(np.stack(person.faceprints) @ faceprint))
            if score > best_score:
                best_label, best_score = person.label, score
        if best_score < self.threshold:
            return None, best_score
        return best_label, best_score

    def save(self, path: str | Path) -> None:
        """Write labels and faceprints (no images) to a .npz file. Explicit opt-in only."""
        labels = np.array(self.labels, dtype=str)
        counts = np.array([len(p.faceprints) for p in self.people], dtype=np.int64)
        prints = (
            np.concatenate([np.stack(p.faceprints) for p in self.people])
            if self.people
            else np.zeros((0, 128), np.float32)
        )
        np.savez(Path(path), labels=labels, counts=counts, faceprints=prints.astype(np.float32))

    @classmethod
    def load(cls, path: str | Path, threshold: float = DEFAULT_MATCH_THRESHOLD) -> KeepList:
        keep = cls(threshold)
        with np.load(Path(path), allow_pickle=False) as data:
            labels, counts, prints = data["labels"], data["counts"], data["faceprints"]
        start = 0
        for label, count in zip(labels, counts, strict=True):
            keep.add(str(label), list(prints[start : start + int(count)]))
            start += int(count)
        return keep


class Enrolment:
    """Collects faceprints from one track to add a new person to the keep list.

    Takes a faceprint every `interval` checks (so the views differ a little),
    and ignores any that don't look like the faceprints collected so far, in
    case the track switched to someone else mid-way.
    """

    def __init__(
        self,
        track_id: int,
        label: str,
        target: int = 10,
        interval: int = 2,
        consistency: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.track_id = track_id
        self.label = label
        self.target = target
        self.interval = interval
        self.consistency = consistency
        self.faceprints: list[Faceprint] = []
        self._offered = 0

    @property
    def done(self) -> bool:
        return len(self.faceprints) >= self.target

    def offer(self, faceprint: Faceprint) -> None:
        if self.done:
            return
        self._offered += 1
        if (self._offered - 1) % self.interval:
            return
        if self.faceprints:
            mean = normalise(np.mean(np.stack(self.faceprints), axis=0))
            if float(mean @ faceprint) < self.consistency:
                return
        self.faceprints.append(faceprint)
