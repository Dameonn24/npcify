"""Head pose and facial expression for each hidden (bystander) face.

For each bystander track, the padded face box is cropped and MediaPipe's Face
Landmarker runs on just that crop (we never use its own face finding to
decide who is in the frame). From its result we read:
- head angles (yaw, pitch, roll) from the facial transformation matrix,
- 52 blendshape scores, mapped to the small AvatarExpression set that avatar
  rendering uses.
Faces too small for the landmarker, or where it fails, get a coarse pose from
YuNet's 5 landmarks and a neutral expression.

All angles and left/right values are in IMAGE terms (see FaceState and
AvatarExpression), measured on the frame as it will be shown. So they are
the right way round for drawing whether or not the video is mirrored.

Privacy: landmarks and expression values live in memory only, for live
tracks, and are dropped as soon as a track is removed or kept real. They are
never logged or written to disk.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from npcify.models import ensure_model
from npcify.smoothing import OneEuroFilter
from npcify.types import AvatarExpression, Decision, FaceState, Frame, Point, Track

# --- head angles --------------------------------------------------------------
#
# MediaPipe's transformation matrix is in camera space: x to the image's
# right, y up, z toward the viewer. A face looking straight at the camera has
# rotation = identity. We decompose R = Ry(yaw) @ Rx(-pitch) @ Rz(-roll), so:
#   yaw   > 0: face turned toward the image's right
#   pitch > 0: looking up
#   roll  > 0: head tilted clockwise as seen in the image


def _ry(a: float) -> NDArray[np.float64]:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(a: float) -> NDArray[np.float64]:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a: float) -> NDArray[np.float64]:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def angles_to_rotation(yaw: float, pitch: float, roll: float) -> NDArray[np.float64]:
    """Rotation matrix (camera space) for angles in degrees, in our convention."""
    return _ry(math.radians(yaw)) @ _rx(math.radians(-pitch)) @ _rz(math.radians(-roll))


def rotation_to_angles(rotation: NDArray[np.floating]) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 3x3 camera-space rotation matrix."""
    r = np.asarray(rotation, dtype=np.float64)[:3, :3]
    # Remove any scale the matrix carries by snapping to the nearest rotation.
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, r[1, 2]))))
    roll = -math.degrees(math.atan2(r[1, 0], r[1, 1]))
    return yaw, pitch, roll


def mirror_rotation(rotation: NDArray[np.floating]) -> NDArray[np.float64]:
    """The rotation as seen in a horizontally flipped image."""
    s = np.diag([-1.0, 1.0, 1.0])
    return s @ np.asarray(rotation, dtype=np.float64)[:3, :3] @ s


# --- expressions --------------------------------------------------------------

# How far jawOpen typically goes for a wide-open mouth; scaled up to reach 1.
_JAW_FULL = 0.6


def blendshapes_to_avatar(bs: Mapping[str, float]) -> AvatarExpression:
    """Map MediaPipe's 52 blendshapes to the AvatarExpression contract.

    MediaPipe names sides from the person's point of view, as in an unmirrored
    photo: its "Right" eye is the one on the image's LEFT (checked by closing
    one eye on a test image). AvatarExpression uses image sides, so the names
    swap here. This holds for mirrored frames too, because the landmarker only
    ever sees the image.
    """

    def g(name: str) -> float:
        return float(bs.get(name, 0.0))

    def clip01(v: float) -> float:
        return max(0.0, min(1.0, v))

    def clip11(v: float) -> float:
        return max(-1.0, min(1.0, v))

    # Image-right eye is MediaPipe's "Left": looking "out" means toward image right.
    look_x = (
        (g("eyeLookOutLeft") - g("eyeLookInLeft")) + (g("eyeLookInRight") - g("eyeLookOutRight"))
    ) / 2
    look_y = (
        (g("eyeLookUpLeft") + g("eyeLookUpRight")) - (g("eyeLookDownLeft") + g("eyeLookDownRight"))
    ) / 2
    return AvatarExpression(
        blink_left=clip01(g("eyeBlinkRight")),
        blink_right=clip01(g("eyeBlinkLeft")),
        mouth_open=clip01(g("jawOpen") / _JAW_FULL),
        smile=clip01((g("mouthSmileLeft") + g("mouthSmileRight")) / 2),
        brows_up=clip01(max(g("browInnerUp"), (g("browOuterUpLeft") + g("browOuterUpRight")) / 2)),
        brows_down=clip01((g("browDownLeft") + g("browDownRight")) / 2),
        look_x=clip11(look_x),
        look_y=clip11(look_y),
    )


def coarse_pose(landmarks: Sequence[Point]) -> tuple[float, float, float]:
    """Rough (yaw, pitch, roll) in degrees from YuNet's 5 points (eyes, nose, mouth).

    roll: angle of the line between the eyes.
    yaw: how far the nose sits from the eyes' midpoint, along the eye line.
    pitch: where the nose sits between the eye line and the mouth.
    The eyes are taken in image left-to-right order, so mirroring doesn't matter.
    """
    eye_a, eye_b = sorted(landmarks[:2], key=lambda p: p[0])
    nose = np.array(landmarks[2], dtype=np.float64)
    mouth = (np.array(landmarks[3]) + np.array(landmarks[4])) / 2
    a, b = np.array(eye_a, dtype=np.float64), np.array(eye_b, dtype=np.float64)
    along = b - a
    eye_dist = float(np.linalg.norm(along))
    if eye_dist < 1e-6:
        return 0.0, 0.0, 0.0
    ux = along / eye_dist
    uy = np.array([-ux[1], ux[0]])  # perpendicular, pointing down the face
    mid = (a + b) / 2

    roll = math.degrees(math.atan2(along[1], along[0]))
    offset = float((nose - mid) @ ux) / eye_dist  # about +-0.5 when side-on
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, offset / 0.6))))
    mouth_depth = float((mouth - mid) @ uy)
    pitch = 0.0
    if mouth_depth > 1e-6:
        ratio = float((nose - mid) @ uy) / mouth_depth  # ~0.55 when level
        pitch = max(-45.0, min(45.0, -(ratio - 0.55) * 100.0))
    return yaw, pitch, roll


# --- landmarker ---------------------------------------------------------------


@dataclass(frozen=True)
class LandmarkResult:
    """One face's landmarker output, in crop coordinates."""

    rotation: NDArray[np.float64]  # 3x3, camera space
    blendshapes: dict[str, float]
    points: NDArray[np.float32]  # (N, 2), normalised 0-1 within the crop


class FaceLandmarkModel(Protocol):
    def run(self, crop: Frame) -> LandmarkResult | None: ...


class MediaPipeLandmarker:
    """MediaPipe Tasks Face Landmarker in IMAGE mode, one face per crop."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        path = Path(model_path) if model_path else ensure_model("face_landmarker")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def run(self, crop: Frame) -> LandmarkResult | None:
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        )
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return None
        points = np.array([(p.x, p.y) for p in result.face_landmarks[0]], dtype=np.float32)
        blendshapes = (
            {b.category_name: float(b.score) for b in result.face_blendshapes[0]}
            if result.face_blendshapes
            else {}
        )
        matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
        return LandmarkResult(matrix[:3, :3], blendshapes, points)


# --- per-track estimation -----------------------------------------------------


@dataclass(frozen=True)
class PoseConfig:
    """Settings for PoseEstimator.

    max_per_frame: landmarker runs per frame at most (~4 ms each); other faces
        hold their last values until their turn comes round.
    min_landmarker_width: faces narrower than this (pixels) use the coarse pose.
    crop_scale: crop size around the face box, as a multiple of its larger side.
    angle_* / expression_*: One Euro filter settings (see smoothing.py). Angles
        are in degrees and expressions 0-1, hence the very different betas.
    """

    max_per_frame: int = 4
    min_landmarker_width: int = 50
    crop_scale: float = 1.6
    # Tuned on synthetic noise: angles follow a 30-degree turn within ~3 frames
    # and cut still-face jitter to ~0.3x; expressions register a blink within
    # ~2 frames and cut jitter to ~0.45x.
    angle_min_cutoff: float = 0.5
    angle_beta: float = 0.02
    expression_min_cutoff: float = 1.0
    expression_beta: float = 3.0
    landmarker_confidence: float = 0.9
    coarse_confidence: float = 0.3


@dataclass
class _Smoother:
    angles: OneEuroFilter
    expression: OneEuroFilter


class PoseEstimator:
    """Keeps a smoothed FaceState for every hidden track.

    Kept (real) people never get a FaceState. A track's state and filters are
    deleted as soon as it is removed or becomes kept.
    """

    def __init__(
        self,
        model: FaceLandmarkModel | None = None,
        config: PoseConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model = model if model is not None else MediaPipeLandmarker()
        self.config = config or PoseConfig()
        self.clock = clock
        self.states: dict[int, FaceState] = {}
        self._smoothers: dict[int, _Smoother] = {}
        self._last_run: dict[int, int] = {}  # run sequence number, for round-robin order
        self._runs = 0

    def update(
        self,
        frame: Frame,
        tracks: list[Track],
        decisions: list[Decision],
        now: float | None = None,
    ) -> dict[int, FaceState]:
        """Update states for this frame and return {track_id: FaceState} for hidden tracks."""
        now = self.clock() if now is None else now
        kept = {d.track_id for d in decisions if d.keep_real}
        hidden = [t for t in tracks if t.track_id not in kept]
        live = {t.track_id for t in hidden}
        for store in (self.states, self._smoothers, self._last_run):
            for tid in [k for k in store if k not in live]:
                del store[tid]

        for t in hidden:
            self.states.setdefault(t.track_id, FaceState(t.track_id))
        for t in self.schedule(hidden):
            self._runs += 1
            self._last_run[t.track_id] = self._runs  # unique, so ties never favour low IDs
            self._measure(frame, t, now)
        return dict(self.states)

    def schedule(self, tracks: list[Track]) -> list[Track]:
        """Round-robin: the tracks seen this frame that waited longest, up to the limit."""
        visible = [t for t in tracks if t.frames_since_seen == 0]
        visible.sort(key=lambda t: (self._last_run.get(t.track_id, -1), t.track_id))
        return visible[: self.config.max_per_frame]

    def _measure(self, frame: Frame, t: Track, now: float) -> None:
        cfg = self.config
        result = None
        crop_origin = (0, 0, 1, 1)
        if t.box.width >= cfg.min_landmarker_width:
            crop, crop_origin = self._crop(frame, t)
            if crop is not None:
                result = self.model.run(crop)

        if result is not None:
            yaw, pitch, roll = rotation_to_angles(result.rotation)
            avatar = blendshapes_to_avatar(result.blendshapes)
            x0, y0, cw, ch = crop_origin
            mesh = (result.points * np.array([cw, ch], np.float32) + [x0, y0]).astype(np.float32)
            self._apply(
                t.track_id,
                now,
                (yaw, pitch, roll),
                avatar,
                result.blendshapes,
                "landmarker",
                cfg.landmarker_confidence,
                mesh,
            )
        elif t.landmarks is not None:
            self._apply(
                t.track_id,
                now,
                coarse_pose(t.landmarks),
                AvatarExpression(),
                {},
                "coarse",
                cfg.coarse_confidence,
                None,
            )
        # Otherwise hold the last values.

    def _crop(self, frame: Frame, t: Track) -> tuple[Frame | None, tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        b = t.box
        side = self.config.crop_scale * max(b.width, b.height)
        cx, cy = b.x + b.width / 2, b.y + b.height / 2
        x0, y0 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
        x1, y1 = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, (0, 0, 1, 1)
        return frame[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)

    def _apply(
        self,
        track_id: int,
        now: float,
        angles: tuple[float, float, float],
        avatar: AvatarExpression,
        raw: dict[str, float],
        source: str,
        confidence: float,
        mesh: NDArray[np.float32] | None,
    ) -> None:
        cfg = self.config
        sm = self._smoothers.get(track_id)
        if sm is None:
            sm = _Smoother(
                OneEuroFilter(cfg.angle_min_cutoff, cfg.angle_beta),
                OneEuroFilter(cfg.expression_min_cutoff, cfg.expression_beta),
            )
            self._smoothers[track_id] = sm
        yaw, pitch, roll = sm.angles(angles, now)
        e = sm.expression(list(vars(avatar).values()), now)
        self.states[track_id] = FaceState(
            track_id=track_id,
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            expressions=raw,
            avatar=AvatarExpression(*(float(v) for v in e)),
            source=source,
            confidence=confidence,
            mesh=mesh,
        )
