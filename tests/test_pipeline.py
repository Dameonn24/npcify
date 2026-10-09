import numpy as np

from npcify import Pipeline
from npcify.types import BoundingBox, Detection, Frame

FAKE = Detection(BoundingBox(10, 10, 50, 50), confidence=0.9)


class FakeDetector:
    def detect(self, frame: Frame) -> list[Detection]:
        return [FAKE]


def test_process_frame_keeps_shape() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = Pipeline(detector=FakeDetector()).process_frame(frame)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_process_frame_stores_detections_and_timing() -> None:
    pipeline = Pipeline(detector=FakeDetector())
    pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert pipeline.detections == [FAKE]
    assert len(pipeline.tracks) == 1
    assert {"detect", "track", "anonymise"} <= set(pipeline.timer.averages())


def test_process_frame_hides_faces_without_touching_input() -> None:
    frame = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    before = frame.copy()
    out = Pipeline(detector=FakeDetector()).process_frame(frame)
    assert (frame == before).all()
    assert (out[10:60, 10:60] != frame[10:60, 10:60]).any(axis=2).mean() > 0.95
    assert (out[200:, 200:] == frame[200:, 200:]).all()


def test_new_face_is_covered_on_its_very_first_frame() -> None:
    frame = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    pipeline = Pipeline(detector=FakeDetector())
    out = pipeline.process_frame(frame)
    assert pipeline.tracks[0].state.value == "tentative"
    assert (out[10:60, 10:60] != frame[10:60, 10:60]).any(axis=2).all()


def test_debug_no_anonymise_returns_faces_visible(capsys) -> None:
    frame = np.random.default_rng(2).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    out = Pipeline(detector=FakeDetector(), debug_no_anonymise=True).process_frame(frame)
    assert (out == frame).all()
    assert "WARNING" in capsys.readouterr().err


class CountingDetector(FakeDetector):
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: Frame) -> list[Detection]:
        self.calls += 1
        return super().detect(frame)


def test_detect_every_n_frames_keeps_tracks_in_between() -> None:
    detector = CountingDetector()
    pipeline = Pipeline(detector=detector, detect_every=3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(9):
        pipeline.process_frame(frame)
        assert len(pipeline.tracks) == 1  # never dropped on skipped frames
    assert detector.calls == 3
