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
    assert "detect" in pipeline.timer.averages()
