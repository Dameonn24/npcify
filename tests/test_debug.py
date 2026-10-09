import numpy as np

from npcify.debug import draw_detections
from npcify.types import BoundingBox, Detection


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
