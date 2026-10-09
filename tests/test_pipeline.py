import numpy as np

from npcify import Pipeline


def test_process_frame_keeps_shape() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = Pipeline().process_frame(frame)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
