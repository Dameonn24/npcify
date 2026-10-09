import numpy as np

from npcify.output import draw_overlay, format_overlay_lines


def test_format_overlay_lines() -> None:
    lines = format_overlay_lines(29.97, 33.4, (1280, 720), {"capture": 1.23})
    assert lines == [" 30.0 FPS  (33.4 ms/frame)", "1280x720", "capture: 1.2 ms"]


def test_draw_overlay_draws_without_touching_input() -> None:
    frame = np.full((120, 200, 3), 128, dtype=np.uint8)
    out = draw_overlay(frame, ["hello"])
    assert out.shape == frame.shape
    assert (frame == 128).all()  # input unchanged
    assert (out != 128).any()  # something was drawn
