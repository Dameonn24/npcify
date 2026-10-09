"""Safety net: blur anything we're unsure about, and delay output.

If a face can't be confidently avatarised, it is blurred instead. A short
delay buffer gives the pipeline time to catch faces it missed for a frame or
two before those frames are shown. A real bystander face must never slip
through.
"""

from __future__ import annotations

from npcify.types import BoundingBox, Frame


def blur_regions(frame: Frame, boxes: list[BoundingBox]) -> Frame:
    """Return a copy of the frame with each box heavily blurred."""
    raise NotImplementedError


class DelayBuffer:
    """Holds the last few frames so output lags input by a fixed amount."""

    def __init__(self, delay_frames: int) -> None:
        self.delay_frames = delay_frames

    def push(self, frame: Frame) -> Frame | None:
        """Add a frame; return the delayed frame once the buffer is full, else None."""
        raise NotImplementedError
