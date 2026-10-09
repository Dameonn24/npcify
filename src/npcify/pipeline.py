"""Run every stage in order for each frame.

capture -> detect -> track -> decide -> pose -> render -> safety -> output.
For now it is a pass-through; stages are wired in as they are built.
"""

from __future__ import annotations

from npcify.types import Frame


class Pipeline:
    """Turns one input frame into one output frame."""

    def process_frame(self, frame: Frame) -> Frame:
        """Process a single frame. Currently returns it unchanged."""
        return frame
