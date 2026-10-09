"""Send processed frames somewhere: a preview window, a virtual camera, or a file.

Saving to disk only happens when the user explicitly asks for it.
"""

from __future__ import annotations

from npcify.types import Frame


def show_window(frame: Frame, title: str = "npcify") -> None:
    """Display the frame in a local preview window."""
    raise NotImplementedError


def send_to_virtual_camera(frame: Frame) -> None:
    """Push the frame to a virtual camera so other apps can use it."""
    raise NotImplementedError


class VideoFileWriter:
    """Writes frames to a video file on disk (opt-in only)."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.fps = fps
        self.size = size

    def write(self, frame: Frame) -> None:
        """Append one frame to the file."""
        raise NotImplementedError

    def close(self) -> None:
        """Finish and close the file."""
        raise NotImplementedError
