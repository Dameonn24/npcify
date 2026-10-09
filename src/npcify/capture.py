"""Read frames from a camera or a video file.

This is the entry point for pixels: everything downstream receives frames
produced here. Frames never leave the device.
"""

from __future__ import annotations

from collections.abc import Iterator

from npcify.types import Frame


def open_camera(index: int = 0) -> Iterator[Frame]:
    """Yield frames from a local camera, one at a time."""
    raise NotImplementedError


def open_video(path: str) -> Iterator[Frame]:
    """Yield frames from a video file on disk, one at a time."""
    raise NotImplementedError
