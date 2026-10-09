"""Send processed frames somewhere: a preview window, a virtual camera, or a file.

Saving to disk only happens when the user explicitly asks for it.
"""

from __future__ import annotations

import cv2

from npcify.types import Frame

_QUIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 = Esc
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_LINE_HEIGHT = 18


def format_overlay_lines(
    fps: float,
    frame_ms: float,
    resolution: tuple[int, int],
    stage_ms: dict[str, float] | None = None,
) -> list[str]:
    """Build the text lines shown in the preview overlay."""
    lines = [
        f"{fps:5.1f} FPS  ({frame_ms:.1f} ms/frame)",
        f"{resolution[0]}x{resolution[1]}",
    ]
    for name, ms in (stage_ms or {}).items():
        lines.append(f"{name}: {ms:.1f} ms")
    return lines


def draw_overlay(frame: Frame, lines: list[str]) -> Frame:
    """Return a copy of the frame with white, dark-outlined text in the top-left corner."""
    out = frame.copy()
    for i, text in enumerate(lines):
        origin = (10, 22 + i * _LINE_HEIGHT)
        # Thick dark stroke first, then thin white text on top: readable on any background.
        cv2.putText(out, text, origin, _FONT, _FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, origin, _FONT, _FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class PreviewWindow:
    """A local on-screen preview using cv2.imshow.

    macOS only allows GUI calls on the main thread, so create and use this from
    the main thread only.
    """

    def __init__(self, title: str = "npcify") -> None:
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)

    def show(self, frame: Frame, overlay_lines: list[str] | None = None) -> bool:
        """Display a frame. Returns True if the user asked to quit (q, Esc, or closed window)."""
        if overlay_lines:
            frame = draw_overlay(frame, overlay_lines)
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in _QUIT_KEYS:
            return True
        return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1

    def close(self) -> None:
        cv2.destroyWindow(self.title)
        # Give the macOS event loop a moment to actually tear the window down.
        cv2.waitKey(1)


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
