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
    extra: list[str] | None = None,
) -> list[str]:
    """Build the text lines shown in the preview overlay. `extra` lines go before timings."""
    lines = [
        f"{fps:5.1f} FPS  ({frame_ms:.1f} ms/frame)",
        f"{resolution[0]}x{resolution[1]}",
        *(extra or []),
    ]
    for name, ms in (stage_ms or {}).items():
        lines.append(f"{name}: {ms:.1f} ms")
    return lines


def put_outlined_text(
    img: Frame,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int] = (255, 255, 255),
    scale: float = _FONT_SCALE,
) -> None:
    """Draw text with a dark outline, in place, so it reads on light and dark scenes.

    The outline is the same text drawn at small offsets rather than with a
    thicker stroke, because in OpenCV 5 thicker text is also wider and would
    not line up with the text on top.
    """
    x, y = origin
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
        cv2.putText(img, text, (x + dx, y + dy), _FONT, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, origin, _FONT, scale, colour, 1, cv2.LINE_AA)


def draw_overlay(frame: Frame, lines: list[str]) -> Frame:
    """Return a copy of the frame with white, dark-outlined text in the top-left corner."""
    out = frame.copy()
    for i, text in enumerate(lines):
        put_outlined_text(out, text, (10, 22 + i * _LINE_HEIGHT))
    return out


class PreviewWindow:
    """A local on-screen preview using cv2.imshow.

    macOS only allows GUI calls on the main thread, so create and use this from
    the main thread only.
    """

    def __init__(self, title: str = "npcify") -> None:
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)

    def show(
        self, frame: Frame, overlay_lines: list[str] | None = None, wait_ms: int = 1
    ) -> str | None:
        """Display a frame and return the key pressed, if any.

        Keys come back as lowercase characters (" " for space). Returns "quit" for
        q, Esc, or the window being closed. Waits up to `wait_ms` for a key.
        """
        if overlay_lines:
            frame = draw_overlay(frame, overlay_lines)
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in _QUIT_KEYS or cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"
        return chr(key).lower() if key != 0xFF else None

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
