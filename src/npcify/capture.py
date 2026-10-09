"""Read frames from a camera or a video file.

This is the entry point for pixels: everything downstream receives frames
produced here. Frames never leave the device.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType

import cv2

from npcify.types import Frame

_CAMERA_PERMISSION_HELP = (
    "On macOS, check System Settings > Privacy & Security > Camera and allow the app "
    "you ran this from (Terminal, iTerm, or VS Code), then restart that app. "
    "Also check that no other app is using the camera and that the index is right."
)


class CameraError(RuntimeError):
    """The camera could not be opened or stopped delivering frames."""


class FrameSource:
    """Frames from a camera (pass an int index) or a video file (pass a path).

    Mirroring defaults to on for cameras (so it feels like a mirror) and off for
    video files. Use as a context manager to make sure the device is released.
    """

    def __init__(
        self,
        source: int | str | Path = 0,
        width: int = 1280,
        height: int = 720,
        mirror: bool | None = None,
    ) -> None:
        self.is_camera = isinstance(source, int)
        self.mirror = self.is_camera if mirror is None else mirror

        if self.is_camera:
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(source, backend)
            if not self._cap.isOpened():
                raise CameraError(f"Could not open camera {source}. {_CAMERA_PERMISSION_HELP}")
            # A request, not a guarantee: the camera picks its nearest supported mode.
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path}")
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise ValueError(f"Could not open video file (unsupported format?): {path}")

    @property
    def resolution(self) -> tuple[int, int]:
        """The actual (width, height) being delivered, which may differ from the request."""
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def source_fps(self) -> float:
        """The FPS the source reports (a video's native rate; often unreliable for cameras)."""
        return float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self) -> Frame | None:
        """Return the next frame, or None at the end of a video or if the camera stops."""
        ok, frame = self._cap.read()
        if not ok:
            return None
        return cv2.flip(frame, 1) if self.mirror else frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
