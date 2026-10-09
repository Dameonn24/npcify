"""Run every stage in order for each frame.

capture -> detect -> track -> decide -> pose -> render -> safety -> output.
Stages are wired in as they are built. So far: detect.
"""

from __future__ import annotations

from npcify.detect import FaceDetector, YuNetDetector
from npcify.timing import StageTimer
from npcify.types import Detection, Frame


class Pipeline:
    """Turns one input frame into one output frame.

    Pass a `timer` to have each stage's time recorded under its own name.
    """

    def __init__(
        self,
        detector: FaceDetector | None = None,
        timer: StageTimer | None = None,
    ) -> None:
        self.detector = detector if detector is not None else YuNetDetector()
        self.timer = timer if timer is not None else StageTimer()
        self.detections: list[Detection] = []

    def process_frame(self, frame: Frame) -> Frame:
        """Process a single frame. Detects faces; the frame itself is not yet changed."""
        with self.timer.stage("detect"):
            self.detections = self.detector.detect(frame)
        return frame
