"""Run every stage in order for each frame.

capture -> detect -> track -> decide -> pose -> render -> safety -> output.
Stages are wired in as they are built. So far: detect, track.
"""

from __future__ import annotations

from npcify.detect import FaceDetector, YuNetDetector
from npcify.timing import StageTimer
from npcify.track import Tracker
from npcify.types import Detection, Frame, Track


class Pipeline:
    """Turns one input frame into one output frame.

    detect_every: run the detector on every Nth frame only; in between, the
        tracker just predicts where faces have moved. 1 = detect every frame.
    timer: if given, each stage's time is recorded under its own name.

    The default detector runs at the tracker's low threshold, so that
    low-confidence detections can keep existing tracks alive.
    """

    def __init__(
        self,
        detector: FaceDetector | None = None,
        tracker: Tracker | None = None,
        timer: StageTimer | None = None,
        detect_every: int = 1,
    ) -> None:
        if detect_every < 1:
            raise ValueError("detect_every must be at least 1")
        self.tracker = tracker if tracker is not None else Tracker()
        self.detector = (
            detector
            if detector is not None
            else YuNetDetector(score_threshold=self.tracker.config.low_threshold)
        )
        self.timer = timer if timer is not None else StageTimer()
        self.detect_every = detect_every
        self.frame_index = 0
        # Latest results. `detections` is empty on frames where detection was skipped.
        self.detections: list[Detection] = []
        self.tracks: list[Track] = []

    def process_frame(self, frame: Frame) -> Frame:
        """Process a single frame. Detects and tracks faces; the frame is not yet changed."""
        frame_size = (frame.shape[1], frame.shape[0])
        run_detector = self.frame_index % self.detect_every == 0
        self.frame_index += 1

        if run_detector:
            with self.timer.stage("detect"):
                self.detections = self.detector.detect(frame)
        else:
            self.detections = []
        with self.timer.stage("track"):
            self.tracks = self.tracker.update(self.detections if run_detector else None, frame_size)
        return frame
