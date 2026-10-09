"""Run every stage in order for each frame.

capture -> detect -> track -> decide -> pose -> render -> safety -> output.
Stages are wired in as they are built. So far: detect, track, decide (with
recognition when a KeepListPolicy is used), and anonymise (every face that
isn't explicitly kept real is hidden).

Detection, tracking and covering all happen on the same frame inside
process_frame, before it is returned, so a face is never shown uncovered while
waiting for a later stage.
"""

from __future__ import annotations

import sys

from npcify.decide import CoveragePolicy, HideEveryone, regions_to_cover
from npcify.detect import FaceDetector, YuNetDetector
from npcify.render import Anonymizer
from npcify.timing import StageTimer
from npcify.track import Tracker
from npcify.types import CoverRegion, Decision, Detection, Frame, Track


class Pipeline:
    """Turns one input frame into one output frame with faces hidden.

    detect_every: run the detector on every Nth frame only; in between, the
        tracker just predicts where faces have moved. 1 = detect every frame.
        WARNING: with N > 1, a new face that appears on a skipped frame is not
        covered until the next detection frame. Only use for benchmarking.
    timer: if given, each stage's time is recorded under its own name.
    debug_no_anonymise: return frames with faces visible. Development only;
        prints a warning. Never use for anything streamed or saved.

    The default detector runs at the tracker's low threshold, so that
    low-confidence detections can keep existing tracks alive (and are covered).
    """

    def __init__(
        self,
        detector: FaceDetector | None = None,
        tracker: Tracker | None = None,
        policy: CoveragePolicy | None = None,
        anonymizer: Anonymizer | None = None,
        timer: StageTimer | None = None,
        detect_every: int = 1,
        debug_no_anonymise: bool = False,
    ) -> None:
        if detect_every < 1:
            raise ValueError("detect_every must be at least 1")
        self.tracker = tracker if tracker is not None else Tracker()
        self.detector = (
            detector
            if detector is not None
            else YuNetDetector(score_threshold=self.tracker.config.low_threshold)
        )
        self.policy = policy if policy is not None else HideEveryone()
        self.anonymizer = anonymizer if anonymizer is not None else Anonymizer()
        self.timer = timer if timer is not None else StageTimer()
        if getattr(self.policy, "timer", False) is None:
            self.policy.timer = self.timer  # type: ignore[attr-defined]  # share "recognise" timings
        self.detect_every = detect_every
        self.debug_no_anonymise = debug_no_anonymise
        if debug_no_anonymise:
            print(
                "WARNING: npcify anonymisation is OFF (debug_no_anonymise). "
                "Faces will be visible. Do not stream or record this output.",
                file=sys.stderr,
            )
        self.frame_index = 0
        # Latest results. `detections` is empty on frames where detection was skipped.
        self.detections: list[Detection] = []
        self.tracks: list[Track] = []
        self.decisions: list[Decision] = []
        self.regions: list[CoverRegion] = []

    def process_frame(self, frame: Frame) -> Frame:
        """Detect, track and hide faces. Returns a new frame; the input is unchanged."""
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
        with self.timer.stage("decide"):
            self.decisions = self.policy.decide(frame, self.tracks)
            self.regions = regions_to_cover(self.tracks, self.detections, self.decisions)
        with self.timer.stage("anonymise"):
            return self.anonymise(frame)

    def anonymise(self, frame: Frame) -> Frame:
        """Hide the latest regions in `frame` (e.g. to re-render after changing style)."""
        if self.debug_no_anonymise:
            return frame.copy()
        return self.anonymizer.apply(frame, [r.box for r in self.regions])
