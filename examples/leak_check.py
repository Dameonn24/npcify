"""Check a video for frames where a face might not have been hidden.

    uv run python examples/leak_check.py some_clip.mp4
    uv run python examples/leak_check.py some_clip.mp4 --no-save

For every frame it checks:
  1. Uncovered detections: a raw detection whose box is not entirely inside
     the fully-hidden part of a cover region. Should never happen: a bug.
  2. Faces found in the OUTPUT: a second detector runs at full resolution on
     the anonymised frame.
     - Centre, or two or more of eyes/nose/mouth corners, outside the hidden
       area: a real leak, e.g. a small or side-on face the pipeline's detector
       missed. Saved for review.
     - Otherwise inside the hidden area: the face is hidden but its rough shape (dark
       blocks where the eyes are) is still face-like to a detector. Counted
       only; a detector finding "a face" is not the same as anyone being able
       to tell who it is. The "solid" style scores zero here.
  3. Lost tracks: the face is being covered by a predicted box. Saved so you
     can check by eye that the prediction really covered the face.

Flagged frames are saved (anonymised, with debug boxes drawn) to
outputs/leak_check/. These images can contain faces that leaked, which is
the point of the check: delete them when you're done. Use --no-save to only
print the summary.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from npcify.capture import FrameSource
from npcify.debug import draw_tracks
from npcify.detect import YuNetDetector
from npcify.pipeline import Pipeline
from npcify.render import STYLES, Anonymizer
from npcify.track import Tracker
from npcify.types import Detection, TrackState

OUT_DIR = Path("outputs/leak_check")


def features_hidden(det: Detection, mask: np.ndarray) -> bool:
    """True if a face found in the output is (essentially) inside the hidden area.

    It counts as hidden if its centre is hidden and at most one of its five
    landmarks (eyes, nose, mouth corners) falls outside the hidden area. On a
    pixelated face the recheck detector's box and landmarks are guesses, and
    one point often lands just past the hidden core; a genuinely half-visible
    face has several features showing.
    """
    h, w = mask.shape

    def hidden(x: float, y: float) -> bool:
        return bool(mask[min(max(round(y), 0), h - 1), min(max(round(x), 0), w - 1)])

    b = det.box
    if not hidden(b.x + b.width / 2, b.y + b.height / 2):
        return False
    outside = sum(not hidden(x, y) for x, y in det.landmarks or ())
    return outside <= 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("video", help="path to a video file")
    p.add_argument("--style", choices=STYLES, default=STYLES[0])
    p.add_argument(
        "--recheck-threshold",
        type=float,
        default=0.5,
        help="confidence at which a face found in the output counts as a leak (default: 0.5)",
    )
    p.add_argument(
        "--det-width",
        type=int,
        default=None,
        help="width the pipeline's detector runs at (default: the pipeline default)",
    )
    p.add_argument("--max-saves", type=int, default=300, help="cap on saved frames (default: 300)")
    p.add_argument("--no-save", action="store_true", help="don't save any frames")
    args = p.parse_args()

    tracker = Tracker()
    detector = None
    if args.det_width:
        detector = YuNetDetector(
            score_threshold=tracker.config.low_threshold, det_width=args.det_width
        )
    pipeline = Pipeline(detector=detector, tracker=tracker, anonymizer=Anonymizer(style=args.style))
    anonymizer = pipeline.anonymizer

    if not args.no_save:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents=True)

    n_frames = 0
    lost_frames = 0
    uncovered_frames = 0
    uncovered_total = 0
    leak_frames = 0
    leak_total = 0
    facelike_total = 0
    saved = 0
    recheck: YuNetDetector | None = None

    with FrameSource(args.video) as source:
        width, height = source.resolution
        print(f"{args.video}: {width}x{height}, style {args.style}")
        # Full-resolution detector for the output check: finds smaller faces than the pipeline's.
        recheck = YuNetDetector(score_threshold=args.recheck_threshold, det_width=max(width, 640))
        while (frame := source.read()) is not None:
            out = pipeline.process_frame(frame)
            mask = anonymizer.coverage_mask(frame.shape, [r.box for r in pipeline.regions])

            uncovered = []
            for d in pipeline.detections:
                b = d.box
                if not mask[b.y : b.y + b.height, b.x : b.x + b.width].all():
                    uncovered.append(d)
            lost = [t for t in pipeline.tracks if t.state is TrackState.LOST]
            # Split output detections into real leaks and still-face-like hidden faces.
            leaks, facelike = [], []
            for d in recheck.detect(out):
                (facelike if features_hidden(d, mask) else leaks).append(d)
            facelike_total += len(facelike)

            reasons = []
            if uncovered:
                uncovered_frames += 1
                uncovered_total += len(uncovered)
                reasons.append("UNCOVERED")
                print(f"  frame {n_frames}: BUG: {len(uncovered)} detection(s) not fully covered")
            if leaks:
                leak_frames += 1
                leak_total += len(leaks)
                reasons.append("LEAK")
            if lost:
                lost_frames += 1
                reasons.append("lost")

            if reasons and not args.no_save and saved < args.max_saves:
                debug = draw_tracks(out, pipeline.tracks, None, pipeline.detections)
                for d in leaks:
                    b = d.box
                    cv2.rectangle(
                        debug, (b.x, b.y), (b.x + b.width, b.y + b.height), (0, 0, 255), 3
                    )
                cv2.imwrite(str(OUT_DIR / f"{n_frames:06d}_{'_'.join(reasons)}.jpg"), debug)
                saved += 1
            n_frames += 1

    if n_frames == 0:
        print("No frames read.", file=sys.stderr)
        return 1

    print()
    print(f"frames checked:          {n_frames}")
    print(f"frames with lost tracks: {lost_frames}")
    print(f"uncovered detections:    {uncovered_total} (in {uncovered_frames} frames)")
    print(f"leaks (face in output):  {leak_total} in {leak_frames} frames")
    print(f"hidden but face-shaped:  {facelike_total} (informational)")
    if not args.no_save:
        capped = " (capped)" if saved >= args.max_saves else ""
        print(f"saved {saved} frames to {OUT_DIR}/{capped}; delete them when done.")
    if uncovered_total:
        print("\nFAIL: some detections were not fully covered. This is a bug.")
        return 2
    if leak_total:
        print(
            "\nCheck the LEAK frames (red boxes): a face was found that no region covered. "
            "Usually a face the pipeline's detector missed (small, side-on); "
            "sometimes a false alarm."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
