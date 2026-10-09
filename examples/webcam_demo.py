"""Live webcam (or video file) preview through the npcify pipeline, with an FPS overlay.

    uv run python examples/webcam_demo.py
    uv run python examples/webcam_demo.py --video some_clip.mp4

Press q or Esc to quit.
"""

from __future__ import annotations

import argparse
import sys
import time

from npcify import Pipeline
from npcify.capture import CameraError, FrameSource
from npcify.debug import TrackTrails, draw_tracks
from npcify.detect import YuNetDetector
from npcify.models import ModelDownloadError
from npcify.output import PreviewWindow, format_overlay_lines
from npcify.timing import FPSCounter, StageTimer
from npcify.track import Tracker, TrackerConfig
from npcify.types import TrackState


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--video", help="path to a video file (overrides --camera)")
    p.add_argument("--width", type=int, default=1280, help="requested width (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="requested height (default: 720)")
    p.add_argument("--no-mirror", action="store_true", help="don't flip the webcam image")
    p.add_argument("--no-overlay", action="store_true", help="hide the FPS overlay")
    p.add_argument(
        "--show-detections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw tracked faces with IDs, trails and landmarks (default: on)",
    )
    p.add_argument(
        "--det-threshold",
        type=float,
        default=0.6,
        help="confidence needed to start tracking a new face (default: 0.6)",
    )
    p.add_argument(
        "--low-threshold",
        type=float,
        default=0.3,
        help="lower confidence that can keep an existing track alive (default: 0.3)",
    )
    p.add_argument(
        "--max-age",
        type=int,
        default=TrackerConfig.max_age,
        help=f"frames to keep a face the detector lost (default: {TrackerConfig.max_age}, ~2 s)",
    )
    p.add_argument(
        "--recover-distance",
        type=float,
        default=TrackerConfig.recover_distance,
        help="how far (in face-widths) a lost face may reappear and keep its ID "
        f"(default: {TrackerConfig.recover_distance})",
    )
    p.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="run detection every N frames; the tracker predicts in between (default: 1)",
    )
    p.add_argument(
        "--det-width",
        type=int,
        default=640,
        help="width detection runs at; larger finds smaller faces but is slower (default: 640)",
    )
    return p.parse_args()


def track_summary(pipeline: Pipeline) -> list[str]:
    def count(state: TrackState) -> int:
        return sum(t.state is state for t in pipeline.tracks)

    return [
        f"faces: {len(pipeline.detections)}  tracks: {len(pipeline.tracks)}",
        f"  {count(TrackState.CONFIRMED)} confirmed, {count(TrackState.TENTATIVE)} tentative, "
        f"{count(TrackState.LOST)} lost",
    ]


def main() -> int:
    args = parse_args()
    source_arg = args.video if args.video else args.camera
    mirror = False if args.no_mirror else None  # None = default for the source type

    config = TrackerConfig(
        high_threshold=args.det_threshold,
        low_threshold=min(args.low_threshold, args.det_threshold),
        max_age=args.max_age,
        recover_distance=args.recover_distance,
    )
    try:
        # The detector runs at the low threshold so weak detections reach the tracker.
        detector = YuNetDetector(score_threshold=config.low_threshold, det_width=args.det_width)
    except ModelDownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        source = FrameSource(source_arg, args.width, args.height, mirror=mirror)
    except (CameraError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    fps = FPSCounter()
    timer = StageTimer()
    pipeline = Pipeline(
        detector=detector,
        tracker=Tracker(config),
        timer=timer,
        detect_every=max(1, args.detect_every),
    )
    trails = TrackTrails()
    window: PreviewWindow | None = None

    w, h = source.resolution
    if source.is_camera and (w, h) != (args.width, args.height):
        print(f"Requested {args.width}x{args.height}, camera gave {w}x{h}.")
    else:
        print(f"Resolution: {w}x{h}")

    # Play video files at their native speed rather than as fast as possible.
    frame_period = 1.0 / source.source_fps if not source.is_camera and source.source_fps else 0.0
    next_frame_at = time.perf_counter()

    try:
        window = PreviewWindow()
        while True:
            with timer.stage("capture"):
                frame = source.read()
            if frame is None:
                if source.is_camera:
                    print("Camera stopped delivering frames.", file=sys.stderr)
                break

            with timer.stage("process"):
                frame = pipeline.process_frame(frame)

            with timer.stage("display"):
                if args.show_detections:
                    trails.update(pipeline.tracks)
                    frame = draw_tracks(frame, pipeline.tracks, trails, pipeline.detections)
                lines = None
                if not args.no_overlay:
                    res = (frame.shape[1], frame.shape[0])
                    lines = format_overlay_lines(
                        fps.fps, fps.frame_ms, res, timer.averages(), track_summary(pipeline)
                    )
                quit_requested = window.show(frame, lines)
            fps.tick()
            if quit_requested:
                break

            if frame_period:
                next_frame_at += frame_period
                delay = next_frame_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_frame_at = time.perf_counter()  # fell behind; don't try to catch up
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        if window is not None:
            window.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
