"""Live webcam (or video file) through the npcify pipeline.

Every face is hidden except people on the keep list.

    uv run python examples/webcam_demo.py
    uv run python examples/webcam_demo.py --video some_clip.mp4

Mouse:
    left-click a face            keep that person real (look at the camera ~1 s)
    right/shift/ctrl-click face  hide that person again (remove from keep list)
Keys:
    q / Esc   quit
    c         clear the keep list
    d         toggle debug boxes, track IDs and keep/hide status
    s         cycle anonymisation style
    space     pause / resume
    n         when paused, step forward one frame (to check for leaks)

There is deliberately no key to turn anonymisation off.
"""

from __future__ import annotations

import argparse
import sys
import time

from npcify import Pipeline
from npcify.capture import CameraError, FrameSource
from npcify.debug import TrackTrails, draw_face_states, draw_tracks
from npcify.decide import KeepListPolicy
from npcify.detect import DEFAULT_DET_WIDTH, YuNetDetector
from npcify.models import ModelDownloadError
from npcify.output import PreviewWindow, format_overlay_lines
from npcify.pose import PoseEstimator
from npcify.recognise import DEFAULT_MATCH_THRESHOLD, KeepList, SFaceRecognizer
from npcify.render import STYLES, Anonymizer, Padding
from npcify.timing import FPSCounter, StageTimer
from npcify.track import Tracker, TrackerConfig
from npcify.types import Track, TrackState

_DEFAULT_PADDING = Padding()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--video", help="path to a video file (overrides --camera)")
    p.add_argument("--width", type=int, default=1280, help="requested width (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="requested height (default: 720)")
    p.add_argument("--no-mirror", action="store_true", help="don't flip the webcam image")
    p.add_argument("--no-overlay", action="store_true", help="hide the FPS overlay")
    p.add_argument(
        "--style",
        choices=STYLES,
        default=STYLES[0],
        help=f"how faces are hidden (default: {STYLES[0]})",
    )
    p.add_argument(
        "--padding",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=(_DEFAULT_PADDING.width, _DEFAULT_PADDING.height),
        help="extra width and height around each face, as fractions "
        f"(default: {_DEFAULT_PADDING.width} {_DEFAULT_PADDING.height})",
    )
    p.add_argument(
        "--debug", action="store_true", help="start with debug boxes and IDs shown (key: d)"
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
        help="lowest confidence that is still covered and keeps a track alive (default: 0.3)",
    )
    p.add_argument(
        "--max-age",
        type=int,
        default=TrackerConfig.max_age,
        help=f"frames to keep covering a face the detector lost (default: {TrackerConfig.max_age})",
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
        help="run detection every N frames (default: 1). N > 1 can leave new faces "
        "uncovered for up to N-1 frames; for benchmarking only",
    )
    p.add_argument(
        "--det-width",
        type=int,
        default=DEFAULT_DET_WIDTH,
        help="width detection runs at; larger finds smaller faces but is slower "
        f"(default: {DEFAULT_DET_WIDTH})",
    )
    p.add_argument(
        "--debug-mesh",
        action="store_true",
        help="in debug view, also draw the full face landmark mesh for hidden faces",
    )
    p.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="how similar a face must be to a kept person to be revealed; higher is "
        f"stricter (default: {DEFAULT_MATCH_THRESHOLD})",
    )
    p.add_argument("--keep-list", metavar="PATH", help="load a saved keep list (.npz)")
    p.add_argument(
        "--save-keep-list",
        metavar="PATH",
        help="save the keep list's faceprints (biometric data, no images) to PATH on exit",
    )
    p.add_argument(
        "--debug-no-anonymise",
        action="store_true",
        help="DEVELOPMENT ONLY: show faces unhidden. Never stream or record with this.",
    )
    return p.parse_args()


def track_summary(pipeline: Pipeline) -> list[str]:
    def count(state: TrackState) -> int:
        return sum(t.state is state for t in pipeline.tracks)

    return [
        f"faces: {len(pipeline.detections)}  covered: {len(pipeline.regions)}",
        f"  {count(TrackState.CONFIRMED)} confirmed, {count(TrackState.TENTATIVE)} tentative, "
        f"{count(TrackState.LOST)} lost",
    ]


def track_at(tracks: list[Track], x: int, y: int) -> Track | None:
    """The smallest track whose (slightly enlarged) box contains the point."""
    hits = []
    for t in tracks:
        b = t.box
        mx, my = 0.2 * b.width, 0.2 * b.height
        if b.x - mx <= x <= b.x + b.width + mx and b.y - my <= y <= b.y + b.height + my:
            hits.append((b.width * b.height, t))
    return min(hits, key=lambda h: h[0])[1] if hits else None


def keep_summary(policy: KeepListPolicy) -> list[str]:
    lines = []
    if len(policy.keep_list):
        lines.append("Keeping: " + ", ".join(policy.keep_list.labels))
    e = policy.enrolment
    if e is not None:
        hint = policy.enrolment_hint or "face the camera, move your head a little"
        lines.append(f"Enrolling {e.label}: {len(e.faceprints)}/{e.target} ({hint})")
    return lines


def main() -> int:
    args = parse_args()
    source_arg = args.video if args.video else args.camera
    mirror = False if args.no_mirror else None  # None = default for the source type

    if args.debug_no_anonymise:
        banner = "!" * 72
        print(
            f"\n{banner}\n  ANONYMISATION IS OFF (--debug-no-anonymise).\n"
            f"  Faces are VISIBLE. Do not stream, share or record this window.\n{banner}\n",
            file=sys.stderr,
        )
    if args.detect_every > 1:
        print(
            f"Warning: --detect-every {args.detect_every}: a face that appears between "
            "detection frames is not covered until the next one.",
            file=sys.stderr,
        )

    config = TrackerConfig(
        high_threshold=args.det_threshold,
        low_threshold=min(args.low_threshold, args.det_threshold),
        max_age=args.max_age,
        recover_distance=args.recover_distance,
    )
    if args.save_keep_list:
        print(
            f"WARNING: on exit, the faceprints of everyone on the keep list will be saved to "
            f"{args.save_keep_list}. Faceprints are biometric data: keep that file private.",
            file=sys.stderr,
        )
    try:
        # The detector runs at the low threshold so weak detections are covered too.
        detector = YuNetDetector(score_threshold=config.low_threshold, det_width=args.det_width)
        recogniser = SFaceRecognizer()
        pose = PoseEstimator()
    except ModelDownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if args.keep_list:
        try:
            keep_list = KeepList.load(args.keep_list, threshold=args.match_threshold)
        except (OSError, KeyError, ValueError) as e:
            print(f"Error: could not load keep list {args.keep_list}: {e}", file=sys.stderr)
            return 1
        print(f"Loaded keep list: {', '.join(keep_list.labels) or '(empty)'}")
    else:
        keep_list = KeepList(threshold=args.match_threshold)

    try:
        source = FrameSource(source_arg, args.width, args.height, mirror=mirror)
    except (CameraError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    padding = Padding(args.padding[0], args.padding[1], _DEFAULT_PADDING.shift_up)
    style = args.style
    fps = FPSCounter()
    timer = StageTimer()
    policy = KeepListPolicy(recogniser, keep_list, timer=timer)
    pipeline = Pipeline(
        detector=detector,
        tracker=Tracker(config),
        policy=policy,
        pose=pose,
        anonymizer=Anonymizer(style=style, padding=padding),
        timer=timer,
        detect_every=max(1, args.detect_every),
        debug_no_anonymise=args.debug_no_anonymise,
    )
    trails = TrackTrails()
    window: PreviewWindow | None = None

    w, h = source.resolution
    if source.is_camera and (w, h) != (args.width, args.height):
        print(f"Requested {args.width}x{args.height}, camera gave {w}x{h}.")
    else:
        print(f"Resolution: {w}x{h}")
    print("Click a face to keep it real; right/shift-click to hide it again.")
    print("Keys: q quit, c clear keep list, d debug, s style, space pause, n next frame")

    # Play video files at their native speed rather than as fast as possible.
    frame_period = 1.0 / source.source_fps if not source.is_camera and source.source_fps else 0.0
    next_frame_at = time.perf_counter()

    show_debug = args.debug
    paused = False
    step = False
    raw = None  # the latest input frame (kept so a style change can re-render when paused)
    output = None

    try:
        window = PreviewWindow()
        while True:
            if not paused or step:
                step = False
                with timer.stage("capture"):
                    raw = source.read()
                if raw is None:
                    if source.is_camera:
                        print("Camera stopped delivering frames.", file=sys.stderr)
                    break
                with timer.stage("process"):
                    output = pipeline.process_frame(raw)
                trails.update(pipeline.tracks)
                fps.tick()

            with timer.stage("display"):
                shown = output
                if show_debug:
                    notes = {t.track_id: policy.status_text(t.track_id) for t in pipeline.tracks}
                    shown = draw_tracks(shown, pipeline.tracks, trails, pipeline.detections, notes)
                    shown = draw_face_states(
                        shown, pipeline.tracks, pipeline.faces, mesh=args.debug_mesh
                    )
                lines = None
                if not args.no_overlay:
                    res = (shown.shape[1], shown.shape[0])
                    extra = [
                        *track_summary(pipeline),
                        f"style: {style}  (s to change)",
                        *keep_summary(policy),
                    ]
                    if paused:
                        extra.append("PAUSED  (space resume, n next frame)")
                    if args.debug_no_anonymise:
                        extra.append("!! ANONYMISATION OFF !!")
                    lines = format_overlay_lines(
                        fps.fps, fps.frame_ms, res, timer.averages(), extra
                    )
                key = window.show(shown, lines, wait_ms=30 if paused else 1)

            for x, y, button in window.pop_clicks():
                clicked = track_at(pipeline.tracks, x, y)
                if clicked is None:
                    continue
                if button == "left":
                    status = policy.statuses.get(clicked.track_id)
                    if status is not None and status.keep:
                        print(f"Already keeping {status.kept_as}.")
                    else:
                        label = policy.enrol(clicked.track_id)
                        print(f"Enrolling {label} (track #{clicked.track_id})...")
                else:
                    label = policy.forget_track_person(clicked.track_id)
                    if label:
                        print(f"Removed {label} from the keep list.")

            if key == "quit":
                break
            if key == "c":
                policy.clear()
                print("Keep list cleared.")
            elif key == "d":
                show_debug = not show_debug
            elif key == "s":
                style = STYLES[(STYLES.index(style) + 1) % len(STYLES)]
                pipeline.anonymizer = Anonymizer(style=style, padding=padding)
                if paused and raw is not None:
                    output = pipeline.anonymise(raw)
            elif key == " ":
                paused = not paused
                next_frame_at = time.perf_counter()
            elif key == "n" and paused:
                step = True

            if frame_period and not paused:
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
        if args.save_keep_list:
            keep_list.save(args.save_keep_list)
            print(
                f"Saved faceprints for {len(keep_list)} kept people to {args.save_keep_list} "
                "(biometric data; no images).",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
