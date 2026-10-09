"""Report how well tracking holds together over a video file.

    uv run python examples/track_report.py some_clip.mp4
    uv run python examples/track_report.py some_clip.mp4 --max-age 30

Runs detect + track on every frame (no window) and prints how many IDs were
created, how long tracks lasted, and how often lost tracks were recovered.
A clip with 3 people should create roughly 3 confirmed IDs; many more means
tracks are breaking up (fragmenting).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict

from npcify.capture import FrameSource
from npcify.detect import YuNetDetector
from npcify.pipeline import Pipeline
from npcify.track import Tracker, TrackerConfig
from npcify.types import TrackState


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("video", help="path to a video file")
    p.add_argument("--det-threshold", type=float, default=0.6, help="high threshold")
    p.add_argument("--low-threshold", type=float, default=0.3)
    p.add_argument("--max-age", type=int, default=15)
    p.add_argument("--det-width", type=int, default=640)
    p.add_argument("--detect-every", type=int, default=1)
    args = p.parse_args()

    config = TrackerConfig(
        high_threshold=args.det_threshold,
        low_threshold=min(args.low_threshold, args.det_threshold),
        max_age=args.max_age,
    )
    tracker = Tracker(config)
    pipeline = Pipeline(
        detector=YuNetDetector(score_threshold=config.low_threshold, det_width=args.det_width),
        tracker=tracker,
        detect_every=max(1, args.detect_every),
    )

    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    confirmed_ids: set[int] = set()
    frames_by_state: dict[TrackState, int] = defaultdict(int)
    peak_confirmed = 0
    n_frames = 0

    with FrameSource(args.video) as source:
        print(f"{args.video}: {source.resolution[0]}x{source.resolution[1]}")
        while (frame := source.read()) is not None:
            pipeline.process_frame(frame)
            confirmed_now = 0
            for t in pipeline.tracks:
                first_seen.setdefault(t.track_id, n_frames)
                last_seen[t.track_id] = n_frames
                frames_by_state[t.state] += 1
                if t.state is TrackState.CONFIRMED:
                    confirmed_ids.add(t.track_id)
                    confirmed_now += 1
            peak_confirmed = max(peak_confirmed, confirmed_now)
            n_frames += 1

    if n_frames == 0:
        print("No frames read.", file=sys.stderr)
        return 1

    s = tracker.stats
    lengths = [last_seen[i] - first_seen[i] + 1 for i in confirmed_ids]
    track_frames = sum(frames_by_state.values()) or 1
    print(f"frames:                 {n_frames}")
    print(f"IDs created:            {s.ids_created}")
    print(f"  confirmed:            {s.ids_confirmed}")
    print(f"  never confirmed:      {s.ids_created - s.ids_confirmed} (brief or false detections)")
    print(f"most faces at once:     {peak_confirmed} (confirmed)")
    if lengths:
        print(
            f"confirmed track length: {statistics.fmean(lengths):.0f} frames avg, "
            f"{statistics.median(lengths):.0f} median, {max(lengths)} max"
        )
    print(f"went lost:              {s.lost_events} times")
    if s.lost_events:
        rate = 100 * s.recovered / s.lost_events
        print(f"  recovered:            {s.recovered} ({rate:.0f}%)")
        print(f"  expired:              {s.lost_events - s.recovered} (or still lost at the end)")
    lost_pct = 100 * frames_by_state[TrackState.LOST] / track_frames
    print(f"time spent lost:        {lost_pct:.1f}% of track-frames")
    if peak_confirmed and s.ids_confirmed > 2 * peak_confirmed:
        print(
            "\nNote: many more confirmed IDs than people on screen at once. Tracks may be "
            "fragmenting; try a larger --max-age or a lower --low-threshold."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
