"""Benchmark a face detector over a video file.

    uv run python examples/benchmark_detect.py some_clip.mp4
    uv run python examples/benchmark_detect.py some_clip.mp4 --det-width 960

Times detection only (not video decoding) and prints average/p95 detection
time, the FPS that implies, and the average number of faces per frame.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from npcify.capture import FrameSource
from npcify.detect import YuNetDetector


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("video", help="path to a video file")
    p.add_argument("--det-threshold", type=float, default=0.6)
    p.add_argument("--det-width", type=int, default=640)
    p.add_argument("--warmup", type=int, default=5, help="frames to run before timing")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    args = p.parse_args()

    detector = YuNetDetector(score_threshold=args.det_threshold, det_width=args.det_width)
    times_ms: list[float] = []
    face_counts: list[int] = []

    with FrameSource(args.video) as source:
        print(f"{args.video}: {source.resolution[0]}x{source.resolution[1]}")
        i = 0
        while (frame := source.read()) is not None:
            start = time.perf_counter()
            detections = detector.detect(frame)
            elapsed = (time.perf_counter() - start) * 1000.0
            i += 1
            if i <= args.warmup:
                continue
            times_ms.append(elapsed)
            face_counts.append(len(detections))
            if args.max_frames and len(times_ms) >= args.max_frames:
                break

    if not times_ms:
        print("Not enough frames to benchmark.", file=sys.stderr)
        return 1

    avg = statistics.fmean(times_ms)
    p95 = sorted(times_ms)[int(0.95 * (len(times_ms) - 1))]
    print(f"detector:      YuNet (threshold {args.det_threshold}, det width {args.det_width})")
    print(f"frames timed:  {len(times_ms)} (after {args.warmup} warm-up)")
    print(f"detect time:   {avg:.2f} ms avg, {p95:.2f} ms p95")
    print(f"detect FPS:    {1000.0 / avg:.1f}")
    print(f"faces/frame:   {statistics.fmean(face_counts):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
