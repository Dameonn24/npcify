"""Measure how fast the pipeline runs.

FPSCounter gives a smoothed frames-per-second figure. StageTimer times named
sections of the loop (e.g. "capture", "process", "display") so we can see
which part is slow.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager

Clock = Callable[[], float]


class FPSCounter:
    """Rolling-average FPS over the last `window` frames."""

    def __init__(self, window: int = 30, clock: Clock = time.perf_counter) -> None:
        # window frames means window intervals, which needs window + 1 timestamps.
        self._stamps: deque[float] = deque(maxlen=window + 1)
        self._clock = clock

    def tick(self, now: float | None = None) -> None:
        """Mark that a frame has been completed. `now` is in seconds."""
        self._stamps.append(self._clock() if now is None else now)

    @property
    def fps(self) -> float:
        """Smoothed frames per second, or 0.0 until two frames have been seen."""
        if len(self._stamps) < 2:
            return 0.0
        elapsed = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def frame_ms(self) -> float:
        """Average time per frame in milliseconds, or 0.0 if unknown."""
        fps = self.fps
        return 1000.0 / fps if fps > 0 else 0.0


class StageTimer:
    """Times named sections; reports a rolling average in milliseconds per stage.

    Usage:
        with timer.stage("capture"):
            frame = source.read()
    """

    def __init__(self, window: int = 30, clock: Clock = time.perf_counter) -> None:
        self._window = window
        self._clock = clock
        self._samples: dict[str, deque[float]] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            self.record(name, (self._clock() - start) * 1000.0)

    def record(self, name: str, ms: float) -> None:
        """Add one measurement (in milliseconds) for a stage."""
        self._samples.setdefault(name, deque(maxlen=self._window)).append(ms)

    def averages(self) -> dict[str, float]:
        """Average milliseconds per stage, in the order stages were first seen."""
        return {name: sum(s) / len(s) for name, s in self._samples.items()}
