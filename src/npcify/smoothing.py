"""One Euro filter: smooths noisy values without making movement feel laggy.

A plain moving average has to choose between "smooth but slow" and "fast but
jittery". The One Euro filter changes how much it smooths based on how fast
the value is changing: heavy smoothing when the value is still (no jitter),
light smoothing when it moves (no lag).

Two settings matter:
    min_cutoff: smoothing when still, in Hz. Lower = steadier at rest but
        slower to start moving. Raise it if still faces drift or lag; lower it
        if they jitter.
    beta: how quickly smoothing backs off as speed increases. Higher = follows
        fast movements (a quick head turn, a blink) more tightly, at the cost
        of a little more jitter during movement. Its scale depends on the
        units of the value (degrees vs 0-1 expressions), so each kind of value
        gets its own filter.

Reference: Casiez, Roussel & Vogel, "1€ Filter", CHI 2012.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _alpha(cutoff: FloatArray | float, dt: float) -> FloatArray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filters a vector of values; each element adapts independently."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: FloatArray | None = None
        self._dx: FloatArray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: NDArray[np.floating] | list[float], t: float) -> FloatArray:
        """Filter one sample taken at time `t` (seconds) and return the smoothed value."""
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or self._dx is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 0:
            return self._x.copy()
        dx = (x - self._x) / dt
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        self._x = self._x + _alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x.copy()
