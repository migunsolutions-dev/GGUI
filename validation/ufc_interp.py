"""Controlled piecewise-linear interpolation. Never extrapolates."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple


def interp1d(x: float, xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Linear interpolation on a monotone independent-variable table. None if outside."""
    if not xs or len(xs) != len(ys):
        return None
    if len(xs) == 1:
        return float(ys[0]) if x == xs[0] else None
    x0 = float(xs[0])
    x1 = float(xs[-1])
    increasing = x1 >= x0
    lo_b, hi_b = (x0, x1) if increasing else (x1, x0)
    xv = float(x)
    if xv < lo_b or xv > hi_b:
        return None
    n = len(xs)
    lo, hi = 0, n - 1
    if increasing:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if float(xs[mid]) <= xv:
                lo = mid
            else:
                hi = mid
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if float(xs[mid]) >= xv:
                lo = mid
            else:
                hi = mid
    x_lo = float(xs[lo])
    x_hi = float(xs[hi])
    y_lo = float(ys[lo])
    y_hi = float(ys[hi])
    if x_hi == x_lo:
        return y_lo
    t = (xv - x_lo) / (x_hi - x_lo)
    return y_lo + t * (y_hi - y_lo)


def bracketing_index(value: float, keys: Sequence[float]) -> Optional[Tuple[int, int, float]]:
    """Return (i0, i1, t) with keys[i0] <= value <= keys[i1]. None if outside."""
    if not keys:
        return None
    xv = float(value)
    if xv < float(keys[0]) or xv > float(keys[-1]):
        return None
    if len(keys) == 1:
        return (0, 0, 0.0)
    for i, key in enumerate(keys):
        span = max(abs(float(key)), 1.0)
        if abs(xv - float(key)) <= 1e-12 * span:
            return i, i, 0.0
    lo, hi = 0, len(keys) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(keys[mid]) <= xv:
            lo = mid
        else:
            hi = mid
    span = float(keys[hi]) - float(keys[lo])
    t = 0.0 if span == 0.0 else (xv - float(keys[lo])) / span
    return lo, hi, t
