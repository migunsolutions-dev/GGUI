"""Temporal sampling checks for high-frequency validation pressure histories.

Probe writes are independent of VTK / native field-output frequency. A fixed
``writeInterval`` of 25 steps is not assumed adequate for impulse: adequacy is
judged against the positive-phase duration and by impulse convergence under
subsampling.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple


def first_zero_crossing(times: Sequence[float], overpressure: Sequence[float]) -> float:
    """Linear interpolation of the first post-arrival zero of ``p - p_atm``."""
    if len(times) != len(overpressure) or len(times) < 2:
        raise ValueError("times and overpressure must be the same length >= 2")
    arrived = False
    for i in range(1, len(times)):
        prev = float(overpressure[i - 1])
        cur = float(overpressure[i])
        if not arrived:
            if cur > 0.0:
                arrived = True
            continue
        if prev > 0.0 and cur <= 0.0:
            dt = float(times[i]) - float(times[i - 1])
            if dt == 0.0:
                return float(times[i])
            frac = prev / (prev - cur)
            return float(times[i - 1]) + frac * dt
    raise ValueError("no positive-phase zero crossing found")


def arrival_time(times: Sequence[float], overpressure: Sequence[float]) -> float:
    for t, dp in zip(times, overpressure):
        if float(dp) > 0.0:
            return float(t)
    raise ValueError("no arrival found")


def positive_impulse(
    times: Sequence[float],
    overpressure: Sequence[float],
    *,
    t_arrival: float | None = None,
    t_zero: float | None = None,
) -> float:
    """Trapezoidal ``I+ = integral (p - p_atm) dt`` over the positive phase."""
    t0 = arrival_time(times, overpressure) if t_arrival is None else float(t_arrival)
    t1 = first_zero_crossing(times, overpressure) if t_zero is None else float(t_zero)
    acc = 0.0
    for i in range(1, len(times)):
        ta = float(times[i - 1])
        tb = float(times[i])
        if tb <= t0 or ta >= t1:
            continue
        left = max(ta, t0)
        right = min(tb, t1)
        pa = _interp(ta, float(overpressure[i - 1]), tb, float(overpressure[i]), left)
        pb = _interp(ta, float(overpressure[i - 1]), tb, float(overpressure[i]), right)
        acc += 0.5 * (pa + pb) * (right - left)
    return acc


def samples_in_positive_phase(
    times: Sequence[float], overpressure: Sequence[float]
) -> int:
    t0 = arrival_time(times, overpressure)
    t1 = first_zero_crossing(times, overpressure)
    return sum(1 for t in times if t0 <= float(t) <= t1)


def subsample(times: Sequence[float], values: Sequence[float], stride: int):
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return times[::stride], values[::stride]


def impulse_converged(
    times: Sequence[float],
    overpressure: Sequence[float],
    *,
    strides: Iterable[int] = (2, 4),
    rel_tol: float = 0.02,
) -> Tuple[bool, float, Tuple[Tuple[int, float], ...]]:
    """True when coarsening the history by ``strides`` keeps I+ within ``rel_tol``.

    This is the check that replaces a fixed writeInterval assumption. If
    dropping every other (or every fourth) sample changes impulse by more
    than ``rel_tol``, the probe frequency is not adequate.
    """
    base = positive_impulse(times, overpressure)
    rows = []
    ok = True
    for stride in strides:
        t_s, p_s = subsample(times, overpressure, int(stride))
        if len(t_s) < 4:
            rows.append((int(stride), float("nan")))
            ok = False
            continue
        try:
            value = positive_impulse(t_s, p_s)
        except ValueError:
            rows.append((int(stride), float("nan")))
            ok = False
            continue
        rows.append((int(stride), value))
        if base == 0.0 or abs(value / base - 1.0) > rel_tol:
            ok = False
    return ok, base, tuple(rows)


def recommended_write_interval_steps(
    *,
    dt_s: float,
    positive_phase_s: float,
    min_samples: int = 50,
) -> int:
    """Largest step interval that still puts ``min_samples`` inside t+."""
    if dt_s <= 0.0 or positive_phase_s <= 0.0:
        raise ValueError("dt_s and positive_phase_s must be > 0")
    needed = max(1, int(math.floor(positive_phase_s / (dt_s * min_samples))))
    return max(1, needed)


def _interp(t0: float, v0: float, t1: float, v1: float, t: float) -> float:
    if t1 == t0:
        return v0
    frac = (t - t0) / (t1 - t0)
    return v0 + frac * (v1 - v0)
