"""Shared Validation error metrics. SI values in, dimensionless errors out."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

NEAR_ZERO_REL = 1e-12
NEAR_ZERO_ABS = 1e-30


def is_finite_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def relative_error_percent(
    blastfoam: Optional[float],
    reference: Optional[float],
    *,
    abs_floor: float = NEAR_ZERO_ABS,
    rel_floor: float = NEAR_ZERO_REL,
) -> Optional[float]:
    """(BF - Reference) / Reference * 100. None if either side is unusable."""
    if not is_finite_number(blastfoam) or not is_finite_number(reference):
        return None
    bf = float(blastfoam)
    ref = float(reference)
    if abs(ref) <= max(abs_floor, rel_floor * max(abs(bf), 1.0)):
        return None
    return (bf - ref) / ref * 100.0


def difference(blastfoam: Optional[float], reference: Optional[float]) -> Optional[float]:
    if not is_finite_number(blastfoam) or not is_finite_number(reference):
        return None
    return float(blastfoam) - float(reference)


def _finite_pairs(
    left: Sequence[float], right: Sequence[float]
) -> Tuple[Tuple[float, float], ...]:
    pairs = []
    for a, b in zip(left, right):
        if is_finite_number(a) and is_finite_number(b):
            pairs.append((float(a), float(b)))
    return tuple(pairs)


def rms_error(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    pairs = _finite_pairs(left, right)
    if not pairs:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))


def mean_absolute_error(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    pairs = _finite_pairs(left, right)
    if not pairs:
        return None
    return sum(abs(a - b) for a, b in pairs) / len(pairs)


def max_absolute_error(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    pairs = _finite_pairs(left, right)
    if not pairs:
        return None
    return max(abs(a - b) for a, b in pairs)


def max_meaningful_relative_error(
    left: Sequence[float],
    right: Sequence[float],
    *,
    abs_floor: float = NEAR_ZERO_ABS,
    rel_floor: float = 1e-6,
) -> Optional[float]:
    """Maximum |(L-R)/R| excluding near-zero references. Returns a fraction, not percent."""
    best: Optional[float] = None
    for a, b in _finite_pairs(left, right):
        if abs(b) <= max(abs_floor, rel_floor * max(abs(a), 1.0)):
            continue
        value = abs((a - b) / b)
        if best is None or value > best:
            best = value
    return best


def reject_nonfinite(values: Iterable[float]) -> Tuple[float, ...]:
    return tuple(float(v) for v in values if is_finite_number(v))
