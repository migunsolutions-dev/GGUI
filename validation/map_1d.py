"""Map automatic 1D Validation Points onto probes1d histories.

Previous behaviour (defective)
------------------------------
Each VAL_1D radius was assigned the *nearest* existing probes1d column.
Linearly spaced visualization probes are coarser than log-spaced VAL
radii in the near field, so two VAL IDs could share one source history.

Required behaviour
------------------
1. New cases insert the exact VAL radii into probes1d (generator_1d).
   Read-back uses an exact radius match.

2. Completed cases that lack those extra locations use linear spatial
   interpolation of the shared-time probe series between bracketing
   radii. Peak pressure is taken from the interpolated p(t). Impulse
   uses the interpolated blastFoam impulse series when present (not a
   Python integral of p). No extrapolation.

Do not silently alias multiple VAL radii onto one nearest probe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from validation.metrics import is_finite_number
from validation.probes import peak_and_impulse, series_for_index

KIND_EXACT = "exact_probe"
KIND_INTERP = "spatial_interp"
KIND_NONE = "none"

EXACT_ABS_TOL = 1.0e-7
EXACT_REL_TOL = 1.0e-5


def radii_close(a: float, b: float) -> bool:
    if not is_finite_number(a) or not is_finite_number(b):
        return False
    scale = max(abs(float(a)), abs(float(b)), 1.0e-12)
    return abs(float(a) - float(b)) <= max(EXACT_ABS_TOL, EXACT_REL_TOL * scale)


def merge_radii(
    base: Sequence[float],
    extra: Sequence[float],
    *,
    r_lo: float,
    r_hi: float,
) -> Tuple[float, ...]:
    """Sorted unique radii in [r_lo, r_hi], extras kept even when near a base sample."""
    lo = float(r_lo)
    hi = float(r_hi)
    out: List[float] = []

    def _add(raw: float, *, replace_close: bool) -> None:
        if not is_finite_number(raw):
            return
        value = float(raw)
        if value < lo or value > hi:
            return
        for i, existing in enumerate(out):
            if radii_close(existing, value):
                if replace_close:
                    out[i] = value
                return
        out.append(value)

    for item in base:
        _add(item, replace_close=False)
    for item in extra:
        _add(item, replace_close=True)
    out.sort()
    return tuple(out)


@dataclass(frozen=True)
class RadiusMap:
    kind: str
    target_m: float
    index_lo: Optional[int] = None
    index_hi: Optional[int] = None
    r_lo: Optional[float] = None
    r_hi: Optional[float] = None
    weight: Optional[float] = None
    gap_m: Optional[float] = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.kind in (KIND_EXACT, KIND_INTERP)


def map_radius(radii: Sequence[Optional[float]], target: float) -> RadiusMap:
    """Exact match, or bracketing interpolation. Never nearest-neighbour aliasing."""
    if not is_finite_number(target) or float(target) <= 0.0:
        return RadiusMap(kind=KIND_NONE, target_m=float(target) if is_finite_number(target) else 0.0, reason="Target radius is not a positive finite value.")
    t = float(target)
    indexed: List[Tuple[int, float]] = []
    for i, raw in enumerate(radii):
        if is_finite_number(raw) and float(raw) > 0.0:
            indexed.append((i, float(raw)))
    if not indexed:
        return RadiusMap(kind=KIND_NONE, target_m=t, reason="No probes1d radii are available.")
    for i, r in indexed:
        if radii_close(r, t):
            return RadiusMap(
                kind=KIND_EXACT,
                target_m=t,
                index_lo=i,
                index_hi=i,
                r_lo=r,
                r_hi=r,
                weight=0.0,
                gap_m=0.0,
            )
    indexed.sort(key=lambda item: item[1])
    if t < indexed[0][1] or t > indexed[-1][1]:
        return RadiusMap(
            kind=KIND_NONE,
            target_m=t,
            reason="Requested radius is outside the probes1d range (no extrapolation).",
        )
    for (i0, r0), (i1, r1) in zip(indexed, indexed[1:]):
        if r0 <= t <= r1:
            span = r1 - r0
            if span <= 0.0:
                continue
            weight = (t - r0) / span
            return RadiusMap(
                kind=KIND_INTERP,
                target_m=t,
                index_lo=i0,
                index_hi=i1,
                r_lo=r0,
                r_hi=r1,
                weight=weight,
                gap_m=span,
            )
    return RadiusMap(kind=KIND_NONE, target_m=t, reason="No bracketing probes1d pair for this radius.")


def _blend(a: Sequence[float], b: Sequence[float], weight: float) -> List[float]:
    n = min(len(a), len(b))
    w = float(weight)
    return [float(a[i]) + w * (float(b[i]) - float(a[i])) for i in range(n)]


def mapped_peak_impulse(
    mapping: RadiusMap,
    times: Sequence[float],
    pressure_cols: Sequence[Sequence[float]],
    impulse_cols: Optional[Sequence[Sequence[float]]] = None,
    *,
    p_atm: float = 101325.0,
) -> Tuple[Optional[float], Optional[float], List[float], List[float]]:
    """Return (peak_overpressure_Pa, impulse_Pa_s, times, interpolated_p)."""
    if not mapping.ok or mapping.index_lo is None:
        return None, None, [], []
    t_lo, p_lo = series_for_index(times, pressure_cols, mapping.index_lo)
    if mapping.kind == KIND_EXACT:
        impulse_series = None
        if impulse_cols is not None:
            _it, ivals = series_for_index(times, impulse_cols, mapping.index_lo)
            impulse_series = ivals or None
        peak, impl = peak_and_impulse(t_lo, p_lo, impulse_series, p_atm=p_atm)
        if impulse_series:
            impl = impulse_series[-1]
        return peak, impl, t_lo, p_lo
    t_hi, p_hi = series_for_index(times, pressure_cols, int(mapping.index_hi))
    n = min(len(t_lo), len(t_hi), len(p_lo), len(p_hi))
    if n == 0 or mapping.weight is None:
        return None, None, [], []
    t_use = list(t_lo[:n])
    p_blend = _blend(p_lo[:n], p_hi[:n], mapping.weight)
    impulse_blend = None
    if impulse_cols is not None:
        _it0, i_lo = series_for_index(times, impulse_cols, mapping.index_lo)
        _it1, i_hi = series_for_index(times, impulse_cols, int(mapping.index_hi))
        m = min(len(i_lo), len(i_hi), n)
        if m:
            impulse_blend = _blend(i_lo[:m], i_hi[:m], mapping.weight)
    peak, impl = peak_and_impulse(t_use, p_blend, impulse_blend, p_atm=p_atm)
    if impulse_blend:
        impl = impulse_blend[-1]
    return peak, impl, t_use, p_blend
