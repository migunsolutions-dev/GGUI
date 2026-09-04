"""HOB / single-reflection feature extraction from a 2D pressure snapshot.

Triple-point tracking uses the instantaneous pressure field and |grad(p)|.
Peak-overpressure maps are never used as the TP source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from validation.metrics import is_finite_number


@dataclass(frozen=True)
class ShockFronts:
    r: np.ndarray
    z: np.ndarray
    pressure: np.ndarray
    grad_mag: np.ndarray
    r_shock: np.ndarray
    z_shock: np.ndarray
    triple_point: Optional[Tuple[float, float]]
    mach_stem_height: Optional[float]
    z_ground: float
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class TriplePointSample:
    time_s: float
    x_tp: Optional[float]
    z_tp: Optional[float]
    hm: Optional[float]
    reason: str = ""


def _regular_grid(
    r: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    *,
    n_r: int = 80,
    n_z: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    mask = np.isfinite(r) & np.isfinite(z) & np.isfinite(p)
    r, z, p = r[mask], z[mask], p[mask]
    if r.size < 8:
        raise ValueError("Not enough finite (r,z,p) samples for HOB extraction.")
    r_edges = np.linspace(float(r.min()), float(r.max()), n_r + 1)
    z_edges = np.linspace(float(z.min()), float(z.max()), n_z + 1)
    sums, _, _ = np.histogram2d(r, z, bins=(r_edges, z_edges), weights=p)
    counts, _, _ = np.histogram2d(r, z, bins=(r_edges, z_edges))
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
    return r_cent, z_cent, grid


def extract_fronts(
    r: Sequence[float],
    z: Sequence[float],
    pressure: Sequence[float],
    *,
    z_ground: Optional[float] = None,
    min_confidence: float = 0.25,
) -> ShockFronts:
    r_c, z_c, grid = _regular_grid(np.asarray(r), np.asarray(z), np.asarray(pressure))
    filled = np.array(grid, copy=True)
    nan = ~np.isfinite(filled)
    if nan.any():
        fill = np.nanmedian(filled)
        if not np.isfinite(fill):
            fill = 0.0
        filled[nan] = fill
    gr, gz = np.gradient(filled, r_c, z_c, edge_order=1)
    mag = np.hypot(gr, gz)
    peak = float(np.nanmax(mag)) if mag.size else 0.0
    p_span = float(np.nanmax(filled) - np.nanmin(filled)) if filled.size else 0.0
    z_g = float(z_ground) if is_finite_number(z_ground) else float(np.nanmin(z_c))
    r_shock = np.full(z_c.shape, np.nan)
    for j, _zj in enumerate(z_c):
        col = mag[:, j]
        if not np.isfinite(col).any():
            continue
        i = int(np.nanargmax(col))
        r_shock[j] = r_c[i]
    # Triple point: strongest change of shock-front slope above ground.
    tp = None
    hm = None
    reason = ""
    confidence = 0.0
    finite = np.isfinite(r_shock)
    if peak <= max(1e-6 * max(p_span, 1.0), 1e-12) or finite.sum() < 6:
        reason = "Shock indicator |grad(p)| is too weak for a confident front."
    else:
        z_f = z_c[finite]
        r_f = r_shock[finite]
        dr_dz = np.gradient(r_f, z_f, edge_order=1)
        d2 = np.gradient(dr_dz, z_f, edge_order=1)
        above = z_f > z_g + 0.02 * max(float(z_c.max() - z_g), 1e-6)
        if not np.any(above):
            reason = "No shock-front samples above the ground plane."
        else:
            idx_local = int(np.nanargmax(np.abs(d2[above])))
            z_cand = z_f[above][idx_local]
            r_cand = r_f[above][idx_local]
            span = max(float(np.nanmax(np.abs(d2[above]))), 1e-30)
            confidence = float(min(1.0, abs(d2[above][idx_local]) / span))
            # Prefer a kink that is not at the domain edge.
            if z_cand <= z_g or z_cand >= 0.98 * float(z_c.max()):
                reason = "Detected kink is at the domain boundary; triple point not claimed."
                confidence = 0.0
            elif confidence < min_confidence:
                reason = "Triple-point curvature confidence is below threshold."
            else:
                tp = (float(r_cand), float(z_cand))
                hm = float(z_cand - z_g)
    rr, zz = np.meshgrid(r_c, z_c, indexing="ij")
    return ShockFronts(
        r=rr,
        z=zz,
        pressure=filled,
        grad_mag=mag,
        r_shock=r_shock,
        z_shock=z_c,
        triple_point=tp,
        mach_stem_height=hm,
        z_ground=z_g,
        confidence=confidence,
        reason=reason,
    )


def trajectory(
    samples: Sequence[TriplePointSample],
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    times: List[float] = []
    xs: List[float] = []
    hs: List[float] = []
    for sample in samples:
        if sample.x_tp is None or sample.hm is None:
            continue
        times.append(float(sample.time_s))
        xs.append(float(sample.x_tp))
        hs.append(float(sample.hm))
    return tuple(times), tuple(xs), tuple(hs)


def image_source_reflected_arrival(
    *,
    source_xyz: Tuple[float, float, float],
    observer_xyz: Tuple[float, float, float],
    z_ground: float,
    shock_speed: float,
) -> Optional[float]:
    """Geometric image-source path length / shock speed. Not a pressure model."""
    if not is_finite_number(shock_speed) or float(shock_speed) <= 0.0:
        return None
    sx, sy, sz = (float(v) for v in source_xyz)
    ox, oy, oz = (float(v) for v in observer_xyz)
    image_z = 2.0 * float(z_ground) - sz
    dist = math_dist((sx, sy, image_z), (ox, oy, oz))
    return dist / float(shock_speed)


def math_dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))
