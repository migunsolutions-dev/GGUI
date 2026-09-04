"""Parse VIPER/GGUI gauge histories and compute comparison metrics."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_viper_th(path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return (labels, times, values[n_times, n_gauges]) from a VIPER TH text file."""
    labels: List[str] = []
    times: List[float] = []
    rows: List[List[float]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.lower().startswith("time"):
                parts = line.lstrip("#").split()
                if len(parts) > 1:
                    labels = parts[1:]
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                vals = [float(p) for p in parts[1:]]
            except ValueError:
                continue
            times.append(t)
            rows.append(vals)
    if not rows:
        return labels, np.zeros(0), np.zeros((0, 0))
    arr = np.asarray(rows, dtype=float)
    if not labels:
        labels = [f"g{i}" for i in range(arr.shape[1])]
    return labels, np.asarray(times, dtype=float), arr


def histories_empty(path: str) -> bool:
    try:
        _labels, times, values = parse_viper_th(path)
    except OSError:
        return True
    return times.size == 0 or values.size == 0 or not np.any(np.isfinite(values))


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def as_overpressure(pressure: np.ndarray, p_atm: float) -> np.ndarray:
    """VIPER overpressure files are already Δp; blastFoam probes write absolute p."""
    p = np.asarray(pressure, dtype=float)
    if p.size == 0:
        return p
    if float(np.nanmin(p)) > 0.4 * float(p_atm):
        return p - float(p_atm)
    return p


def peak_overpressure(times: np.ndarray, pressure: np.ndarray, p_atm: float) -> Tuple[float, float]:
    over = as_overpressure(pressure, p_atm)
    i = int(np.nanargmax(over))
    return float(over[i]), float(times[i])


def derived_positive_impulse(
    times: np.ndarray,
    pressure: np.ndarray,
    p_atm: float,
    threshold_pa: float = 100.0,
) -> float:
    """First positive lobe of overpressure. Same integrator for VIPER and GGUI."""
    t = np.asarray(times, dtype=float)
    over = as_overpressure(pressure, p_atm)
    if t.size < 2:
        return 0.0
    hit = np.where(over >= float(threshold_pa))[0]
    if hit.size == 0:
        return 0.0
    start = int(hit[0])
    left = start
    while left > 0 and over[left - 1] > 0.0:
        left -= 1
    if left > 0:
        left -= 1
    right = start
    n = over.size
    while right + 1 < n and over[right + 1] > 0.0:
        right += 1
    if right + 1 < n:
        right += 1
    sl = slice(left, right + 1)
    if int(np.count_nonzero(np.isfinite(t[sl]))) < 2:
        return 0.0
    return _trapz(np.maximum(over[sl], 0.0), t[sl])


def arrival_time(
    times: np.ndarray,
    pressure: np.ndarray,
    p_atm: float,
    threshold_pa: float = 8000.0,
) -> Optional[float]:
    over = as_overpressure(pressure, p_atm)
    t = np.asarray(times, dtype=float)
    hit = np.where(over >= float(threshold_pa))[0]
    if hit.size == 0:
        return None
    return float(t[int(hit[0])])


def waveform_l1(
    t_a: np.ndarray,
    p_a: np.ndarray,
    t_b: np.ndarray,
    p_b: np.ndarray,
    t0: float,
    t1: float,
    p_atm: float,
) -> Optional[float]:
    """Normalized L1 of overpressure on [t0, t1] vs reference a."""
    mask_a = (t_a >= t0) & (t_a <= t1)
    if int(np.count_nonzero(mask_a)) < 4:
        return None
    grid = t_a[mask_a]
    oa = as_overpressure(p_a, p_atm)[mask_a]
    ob = np.interp(grid, t_b, as_overpressure(p_b, p_atm))
    denom = _trapz(np.abs(oa), grid)
    if denom <= 0.0:
        return None
    return _trapz(np.abs(ob - oa), grid) / denom


def pct_error(ggui: float, viper: float) -> Optional[float]:
    if viper == 0 or not math.isfinite(viper) or not math.isfinite(ggui):
        return None
    return 100.0 * (ggui - viper) / viper


def column_history(times: np.ndarray, values: np.ndarray, index: int) -> np.ndarray:
    if values.size == 0 or index >= values.shape[1]:
        return np.zeros(0)
    return values[:, index]


def load_ggui_probe(case_dir: str, dim: str, field: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    from validation.probes import PROBE_FO, latest_probe_field_file, parse_probe_history

    path = latest_probe_field_file(case_dir, PROBE_FO[dim], field)
    if not path:
        return [], np.zeros(0), np.zeros((0, 0))
    locs, times, cols = parse_probe_history(path)
    if not times or not cols:
        return locs, np.zeros(0), np.zeros((0, 0))
    arr = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    return locs, np.asarray(times, dtype=float), arr
