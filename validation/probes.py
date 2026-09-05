"""OpenFOAM probe/gauge readers for Validation. Qt-free copies of Time History parsers."""
from __future__ import annotations

import math
import os
import re
from typing import List, Optional, Sequence, Tuple

PROBE_FO = {"1d": "gauges1d", "2d": "probes2d", "3d": "probes3d"}
VALIDATION_FO = {"1d": "validationGauges1d", "2d": "validationGauges2d"}
EXISTING_1D_GRAPH_FO = "probes1d"
_PROBE_HEADER = re.compile(r"Probe\s+(\d+)\s+\(([^)]+)\)")


def latest_probe_field_file(case_dir: str, fo_name: str, field: str) -> str:
    root = os.path.join(case_dir or "", "postProcessing", fo_name)
    if not os.path.isdir(root):
        return ""
    best_t = None
    best_path = ""
    try:
        names = os.listdir(root)
    except OSError:
        return ""
    for name in names:
        path = os.path.join(root, name, field)
        if not os.path.isfile(path):
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        if best_t is None or t >= best_t:
            best_t = t
            best_path = path
    return best_path


def parse_probe_history(path: str) -> Tuple[List[str], List[float], List[List[float]]]:
    locations: List[str] = []
    times: List[float] = []
    columns: List[List[float]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return locations, times, columns
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines = lines[:-1]
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = _PROBE_HEADER.search(line)
            if match:
                idx = int(match.group(1))
                while len(locations) <= idx:
                    locations.append("")
                locations[idx] = match.group(2).strip()
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            values = [float(part) for part in parts[1:]]
        except ValueError:
            continue
        times.append(t)
        if not columns:
            columns = [[] for _ in values]
        if len(columns) < len(values):
            columns.extend([] for _ in range(len(values) - len(columns)))
        for index, value in enumerate(values):
            columns[index].append(value)
        for extra in columns[len(values) :]:
            extra.append(float("nan"))
    return locations, times, columns


def radii_from_locations(locations: Sequence[str], *, dim: str = "1d") -> List[Optional[float]]:
    """Cartesian probe header '(x y z)' → radius used as standoff for 1D/2D."""
    out: List[Optional[float]] = []
    for loc in locations:
        xyz = xyz_from_location(loc)
        if xyz is None:
            out.append(None)
            continue
        if dim == "1d":
            out.append(radial_distance(*xyz))
        else:
            out.append(abs(float(xyz[0])))
    return out


def radial_distance(x: float, y: float = 0.0, z: float = 0.0) -> float:
    """True spherical radius. Use for 1D wedge/spherical geometry, not x alone."""
    return math.sqrt(float(x) ** 2 + float(y) ** 2 + float(z) ** 2)


def xyz_from_location(location: str) -> Optional[Tuple[float, float, float]]:
    parts = str(location or "").replace(",", " ").split()
    if not parts:
        return None
    try:
        nums = [float(p) for p in parts[:3]]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.append(0.0)
    return float(nums[0]), float(nums[1]), float(nums[2])


PROBE_MATCH_ABS_TOL = 1.0e-4
PROBE_MATCH_REL_TOL = 1.0e-5
PROBE_MISSING = "Probe is missing for this Validation Point; comparison is N/A."
PROBE_MISMATCH = "Probe location does not match the planned Validation Point; comparison is N/A."
# OpenFOAM GREAT / VGREAT are IEEE-finite (~1e300) but are unwritten sentinels.
UNPHYSICAL_PROBE_ABS = 1.0e20


def is_physical_probe_value(value: object) -> bool:
    """True for a usable probe sample; GREAT/NaN/Inf are missing data."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and abs(number) < UNPHYSICAL_PROBE_ABS


def match_probe_to_point(
    locations: Sequence[str],
    point_xyz: Tuple[float, float, float],
    *,
    abs_tol: float = PROBE_MATCH_ABS_TOL,
    rel_tol: float = PROBE_MATCH_REL_TOL,
) -> Tuple[Optional[int], str]:
    """Match a planned Validation Point to a probe header by coordinates, not index.

    Returns (column_index, reason). reason is empty on success.
    """
    target = (float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]))
    hits: List[int] = []
    for index, loc in enumerate(locations):
        xyz = xyz_from_location(loc)
        if xyz is None:
            continue
        if _xyz_close(xyz, target, abs_tol=abs_tol, rel_tol=rel_tol):
            hits.append(index)
    if not hits:
        if not any(str(loc).strip() for loc in locations):
            return None, PROBE_MISSING
        return None, PROBE_MISMATCH
    if len(hits) > 1:
        return None, PROBE_MISMATCH
    return hits[0], ""


def _xyz_close(
    left: Tuple[float, float, float],
    right: Tuple[float, float, float],
    *,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    for a, b in zip(left, right):
        scale = max(abs(float(a)), abs(float(b)), 1.0)
        if abs(float(a) - float(b)) > max(abs_tol, rel_tol * scale):
            return False
    return True


def series_for_index(
    times: Sequence[float], columns: Sequence[Sequence[float]], index: int
) -> Tuple[List[float], List[float]]:
    if index < 0 or index >= len(columns):
        return [], []
    values = list(columns[index])
    out_t: List[float] = []
    out_v: List[float] = []
    for t, v in zip(times, values):
        try:
            tf = float(t)
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(tf) and is_physical_probe_value(vf):
            out_t.append(tf)
            out_v.append(vf)
    return out_t, out_v


def peak_and_impulse(
    times: Sequence[float],
    pressure_pa: Sequence[float],
    impulse_pa_s: Optional[Sequence[float]] = None,
    *,
    p_atm: float = 101325.0,
) -> Tuple[Optional[float], Optional[float]]:
    """Peak overpressure (Pa) and last positive-phase impulse (Pa·s) if provided."""
    over = [float(p) - float(p_atm) for p in pressure_pa if is_physical_probe_value(p)]
    peak = max(over) if over else None
    impulse = None
    if impulse_pa_s:
        finite = [float(v) for v in impulse_pa_s if is_physical_probe_value(v)]
        if finite:
            impulse = finite[-1]
    return peak, impulse


def standoff_m(gauge_xyz: Tuple[float, float, float], charge_xyz: Tuple[float, float, float]) -> float:
    dx = float(gauge_xyz[0]) - float(charge_xyz[0])
    dy = float(gauge_xyz[1]) - float(charge_xyz[1])
    dz = float(gauge_xyz[2]) - float(charge_xyz[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)
