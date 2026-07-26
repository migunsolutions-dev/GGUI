"""Numeric OpenFOAM time-directory helpers for the Cylindrical–2D viewer."""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

_SKIP_DIR_NAMES = frozenset({"constant", "system", "postProcessing"})
LIVE_FOLLOW_LABEL = "Live"


def list_numeric_time_entries(case_dir: str) -> List[Tuple[float, str]]:
    """Return ``(numeric_value, directory_label)`` sorted numerically ascending.

    Skips ``constant``, ``system``, ``processor*``, ``postProcessing``, and any
    non-numeric directory names. Labels keep the on-disk spelling.
    """
    times: List[Tuple[float, str]] = []
    try:
        for name in os.listdir(case_dir):
            path = os.path.join(case_dir, name)
            if not os.path.isdir(path):
                continue
            if name in _SKIP_DIR_NAMES or name.startswith("processor"):
                continue
            try:
                tval = float(name)
            except ValueError:
                continue
            times.append((tval, name))
    except OSError:
        return []
    times.sort(key=lambda item: item[0])
    return times


def list_numeric_time_labels(case_dir: str) -> List[str]:
    return [label for _, label in list_numeric_time_entries(case_dir)]


def pick_opening_time(entries: Sequence[Tuple[float, str]]) -> Tuple[str, float]:
    """Default viewer selection on case open: always prefer time ``0``."""
    for tval, label in entries:
        if tval == 0.0 or label == "0":
            return label, float(tval)
    return "0", 0.0


def match_reader_time_value(
    time_values: Iterable[float],
    target: float,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-15,
) -> Optional[float]:
    """Map a selected numeric time onto a PyVista/OpenFOAM reader time value."""
    values = [float(v) for v in time_values]
    if not values:
        return None
    best = min(values, key=lambda v: abs(v - float(target)))
    tol = max(abs_tol, abs(float(target)) * rel_tol, abs(best) * rel_tol)
    if abs(best - float(target)) <= tol:
        return best
    return None


def poly_mesh_dir_at_or_before(case_dir: str, time_value: float) -> Optional[str]:
    """Latest ``polyMesh`` at or before ``time_value``, else ``constant/polyMesh``."""
    best_time: Optional[float] = None
    best_path: Optional[str] = None
    try:
        for tval, name in list_numeric_time_entries(case_dir):
            if tval > float(time_value) + abs_tol_for(time_value):
                continue
            owner = os.path.join(case_dir, name, "polyMesh", "owner")
            if os.path.isfile(owner) and (best_time is None or tval >= best_time):
                best_time = tval
                best_path = os.path.join(case_dir, name, "polyMesh")
    except OSError:
        best_path = None
    if best_path is not None:
        return best_path
    const_owner = os.path.join(case_dir, "constant", "polyMesh", "owner")
    if os.path.isfile(const_owner):
        return os.path.join(case_dir, "constant", "polyMesh")
    return None


def abs_tol_for(time_value: float) -> float:
    return max(1e-15, abs(float(time_value)) * 1e-12)
