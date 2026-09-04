"""Geometric 1D -> 2D remap handoff: stop 10 source cells before R_remap.

The remap field still extends to the requested remap radius. Only the
watchdog / capture trigger is moved inward so an undisturbed-air strip
remains between the front and the outer remap boundary.

Handoff fires when the *primary shock* reaches the existing watchdog probe
at R_handoff. That is a compression-ratio test against the known ambient,
not a fixed overpressure in Pa.

Qt-free. No full-field scan during the run.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, Optional, Sequence

REMAP_FRONT_BUFFER_CELLS_1D = 10
HANDOFF_FILENAME = "ggui_remap_handoff.json"
HANDOFF_RULE = "R_handoff = R_remap - REMAP_FRONT_BUFFER_CELLS_1D * dr_1D"
# Strong-shock discriminator: p / p_atm. Acoustic/precursor plateaus stay
# near 1; the primary blast front is a compression of this order or larger.
PRIMARY_SHOCK_COMPRESSION_RATIO = 2.0
HANDOFF_CRITERION = (
    "primary_shock_at_handoff_radius: watchdog_probe at R_handoff "
    "reaches p/p_atm >= PRIMARY_SHOCK_COMPRESSION_RATIO; "
    "R_handoff = R_remap - REMAP_FRONT_BUFFER_CELLS_1D * dr_1D"
)


class HandoffGeometryError(ValueError):
    """R_remap is too small for the fixed 10-cell front buffer."""


def physical_buffer_m(
    dr_1d: float,
    buffer_cells: int = REMAP_FRONT_BUFFER_CELLS_1D,
) -> float:
    return int(buffer_cells) * float(dr_1d)


def handoff_radius_m(
    r_remap: float,
    dr_1d: float,
    buffer_cells: int = REMAP_FRONT_BUFFER_CELLS_1D,
) -> float:
    """Return R_handoff. Does not shrink R_remap."""
    try:
        r = float(r_remap)
        dr = float(dr_1d)
        n = int(buffer_cells)
    except (TypeError, ValueError) as exc:
        raise HandoffGeometryError(
            "Remap handoff needs a positive remap radius and 1D cell size."
        ) from exc
    if not math.isfinite(r) or r <= 0.0:
        raise HandoffGeometryError("Remap radius must be a positive finite length.")
    if not math.isfinite(dr) or dr <= 0.0:
        raise HandoffGeometryError("1D cell size must be a positive finite length.")
    if n < 1:
        raise HandoffGeometryError("Remap front buffer cells must be at least 1.")
    buffer_m = physical_buffer_m(dr, n)
    if r <= buffer_m:
        raise HandoffGeometryError(
            f"Remap radius {r:g} m is too small for a {n}-cell front buffer "
            f"at dr_1D={dr:g} m (need R_remap > {buffer_m:g} m)."
        )
    return r - buffer_m


def primary_shock_compression_ratio(pressure: float, p_atm: float) -> Optional[float]:
    """Return p / p_atm, or None if the sample is not usable."""
    try:
        p = float(pressure)
        atm = float(p_atm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or not math.isfinite(atm) or atm <= 0.0 or p <= 0.0:
        return None
    return p / atm


def primary_shock_at_probe(
    pressure: float,
    p_atm: float,
    *,
    ratio: float = PRIMARY_SHOCK_COMPRESSION_RATIO,
) -> bool:
    """True when the primary shock, not a weak precursor, is at the probe.

    The existing watchdog probe already sits at R_handoff. A trailing
    positive-pressure plateau at ~1.4 atm does not satisfy this test; a
    blast front does. This is a dimensionless compression, not a kPa gate.
    """
    value = primary_shock_compression_ratio(pressure, p_atm)
    if value is None:
        return False
    try:
        need = float(ratio)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(need) or need <= 1.0:
        return False
    return value >= need


def primary_shock_arrival_time(
    times: Sequence[Any],
    pressures: Sequence[Any],
    p_atm: float,
    *,
    ratio: float = PRIMARY_SHOCK_COMPRESSION_RATIO,
) -> Optional[float]:
    """First sample time at which ``primary_shock_at_probe`` is true."""
    for raw_t, raw_p in zip(times, pressures):
        try:
            t = float(raw_t)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(t):
            continue
        if primary_shock_at_probe(raw_p, p_atm, ratio=ratio):
            return t
    return None


def leading_primary_front_radius_m(
    radii: Iterable[Any],
    pressures: Iterable[Any],
    p_atm: float,
    *,
    ratio: float = PRIMARY_SHOCK_COMPRESSION_RATIO,
) -> Optional[float]:
    """Outermost radius where the primary-shock compression is present.

    Walks from the far field inward. Used to audit a snapshot; not a
    per-step field scan.
    """
    pairs = []
    for raw_r, raw_p in zip(radii, pressures):
        try:
            r = float(raw_r)
            p = float(raw_p)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(r) or not math.isfinite(p):
            continue
        pairs.append((r, p))
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[0])
    front = None
    for r, p in pairs:
        if primary_shock_at_probe(p, p_atm, ratio=ratio):
            front = r
    return front


def uses_remap_handoff(inputs: Any) -> bool:
    """True when a 1D remap precursor should stop at R_handoff."""
    if not bool(getattr(inputs, "remap_for_2d", False)):
        return False
    mode = str(getattr(inputs, "stop_mode", "") or "").strip().lower()
    right = str(getattr(inputs, "right_boundary", "") or "").strip().lower()
    if right in ("reflect", "reflecting"):
        return False
    if mode in ("reflect", "reflecting", "end_time"):
        return False
    return True


def handoff_plan(
    r_remap: float,
    dr_1d: float,
    *,
    buffer_cells: int = REMAP_FRONT_BUFFER_CELLS_1D,
    source_1d_case: str = "",
) -> Dict[str, Any]:
    r_h = handoff_radius_m(r_remap, dr_1d, buffer_cells)
    return {
        "remap_radius_m": float(r_remap),
        "dr_1d_m": float(dr_1d),
        "remap_front_buffer_cells": int(buffer_cells),
        "physical_buffer_m": physical_buffer_m(dr_1d, buffer_cells),
        "handoff_radius_m": float(r_h),
        "handoff_rule": HANDOFF_RULE,
        "handoff_criterion": HANDOFF_CRITERION,
        "source_1d_case": str(source_1d_case or ""),
        "handoff_time_s": None,
        "target_2d_case": "",
        "hob_m": None,
        "charge_center": None,
        "actual_remap_geometry": None,
    }


def handoff_path(case_dir: str) -> str:
    return os.path.join(case_dir or "", HANDOFF_FILENAME)


def write_handoff_metadata(case_dir: str, payload: Dict[str, Any]) -> str:
    path = handoff_path(case_dir)
    os.makedirs(case_dir or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def read_handoff_metadata(case_dir: str) -> Optional[Dict[str, Any]]:
    path = handoff_path(case_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def merge_target_handoff(
    source: Optional[Dict[str, Any]],
    *,
    target_2d_case: str = "",
    hob_m: Optional[float] = None,
    charge_center: Optional[Any] = None,
    actual_remap_geometry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(source or {})
    if target_2d_case:
        payload["target_2d_case"] = str(target_2d_case)
    if hob_m is not None:
        payload["hob_m"] = float(hob_m)
    if charge_center is not None:
        payload["charge_center"] = list(charge_center)
    if actual_remap_geometry is not None:
        payload["actual_remap_geometry"] = dict(actual_remap_geometry)
    return payload
