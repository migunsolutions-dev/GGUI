"""Common KB propagation geometry: physical R and remap-region exclusion.

Kingery-Bulmash is one reference. Numerical series are BF 1D / BF 2D / BF 3D
plotted at independently computed physical standoff. Copied remap data is not
credited as independent target-solver propagation.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

from validation.metrics import is_finite_number
from validation.probes import radial_distance, standoff_m
from validation.ufc_airblast import scaled_distance

CLASS_INSIDE = "inside"
CLASS_ON_BOUNDARY = "on_boundary"
CLASS_OUTSIDE = "outside"

SERIES_1D = "BF 1D"
SERIES_2D = "BF 2D"
SERIES_3D = "BF 3D"
PLANNED_LABEL = "Planned"


def _positive(value: Any) -> Optional[float]:
    if not is_finite_number(value):
        return None
    number = float(value)
    if number <= 0.0:
        return None
    return number


def _from_remap_dict(data: Optional[Dict[str, Any]]) -> Optional[float]:
    """Physical remap radius: the user-defined incident-front limit.

    Numerical source-mesh padding (field_r_max) must not enlarge Validation
    exclusion. Never infer the region from the target-domain size.
    """
    if not isinstance(data, dict):
        return None
    geometry = data.get("actual_remap_geometry")
    if isinstance(geometry, dict):
        for key in (
            "requested_mapped_radius_m",
            "remap_radius_m",
            "radius_m",
        ):
            found = _positive(geometry.get(key))
            if found is not None:
                return found
        copied = _positive(geometry.get("copied_radius_m"))
        field_max = _positive(geometry.get("field_r_max_m"))
        # Old source_1d_r_max policy stored padding as copied_radius_m.
        if copied is not None and (
            field_max is None or abs(copied - field_max) > 1.0e-12
        ):
            return copied
    for key in (
        "remap_radius_m",
        "requested_mapped_radius_m",
        "copied_radius_m",
    ):
        found = _positive(data.get(key))
        if found is not None:
            return found
    return None


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def copied_1d2d_radius_m(
    *,
    target_2d_case: Optional[str] = None,
    source_1d_case: Optional[str] = None,
    widget_mapped_radius: Optional[float] = None,
) -> Optional[float]:
    """Physical 1D→2D remap radius from case metadata.

    Prefer the user-defined remap/handoff radius. Source-mesh field_r_max is
    numerical padding and is not the independent-propagation boundary.
    The GUI mapped-radius widget is the fallback when metadata is absent.
    """
    from remap_handoff_1d import read_handoff_metadata
    from remap_snapshot_1d import read_snapshot_arrays, read_snapshot_metadata

    if target_2d_case:
        found = _from_remap_dict(read_handoff_metadata(target_2d_case))
        if found is not None:
            return found
        case2 = _read_json(os.path.join(target_2d_case, "case_2d.json"))
        if case2 is not None:
            found = _from_remap_dict(case2.get("remap_handoff"))
            if found is not None:
                return found
            found = _from_remap_dict(case2.get("remap_region"))
            if found is not None:
                return found
    if source_1d_case:
        found = _from_remap_dict(read_handoff_metadata(source_1d_case))
        if found is not None:
            return found
        found = _from_remap_dict(read_snapshot_metadata(source_1d_case))
        if found is not None:
            return found
    return _positive(widget_mapped_radius)


def copied_2d3d_radius_m(
    *,
    prepare_path: Optional[str] = None,
    prepare_payload: Optional[Dict[str, Any]] = None,
    source_2d_case: Optional[str] = None,
) -> Optional[float]:
    """Actual 2D→3D remap-volume radius from transfer metadata.

    Does not use the overall 3D domain size.
    """
    data = prepare_payload if isinstance(prepare_payload, dict) else None
    if data is None and prepare_path:
        data = _read_json(prepare_path)
        if data is None and source_2d_case:
            data = _read_json(os.path.join(source_2d_case, "prepare_3d_transfer.json"))
    if not isinstance(data, dict):
        return None
    volume = data.get("remap_volume")
    found = _from_remap_dict(volume if isinstance(volume, dict) else None)
    if found is not None:
        return found
    for key in ("copied_radius_m", "field_r_max_m", "remap_radius_m"):
        found = _positive(data.get(key))
        if found is not None:
            return found
    mapped = _positive(data.get("mapped_radius"))
    mapped_h = _positive(data.get("mapped_height"))
    if mapped is None:
        return None
    # Ignore leftover 1D mapped-radius widgets when the 2D field is much larger.
    if mapped_h is not None and mapped < 0.25 * mapped_h and mapped <= 0.5:
        return None
    return mapped


def target_cell_guard_m(cell_size: Optional[float]) -> float:
    found = _positive(cell_size)
    return found if found is not None else 0.0


def first_independent_r_m(receive_r_max: float, cell_size: Optional[float]) -> float:
    """Minimum independent propagation radius: strictly beyond remap + one cell."""
    return float(receive_r_max) + target_cell_guard_m(cell_size)


def exclusive_independent_r_min(receive_r_max: float, cell_size: Optional[float]) -> float:
    """First allowed sample location: R > remap + one target cell."""
    limit = first_independent_r_m(receive_r_max, cell_size)
    if not math.isfinite(limit) or limit <= 0.0:
        return limit
    return limit + max(1.0e-9, 1.0e-12 * abs(limit))


def classify_vs_remap(
    range_m: float,
    receive_r_max: Optional[float],
    cell_size: Optional[float] = None,
    *,
    atol: float = 1.0e-12,
) -> str:
    """Classify a gauge as inside / on-boundary / outside the copied remap region."""
    if _positive(receive_r_max) is None:
        return CLASS_OUTSIDE
    if not is_finite_number(range_m):
        return CLASS_INSIDE
    radius = float(range_m)
    receive = float(receive_r_max)
    guard = target_cell_guard_m(cell_size)
    if radius < receive - atol:
        return CLASS_INSIDE
    if radius <= receive + guard + atol:
        return CLASS_ON_BOUNDARY
    return CLASS_OUTSIDE


def kb_propagation_eligible(
    range_m: float,
    receive_r_max: Optional[float],
    cell_size: Optional[float] = None,
) -> bool:
    return classify_vs_remap(range_m, receive_r_max, cell_size) == CLASS_OUTSIDE


def physical_standoff_m(
    dim: str,
    xyz: Tuple[float, float, float],
    charge_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> float:
    """Independent physical R from the actual charge centre. No series offset."""
    wanted = str(dim or "").strip().lower()
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    if wanted == "1d":
        return radial_distance(x, y, z)
    return standoff_m((x, y, z), charge_center)


def scaled_z_from_r(range_m: float, mass_kg: float) -> Optional[float]:
    """Z = R / W**(1/3). Same physical point; X transform only."""
    if not is_finite_number(range_m) or not is_finite_number(mass_kg):
        return None
    if float(mass_kg) <= 0.0:
        return None
    return float(scaled_distance(float(range_m), float(mass_kg)))


def series_label(dim: str) -> str:
    wanted = str(dim or "").strip().lower()
    return {"1d": SERIES_1D, "2d": SERIES_2D, "3d": SERIES_3D}.get(wanted, f"BF {wanted.upper()}")
