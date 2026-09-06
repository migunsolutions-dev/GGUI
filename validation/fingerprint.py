"""Sampling-plan provenance. Reject stale JSON instead of silently reusing it."""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from models import normalize_source_model
from validation.metrics import is_finite_number

FLOAT_ABS = 1.0e-9
FLOAT_REL = 1.0e-9

# Config fields that must match the live case before a stored plan is reused.
CONFIG_KEYS = (
    "dim",
    "case_id",
    "mass_kg",
    "domain_size",
    "hob_m",
    "charge_center",
    "cell_size",
    "burst_mode",
    "reference_mode",
    "n_points",
    "remap_receive_r_max",
    # A JWL plan must never be reused for an IG run at the same geometry, and
    # vice versa. Plans cached before this key existed miss once and replan.
    "source_model",
)


def _round_float(value: Optional[float], digits: int = 12) -> Optional[float]:
    if not is_finite_number(value):
        return None
    return round(float(value), digits)


def case_id_from_path(case_path: Optional[str]) -> str:
    text = str(case_path or "").strip()
    if not text:
        return ""
    return os.path.basename(os.path.normpath(text))


def normalize_case_path(case_path: Optional[str]) -> str:
    text = str(case_path or "").strip()
    if not text:
        return ""
    return os.path.normcase(os.path.normpath(text))


def coordinates_payload(points: Sequence[Any]) -> Tuple[Tuple[float, float, float], ...]:
    out = []
    for point in points:
        try:
            out.append((float(point.x), float(point.y), float(point.z)))
        except (AttributeError, TypeError, ValueError):
            continue
    return tuple(out)


def build_fingerprint(
    *,
    dim: str,
    case_path: Optional[str],
    mass_kg: Optional[float],
    domain_size: Mapping[str, Optional[float]],
    hob_m: Optional[float],
    charge_center: Sequence[float],
    cell_size: Optional[float],
    burst_mode: str,
    reference_mode: str,
    points: Sequence[Any] = (),
    remap_receive_r_max: Optional[float] = None,
    source_model: Optional[str] = None,
) -> Dict[str, Any]:
    coords = coordinates_payload(points)
    domain = {
        key: _round_float(value)
        for key, value in dict(domain_size).items()
    }
    cc = tuple(_round_float(v) or 0.0 for v in list(charge_center)[:3])
    while len(cc) < 3:
        cc = cc + (0.0,)
    payload: Dict[str, Any] = {
        "dim": str(dim or "").strip().lower(),
        "case_path": normalize_case_path(case_path),
        "case_id": case_id_from_path(case_path),
        "mass_kg": _round_float(mass_kg),
        "domain_size": domain,
        "hob_m": _round_float(hob_m),
        "charge_center": list(cc),
        "cell_size": _round_float(cell_size),
        "burst_mode": str(burst_mode or ""),
        "reference_mode": str(reference_mode or ""),
        "n_points": len(coords),
        "coordinates": [list(item) for item in coords],
        "source_model": normalize_source_model(source_model),
    }
    if is_finite_number(remap_receive_r_max) and float(remap_receive_r_max) > 0.0:
        payload["remap_receive_r_max"] = _round_float(remap_receive_r_max)
    return payload


def _floats_close(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if not is_finite_number(left) or not is_finite_number(right):
        return left == right
    a = float(left)
    b = float(right)
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(FLOAT_ABS, FLOAT_REL * scale)


def _values_close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        return all(_values_close(left.get(key), right.get(key)) for key in keys)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_values_close(a, b) for a, b in zip(left, right))
    if is_finite_number(left) or is_finite_number(right):
        return _floats_close(left, right)
    return left == right


def fingerprints_match(
    stored: Optional[Mapping[str, Any]],
    expected: Optional[Mapping[str, Any]],
    *,
    keys: Sequence[str] = CONFIG_KEYS,
) -> bool:
    """True when every expected config field is present and equal in the stored record.

    A missing stored fingerprint is never a match (legacy / stale JSON).
    Coordinates are compared when both sides include them; expected configs
    built from a live snapshot may omit coordinates and still match on physics.
    """
    if not stored or not expected:
        return False
    stored_map = dict(stored)
    expected_map = dict(expected)
    for key in keys:
        if key not in expected_map:
            continue
        if key == "coordinates":
            continue
        if key == "n_points" and "n_points" not in expected_map:
            continue
        if key not in stored_map:
            return False
        if not _values_close(stored_map.get(key), expected_map.get(key)):
            return False
    if "coordinates" in expected_map and "coordinates" in stored_map:
        if not _values_close(stored_map["coordinates"], expected_map["coordinates"]):
            return False
    return True


def attach_plan_fingerprint(plan: Any, fingerprint: Mapping[str, Any]) -> Any:
    return replace(plan, fingerprint=dict(fingerprint))
