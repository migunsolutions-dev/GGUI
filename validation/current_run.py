"""Resolve the current active run. Never pick an arbitrary file on disk."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple

SOURCE_CURRENT = "current_run"
SOURCE_MANUAL = "manual"

MISSING_CURRENT_RUN = "Current run does not contain the required validation data."


@dataclass(frozen=True)
class RunSnapshot:
    """Immutable view of the GUI's current cases and charge inputs."""

    source: str = SOURCE_CURRENT
    live_mode: Optional[str] = None
    live_case_dir: Optional[str] = None
    case_1d: Optional[str] = None
    case_2d: Optional[str] = None
    case_3d: Optional[str] = None
    last_run_1d: Optional[str] = None
    last_run_2d: Optional[str] = None
    last_run_3d: Optional[str] = None
    mass_kg: Optional[float] = None
    material_name: str = ""
    hob_m: Optional[float] = None
    charge_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    p_atm: float = 101325.0
    keep_openfoam_2d: bool = False
    keep_openfoam_3d: bool = False
    output_options: Any = None
    mapping_source_2d: Optional[str] = None
    mapping_time_2d: Optional[str] = None
    mapped_radius: Optional[float] = None
    remap_3d_source: Optional[str] = None
    remap_3d_source_type: Optional[str] = None
    prepare_3d_transfer: Optional[str] = None
    domain_radius_1d: Optional[float] = None
    domain_radius_2d: Optional[float] = None
    domain_height_2d: Optional[float] = None
    domain_cell_1d: Optional[float] = None
    domain_cell_2d: Optional[float] = None
    charge_center_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    extra: Dict[str, Any] = field(default_factory=dict)


def _norm(path: Optional[str]) -> Optional[str]:
    text = str(path or "").strip()
    return text or None


def _mode_to_dim(mode: Optional[str]) -> Optional[str]:
    if not mode:
        return None
    key = str(mode).strip().lower()
    if key in ("1d", "2d", "3d"):
        return key
    return None


def case_dir_for_dim(snapshot: RunSnapshot, dim: str) -> Optional[str]:
    """Pick the current-run case for one dimension. No directory scanning."""
    wanted = str(dim).strip().lower()
    live_dim = _mode_to_dim(snapshot.live_mode)
    if live_dim == wanted:
        found = _norm(snapshot.live_case_dir)
        if found:
            return found
    initialized = {
        "1d": snapshot.case_1d,
        "2d": snapshot.case_2d,
        "3d": snapshot.case_3d,
    }.get(wanted)
    found = _norm(initialized)
    if found:
        return found
    last = {
        "1d": snapshot.last_run_1d,
        "2d": snapshot.last_run_2d,
        "3d": snapshot.last_run_3d,
    }.get(wanted)
    return _norm(last)


def primary_case_dir(snapshot: RunSnapshot) -> Optional[str]:
    live = _norm(snapshot.live_case_dir)
    if live:
        return live
    for dim in ("2d", "3d", "1d"):
        found = case_dir_for_dim(snapshot, dim)
        if found:
            return found
    return None


def with_manual_case(snapshot: RunSnapshot, dim: str, case_dir: str) -> RunSnapshot:
    path = _norm(case_dir)
    updates = {"source": SOURCE_MANUAL}
    if dim == "1d":
        updates["case_1d"] = path
    elif dim == "2d":
        updates["case_2d"] = path
    elif dim == "3d":
        updates["case_3d"] = path
    return replace(snapshot, **updates)


def reset_to_current(snapshot: RunSnapshot) -> RunSnapshot:
    return replace(snapshot, source=SOURCE_CURRENT)


def live_dimension(snapshot: RunSnapshot) -> Optional[str]:
    return _mode_to_dim(snapshot.live_mode)


def histories_available(snapshot: RunSnapshot, dim: str) -> bool:
    """True when high-resolution probe histories exist for this dimension."""
    from validation.probes import EXISTING_1D_GRAPH_FO, PROBE_FO, VALIDATION_FO, latest_probe_field_file

    case = case_dir_for_dim(snapshot, dim)
    if not case:
        return False
    wanted = str(dim).strip().lower()
    if wanted == "1d":
        fo = EXISTING_1D_GRAPH_FO
    elif wanted == "2d":
        fo = VALIDATION_FO.get("2d", "")
    else:
        fo = PROBE_FO.get(wanted, "")
    return bool(fo and latest_probe_field_file(case, fo, "p"))


def default_display_dims(snapshot: RunSnapshot) -> set:
    """Select completed Current Run dimensions that have simulation histories.

    A dimension without histories is not shown as a computed result. If nothing
    has finished yet, preview only the live dimension (labelled Planned).
    """
    available = {name for name in ("1d", "2d", "3d") if histories_available(snapshot, name)}
    if available:
        return available
    live = live_dimension(snapshot)
    if live in ("1d", "2d", "3d"):
        return {live}
    return set()


def charge_center_for_dim(snapshot: RunSnapshot, dim: str) -> Tuple[float, float, float]:
    wanted = str(dim).strip().lower()
    if wanted == "1d":
        return (0.0, 0.0, 0.0)
    if wanted == "2d":
        hob = float(snapshot.hob_m) if snapshot.hob_m is not None else 0.0
        return (0.0, hob, 0.0)
    cc = snapshot.charge_center_3d or snapshot.charge_center
    return (float(cc[0]), float(cc[1]), float(cc[2]))


ContextProvider = Callable[[], RunSnapshot]
