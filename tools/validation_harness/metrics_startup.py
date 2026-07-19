"""Collect startup-mesh metrics from case_init_mode.json and 0/ fields."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from tools.validation_harness.io_case import captured_mass_kg, load_case_init_mode
from tools.validation_harness.models import StartupMetrics


def collect_startup_metrics(case_dir: Path, mode: Optional[Dict[str, Any]] = None) -> StartupMetrics:
    mode = mode if mode is not None else load_case_init_mode(case_dir)
    cap = mode.get("charge_capture") or {}
    smm = mode.get("startup_mesh_metadata") or {}
    deep = smm.get("deep_seeding") or {}
    backup = smm.get("backup_region") or {}
    sm_block = smm.get("startup_mesh") or {}
    ccq = smm.get("charge_capture_quality") or {}
    rp = smm.get("runtime_planning") or {}

    rho = float(mode.get("rho_charge") or cap.get("rho_charge") or 1601.0)
    nominal = ccq.get("nominal_mass_kg")
    if nominal is None:
        nominal = mode.get("nominal_mass_kg")

    captured = ccq.get("captured_mass_kg")
    charge_cells = sm_block.get("charge_cells_alpha_ge_half")
    startup_cells = sm_block.get("initial_cell_count_after_init")

    if captured is None or startup_cells is None:
        mass, n_cells, n_charge = captured_mass_kg(
            case_dir / "0" / "alpha.c4",
            case_dir / "0" / "V",
            rho,
        )
        if captured is None:
            captured = mass
        if startup_cells is None and n_cells is not None:
            startup_cells = n_cells
        if charge_cells is None and n_charge is not None:
            charge_cells = n_charge

    mass_ratio = ccq.get("mass_ratio")
    if mass_ratio is None and captured is not None and nominal:
        try:
            mass_ratio = float(captured) / float(nominal)
        except (TypeError, ValueError, ZeroDivisionError):
            mass_ratio = None

    seed_req = mode.get("charge_refinement_requested", mode.get("user_requested_inside"))
    seed_eff = mode.get("charge_refinement_effective", mode.get("inside_levels"))

    outer_on = sm_block.get("charge_refine_outer_enabled")
    if outer_on is None:
        # Infer from snappy dict presence is not reliable; leave None if unknown.
        enable = getattr(mode, "charge_outer_refine_enable", None) if hasattr(mode, "charge_outer_refine_enable") else None
        if enable is not None:
            outer_on = enable is not False

    return StartupMetrics(
        nominal_mass_kg=float(nominal) if nominal is not None else None,
        captured_mass_kg=float(captured) if captured is not None else None,
        mass_ratio=float(mass_ratio) if mass_ratio is not None else None,
        backup_radius_m=_f(backup.get("backup_radius_m") or cap.get("charge_capture_radius_used_m")),
        backup_to_charge_ratio=_f(backup.get("backup_to_charge_ratio") or cap.get("ratio_capture_to_physical")),
        seed_level_requested=_i(seed_req),
        seed_level_effective=_i(seed_eff),
        charge_cells=_i(charge_cells),
        startup_cell_count=_i(startup_cells),
        base_cell_count_blockmesh=_i(mode.get("base_cell_count")),
        set_cmd=mode.get("set_cmd"),
        charge_refine_outer_enabled=bool(outer_on) if outer_on is not None else None,
        cells_across_charge_estimate=_f(deep.get("cells_across_charge_estimate")),
        projected_startup_cells_estimate=_i(rp.get("projected_startup_cells_estimate")),
        extra={
            "charge_shape": mode.get("charge_shape"),
            "base_cell_size_m": mode.get("base_cell_size"),
            "charge_size_info": mode.get("charge_size_info"),
        },
    )


def _f(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
