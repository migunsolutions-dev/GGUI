"""Startup mesh metadata and advisory warnings for 3D case generation.

Pure observability: does not change meshing, capture, seeding, AMR, or defaults.
Used to populate ``case_init_mode.json`` for validation (EX/B) and GUI Info panels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Architecture-recommended auto-seed policy (metadata / recommendation only — not applied).
RECOMMENDED_CELLS_ACROSS_CHARGE_N = 6
RECOMMENDED_AUTO_SEED_L_MAX = 5

# Advisory thresholds (documentation / warnings only).
MASS_RATIO_WARN_BELOW = 0.98
PROJECTED_STARTUP_CELLS_WARN_ABOVE = 200_000
BACKUP_TO_CHARGE_RATIO_WARN_ABOVE = 2.0
CELLS_ACROSS_CHARGE_WARN_BELOW = 4.0
BAND_PLUS_DEEP_SEED_RISK_NOTE = (
    "chargeRefineOuter is enabled together with deep setRefinedFields seeding; "
    "investigation showed this can multiply initial cell count dramatically (~159× in one benchmark)."
)


@dataclass
class StartupMeshWarning:
    """Structured advisory warning (never alters generated case behavior)."""

    code: str
    severity: str  # "info" | "warning" | "strong_warning"
    message: str
    consequence: str

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "consequence": self.consequence,
        }

    def as_flat_string(self) -> str:
        return f"{self.message} Consequence: {self.consequence}"


def smallest_charge_dimension_m(charge_shape: str, dims: Dict[str, float]) -> Tuple[float, str]:
    """Return (d_min, binding_dimension_name) per architecture §12 / §16."""
    shape = (charge_shape or "Sphere").strip()
    if shape == "Cuboid":
        if "length" in dims and "width" in dims and "height" in dims:
            L, W, H = float(dims["length"]), float(dims["width"]), float(dims["height"])
            d_min = min(L, W, H)
            name = "min_edge"
        else:
            s = float(dims.get("side", 0.1))
            d_min = s
            name = "cube_side"
        return max(d_min, 1e-12), name
    if shape == "Cylinder":
        r = float(dims.get("radius", 0.05))
        length = float(dims.get("length", 0.1))
        d_diam = 2.0 * r
        if d_diam <= length + 1e-12:
            return max(d_diam, 1e-12), "diameter"
        return max(length, 1e-12), "length"
    r = float(dims.get("radius", 0.05))
    return max(2.0 * r, 1e-12), "diameter"


def recommended_auto_seed_level(
    base_cell_m: float,
    d_min_m: float,
    *,
    target_cells_across: int = RECOMMENDED_CELLS_ACROSS_CHARGE_N,
    l_max: int = RECOMMENDED_AUTO_SEED_L_MAX,
) -> dict:
    """Architecture auto-seed formula (recommendation only — not applied to cases)."""
    dx = max(1e-9, float(base_cell_m))
    d = max(1e-12, float(d_min_m))
    n = max(1, int(target_cells_across))
    lmx = max(0, int(l_max))
    raw = math.log2(n * dx / d) if d > 0 else 0.0
    l_ceil = int(math.ceil(raw)) if raw > 0 else 0
    l_clamped = max(0, min(lmx, l_ceil))
    return {
        "target_cells_across_charge_N": n,
        "l_max": lmx,
        "formula": "L_seed = clamp(ceil(log2(N * base_cell / d_min)), 0, L_max)",
        "raw_log2_value": raw,
        "recommended_level_before_clamp": l_ceil,
        "recommended_level": l_clamped,
        "clamped_at_l_max": l_ceil > lmx,
    }


def cells_across_charge_estimate(
    d_min_m: float,
    base_cell_m: float,
    effective_seed_level: int,
) -> float:
    """Estimated cells across smallest charge dimension at effective seed level."""
    dx = max(1e-9, float(base_cell_m))
    d = max(1e-12, float(d_min_m))
    L = max(0, int(effective_seed_level))
    h_eff = dx / (2.0 ** L) if L > 0 else dx
    return d / h_eff


def estimate_projected_startup_cells(
    base_cell_count: Optional[int],
    *,
    uses_set_refined_fields: bool,
    seed_level: int,
    charge_refine_outer_enabled: bool,
    backup_radius_m: Optional[float],
    base_cell_m: float,
) -> Optional[int]:
    """Heuristic upper-bound style estimate for advisory use only."""
    if base_cell_count is None or base_cell_count <= 0:
        return None
    bc = int(base_cell_count)
    L = max(0, int(seed_level))
    if not uses_set_refined_fields or L == 0:
        return bc
    dx = max(1e-9, float(base_cell_m))
    r_back = float(backup_radius_m) if backup_radius_m and backup_radius_m > 0 else dx
    # Volume ratio of backup ball to one base cell (very rough).
    vol_ratio = (4.0 / 3.0) * math.pi * (r_back ** 3) / (dx ** 3)
    vol_ratio = max(1.0, min(vol_ratio, 1e6))
    level_factor = 8.0 ** min(L, 6)
    est = int(bc * vol_ratio * level_factor * 0.15)
    if charge_refine_outer_enabled and L > 0:
        est = int(est * 30)
    return max(bc, est)


def charge_smaller_than_base_cell(d_min_m: float, base_cell_m: float) -> bool:
    return float(d_min_m) < max(1e-9, float(base_cell_m)) - 1e-9


def validation_profile_tags(
    *,
    charge_shape: str,
    mass_kg: float,
    base_cell_m: float,
    d_min_m: float,
    seed_level: int,
    charge_refine_outer_enabled: bool,
    sub_cell: bool,
) -> List[str]:
    """Tags useful for grouping cases in EX-1 / EX-2 / B1 / B2 matrices."""
    tags: List[str] = []
    if sub_cell:
        tags.append("sub_cell_charge")
    if base_cell_m >= 0.45:
        tags.append("coarse_base_mesh")
    if base_cell_m <= 0.25:
        tags.append("fine_base_mesh")
    if mass_kg <= 6.0:
        tags.append("small_mass_5kg_class")
    if mass_kg >= 20.0:
        tags.append("reference_mass_25kg_class")
    if seed_level >= 4:
        tags.append("deep_seed")
    if seed_level == 0:
        tags.append("no_deep_seed")
    if charge_refine_outer_enabled:
        tags.append("charge_refine_outer_on")
    else:
        tags.append("charge_refine_outer_off")
    shape = (charge_shape or "").lower()
    if shape:
        tags.append(f"shape_{shape.lower()}")
    if d_min_m / max(base_cell_m, 1e-9) < 0.6:
        tags.append("charge_much_smaller_than_cell")
    if charge_refine_outer_enabled and seed_level > 0:
        tags.append("band_plus_deep_seed_risk")
    return tags


def build_startup_mesh_warnings(
    *,
    inputs: Any,
    dims: Dict[str, float],
    charge_capture: Optional[dict],
    base_cell_count: Optional[int],
    seed_requested: int,
    seed_effective: int,
    uses_set_refined_fields: bool,
    charge_refine_outer_enabled: bool,
    outer_snappy_level_max: Optional[int],
    amr_max_refinement: Optional[int],
    auto_seed_rec: dict,
    d_min_m: float,
    cells_across: float,
    projected_cells: Optional[int],
) -> List[StartupMeshWarning]:
    """Advisory warnings only — never modify inputs or generated dictionaries."""
    out: List[StartupMeshWarning] = []
    dx = max(1e-9, float(getattr(inputs, "cell_size", 0.5) or 0.5))
    mass = float(getattr(inputs, "mass_kg", 0.0) or 0.0)

    if charge_smaller_than_base_cell(d_min_m, dx):
        out.append(
            StartupMeshWarning(
                code="sub_cell_charge",
                severity="warning",
                message=(
                    f"Smallest charge dimension ({d_min_m:.4g} m) is smaller than the base cell "
                    f"({dx:.4g} m)."
                ),
                consequence=(
                    "Charge geometry may occupy less than one base cell; robust capture depends on "
                    "the backup region and setRefinedFields cell selection (verify mass ratio after init)."
                ),
            )
        )

    cap = charge_capture or {}
    ratio_cap = cap.get("ratio_capture_to_physical")
    if ratio_cap is not None and float(ratio_cap) > BACKUP_TO_CHARGE_RATIO_WARN_ABOVE:
        r_cap_v = cap.get("charge_capture_radius_used_m")
        r_cap_s = f"{float(r_cap_v):.4g}" if r_cap_v is not None else "?"
        out.append(
            StartupMeshWarning(
                code="large_backup_to_charge_ratio",
                severity="warning",
                message=(
                    f"Backup capture radius is {float(ratio_cap):.2f}× the physical charge size "
                    f"(R_cap={r_cap_s} m)."
                ),
                consequence=(
                    "A large backup/seed region can inflate initial cell count on coarse meshes and "
                    "make runtime unrefinement behind the wave harder."
                ),
            )
        )

    if cells_across < CELLS_ACROSS_CHARGE_WARN_BELOW and seed_requested > 0:
        out.append(
            StartupMeshWarning(
                code="low_cells_across_charge",
                severity="warning",
                message=(
                    f"Estimated cells across smallest charge dimension ≈ {cells_across:.2f} "
                    f"at seed level {seed_effective} (architecture target N={RECOMMENDED_CELLS_ACROSS_CHARGE_N} "
                    f"would recommend level {auto_seed_rec.get('recommended_level')})."
                ),
                consequence=(
                    "The charge may be under-resolved at startup; early blast formation may depend "
                    "heavily on the first runtime AMR cycles."
                ),
            )
        )

    if seed_requested >= 5:
        out.append(
            StartupMeshWarning(
                code="very_high_seed_level",
                severity="warning",
                message=f"Requested charge seed level is {seed_requested} (deep refineInternal).",
                consequence=(
                    "Deep seeding increases initial cell count; combined with chargeRefineOuter it "
                    "has produced explosive mesh growth in validation."
                ),
            )
        )

    if charge_refine_outer_enabled and seed_requested > 0:
        out.append(
            StartupMeshWarning(
                code="band_plus_deep_seed",
                severity="strong_warning",
                message=BAND_PLUS_DEEP_SEED_RISK_NOTE,
                consequence=(
                    "Case may become computationally intractable before the early propagation window; "
                    "compare band-OFF with the same seed depth for validation."
                ),
            )
        )

    if projected_cells is not None and projected_cells > PROJECTED_STARTUP_CELLS_WARN_ABOVE:
        out.append(
            StartupMeshWarning(
                code="projected_excessive_startup_cells",
                severity="strong_warning",
                message=(
                    f"Heuristic projected initial cell count after seeding ≈ {projected_cells:,} "
                    f"(advisory threshold {PROJECTED_STARTUP_CELLS_WARN_ABOVE:,})."
                ),
                consequence=(
                    "Startup mesh cost may dominate runtime; consider band-OFF, lower seed level, "
                    "or a finer base mesh before long solver runs."
                ),
            )
        )

    if amr_max_refinement is not None and seed_requested > int(amr_max_refinement):
        out.append(
            StartupMeshWarning(
                code="seed_exceeds_amr_max",
                severity="warning",
                message=(
                    f"Requested seed level ({seed_requested}) exceeds runtime maxRefinement "
                    f"({int(amr_max_refinement)})."
                ),
                consequence=(
                    "Runtime AMR may unrefine seeded regions at t≈0, reducing near-field resolution "
                    "until the first refinement pass."
                ),
            )
        )

    if auto_seed_rec.get("clamped_at_l_max"):
        out.append(
            StartupMeshWarning(
                code="auto_seed_would_clamp",
                severity="info",
                message=(
                    f"Architecture auto-seed recommendation would clamp at L_max={auto_seed_rec.get('l_max')} "
                    f"(uncapped level {auto_seed_rec.get('recommended_level_before_clamp')})."
                ),
                consequence=(
                    "Very small charge on a coarse base mesh may be under-resolved relative to target N; "
                    "consider a finer base cell size (advisory only — current seed level unchanged)."
                ),
            )
        )

    if dx >= 0.8 and mass <= 6.0:
        out.append(
            StartupMeshWarning(
                code="extreme_coarse_operating_region",
                severity="warning",
                message=(
                    f"Operating in an extremely coarse regime (base cell {dx:.4g} m, mass {mass:.4g} kg)."
                ),
                consequence=(
                    "Validate capture and clamp/warn behavior (roadmap B5); auto-seed recommendation "
                    "may hit L_max."
                ),
            )
        )

    return out


def build_startup_mesh_metadata(
    inputs: Any,
    dims: Dict[str, float],
    *,
    charge_capture: Optional[dict],
    base_cell_count: Optional[int],
    base_cell_size_m: float,
    charge_shape: str,
    seed_requested: int,
    seed_effective: int,
    uses_set_refined_fields: bool,
    set_cmd: str,
    charge_refine_outer_enabled: bool,
    outer_snappy_level_min: Optional[int],
    outer_snappy_level_max: Optional[int],
    amr_written: Optional[dict],
    transition_region: Optional[dict],
    nominal_mass_kg: Optional[float] = None,
) -> dict:
    """Assemble the ``startup_mesh_metadata`` object for case_init_mode.json."""
    dx = max(1e-9, float(base_cell_size_m))
    mass = float(nominal_mass_kg if nominal_mass_kg is not None else getattr(inputs, "mass_kg", 0.0) or 0.0)
    d_min, d_binding = smallest_charge_dimension_m(charge_shape, dims)
    auto_rec = recommended_auto_seed_level(dx, d_min)
    L_eff = int(seed_effective) if uses_set_refined_fields else 0
    cells_across = cells_across_charge_estimate(d_min, dx, L_eff)

    cap = charge_capture or {}
    backup_r = cap.get("charge_capture_radius_used_m")
    projected = estimate_projected_startup_cells(
        base_cell_count,
        uses_set_refined_fields=uses_set_refined_fields,
        seed_level=L_eff,
        charge_refine_outer_enabled=charge_refine_outer_enabled,
        backup_radius_m=float(backup_r) if backup_r is not None else None,
        base_cell_m=dx,
    )

    amr_max = None
    if isinstance(amr_written, dict):
        try:
            amr_max = int(amr_written.get("maxRefinement"))
        except (TypeError, ValueError):
            amr_max = None

    warnings = build_startup_mesh_warnings(
        inputs=inputs,
        dims=dims,
        charge_capture=charge_capture,
        base_cell_count=base_cell_count,
        seed_requested=int(seed_requested),
        seed_effective=L_eff,
        uses_set_refined_fields=uses_set_refined_fields,
        charge_refine_outer_enabled=charge_refine_outer_enabled,
        outer_snappy_level_max=outer_snappy_level_max,
        amr_max_refinement=amr_max,
        auto_seed_rec=auto_rec,
        d_min_m=d_min,
        cells_across=cells_across,
        projected_cells=projected,
    )

    sub_cell = charge_smaller_than_base_cell(d_min, dx)
    tags = validation_profile_tags(
        charge_shape=charge_shape,
        mass_kg=mass,
        base_cell_m=dx,
        d_min_m=d_min,
        seed_level=int(seed_requested),
        charge_refine_outer_enabled=charge_refine_outer_enabled,
        sub_cell=sub_cell,
    )

    indicator = None
    if isinstance(amr_written, dict):
        indicator = (amr_written.get("errorEstimator_line") or "").strip() or None

    return {
        "schema_version": 1,
        "charge_capture_quality": {
            "nominal_mass_kg": mass,
            "captured_mass_kg": None,
            "mass_ratio": None,
            "note": (
                "captured_mass_kg and mass_ratio are populated after setFields/setRefinedFields "
                "(validation scripts or post-init analysis); mass-conserving sources target nominal mass."
            ),
        },
        "startup_mesh": {
            "base_cell_count_blockmesh": base_cell_count,
            "initial_cell_count_after_init": None,
            "charge_cells_alpha_ge_half": None,
            "set_cmd": set_cmd,
            "uses_set_refined_fields": uses_set_refined_fields,
            "seed_level_requested": int(seed_requested),
            "seed_level_effective": L_eff,
            "charge_refine_outer_enabled": charge_refine_outer_enabled,
            "outer_snappy_levels": (
                [outer_snappy_level_min, outer_snappy_level_max]
                if outer_snappy_level_min is not None and outer_snappy_level_max is not None
                else None
            ),
        },
        "backup_region": {
            "mode": cap.get("mode"),
            "backup_radius_m": backup_r,
            "physical_charge_radius_m": cap.get("physical_charge_radius_m"),
            "backup_to_charge_ratio": cap.get("ratio_capture_to_physical"),
            "capture_factor": cap.get("charge_capture_factor"),
            "formula_description": cap.get("formula_description"),
            "base_cell_m": dx,
        },
        "deep_seeding": {
            "requested_level": int(seed_requested),
            "effective_level": L_eff,
            "smallest_charge_dimension_m": d_min,
            "binding_dimension": d_binding,
            "cells_across_charge_estimate": cells_across,
        },
        "auto_seed_recommendation": {
            **auto_rec,
            "applied": False,
            "note": "Recommendation only — does not change charge_refinement_level or generated dictionaries.",
        },
        "runtime_planning": {
            "projected_startup_cells_estimate": projected,
            "projected_cells_estimate_is_heuristic": True,
            "amr_indicator_written": indicator,
            "amr_max_refinement": amr_max,
            "refinement_burden_indicators": {
                "deep_seed": L_eff >= 4,
                "band_enabled": charge_refine_outer_enabled,
                "band_plus_deep_seed": charge_refine_outer_enabled and L_eff > 0,
                "sub_cell_charge": sub_cell,
            },
        },
        "validation_profile": {
            "tags": tags,
            "suggested_experiments": _suggested_experiments(tags),
        },
        "transition_region_summary": (
            {
                "outside_extent_m": transition_region.get("outside_extent_m"),
                "outside_extent_auto": transition_region.get("outside_extent_auto"),
            }
            if isinstance(transition_region, dict)
            else None
        ),
        "warnings_structured": [w.as_dict() for w in warnings],
    }


def _suggested_experiments(tags: List[str]) -> List[str]:
    """Map profile tags to roadmap experiment IDs (advisory)."""
    ex: List[str] = []
    if "sub_cell_charge" in tags or "charge_much_smaller_than_cell" in tags:
        ex.append("EX-2")
        ex.append("B5")
    if "band_plus_deep_seed_risk" in tags:
        ex.append("EX-1")
        ex.append("EX-6")
    if "deep_seed" in tags and "charge_refine_outer_off" in tags:
        ex.append("EX-3")
    if "coarse_base_mesh" in tags:
        ex.append("EX-2")
        ex.append("B5")
    if "no_deep_seed" in tags and "charge_refine_outer_on" in tags:
        ex.append("EX-1")
    if not ex:
        ex.append("EX-2")
    return sorted(set(ex))


def flatten_warnings_for_charge_warnings(metadata: dict) -> List[str]:
    """Flat strings for legacy charge_warnings / GUI list (no behavior change)."""
    structured = metadata.get("warnings_structured") or []
    flat: List[str] = []
    for w in structured:
        if isinstance(w, dict):
            msg = w.get("message", "")
            cons = w.get("consequence", "")
            if msg and cons:
                flat.append(f"{msg} Consequence: {cons}")
            elif msg:
                flat.append(str(msg))
    return flat
