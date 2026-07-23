"""Central automatic charge-seed plan for 3D meshing (Qt/generator independent).

Shared by Mesh Plan UI, generator, initialization plan, capture guard,
metadata, project I/O, and tests.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

SEED_MODE_AUTO = "Auto"
SEED_MODE_MANUAL = "Manual"
SEED_MODE_OFF = "Off"
SEED_MODES = (SEED_MODE_AUTO, SEED_MODE_MANUAL, SEED_MODE_OFF)

DEFAULT_TARGET_CELLS = 8
DEFAULT_MIN_CELLS = 6
DEFAULT_MAX_AUTO_LEVEL = 5

UNSAFE_AUTO_SEED_MESSAGE = (
    "Automatic charge seeding cannot achieve the minimum of {min_cells} cells "
    "across the smallest charge dimension (d_min={d_min:.4g} m) even at the "
    "maximum automatic seed level ({max_level}). "
    "Achieved ≈ {achieved:.2f} cells at h_seed={h_seed:.4g} m.\n\n"
    "Reduce the base cell size or use 1D-to-3D remap. "
    "Remap bypasses this guard. An explicit Manual seed remains available "
    "only as an expert override and is reported clearly in the Mesh Plan."
)


class SeedPolicyError(ValueError):
    """Invalid explicit charge-seed policy (mode or numeric)."""


@dataclass(frozen=True)
class ChargeSeedPlan:
    mode: str
    target_cells: int
    min_cells: int
    max_level: int
    d_min_m: float
    d_min_name: str
    h0_m: float
    level_required: int
    level_requested: int
    level_effective: int
    achieved_cells: float
    cap_applied: bool
    is_safe: bool
    reason: str
    h_seed_m: float
    independence_note: str = (
        "Startup charge seed and runtime Wave AMR level are intentionally independent."
    )
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_seed_mode(raw: Any) -> str:
    """Normalize an *explicit* seed-mode value.

    Raises SeedPolicyError for unrecognized explicit strings.
    Does not accept None/empty — callers must apply legacy migration first.
    """
    if raw is None:
        raise SeedPolicyError("charge_seed_mode is missing; use resolve_seed_mode for legacy migration")
    text = str(raw).strip()
    if text == "":
        raise SeedPolicyError("charge_seed_mode is empty; use resolve_seed_mode for legacy migration")
    low = text.lower()
    if low == "auto":
        return SEED_MODE_AUTO
    if low in ("manual", "man"):
        return SEED_MODE_MANUAL
    if low in ("off", "none", "disabled", "0"):
        return SEED_MODE_OFF
    if text in SEED_MODES:
        return text
    raise SeedPolicyError(
        f"Invalid charge_seed_mode {raw!r}. "
        f"Expected one of: {', '.join(SEED_MODES)}."
    )


def resolve_seed_mode(inputs: Any) -> str:
    """Return explicit mode, migrating absent keys from historical level-only projects."""
    raw = getattr(inputs, "charge_seed_mode", None)
    if raw is None or str(raw).strip() == "":
        # Historical: level>0 ⇒ Manual, level==0 ⇒ Off (never silently Auto).
        lvl_raw = getattr(inputs, "charge_refinement_level", 0)
        try:
            lvl = max(0, int(0 if lvl_raw is None else lvl_raw))
        except (TypeError, ValueError) as exc:
            raise SeedPolicyError(
                f"charge_refinement_level must be an integer >= 0 (got {lvl_raw!r})"
            ) from exc
        return SEED_MODE_MANUAL if lvl > 0 else SEED_MODE_OFF
    return normalize_seed_mode(raw)


def _policy_int(
    inputs: Any,
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    """Read an integer policy field: missing/None → default; explicit 0 preserved.

    Raises SeedPolicyError for non-numeric or below-minimum values.
    """
    if not hasattr(inputs, name):
        return int(default)
    raw = getattr(inputs, name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SeedPolicyError(
            f"{name} must be an integer >= {minimum} (got {raw!r})"
        ) from exc
    if value < minimum:
        raise SeedPolicyError(
            f"{name} must be an integer >= {minimum} (got {value})"
        )
    return value


def charge_dims_from_inputs(inputs: Any) -> Dict[str, float]:
    """Physical charge dimensions — delegates to physical_charge_geometry."""
    from physical_charge_geometry import dims_dict_from_inputs

    return dims_dict_from_inputs(inputs)


def smallest_charge_dimension_m(charge_shape: str, dims: Dict[str, float]) -> Tuple[float, str]:
    shape = (charge_shape or "Sphere").strip()
    if shape == "Cuboid":
        if "length" in dims and "width" in dims and "height" in dims:
            L, W, H = float(dims["length"]), float(dims["width"]), float(dims["height"])
            return max(min(L, W, H), 1e-12), "min_edge"
        s = float(dims.get("side", 0.1))
        return max(s, 1e-12), "cube_side"
    if shape == "Cylinder":
        r = float(dims.get("radius", 0.05))
        length = float(dims.get("length", 0.1))
        d_diam = 2.0 * r
        if d_diam <= length + 1e-12:
            return max(d_diam, 1e-12), "diameter"
        return max(length, 1e-12), "length"
    r = float(dims.get("radius", 0.05))
    return max(2.0 * r, 1e-12), "diameter"


def required_seed_level(h0_m: float, d_min_m: float, target_cells: int) -> int:
    """L_required = ceil(log2(N_target * h0 / d_min)), lower-bounded at 0."""
    h0 = max(1e-12, float(h0_m))
    d = max(1e-12, float(d_min_m))
    n = max(1, int(target_cells))
    raw = math.log2(n * h0 / d)
    return max(0, int(math.ceil(raw))) if raw > 0 else 0


def cells_across_d_min(d_min_m: float, h0_m: float, level: int) -> float:
    h0 = max(1e-12, float(h0_m))
    d = max(1e-12, float(d_min_m))
    L = max(0, int(level))
    h_seed = h0 / (2.0 ** L)
    return d / h_seed


def build_charge_seed_plan(inputs: Any) -> ChargeSeedPlan:
    """Authoritative charge-seed decision for the current CaseInputs3D-like object."""
    mode = resolve_seed_mode(inputs)
    target = _policy_int(inputs, "charge_seed_target_cells", DEFAULT_TARGET_CELLS, minimum=1)
    min_cells = _policy_int(inputs, "charge_seed_min_cells", DEFAULT_MIN_CELLS, minimum=1)
    max_level = _policy_int(inputs, "charge_seed_max_level", DEFAULT_MAX_AUTO_LEVEL, minimum=0)
    h0_raw = getattr(inputs, "cell_size", 0.1)
    if h0_raw is None:
        h0 = 0.1
    else:
        try:
            h0 = max(1e-12, float(h0_raw))
        except (TypeError, ValueError) as exc:
            raise SeedPolicyError(f"cell_size must be a positive number (got {h0_raw!r})") from exc
    shape = str(getattr(inputs, "charge_shape", "Sphere") or "Sphere")
    dims = charge_dims_from_inputs(inputs)
    d_min, d_name = smallest_charge_dimension_m(shape, dims)

    wave_raw = getattr(inputs, "dyn_refine_max", 1)
    try:
        wave = max(0, int(0 if wave_raw is None else wave_raw))
    except (TypeError, ValueError):
        wave = 1
    independence = (
        f"Startup charge seed and runtime Wave AMR L{wave} are intentionally independent."
    )

    if mode == SEED_MODE_OFF:
        return ChargeSeedPlan(
            mode=mode,
            target_cells=target,
            min_cells=min_cells,
            max_level=max_level,
            d_min_m=d_min,
            d_min_name=d_name,
            h0_m=h0,
            level_required=0,
            level_requested=0,
            level_effective=0,
            achieved_cells=cells_across_d_min(d_min, h0, 0),
            cap_applied=False,
            is_safe=True,
            reason="charge seed mode Off — no startup internal refinement",
            h_seed_m=h0,
            independence_note=independence,
        )

    if mode == SEED_MODE_MANUAL:
        lvl_raw = getattr(inputs, "charge_refinement_level", 0)
        try:
            requested = max(0, int(0 if lvl_raw is None else lvl_raw))
        except (TypeError, ValueError) as exc:
            raise SeedPolicyError(
                f"charge_refinement_level must be an integer >= 0 (got {lvl_raw!r})"
            ) from exc
        effective = requested
        achieved = cells_across_d_min(d_min, h0, effective)
        h_seed = h0 / (2.0 ** effective)
        return ChargeSeedPlan(
            mode=mode,
            target_cells=target,
            min_cells=min_cells,
            max_level=max_level,
            d_min_m=d_min,
            d_min_name=d_name,
            h0_m=h0,
            level_required=requested,
            level_requested=requested,
            level_effective=effective,
            achieved_cells=achieved,
            cap_applied=False,
            is_safe=True,
            reason=f"manual charge seed level {effective}",
            h_seed_m=h_seed,
            independence_note=independence,
            warnings=(independence,) if effective != wave else (),
        )

    # Auto
    required = required_seed_level(h0, d_min, target)
    cap_applied = required > max_level
    effective = min(required, max_level)
    achieved = cells_across_d_min(d_min, h0, effective)
    h_seed = h0 / (2.0 ** effective)
    is_safe = achieved + 1e-9 >= float(min_cells)
    if not is_safe:
        reason = (
            f"Auto seed capped at L{effective}: achieved {achieved:.2f} cells across "
            f"{d_name} < minimum {min_cells}"
        )
    elif cap_applied:
        reason = (
            f"Auto seed requires L{required} for {target} cells across {d_name}; "
            f"capped at L{max_level} (still ≥ minimum {min_cells})"
        )
    else:
        reason = (
            f"Auto seed L{effective} for target {target} cells across {d_name} "
            f"(d_min={d_min:.4g} m, h0={h0:.4g} m)"
        )
    warns = [independence]
    if cap_applied:
        warns.append(f"Auto seed level capped at {max_level} (required {required}).")
    if not is_safe:
        warns.append(
            UNSAFE_AUTO_SEED_MESSAGE.format(
                min_cells=min_cells,
                d_min=d_min,
                max_level=max_level,
                achieved=achieved,
                h_seed=h_seed,
            )
        )
    return ChargeSeedPlan(
        mode=mode,
        target_cells=target,
        min_cells=min_cells,
        max_level=max_level,
        d_min_m=d_min,
        d_min_name=d_name,
        h0_m=h0,
        level_required=required,
        level_requested=required,
        level_effective=effective,
        achieved_cells=achieved,
        cap_applied=cap_applied,
        is_safe=is_safe,
        reason=reason,
        h_seed_m=h_seed,
        independence_note=independence,
        warnings=tuple(warns),
    )


def _coerce_policy_int(raw: Any, name: str, *, minimum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SeedPolicyError(
            f"{name} must be an integer >= {minimum} (got {raw!r})"
        ) from exc
    if value < minimum:
        raise SeedPolicyError(
            f"{name} must be an integer >= {minimum} (got {value})"
        )
    return value


def migrate_case_inputs_seed_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy project/case dicts to explicit seed mode (non-mutating copy).

    Explicit invalid ``charge_seed_mode`` raises SeedPolicyError (do not silently Auto).
    """
    out = dict(data)
    had_mode = "charge_seed_mode" in out and out.get("charge_seed_mode") not in (None, "")
    if not had_mode:
        lvl = max(0, int(out.get("charge_refinement_level", 0) or 0))
        out["charge_seed_mode"] = SEED_MODE_MANUAL if lvl > 0 else SEED_MODE_OFF
    else:
        out["charge_seed_mode"] = normalize_seed_mode(out.get("charge_seed_mode"))

    # Defaults for new keys when absent/None. Explicit 0 for max_level is preserved.
    if "charge_seed_target_cells" not in out or out.get("charge_seed_target_cells") is None:
        out["charge_seed_target_cells"] = DEFAULT_TARGET_CELLS
    else:
        out["charge_seed_target_cells"] = _coerce_policy_int(
            out["charge_seed_target_cells"], "charge_seed_target_cells", minimum=1
        )
    if "charge_seed_min_cells" not in out or out.get("charge_seed_min_cells") is None:
        out["charge_seed_min_cells"] = DEFAULT_MIN_CELLS
    else:
        out["charge_seed_min_cells"] = _coerce_policy_int(
            out["charge_seed_min_cells"], "charge_seed_min_cells", minimum=1
        )
    if "charge_seed_max_level" not in out or out.get("charge_seed_max_level") is None:
        out["charge_seed_max_level"] = DEFAULT_MAX_AUTO_LEVEL
    else:
        out["charge_seed_max_level"] = _coerce_policy_int(
            out["charge_seed_max_level"], "charge_seed_max_level", minimum=0
        )

    # Outer band: missing key historically meant enabled (None/True).
    if "charge_outer_refine_enable" not in data:
        out["charge_outer_refine_enable"] = True
    elif out.get("charge_outer_refine_enable") is None:
        out["charge_outer_refine_enable"] = True

    # Single effective level for mode-inside; legacy min/max → max as level.
    if "charge_outer_refine_level" not in out or out.get("charge_outer_refine_level") is None:
        rmax = out.get("charge_outer_refine_max")
        rmin = out.get("charge_outer_refine_min")
        if rmax is not None or rmin is not None:
            try:
                a = int(rmin) if rmin is not None else 0
                b = int(rmax) if rmax is not None else a
                out["charge_outer_refine_level"] = max(a, b)
            except (TypeError, ValueError):
                out["charge_outer_refine_level"] = int(out.get("refine_max", 3) or 3)
        elif out.get("charge_outer_refine_enable"):
            out["charge_outer_refine_level"] = int(out.get("refine_max", 3) or 3)
        else:
            out["charge_outer_refine_level"] = 0

    return out


def seed_status_label(plan: ChargeSeedPlan) -> str:
    if plan.mode == SEED_MODE_OFF:
        return "Off"
    if not plan.is_safe:
        return "Unsafe (need finer base or remap)"
    if plan.cap_applied:
        return f"Capped at L{plan.level_effective}"
    if plan.mode == SEED_MODE_AUTO:
        return f"Auto OK (~{plan.achieved_cells:.1f} cells)"
    return f"Manual L{plan.level_effective}"
