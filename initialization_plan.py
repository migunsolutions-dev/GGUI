"""Authoritative 3D field-initialization decision.

This module is deliberately free of Qt and generator imports so the GUI,
generator, metadata, and tests can consume exactly the same decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class InitializationPlan:
    command: str
    uses_set_refined_fields: bool
    performs_internal_refinement: bool
    seed_requested: int
    seed_effective: int
    reason: str
    requires_separate_placement: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Keep the long-standing metadata key while making the plan explicit.
        data["set_cmd"] = self.command
        return data


def effective_dyn_refine_enabled(inputs: Any) -> bool:
    """Shared Dyn Mesh gate: None falls back to enable_local_refinement (default True)."""
    dyn = getattr(inputs, "enable_dyn_refine", None)
    if dyn is None:
        return bool(getattr(inputs, "enable_local_refinement", True))
    return bool(dyn)


def outer_band_level_string(inputs: Any) -> Optional[str]:
    """Return snappy outer-band level string if the band will be emitted, else None.

    Matches Generator3D._charge_outer_refine_levels emission rules using the shared
    effective Dyn Mesh gate. Fixed mesh never emits the outer band.
    """
    if not effective_dyn_refine_enabled(inputs):
        return None
    if getattr(inputs, "charge_outer_refine_enable", None) is False:
        return None
    rmin_outer = getattr(inputs, "charge_outer_refine_min", None)
    rmax_outer = getattr(inputs, "charge_outer_refine_max", None)
    rmin = rmin_outer if rmin_outer is not None else getattr(inputs, "refine_min", 2)
    rmax = rmax_outer if rmax_outer is not None else getattr(inputs, "refine_max", 3)
    if int(rmin) == 0 and int(rmax) == 0:
        return None
    if not getattr(inputs, "enable_local_refinement", True):
        return "2 2"
    rmax = max(int(rmin), int(rmax))
    return f"{int(rmin)} {rmax}"


def outer_band_will_be_applied(inputs: Any) -> bool:
    """True when the generator will emit a non-zero outer charge-refinement band."""
    level_str = outer_band_level_string(inputs)
    if not level_str:
        return False
    parts = level_str.split()
    try:
        levels = [int(p) for p in parts]
    except ValueError:
        return False
    return max(levels) > 0


def build_initialization_plan(inputs: Any) -> InitializationPlan:
    requested = max(0, int(getattr(inputs, "charge_refinement_level", 0) or 0))
    shape = str(getattr(inputs, "charge_shape", "Sphere") or "Sphere")
    remap = bool(getattr(inputs, "remap_enabled", False))
    dyn = effective_dyn_refine_enabled(inputs)

    if remap:
        return InitializationPlan(
            command="remap_radial.py",
            uses_set_refined_fields=False,
            performs_internal_refinement=False,
            seed_requested=requested,
            seed_effective=0,
            reason="remap initialization supplies fields from the selected source case",
            requires_separate_placement=True,
        )
    if not dyn:
        return InitializationPlan(
            command="setFields",
            uses_set_refined_fields=False,
            performs_internal_refinement=False,
            seed_requested=requested,
            seed_effective=0,
            reason="fixed mesh disables startup internal charge refinement",
        )
    if requested <= 0:
        return InitializationPlan(
            command="setFields",
            uses_set_refined_fields=False,
            performs_internal_refinement=False,
            seed_requested=requested,
            seed_effective=0,
            reason="requested internal charge seed level is zero",
        )
    if shape not in ("Sphere", "Cylinder", "Cuboid"):
        raise ValueError(f"Unsupported 3D charge shape for initialization: {shape}")
    return InitializationPlan(
        command="setRefinedFields",
        uses_set_refined_fields=True,
        performs_internal_refinement=True,
        seed_requested=requested,
        seed_effective=requested,
        reason=f"{shape} seed level {requested} requires refineInternal",
    )
