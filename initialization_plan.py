"""Authoritative 3D field-initialization decision.

This module is deliberately free of Qt and generator imports so the GUI,
generator, metadata, and tests can consume exactly the same decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from charge_seed_plan import (
    SEED_MODE_OFF,
    build_charge_seed_plan,
    resolve_seed_mode,
)


@dataclass(frozen=True)
class InitializationPlan:
    command: str
    uses_set_refined_fields: bool
    performs_internal_refinement: bool
    seed_requested: int
    seed_effective: int
    reason: str
    requires_separate_placement: bool = False
    seed_mode: str = "Off"

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


def outer_band_effective_level(inputs: Any) -> Optional[int]:
    """Single mode-inside refinement level for chargeRefineOuter, or None if off."""
    if not effective_dyn_refine_enabled(inputs):
        return None
    enable = getattr(inputs, "charge_outer_refine_enable", False)
    if enable is False:
        return None
    # Legacy None means enabled (pre-migration projects).
    if enable is None:
        enable = True
    if not enable:
        return None
    level = getattr(inputs, "charge_outer_refine_level", None)
    if level is None:
        rmax = getattr(inputs, "charge_outer_refine_max", None)
        rmin = getattr(inputs, "charge_outer_refine_min", None)
        if rmax is None and rmin is None:
            rmax = getattr(inputs, "refine_max", 3)
            rmin = getattr(inputs, "refine_min", 2)
        try:
            a = int(rmin) if rmin is not None else 0
            b = int(rmax) if rmax is not None else a
            level = max(a, b)
        except (TypeError, ValueError):
            level = 0
    try:
        level_i = int(level)
    except (TypeError, ValueError):
        return None
    if level_i <= 0:
        return None
    return level_i


def outer_band_level_string(inputs: Any) -> Optional[str]:
    """Return snappy outer-band level token if the band will be emitted, else None.

    For mode inside, GGUI emits a single effective level (not a min/max pair).
    """
    level = outer_band_effective_level(inputs)
    if level is None:
        return None
    if not getattr(inputs, "enable_local_refinement", True) and getattr(
        inputs, "charge_outer_refine_enable", None
    ) is None:
        # Extremely old path: local refinement globally off — keep prior "2" emission.
        return "2"
    return str(int(level))


def outer_band_will_be_applied(inputs: Any) -> bool:
    """True when the generator will emit a non-zero outer charge-refinement band."""
    return outer_band_effective_level(inputs) is not None


def build_initialization_plan(inputs: Any) -> InitializationPlan:
    seed_plan = build_charge_seed_plan(inputs)
    mode = seed_plan.mode
    requested = int(seed_plan.level_requested)
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
            seed_mode=mode,
        )
    if not dyn:
        return InitializationPlan(
            command="setFields",
            uses_set_refined_fields=False,
            performs_internal_refinement=False,
            seed_requested=requested,
            seed_effective=0,
            reason="fixed mesh disables startup internal charge refinement",
            seed_mode=mode,
        )
    if mode == SEED_MODE_OFF or seed_plan.level_effective <= 0:
        return InitializationPlan(
            command="setFields",
            uses_set_refined_fields=False,
            performs_internal_refinement=False,
            seed_requested=requested,
            seed_effective=0,
            reason=seed_plan.reason if mode == SEED_MODE_OFF else "effective internal charge seed level is zero",
            seed_mode=mode,
        )
    if shape not in ("Sphere", "Cylinder", "Cuboid"):
        raise ValueError(f"Unsupported 3D charge shape for initialization: {shape}")
    return InitializationPlan(
        command="setRefinedFields",
        uses_set_refined_fields=True,
        performs_internal_refinement=True,
        seed_requested=requested,
        seed_effective=int(seed_plan.level_effective),
        reason=seed_plan.reason,
        seed_mode=mode,
    )
