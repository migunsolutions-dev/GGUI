"""Authoritative 3D field-initialization decision.

This module is deliberately free of Qt and generator imports so the GUI,
generator, metadata, and tests can consume exactly the same decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


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


def build_initialization_plan(inputs: Any) -> InitializationPlan:
    requested = max(0, int(getattr(inputs, "charge_refinement_level", 0) or 0))
    shape = str(getattr(inputs, "charge_shape", "Sphere") or "Sphere")
    remap = bool(getattr(inputs, "remap_enabled", False))
    dyn = getattr(inputs, "enable_dyn_refine", None)
    if dyn is None:
        dyn = bool(getattr(inputs, "enable_local_refinement", True))

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
