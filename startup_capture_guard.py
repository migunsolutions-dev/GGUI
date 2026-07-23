"""Non-mutating preflight for unsafe seed-0/no-band 3D charge capture."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from charge_seed_plan import (
    SEED_MODE_AUTO,
    UNSAFE_AUTO_SEED_MESSAGE,
    build_charge_seed_plan,
)
from initialization_plan import (
    build_initialization_plan,
    outer_band_will_be_applied,
)
from mesh_domain import align_domain_to_cell_size


@dataclass(frozen=True)
class CaptureGuardResult:
    safe: bool
    cell_centre_inside: bool
    reason: str


def _charge_extents(inputs: Any) -> Tuple[float, float, float]:
    """Half-extents from authoritative physical_charge_geometry."""
    from physical_charge_geometry import physical_charge_geometry

    geom = physical_charge_geometry(inputs)
    shape = geom.shape
    if shape == "Sphere":
        r = geom.radius_m
        return r, r, r
    if shape == "Cuboid":
        return geom.length_box_m / 2.0, geom.width_m / 2.0, geom.height_m / 2.0
    radius = geom.cylinder_radius_m
    half = geom.length_m / 2.0
    axis = str(getattr(inputs, "cylinder_axis", "Z")).upper()
    return (
        (half, radius, radius)
        if axis == "X"
        else (radius, half, radius)
        if axis == "Y"
        else (radius, radius, half)
    )


def _candidate_indices(lo: float, hi: float, origin: float, dx: float, count: int) -> Iterable[int]:
    first = max(0, int(math.ceil((lo - origin) / dx - 0.5)))
    last = min(count - 1, int(math.floor((hi - origin) / dx - 0.5)))
    return range(first, last + 1) if first <= last else ()


def aligned_base_cell_centre_inside(inputs: Any) -> bool:
    alignment = align_domain_to_cell_size(inputs.min_point, inputs.max_point, inputs.cell_size)
    mins = alignment.min_point
    counts = (alignment.nx, alignment.ny, alignment.nz)
    dx = float(inputs.cell_size)
    center = tuple(float(v) for v in inputs.charge_center)
    ext = _charge_extents(inputs)
    ranges = [
        _candidate_indices(center[i] - ext[i], center[i] + ext[i], mins[i], dx, counts[i])
        for i in range(3)
    ]
    shape = str(inputs.charge_shape)
    radius = ext[1] if shape == "Cylinder" and str(inputs.cylinder_axis).upper() == "X" else ext[0]
    axis = str(getattr(inputs, "cylinder_axis", "X")).upper()
    for ix in ranges[0]:
        x = mins[0] + (ix + 0.5) * dx
        for iy in ranges[1]:
            y = mins[1] + (iy + 0.5) * dx
            for iz in ranges[2]:
                z = mins[2] + (iz + 0.5) * dx
                q = (x - center[0], y - center[1], z - center[2])
                if shape == "Sphere" and sum(v * v for v in q) <= ext[0] ** 2:
                    return True
                if shape == "Cuboid" and all(abs(q[i]) <= ext[i] for i in range(3)):
                    return True
                if shape == "Cylinder":
                    axial = {"X": 0, "Y": 1, "Z": 2}.get(axis, 0)
                    radial2 = sum(q[i] ** 2 for i in range(3) if i != axial)
                    if abs(q[axial]) <= ext[axial] and radial2 <= radius ** 2:
                        return True
    return False


UNSAFE_CAPTURE_MESSAGE = (
    "Initialization is blocked because no applied internal seed or "
    "outer refinement band protects capture, and the aligned base mesh "
    "has no cell centre inside the physical charge.\n\n"
    "Choose one remedy without changing charge mass:\n"
    "• reduce the base cell size;\n"
    "• enable Dyn Mesh with Auto seed (or Manual seed level > 0); or\n"
    "• deliberately enable the advanced outer refinement band."
)


def evaluate_unsafe_capture(inputs: Any) -> CaptureGuardResult:
    """Non-mutating guard using the same Dyn Mesh / init-plan / outer-band rules as generation.

    Requested seed or outer-band UI values protect capture only when they will
    actually be applied (Dyn Mesh + setRefinedFields / emitted snappy band).
    """
    plan = build_initialization_plan(inputs)
    if plan.uses_set_refined_fields and plan.seed_effective > 0:
        return CaptureGuardResult(True, False, "internal seed protects startup capture")
    if outer_band_will_be_applied(inputs):
        return CaptureGuardResult(True, False, "outer refinement band protects startup capture")
    inside = aligned_base_cell_centre_inside(inputs)
    return CaptureGuardResult(
        inside,
        inside,
        "aligned base mesh contains a charge cell centre"
        if inside
        else "no aligned base-mesh cell centre lies inside the physical charge",
    )


def require_safe_capture(inputs: Any) -> None:
    """Raise ValueError before writing an unsafe non-remap 3D case. Non-mutating.

    Policy (``build_initialization_plan`` is authoritative):

    1. Remap — bypass entirely.
    2. Effective Auto seed with ``setRefinedFields`` — enforce Auto seed safety.
    3. Fixed Mesh / seed Off / Manual — never raise the Auto-seed-specific error;
       fall through to the physical base-cell-centre capture check when no
       applied seed or outer band protects capture.
    """
    if getattr(inputs, "remap_enabled", False):
        return
    plan = build_initialization_plan(inputs)
    # Auto-seed target failure only when the effective plan will actually run
    # setRefinedFields with a non-zero Auto seed — not raw GUI selection alone.
    if (
        plan.uses_set_refined_fields
        and int(plan.seed_effective) > 0
        and plan.seed_mode == SEED_MODE_AUTO
    ):
        seed_plan = build_charge_seed_plan(inputs)
        if not seed_plan.is_safe:
            raise ValueError(
                UNSAFE_AUTO_SEED_MESSAGE.format(
                    min_cells=seed_plan.min_cells,
                    d_min=seed_plan.d_min_m,
                    max_level=seed_plan.max_level,
                    achieved=seed_plan.achieved_cells,
                    h_seed=seed_plan.h_seed_m,
                )
            )
    guard = evaluate_unsafe_capture(inputs)
    if not guard.safe:
        raise ValueError(UNSAFE_CAPTURE_MESSAGE)
