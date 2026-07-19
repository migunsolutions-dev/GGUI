"""Non-mutating preflight for unsafe seed-0/no-band 3D charge capture."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from mesh_domain import align_domain_to_cell_size


@dataclass(frozen=True)
class CaptureGuardResult:
    safe: bool
    cell_centre_inside: bool
    reason: str


def _charge_extents(inputs: Any) -> Tuple[float, float, float]:
    volume = float(inputs.mass_kg) / float(inputs.rho_charge)
    shape = str(inputs.charge_shape)
    if shape == "Sphere":
        r = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        return r, r, r
    if shape == "Cuboid":
        length = float(getattr(inputs, "charge_length", 0.0) or 0.0)
        width = float(getattr(inputs, "charge_width", 0.0) or 0.0)
        height = float(getattr(inputs, "charge_height", 0.0) or 0.0)
        if not (
            length > 0
            and width > 0
            and height > 0
            and abs(length * width * height - volume) <= 0.02 * volume
        ):
            length = width = height = volume ** (1.0 / 3.0)
        return length / 2.0, width / 2.0, height / 2.0
    radius = float(getattr(inputs, "cylinder_radius", 0.05) or 0.05)
    length = float(getattr(inputs, "charge_length", 0.0) or 0.0)
    if length <= 0:
        aspect = float(getattr(inputs, "charge_aspect", 0.0) or 0.0)
        length = 2.0 * radius * aspect if aspect > 0 else volume / (math.pi * radius * radius)
    axis = str(getattr(inputs, "cylinder_axis", "X")).upper()
    half = length / 2.0
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
    "• enable Dyn Mesh and select an internal charge refinement level greater than zero; or\n"
    "• deliberately enable the advanced outer refinement band."
)


def evaluate_unsafe_capture(inputs: Any) -> CaptureGuardResult:
    seed = max(0, int(getattr(inputs, "charge_refinement_level", 0) or 0))
    band_enabled = bool(getattr(inputs, "charge_outer_refine_enable", False))
    band_levels = max(
        int(getattr(inputs, "charge_outer_refine_min", 0) or 0),
        int(getattr(inputs, "charge_outer_refine_max", 0) or 0),
    )
    if seed > 0:
        return CaptureGuardResult(True, False, "internal seed protects startup capture")
    if band_enabled and band_levels > 0:
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
    """Raise ValueError before writing an unsafe non-remap 3D case. Non-mutating."""
    if getattr(inputs, "remap_enabled", False):
        return
    guard = evaluate_unsafe_capture(inputs)
    if not guard.safe:
        raise ValueError(UNSAFE_CAPTURE_MESSAGE)
