"""Charge capture radius for 3D setRefinedFields (sphericalMassToCell / cylindericalMassToCell).

The capture radius is written to the ``backup {{ ... }}`` sub-dictionary in setFieldsDict.
It is only for seeding the explosive on a coarse base mesh — not a physical charge size,
not a general-purpose refinement bubble, and not the snappy outer transition zone.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple


def base_cell_spacings_m(inputs: Any) -> Tuple[float, float, float]:
    """Uniform blockMesh cells: dx = dy = dz = cell_size."""
    h = max(1e-9, float(getattr(inputs, "cell_size", 0.5) or 0.5))
    return (h, h, h)


# Safety multiplier on the half cell-diagonal for the geometric capture term.
# 0.5*sqrt(dx^2+dy^2+dz^2) is the half cell-diagonal == the WORST-CASE distance from an
# arbitrary point (e.g. a charge centred on a cell corner) to the nearest base-cell centre.
# With factor 1.0 the capture sphere's surface passes exactly through that centre, so on a
# coarse mesh no cell centre is strictly inside and setRefinedFields selects zero cells
# ("No cells were selected"). A multiplier > 1 guarantees at least one base-cell centre lies
# strictly inside the capture sphere for ANY charge placement on the grid, which is what the
# setRefinedFields ``backup`` region needs to bootstrap refineInternal. This does not change
# the captured mass (sphericalMassToCell/cylindericalMassToCell set mass from the charge
# radius, not the backup radius); it only guarantees the bootstrap selection is non-empty.
CAPTURE_CELL_SAFETY = 1.5


def auto_charge_capture_radius_m(
    r_charge: float,
    dx: float,
    dy: float,
    dz: float,
    capture_factor: float,
) -> float:
    """Geometric auto capture radius (documented GUI policy).

    R_capture = max(
        1.05 * R_charge,
        0.5 * sqrt(dx^2 + dy^2 + dz^2) * max(captureFactor, CAPTURE_CELL_SAFETY)
    )

    The geometric term is floored by CAPTURE_CELL_SAFETY (> 1) so the capture sphere
    strictly encloses at least one base-cell centre even for charges smaller than a base
    cell that sit on a cell corner (otherwise capture fails on coarse meshes). A user
    captureFactor larger than the floor is still honoured.
    """
    r_c = max(0.0, float(r_charge))
    cf = max(1e-9, float(capture_factor))
    diag = math.sqrt(max(0.0, float(dx) ** 2 + float(dy) ** 2 + float(dz) ** 2))
    cell_term = 0.5 * diag * max(cf, CAPTURE_CELL_SAFETY)
    return max(1.05 * r_c, cell_term)


@dataclass
class ChargeCaptureReport:
    mode: str  # "auto" | "manual"
    physical_charge_radius_m: float
    base_dx_m: float
    base_dy_m: float
    base_dz_m: float
    charge_capture_factor: Optional[float]
    charge_capture_radius_used_m: float
    ratio_capture_to_physical: float
    formula_description: str
    warnings: List[str]

    def as_json_dict(self) -> dict:
        d = asdict(self)
        return d


def resolve_charge_capture_radius_m(
    inputs: Any,
    r_phys: float,
) -> Tuple[float, ChargeCaptureReport]:
    """Return (radius_for_setfields_backup_m, report). No hidden minimums."""
    dx, dy, dz = base_cell_spacings_m(inputs)
    r_phys = float(r_phys)
    rp = max(1e-12, abs(r_phys))

    mode = str(getattr(inputs, "charge_capture_mode", "auto") or "auto").strip().lower()
    if mode not in ("auto", "manual"):
        mode = "auto"

    manual_raw = getattr(inputs, "charge_capture_radius", None)
    legacy_ov = getattr(inputs, "charge_backup_radius_override", None)

    manual_f: Optional[float] = None
    if manual_raw is not None:
        try:
            manual_f = float(manual_raw)
        except (TypeError, ValueError):
            manual_f = None
    if manual_f is None and legacy_ov is not None:
        try:
            manual_f = float(legacy_ov)
            mode = "manual"
        except (TypeError, ValueError):
            pass

    factor = float(getattr(inputs, "charge_capture_factor", 1.0) or 1.0)

    warnings: List[str] = []
    cap_fact: Optional[float] = None

    if mode == "manual" and manual_f is not None and manual_f > 0:
        r_cap = float(manual_f)
        desc = (
            "Manual charge capture radius: exact user value (no hidden minimum, no cell-size floor)."
        )
    else:
        if mode == "manual" and manual_f is not None and manual_f <= 0:
            warnings.append(
                "Charge capture mode was manual but radius was invalid; fell back to auto capture."
            )
        mode = "auto"
        cap_fact = factor
        r_cap = auto_charge_capture_radius_m(r_phys, dx, dy, dz, factor)
        desc = (
            "Auto charge capture radius: max(1.05 * R_charge, "
            "0.5 * sqrt(dx² + dy² + dz²) * max(charge_capture_factor, "
            f"{CAPTURE_CELL_SAFETY})). The half cell-diagonal is floored by a safety factor "
            "so at least one base-cell centre is strictly enclosed (robust sub-cell capture)."
        )

    # Manual radius: advisory only (never changes r_cap). Order: physical → marginal → geometric
    # → large-radius (4× / 2×).
    if mode == "manual":
        r_geom = 0.5 * math.sqrt(max(0.0, dx * dx + dy * dy + dz * dz))
        r_ch = float(r_phys)
        if r_ch > 0.0:
            if r_cap < r_ch:
                warnings.append(
                    "The manual charge capture radius is smaller than the physical charge radius. "
                    "The charge may not be seeded correctly. Increase the capture radius or use Auto mode."
                )
            elif r_cap < 1.05 * r_ch:
                warnings.append(
                    "The manual charge capture radius is very close to the physical charge radius. "
                    "On a coarse base mesh, this may fail to capture enough cells for setRefinedFields."
                )
        if r_cap < r_geom:
            warnings.append(
                "The manual charge capture radius is smaller than half the base-cell diagonal. "
                "If the charge center is not near a cell center, the charge may not be captured on the current base mesh."
            )

    ratio = r_cap / rp
    if r_cap > 4.0 * rp:
        warnings.append(
            "STRONG WARNING: The charge capture radius is more than 4× the physical charge radius. "
            "This may create a very large initial refined region and make AMR unrefinement behind "
            "the shock more difficult."
        )
    elif r_cap > 2.0 * rp:
        warnings.append(
            "WARNING: The charge capture radius is larger than 2× the physical charge radius. "
            "This may create a large initial refined region and make AMR unrefinement behind "
            "the shock more difficult."
        )

    report = ChargeCaptureReport(
        mode=mode,
        physical_charge_radius_m=r_phys,
        base_dx_m=dx,
        base_dy_m=dy,
        base_dz_m=dz,
        charge_capture_factor=cap_fact,
        charge_capture_radius_used_m=r_cap,
        ratio_capture_to_physical=ratio,
        formula_description=desc,
        warnings=list(warnings),
    )
    return r_cap, report
