"""Pure validation and derived calculations for axisymmetric 2D cases."""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

from charge_seed_plan import ChargeSeedPlan, build_charge_seed_plan
from models_2d import CaseInputs2D
from physical_charge_geometry import PhysicalChargeGeometry, physical_charge_geometry


DIRECT_SOURCE = "Direct Charge"
REMAP_SOURCE = "From 1D"
FIXED_MESH = "Fixed Mesh"
DYNAMIC_MESH = "Dynamic Mesh (AMR)"
BOUNDARY_OPEN = "Open"
BOUNDARY_SLIP = "Reflecting slip wall"
REQUIRED_FIELDS = ("p", "T", "U", "rho.air", "rho.c4", "alpha.c4")


@dataclass(frozen=True)
class AxisymmetricDomain:
    requested_radius: float
    requested_height: float
    effective_radius: float
    effective_height: float
    cell_size: float
    radial_cells: int
    vertical_cells: int
    adjusted: bool

    @property
    def total_cells(self) -> int:
        """Computational wedge cells; display mirroring never changes this value."""
        return self.radial_cells * self.vertical_cells

    def to_metadata(self) -> Dict[str, Any]:
        return asdict(self) | {"total_computational_cells": self.total_cells}


@dataclass(frozen=True)
class ValidationResult2D:
    domain: Optional[AxisymmetricDomain]
    charge: Optional[PhysicalChargeGeometry]
    seed_plan: Optional[ChargeSeedPlan]
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> "ValidationResult2D":
        if self.errors:
            raise ValueError("\n".join(self.errors))
        return self


@dataclass(frozen=True)
class MappingValidationReport:
    valid: bool
    source_case: str
    source_time: str
    target_cell_count: int
    mapped_radius: float
    mapped_fields: Tuple[str, ...]
    missing_fields: Tuple[str, ...]
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]
    source_cell_count: Optional[int] = None
    source_resolution: Optional[float] = None
    target_resolution: Optional[float] = None
    mass_before: Optional[float] = None
    mass_after: Optional[float] = None
    energy_before: Optional[float] = None
    energy_after: Optional[float] = None
    conservation_verified: bool = False

    remap_source_type: str = ""
    source_physical_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["method"] = (
            "radial interpolation of the 1D spherical profile about the target "
            "charge centre [0, HOB, 0]; ambient outside the 1D extent; "
            "no below-ground mirror"
        )
        data["conservative"] = False
        data["conservation_note"] = (
            "Mass and energy are not reported unless an independently verified "
            "integration definition is available."
        )
        return data


def align_axisymmetric_domain(radius: float, height: float, cell_size: float) -> AxisymmetricDomain:
    """Align domain to integer cells without shrinking below the request.

    User-entered radius and height are minimum requested dimensions. Domain size
    affects wave propagation and boundary-reflection timing, so reducing either
    dimension below the request is not acceptable. Cell counts therefore use
    ``math.ceil(requested / cell_size)`` rather than rounding to nearest.
    """
    r = float(radius)
    h = float(height)
    dx = float(cell_size)
    if not all(math.isfinite(v) for v in (r, h, dx)) or min(r, h, dx) <= 0:
        raise ValueError("Radius, Height and Base Cell Size must be finite and > 0.")
    nr = max(1, int(math.ceil(r / dx - 1e-15)))
    nz = max(1, int(math.ceil(h / dx - 1e-15)))
    er = nr * dx
    eh = nz * dx
    tol_r = 1e-9 * max(1.0, r)
    tol_h = 1e-9 * max(1.0, h)
    return AxisymmetricDomain(
        requested_radius=r,
        requested_height=h,
        effective_radius=er,
        effective_height=eh,
        cell_size=dx,
        radial_cells=nr,
        vertical_cells=nz,
        adjusted=abs(er - r) > tol_r or abs(eh - h) > tol_h,
    )


def _charge_fits(inputs: CaseInputs2D, domain: AxisymmetricDomain, charge: PhysicalChargeGeometry) -> bool:
    zc = float(inputs.height_of_burst)
    if charge.shape == "Sphere":
        radial = charge.radius_m
        half_height = charge.radius_m
    else:
        radial = charge.cylinder_radius_m
        half_height = 0.5 * charge.length_m
    if radial > domain.effective_radius + 1e-12:
        return False
    if zc + half_height > domain.effective_height + 1e-12:
        return False
    # VIPER ground burst: complete sphere may be centred on a reflecting bottom
    # (HOB=0). The computational domain holds the upper hemisphere; do not
    # require zc - r >= 0 in that case.
    if zc - half_height >= -1e-12:
        return True
    return (
        charge.shape == "Sphere"
        and inputs.bottom_boundary == BOUNDARY_SLIP
        and abs(zc) <= 1e-12
        and abs(float(inputs.charge_center_r)) <= 1e-12
    )


def validate_case_inputs_2d(inputs: CaseInputs2D) -> ValidationResult2D:
    errors = []
    warnings = []
    domain = None
    charge = None
    seed_plan = None

    if inputs.cell_size is None:
        errors.append("Base Cell Size is undefined.")
    else:
        try:
            domain = align_axisymmetric_domain(inputs.radius, inputs.height, inputs.cell_size)
            if domain.adjusted:
                warnings.append(
                    "Effective domain was aligned to the requested Base Cell Size: "
                    f"R={domain.effective_radius:.6g} m, H={domain.effective_height:.6g} m "
                    f"({domain.radial_cells}×{domain.vertical_cells} cells)."
                )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    if inputs.initialization_source not in (DIRECT_SOURCE, REMAP_SOURCE):
        errors.append("Initialization Source must be Direct Charge or From 1D.")
    if inputs.mesh_mode not in (FIXED_MESH, DYNAMIC_MESH):
        errors.append("Mesh Mode must be Fixed Mesh or Dynamic Mesh (AMR).")
    if abs(float(inputs.charge_center_r)) > 1e-12:
        errors.append("Axisymmetric charge centre Radius r is locked to 0.")
    if abs(float(inputs.detonation_radius)) > 1e-12:
        errors.append("Axisymmetric detonation Radius r is locked to 0.")
    if inputs.charge_shape not in ("Sphere", "Cylinder"):
        errors.append("Axisymmetric Direct Charge supports only Sphere or axial Cylinder.")
    for name, preset in (
        ("Outer Radius", inputs.outer_boundary),
        ("Top", inputs.top_boundary),
        ("Ground / Bottom", inputs.bottom_boundary),
    ):
        if preset not in (BOUNDARY_OPEN, BOUNDARY_SLIP):
            errors.append(f"{name} boundary must be Open or Reflecting slip wall.")
    if inputs.refine_indicator_field != "densityGradient":
        errors.append("Only the validated densityGradient runtime AMR estimator is supported.")
    if inputs.cores < 1:
        errors.append("Processor cores must be at least 1.")
    if inputs.max_co <= 0 or inputs.end_time_s <= 0 or inputs.delta_t <= 0:
        errors.append("maxCo, End Time and initial time step must be > 0.")

    if inputs.initialization_source == DIRECT_SOURCE:
        try:
            charge = physical_charge_geometry(inputs)
            if domain is not None and not _charge_fits(inputs, domain, charge):
                errors.append("The complete on-axis charge must fit inside the effective 2D domain.")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

        if inputs.mesh_mode == DYNAMIC_MESH:
            try:
                seed_plan = build_charge_seed_plan(inputs)
                if not seed_plan.is_safe:
                    errors.append(seed_plan.reason)
                warnings.extend(seed_plan.warnings)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        elif charge is not None:
            cells = charge.d_min_m / max(float(inputs.cell_size), 1e-12)
            if cells + 1e-9 < float(inputs.charge_seed_min_cells):
                errors.append(
                    "Fixed Mesh cannot resolve the charge at the requested Base Cell Size "
                    f"({cells:.2f} cells across {charge.d_min_name}; minimum "
                    f"{inputs.charge_seed_min_cells}). Reduce Base Cell Size."
                )
    else:
        source = inputs.mapping
        if not (source.case_path or "").strip():
            errors.append("From 1D requires a source case.")
        if source.time_mode not in ("latest", "specific"):
            errors.append("Source time mode must be latest or specific.")
        if source.time_mode == "specific" and not (source.specific_time or "").strip():
            errors.append("A specific source time is required.")
        if source.mapped_radius <= 0:
            errors.append("Mapped radius must be > 0.")
        if domain is not None and source.mapped_radius > min(
            domain.effective_radius, domain.effective_height
        ):
            errors.append("The target domain does not contain the requested mapped radius.")

    for probe in inputs.probes:
        if probe.radius < 0 or probe.height < 0:
            errors.append(f"Probe {probe.name!r} coordinates must be non-negative.")
        elif domain and (
            probe.radius > domain.effective_radius or probe.height > domain.effective_height
        ):
            errors.append(f"Probe {probe.name!r} lies outside the computational domain.")

    return ValidationResult2D(
        domain=domain,
        charge=charge,
        seed_plan=seed_plan,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def resolve_source_time(case_path: str, time_mode: str, specific_time: str = "") -> Optional[str]:
    if time_mode == "specific":
        return specific_time.strip() or None
    values = []
    try:
        entries = os.listdir(case_path)
    except OSError:
        return None
    for entry in entries:
        try:
            value = float(entry)
        except ValueError:
            continue
        if value >= 0 and os.path.isdir(os.path.join(case_path, entry)):
            values.append((value, entry))
    return max(values, default=(None, None))[1]


def validate_mapping_source(
    inputs: CaseInputs2D,
    *,
    required_fields: Iterable[str] = REQUIRED_FIELDS,
) -> MappingValidationReport:
    result = validate_case_inputs_2d(inputs)
    source = inputs.mapping
    source_case = os.path.abspath(source.case_path) if source.case_path else ""
    source_time = resolve_source_time(source_case, source.time_mode, source.specific_time)
    errors = list(result.errors)
    warnings = list(result.warnings)
    mapped = []
    missing = []
    remap_source_type = ""
    source_physical_time = None
    snapshot_ok = False
    remap_blocked = False
    if not os.path.isdir(source_case):
        errors.append("The selected 1D source case does not exist.")
    else:
        from case_topology import CaseDimension, classify_case_topology
        from remap_snapshot_1d import SOURCE_SNAPSHOT, resolve_remap_source

        classification = classify_case_topology(source_case)
        if classification.classification != CaseDimension.AXISYMMETRIC_WEDGE:
            errors.append(
                "The source case is not verified as an axisymmetric wedge "
                f"({classification.classification.value}: {classification.reason})."
            )
        resolved = resolve_remap_source(source_case)
        remap_source_type = resolved.source_type
        source_physical_time = resolved.physical_time
        if resolved.blocked:
            errors.append(resolved.message)
            remap_blocked = True
        elif resolved.ok and resolved.source_type == SOURCE_SNAPSHOT:
            source_time = resolved.time_label
            mapped = list(resolved.field_names) or ["p", "T", "U"]
            missing = []
            snapshot_ok = True
            warnings.append(resolved.message)
        elif resolved.ok:
            source_time = resolved.time_label or source_time
            warnings.append(resolved.message)
        else:
            errors.append(resolved.message)
        phase_path = os.path.join(source_case, "constant", "phaseProperties")
        try:
            with open(phase_path, "r", encoding="utf-8", errors="ignore") as stream:
                phase_text = stream.read()
            if "JWL" not in phase_text or "phases" not in phase_text:
                errors.append("Source material/EOS is not compatible with the target JWL case.")
            rho_tokens = (
                f"rho0 {inputs.rho_charge:g}",
                f"rho0 {inputs.rho_charge:.1f}",
            )
            if not any(token in phase_text for token in rho_tokens):
                warnings.append(
                    "Source explosive density could not be matched exactly to the target; "
                    "review phaseProperties before mapping."
                )
        except OSError:
            errors.append("Source constant/phaseProperties is missing.")
    if snapshot_ok or remap_blocked:
        pass
    elif not source_time:
        errors.append("The selected source time does not exist.")
    else:
        tdir = os.path.join(source_case, source_time)
        if not os.path.isdir(tdir):
            errors.append(f"Source time {source_time!r} does not exist.")
        else:
            for name in required_fields:
                if os.path.isfile(os.path.join(tdir, name)):
                    mapped.append(name)
                else:
                    missing.append(name)
            if missing:
                errors.append("Required source fields are missing: " + ", ".join(missing))
    warnings.append(
        "Radial mapping about the target charge centre [0, HOB, 0] is not conservative."
    )
    domain = result.domain
    return MappingValidationReport(
        valid=not errors,
        source_case=source_case,
        source_time=source_time or "",
        target_cell_count=domain.total_cells if domain else 0,
        mapped_radius=source.mapped_radius,
        mapped_fields=tuple(mapped),
        missing_fields=tuple(missing),
        warnings=tuple(dict.fromkeys(warnings)),
        errors=tuple(dict.fromkeys(errors)),
        source_resolution=source.source_resolution,
        target_resolution=inputs.cell_size,
        remap_source_type=remap_source_type,
        source_physical_time=source_physical_time,
    )
