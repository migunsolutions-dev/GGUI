"""Source-state derivation for the Ideal-Gas Isothermal Burst blast source model.

At t = 0 the charge is already fully detonated. Its volume is replaced by the
same single calorically-perfect ideal gas that fills the rest of the domain, at
uniform density and uniform specific internal energy -- and therefore, since
``e = Cv*T`` and ``p = (gamma - 1)*rho*e``, at uniform temperature and uniform
pressure. There is no reaction zone, no detonation front, no product equation of
state and no activation model. ``E_charge`` is the specific detonation/chemical energy **added** to the
initial thermodynamic state, matching blastFoam activation-model ``e0``
semantics. It is not the final specific internal energy:

    e_initial = Cv * T_atm
    e_source  = e_initial + E_charge
    T_source  = T_atm + E_charge / Cv
    p_source  = (gamma - 1) * rho_source * e_source

The conserved detonation-energy quantity is
``DeltaE = E_final - E_initial = W * E_charge``. Total initialized
internal energy is larger than ``W * E_charge`` by the sensible energy
already present at ``T_atm``. This convention is not a VIPER/UFC/KB fit.

"Isothermal" describes the *initialization*, not a process assumption: the burst
region is isothermal in space at t = 0. The later expansion is whatever the Euler
equations produce.

blastFoam only needs ``p`` and ``rho``. ``fluidBlastThermo::initializeFields``
calls ``initializeEnergy``, which Newton-solves ``e`` until ``p(rho, e)`` matches
the ``p`` field; for an ideal gas that is linear, so ``e = p/((gamma-1)*rho)``
exactly, and ``T = e/Cv`` follows from ``correct()``. The production pressure
therefore initializes ``T_source = T_atm + E_charge/Cv``, not ``E_charge/Cv``.

Everything here is a closed-form function of the physical charge definition. There
are no fitted coefficients, no multipliers, no lookup tables and no dependence on
scaled distance. That is deliberate: BF-IG must not be calibrated against VIPER or
against any downstream pressure measurement.

Two geometric decisions are load-bearing, and both are driven by measurements of
the real GGUI 1D wedge mesh rather than by assumption:

1.  The source state is derived from **radii only**, never from summed mesh cell
    volumes. ``rho`` and ``e`` are intensive, so the physically correct
    initialization for a spherically symmetric problem is fixed by the physical
    spherical shell the source cells represent. GGUI's 1D ``blockMeshDict`` builds
    a *twisted* hex (the r_max vertices at -wedge_half are paired against the
    r_min vertices at +wedge_half on the axis side), so its cell volumes deviate
    from a true spherical cone by a factor that runs from 0.986 at r = 1.1 m down
    to 0.377 at the innermost cell. Deriving density from those volumes would be
    both wrong and mesh-dependent. See ``wedge_cone_volume_ratio``.

2.  The radius handed to ``setFields`` is snapped to a **cell face** radius. OpenFOAM's
    ``sphereToCell`` selects cells whose *centre* lies inside the sphere, and on a
    twisted mesh the cell-centre radius is not exactly the ideal frustum centroid.
    A face radius always lies strictly between the centres of the cells either side
    of it, so the selected cell count is exact for any centroid convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from models import SOURCE_MODEL_IG, SOURCE_MODEL_SCHEMA_VERSION

# Single-material architecture: one gas fills the whole domain, so the burst gas
# necessarily uses air's properties. These are the values the existing JWL case
# already gives its `air` phase, so BF-IG introduces no new thermo constant.
GAMMA_IDEAL_GAS = 1.4
CV_IDEAL_GAS = 718.0

DERIVATION_ID = "ig_isothermal_burst/radial_shell_face_snapped/detonation_added/v2"

# Rejected candidate: treat E_charge as the final specific internal energy.
# Kept only so audits can show the numerical difference.
ENERGY_CONVENTION_FINAL_INTERNAL = "final_internal_energy"

# Production: E_charge is energy added by detonation.
ENERGY_CONVENTION_DETONATION_ADDED = "detonation_energy_added"

PRODUCTION_ENERGY_CONVENTION = ENERGY_CONVENTION_DETONATION_ADDED
WORKING_ENERGY_CONVENTION = ENERGY_CONVENTION_DETONATION_ADDED


class IgSourceStateError(ValueError):
    """Raised when the charge definition cannot produce a valid burst state."""


def specific_gas_constant(gamma: float = GAMMA_IDEAL_GAS, cv: float = CV_IDEAL_GAS) -> float:
    """``R = (gamma - 1) * Cv``, the value implied by blastFoam's eConst + idealGas pair."""
    return (float(gamma) - 1.0) * float(cv)


def spherical_shell_volume(r_inner: float, r_outer: float) -> float:
    """Full-sphere volume of the shell ``[r_inner, r_outer]``."""
    return 4.0 / 3.0 * math.pi * (float(r_outer) ** 3 - float(r_inner) ** 3)


def frustum_centroid_radius(r_inner: float, r_outer: float) -> float:
    """Volume-centroid radius of a conical shell, ``(3/4)(rb^4-ra^4)/(rb^3-ra^3)``.

    Reported for diagnostics. The selection rule does not depend on it, by design.
    """
    a, b = float(r_inner), float(r_outer)
    denom = b ** 3 - a ** 3
    if denom <= 0.0:
        raise IgSourceStateError("frustum_centroid_radius requires r_outer > r_inner")
    return 0.75 * (b ** 4 - a ** 4) / denom


def _require_positive(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IgSourceStateError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise IgSourceStateError(f"{name} must be finite and > 0, got {number!r}")
    return number


@dataclass(frozen=True)
class AmbientState:
    """Uniform far-field gas state, derived so it is exactly consistent with the EOS."""

    p_atm: float
    t_atm: float
    gamma: float
    cv: float
    r_specific: float
    rho: float
    e: float


def ambient_state(
    p_atm: float,
    t_atm: float,
    gamma: float = GAMMA_IDEAL_GAS,
    cv: float = CV_IDEAL_GAS,
) -> AmbientState:
    """Ambient density from the EOS rather than a hard-coded 1.225 kg/m^3.

    ``rho = p/(R*T)`` is only 1.225 at exactly 101325 Pa / 288 K. Writing a constant
    at any other ambient seeds a spurious starting wave, because blastFoam derives
    ``e`` from the ``p`` and ``rho`` it is given.
    """
    p = _require_positive("p_atm", p_atm)
    t = _require_positive("t_atm", t_atm)
    g = _require_positive("gamma", gamma)
    c = _require_positive("cv", cv)
    if g <= 1.0:
        raise IgSourceStateError(f"gamma must be > 1, got {g!r}")
    r_specific = (g - 1.0) * c
    return AmbientState(
        p_atm=p,
        t_atm=t,
        gamma=g,
        cv=c,
        r_specific=r_specific,
        rho=p / (r_specific * t),
        e=c * t,
    )


@dataclass(frozen=True)
class ChargeGeometry:
    """The ideal continuum charge, before any mesh is involved."""

    mass_kg: float
    rho_charge: float
    energy_j_per_kg: float
    volume_m3: float
    radius_m: float
    source_energy_j: float


def charge_geometry(mass_kg: float, rho_charge: float, energy_j_per_kg: float) -> ChargeGeometry:
    mass = _require_positive("mass_kg", mass_kg)
    rho = _require_positive("rho_charge", rho_charge)
    energy = _require_positive("energy_j_per_kg", energy_j_per_kg)
    volume = mass / rho
    return ChargeGeometry(
        mass_kg=mass,
        rho_charge=rho,
        energy_j_per_kg=energy,
        volume_m3=volume,
        radius_m=(3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0),
        source_energy_j=mass * energy,
    )


@dataclass(frozen=True)
class SourceShell:
    """The discrete radial shell that will actually carry the burst state.

    The 1D mesh starts at ``r_min > 0`` with a reflecting ``origin`` patch, so the
    charge core is not part of the domain -- a pre-existing property of GGUI's 1D
    mesh, noted in ``profiles.compute_recommended_1d`` as "Missing mass is
    (r_min/R)^3". ``r_outer`` is chosen so the shell's full-sphere volume matches the
    charge volume as closely as a whole number of cells allows, which keeps
    ``rho_source`` close to ``rho_charge`` instead of inflating it to cover the core.
    """

    r_min_m: float
    cell_size_m: float
    n_cells: int
    r_outer_m: float
    set_fields_radius_m: float
    volume_full_sphere_m3: float
    radius_equivalent_m: float
    core_volume_fraction: float
    volume_ratio_to_charge: float


def source_shell(r_min_m: float, cell_size_m: float, charge_radius_m: float) -> SourceShell:
    """Pick the whole number of cells whose full-sphere shell volume best matches the charge.

    Matching *volume* rather than radius means ``r_outer`` compensates for the missing
    core: ``r_outer^3 = R_charge^3 + r_min^3`` in the continuum limit. The returned
    ``set_fields_radius_m`` is the cell-face radius, which selects exactly ``n_cells``
    cells regardless of how OpenFOAM places cell centres on the twisted wedge.
    """
    r_min = float(r_min_m)
    if not math.isfinite(r_min) or r_min < 0.0:
        raise IgSourceStateError(f"r_min_m must be finite and >= 0, got {r_min_m!r}")
    dx = _require_positive("cell_size_m", cell_size_m)
    r_charge = _require_positive("charge_radius_m", charge_radius_m)
    if r_min >= r_charge:
        raise IgSourceStateError(
            f"mesh inner radius {r_min!r} m is not inside the charge radius {r_charge!r} m; "
            "the source region would be empty"
        )

    # Volume-matched target: the shell [r_min, r_target] has exactly the charge volume.
    r_target = (r_charge ** 3 + r_min ** 3) ** (1.0 / 3.0)
    n_cells = int(round((r_target - r_min) / dx))
    n_cells = max(1, n_cells)
    r_outer = r_min + n_cells * dx

    volume = spherical_shell_volume(r_min, r_outer)
    charge_volume = 4.0 / 3.0 * math.pi * r_charge ** 3
    return SourceShell(
        r_min_m=r_min,
        cell_size_m=dx,
        n_cells=n_cells,
        r_outer_m=r_outer,
        set_fields_radius_m=r_outer,
        volume_full_sphere_m3=volume,
        radius_equivalent_m=(3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0),
        core_volume_fraction=(r_min / r_charge) ** 3,
        volume_ratio_to_charge=volume / charge_volume,
    )


@dataclass(frozen=True)
class IgBurstState:
    """The complete initialized burst state plus everything the audit has to report."""

    ambient: AmbientState
    charge: ChargeGeometry
    shell: SourceShell

    rho_source: float
    e_source: float
    p_source: float
    t_source: float
    speed_of_sound_source: float

    source_mass_kg: float
    e_initial_j_per_kg: float
    e_detonation_added_j_per_kg: float
    e_initial_sensible_j: float
    e_detonation_added_j: float
    e_final_internal_j: float
    delta_e_j: float
    mass_error_rel: float
    detonation_energy_error_rel: float

    ambient_energy_in_shell_j: float
    e_source_if_final_internal: float
    p_source_if_final_internal: float
    t_source_if_final_internal: float

    energy_convention: str = WORKING_ENERGY_CONVENTION
    production_energy_convention: Optional[str] = PRODUCTION_ENERGY_CONVENTION

    derivation_id: str = DERIVATION_ID
    source_model: str = SOURCE_MODEL_IG
    source_model_schema_version: int = SOURCE_MODEL_SCHEMA_VERSION


def derive_ig_state(
    *,
    mass_kg: float,
    rho_charge: float,
    energy_j_per_kg: float,
    p_atm: float,
    t_atm: float,
    r_min_m: float,
    cell_size_m: float,
    gamma: float = GAMMA_IDEAL_GAS,
    cv: float = CV_IDEAL_GAS,
) -> IgBurstState:
    """Derive the burst state from the physical charge definition.

    ``rho_source = W / V_shell`` makes the represented mass exactly ``W``.
    ``E_charge`` is added to the initial sensible energy:

        e_initial = Cv * T_atm
        e_source  = e_initial + E_charge
        T_source  = T_atm + E_charge / Cv
        p_source  = (gamma - 1) * rho_source * e_source
        DeltaE    = W * e_source - W * e_initial = W * E_charge
    """
    ambient = ambient_state(p_atm, t_atm, gamma=gamma, cv=cv)
    charge = charge_geometry(mass_kg, rho_charge, energy_j_per_kg)
    shell = source_shell(r_min_m, cell_size_m, charge.radius_m)

    rho_source = charge.mass_kg / shell.volume_full_sphere_m3
    e_initial = ambient.e
    e_added = charge.energy_j_per_kg
    e_source = e_initial + e_added
    t_source = ambient.t_atm + e_added / ambient.cv
    p_source = (ambient.gamma - 1.0) * rho_source * e_source
    e_if_final = e_added
    p_if_final = (ambient.gamma - 1.0) * rho_source * e_if_final

    source_mass = rho_source * shell.volume_full_sphere_m3
    e_initial_j = source_mass * e_initial
    e_det_j = source_mass * e_added
    e_final_j = source_mass * e_source
    delta_e = e_final_j - e_initial_j

    return IgBurstState(
        ambient=ambient,
        charge=charge,
        shell=shell,
        rho_source=rho_source,
        e_source=e_source,
        p_source=p_source,
        t_source=t_source,
        speed_of_sound_source=math.sqrt(ambient.gamma * p_source / rho_source),
        source_mass_kg=source_mass,
        e_initial_j_per_kg=e_initial,
        e_detonation_added_j_per_kg=e_added,
        e_initial_sensible_j=e_initial_j,
        e_detonation_added_j=e_det_j,
        e_final_internal_j=e_final_j,
        delta_e_j=delta_e,
        mass_error_rel=source_mass / charge.mass_kg - 1.0,
        detonation_energy_error_rel=delta_e / charge.source_energy_j - 1.0,
        ambient_energy_in_shell_j=ambient.rho * ambient.e * shell.volume_full_sphere_m3,
        e_source_if_final_internal=e_if_final,
        p_source_if_final_internal=p_if_final,
        t_source_if_final_internal=e_if_final / ambient.cv,
        energy_convention=WORKING_ENERGY_CONVENTION,
        production_energy_convention=PRODUCTION_ENERGY_CONVENTION,
    )


def wedge_cone_volume_ratio(cell_volumes: Tuple[float, ...], r_min_m: float, cell_size_m: float,
                            wedge_solid_angle: float) -> Tuple[float, ...]:
    """Per-cell ratio of measured mesh volume to the ideal conical shell volume.

    Mesh-quality diagnostic only. GGUI's twisted 1D wedge returns values well below 1
    near the axis, which is exactly why the source derivation ignores mesh volumes.
    ``cell_volumes`` must be ordered inner -> outer.
    """
    dx = _require_positive("cell_size_m", cell_size_m)
    r_min = float(r_min_m)
    out = []
    for i, v in enumerate(cell_volumes):
        a = r_min + i * dx
        b = a + dx
        ideal = wedge_solid_angle / 3.0 * (b ** 3 - a ** 3)
        out.append(float(v) / ideal if ideal > 0.0 else float("nan"))
    return tuple(out)


def wedge_solid_angle(axis_eps: float, cone_half: float, wedge_half: float) -> float:
    """True solid angle of the 1D spherical wedge, for mesh diagnostics."""
    return (math.cos(float(axis_eps)) - math.cos(float(cone_half))) * 2.0 * float(wedge_half)


def audit_dict(
    state: IgBurstState,
    *,
    case_path: Optional[str] = None,
    material_name: Optional[str] = None,
    measured_source_cells: Optional[int] = None,
    measured_wedge_source_volume_m3: Optional[float] = None,
) -> Dict[str, Any]:
    """The `ggui_ig_source_audit.json` payload.

    Everything a future engineer needs to reconstruct what was initialized, without
    reverse-engineering the generated dictionaries.
    """
    payload: Dict[str, Any] = {
        "source_model": state.source_model,
        "source_model_schema_version": state.source_model_schema_version,
        "derivation_id": state.derivation_id,
        "case_path": case_path,
        "material_name": material_name,
        "user_inputs": {
            "W_kg": state.charge.mass_kg,
            "rho_charge_kg_m3": state.charge.rho_charge,
            "E_charge_J_kg": state.charge.energy_j_per_kg,
            "p_atm_Pa": state.ambient.p_atm,
            "T_atm_K": state.ambient.t_atm,
        },
        "gas": {
            "gamma": state.ambient.gamma,
            "Cv_J_kgK": state.ambient.cv,
            "R_specific_J_kgK": state.ambient.r_specific,
        },
        "ambient": {
            "rho_kg_m3": state.ambient.rho,
            "e_J_kg": state.ambient.e,
            "p_Pa": state.ambient.p_atm,
            "T_K": state.ambient.t_atm,
        },
        "charge_ideal": {
            "V_charge_m3": state.charge.volume_m3,
            "R_charge_m": state.charge.radius_m,
            "intended_detonation_energy_J": state.charge.source_energy_j,
        },
        "source_region": {
            "r_min_m": state.shell.r_min_m,
            "cell_size_m": state.shell.cell_size_m,
            "n_source_cells": state.shell.n_cells,
            "r_outer_m": state.shell.r_outer_m,
            "set_fields_radius_m": state.shell.set_fields_radius_m,
            "V_source_full_sphere_m3": state.shell.volume_full_sphere_m3,
            "R_source_equivalent_m": state.shell.radius_equivalent_m,
            "V_source_over_V_charge": state.shell.volume_ratio_to_charge,
            "R_source_equivalent_over_R_charge": (
                state.shell.radius_equivalent_m / state.charge.radius_m
            ),
            "core_volume_fraction_excluded_by_mesh": state.shell.core_volume_fraction,
            "cells_across_charge_radius": state.charge.radius_m / state.shell.cell_size_m,
        },
        "energy_convention": {
            "production": state.production_energy_convention,
            "basis": (
                "E_charge is specific detonation energy added to Cv*T_atm. "
                "Chosen from blastFoam activation e0 semantics, not VIPER/UFC/KB fit."
            ),
            "rejected_final_internal_energy": {
                "e_J_kg": state.e_source_if_final_internal,
                "p_Pa": state.p_source_if_final_internal,
                "T_K": state.t_source_if_final_internal,
            },
        },
        "initialized_state": {
            "rho_source_kg_m3": state.rho_source,
            "rho_source_over_rho_charge": state.rho_source / state.charge.rho_charge,
            "e_initial_J_kg": state.e_initial_j_per_kg,
            "e_detonation_added_J_kg": state.e_detonation_added_j_per_kg,
            "e_source_J_kg": state.e_source,
            "p_source_Pa": state.p_source,
            "T_source_K": state.t_source,
            "speed_of_sound_source_m_s": state.speed_of_sound_source,
        },
        "conservation": {
            "initialized_mass_kg": state.source_mass_kg,
            "mass_error_rel": state.mass_error_rel,
            "E_initial_sensible_J": state.e_initial_sensible_j,
            "E_detonation_added_J": state.e_detonation_added_j,
            "E_final_internal_J": state.e_final_internal_j,
            "DeltaE_J": state.delta_e_j,
            "intended_W_E_charge_J": state.charge.source_energy_j,
            "DeltaE_error_rel": state.detonation_energy_error_rel,
            "note": (
                "DeltaE = E_final - E_initial is the conserved detonation energy. "
                "E_final_internal is not equal to W*E_charge."
            ),
            "basis": "full-sphere-equivalent radial shell; mesh cell volumes deliberately unused",
        },
        "displaced_ambient_air": {
            "note": (
                "Energy of the ambient air that occupied V_source before the "
                "burst gas replaced it. Distinct from E_initial_sensible of "
                "the charge mass."
            ),
            "E_ambient_in_source_shell_J": state.ambient_energy_in_shell_j,
        },
    }
    if measured_source_cells is not None or measured_wedge_source_volume_m3 is not None:
        payload["mesh_verification"] = {
            "measured_source_cells": measured_source_cells,
            "predicted_source_cells": state.shell.n_cells,
            "cell_count_matches": (
                None if measured_source_cells is None
                else int(measured_source_cells) == state.shell.n_cells
            ),
            "measured_wedge_source_volume_m3": measured_wedge_source_volume_m3,
        }
    return payload
