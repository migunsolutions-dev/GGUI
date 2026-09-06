"""Energy-budget audit for the existing blastFoam JWL detonation path.

This module does not change JWL generation. It only states, as far as the
installed blastFoam sources and the current GGUI dictionaries permit, what
``energy_j_per_kg`` / initiation ``E0`` mean and how much energy that path
actually represents.

blastFoam documented semantics (activationModel.C):

* If the initiation dictionary contains ``E0``, it is read as
  ``dimensionedScalar("E0", dimPressure, dict)`` and converted to specific
  energy by ``e0 = E0 / rho0``.
* If it contains ``e0`` instead, that value is already ``dimEnergy/dimMass``.
* The energy equation source is ``ESource = d(lambda)/dt * e0``.
  When lambda goes 0 -> 1 over the charge, the chemical energy *added*
  is ``e0`` [J/kg], not a replacement of the cell internal energy.
* ``noneActivation::initESource`` is ``e0 * (1 - lambda)``; the default
  pressureBased path uses the rate form above.

The reactant cells are initialized separately by
``fluidBlastThermo::initializeFields`` -> ``initializeEnergy(p, rho, T)``.
For BirchMurnaghan3, ``p`` does not depend on ``e`` (``dpde`` is
``NotImplemented``), so the initial specific energy is the eConst + BM3
cold-curve evaluation at ``(rho_charge, T_atm)``. That initial energy is
*not* ``energy_j_per_kg``.

Production V2 writes the same initiation energy in every dimension:

    E0 = rho0 * E_charge     [Pa]
    e0 = E_charge            [J/kg]

``JWL_PARAMETERS["E0"]`` is legacy provenance only and is not an activation
source. The pre-V2 writers (1D GUI J/kg into Pa; 2D/3D catalog E0) remain
reconstructable via ``schema=JWL_ENERGY_SCHEMA_LEGACY``.

A unit-correct chemical add is still not energy-equivalent to an IG burst:
JWL also has a BM3 reactant reference energy and a different EOS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from jwl_activation_energy import (
    JWL_ENERGY_SCHEMA_LEGACY,
    JWL_ENERGY_SCHEMA_V2,
    PRODUCTION_JWL_ENERGY_SCHEMA,
    legacy_written_E0_pa,
    v2_activation,
)
from material_catalog import JWL_PARAMETERS, MATERIALS


# Values written into the 1D/2D/3D detonating reactant block today.
BM3_GAMMA = 0.25
BM3_PREF = 101298.0
BM3_K0 = 8.04e9
BM3_K0_PRIME = 7.97
REACTANT_CV = 1400.0
REACTANT_HF = 0.0


def birch_murnaghan3_cold_energy(rho: float, rho0: float) -> float:
    """BirchMurnaghan3::E at the given density, matching blastFoam's formula.

    ``E`` is the EOS internal-energy *correction* added to eConst
    ``Cv*T``. At ``rho == rho0`` this is still nonzero because of the
    ``K0Prime`` terms and ``-pRef/rho``.
    """
    density = float(rho)
    ref = float(rho0)
    if density <= 0.0 or ref <= 0.0:
        raise ValueError("rho and rho0 must be > 0")
    x = density / ref
    cold = (
        9.0 / 16.0 * BM3_K0
        * (
            (x ** 5) ** (1.0 / 3.0)
            * ((x ** 2) ** (1.0 / 3.0) * (14.0 - 3.0 * BM3_K0_PRIME) + 3.0 * BM3_K0_PRIME - 16.0)
            + (x ** 3) * (BM3_K0_PRIME - 4.0)
        )
        - BM3_PREF
    )
    return cold / density


def reactant_initial_specific_energy(rho_charge: float, t_atm: float, rho0: Optional[float] = None) -> float:
    """Specific internal energy of a 1D/2D JWL reactant cell at t = 0.

    eConst defaults make ``Es = Cv*T + EOS::E(rho)``. blastFoam then uses that
    value as ``e`` because BM3 ``dpde`` is zero, so ``initializeEnergy`` does
    not iterate.
    """
    rho = float(rho_charge)
    t = float(t_atm)
    ref = float(rho0) if rho0 is not None else rho
    return REACTANT_CV * t + birch_murnaghan3_cold_energy(rho, ref)


def blastfoam_e0_from_initiation(e0_dict_value: float, rho0: float) -> float:
    """Specific detonation energy blastFoam stores as ``activationModel::e0_``.

    ``E0`` in the initiation dictionary has dimensions of pressure
    (J/m^3). The solver divides by product/reactant ``rho0``.
    """
    return float(e0_dict_value) / float(rho0)


@dataclass(frozen=True)
class JwlEnergyBudget:
    """Quantities needed to compare a JWL case against an IG burst."""

    dimension: str
    material_name: str
    mass_kg: float
    rho_charge: float
    energy_j_per_kg_gui: float
    t_atm: float
    p_atm: float

    jwl_energy_schema: str
    written_initiation_E0: float
    written_E0_meaning: str
    blastfoam_e0_j_per_kg: float
    catalog_jwl_E0: Optional[float]
    catalog_energy_j_per_kg: Optional[float]

    reactant_e_init_j_per_kg: float
    chemical_energy_added_j_per_kg: float
    chemical_energy_added_j: float
    intended_gui_chemical_energy_j: float
    represented_chemical_over_intended: float

    total_specific_energy_after_burn_j_per_kg: float
    total_energy_after_burn_j: float

    volumetric_E0_if_gui_energy_were_specific: float
    e0_if_gui_energy_were_written_as_e0: float


def audit_jwl_energy(
    *,
    dimension: str,
    mass_kg: float,
    rho_charge: float,
    energy_j_per_kg: float,
    p_atm: float = 101325.0,
    t_atm: float = 288.0,
    material_name: str = "TNT",
    written_initiation_E0: Optional[float] = None,
    schema: Optional[str] = None,
) -> JwlEnergyBudget:
    """Build the JWL energy budget for a GGUI case definition.

    Default ``schema`` is production V2: ``E0 = rho0 * E_charge`` in every
    dimension. Pass ``JWL_ENERGY_SCHEMA_LEGACY`` to reconstruct the pre-V2
    writers (1D GUI J/kg into Pa; 2D/3D catalog E0).
    """
    dim = str(dimension).strip().upper()
    if dim not in {"1D", "2D", "3D"}:
        raise ValueError(f"dimension must be 1D, 2D or 3D, got {dimension!r}")

    catalog = JWL_PARAMETERS.get(material_name)
    materials = MATERIALS.get(material_name)
    catalog_e0 = None if catalog is None else float(catalog["E0"])
    catalog_energy = None if materials is None else float(materials["energy"])

    resolved_schema = str(schema or PRODUCTION_JWL_ENERGY_SCHEMA).strip()
    if written_initiation_E0 is None:
        if resolved_schema == JWL_ENERGY_SCHEMA_LEGACY:
            written = legacy_written_E0_pa(
                dimension=dim,
                energy_j_per_kg=energy_j_per_kg,
                material_name=material_name,
            )
            meaning = (
                "LEGACY writer: 1D stuffed E_charge [J/kg] into E0 [Pa]; "
                "2D/3D copied JWL_PARAMETERS E0"
            )
        else:
            written = v2_activation(
                energy_j_per_kg, rho_charge, material_name=material_name, dimension=dim
            ).E0_pa
            meaning = (
                "V2: E0 = rho0 * E_charge in every dimension; "
                "blastFoam reads E0 as Pa and divides by rho0"
            )
    else:
        written = float(written_initiation_E0)
        meaning = "Caller-supplied initiation E0, interpreted as blastFoam Pa"

    rho = float(rho_charge)
    e0 = blastfoam_e0_from_initiation(written, rho)
    e_init = reactant_initial_specific_energy(rho, t_atm, rho0=rho)
    mass = float(mass_kg)
    intended = mass * float(energy_j_per_kg)
    added = mass * e0
    return JwlEnergyBudget(
        dimension=dim,
        material_name=str(material_name),
        mass_kg=mass,
        rho_charge=rho,
        energy_j_per_kg_gui=float(energy_j_per_kg),
        t_atm=float(t_atm),
        p_atm=float(p_atm),
        jwl_energy_schema=resolved_schema,
        written_initiation_E0=written,
        written_E0_meaning=meaning,
        blastfoam_e0_j_per_kg=e0,
        catalog_jwl_E0=catalog_e0,
        catalog_energy_j_per_kg=catalog_energy,
        reactant_e_init_j_per_kg=e_init,
        chemical_energy_added_j_per_kg=e0,
        chemical_energy_added_j=added,
        intended_gui_chemical_energy_j=intended,
        represented_chemical_over_intended=added / intended if intended else math.nan,
        total_specific_energy_after_burn_j_per_kg=e_init + e0,
        total_energy_after_burn_j=mass * (e_init + e0),
        volumetric_E0_if_gui_energy_were_specific=float(energy_j_per_kg) * rho,
        e0_if_gui_energy_were_written_as_e0=float(energy_j_per_kg),
    )


def budget_dict(budget: JwlEnergyBudget) -> Dict[str, Any]:
    v2_equivalent = abs(
        budget.represented_chemical_over_intended - 1.0
    ) < 1e-9
    return {
        "source_model": "JWL_DETONATION",
        "jwl_energy_schema": budget.jwl_energy_schema,
        "dimension": budget.dimension,
        "material_name": budget.material_name,
        "user_inputs": {
            "W_kg": budget.mass_kg,
            "rho_charge_kg_m3": budget.rho_charge,
            "E_charge_J_kg_gui": budget.energy_j_per_kg_gui,
            "p_atm_Pa": budget.p_atm,
            "T_atm_K": budget.t_atm,
        },
        "what_blastfoam_reads": {
            "initiation_key": "E0",
            "initiation_value_as_written": budget.written_initiation_E0,
            "blastfoam_dimension": "Pa (J/m^3)",
            "rho0_used_for_conversion": budget.rho_charge,
            "e0_J_kg": budget.blastfoam_e0_j_per_kg,
            "meaning": budget.written_E0_meaning,
        },
        "catalog": {
            "JWL_E0": budget.catalog_jwl_E0,
            "MATERIALS_energy_J_kg": budget.catalog_energy_j_per_kg,
        },
        "reactant_initial_state": {
            "eos": "BirchMurnaghan3 + eConst Cv=1400",
            "e_init_J_kg": budget.reactant_e_init_j_per_kg,
            "note": (
                "This is the cell internal energy at t=0, before any lambda "
                "progress. It is not E_charge."
            ),
        },
        "chemical_release": {
            "e0_added_J_kg": budget.chemical_energy_added_j_per_kg,
            "E_added_J": budget.chemical_energy_added_j,
            "intended_W_times_E_charge_J": budget.intended_gui_chemical_energy_j,
            "represented_chemical_over_intended": budget.represented_chemical_over_intended,
            "source_term": "ESource = d(lambda)/dt * e0",
        },
        "after_complete_burn": {
            "e_total_J_kg": budget.total_specific_energy_after_burn_j_per_kg,
            "E_total_J": budget.total_energy_after_burn_j,
            "note": "e_init + e0; products then follow the JWL EOS, not idealGas",
        },
        "counterfactuals_not_applied": {
            "E0_Pa_if_GUI_energy_were_specific": (
                budget.volumetric_E0_if_gui_energy_were_specific
            ),
            "e0_if_initiation_used_lowercase_e0": (
                budget.e0_if_gui_energy_were_written_as_e0
            ),
        },
        "equivalence_to_ig": {
            "same_W_rho_E_charge_chemical_add_matches": v2_equivalent,
            "same_W_rho_E_charge_is_energy_equivalent": False,
            "hard_gate": (
                "Reference comparisons are validation only. A unit-correct "
                "JWL chemical add is not a reason to retune JWL or IG to "
                "UFC, KB, VIPER, IG, legacy JWL, or any waveform."
            ),
            "reason": (
                "V2 writes E0 = rho0 * E_charge so blastFoam e0 = E_charge "
                "in every dimension. IG still adds W*E_charge to Cv*T_atm "
                "on an ideal-gas EOS; JWL also has BM3 reactant reference "
                "energy. Same GUI W/rho/E_charge is not EOS-equivalent."
            ),
        },
    }


def print_budget(budget: JwlEnergyBudget) -> str:
    lines = [
        f"BF-JWL energy budget ({budget.dimension}, {budget.material_name})",
        f"  GUI W={budget.mass_kg} kg  rho={budget.rho_charge}  E_charge={budget.energy_j_per_kg_gui:g} J/kg",
        f"  Written initiation E0 = {budget.written_initiation_E0:g}  ({budget.written_E0_meaning})",
        f"  blastFoam e0 = E0/rho0 = {budget.blastfoam_e0_j_per_kg:g} J/kg",
        f"  Reactant e_init (BM3+eConst) = {budget.reactant_e_init_j_per_kg:g} J/kg",
        f"  Chemical energy added = {budget.chemical_energy_added_j:g} J"
        f"  ({budget.represented_chemical_over_intended:.4g} of W*E_charge)",
        f"  After complete burn e = {budget.total_specific_energy_after_burn_j_per_kg:g} J/kg"
        f"  ({budget.total_energy_after_burn_j:g} J)",
        f"  Intended W*E_charge = {budget.intended_gui_chemical_energy_j:g} J",
    ]
    return "\n".join(lines)
