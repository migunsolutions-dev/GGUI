"""Authoritative JWL activation-energy conversion for GGUI V2.

The GUI / material input ``E_charge`` is specific detonation energy [J/kg].
blastFoam reads initiation ``E0`` as Pa (J/m^3) and forms ``e0 = E0 / rho0``.

V2 therefore uses one mapping in every dimension:

    e0 = E_charge
    E0 = rho0 * E_charge

``JWL_PARAMETERS["E0"]`` is not an activation-energy source. It is retained
only as legacy provenance. This module does not change JWL EOS coefficients
and does not fit anything to UFC, KB, VIPER, IG, or waveforms.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from material_catalog import JWL_PARAMETERS

JWL_ENERGY_SCHEMA_LEGACY = "JWL_ENERGY_SCHEMA_LEGACY"
JWL_ENERGY_SCHEMA_V2 = "JWL_ENERGY_SCHEMA_V2"
PRODUCTION_JWL_ENERGY_SCHEMA = JWL_ENERGY_SCHEMA_V2
JWL_ENERGY_AUDIT_FILENAME = "ggui_jwl_energy.json"


class JwlActivationEnergyError(ValueError):
    """Raised when E_charge or rho0 cannot form a valid activation energy."""


def _require_positive(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise JwlActivationEnergyError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise JwlActivationEnergyError(f"{name} must be finite and > 0, got {number!r}")
    return number


@dataclass(frozen=True)
class JwlActivationEnergy:
    """Values written to blastFoam initiation, plus reconstruction metadata."""

    schema: str
    E_charge_j_per_kg: float
    rho0_kg_m3: float
    E0_pa: float
    e0_j_per_kg: float
    material_name: str = ""
    catalog_E0_legacy_pa: Optional[float] = None
    dimension: str = ""


def v2_activation(
    energy_j_per_kg: float,
    rho0: float,
    *,
    material_name: str = "",
    dimension: str = "",
) -> JwlActivationEnergy:
    """Production mapping: ``E0 = rho0 * E_charge``, ``e0 = E_charge``."""
    e_charge = _require_positive("E_charge", energy_j_per_kg)
    rho = _require_positive("rho0", rho0)
    name = str(material_name or "").strip()
    catalog = JWL_PARAMETERS.get(name)
    return JwlActivationEnergy(
        schema=JWL_ENERGY_SCHEMA_V2,
        E_charge_j_per_kg=e_charge,
        rho0_kg_m3=rho,
        E0_pa=rho * e_charge,
        e0_j_per_kg=e_charge,
        material_name=name,
        catalog_E0_legacy_pa=None if catalog is None else float(catalog["E0"]),
        dimension=str(dimension or "").strip().upper(),
    )


def legacy_written_E0_pa(
    *,
    dimension: str,
    energy_j_per_kg: float,
    material_name: str = "",
) -> float:
    """Reproduce the pre-V2 written E0. Comparison / identification only."""
    dim = str(dimension).strip().upper()
    e_charge = _require_positive("E_charge", energy_j_per_kg)
    name = str(material_name or "").strip()
    if dim == "1D":
        return e_charge
    catalog = JWL_PARAMETERS.get(name)
    if catalog is not None:
        return float(catalog["E0"])
    return e_charge


def blastfoam_e0_j_per_kg(E0_pa: float, rho0: float) -> float:
    """What installed blastFoam stores after reading uppercase ``E0``."""
    return _require_positive("E0", E0_pa) / _require_positive("rho0", rho0)


def audit_dict(
    state: JwlActivationEnergy,
    *,
    case_path: Optional[str] = None,
    mass_kg: Optional[float] = None,
) -> Dict[str, Any]:
    catalog_e0 = state.catalog_E0_legacy_pa
    payload: Dict[str, Any] = {
        "source_model": "JWL_DETONATION",
        "jwl_energy_schema": state.schema,
        "dimension": state.dimension or None,
        "case_path": case_path,
        "material_name": state.material_name or None,
        "user_inputs": {
            "E_charge_J_kg": state.E_charge_j_per_kg,
            "rho0_kg_m3": state.rho0_kg_m3,
            "W_kg": mass_kg,
        },
        "blastfoam_initiation": {
            "key": "E0",
            "E0_Pa": state.E0_pa,
            "blastfoam_dimension": "Pa (J/m^3)",
            "e0_J_kg": state.e0_j_per_kg,
            "identity": "e0 = E_charge; E0 = rho0 * E_charge",
        },
        "legacy_catalog_E0_Pa": catalog_e0,
        "legacy_catalog_note": (
            "JWL_PARAMETERS['E0'] is provenance only and is not used for V2 "
            "activation energy."
        ),
    }
    if mass_kg is not None:
        payload["activation_energy_J"] = float(mass_kg) * state.e0_j_per_kg
    return payload


def write_jwl_energy_audit(case_dir: str, state: JwlActivationEnergy, *, mass_kg: Optional[float] = None) -> str:
    path = os.path.join(case_dir, JWL_ENERGY_AUDIT_FILENAME)
    payload = audit_dict(state, case_path=case_dir, mass_kg=mass_kg)
    os.makedirs(case_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def read_jwl_energy_schema(case_dir: str) -> str:
    """Identify a case folder. Missing sidecar means a pre-V2 (legacy) case."""
    path = os.path.join(case_dir or "", JWL_ENERGY_AUDIT_FILENAME)
    if not os.path.isfile(path):
        return JWL_ENERGY_SCHEMA_LEGACY
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return JWL_ENERGY_SCHEMA_LEGACY
    schema = str(payload.get("jwl_energy_schema") or "").strip()
    if schema == JWL_ENERGY_SCHEMA_V2:
        return JWL_ENERGY_SCHEMA_V2
    return JWL_ENERGY_SCHEMA_LEGACY
