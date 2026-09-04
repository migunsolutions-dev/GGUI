"""Canonical explosive material values shared by dimensional workflows."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from domain_errors import IncompleteMaterialError, UnknownMaterialError

MATERIALS: Dict[str, Dict[str, Any]] = {
    "TNT": {"rho": 1630, "energy": 4.29e6},
    "C4": {"rho": 1601, "energy": 4.52e6},
    "PETN": {"rho": 1770, "energy": 6.11e6},
    "ANFO": {"rho": 840, "energy": 3.79e6},
    "Custom": {
        "rho": 1600,
        "energy": 4.50e6,
        "A": 300.0e9,
        "B": 3.0e9,
        "R1": 4.0,
        "R2": 1.0,
        "omega": 0.30,
    },
}


JWL_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "TNT": {
        "A": 373.77e9, "B": 3.7471e9, "R1": 4.15, "R2": 0.90,
        "omega": 0.35, "E0": 4.29e9, "CvCoeffs": (413.15, 2.1538),
    },
    "C4": {
        "A": 609.77e9, "B": 12.95e9, "R1": 4.50, "R2": 1.40,
        "omega": 0.25, "E0": 9.0e9, "CvCoeffs": (413.15, 2.1538),
    },
    "PETN": {
        "A": 617.0e9, "B": 16.9e9, "R1": 4.40, "R2": 1.20,
        "omega": 0.25, "E0": 6.11e9, "CvCoeffs": (413.15, 2.1538),
    },
    "ANFO": {
        "A": 49.46e9, "B": 1.89e9, "R1": 3.90, "R2": 1.10,
        "omega": 0.33, "E0": 3.79e9, "CvCoeffs": (413.15, 2.1538),
    },
}

# Shared thermodynamic template coefficients (not a material identity fallback).
DEFAULT_CV_COEFFS = (413.15, 2.1538)

# JWL-specific keys validated by ``jwl_parameters`` for Custom materials.
# Density (rho) and GUI energy are validated by ``material_validation`` /
# ``validate_required_values`` before any generator calls ``jwl_parameters``.
REQUIRED_CUSTOM_JWL_KEYS = ("A", "B", "R1", "R2", "omega")


def materials_copy() -> Dict[str, Dict[str, Any]]:
    return deepcopy(MATERIALS)


def jwl_parameters(material_name: str, custom: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return JWL EOS parameters for a known catalog material or complete Custom.

    Responsibility split (Checkpoint 3):
    * ``jwl_parameters`` validates **JWL-specific** Custom keys
      (``A``, ``B``, ``R1``, ``R2``, ``omega``, and ``E0``/``energy``).
    * Density (``rho`` / ``rho_charge``) is **not** validated here — it is
      enforced by ``material_validation.validate_material_definition`` /
      ``validate_required_values`` before generation. Generators also write
      ``inputs.rho_charge`` separately into phaseProperties.

    Never substitutes another catalog material when the name is unknown or
    Custom JWL parameters are incomplete.
    """
    name = str(material_name or "").strip()
    if name == "Custom":
        if not isinstance(custom, dict):
            raise IncompleteMaterialError(
                "Custom material requires a complete parameter dictionary."
            )
        missing = [key for key in REQUIRED_CUSTOM_JWL_KEYS if key not in custom]
        if "E0" not in custom and "energy" not in custom:
            missing.append("E0 (or energy)")
        if missing:
            raise IncompleteMaterialError(
                "Custom material is incomplete; missing required JWL parameter(s): "
                + ", ".join(missing)
                + "."
            )
        try:
            values = {
                "A": float(custom["A"]),
                "B": float(custom["B"]),
                "R1": float(custom["R1"]),
                "R2": float(custom["R2"]),
                "omega": float(custom["omega"]),
                "E0": float(custom["E0"] if "E0" in custom else custom["energy"]),
                "CvCoeffs": tuple(custom.get("CvCoeffs", DEFAULT_CV_COEFFS)),
            }
        except (TypeError, ValueError) as exc:
            raise IncompleteMaterialError(
                f"Custom material parameters are malformed: {exc}"
            ) from exc
        for key, number in values.items():
            if key == "CvCoeffs":
                continue
            if not isinstance(number, float) or number != number or number in (
                float("inf"),
                float("-inf"),
            ):
                raise IncompleteMaterialError(
                    f"Custom material parameter {key!r} is non-finite."
                )
            if number <= 0:
                raise IncompleteMaterialError(
                    f"Custom material parameter {key!r} must be > 0."
                )
        return values

    if name not in JWL_PARAMETERS:
        raise UnknownMaterialError(
            f"Unknown material {name!r}. Refusing to substitute catalog parameters "
            "from another material."
        )
    return deepcopy(JWL_PARAMETERS[name])
