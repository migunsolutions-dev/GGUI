"""Canonical explosive material values shared by dimensional workflows."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


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


def materials_copy() -> Dict[str, Dict[str, Any]]:
    return deepcopy(MATERIALS)


def jwl_parameters(material_name: str, custom: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if material_name == "Custom" and isinstance(custom, dict):
        required = ("A", "B", "R1", "R2", "omega")
        if all(key in custom for key in required):
            return {
                "A": float(custom["A"]),
                "B": float(custom["B"]),
                "R1": float(custom["R1"]),
                "R2": float(custom["R2"]),
                "omega": float(custom["omega"]),
                "E0": float(custom.get("E0", custom.get("energy", 9.0e9))),
                "CvCoeffs": tuple(custom.get("CvCoeffs", (413.15, 2.1538))),
            }
    return deepcopy(JWL_PARAMETERS.get(material_name, JWL_PARAMETERS["C4"]))
