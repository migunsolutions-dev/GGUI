from __future__ import annotations

from typing import Any, Dict, Optional

from models import RecommendedParams1D


def get_profile(name: str = "Balanced") -> Dict[str, Any]:
    # Keep your current Balanced defaults here; later you can add more profiles.
    # NOTE: values match the ones you currently use in main.py.
    if not name:
        name = "Balanced"

    if name.lower() == "balanced":
        return {
            "name": "Balanced",
            "n_cells_on_charge_radius": 40,
            "domain_factor": 10.0,
            "maxCo_default": 0.4,
            "dt_cap": 5e-8,
            "dt_factor": 0.05,
            "a_ref": 6000.0,
            "maxDeltaT": 1e-5,
            "ignition_frac": 0.02,
            "ignition_dx_mult": 2.0,
            "rmin_frac": 0.002,
            "rmin_dx_mult": 0.5,
        }

    # Fallback
    return get_profile("Balanced")


def compute_recommended_1d(
    *,
    radius: float,
    cell_size: float,
    charge_radius: float,
    profile: Dict[str, Any],
    max_cfl_from_ui: Optional[float] = None,
) -> RecommendedParams1D:
    dx = max(float(cell_size), 1e-6)
    R = max(float(charge_radius), 1e-6)

    r_min = max(float(profile["rmin_dx_mult"]) * dx, float(profile["rmin_frac"]) * R, 1e-6)
    r_ign = r_min + 0.5 * dx
    ignition_point = (float(r_ign), 0.0, 0.0)

    ign_r = max(float(profile["ignition_dx_mult"]) * dx, float(profile["ignition_frac"]) * R)
    ign_r = min(ign_r, 0.25 * R)

    dt0 = float(profile["dt_factor"]) * dx / max(float(profile["a_ref"]), 1.0)
    dt0 = min(dt0, float(profile["dt_cap"]))
    dt0 = max(dt0, 1e-10)

    maxCo = float(profile["maxCo_default"]) if max_cfl_from_ui is None else max(0.05, min(1.0, float(max_cfl_from_ui)))
    maxDeltaT = float(profile["maxDeltaT"])

    return RecommendedParams1D(
        r_min=float(r_min),
        ignition_point=ignition_point,
        ignition_radius=float(ign_r),
        dt0=float(dt0),
        maxCo=float(maxCo),
        maxDeltaT=float(maxDeltaT),
    )
