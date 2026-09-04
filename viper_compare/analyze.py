"""Build comparison rows. Physical R only; no series offset."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from validation.kb_propagation import CLASS_OUTSIDE, classify_vs_remap, kb_propagation_eligible
from validation.ufc_airblast import (
    BURST_SPHERICAL,
    QUANTITY_INCIDENT_IMPULSE,
    QUANTITY_PEAK_PRESSURE,
    evaluate,
    scaled_distance,
)
from viper_compare.extract import (
    arrival_time,
    derived_positive_impulse,
    peak_overpressure,
    pct_error,
    waveform_l1,
)


def gauge_row(
    *,
    solver: str,
    configuration: str,
    dimension: str,
    remapped: bool,
    gauge_label: str,
    r_m: float,
    mass_kg: float,
    times: np.ndarray,
    pressure: np.ndarray,
    native_impulse: Optional[np.ndarray],
    p_atm: float,
    receive_r_max: Optional[float],
    dx_2d: float,
    source_case: str,
    source_file: str,
) -> Dict[str, Any]:
    z = scaled_distance(r_m, mass_kg)
    peak_p, peak_t = peak_overpressure(times, pressure, p_atm)
    native_i = None
    if native_impulse is not None and native_impulse.size:
        native_i = float(np.nanmax(native_impulse))
    derived_i = derived_positive_impulse(times, pressure, p_atm)
    arr = arrival_time(times, pressure, p_atm)
    inside = None
    if remapped and dimension == "2d":
        inside = classify_vs_remap(r_m, receive_r_max, dx_2d)
    kb_p = evaluate(
        QUANTITY_PEAK_PRESSURE, range_m=r_m, mass_kg=mass_kg, burst_type=BURST_SPHERICAL
    )
    kb_i = evaluate(
        QUANTITY_INCIDENT_IMPULSE, range_m=r_m, mass_kg=mass_kg, burst_type=BURST_SPHERICAL
    )
    return {
        "solver": solver,
        "configuration": configuration,
        "dimension": dimension,
        "remapped": remapped,
        "gauge_label": gauge_label,
        "R_m": r_m,
        "Z": z,
        "inside_remap": inside,
        "independent_2d": (
            True
            if not remapped or dimension != "2d"
            else kb_propagation_eligible(r_m, receive_r_max, dx_2d)
        ),
        "peak_pressure_pa": peak_p,
        "native_impulse_pa_s": native_i,
        "derived_impulse_pa_s": derived_i,
        "peak_time_s": peak_t,
        "arrival_time_s_if_available": arr,
        "kb_peak_pressure_pa": kb_p.value_si if kb_p.ok else None,
        "kb_impulse_pa_s": kb_i.value_si if kb_i.ok else None,
        "source_case": source_case,
        "source_file": source_file,
    }


def pair_error(ggui_row: Dict[str, Any], viper_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "R_m": ggui_row["R_m"],
        "gauge_label": ggui_row["gauge_label"],
        "peak_pressure_error_pct": pct_error(
            ggui_row["peak_pressure_pa"], viper_row["peak_pressure_pa"]
        ),
        "derived_impulse_diff_pa_s": (
            None
            if ggui_row["derived_impulse_pa_s"] is None
            or viper_row["derived_impulse_pa_s"] is None
            else ggui_row["derived_impulse_pa_s"] - viper_row["derived_impulse_pa_s"]
        ),
        "peak_time_diff_s": (
            None
            if ggui_row["peak_time_s"] is None or viper_row["peak_time_s"] is None
            else ggui_row["peak_time_s"] - viper_row["peak_time_s"]
        ),
    }


def remap_eligible_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("remapped") and row.get("dimension") == "2d":
            if row.get("inside_remap") != CLASS_OUTSIDE:
                continue
        out.append(row)
    return list(out)


def waveform_window(t_a: np.ndarray, t_b: np.ndarray, peak_t: float) -> tuple:
    """Window from min(t) to peak + 2 ms, clipped to overlap. Avoids long zero tails."""
    t0 = max(float(t_a.min()), float(t_b.min()))
    t1 = min(float(t_a.max()), float(t_b.max()), float(peak_t) + 0.002)
    return t0, t1
