"""UFC Calc workbook pressure-time reconstruction.

The UFC Calc worksheet builds a modified-Friedlander history from interpolated
Figure 2-7 / 2-15 table parameters:

    P(t) = P * (1 - (t - ta)/t0) * exp(-b * (t - ta)/t0)

for ta ≤ t ≤ ta + t0. The decay b is tabulated in the workbook and is a
derived Friedlander coefficient (I = P t0 (b-1+e^{-b})/b^2), not a UFC figure
series and not CONWEP.

Do not label this engine CONWEP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from validation import ufc_airblast as ufc_ab
from validation.metrics import is_finite_number
from validation.ufc_units import friedlander_impulse_factor

CITATION = (
    "UFC Calc.xlsx worksheet 'UFC Calc': modified Friedlander using interpolated "
    "DataSpherical / DataHemiSpherical parameters (UFC 3-340-02 Figures 2-7 and 2-15). "
    "Decay b is derived from tabulated P, I, and t0; it is not a CONWEP source."
)

FAMILY_INCIDENT = "incident"
FAMILY_REFLECTED = "reflected"


@dataclass(frozen=True)
class UfcWaveform:
    family: str
    burst_type: str
    times_s: Tuple[float, ...]
    overpressure_pa: Tuple[float, ...]
    impulse_pa_s: Tuple[float, ...]
    peak_pa: float
    impulse_table_pa_s: float
    arrival_s: float
    duration_s: float
    decay_b: float
    citation: str = CITATION
    unavailable_reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.unavailable_reason and bool(self.times_s)


def _family_quantities(family: str) -> Tuple[str, str, str]:
    if family == FAMILY_REFLECTED:
        return (
            ufc_ab.QUANTITY_REFLECTED_PRESSURE,
            ufc_ab.QUANTITY_REFLECTED_IMPULSE,
            ufc_ab.QUANTITY_B_REFLECTED,
        )
    return (
        ufc_ab.QUANTITY_PEAK_PRESSURE,
        ufc_ab.QUANTITY_INCIDENT_IMPULSE,
        ufc_ab.QUANTITY_B_INCIDENT,
    )


def friedlander_overpressure(t_s: float, peak_pa: float, arrival_s: float, duration_s: float, decay_b: float) -> float:
    if duration_s <= 0.0 or t_s < arrival_s or t_s > arrival_s + duration_s:
        return 0.0
    tau = (t_s - arrival_s) / duration_s
    return float(peak_pa) * (1.0 - tau) * math.exp(-float(decay_b) * tau)


def evaluate(
    *,
    range_m: float,
    mass_kg: float,
    burst_type: str,
    family: str = FAMILY_INCIDENT,
    n_points: int = 201,
) -> UfcWaveform:
    q_p, q_i, q_b = _family_quantities(family)
    peak = ufc_ab.evaluate(q_p, range_m=range_m, mass_kg=mass_kg, burst_type=burst_type)
    impulse = ufc_ab.evaluate(q_i, range_m=range_m, mass_kg=mass_kg, burst_type=burst_type)
    arrival = ufc_ab.evaluate(
        ufc_ab.QUANTITY_ARRIVAL, range_m=range_m, mass_kg=mass_kg, burst_type=burst_type
    )
    duration = ufc_ab.evaluate(
        ufc_ab.QUANTITY_DURATION, range_m=range_m, mass_kg=mass_kg, burst_type=burst_type
    )
    decay = ufc_ab.evaluate(q_b, range_m=range_m, mass_kg=mass_kg, burst_type=burst_type)
    empty = UfcWaveform(
        family=family,
        burst_type=burst_type,
        times_s=(),
        overpressure_pa=(),
        impulse_pa_s=(),
        peak_pa=0.0,
        impulse_table_pa_s=0.0,
        arrival_s=0.0,
        duration_s=0.0,
        decay_b=0.0,
        unavailable_reason="",
    )
    missing = [
        ev.unavailable_reason
        for ev in (peak, impulse, arrival, duration, decay)
        if not ev.ok
    ]
    if missing:
        return UfcWaveform(**{**empty.__dict__, "unavailable_reason": missing[0]})
    p0 = float(peak.value_si)
    i0 = float(impulse.value_si)
    ta = float(arrival.value_si)
    t0 = float(duration.value_si)
    b = float(decay.value_si)
    if t0 <= 0.0 or p0 <= 0.0:
        return UfcWaveform(
            **{**empty.__dict__, "unavailable_reason": "UFC Calc P(t): N/A — peak or duration is not positive."}
        )
    if n_points < 2:
        n_points = 2
    times = []
    pressures = []
    impulses = []
    acc = 0.0
    prev_t = ta
    prev_p = p0
    for i in range(n_points):
        t = ta + t0 * i / (n_points - 1)
        p = friedlander_overpressure(t, p0, ta, t0, b)
        if i:
            acc += 0.5 * (prev_p + p) * (t - prev_t)
        times.append(t)
        pressures.append(p)
        impulses.append(acc)
        prev_t = t
        prev_p = p
    return UfcWaveform(
        family=family,
        burst_type=burst_type,
        times_s=tuple(times),
        overpressure_pa=tuple(pressures),
        impulse_pa_s=tuple(impulses),
        peak_pa=p0,
        impulse_table_pa_s=i0,
        arrival_s=ta,
        duration_s=t0,
        decay_b=b,
    )


def table_impulse_from_decay(peak_pa: float, duration_s: float, decay_b: float) -> float:
    return float(peak_pa) * float(duration_s) * friedlander_impulse_factor(decay_b)
