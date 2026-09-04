"""Kingery-Bulmash airblast from Swisdak 1994 (ADA526744).

Primary source
--------------
Michael M. Swisdak Jr., "Simplified Kingery Airblast Calculations",
Naval Surface Warfare Center Indian Head Division, 1994, ADA526744.

The paper publishes simplified piecewise polynomials that reproduce the
Kingery-Bulmash 1984 hemispherical *surface-burst* compilation to within 1%.

Form (Swisdak 1994)::

    Y = exp(A + B ln Z + C (ln Z)^2 + ... + G (ln Z)^6)

where Z = R / W**(1/3) is scaled distance in m / kg**(1/3) for the metric
coefficients used here. Pressure Y is in kPa. Impulse Y is in kPa·ms / kg**(1/3)
before multiplying by W**(1/3). Arrival and positive duration are in
ms / kg**(1/3) before multiplying by W**(1/3).

This module does **not** extrapolate outside a published Z interval.
This module does **not** treat the hemispherical fit as a free-air spherical
fit. Swisdak 1994 covers the hemispherical surface-burst family only.

Mass convention: W is the user-supplied charge mass. No silent TNT
equivalence and no silent hemisphere-to-sphere conversion.

Coefficients below are the metric columns of Swisdak 1994 as transcribed
in open secondary sources that quote ADA526744 Table values (incident
peak overpressure matches Computation 2017, 5, 41, Table 2 citing [17]
Swisdak 1994).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from validation.metrics import is_finite_number
from validation.units import kpa_ms_to_pa_s, kpa_to_pa, ms_to_s

CITATION = (
    "Michael M. Swisdak Jr., Simplified Kingery Airblast Calculations, "
    "Naval Surface Warfare Center Indian Head Division, 1994, ADA526744."
)
DOCUMENT_ID = "ADA526744"
EQUATION = "Y = exp(A + B*ln(Z) + C*(ln(Z))^2 + D*(ln(Z))^3 + E*(ln(Z))^4 + F*(ln(Z))^5 + G*(ln(Z))^6)"
Z_UNITS = "m/kg**(1/3)"
MASS_CONVENTION = (
    "W is the entered charge mass (kg). No TNT-equivalence factor is applied. "
    "Swisdak 1994 / Kingery hemispherical surface-burst compilation assumes TNT; "
    "the entered mass is used as W without conversion."
)

BURST_HEMISPHERICAL = "hemispherical_surface"
BURST_SPHERICAL = "free_air_spherical"

QUANTITY_PEAK_PRESSURE = "peak_incident_overpressure"
QUANTITY_REFLECTED_PRESSURE = "peak_reflected_overpressure"
QUANTITY_INCIDENT_IMPULSE = "positive_incident_impulse"
QUANTITY_REFLECTED_IMPULSE = "positive_reflected_impulse"
QUANTITY_ARRIVAL = "arrival_time"
QUANTITY_DURATION = "positive_phase_duration"
QUANTITY_SHOCK_SPEED = "shock_front_velocity"

# Piecewise metric coefficients: (z_min, z_max_inclusive, (A..G), scale_W_third)
# Intervals are closed at the upper bound of each piece except the first lower bound.
# At a shared breakpoint the earlier (near-field) piece is used (Swisdak listing order).
_HEMI: Dict[str, Tuple[Tuple[float, float, Tuple[float, ...], bool], ...]] = {
    QUANTITY_ARRIVAL: (
        (0.06, 1.50, (-0.7604, 1.8058, 0.1257, -0.0437, -0.0310, -0.00669, 0.0), True),
        (1.50, 40.0, (-0.7137, 1.5732, 0.5561, -0.4213, 0.1054, -0.00929, 0.0), True),
    ),
    QUANTITY_PEAK_PRESSURE: (
        (0.2, 2.9, (7.2106, -2.1069, -0.3229, 0.1117, 0.0685, 0.0, 0.0), False),
        (2.9, 23.8, (7.5938, -3.0523, 0.40977, 0.0261, -0.01267, 0.0, 0.0), False),
        (23.8, 198.5, (6.0536, -1.4066, 0.0, 0.0, 0.0, 0.0, 0.0), False),
    ),
    QUANTITY_REFLECTED_PRESSURE: (
        (0.06, 2.00, (9.006, -2.6893, -0.6295, 0.1011, 0.29255, 0.13505, 0.019736), False),
        (2.00, 40.0, (8.8396, -1.733, -2.64, 2.293, -0.8232, 0.14247, -0.0099), False),
    ),
    QUANTITY_DURATION: (
        (0.2, 1.02, (0.5426, 3.2299, -1.5931, -5.9667, -4.0815, -0.9149, 0.0), True),
        (1.02, 2.8, (0.5440, 2.7082, -9.7354, 14.3425, -9.7791, 2.8535, 0.0), True),
        (2.8, 40.0, (-2.4608, 7.1639, -5.6215, 2.2711, -0.44994, 0.03486, 0.0), True),
    ),
    QUANTITY_INCIDENT_IMPULSE: (
        (0.2, 0.96, (5.522, 1.117, 0.6, -0.292, -0.087, 0.0, 0.0), True),
        (0.96, 2.38, (5.465, -0.308, -1.464, 1.362, -0.432, 0.0, 0.0), True),
        (2.38, 33.7, (5.2749, -0.4677, -0.2499, 0.0588, -0.00554, 0.0, 0.0), True),
        (33.7, 158.7, (5.9825, -1.062, 0.0, 0.0, 0.0, 0.0, 0.0), True),
    ),
    QUANTITY_REFLECTED_IMPULSE: (
        (0.06, 40.0, (6.7853, -1.3466, 0.101, -0.01123, 0.0, 0.0, 0.0), True),
    ),
    QUANTITY_SHOCK_SPEED: (
        (0.06, 1.50, (0.1794, -0.956, -0.0866, 0.109, 0.0699, 0.01218, 0.0), False),
        (1.50, 40.0, (0.2597, -1.326, 0.3767, 0.0396, -0.0351, 0.00432, 0.0), False),
    ),
}

SPHERICAL_UNAVAILABLE = (
    "Swisdak 1994 ADA526744 publishes simplified polynomials for the "
    "Kingery hemispherical surface-burst compilation only. Free-air spherical "
    "Kingery-Bulmash 1984 (ARBRL-TR-02555) polynomials are not bundled."
)


@dataclass(frozen=True)
class ReferenceEval:
    quantity: str
    burst_type: str
    value_si: Optional[float]
    z: Optional[float]
    range_m: Optional[float]
    mass_kg: Optional[float]
    z_min: Optional[float]
    z_max: Optional[float]
    equation_id: str
    citation: str
    mass_convention: str
    unavailable_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value_si is not None and is_finite_number(self.value_si)


def scaled_distance(range_m: float, mass_kg: float) -> Optional[float]:
    if not is_finite_number(range_m) or not is_finite_number(mass_kg):
        return None
    if float(range_m) <= 0.0 or float(mass_kg) <= 0.0:
        return None
    return float(range_m) / (float(mass_kg) ** (1.0 / 3.0))


def _eval_poly(z: float, coeffs: Sequence[float]) -> float:
    ln_z = math.log(z)
    total = 0.0
    term = 1.0
    for coef in coeffs:
        total += float(coef) * term
        term *= ln_z
    return math.exp(total)


def _pick_piece(
    pieces: Sequence[Tuple[float, float, Tuple[float, ...], bool]], z: float
) -> Optional[Tuple[float, float, Tuple[float, ...], bool]]:
    for index, piece in enumerate(pieces):
        z_min, z_max, _coeffs, _scale = piece
        last = index == len(pieces) - 1
        if index == 0:
            if z_min <= z <= z_max:
                return piece
            continue
        if last:
            if z_min < z <= z_max:
                return piece
        elif z_min < z <= z_max:
            return piece
    return None


def _to_si(quantity: str, raw: float, mass_kg: float, scale_w: bool) -> float:
    scaled = raw * (float(mass_kg) ** (1.0 / 3.0)) if scale_w else raw
    if quantity in (QUANTITY_PEAK_PRESSURE, QUANTITY_REFLECTED_PRESSURE):
        return kpa_to_pa(scaled)
    if quantity in (QUANTITY_INCIDENT_IMPULSE, QUANTITY_REFLECTED_IMPULSE):
        return kpa_ms_to_pa_s(scaled)
    if quantity in (QUANTITY_ARRIVAL, QUANTITY_DURATION):
        return ms_to_s(scaled)
    if quantity == QUANTITY_SHOCK_SPEED:
        return float(scaled) * 1000.0  # Swisdak metric shock speed poly is in km/s units * 1000 → m/s
    return float(scaled)


def evaluate(
    quantity: str,
    *,
    range_m: float,
    mass_kg: float,
    burst_type: str = BURST_HEMISPHERICAL,
) -> ReferenceEval:
    z = scaled_distance(range_m, mass_kg)
    base = dict(
        quantity=quantity,
        burst_type=burst_type,
        value_si=None,
        z=z,
        range_m=float(range_m) if is_finite_number(range_m) else None,
        mass_kg=float(mass_kg) if is_finite_number(mass_kg) else None,
        z_min=None,
        z_max=None,
        equation_id=EQUATION,
        citation=CITATION,
        mass_convention=MASS_CONVENTION,
    )
    if burst_type == BURST_SPHERICAL:
        return ReferenceEval(unavailable_reason=SPHERICAL_UNAVAILABLE, **base)
    if burst_type != BURST_HEMISPHERICAL:
        return ReferenceEval(
            unavailable_reason=f"Unknown burst type {burst_type!r}.", **base
        )
    if z is None:
        return ReferenceEval(
            unavailable_reason="Range and charge mass must be positive and finite.",
            **base,
        )
    pieces = _HEMI.get(quantity)
    if not pieces:
        return ReferenceEval(
            unavailable_reason=f"Quantity {quantity!r} is not in Swisdak 1994.",
            **base,
        )
    piece = _pick_piece(pieces, z)
    if piece is None:
        z_lo = pieces[0][0]
        z_hi = pieces[-1][1]
        return ReferenceEval(
            unavailable_reason=(
                f"Z={z:.4g} {Z_UNITS} is outside the published Swisdak 1994 "
                f"interval [{z_lo:g}, {z_hi:g}] for {quantity}."
            ),
            z_min=z_lo,
            z_max=z_hi,
            **{k: v for k, v in base.items() if k not in ("z_min", "z_max")},
        )
    z_min, z_max, coeffs, scale_w = piece
    raw = _eval_poly(z, coeffs)
    value = _to_si(quantity, raw, mass_kg, scale_w)
    if not is_finite_number(value):
        return ReferenceEval(
            unavailable_reason="Polynomial evaluated to a non-finite value.",
            z_min=z_min,
            z_max=z_max,
            **{k: v for k, v in base.items() if k not in ("z_min", "z_max")},
        )
    return ReferenceEval(
        quantity=quantity,
        burst_type=burst_type,
        value_si=float(value),
        z=z,
        range_m=float(range_m),
        mass_kg=float(mass_kg),
        z_min=z_min,
        z_max=z_max,
        equation_id=EQUATION,
        citation=CITATION,
        mass_convention=MASS_CONVENTION,
    )


def curve(
    quantity: str,
    *,
    mass_kg: float,
    burst_type: str = BURST_HEMISPHERICAL,
    n_points: int = 80,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Reference curve vs range (m) across the published Z envelope for ``quantity``."""
    if burst_type != BURST_HEMISPHERICAL:
        return (), ()
    pieces = _HEMI.get(quantity)
    if not pieces or not is_finite_number(mass_kg) or float(mass_kg) <= 0.0:
        return (), ()
    z_lo = pieces[0][0]
    z_hi = pieces[-1][1]
    w13 = float(mass_kg) ** (1.0 / 3.0)
    ranges = []
    values = []
    if n_points < 2:
        n_points = 2
    log_lo = math.log(z_lo)
    log_hi = math.log(z_hi)
    for i in range(n_points):
        z = math.exp(log_lo + (log_hi - log_lo) * i / (n_points - 1))
        r = z * w13
        ev = evaluate(quantity, range_m=r, mass_kg=mass_kg, burst_type=burst_type)
        if ev.ok:
            ranges.append(r)
            values.append(float(ev.value_si))
    return tuple(ranges), tuple(values)


def curve_vs_z(
    quantity: str,
    *,
    mass_kg: float,
    burst_type: str = BURST_HEMISPHERICAL,
    n_points: int = 80,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    ranges, values = curve(
        quantity, mass_kg=mass_kg, burst_type=burst_type, n_points=n_points
    )
    if not ranges:
        return (), ()
    w13 = float(mass_kg) ** (1.0 / 3.0)
    return tuple(r / w13 for r in ranges), values
