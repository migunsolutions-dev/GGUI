"""CONWEP-facing airblast interface.

CONWEP (Hyde, TM 5-855-1 / Conventional Weapons Effects) uses the
Kingery-Bulmash airblast compilation for peak overpressure, impulse,
arrival, and positive duration.

This module reuses Swisdak 1994 (ADA526744) Kingery-Bulmash *scalars*
for hemispherical surface burst, which are the published simplified
polynomials of that compilation. It does **not** label the Kingery-Bulmash
engine as CONWEP.

The CONWEP pressure-time waveform is not reconstructed here: Swisdak 1994
does not publish a CONWEP-equivalent P(t) history, and no in-repo UFC
waveform definition is available. ``pressure_history`` / ``impulse_history``
therefore return None (N/A).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from validation import kingery_bulmash as kb
from validation.metrics import is_finite_number

PRESSURE_INCIDENT = "incident"
PRESSURE_REFLECTED = "reflected"

_SCALAR_MAP = {
    PRESSURE_INCIDENT: {
        "peak_pressure": kb.QUANTITY_PEAK_PRESSURE,
        "positive_impulse": kb.QUANTITY_INCIDENT_IMPULSE,
        "arrival_time": kb.QUANTITY_ARRIVAL,
        "positive_duration": kb.QUANTITY_DURATION,
    },
    PRESSURE_REFLECTED: {
        "peak_pressure": kb.QUANTITY_REFLECTED_PRESSURE,
        "positive_impulse": kb.QUANTITY_REFLECTED_IMPULSE,
        "arrival_time": kb.QUANTITY_ARRIVAL,
        "positive_duration": kb.QUANTITY_DURATION,
    },
}

WAVEFORM_UNAVAILABLE = (
    "CONWEP pressure/impulse time histories are N/A: no authoritative CONWEP "
    "waveform (TM 5-855-1 / UFC 3-340-02) is bundled. Scalars reuse Swisdak "
    "1994 Kingery-Bulmash airblast parameters for hemispherical surface burst."
)

SCALAR_PROVENANCE = (
    "CONWEP airblast scalars reuse Kingery-Bulmash parameters via Swisdak 1994 "
    "ADA526744 (hemispherical surface burst). This is not a second undocumented fit."
)


@dataclass(frozen=True)
class ConwepScalar:
    name: str
    value_si: Optional[float]
    kb_quantity: str
    citation: str
    provenance: str
    unavailable_reason: str = ""


@dataclass(frozen=True)
class ConwepResult:
    pressure_type: str
    peak_pressure: ConwepScalar
    positive_impulse: ConwepScalar
    arrival_time: ConwepScalar
    positive_duration: ConwepScalar
    pressure_history: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]
    impulse_history: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]
    waveform_reason: str = WAVEFORM_UNAVAILABLE


def _scalar(name: str, quantity: str, range_m: float, mass_kg: float) -> ConwepScalar:
    ev = kb.evaluate(
        quantity,
        range_m=range_m,
        mass_kg=mass_kg,
        burst_type=kb.BURST_HEMISPHERICAL,
    )
    return ConwepScalar(
        name=name,
        value_si=ev.value_si if ev.ok else None,
        kb_quantity=quantity,
        citation=ev.citation,
        provenance=SCALAR_PROVENANCE,
        unavailable_reason=ev.unavailable_reason,
    )


def evaluate(
    *,
    range_m: float,
    mass_kg: float,
    pressure_type: str = PRESSURE_INCIDENT,
    explosive_type: str = "",
) -> ConwepResult:
    """Return CONWEP-facing scalars. ``explosive_type`` is recorded only; W is mass_kg."""
    kind = str(pressure_type or PRESSURE_INCIDENT).strip().lower()
    if kind not in _SCALAR_MAP:
        kind = PRESSURE_INCIDENT
    mapping = _SCALAR_MAP[kind]
    _ = explosive_type  # displayed by the UI; no undocumented TNT conversion
    if not is_finite_number(range_m) or not is_finite_number(mass_kg):
        def missing(name: str, quantity: str) -> ConwepScalar:
            return ConwepScalar(
                name=name,
                value_si=None,
                kb_quantity=quantity,
                citation=kb.CITATION,
                provenance=SCALAR_PROVENANCE,
                unavailable_reason="Standoff and explosive weight must be positive and finite.",
            )
        return ConwepResult(
            pressure_type=kind,
            peak_pressure=missing("peak_pressure", mapping["peak_pressure"]),
            positive_impulse=missing("positive_impulse", mapping["positive_impulse"]),
            arrival_time=missing("arrival_time", mapping["arrival_time"]),
            positive_duration=missing("positive_duration", mapping["positive_duration"]),
            pressure_history=None,
            impulse_history=None,
        )
    return ConwepResult(
        pressure_type=kind,
        peak_pressure=_scalar("peak_pressure", mapping["peak_pressure"], range_m, mass_kg),
        positive_impulse=_scalar("positive_impulse", mapping["positive_impulse"], range_m, mass_kg),
        arrival_time=_scalar("arrival_time", mapping["arrival_time"], range_m, mass_kg),
        positive_duration=_scalar("positive_duration", mapping["positive_duration"], range_m, mass_kg),
        pressure_history=None,
        impulse_history=None,
    )


def pressure_history(*_args, **_kwargs) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
    return None


def impulse_history(*_args, **_kwargs) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
    return None
