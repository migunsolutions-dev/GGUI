"""UFC 3-340-02 spherical (Fig 2-7) and hemispherical (Fig 2-15) airblast tables.

Tables come from UFC Calc.xlsx sheets DataSpherical / DataHemiSpherical, which
reproduce the official DPlot series in SI. This engine is not Kingery-Bulmash
and not CONWEP. Swisdak 1994 remains a separate hemispherical implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from validation.metrics import is_finite_number
from validation.ufc_data import load_json
from validation.ufc_interp import interp1d
from validation.ufc_units import cube_root
from validation.units import kpa_ms_to_pa_s, kpa_to_pa, ms_to_s

BURST_SPHERICAL = "ufc_free_air_spherical"
BURST_HEMISPHERICAL = "ufc_hemispherical_surface"

QUANTITY_PEAK_PRESSURE = "peak_incident_overpressure"
QUANTITY_REFLECTED_PRESSURE = "peak_reflected_overpressure"
QUANTITY_INCIDENT_IMPULSE = "positive_incident_impulse"
QUANTITY_REFLECTED_IMPULSE = "positive_reflected_impulse"
QUANTITY_ARRIVAL = "arrival_time"
QUANTITY_DURATION = "positive_phase_duration"
QUANTITY_B_INCIDENT = "friedlander_decay_incident"
QUANTITY_B_REFLECTED = "friedlander_decay_reflected"

_COL = {
    QUANTITY_ARRIVAL: 1,
    QUANTITY_DURATION: 2,
    QUANTITY_PEAK_PRESSURE: 3,
    QUANTITY_INCIDENT_IMPULSE: 4,
    QUANTITY_B_INCIDENT: 5,
    QUANTITY_REFLECTED_PRESSURE: 6,
    QUANTITY_REFLECTED_IMPULSE: 7,
    QUANTITY_B_REFLECTED: 8,
}

_SCALE_W13 = {
    QUANTITY_ARRIVAL,
    QUANTITY_DURATION,
    QUANTITY_INCIDENT_IMPULSE,
    QUANTITY_REFLECTED_IMPULSE,
}

_FILES = {
    BURST_SPHERICAL: "ufc_3_340_02_fig_2_7.json",
    BURST_HEMISPHERICAL: "ufc_3_340_02_fig_2_15.json",
}


def _table(burst_type: str) -> Optional[dict]:
    name = _FILES.get(burst_type)
    if not name:
        return None
    return load_json(name)


@dataclass(frozen=True)
class UfcAirblastEval:
    quantity: str
    burst_type: str
    value_si: Optional[float]
    z: Optional[float]
    range_m: Optional[float]
    mass_kg: Optional[float]
    z_min: Optional[float]
    z_max: Optional[float]
    figure: str
    sheet: str
    citation: str
    interpolation_method: str
    data_kind: str
    unavailable_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value_si is not None and is_finite_number(self.value_si)


def z_interval(burst_type: str) -> Optional[Tuple[float, float]]:
    """Published SI scaled-distance interval for a UFC airblast table. No extrapolation."""
    table = _table(burst_type)
    if table is None:
        return None
    z_lo, z_hi = table["valid_range_z_si"]
    return float(z_lo), float(z_hi)


def figure_id(burst_type: str) -> str:
    table = _table(burst_type)
    if table is None:
        return ""
    return str(table.get("source_figure") or "")


def scaled_distance(range_m: float, mass_kg: float) -> Optional[float]:
    if not is_finite_number(range_m) or not is_finite_number(mass_kg):
        return None
    if float(range_m) <= 0.0 or float(mass_kg) <= 0.0:
        return None
    return float(range_m) / cube_root(float(mass_kg))


def interpolate_row(burst_type: str, z: float) -> Optional[Tuple[Sequence[float], dict]]:
    table = _table(burst_type)
    if table is None:
        return None
    rows = table["rows"]
    zs = [float(r[0]) for r in rows]
    cols = []
    for j in range(len(rows[0])):
        ys = [float(r[j]) for r in rows]
        val = interp1d(z, zs, ys)
        if val is None:
            return None
        cols.append(val)
    return cols, table


def _to_si(quantity: str, raw: float, mass_kg: float) -> float:
    scaled = raw * cube_root(float(mass_kg)) if quantity in _SCALE_W13 else raw
    if quantity in (QUANTITY_PEAK_PRESSURE, QUANTITY_REFLECTED_PRESSURE):
        return kpa_to_pa(scaled)
    if quantity in (QUANTITY_INCIDENT_IMPULSE, QUANTITY_REFLECTED_IMPULSE):
        return kpa_ms_to_pa_s(scaled)
    if quantity in (QUANTITY_ARRIVAL, QUANTITY_DURATION):
        return ms_to_s(scaled)
    return float(scaled)


def evaluate(
    quantity: str,
    *,
    range_m: float,
    mass_kg: float,
    burst_type: str,
) -> UfcAirblastEval:
    table = _table(burst_type)
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
        figure="",
        sheet="",
        citation="",
        interpolation_method="",
        data_kind="ufc_workbook_table_converted_from_dplot",
    )
    if table is None:
        return UfcAirblastEval(
            unavailable_reason=f"Unknown UFC burst type {burst_type!r}.",
            **base,
        )
    base["figure"] = f"Figure {table['source_figure']}"
    base["sheet"] = str(table.get("source_sheet") or "")
    base["citation"] = (
        f"{table['source_document']}, {base['figure']}, "
        f"workbook {table.get('source_workbook')} sheet {base['sheet']}."
    )
    base["interpolation_method"] = str(table.get("interpolation_method") or "")
    z_lo, z_hi = table["valid_range_z_si"]
    base["z_min"] = float(z_lo)
    base["z_max"] = float(z_hi)
    if quantity not in _COL:
        return UfcAirblastEval(
            unavailable_reason=f"Quantity {quantity!r} is not in the UFC {base['figure']} table.",
            **base,
        )
    if z is None:
        return UfcAirblastEval(
            unavailable_reason="Range and charge mass must be positive and finite.",
            **base,
        )
    got = interpolate_row(burst_type, z)
    if got is None:
        return UfcAirblastEval(
            unavailable_reason=(
                f"UFC {base['figure']}: N/A — Z={z:.4g} m/kg^(1/3) is outside "
                f"the tabulated interval [{z_lo:g}, {z_hi:g}]."
            ),
            **base,
        )
    cols, _t = got
    raw = cols[_COL[quantity]]
    value = _to_si(quantity, raw, mass_kg)
    if not is_finite_number(value):
        return UfcAirblastEval(unavailable_reason="Interpolated value is not finite.", **base)
    base["value_si"] = float(value)
    return UfcAirblastEval(**base)


def curve(
    quantity: str,
    *,
    mass_kg: float,
    burst_type: str,
    n_points: int = 80,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    table = _table(burst_type)
    if table is None or quantity not in _COL:
        return (), ()
    if not is_finite_number(mass_kg) or float(mass_kg) <= 0.0:
        return (), ()
    z_lo, z_hi = table["valid_range_z_si"]
    w13 = cube_root(float(mass_kg))
    if n_points < 2:
        n_points = 2
    import math

    ranges = []
    values = []
    log_lo = math.log(z_lo)
    log_hi = math.log(z_hi)
    for i in range(n_points):
        z = math.exp(log_lo + (log_hi - log_lo) * i / (n_points - 1))
        ev = evaluate(quantity, range_m=z * w13, mass_kg=mass_kg, burst_type=burst_type)
        if ev.ok:
            ranges.append(float(ev.range_m))
            values.append(float(ev.value_si))
    return tuple(ranges), tuple(values)


def curve_vs_z(
    quantity: str,
    *,
    mass_kg: float,
    burst_type: str,
    n_points: int = 80,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    ranges, values = curve(
        quantity, mass_kg=mass_kg, burst_type=burst_type, n_points=n_points
    )
    if not ranges:
        return (), ()
    w13 = cube_root(float(mass_kg))
    return tuple(r / w13 for r in ranges), values
