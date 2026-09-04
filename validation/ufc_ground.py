"""UFC 3-340-02 Figures 2-9 and 2-10 — ground-surface reflected pressure / impulse.

These charts are families of scaled charge height versus angle of incidence at
a reflecting surface. They are not free-field gauges and are not CONWEP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from validation.metrics import is_finite_number
from validation.ufc_data import load_json
from validation.ufc_interp import bracketing_index, interp1d
from validation.ufc_units import (
    PSI_MS_LB13_TO_KPA_MS_KG13,
    PSI_TO_PA,
    cube_root,
    si_scaled_to_english,
)
from validation.units import kpa_ms_to_pa_s

FIGURE_PRESSURE = "2-9"
FIGURE_IMPULSE = "2-10"
GROUND_FRACTION_OF_HOB = 0.01
GROUND_ABS_M = 1.0e-3


def _file(figure: str) -> str:
    if figure == FIGURE_PRESSURE:
        return "ufc_3_340_02_fig_2_09.json"
    if figure == FIGURE_IMPULSE:
        return "ufc_3_340_02_fig_2_10.json"
    raise KeyError(figure)


def _payload(figure: str) -> dict:
    return load_json(_file(figure))


def labeled_curves(figure: str) -> Tuple[dict, ...]:
    curves = []
    for curve in _payload(figure)["curves"]:
        if curve.get("missing_y"):
            continue
        if curve.get("hc_published") is None:
            continue
        if not curve.get("y_published"):
            continue
        curves.append(curve)
    return tuple(curves)


def published_hc_english(figure: str) -> Tuple[float, ...]:
    return tuple(float(c["hc_published"]) for c in labeled_curves(figure))


def alpha_grid(alpha_max_deg: float, npts: int) -> Tuple[float, ...]:
    if npts <= 1:
        return (0.0,)
    amax = float(alpha_max_deg)
    return tuple(amax * i / (npts - 1) for i in range(npts))


def incidence_deg(ground_range_m: float, hob_m: float) -> Optional[float]:
    if not is_finite_number(ground_range_m) or not is_finite_number(hob_m):
        return None
    if float(hob_m) <= 0.0 or float(ground_range_m) < 0.0:
        return None
    return math.degrees(math.atan2(float(ground_range_m), float(hob_m)))


def on_reflecting_surface(
    observer_z_m: Optional[float],
    z_ground_m: float,
    hob_m: float,
) -> bool:
    if observer_z_m is None or not is_finite_number(observer_z_m):
        return False
    tol = max(GROUND_ABS_M, GROUND_FRACTION_OF_HOB * max(abs(float(hob_m)), 0.0))
    return abs(float(observer_z_m) - float(z_ground_m)) <= tol


@dataclass(frozen=True)
class UfcGroundEval:
    figure: str
    quantity: str
    value_si: Optional[float]
    alpha_deg: Optional[float]
    hc_scaled_english: Optional[float]
    curve_hc_lo: Optional[float]
    curve_hc_hi: Optional[float]
    citation: str
    unavailable_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value_si is not None and is_finite_number(self.value_si)


def _lookup_y(figure: str, alpha_deg: float, hc_en: float) -> Optional[Tuple[float, float, float]]:
    keys = published_hc_english(figure)
    bracket = bracketing_index(hc_en, keys)
    if bracket is None:
        return None
    i0, i1, t = bracket
    curves = labeled_curves(figure)
    npts = int(_payload(figure)["npts"])

    def on_curve(curve: dict) -> Optional[float]:
        amax = curve.get("alpha_max_deg")
        if amax is None:
            return None
        xs = alpha_grid(float(amax), npts)
        return interp1d(alpha_deg, xs, curve["y_published"])

    y0 = on_curve(curves[i0])
    y1 = on_curve(curves[i1])
    if y0 is None or y1 is None:
        return None
    return y0 + t * (y1 - y0), keys[i0], keys[i1]


def lookup(
    figure: str,
    *,
    ground_range_m: float,
    hob_m: float,
    mass_kg: float,
    observer_z_m: Optional[float] = 0.0,
    z_ground_m: float = 0.0,
) -> UfcGroundEval:
    data = _payload(figure)
    citation = (
        f"{data['source_document']}, Figure {data['source_figure']}. "
        f"{data['source_figure_title']}."
    )
    quantity = "peak_reflected_overpressure" if figure == FIGURE_PRESSURE else "positive_reflected_impulse"
    if not on_reflecting_surface(observer_z_m, z_ground_m, hob_m):
        return UfcGroundEval(
            figure=f"Figure {data['source_figure']}",
            quantity=quantity,
            value_si=None,
            alpha_deg=None,
            hc_scaled_english=None,
            curve_hc_lo=None,
            curve_hc_hi=None,
            citation=citation,
            unavailable_reason=(
                f"UFC Figure {data['source_figure']}: N/A — comparison applies only "
                "at the reflecting surface, not to a gauge above the ground."
            ),
        )
    alpha = incidence_deg(ground_range_m, hob_m)
    if alpha is None:
        return UfcGroundEval(
            figure=f"Figure {data['source_figure']}",
            quantity=quantity,
            value_si=None,
            alpha_deg=None,
            hc_scaled_english=None,
            curve_hc_lo=None,
            curve_hc_hi=None,
            citation=citation,
            unavailable_reason=(
                f"UFC Figure {data['source_figure']}: N/A — HOB must be positive "
                "to define the angle of incidence."
            ),
        )
    if not is_finite_number(mass_kg) or float(mass_kg) <= 0.0:
        return UfcGroundEval(
            figure=f"Figure {data['source_figure']}",
            quantity=quantity,
            value_si=None,
            alpha_deg=alpha,
            hc_scaled_english=None,
            curve_hc_lo=None,
            curve_hc_hi=None,
            citation=citation,
            unavailable_reason=f"UFC Figure {data['source_figure']}: N/A — charge mass W must be positive.",
        )
    hc_en = si_scaled_to_english(float(hob_m) / cube_root(float(mass_kg)))
    keys = published_hc_english(figure)
    got = _lookup_y(figure, alpha, hc_en)
    if got is None:
        return UfcGroundEval(
            figure=f"Figure {data['source_figure']}",
            quantity=quantity,
            value_si=None,
            alpha_deg=alpha,
            hc_scaled_english=hc_en,
            curve_hc_lo=None,
            curve_hc_hi=None,
            citation=citation,
            unavailable_reason=(
                f"UFC Figure {data['source_figure']}: N/A — scaled Hc/W^(1/3)="
                f"{hc_en:.4g} ft/lb^(1/3) or α={alpha:.4g} deg is outside the "
                f"labeled published family [{keys[0]:g}, {keys[-1]:g}] "
                "(unlabeled first GRF series and extra annotated Hc without Y are excluded)."
            ),
        )
    y_pub, hc_lo, hc_hi = got
    if figure == FIGURE_PRESSURE:
        value = float(y_pub) * PSI_TO_PA
    else:
        scaled_kpa_ms = float(y_pub) * PSI_MS_LB13_TO_KPA_MS_KG13
        value = kpa_ms_to_pa_s(scaled_kpa_ms * cube_root(float(mass_kg)))
    return UfcGroundEval(
        figure=f"Figure {data['source_figure']}",
        quantity=quantity,
        value_si=value,
        alpha_deg=alpha,
        hc_scaled_english=hc_en,
        curve_hc_lo=hc_lo,
        curve_hc_hi=hc_hi,
        citation=citation,
    )


def reference_curve_vs_range(
    figure: str,
    *,
    hob_m: float,
    mass_kg: float,
    n_points: int = 80,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Ground range [m] vs SI quantity for the current scaled HOB, on the surface."""
    if not is_finite_number(hob_m) or not is_finite_number(mass_kg):
        return (), ()
    if float(hob_m) <= 0.0 or float(mass_kg) <= 0.0:
        return (), ()
    keys = published_hc_english(figure)
    hc_en = si_scaled_to_english(float(hob_m) / cube_root(float(mass_kg)))
    if bracketing_index(hc_en, keys) is None:
        return (), ()
    curves = labeled_curves(figure)
    amax = min(float(c["alpha_max_deg"]) for c in curves if c.get("alpha_max_deg") is not None)
    if n_points < 2:
        n_points = 2
    xs = []
    ys = []
    for i in range(n_points):
        alpha = amax * i / (n_points - 1)
        r = float(hob_m) * math.tan(math.radians(alpha))
        ev = lookup(
            figure,
            ground_range_m=r,
            hob_m=hob_m,
            mass_kg=mass_kg,
            observer_z_m=0.0,
            z_ground_m=0.0,
        )
        if ev.ok:
            xs.append(r)
            ys.append(float(ev.value_si))
    return tuple(xs), tuple(ys)
