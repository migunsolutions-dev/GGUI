"""UFC 3-340-02 Figure 2-13 — scaled height of triple point.

Empirical DPlot series from the official UFC figure file. This is not an
analytical HOB equation and is not Kingery-Bulmash or CONWEP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from validation.metrics import is_finite_number
from validation.ufc_data import load_json
from validation.ufc_interp import bracketing_index, interp1d
from validation.ufc_units import cube_root, english_scaled_to_si, si_scaled_to_english

DOCUMENT = "UFC 3-340-02 Structures to Resist the Effects of Accidental Explosions"
FIGURE = "2-13"
REQUIRED_CHART = (
    "UFC 3-340-02 Figure 2-13. Scaled height of triple point: "
    "HT/W^(1/3) versus scaled horizontal distance, family of Hc/W^(1/3)."
)


def _payload() -> dict:
    return load_json("ufc_3_340_02_fig_2_13.json")


@dataclass(frozen=True)
class UfcHobProvenance:
    document: str = DOCUMENT
    figure_or_table: str = f"Figure {FIGURE}"
    quantity: str = "scaled_triple_point_height"
    data_sha256: str = ""
    populated: bool = True
    required_chart: str = REQUIRED_CHART
    data_kind: str = "ufc_dplot_empirical"
    interpolation_method: str = ""
    revision: str = ""


def _provenance() -> UfcHobProvenance:
    data = _payload()
    return UfcHobProvenance(
        document=data["source_document"],
        figure_or_table=f"Figure {data['source_figure']}",
        data_sha256=str(data.get("source_sha256") or ""),
        populated=True,
        interpolation_method=str(data.get("interpolation_method") or ""),
        revision=str(data.get("source_revision") or ""),
    )


UFC_HOB_PROVENANCE = _provenance()
UFC_HOB_DATASET = tuple(
    (float(curve["hc_published"]), tuple(curve["y_published"]))
    for curve in _payload()["curves"]
)


def published_hc_english() -> Tuple[float, ...]:
    return tuple(float(c["hc_published"]) for c in _payload()["curves"])


@dataclass(frozen=True)
class UfcHobEval:
    ground_range_m: Optional[float]
    hm_m: Optional[float]
    band: Optional[Tuple[float, float]]
    provenance: UfcHobProvenance
    unavailable_reason: str = ""
    hc_scaled_english: Optional[float] = None
    r_scaled_english: Optional[float] = None
    curve_hc_lo: Optional[float] = None
    curve_hc_hi: Optional[float] = None


def _scaled_english(length_m: float, mass_kg: float) -> Optional[float]:
    if not is_finite_number(length_m) or not is_finite_number(mass_kg):
        return None
    if float(mass_kg) <= 0.0:
        return None
    return si_scaled_to_english(float(length_m) / cube_root(float(mass_kg)))


def _ht_on_curve(r_en: float, y_published: Sequence[float]) -> Optional[float]:
    xs = _payload()["x_published"]
    return interp1d(r_en, xs, y_published)


def lookup_mach_stem_height(
    ground_range_m: float,
    hob_m: Optional[float] = None,
    mass_kg: Optional[float] = None,
) -> UfcHobEval:
    prov = UFC_HOB_PROVENANCE
    r_m = float(ground_range_m) if is_finite_number(ground_range_m) else None
    if hob_m is None or mass_kg is None:
        return UfcHobEval(
            ground_range_m=r_m,
            hm_m=None,
            band=None,
            provenance=prov,
            unavailable_reason=(
                "UFC 3-340-02 Figure 2-13: N/A — charge mass W and height of burst "
                "are required to select a scaled-Hc curve."
            ),
        )
    r_en = _scaled_english(ground_range_m, mass_kg)
    hc_en = _scaled_english(hob_m, mass_kg)
    if r_en is None or hc_en is None:
        return UfcHobEval(
            ground_range_m=r_m,
            hm_m=None,
            band=None,
            provenance=prov,
            unavailable_reason="UFC 3-340-02 Figure 2-13: N/A — range, HOB, and mass must be finite and W > 0.",
        )
    keys = published_hc_english()
    bracket = bracketing_index(hc_en, keys)
    if bracket is None:
        return UfcHobEval(
            ground_range_m=r_m,
            hm_m=None,
            band=None,
            provenance=prov,
            hc_scaled_english=hc_en,
            r_scaled_english=r_en,
            unavailable_reason=(
                f"UFC 3-340-02 Figure 2-13: N/A — scaled charge height "
                f"Hc/W^(1/3)={hc_en:.4g} ft/lb^(1/3) is outside the published "
                f"family [{keys[0]:g}, {keys[-1]:g}]. Hc=7 is annotated in the "
                "GRF but has no Y samples."
            ),
        )
    i0, i1, t = bracket
    curves = _payload()["curves"]
    ht0 = _ht_on_curve(r_en, curves[i0]["y_published"])
    ht1 = _ht_on_curve(r_en, curves[i1]["y_published"])
    if ht0 is None or ht1 is None:
        xs = _payload()["x_published"]
        return UfcHobEval(
            ground_range_m=r_m,
            hm_m=None,
            band=None,
            provenance=prov,
            hc_scaled_english=hc_en,
            r_scaled_english=r_en,
            curve_hc_lo=keys[i0],
            curve_hc_hi=keys[i1],
            unavailable_reason=(
                f"UFC 3-340-02 Figure 2-13: N/A — scaled ground range "
                f"{r_en:.4g} ft/lb^(1/3) is outside the published interval "
                f"[{xs[0]:g}, {xs[-1]:g}]."
            ),
        )
    ht_en = ht0 + t * (ht1 - ht0)
    w13 = cube_root(float(mass_kg))
    hm_m = english_scaled_to_si(ht_en) * w13
    lo_m = english_scaled_to_si(ht0) * w13
    hi_m = english_scaled_to_si(ht1) * w13
    band = None if i0 == i1 else (min(lo_m, hi_m), max(lo_m, hi_m))
    return UfcHobEval(
        ground_range_m=float(ground_range_m),
        hm_m=float(hm_m),
        band=band,
        provenance=prov,
        hc_scaled_english=hc_en,
        r_scaled_english=r_en,
        curve_hc_lo=keys[i0],
        curve_hc_hi=keys[i1],
    )


def reference_curve(
    hob_m: Optional[float] = None,
    mass_kg: Optional[float] = None,
) -> Tuple[Sequence[float], Sequence[float]]:
    """Physical (ground range [m], HT [m]) for the current scaled HOB. Empty if N/A."""
    if hob_m is None or mass_kg is None:
        return (), ()
    if not is_finite_number(hob_m) or not is_finite_number(mass_kg) or float(mass_kg) <= 0.0:
        return (), ()
    hc_en = _scaled_english(hob_m, mass_kg)
    keys = published_hc_english()
    if hc_en is None:
        return (), ()
    bracket = bracketing_index(hc_en, keys)
    if bracket is None:
        return (), ()
    i0, i1, t = bracket
    data = _payload()
    xs = data["x_published"]
    y0 = data["curves"][i0]["y_published"]
    y1 = data["curves"][i1]["y_published"]
    w13 = cube_root(float(mass_kg))
    ranges = []
    heights = []
    for x_en, a, b in zip(xs, y0, y1):
        ht_en = float(a) + t * (float(b) - float(a))
        ranges.append(english_scaled_to_si(float(x_en)) * w13)
        heights.append(english_scaled_to_si(ht_en) * w13)
    return tuple(ranges), tuple(heights)
