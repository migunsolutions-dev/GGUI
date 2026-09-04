"""Per-sample UFC/KB comparison. Never apply one global reference to mixed series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from validation import kingery_bulmash as kb
from validation import ufc_airblast as ufc_ab
from validation.metrics import is_finite_number, relative_error_percent
from validation.ufc_airblast import scaled_distance

SOURCE_UFC = "UFC 3-340-02"
SOURCE_SWISDAK = "Swisdak 1994"

THREE_D_HEMI_NA = (
    "Hemispherical/reflected reference is not applied to an arbitrary 3D gauge "
    "without surface/orientation data."
)
INCOMPATIBLE_REFERENCE = "Sample reference is incompatible with this series; comparison is N/A."
INVALID_HISTORY = "History is not valid for a UFC/KB comparison."


@dataclass(frozen=True)
class OverlaySample:
    point_id: str
    dim: str
    mass_kg: float
    burst: str
    figure: str
    reference_source: str
    range_m: float
    scaled_z: Optional[float]
    bf_peak: Optional[float] = None
    bf_impulse: Optional[float] = None
    comparable: bool = False
    validity_reason: str = ""
    kind: str = "planned"
    probe_ok: bool = True


@dataclass(frozen=True)
class SampleEval:
    sample: OverlaySample
    ref_peak: Optional[float]
    ref_impulse: Optional[float]
    error_peak_pct: Optional[float]
    error_impulse_pct: Optional[float]
    applicable: bool
    na_reason: str = ""


@dataclass(frozen=True)
class ReferenceGroup:
    key: Tuple[Any, ...]
    source: str
    burst: str
    figure: str
    mass_kg: float
    label: str
    samples: Tuple[OverlaySample, ...]


def is_ufc_source(source: str) -> bool:
    return str(source or "").startswith("UFC")


def is_hemispherical(burst: str) -> bool:
    return "hemi" in str(burst or "").lower()


def reference_label(source: str, burst: str, figure: str = "") -> str:
    if is_ufc_source(source):
        fig = figure or ufc_ab.figure_id(burst)
        if fig:
            return f"{SOURCE_UFC} Figure {fig}"
        if burst == ufc_ab.BURST_SPHERICAL:
            return f"{SOURCE_UFC} Figure 2-7"
        if burst == ufc_ab.BURST_HEMISPHERICAL:
            return f"{SOURCE_UFC} Figure 2-15"
        return SOURCE_UFC
    if burst == kb.BURST_SPHERICAL:
        return "Kingery-Bulmash / Swisdak 1994 (spherical)"
    return "Kingery-Bulmash / Swisdak 1994"


def group_key(sample: OverlaySample) -> Tuple[Any, ...]:
    mass = round(float(sample.mass_kg), 9) if is_finite_number(sample.mass_kg) else None
    return (
        str(sample.reference_source or ""),
        str(sample.burst or ""),
        str(sample.figure or ""),
        mass,
    )


def reference_applicable(dim: str, burst: str) -> bool:
    if str(dim).strip().lower() != "3d":
        return True
    return not is_hemispherical(burst)


def engine_for(source: str):
    return ufc_ab if is_ufc_source(source) else kb


def evaluate_sample(sample: OverlaySample) -> SampleEval:
    """Compare a sample only to *its* mass/burst/figure, never a global overlay radio."""
    if not sample.probe_ok:
        return SampleEval(
            sample=sample,
            ref_peak=None,
            ref_impulse=None,
            error_peak_pct=None,
            error_impulse_pct=None,
            applicable=False,
            na_reason=sample.validity_reason
            or "Probe location does not match the planned Validation Point.",
        )
    if str(sample.dim).strip().lower() == "3d" and is_hemispherical(sample.burst):
        return SampleEval(
            sample=sample,
            ref_peak=None,
            ref_impulse=None,
            error_peak_pct=None,
            error_impulse_pct=None,
            applicable=False,
            na_reason=THREE_D_HEMI_NA,
        )
    engine = engine_for(sample.reference_source)
    burst = sample.burst
    mass = sample.mass_kg
    rng = sample.range_m
    if not is_finite_number(mass) or float(mass) <= 0.0 or not is_finite_number(rng):
        return SampleEval(
            sample=sample,
            ref_peak=None,
            ref_impulse=None,
            error_peak_pct=None,
            error_impulse_pct=None,
            applicable=False,
            na_reason=INCOMPATIBLE_REFERENCE,
        )
    peak_ev = engine.evaluate(
        engine.QUANTITY_PEAK_PRESSURE, range_m=float(rng), mass_kg=float(mass), burst_type=burst
    )
    imp_ev = engine.evaluate(
        engine.QUANTITY_INCIDENT_IMPULSE, range_m=float(rng), mass_kg=float(mass), burst_type=burst
    )
    ref_p = peak_ev.value_si if peak_ev.ok else None
    ref_i = imp_ev.value_si if imp_ev.ok else None
    comparable = bool(sample.comparable and sample.kind in ("bf", "user"))
    if not comparable:
        reason = sample.validity_reason or (
            INVALID_HISTORY if sample.kind in ("bf", "user", "invalid") else ""
        )
        return SampleEval(
            sample=sample,
            ref_peak=ref_p,
            ref_impulse=ref_i,
            error_peak_pct=None,
            error_impulse_pct=None,
            applicable=bool(ref_p is not None or ref_i is not None),
            na_reason=reason,
        )
    return SampleEval(
        sample=sample,
        ref_peak=ref_p,
        ref_impulse=ref_i,
        error_peak_pct=relative_error_percent(sample.bf_peak, ref_p),
        error_impulse_pct=relative_error_percent(sample.bf_impulse, ref_i),
        applicable=True,
        na_reason="",
    )


def group_samples(samples: Sequence[OverlaySample]) -> Tuple[ReferenceGroup, ...]:
    buckets: Dict[Tuple[Any, ...], List[OverlaySample]] = {}
    order: List[Tuple[Any, ...]] = []
    for sample in samples:
        key = group_key(sample)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(sample)
    groups = []
    for key in order:
        items = tuple(buckets[key])
        first = items[0]
        groups.append(
            ReferenceGroup(
                key=key,
                source=first.reference_source,
                burst=first.burst,
                figure=first.figure,
                mass_kg=float(first.mass_kg) if is_finite_number(first.mass_kg) else 0.0,
                label=reference_label(first.reference_source, first.burst, first.figure),
                samples=items,
            )
        )
    return tuple(groups)


def mixed_references(groups: Sequence[ReferenceGroup]) -> bool:
    if len(groups) <= 1:
        return False
    signatures = {(g.source, g.burst, g.figure) for g in groups}
    return len(signatures) > 1


def curve_for_group(
    group: ReferenceGroup,
    *,
    quantity: str,
    vs_z: bool,
) -> Tuple[List[float], List[float]]:
    engine = engine_for(group.source)
    fn = engine.curve_vs_z if vs_z else engine.curve
    xr, yr = fn(quantity, mass_kg=group.mass_kg, burst_type=group.burst)
    return list(xr or []), list(yr or [])


def sample_from_plan_point(plan: Any, point: Any, **kwargs: Any) -> OverlaySample:
    mass = float(getattr(point, "mass_kg", None) or plan.mass_kg or 0.0)
    burst = str(getattr(point, "burst", None) or plan.burst_master or "")
    figure = str(getattr(point, "figure", None) or plan.figure or "")
    source = str(getattr(point, "reference_source", None) or SOURCE_UFC)
    rng = float(point.range_m)
    z_val = getattr(point, "scaled_z", None)
    if z_val is None:
        z_val = scaled_distance(rng, mass)
    return OverlaySample(
        point_id=str(point.point_id),
        dim=str(point.dim or plan.dim),
        mass_kg=mass,
        burst=burst,
        figure=figure,
        reference_source=source,
        range_m=rng,
        scaled_z=z_val,
        **kwargs,
    )
