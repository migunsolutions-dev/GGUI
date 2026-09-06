"""Automatic Validation Point generation for 1D/2D reference comparison.

Sampling master is the verified UFC airblast table for the current model.
User gauges are never required for the normal 1D/2D workflow.

1D line
-------
Radial spherical radius from the charge centre at the origin. R is the
physical radius along the 1D wedge. Domain limit is CaseInputs1D.radius.

2D line
-------
One horizontal sampling line through the charge centre in the r–z plane:
z = HOB, r increasing. Standoff equals r because the charge sits on the axis.
This is the free-field incident line of an elevated spherical charge before
ground reflection contaminates the sample.

If HOB is effectively zero (surface burst), that same rule is the ground
surface line used by hemispherical charts.

UFC table used as the sampling master
-------------------------------------
* 1D: always Figure 2-7 (free-air spherical).
* 2D with HOB > 0: Figure 2-7.
* 2D with HOB ≈ 0: Figure 2-15 (hemispherical surface burst).

Switching the displayed reference (UFC ↔ Swisdak, spherical ↔ hemi) reuses
these points. Values outside that reference's own Z interval are N/A.
No second solver-probe set is created for a display-only reference change.

R_min = Z_min,UFC * W**(1/3) using validation.ufc_units.cube_root.
R_max = min(Z_max,UFC * W**(1/3), available simulation range).
No extrapolation.

Point count uses the default Validation plot width and Qt logical DPI so
neighbouring markers are about 5 mm apart at the normal application size.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ui_metrics import COMPUTATIONAL_LEFT_PANEL_WIDTH, DEFAULT_WINDOW_WIDTH
from validation import ufc_airblast as ufc_ab
from validation.fingerprint import attach_plan_fingerprint, build_fingerprint
from validation.kb_overlay import SOURCE_UFC
from validation.kb_propagation import (
    exclusive_independent_r_min,
    first_independent_r_m,
    physical_standoff_m,
)
from validation.metrics import is_finite_number
from validation.ufc_airblast import scaled_distance
from validation.ufc_units import cube_root

PURPOSE_VALIDATION = "validation"
SAMPLING_AUTOMATIC = "automatic"
SOURCE_SOLVER_PROBE = "solver_probe"
SOURCE_EXISTING_1D_PROBES = "existing_high_res_probes1d"
SOURCE_UNAVAILABLE = "unavailable"

FO_1D_EXISTING = "probes1d"
FO_1D_VALIDATION = "validationGauges1d"
FO_2D_VALIDATION = "validationGauges2d"

LINE_RADIAL_1D = "radial_from_charge_centre"
LINE_HORIZONTAL_2D = "horizontal_through_charge_centre"

TARGET_SPACING_MM = 5.0
DEFAULT_LOGICAL_DPI_X = 96.0
DEFAULT_PLOT_WIDTH_PX = float(DEFAULT_WINDOW_WIDTH - COMPUTATIONAL_LEFT_PANEL_WIDTH - 16)
MIN_POINTS = 6
MAX_POINTS = 80
SURFACE_HOB_M = 1.0e-9
REMAP_NO_VALID_DOMAIN = (
    "Validation cannot be generated reliably: not enough physical domain "
    "outside the remap receiving region."
)
REMAP_RECEIVE_NOTE = (
    "Automatic KB propagation points are placed outside the copied 1D remap "
    "region plus one target cell, using the actual remap metadata. Inside/"
    "on-boundary locations belong to Remap Validation, not KB propagation."
)


@dataclass(frozen=True)
class ValidationPoint:
    point_id: str
    dim: str
    index: int
    range_m: float
    x: float
    y: float
    z: float
    purpose: str = PURPOSE_VALIDATION
    reference_sampling: str = SAMPLING_AUTOMATIC
    mass_kg: float = 0.0
    burst: str = ""
    figure: str = ""
    reference_source: str = SOURCE_UFC
    scaled_z: Optional[float] = None


@dataclass(frozen=True)
class SamplingPlan:
    dim: str
    burst_master: str
    figure: str
    mass_kg: float
    charge_center: Tuple[float, float, float]
    r_min: float
    r_max: float
    z_min: float
    z_max: float
    n_points: int
    line_kind: str
    line_z: Optional[float]
    points: Tuple[ValidationPoint, ...]
    function_object: str
    data_source: str
    notes: Tuple[str, ...] = ()
    domain_r_max: float = 0.0
    purpose: str = PURPOSE_VALIDATION
    reference_sampling: str = SAMPLING_AUTOMATIC
    extra: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Dict[str, Any] = field(default_factory=dict)
    remap_timing: Dict[str, Any] = field(default_factory=dict)
    remap_receive_r_max: Optional[float] = None

    @property
    def ok(self) -> bool:
        return bool(self.points) and self.r_max > self.r_min > 0.0


def runtime_logical_dpi_x() -> float:
    """Qt screen logical DPI when a QApplication exists; otherwise 96.

    Generation must not assume a particular monitor. The default plot width is
    still the nominal Validation pane, so resize after a run only thins markers.
    """
    try:
        from PyQt5.QtWidgets import QApplication
    except Exception:
        return DEFAULT_LOGICAL_DPI_X
    app = QApplication.instance()
    if app is None:
        return DEFAULT_LOGICAL_DPI_X
    try:
        screen = app.primaryScreen()
        dpi = float(screen.logicalDotsPerInchX()) if screen is not None else 0.0
    except Exception:
        return DEFAULT_LOGICAL_DPI_X
    if not math.isfinite(dpi) or dpi <= 0.0:
        return DEFAULT_LOGICAL_DPI_X
    return dpi


def pixels_per_mm(logical_dpi_x: float) -> float:
    dpi = float(logical_dpi_x)
    if not math.isfinite(dpi) or dpi <= 0.0:
        dpi = DEFAULT_LOGICAL_DPI_X
    return dpi / 25.4


def target_spacing_px(
    logical_dpi_x: float = DEFAULT_LOGICAL_DPI_X,
    spacing_mm: float = TARGET_SPACING_MM,
) -> float:
    return float(spacing_mm) * pixels_per_mm(logical_dpi_x)


def point_count(
    usable_width_px: float,
    logical_dpi_x: float = DEFAULT_LOGICAL_DPI_X,
    *,
    spacing_mm: float = TARGET_SPACING_MM,
) -> int:
    """N ≈ usable_plot_width_px / target_spacing_px + 1, clamped to guards."""
    width = float(usable_width_px)
    if not math.isfinite(width) or width <= 0.0:
        width = DEFAULT_PLOT_WIDTH_PX
    spacing = target_spacing_px(logical_dpi_x, spacing_mm)
    if not math.isfinite(spacing) or spacing <= 0.0:
        return MIN_POINTS
    n = int(round(width / spacing + 1.0))
    return max(MIN_POINTS, min(MAX_POINTS, n))


def marker_stride(
    n_stored: int,
    current_width_px: float,
    logical_dpi_x: float = DEFAULT_LOGICAL_DPI_X,
) -> int:
    """Keep every k-th marker so on-screen spacing stays near 5 mm after resize."""
    n = int(n_stored)
    if n <= 1:
        return 1
    target = point_count(current_width_px, logical_dpi_x)
    if target >= n:
        return 1
    return max(1, int(math.ceil(n / float(target))))


def log_spaced(r_min: float, r_max: float, n: int) -> Tuple[float, ...]:
    """Radii whose log(R) is uniformly spaced. No points outside [r_min, r_max]."""
    if not is_finite_number(r_min) or not is_finite_number(r_max):
        return ()
    lo = float(r_min)
    hi = float(r_max)
    if lo <= 0.0 or hi <= lo:
        return ()
    count = int(n)
    if count <= 1:
        return (lo,)
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return tuple(
        math.exp(log_lo + (log_hi - log_lo) * i / (count - 1)) for i in range(count)
    )


def sampling_burst_1d() -> str:
    return ufc_ab.BURST_SPHERICAL


def sampling_burst_2d(hob_m: Optional[float]) -> str:
    if is_finite_number(hob_m) and float(hob_m) > SURFACE_HOB_M:
        return ufc_ab.BURST_SPHERICAL
    return ufc_ab.BURST_HEMISPHERICAL


def ufc_range_m(mass_kg: float, burst_type: str) -> Optional[Tuple[float, float, float, float, str]]:
    """Return (r_min, r_max, z_min, z_max, figure) from the UFC table. None if unusable."""
    interval = ufc_ab.z_interval(burst_type)
    if interval is None or not is_finite_number(mass_kg) or float(mass_kg) <= 0.0:
        return None
    z_lo, z_hi = interval
    w13 = cube_root(float(mass_kg))
    r_lo = float(z_lo) * w13
    r_hi = float(z_hi) * w13
    if r_lo <= 0.0 or r_hi <= r_lo:
        return None
    return r_lo, r_hi, float(z_lo), float(z_hi), ufc_ab.figure_id(burst_type)


def intersect_range(r_lo: float, r_hi: float, domain_r_max: float, *, inset_m: float) -> Optional[Tuple[float, float]]:
    """Clip UFC radii to the simulated domain. Returns None when the intersection is empty."""
    if not is_finite_number(domain_r_max) or float(domain_r_max) <= 0.0:
        return None
    available = float(domain_r_max) - max(float(inset_m), 0.0)
    if available <= 0.0:
        return None
    r_min = float(r_lo)
    r_max = min(float(r_hi), available)
    if r_max <= r_min:
        return None
    return r_min, r_max


def _inset(domain_r_max: float, cell_size: Optional[float]) -> float:
    if is_finite_number(cell_size) and float(cell_size) > 0.0:
        return min(0.5 * float(cell_size), 0.05 * float(domain_r_max))
    return max(1.0e-4 * float(domain_r_max), 1.0e-6)


def _pid(dim: str, index: int) -> str:
    return f"VAL_{dim.upper()}_{index + 1:03d}"


def _point(
    *,
    dim: str,
    index: int,
    range_m: float,
    x: float,
    y: float,
    z: float,
    mass_kg: float,
    burst: str,
    figure: str,
) -> ValidationPoint:
    return ValidationPoint(
        point_id=_pid(dim, index),
        dim=dim,
        index=index,
        range_m=range_m,
        x=x,
        y=y,
        z=z,
        mass_kg=float(mass_kg),
        burst=burst,
        figure=figure,
        reference_source=SOURCE_UFC,
        scaled_z=scaled_distance(range_m, mass_kg),
    )


def plan_1d(
    *,
    mass_kg: float,
    domain_radius_m: float,
    cell_size: Optional[float] = None,
    usable_width_px: float = DEFAULT_PLOT_WIDTH_PX,
    logical_dpi_x: float = DEFAULT_LOGICAL_DPI_X,
) -> SamplingPlan:
    notes: List[str] = []
    empty = SamplingPlan(
        dim="1d",
        burst_master=sampling_burst_1d(),
        figure="",
        mass_kg=float(mass_kg) if is_finite_number(mass_kg) else 0.0,
        charge_center=(0.0, 0.0, 0.0),
        r_min=0.0,
        r_max=0.0,
        z_min=0.0,
        z_max=0.0,
        n_points=0,
        line_kind=LINE_RADIAL_1D,
        line_z=0.0,
        points=(),
        function_object=FO_1D_EXISTING,
        data_source=SOURCE_EXISTING_1D_PROBES,
        notes=("1D automatic sampling requires a positive charge mass and domain radius.",),
        domain_r_max=float(domain_radius_m) if is_finite_number(domain_radius_m) else 0.0,
    )
    ufc = ufc_range_m(mass_kg, sampling_burst_1d())
    if ufc is None:
        return empty
    r_lo, r_hi, z_lo, z_hi, figure = ufc
    clipped = intersect_range(r_lo, r_hi, domain_radius_m, inset_m=_inset(domain_radius_m, cell_size))
    if clipped is None:
        notes.append(
            "UFC Figure 2-7 range does not intersect the 1D domain; automatic points are N/A (no extrapolation)."
        )
        return SamplingPlan(
            dim="1d",
            burst_master=sampling_burst_1d(),
            figure=figure,
            mass_kg=float(mass_kg),
            charge_center=(0.0, 0.0, 0.0),
            r_min=r_lo,
            r_max=r_hi,
            z_min=z_lo,
            z_max=z_hi,
            n_points=0,
            line_kind=LINE_RADIAL_1D,
            line_z=0.0,
            points=(),
            function_object=FO_1D_EXISTING,
            data_source=SOURCE_UNAVAILABLE,
            notes=tuple(notes),
            domain_r_max=float(domain_radius_m),
        )
    r_min, r_max = clipped
    n = point_count(usable_width_px, logical_dpi_x)
    radii = log_spaced(r_min, r_max, n)
    points = tuple(
        _point(
            dim="1d",
            index=i,
            range_m=r,
            x=r,
            y=0.0,
            z=0.0,
            mass_kg=float(mass_kg),
            burst=sampling_burst_1d(),
            figure=figure,
        )
        for i, r in enumerate(radii)
    )
    return SamplingPlan(
        dim="1d",
        burst_master=sampling_burst_1d(),
        figure=figure,
        mass_kg=float(mass_kg),
        charge_center=(0.0, 0.0, 0.0),
        r_min=r_min,
        r_max=r_max,
        z_min=z_lo,
        z_max=z_hi,
        n_points=len(points),
        line_kind=LINE_RADIAL_1D,
        line_z=0.0,
        points=points,
        function_object=FO_1D_EXISTING,
        data_source=SOURCE_EXISTING_1D_PROBES,
        notes=(
            "1D Validation Points are log-spaced along the spherical radius. "
            "High-resolution histories come from the existing probes1d function object; "
            "no extra solver probe set is written.",
        ),
        domain_r_max=float(domain_radius_m),
    )


def plan_2d(
    *,
    mass_kg: float,
    domain_radius_m: float,
    domain_height_m: float,
    hob_m: float,
    cell_size: Optional[float] = None,
    usable_width_px: float = DEFAULT_PLOT_WIDTH_PX,
    logical_dpi_x: float = DEFAULT_LOGICAL_DPI_X,
    remap_receive_r_max: Optional[float] = None,
) -> SamplingPlan:
    burst = sampling_burst_2d(hob_m)
    notes: List[str] = []
    empty = SamplingPlan(
        dim="2d",
        burst_master=burst,
        figure="",
        mass_kg=float(mass_kg) if is_finite_number(mass_kg) else 0.0,
        charge_center=(0.0, float(hob_m) if is_finite_number(hob_m) else 0.0, 0.0),
        r_min=0.0,
        r_max=0.0,
        z_min=0.0,
        z_max=0.0,
        n_points=0,
        line_kind=LINE_HORIZONTAL_2D,
        line_z=float(hob_m) if is_finite_number(hob_m) else None,
        points=(),
        function_object=FO_2D_VALIDATION,
        data_source=SOURCE_SOLVER_PROBE,
        notes=("2D automatic sampling requires mass, domain radius/height, and HOB.",),
        domain_r_max=float(domain_radius_m) if is_finite_number(domain_radius_m) else 0.0,
    )
    if not is_finite_number(hob_m) or not is_finite_number(domain_height_m):
        return empty
    z_line = float(hob_m)
    if z_line < -1.0e-12 or z_line > float(domain_height_m) + 1.0e-12:
        notes.append(
            "Charge centre height is outside the 2D domain; automatic validation line is N/A."
        )
        return SamplingPlan(
            dim="2d",
            burst_master=burst,
            figure=ufc_ab.figure_id(burst),
            mass_kg=float(mass_kg) if is_finite_number(mass_kg) else 0.0,
            charge_center=(0.0, z_line, 0.0),
            r_min=0.0,
            r_max=0.0,
            z_min=0.0,
            z_max=0.0,
            n_points=0,
            line_kind=LINE_HORIZONTAL_2D,
            line_z=z_line,
            points=(),
            function_object=FO_2D_VALIDATION,
            data_source=SOURCE_UNAVAILABLE,
            notes=tuple(notes),
            domain_r_max=float(domain_radius_m) if is_finite_number(domain_radius_m) else 0.0,
        )
    ufc = ufc_range_m(mass_kg, burst)
    if ufc is None:
        return empty
    r_lo, r_hi, z_lo, z_hi, figure = ufc
    clipped = intersect_range(r_lo, r_hi, domain_radius_m, inset_m=_inset(domain_radius_m, cell_size))
    if clipped is None:
        notes.append(
            f"UFC Figure {figure} range does not intersect the 2D domain; "
            "automatic points are N/A (no extrapolation)."
        )
        return SamplingPlan(
            dim="2d",
            burst_master=burst,
            figure=figure,
            mass_kg=float(mass_kg),
            charge_center=(0.0, z_line, 0.0),
            r_min=r_lo,
            r_max=r_hi,
            z_min=z_lo,
            z_max=z_hi,
            n_points=0,
            line_kind=LINE_HORIZONTAL_2D,
            line_z=z_line,
            points=(),
            function_object=FO_2D_VALIDATION,
            data_source=SOURCE_UNAVAILABLE,
            notes=tuple(notes),
            domain_r_max=float(domain_radius_m),
        )
    r_min, r_max = clipped
    receive = (
        float(remap_receive_r_max)
        if is_finite_number(remap_receive_r_max) and float(remap_receive_r_max) > 0.0
        else None
    )
    charge_center = (0.0, z_line, 0.0)
    dx = float(cell_size) if is_finite_number(cell_size) and float(cell_size) > 0.0 else None
    if receive is not None:
        lo_phys = exclusive_independent_r_min(receive, dx)
        if r_max <= first_independent_r_m(receive, dx):
            notes.append(REMAP_NO_VALID_DOMAIN)
            return SamplingPlan(
                dim="2d",
                burst_master=burst,
                figure=figure,
                mass_kg=float(mass_kg),
                charge_center=charge_center,
                r_min=r_min,
                r_max=r_max,
                z_min=z_lo,
                z_max=z_hi,
                n_points=0,
                line_kind=LINE_HORIZONTAL_2D,
                line_z=z_line,
                points=(),
                function_object=FO_2D_VALIDATION,
                data_source=SOURCE_UNAVAILABLE,
                notes=tuple(notes),
                domain_r_max=float(domain_radius_m),
                remap_receive_r_max=receive,
                extra={
                    "remap_region": {
                        "center": [0.0, z_line, 0.0],
                        "radius_m": receive,
                        "exclusion_guard_m": float(dx or 0.0),
                        "first_independent_r_m": first_independent_r_m(receive, dx),
                    }
                },
            )
        r_min = max(r_min, lo_phys)
        notes.append(
            REMAP_RECEIVE_NOTE
            + f" (R > {first_independent_r_m(receive, dx):.6g} m)."
        )
    n = point_count(usable_width_px, logical_dpi_x)
    radii = log_spaced(r_min, r_max, n)
    points = []
    for i, r in enumerate(radii):
        rng = physical_standoff_m("2d", (r, z_line, 0.0), charge_center)
        points.append(
            _point(
                dim="2d",
                index=i,
                range_m=rng,
                x=r,
                y=z_line,
                z=0.0,
                mass_kg=float(mass_kg),
                burst=burst,
                figure=figure,
            )
        )
    points = tuple(points)
    notes.append(
        "2D Validation Points lie on the horizontal line through the charge centre "
        "(z = HOB, r increasing). One line only; no angular rays."
    )
    return SamplingPlan(
        dim="2d",
        burst_master=burst,
        figure=figure,
        mass_kg=float(mass_kg),
        charge_center=charge_center,
        r_min=r_min,
        r_max=r_max,
        z_min=z_lo,
        z_max=z_hi,
        n_points=len(points),
        line_kind=LINE_HORIZONTAL_2D,
        line_z=z_line,
        points=points,
        function_object=FO_2D_VALIDATION,
        data_source=SOURCE_SOLVER_PROBE,
        notes=tuple(notes),
        domain_r_max=float(domain_radius_m),
        remap_receive_r_max=receive,
        extra=(
            {
                "remap_region": {
                    "center": [0.0, z_line, 0.0],
                    "radius_m": receive,
                    "exclusion_guard_m": float(dx or 0.0),
                    "first_independent_r_m": first_independent_r_m(receive, dx),
                }
            }
            if receive is not None
            else {}
        ),
    )


def plan_to_dict(plan: SamplingPlan) -> Dict[str, Any]:
    payload = asdict(plan)
    payload["charge_center"] = list(plan.charge_center)
    payload["points"] = [asdict(p) for p in plan.points]
    payload["notes"] = list(plan.notes)
    return payload


def plan_from_dict(data: Dict[str, Any]) -> Optional[SamplingPlan]:
    if not isinstance(data, dict):
        return None
    try:
        pts = []
        for raw in data.get("points") or ():
            mass = float(raw.get("mass_kg") or data.get("mass_kg") or 0.0)
            burst = str(raw.get("burst") or data.get("burst_master") or "")
            figure = str(raw.get("figure") or data.get("figure") or "")
            rng = float(raw["range_m"])
            z_val = raw.get("scaled_z")
            pts.append(
                ValidationPoint(
                    point_id=str(raw["point_id"]),
                    dim=str(raw["dim"]),
                    index=int(raw["index"]),
                    range_m=rng,
                    x=float(raw["x"]),
                    y=float(raw["y"]),
                    z=float(raw["z"]),
                    purpose=str(raw.get("purpose") or PURPOSE_VALIDATION),
                    reference_sampling=str(raw.get("reference_sampling") or SAMPLING_AUTOMATIC),
                    mass_kg=mass,
                    burst=burst,
                    figure=figure,
                    reference_source=str(raw.get("reference_source") or SOURCE_UFC),
                    scaled_z=None if z_val is None else float(z_val),
                )
            )
        cc = data.get("charge_center") or (0.0, 0.0, 0.0)
        receive = data.get("remap_receive_r_max")
        return SamplingPlan(
            dim=str(data.get("dim") or ""),
            burst_master=str(data.get("burst_master") or ""),
            figure=str(data.get("figure") or ""),
            mass_kg=float(data.get("mass_kg") or 0.0),
            charge_center=(float(cc[0]), float(cc[1]), float(cc[2])),
            r_min=float(data.get("r_min") or 0.0),
            r_max=float(data.get("r_max") or 0.0),
            z_min=float(data.get("z_min") or 0.0),
            z_max=float(data.get("z_max") or 0.0),
            n_points=int(data.get("n_points") or len(pts)),
            line_kind=str(data.get("line_kind") or ""),
            line_z=None if data.get("line_z") is None else float(data.get("line_z")),
            points=tuple(pts),
            function_object=str(data.get("function_object") or ""),
            data_source=str(data.get("data_source") or ""),
            notes=tuple(data.get("notes") or ()),
            domain_r_max=float(data.get("domain_r_max") or 0.0),
            purpose=str(data.get("purpose") or PURPOSE_VALIDATION),
            reference_sampling=str(data.get("reference_sampling") or SAMPLING_AUTOMATIC),
            extra=dict(data.get("extra") or {}),
            fingerprint=dict(data.get("fingerprint") or {}),
            remap_timing=dict(data.get("remap_timing") or {}),
            remap_receive_r_max=None if receive is None else float(receive),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def cache_key(
    *,
    case_1d: Optional[str],
    case_2d: Optional[str],
    mass_kg: Optional[float],
    domain_1d: Optional[float],
    domain_2d: Optional[float],
    height_2d: Optional[float],
    hob_m: Optional[float],
) -> Tuple[Any, ...]:
    return (
        str(case_1d or ""),
        str(case_2d or ""),
        None if mass_kg is None else round(float(mass_kg), 12),
        None if domain_1d is None else round(float(domain_1d), 12),
        None if domain_2d is None else round(float(domain_2d), 12),
        None if height_2d is None else round(float(height_2d), 12),
        None if hob_m is None else round(float(hob_m), 12),
    )


def nearest_index(radii: Sequence[float], target: float) -> Optional[int]:
    if not radii or not is_finite_number(target):
        return None
    best_i = None
    best_d = None
    t = float(target)
    for i, r in enumerate(radii):
        if not is_finite_number(r):
            continue
        d = abs(float(r) - t)
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    return best_i


def expected_plan_fingerprint(
    plan: SamplingPlan,
    *,
    case_path: Optional[str],
    cell_size: Optional[float] = None,
    hob_m: Optional[float] = None,
    domain_height_m: Optional[float] = None,
    remap_receive_r_max: Optional[float] = None,
    include_coordinates: bool = True,
    source_model: Optional[str] = None,
) -> Dict[str, Any]:
    domain: Dict[str, Optional[float]] = {"radius": plan.domain_r_max}
    if plan.dim == "2d":
        domain["height"] = domain_height_m
    hob = hob_m
    if hob is None and plan.charge_center:
        hob = plan.charge_center[1]
    receive = remap_receive_r_max if remap_receive_r_max is not None else plan.remap_receive_r_max
    payload = build_fingerprint(
        dim=plan.dim,
        case_path=case_path,
        mass_kg=plan.mass_kg,
        domain_size=domain,
        hob_m=hob,
        charge_center=plan.charge_center,
        cell_size=cell_size,
        burst_mode=plan.burst_master,
        reference_mode=f"{SOURCE_UFC} Figure {plan.figure}".strip(),
        points=plan.points if include_coordinates else (),
        remap_receive_r_max=receive,
        source_model=source_model,
    )
    if not include_coordinates:
        payload.pop("coordinates", None)
        payload.pop("n_points", None)
    return payload


def stamp_plan(
    plan: SamplingPlan,
    *,
    case_path: Optional[str],
    cell_size: Optional[float] = None,
    hob_m: Optional[float] = None,
    domain_height_m: Optional[float] = None,
    remap_receive_r_max: Optional[float] = None,
    remap_timing: Optional[Dict[str, Any]] = None,
    source_model: Optional[str] = None,
) -> SamplingPlan:
    fingerprint = expected_plan_fingerprint(
        plan,
        case_path=case_path,
        cell_size=cell_size,
        hob_m=hob_m,
        domain_height_m=domain_height_m,
        remap_receive_r_max=remap_receive_r_max,
        include_coordinates=True,
        source_model=source_model,
    )
    return replace(
        attach_plan_fingerprint(plan, fingerprint),
        remap_timing=dict(remap_timing or plan.remap_timing or {}),
        remap_receive_r_max=(
            remap_receive_r_max if remap_receive_r_max is not None else plan.remap_receive_r_max
        ),
    )


def live_fingerprint(
    *,
    dim: str,
    case_path: Optional[str],
    mass_kg: Optional[float],
    domain_radius_m: Optional[float],
    domain_height_m: Optional[float] = None,
    hob_m: Optional[float] = None,
    charge_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cell_size: Optional[float] = None,
    burst_mode: str = "",
    figure: str = "",
    remap_receive_r_max: Optional[float] = None,
    source_model: Optional[str] = None,
) -> Dict[str, Any]:
    domain: Dict[str, Optional[float]] = {"radius": domain_radius_m}
    if str(dim).strip().lower() == "2d":
        domain["height"] = domain_height_m
    payload = build_fingerprint(
        dim=dim,
        case_path=case_path,
        mass_kg=mass_kg,
        domain_size=domain,
        hob_m=hob_m,
        charge_center=charge_center,
        cell_size=cell_size,
        burst_mode=burst_mode,
        reference_mode=f"{SOURCE_UFC} Figure {figure}".strip() if figure else SOURCE_UFC,
        points=(),
        remap_receive_r_max=remap_receive_r_max,
        source_model=source_model,
    )
    payload.pop("coordinates", None)
    payload.pop("n_points", None)
    return payload
