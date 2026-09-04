"""1D→2D and 2D→3D remap comparison at the same physical time."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from generator_2d import WEDGE_HALF_ANGLE_DEG
from output_options import REMAP_2D_FILENAME
from validation.metrics import (
    is_finite_number,
    max_absolute_error,
    max_meaningful_relative_error,
    mean_absolute_error,
    rms_error,
)
from validation.remap_timing import (
    RemapTiming,
    build_remap_timing,
    physical_times_synchronized,
    remap_timing_from_mapping,
)
from validation.spatial import (
    first_initialized_time,
    parse_remap2d_ggui,
    read_1d_profile,
    read_cell_centres,
    read_vol_field,
)

REMAP_FIELDS = ("p", "rho", "U", "T", "alpha.c4")


@dataclass(frozen=True)
class ProfileCompare:
    field: str
    r: Tuple[float, ...]
    source: Tuple[float, ...]
    target: Tuple[float, ...]
    abs_diff: Tuple[float, ...]
    rel_diff: Tuple[Optional[float], ...]
    source_time: Optional[float]
    target_time: Optional[float]
    delta_t: Optional[float]
    synchronized: bool
    interval: Tuple[float, float]
    rms: Optional[float]
    mae: Optional[float]
    max_abs: Optional[float]
    max_rel: Optional[float]
    shock_source: Optional[float]
    shock_target: Optional[float]
    peak_source: Optional[float]
    peak_target: Optional[float]
    peak_r_source: Optional[float]
    peak_r_target: Optional[float]
    source_physical_time: Optional[float]
    target_physical_time: Optional[float]
    physical_time_offset: Optional[float]
    message: str = ""


@dataclass(frozen=True)
class ConservationCompare:
    quantity: str
    source: Optional[float]
    target: Optional[float]
    difference: Optional[float]
    relative: Optional[float]
    geometry: str
    message: str = ""


def resolve_1d_to_2d(
    *,
    target_case: str,
    mapping_source: Optional[str],
    mapping_time: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Return (source_case, source_time, target_time, message) from metadata only."""
    if not target_case or not os.path.isdir(target_case):
        return None, None, None, "Current run does not contain the required validation data."
    source = str(mapping_source or "").strip()
    meta_path = os.path.join(target_case, "case_2d.json")
    if (not source) and os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as handle:
                payload = json.loads(handle.read())
        except (OSError, json.JSONDecodeError, TypeError):
            payload = {}
        mapping = payload.get("mapping") if isinstance(payload, dict) else None
        if isinstance(mapping, dict):
            source = str(mapping.get("case_path") or "").strip()
            if not mapping_time:
                mapping_time = str(mapping.get("specific_time") or mapping.get("time_mode") or "")
    remap_meta = os.path.join(target_case, REMAP_2D_FILENAME)
    if os.path.isfile(remap_meta) and not mapping_time:
        parsed = parse_remap2d_ggui(remap_meta)
        mapping_time = parsed.get("time") or mapping_time
    if not source or not os.path.isdir(source):
        return None, None, None, "Remap source is not recorded in case metadata."
    target_time = first_initialized_time(target_case)
    source_time = mapping_time if mapping_time and mapping_time not in ("latest",) else first_initialized_time(source)
    if mapping_time == "latest":
        from openfoam_times_2d import list_numeric_time_entries

        times = list_numeric_time_entries(source)
        source_time = times[-1][1] if times else source_time
    if not target_time:
        return source, source_time, None, "Target initialized time is not available."
    return source, source_time, target_time, ""


def resolve_2d_to_3d(
    *,
    target_case: str,
    remap_source_type: Optional[str],
    prepare_3d_transfer: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    if str(remap_source_type or "").strip().upper() == "1D":
        return None, None, None, (
            "Current 3D remap source is 1D radial (remap_radial.py), not a 2D→3D remap."
        )
    path = str(prepare_3d_transfer or "").strip()
    if not path:
        candidate = os.path.join(target_case or "", "prepare_3d_transfer.json")
        if os.path.isfile(candidate):
            path = candidate
        else:
            # Look on a sibling 2D case only if explicitly provided. Do not scan disk.
            return None, None, None, "2D→3D transfer metadata (prepare_3d_transfer.json) is not available."
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.loads(handle.read())
    except (OSError, json.JSONDecodeError, TypeError):
        return None, None, None, "2D→3D transfer metadata could not be read."
    source = str(payload.get("source_case") or "").strip()
    source_time = str(payload.get("source_time") or "").strip()
    if not source or not os.path.isdir(source):
        return None, None, None, "2D remap source path in transfer metadata does not exist."
    target_time = first_initialized_time(target_case)
    return source, source_time, target_time, ""


def _interpolate(src_r: np.ndarray, src_v: np.ndarray, query_r: np.ndarray) -> np.ndarray:
    order = np.argsort(src_r)
    r = src_r[order]
    v = src_v[order]
    if r.size < 2:
        return np.full(query_r.shape, np.nan)
    return np.interp(query_r, r, v, left=np.nan, right=np.nan)


def _shock_position(r: np.ndarray, values: np.ndarray) -> Optional[float]:
    if r.size < 3:
        return None
    g = np.gradient(values, r, edge_order=1)
    if not np.isfinite(g).any():
        return None
    return float(r[int(np.nanargmax(np.abs(g)))])


def compare_profiles(
    *,
    field: str,
    source_r: Sequence[float],
    source_v: Sequence[float],
    target_r: Sequence[float],
    target_v: Sequence[float],
    r_max: float,
    source_time: Optional[float],
    target_time: Optional[float],
    physical_time_offset: Optional[float] = None,
) -> ProfileCompare:
    r_src = np.asarray(source_r, dtype=float)
    v_src = np.asarray(source_v, dtype=float)
    r_tgt = np.asarray(target_r, dtype=float)
    v_tgt = np.asarray(target_v, dtype=float)
    mask = np.isfinite(r_tgt) & np.isfinite(v_tgt) & (r_tgt <= float(r_max) + 1e-12)
    r = r_tgt[mask]
    tgt = v_tgt[mask]
    src = _interpolate(r_src, v_src, r)
    abs_diff = tgt - src
    rel: list[Optional[float]] = []
    for a, b in zip(tgt, src):
        if is_finite_number(a) and is_finite_number(b) and abs(float(b)) > 1e-12:
            rel.append(float((a - b) / b))
        else:
            rel.append(None)
    dt = None
    src_phys = float(source_time) if is_finite_number(source_time) else None
    tgt_of = float(target_time) if is_finite_number(target_time) else None
    offset = float(physical_time_offset) if is_finite_number(physical_time_offset) else None
    tgt_phys = None
    if tgt_of is not None:
        tgt_phys = tgt_of + offset if offset is not None else tgt_of
    if src_phys is not None and tgt_phys is not None:
        dt = float(tgt_phys) - float(src_phys)
    sync = physical_times_synchronized(src_phys, tgt_of, offset=offset)
    src_list = [float(v) if is_finite_number(v) else float("nan") for v in src]
    tgt_list = [float(v) if is_finite_number(v) else float("nan") for v in tgt]
    interval = (float(r.min()) if r.size else 0.0, float(r.max()) if r.size else 0.0)
    peak_s = float(np.nanmax(v_src)) if v_src.size else None
    peak_t = float(np.nanmax(v_tgt)) if v_tgt.size else None
    peak_rs = float(r_src[int(np.nanargmax(v_src))]) if v_src.size and np.isfinite(v_src).any() else None
    peak_rt = float(r_tgt[int(np.nanargmax(v_tgt))]) if v_tgt.size and np.isfinite(v_tgt).any() else None
    msg = ""
    if not sync:
        msg = "Source and target physical times do not match; comparison is not synchronized."
    if r.size == 0:
        msg = "No overlapping samples inside the remap radius."
    return ProfileCompare(
        field=field,
        r=tuple(float(x) for x in r),
        source=tuple(src_list),
        target=tuple(tgt_list),
        abs_diff=tuple(float(x) if is_finite_number(x) else float("nan") for x in abs_diff),
        rel_diff=tuple(rel),
        source_time=float(source_time) if is_finite_number(source_time) else None,
        target_time=float(target_time) if is_finite_number(target_time) else None,
        delta_t=dt,
        synchronized=sync,
        interval=interval,
        rms=rms_error(tgt_list, src_list),
        mae=mean_absolute_error(tgt_list, src_list),
        max_abs=max_absolute_error(tgt_list, src_list),
        max_rel=max_meaningful_relative_error(tgt_list, src_list),
        shock_source=_shock_position(r_src, v_src),
        shock_target=_shock_position(r_tgt, v_tgt),
        peak_source=peak_s,
        peak_target=peak_t,
        peak_r_source=peak_rs,
        peak_r_target=peak_rt,
        source_physical_time=src_phys,
        target_physical_time=tgt_phys,
        physical_time_offset=offset,
        message=msg,
    )


def spherical_mass(r: Sequence[float], rho: Sequence[float]) -> Optional[float]:
    r_a = np.asarray(r, dtype=float)
    rho_a = np.asarray(rho, dtype=float)
    if r_a.size < 2:
        return None
    order = np.argsort(r_a)
    r_a = r_a[order]
    rho_a = rho_a[order]
    dr = np.diff(r_a)
    r_mid = 0.5 * (r_a[1:] + r_a[:-1])
    rho_mid = 0.5 * (rho_a[1:] + rho_a[:-1])
    return float(np.sum(rho_mid * 4.0 * math.pi * r_mid**2 * dr))


def wedge_mass(
    r: Sequence[float],
    z: Sequence[float],
    rho: Sequence[float],
    *,
    half_angle_deg: float = WEDGE_HALF_ANGLE_DEG,
    cell_volume: Optional[Sequence[float]] = None,
) -> Optional[float]:
    rho_a = np.asarray(rho, dtype=float)
    if cell_volume is not None:
        vol = np.asarray(cell_volume, dtype=float)
        n = min(rho_a.size, vol.size)
        if n == 0:
            return None
        return float(np.sum(rho_a[:n] * vol[:n]))
    r_a = np.asarray(r, dtype=float)
    z_a = np.asarray(z, dtype=float)
    n = min(r_a.size, z_a.size, rho_a.size)
    if n < 2:
        return None
    theta = 2.0 * math.radians(float(half_angle_deg))
    # Approximate dV = theta * r * dr * dz using unique spacings.
    dr = max(float(np.median(np.diff(np.unique(np.round(r_a[:n], 9))))), 1e-12)
    dz = max(float(np.median(np.diff(np.unique(np.round(z_a[:n], 9))))), 1e-12)
    return float(np.sum(rho_a[:n] * theta * np.maximum(r_a[:n], 0.0) * dr * dz))


def conservation_1d_2d(
    *,
    r_1d: Sequence[float],
    rho_1d: Sequence[float],
    alpha_1d: Optional[Sequence[float]],
    r_2d: Sequence[float],
    z_2d: Sequence[float],
    rho_2d: Sequence[float],
    alpha_2d: Optional[Sequence[float]],
    cell_volume_2d: Optional[Sequence[float]] = None,
    r_max: float,
) -> Tuple[ConservationCompare, ...]:
    mask1 = np.asarray(r_1d, dtype=float) <= float(r_max) + 1e-12
    r1 = np.asarray(r_1d, dtype=float)[mask1]
    rho1 = np.asarray(rho_1d, dtype=float)[mask1]
    m1 = spherical_mass(r1, rho1)
    m2_wedge = wedge_mass(r_2d, z_2d, rho_2d, cell_volume=cell_volume_2d)
    # Scale wedge to full sphere: full / wedge = 4π / theta.
    theta = 2.0 * math.radians(WEDGE_HALF_ANGLE_DEG)
    scale = 4.0 * math.pi / theta if theta > 0.0 else None
    m2 = m2_wedge * scale if m2_wedge is not None and scale is not None else None
    geom = "1D: dV=4*pi*r^2*dr ; 2D wedge scaled to 4*pi by 4*pi/theta_wedge"
    items = [
        ConservationCompare(
            quantity="total_mass",
            source=m1,
            target=m2,
            difference=(m2 - m1) if m1 is not None and m2 is not None else None,
            relative=((m2 - m1) / m1) if m1 not in (None, 0.0) and m2 is not None else None,
            geometry=geom,
        )
    ]
    if alpha_1d is not None and alpha_2d is not None:
        a1 = np.asarray(alpha_1d, dtype=float)[mask1] if len(alpha_1d) == len(r_1d) else np.asarray(alpha_1d, dtype=float)
        expl1 = spherical_mass(r1, rho1 * a1[: r1.size])
        expl2_w = wedge_mass(r_2d, z_2d, np.asarray(rho_2d) * np.asarray(alpha_2d), cell_volume=cell_volume_2d)
        expl2 = expl2_w * scale if expl2_w is not None and scale is not None else None
        items.append(
            ConservationCompare(
                quantity="explosive_mass",
                source=expl1,
                target=expl2,
                difference=(expl2 - expl1) if expl1 is not None and expl2 is not None else None,
                relative=((expl2 - expl1) / expl1) if expl1 not in (None, 0.0) and expl2 is not None else None,
                geometry=geom,
            )
        )
    return tuple(items)


def load_line_from_case(
    case_dir: str, time_label: str, field: str, *, dim: str = "1d"
) -> Tuple[np.ndarray, np.ndarray]:
    geometry = "spherical_1d" if str(dim).strip().lower() == "1d" else "axisymmetric_2d"
    if field == "U":
        return read_1d_profile(case_dir, time_label, "U", geometry=geometry)
    return read_1d_profile(case_dir, time_label, field, geometry=geometry)
