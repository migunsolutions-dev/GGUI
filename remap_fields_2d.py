"""1D spherical profile → 2D r-z remap about the target charge centre.

The 2D computational domain is height above ground, z >= 0. Each target cell
samples the 1D radial profile at

    r_source = hypot(r_2d, z_2d - HOB)

so the remapped field is centred at ``[0, HOB, 0]``. The portion of the sphere
that would fall below the ground is omitted: those cells are simply not in the
mesh. This module never mirrors or duplicates the charge through the ground
plane.

This file is copied into generated 2D remap cases and must stay Qt-free and
importable from the case directory (numpy + stdlib only).
"""
from __future__ import annotations

import io
import os
import re
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from remap_native_io import (
    _parse_internal_field, _read_3d_cell_centres as _read_cell_centres,
    _read_0_orig_internal, _read_bc, verified_remap_source,
)

MAPPING_METHOD = "radial_from_target_charge_center"
GROUND_CLIP = "domain_z_ge_0_no_mirror"
RADIUS_POLICY = "user_mapped_radius"

_BELOW_GROUND = -1.0e-15
_R_FLOOR = 1.0e-20


def charge_center_xyz(hob_m: float) -> Tuple[float, float, float]:
    return (0.0, float(hob_m), 0.0)


def source_radius_rz(r_2d, z_2d, hob_m: float):
    """Spherical radius from the target 2D charge centre ``[0, HOB, 0]``."""
    r = np.asarray(r_2d, dtype=float)
    z = np.asarray(z_2d, dtype=float)
    return np.hypot(r, z - float(hob_m))


def effective_mapped_radius(r_1d, mapped_radius: float = 0.0) -> float:
    """Physical receiving-region radius: the user remap limit.

    Source-mesh padding beyond that limit is numerical only and is not copied.
    If the user radius is unset, fall back to the 1D profile extent.
    """
    r_1d = np.asarray(r_1d, dtype=float)
    r_max = float(np.max(r_1d)) if r_1d.size else 0.0
    try:
        user = float(mapped_radius or 0.0)
    except (TypeError, ValueError):
        user = 0.0
    if not np.isfinite(user) or user <= 0.0:
        return max(r_max, 0.0)
    if r_max <= 0.0:
        return user
    return min(user, r_max)


def interpolate_radial(r_source, r_1d, values_1d):
    r_src = np.asarray(r_source, dtype=float)
    r_tab = np.asarray(r_1d, dtype=float)
    vals = np.asarray(values_1d, dtype=float)
    order = np.argsort(r_tab, kind="mergesort")
    return np.interp(r_src, r_tab[order], vals[order])


def map_scalar_profile(
    r_2d,
    z_2d,
    hob_m: float,
    r_1d,
    values_1d,
    *,
    mapped_radius: float = 0.0,
    ambient=0.0,
):
    """Interpolate a 1D radial scalar onto 2D cell centres.

    Cells with ``z < 0`` (should not exist in the wedge) stay at *ambient*.
    Cells outside the 1D extent stay at *ambient*. There is no image charge.
    """
    r_2d = np.asarray(r_2d, dtype=float)
    z_2d = np.asarray(z_2d, dtype=float)
    r_src = source_radius_rz(r_2d, z_2d, hob_m)
    r_lim = effective_mapped_radius(r_1d, mapped_radius)
    mapped = interpolate_radial(r_src, r_1d, values_1d)
    ambient_arr = np.broadcast_to(np.asarray(ambient, dtype=float), r_src.shape).copy()
    outside = (r_src > r_lim) | (z_2d < _BELOW_GROUND)
    return np.where(outside, ambient_arr, mapped)


def map_radial_velocity(
    r_2d,
    z_2d,
    hob_m: float,
    r_1d,
    u_mag_1d,
    *,
    mapped_radius: float = 0.0,
    ambient_u: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reconstruct Cartesian velocity from 1D radial speed about ``[0, HOB, 0]``."""
    r_2d = np.asarray(r_2d, dtype=float).reshape(-1)
    z_2d = np.asarray(z_2d, dtype=float).reshape(-1)
    n = r_2d.size
    r_src = source_radius_rz(r_2d, z_2d, hob_m)
    r_safe = np.maximum(r_src, _R_FLOOR)
    u_mag = interpolate_radial(r_src, r_1d, u_mag_1d)
    ux = u_mag * r_2d / r_safe
    uy = u_mag * (z_2d - float(hob_m)) / r_safe
    uz = np.zeros(n, dtype=float)
    mapped = np.column_stack((ux, uy, uz))
    if ambient_u is None:
        ambient_u = np.zeros((n, 3), dtype=float)
    else:
        ambient_u = np.asarray(ambient_u, dtype=float)
        if ambient_u.ndim == 1:
            ambient_u = np.broadcast_to(ambient_u.reshape(1, 3), (n, 3)).copy()
    r_lim = effective_mapped_radius(r_1d, mapped_radius)
    outside = (r_src > r_lim) | (z_2d < _BELOW_GROUND)
    return np.where(outside[:, np.newaxis], ambient_u, mapped)


def map_fields_to_2d_cells(
    r_2d,
    z_2d,
    hob_m: float,
    r_1d,
    profiles: Mapping[str, Sequence[float]],
    *,
    mapped_radius: float = 0.0,
    ambient: Optional[Mapping[str, object]] = None,
) -> Dict[str, np.ndarray]:
    """Map named 1D profiles onto 2D cells. ``U`` is reconstructed from ``U_mag``."""
    ambient = dict(ambient or {})
    out: Dict[str, np.ndarray] = {}
    for name, values in profiles.items():
        if name in ("U_mag", "U", "r"):
            continue
        out[name] = map_scalar_profile(
            r_2d,
            z_2d,
            hob_m,
            r_1d,
            values,
            mapped_radius=mapped_radius,
            ambient=ambient.get(name, 0.0),
        )
    if "U_mag" in profiles:
        out["U"] = map_radial_velocity(
            r_2d,
            z_2d,
            hob_m,
            r_1d,
            profiles["U_mag"],
            mapped_radius=mapped_radius,
            ambient_u=ambient.get("U"),
        )
    return out


def carry_mixture_mass_in_air(
    mapped: Dict[str, np.ndarray],
    *,
    unused_rho_c4: float,
) -> Dict[str, np.ndarray]:
    """Drop the HE phase after remap without creating vacuum cells.

    1D product cells have ``alpha.c4 ~ 1`` and ``rho.air ~ 0``. The remapped
    2D case uses ``activationModel none`` and carries the blast in air, so
    those cells must keep the 1D *mixture* density in ``rho.air``. Zeroing
    ``alpha.c4`` / ``rho.c4`` without this step leaves ``rho_mix ~ 0`` and
    ``compressibleBlastSystem::decode`` divides by density on the first step.
    Unused HE-phase density stays at the generated ambient value (0.orig).
    """
    if "rho.air" not in mapped or "rho.c4" not in mapped or "alpha.c4" not in mapped:
        return mapped
    alpha = np.asarray(mapped["alpha.c4"], dtype=float)
    rho_c4 = np.asarray(mapped["rho.c4"], dtype=float)
    rho_air = np.asarray(mapped["rho.air"], dtype=float)
    mapped["rho.air"] = alpha * rho_c4 + (1.0 - alpha) * rho_air
    mapped["alpha.c4"] = np.zeros_like(alpha)
    mapped["rho.c4"] = np.full_like(alpha, float(unused_rho_c4))
    return mapped


def remap_region_metadata(
    hob_m: float,
    *,
    mapped_radius: float = 0.0,
    source_time: str = "",
    time_mode: str = "",
    target_time: str = "0",
) -> Dict[str, object]:
    center = list(charge_center_xyz(hob_m))
    return {
        "center": center,
        "radius_policy": RADIUS_POLICY,
        "requested_mapped_radius_m": float(mapped_radius or 0.0),
        "ground_clip": GROUND_CLIP,
        "mapping_method": MAPPING_METHOD,
        "source_time": str(source_time or ""),
        "time_mode": str(time_mode or ""),
        "target_time": str(target_time or "0"),
    }


# ---------------------------------------------------------------------------
# OpenFOAM case I/O — used when this module is copied into a generated case.
# ---------------------------------------------------------------------------

_SKIP_TIME = {"constant", "system", "0.orig", "postProcessing"}


def _find_latest_time(case_path: str) -> Optional[str]:
    best = None
    try:
        for name in os.listdir(case_path):
            if name in _SKIP_TIME or name.startswith("processor"):
                continue
            path = os.path.join(case_path, name)
            if not os.path.isdir(path):
                continue
            try:
                t = float(name)
            except ValueError:
                continue
            if best is None or t > best[0]:
                best = (t, name)
    except OSError:
        return None
    return best[1] if best else None


def _available_times(case_path: str):
    times = []
    try:
        for name in os.listdir(case_path):
            if name in _SKIP_TIME or name.startswith("processor"):
                continue
            path = os.path.join(case_path, name)
            if not os.path.isdir(path):
                continue
            try:
                times.append((float(name), name))
            except ValueError:
                continue
    except OSError:
        return []
    times.sort(key=lambda item: item[0])
    return times


def _resolve_time_dir(case_path: str, requested_time: str) -> Optional[str]:
    raw = str(requested_time or "").strip().strip("'\"")
    if raw.lower() in ("latest", "latesttime", ""):
        return _find_latest_time(case_path)
    available = _available_times(case_path)
    if not available:
        return None
    for _, name in available:
        if name == raw:
            return name
    try:
        requested = float(raw)
    except ValueError:
        return None
    for value, name in available:
        if value == requested:
            return name
    return None




def _parse_cell_centres_file(path: str) -> Optional[np.ndarray]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    start = text.find("internalField")
    if start < 0:
        return None
    end = text.find("boundaryField", start)
    block = text[start:end if end >= 0 else len(text)]
    pts = [
        (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        for m in re.finditer(
            r"\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)",
            block,
        )
    ]
    return np.array(pts, dtype=float) if pts else None


def _read_1d_data(source_case, time_dir):
    from remap_snapshot_1d import capture_arrays_from_time_dir
    return capture_arrays_from_time_dir(source_case, time_dir)








def _fast_write(path: str, name: str, dim: str, arr, bc: str, is_vector: bool = False) -> None:
    n = len(arr)
    cls = "volVectorField" if is_vector else "volScalarField"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "FoamFile\n{ version 2.0; format ascii; class "
            + cls
            + "; object "
            + name
            + "; }\n\n"
        )
        if is_vector:
            handle.write("dimensions [0 1 -1 0 0 0 0];\n\ninternalField nonuniform List<vector>\n")
            handle.write("%d\n(\n" % n)
            buf = io.StringIO()
            np.savetxt(buf, np.asarray(arr).reshape(-1, 3), fmt=" (%.10e %.10e %.10e)")
            handle.write(buf.getvalue())
        else:
            handle.write("dimensions " + dim + ";\n\ninternalField nonuniform List<scalar>\n")
            handle.write("%d\n(\n" % n)
            buf = io.StringIO()
            np.savetxt(buf, np.asarray(arr).reshape(-1, 1), fmt=" %.10e")
            handle.write(buf.getvalue())
        handle.write(");\n\n")
        handle.write(bc if bc.endswith("\n") else bc + "\n")


def _profile_is_ideal_gas(resolved) -> bool:
    """True only for a positively identified Ideal-Gas profile."""
    from models import SOURCE_MODEL_IG, normalize_source_model

    declared = str((resolved.metadata or {}).get("source_model") or "")
    if declared:
        return normalize_source_model(declared) == SOURCE_MODEL_IG
    profile = resolved.profile or {}
    return (
        "rho" in profile
        and "alpha.c4" not in profile
        and "rho.air" not in profile
        and "rho.c4" not in profile
    )


def _case_gas_constant() -> float:
    """R = (gamma-1)*Cv from the case phaseProperties, matching eConst idealGas."""
    path = os.path.join("constant", "phaseProperties")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    gamma = float(re.search(r"\bgamma\s+([0-9.eE+-]+)\s*;", text).group(1))
    cv = float(re.search(r"\bCv\s+([0-9.eE+-]+)\s*;", text).group(1))
    if gamma <= 1.0 or cv <= 0.0:
        raise ValueError("Ideal-Gas thermo constants are not usable")
    return (gamma - 1.0) * cv


def _write_ideal_gas_state(
    *,
    data_1d,
    r_2d,
    z_2d,
    hob,
    mapped_radius,
    zero_dir,
    out_dir,
    n_cells,
    r_lim,
    r_src,
) -> int:
    """Plant single-phase p, rho, T and U. T is rebuilt from p and rho.

    blastFoam's eConst ideal gas sets ``e = p/((gamma-1)*rho)`` and ``T = e/Cv``.
    Rebuilding T from the planted p and rho keeps that relation exact and does
    not add a second energy source.
    """
    p_orig = _read_0_orig_internal(zero_dir, "p", n_cells)
    t_orig = _read_0_orig_internal(zero_dir, "T", n_cells)
    rho_orig = _read_0_orig_internal(zero_dir, "rho", n_cells)
    u_orig = _read_0_orig_internal(zero_dir, "U", n_cells, is_vector=True)
    mapped = map_fields_to_2d_cells(
        r_2d,
        z_2d,
        hob,
        data_1d["r"],
        data_1d,
        mapped_radius=mapped_radius,
        ambient={"p": p_orig, "T": t_orig, "rho": rho_orig, "U": u_orig},
    )
    rho = np.asarray(mapped["rho"], dtype=float)
    pressure = np.asarray(mapped["p"], dtype=float)
    if (
        not np.all(np.isfinite(rho))
        or not np.all(np.isfinite(pressure))
        or np.any(rho <= 0.0)
    ):
        print("remap_2d: Ideal-Gas density or pressure is not a finite positive state", file=sys.stderr)
        return 1
    mapped["T"] = pressure / (rho * _case_gas_constant())
    os.makedirs(out_dir, exist_ok=True)
    for name, dim in (
        ("p", "[1 -1 -2 0 0 0 0]"),
        ("T", "[0 0 0 1 0 0 0]"),
        ("rho", "[1 -3 0 0 0 0 0]"),
    ):
        _fast_write(
            os.path.join(out_dir, name),
            name,
            dim,
            mapped[name],
            _read_bc(os.path.join(zero_dir, name)),
            is_vector=False,
        )
    _fast_write(
        os.path.join(out_dir, "U"),
        "U",
        "",
        mapped["U"],
        _read_bc(os.path.join(zero_dir, "U")),
        is_vector=True,
    )
    inside = int(np.sum(r_src <= r_lim))
    print(
        "remap_2d: wrote %s/p,T,rho,U (inside r<=%.6g: %d/%d)"
        % (out_dir, r_lim, inside, n_cells)
    )
    return 0


def run_case_remap(
    *,
    source_case: str,
    source_time: str,
    hob: float,
    mapped_radius: float = 0.0,
    out_dir: str = "0",
) -> int:
    """Run the 1D→2D remap from the generated 2D case root. Returns a process code."""
    try:
        resolved = verified_remap_source(source_case, source_time)
    except ValueError as exc:
        print("remap_2d: %s" % exc, file=sys.stderr)
        return 1
    data_1d = resolved.profile
    centres = _read_cell_centres()
    if centres is None or len(centres) == 0:
        print("remap_2d: run postProcess -func writeCellCentres first", file=sys.stderr)
        return 1
    origin = np.array(charge_center_xyz(hob), dtype=float)
    r_vec = centres - origin
    r_src = np.linalg.norm(r_vec, axis=1)
    r_2d = centres[:, 0]
    z_2d = centres[:, 1]
    n_cells = len(r_src)
    zero_dir = "0.orig" if os.path.isdir("0.orig") else "0"
    r_lim = effective_mapped_radius(data_1d["r"], mapped_radius)
    if _profile_is_ideal_gas(resolved):
        return _write_ideal_gas_state(
            data_1d=data_1d,
            r_2d=r_2d,
            z_2d=z_2d,
            hob=hob,
            mapped_radius=mapped_radius,
            zero_dir=zero_dir,
            out_dir=out_dir,
            n_cells=n_cells,
            r_lim=r_lim,
            r_src=r_src,
        )
    print(
        "remap_2d: HOB=%.6g origin=%s R_remap=%.6g cells=%d"
        % (float(hob), origin.tolist(), r_lim, n_cells),
        file=sys.stderr,
    )
    p_orig = _read_0_orig_internal(zero_dir, "p", n_cells)
    t_orig = _read_0_orig_internal(zero_dir, "T", n_cells)
    rho4_orig = _read_0_orig_internal(zero_dir, "rho.c4", n_cells)
    rhoa_orig = _read_0_orig_internal(zero_dir, "rho.air", n_cells)
    a4_orig = _read_0_orig_internal(zero_dir, "alpha.c4", n_cells)
    u_orig = _read_0_orig_internal(zero_dir, "U", n_cells, is_vector=True)
    ambient = {
        "p": p_orig,
        "T": t_orig,
        "rho.c4": rho4_orig,
        "rho.air": rhoa_orig,
        "alpha.c4": a4_orig,
        "U": u_orig,
    }
    mapped = map_fields_to_2d_cells(
        r_2d,
        z_2d,
        hob,
        data_1d["r"],
        data_1d,
        mapped_radius=mapped_radius,
        ambient=ambient,
    )
    unused_rho_c4 = 1600.0
    raw_c4 = ambient.get("rho.c4")
    if raw_c4 is not None:
        arr_c4 = np.asarray(raw_c4, dtype=float)
        if arr_c4.size:
            candidate = float(arr_c4.reshape(-1)[0])
            if candidate > 0.0:
                unused_rho_c4 = candidate
    carry_mixture_mass_in_air(mapped, unused_rho_c4=unused_rho_c4)
    os.makedirs(out_dir, exist_ok=True)
    scalars = (
        ("p", "[1 -1 -2 0 0 0 0]"),
        ("T", "[0 0 0 1 0 0 0]"),
        ("rho.c4", "[1 -3 0 0 0 0 0]"),
        ("rho.air", "[1 -3 0 0 0 0 0]"),
        ("alpha.c4", "[0 0 0 0 0 0 0]"),
    )
    for name, dim in scalars:
        src = os.path.join(zero_dir, name)
        _fast_write(
            os.path.join(out_dir, name),
            name,
            dim,
            mapped[name],
            _read_bc(src),
            is_vector=False,
        )
    _fast_write(
        os.path.join(out_dir, "U"),
        "U",
        "",
        mapped["U"],
        _read_bc(os.path.join(zero_dir, "U")),
        is_vector=True,
    )
    inside = int(np.sum(r_src <= r_lim))
    print(
        "remap_2d: wrote %s/p,T,U,rho.c4,rho.air,alpha.c4 (inside r<=%.6g: %d/%d)"
        % (out_dir, r_lim, inside, n_cells)
    )
    return 0
