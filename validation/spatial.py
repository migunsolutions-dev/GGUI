"""Lazy OpenFOAM spatial sampling for HOB and remap. No GUI imports."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from openfoam_times_2d import list_numeric_time_entries

_INTERNAL = re.compile(
    r"internalField\s+nonuniform\s+List<(?:scalar|vector)>\s+(\d+)\s*\((.*?)\)\s*;",
    re.DOTALL,
)
_UNIFORM = re.compile(r"internalField\s+uniform\s+([^;]+);")
_REMAP_TIME = re.compile(r"^\s*time\s+([^;]+);", re.MULTILINE)
_REMAP_SOURCE = re.compile(r'^\s*sourceCase\s+"([^"]+)";', re.MULTILINE)


def list_saved_times(case_dir: str) -> List[Tuple[float, str]]:
    return list_numeric_time_entries(case_dir or "")


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def parse_internal_field(text: str) -> Optional[np.ndarray]:
    match = _UNIFORM.search(text)
    if match:
        parts = match.group(1).replace("(", " ").replace(")", " ").split()
        try:
            values = [float(p) for p in parts]
        except ValueError:
            return None
        return np.array(values, dtype=float)
    match = _INTERNAL.search(text)
    if not match:
        return None
    body = match.group(2)
    numbers = []
    for tok in body.replace("(", " ").replace(")", " ").split():
        try:
            numbers.append(float(tok))
        except ValueError:
            continue
    count = int(match.group(1))
    if not numbers:
        return None
    arr = np.array(numbers, dtype=float)
    if arr.size == count:
        return arr
    if count > 0 and arr.size % count == 0:
        return arr.reshape((count, arr.size // count))
    return arr


def read_vol_field(case_dir: str, time_label: str, name: str) -> Optional[np.ndarray]:
    path = os.path.join(case_dir, time_label, name)
    if not os.path.isfile(path):
        return None
    return parse_internal_field(read_text(path))


def read_cell_centres(case_dir: str, time_label: str) -> Optional[np.ndarray]:
    for candidate in (os.path.join(case_dir, time_label, "C"), os.path.join(case_dir, "constant", "C")):
        if os.path.isfile(candidate):
            data = parse_internal_field(read_text(candidate))
            if data is not None and data.ndim == 2 and data.shape[1] >= 2:
                return data
    return None


def axisymmetric_rz(
    centres: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wedge convention: x ~ radius, y ~ height."""
    r = np.asarray(centres[:, 0], dtype=float)
    z = np.asarray(centres[:, 1], dtype=float)
    return r, z


def load_pressure_rz(
    case_dir: str,
    time_label: str,
    field: str = "p",
    *,
    plane: str = "axisymmetric",
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    values = read_vol_field(case_dir, time_label, field)
    centres = read_cell_centres(case_dir, time_label)
    if values is None:
        return None, None, None, f"Field {field} is not available at time {time_label}."
    if centres is None:
        return None, None, None, "Cell centres (C) are not available; cannot place the spatial field."
    n = min(len(values) if values.ndim == 1 else values.shape[0], centres.shape[0])
    if values.ndim > 1:
        mag = np.linalg.norm(values[:n], axis=1)
        p = mag
    else:
        p = np.asarray(values[:n], dtype=float)
    pts = centres[:n]
    kind = str(plane or "axisymmetric").strip().upper()
    if kind in ("", "AXISYMMETRIC", "2D"):
        r, z = axisymmetric_rz(pts)
        return r, z, p, ""
    if pts.shape[1] < 3:
        return None, None, None, "3D cell centres are required for an X-Z or Y-Z section."
    ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))
    span = float(np.max(np.max(pts, axis=0) - np.min(pts, axis=0))) if pts.size else 1.0
    thickness = max(1e-6, 0.02 * max(span, 1e-6))
    if kind in ("X-Z", "XZ"):
        dist = np.abs(pts[:, 1] - oy)
        r = pts[:, 0]
        z = pts[:, 2]
    elif kind in ("Y-Z", "YZ"):
        dist = np.abs(pts[:, 0] - ox)
        r = pts[:, 1]
        z = pts[:, 2]
    else:
        return None, None, None, "Only X-Z and Y-Z sections are supported."
    mask = dist <= thickness
    if not np.any(mask):
        return None, None, None, f"No cells lie on the {kind} section through the charge centre."
    return r[mask], z[mask], p[mask], ""


def parse_remap2d_ggui(path: str) -> Dict[str, str]:
    text = read_text(path)
    out: Dict[str, str] = {}
    match = _REMAP_TIME.search(text)
    if match:
        out["time"] = match.group(1).strip()
    match = _REMAP_SOURCE.search(text)
    if match:
        out["sourceCase"] = match.group(1).strip()
    return out


def radial_distance_xyz(x: float, y: float = 0.0, z: float = 0.0) -> float:
    return float(np.sqrt(float(x) ** 2 + float(y) ** 2 + float(z) ** 2))


def spherical_radii(centres: np.ndarray) -> np.ndarray:
    """True radial distance sqrt(x^2+y^2+z^2) for a spherical/wedge 1D mesh."""
    pts = np.asarray(centres, dtype=float)
    if pts.ndim == 1:
        return np.abs(pts)
    n = pts.shape[1]
    if n >= 3:
        return np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2)
    if n == 2:
        return np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    return np.abs(pts[:, 0])


def physical_radius_from_centre(
    centres: np.ndarray,
    charge_centre: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Euclidean R from the charge centre. Not cylindrical x and not storage order."""
    pts = np.asarray(centres, dtype=float)
    cx, cy, cz = (float(charge_centre[0]), float(charge_centre[1]), float(charge_centre[2]))
    if pts.ndim == 1:
        return np.abs(pts - cx)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy if pts.shape[1] > 1 else 0.0
    dz = pts[:, 2] - cz if pts.shape[1] > 2 else 0.0
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def read_1d_profile(
    case_dir: str, time_label: str, field: str, *, geometry: str = "spherical_1d"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (r, values). 1D wedge/spherical uses sqrt(x^2+y^2+z^2), not x alone."""
    values = read_vol_field(case_dir, time_label, field)
    centres = read_cell_centres(case_dir, time_label)
    if values is None:
        return np.array([]), np.array([])
    if values.ndim > 1:
        if field.upper() == "U" or values.shape[1] >= 3:
            # Radial velocity in 1D is the x-component.
            values = values[:, 0]
        else:
            values = np.linalg.norm(values, axis=1)
    if centres is None:
        r = np.arange(values.size, dtype=float) + 0.5
        return r, np.asarray(values, dtype=float)
    pts = np.asarray(centres[: values.size], dtype=float)
    kind = str(geometry or "spherical_1d").strip().lower()
    if kind in ("axisymmetric_2d", "cylindrical", "2d"):
        r = np.abs(pts[:, 0]) if pts.ndim == 2 else np.abs(pts)
    else:
        r = spherical_radii(pts)
    return r, np.asarray(values[: r.size], dtype=float)


def first_initialized_time(case_dir: str) -> Optional[str]:
    times = list_saved_times(case_dir)
    if not times:
        return None
    for value, label in times:
        if value == 0.0 or label == "0":
            return label
    return times[0][1]
