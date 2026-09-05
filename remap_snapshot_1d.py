"""Dedicated 1D remap snapshot: compact npz + versioned JSON metadata.

Written from the final 1D solver state so 2D/3D remap does not require a
full OpenFOAM time directory at the stop time. Qt-free (numpy + stdlib) so
the module can be copied into generated 2D/3D cases.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1
SNAPSHOT_NPZ = "ggui_remap_snapshot_1d.npz"
SNAPSHOT_JSON = "ggui_remap_snapshot_1d.json"
SOURCE_SNAPSHOT = "snapshot"
SOURCE_OPENFOAM = "openfoam_time_directory"
SOURCE_DIMENSION = "1D"
COORDINATE_CONVENTION = "spherical_radius_m"
UNITS = {
    "r": "m",
    "p": "Pa",
    "T": "K",
    "U_mag": "m/s",
    "rho.air": "kg/m3",
    "rho.c4": "kg/m3",
    "alpha.c4": "1",
}
REQUIRED_ARRAYS = ("r", "p", "T", "U_mag")
OPTIONAL_ARRAYS = ("rho.air", "rho.c4", "alpha.c4")
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION,)

_SKIP_TIME = frozenset({"constant", "system", "0.orig", "postProcessing"})
_TIME_MATCH_REL = 1.0e-5
_TIME_MATCH_ABS = 1.0e-12


@dataclass
class RemapSourceResolution:
    ok: bool
    blocked: bool
    source_type: str = ""
    time_label: str = ""
    physical_time: Optional[float] = None
    message: str = ""
    field_names: Tuple[str, ...] = ()
    profile: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RemapAvailability:
    solver_completed: bool = False
    snapshot_available: bool = False
    snapshot_invalid: bool = False
    openfoam_fallback_available: bool = False
    status: str = "missing"
    message: str = ""
    physical_time: Optional[float] = None
    source_type: str = ""


def snapshot_npz_path(case_dir: str) -> str:
    return os.path.join(case_dir or "", SNAPSHOT_NPZ)


def snapshot_json_path(case_dir: str) -> str:
    return os.path.join(case_dir or "", SNAPSHOT_JSON)


def snapshot_exists(case_dir: str) -> bool:
    return os.path.isfile(snapshot_npz_path(case_dir)) and os.path.isfile(
        snapshot_json_path(case_dir)
    )


def invalidate_snapshot(case_dir: str) -> None:
    for path in (snapshot_npz_path(case_dir), snapshot_json_path(case_dir)):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


def canonical_case_path(case_dir: str) -> str:
    """Stable case identity across Windows, WSL UNC, and Linux paths.

    ``\\\\wsl.localhost\\Ubuntu-20.04\\home\\naor\\...`` and
    ``/home/naor/...`` must compare as the same source case. Windows
    ``normcase`` lowercases UNC paths, so equality is case-insensitive.
    """
    raw = (case_dir or "").strip()
    if not raw:
        return ""
    if raw.startswith("/") and not raw.startswith("//"):
        linux = raw
    elif raw.startswith("\\\\") or raw.startswith("//"):
        parts = [part for part in raw.replace("/", "\\").split("\\") if part]
        if len(parts) >= 3 and parts[0].lower() in ("wsl.localhost", "wsl$"):
            linux = "/" + "/".join(parts[2:])
        else:
            linux = "/" + "/".join(parts)
    elif len(raw) >= 2 and raw[1] == ":":
        rest = raw[2:].replace("\\", "/").lstrip("/")
        linux = f"/mnt/{raw[0].lower()}/{rest}" if rest else f"/mnt/{raw[0].lower()}"
    else:
        linux = raw.replace("\\", "/")
    linux = linux.replace("\\", "/")
    while "//" in linux:
        linux = linux.replace("//", "/")
    if len(linux) > 1:
        linux = linux.rstrip("/")
    return linux


def _norm_case_path(case_dir: str) -> str:
    return canonical_case_path(case_dir).lower()


def same_source_case(left: str, right: str) -> bool:
    a = canonical_case_path(left)
    b = canonical_case_path(right)
    if not a or not b:
        return False
    return a.lower() == b.lower()


def _sha256_text(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8", "replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def arrays_checksum(arrays: Dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(arrays[name], dtype=np.float64).tobytes())
    return digest.hexdigest()


def identity_fingerprint(
    case_dir: str,
    *,
    physical_time: Optional[float] = None,
    stop_reason: str = "",
    mode: str = "",
    wave_radius_reached: bool = False,
) -> str:
    return _sha256_text(
        _norm_case_path(case_dir),
        None if physical_time is None else round(float(physical_time), 12),
        str(stop_reason or ""),
        str(mode or ""),
        bool(wave_radius_reached),
    )


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Snapshot field {name!r} is empty.")
    return arr


def _completion_info(case_dir: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "mode": "",
        "stop_reason": "",
        "wave_radius_reached": False,
        "requested_stop_radius_m": None,
        "detected_arrival_time_s": None,
        "arrival_criterion": "",
        "final_solver_time_s": None,
            "end_time_s": None,
            "remap_for_2d": False,
            "remap_radius_m": None,
            "dr_1d_m": None,
            "remap_front_buffer_cells": None,
            "handoff_radius_m": None,
        }
    try:
        from completion_1d import read_completion_record
    except Exception:
        return info
    record = read_completion_record(case_dir)
    if record is None:
        return info
    info.update(
        {
            "mode": str(record.mode or ""),
            "stop_reason": str(record.stop_reason or ""),
            "wave_radius_reached": bool(record.wave_radius_reached),
            "requested_stop_radius_m": record.requested_stop_radius_m,
            "detected_arrival_time_s": record.detected_arrival_time_s,
            "arrival_criterion": str(record.criterion or ""),
            "final_solver_time_s": record.final_solver_time_s,
            "end_time_s": record.end_time_s,
            "remap_for_2d": bool(record.remap_for_2d),
            "remap_radius_m": record.remap_radius_m,
            "dr_1d_m": record.dr_1d_m,
            "remap_front_buffer_cells": record.remap_front_buffer_cells,
            "handoff_radius_m": record.handoff_radius_m,
        }
    )
    return info


def write_snapshot(
    case_dir: str,
    arrays: Dict[str, Sequence[float]],
    *,
    physical_time: Optional[float] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write npz + json. *arrays* must include r, p, T, U_mag."""
    packed = {name: _as_1d(values, name) for name, values in arrays.items()}
    n = packed["r"].size
    for name in REQUIRED_ARRAYS:
        if name not in packed:
            raise ValueError(f"Snapshot is missing required field {name!r}.")
        if packed[name].size != n:
            raise ValueError(
                f"Snapshot field {name!r} length {packed[name].size} does not match r ({n})."
            )
    for name in OPTIONAL_ARRAYS:
        if name not in packed:
            packed[name] = np.zeros(n, dtype=np.float64)
        elif packed[name].size != n:
            raise ValueError(
                f"Snapshot field {name!r} length {packed[name].size} does not match r ({n})."
            )
    completion = _completion_info(case_dir)
    phys = physical_time
    if phys is None:
        phys = completion.get("final_solver_time_s")
    shapes = {name: [int(packed[name].size)] for name in list(REQUIRED_ARRAYS) + list(OPTIONAL_ARRAYS)}
    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_dimension": SOURCE_DIMENSION,
        "source_case_path": canonical_case_path(case_dir),
        "source_case_path_canonical": canonical_case_path(case_dir),
        "source_case_id": os.path.basename(canonical_case_path(case_dir)),
        "source_physical_time": None if phys is None else float(phys),
        "source_final_solver_time": completion.get("final_solver_time_s"),
        "completion_mode": completion.get("mode") or "",
        "stop_reason": completion.get("stop_reason") or "",
        "wave_radius_reached": bool(completion.get("wave_radius_reached")),
        "requested_stop_radius_m": completion.get("requested_stop_radius_m"),
        "arrival_time_s": completion.get("detected_arrival_time_s"),
        "arrival_criterion": completion.get("arrival_criterion") or "",
        "coordinate_convention": COORDINATE_CONVENTION,
        "units": dict(UNITS),
        "field_names": list(REQUIRED_ARRAYS) + list(OPTIONAL_ARRAYS),
        "field_shapes": shapes,
        "n_points": int(n),
        "arrays_checksum": arrays_checksum(packed),
        "identity_fingerprint": identity_fingerprint(
            case_dir,
            physical_time=phys,
            stop_reason=str(completion.get("stop_reason") or ""),
            mode=str(completion.get("mode") or ""),
            wave_radius_reached=bool(completion.get("wave_radius_reached")),
        ),
        "created_utc": _utcnow(),
        "remap_source_type": SOURCE_SNAPSHOT,
        "remap_radius_m": completion.get("remap_radius_m"),
        "dr_1d_m": completion.get("dr_1d_m"),
        "remap_front_buffer_cells": completion.get("remap_front_buffer_cells"),
        "handoff_radius_m": completion.get("handoff_radius_m"),
        "handoff_time_s": completion.get("detected_arrival_time_s") or phys,
        "source_1d_case": canonical_case_path(case_dir),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    try:
        r_arr = packed.get("r")
        if r_arr is not None and getattr(r_arr, "size", 0):
            metadata["field_r_max_m"] = float(np.max(r_arr))
            metadata["field_r_min_m"] = float(np.min(r_arr))
    except (TypeError, ValueError):
        pass
    os.makedirs(case_dir, exist_ok=True)
    np.savez_compressed(snapshot_npz_path(case_dir), **packed)
    with open(snapshot_json_path(case_dir), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if metadata.get("handoff_radius_m") is not None:
        try:
            from remap_handoff_1d import read_handoff_metadata, write_handoff_metadata

            existing = read_handoff_metadata(case_dir) or {}
            existing.update(
                {
                    "remap_radius_m": metadata.get("remap_radius_m"),
                    "dr_1d_m": metadata.get("dr_1d_m"),
                    "remap_front_buffer_cells": metadata.get("remap_front_buffer_cells"),
                    "handoff_radius_m": metadata.get("handoff_radius_m"),
                    "handoff_time_s": metadata.get("handoff_time_s"),
                    "source_1d_case": metadata.get("source_1d_case")
                    or canonical_case_path(case_dir),
                    "field_r_max_m": metadata.get("field_r_max_m"),
                    "field_r_min_m": metadata.get("field_r_min_m"),
                    "source_physical_time": metadata.get("source_physical_time"),
                }
            )
            write_handoff_metadata(case_dir, existing)
        except Exception:
            pass
    return metadata


def read_snapshot_metadata(case_dir: str) -> Optional[Dict[str, Any]]:
    path = snapshot_json_path(case_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_snapshot_arrays(case_dir: str) -> Optional[Dict[str, np.ndarray]]:
    path = snapshot_npz_path(case_dir)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as payload:
            return {str(name): np.asarray(payload[name], dtype=np.float64) for name in payload.files}
    except (OSError, ValueError):
        return None


def validate_snapshot(
    case_dir: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    arrays: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[bool, str]:
    meta = metadata if metadata is not None else read_snapshot_metadata(case_dir)
    packed = arrays if arrays is not None else read_snapshot_arrays(case_dir)
    if meta is None or packed is None:
        return False, "Remap snapshot is missing or unreadable."
    try:
        version = int(meta.get("schema_version"))
    except (TypeError, ValueError):
        return False, "Remap snapshot schema version is missing or invalid."
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        return False, f"Remap snapshot schema version {version} is not supported."
    if str(meta.get("source_dimension") or "") != SOURCE_DIMENSION:
        return False, "Remap snapshot is not a 1D source."
    phys = meta.get("source_physical_time")
    if phys is not None:
        try:
            phys_f = float(phys)
        except (TypeError, ValueError):
            return False, "Remap snapshot physical time is invalid."
        if not np.isfinite(phys_f) or phys_f < 0.0:
            return False, "Remap snapshot physical time is invalid."
    for name in REQUIRED_ARRAYS:
        if name not in packed:
            return False, f"Remap snapshot is missing required field {name!r}."
    n = packed["r"].size
    if n < 2:
        return False, "Remap snapshot radial profile is too short."
    for name, arr in packed.items():
        if np.asarray(arr).size != n:
            return False, f"Remap snapshot field {name!r} has inconsistent length."
    expected_sum = arrays_checksum(packed)
    if str(meta.get("arrays_checksum") or "") != expected_sum:
        return False, "Remap snapshot checksum does not match the stored arrays."
    stored_paths = [
        str(meta.get("source_case_path") or ""),
        str(meta.get("source_case_path_canonical") or ""),
    ]
    stored_paths = [path for path in stored_paths if path]
    if stored_paths and not all(same_source_case(path, case_dir) for path in stored_paths):
        return False, "Remap snapshot belongs to a different source case."
    stored_path = stored_paths[0] if stored_paths else ""
    stored_id = str(meta.get("identity_fingerprint") or "")
    completion = _completion_info(case_dir)
    if completion.get("stop_reason") or completion.get("final_solver_time_s") is not None:
        live_id = identity_fingerprint(
            case_dir,
            physical_time=completion.get("final_solver_time_s"),
            stop_reason=str(completion.get("stop_reason") or ""),
            mode=str(completion.get("mode") or ""),
            wave_radius_reached=bool(completion.get("wave_radius_reached")),
        )
        if stored_id and live_id != stored_id:
            same_case = not stored_path or same_source_case(stored_path, case_dir)
            same_completion = (
                str(meta.get("stop_reason") or "") == str(completion.get("stop_reason") or "")
                and str(meta.get("completion_mode") or "") == str(completion.get("mode") or "")
                and bool(meta.get("wave_radius_reached"))
                == bool(completion.get("wave_radius_reached"))
            )
            if not (same_case and same_completion):
                return False, (
                    "Remap snapshot is stale relative to the current 1D completion record. "
                    "Re-run 1D or recapture the snapshot before remapping."
                )
        snap_t = meta.get("source_physical_time")
        final_t = completion.get("final_solver_time_s")
        if snap_t is not None and final_t is not None:
            try:
                if abs(float(snap_t) - float(final_t)) > max(
                    _TIME_MATCH_ABS, _TIME_MATCH_REL * max(abs(float(final_t)), 1.0)
                ):
                    return False, (
                        "Remap snapshot physical time does not match the 1D final solver time."
                    )
            except (TypeError, ValueError):
                return False, "Remap snapshot physical time is invalid."
    units = meta.get("units") or {}
    for key in ("r", "p", "T"):
        if key in units and str(units.get(key) or "") != UNITS[key]:
            return False, "Remap snapshot units are not the supported SI set."
    return True, ""


def profile_from_snapshot(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "r": np.asarray(arrays["r"], dtype=float),
        "p": np.asarray(arrays["p"], dtype=float),
        "T": np.asarray(arrays["T"], dtype=float),
        "rho.c4": np.asarray(arrays.get("rho.c4", np.zeros_like(arrays["r"])), dtype=float),
        "rho.air": np.asarray(arrays.get("rho.air", np.zeros_like(arrays["r"])), dtype=float),
        "alpha.c4": np.asarray(arrays.get("alpha.c4", np.zeros_like(arrays["r"])), dtype=float),
        "U_mag": np.asarray(arrays["U_mag"], dtype=float),
    }


def load_profile_for_remap(case_dir: str) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[str]]:
    """Return (profile, error). error set => do not fall back. both None => no snapshot."""
    if not snapshot_exists(case_dir):
        return None, None
    ok, message = validate_snapshot(case_dir)
    if not ok:
        return None, message
    arrays = read_snapshot_arrays(case_dir)
    if arrays is None:
        return None, "Remap snapshot arrays could not be loaded."
    return profile_from_snapshot(arrays), None


def _list_time_dirs(case_dir: str) -> List[Tuple[float, str]]:
    times: List[Tuple[float, str]] = []
    try:
        names = os.listdir(case_dir)
    except OSError:
        return []
    for name in names:
        if name in _SKIP_TIME or name.startswith("processor"):
            continue
        path = os.path.join(case_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            times.append((float(name), name))
        except ValueError:
            continue
    times.sort(key=lambda item: item[0])
    return times


def _times_match(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= max(
        _TIME_MATCH_ABS, _TIME_MATCH_REL * max(abs(float(right)), abs(float(left)), 1.0)
    )


def matching_time_dir(case_dir: str, physical_time: Optional[float]) -> Optional[str]:
    if physical_time is None:
        return None
    try:
        target = float(physical_time)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(target):
        return None
    best = None
    for value, name in _list_time_dirs(case_dir):
        if value <= 0.0:
            continue
        delta = abs(value - target)
        if _times_match(value, target) and (best is None or delta < best[0]):
            best = (delta, name)
    return None if best is None else best[1]


def _parse_internal_field(path: str, is_vector: bool = False):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None, None
    if "uniform" in text:
        match = re.search(r"internalField\s+uniform\s+([^;]+);", text)
        if match:
            token = match.group(1).strip()
            if is_vector:
                nums = tuple(float(x) for x in re.findall(r"[\d.eE+-]+", token))
                return None, np.array(nums[:3] if len(nums) >= 3 else (0.0, 0.0, 0.0))
            found = re.search(r"[\d.eE+-]+", token)
            return None, float(found.group()) if found else 0.0
    match = re.search(
        r"internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    if not match:
        return None, None
    count = int(match.group(1))
    inner = match.group(2)
    if is_vector:
        vals = []
        for triple in re.finditer(r"\(([^)]+)\)", inner):
            parts = triple.group(1).split()
            if len(parts) >= 3:
                vals.append((float(parts[0]), float(parts[1]), float(parts[2])))
        return (np.array(vals[:count], dtype=float) if vals else None), np.zeros(3)
    nums = [float(x) for x in re.findall(r"[\d.eE+-]+", inner)[:count]]
    return np.array(nums, dtype=float), 0.0


def _read_field(time_path: str, name: str, is_vector: bool = False):
    path = os.path.join(time_path, name)
    if not os.path.isfile(path):
        return None, None
    return _parse_internal_field(path, is_vector=is_vector)


def _foam_list_body(text: str) -> str:
    start = text.find("(")
    end = text.rfind(")")
    if start < 0 or end <= start:
        return ""
    return text[start + 1 : end]


def _parse_mesh_points(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    pts = [
        (float(a), float(b), float(c))
        for a, b, c in re.findall(
            r"\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)",
            _foam_list_body(text),
        )
    ]
    if not pts:
        return None
    return np.asarray(pts, dtype=float)


def _parse_mesh_faces(path: str) -> Optional[List[np.ndarray]]:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    faces: List[np.ndarray] = []
    for match in re.finditer(r"\d+\s*\(([^)]+)\)", _foam_list_body(text)):
        idx = [int(tok) for tok in match.group(1).split() if tok]
        if idx:
            faces.append(np.asarray(idx, dtype=int))
    return faces or None


def _parse_mesh_labels(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    nums = [int(tok) for tok in _foam_list_body(text).split() if tok]
    if not nums:
        return None
    return np.asarray(nums, dtype=int)


def cell_radii_from_poly_mesh(case_dir: str, n_cells: int) -> Optional[np.ndarray]:
    """True cell radii from constant/polyMesh. Do not linspace the point cloud."""
    mesh = os.path.join(case_dir or "", "constant", "polyMesh")
    points = _parse_mesh_points(os.path.join(mesh, "points"))
    faces = _parse_mesh_faces(os.path.join(mesh, "faces"))
    owner = _parse_mesh_labels(os.path.join(mesh, "owner"))
    if points is None or faces is None or owner is None:
        return None
    n_owner = int(owner.max()) + 1 if owner.size else 0
    n = max(int(n_cells), n_owner)
    if n < 2 or len(faces) != len(owner):
        return None
    acc = np.zeros((n, 3), dtype=float)
    weight = np.zeros(n, dtype=float)
    n_pts = len(points)

    def accumulate(cell: int, face_idx: int) -> None:
        if cell < 0 or cell >= n or face_idx < 0 or face_idx >= len(faces):
            return
        idx = faces[face_idx]
        if idx.size == 0 or int(idx.max()) >= n_pts:
            return
        acc[cell] += points[idx].mean(axis=0)
        weight[cell] += 1.0

    for face_i, cell in enumerate(owner):
        accumulate(int(cell), face_i)
    neighbour = _parse_mesh_labels(os.path.join(mesh, "neighbour"))
    if neighbour is not None:
        for face_i, cell in enumerate(neighbour):
            accumulate(int(cell), face_i)
    if not np.all(weight[:n_cells] > 0.0):
        return None
    centres = acc[:n_cells] / weight[:n_cells, None]
    return np.linalg.norm(centres, axis=1)


def capture_arrays_from_time_dir(case_dir: str, time_label: str) -> Optional[Dict[str, np.ndarray]]:
    time_path = os.path.join(case_dir, time_label)
    p_arr, p_def = _read_field(time_path, "p")
    t_arr, t_def = _read_field(time_path, "T")
    u_arr, _u_def = _read_field(time_path, "U", is_vector=True)
    rhoa_arr, ra_def = _read_field(time_path, "rho.air")
    if rhoa_arr is None and ra_def is None:
        rhoa_arr, ra_def = _read_field(time_path, "rho")
    rho4_arr, r4_def = _read_field(time_path, "rho.c4")
    a4_arr, a4_def = _read_field(time_path, "alpha.c4")
    n = 0
    for arr in (p_arr, t_arr, u_arr, rhoa_arr, rho4_arr, a4_arr):
        if arr is not None:
            n = max(n, len(arr))
    if n < 2:
        return None
    if p_arr is None:
        p_arr = np.full(n, 101325.0 if p_def is None else p_def)
    if t_arr is None:
        t_arr = np.full(n, 300.0 if t_def is None else t_def)
    if u_arr is None:
        u_mag = np.zeros(n)
    else:
        u_mag = np.linalg.norm(np.asarray(u_arr, dtype=float), axis=1)
    if rhoa_arr is None:
        rhoa_arr = np.full(n, 1.225 if ra_def is None else ra_def)
    if rho4_arr is None:
        rho4_arr = np.full(n, 0.0 if r4_def is None else r4_def)
    if a4_arr is None:
        a4_arr = np.full(n, 0.0 if a4_def is None else a4_def)
    centres_path = os.path.join(time_path, "C")
    r_1d = None
    if os.path.isfile(centres_path):
        c_arr, _ = _parse_internal_field(centres_path, is_vector=True)
        if c_arr is not None and len(c_arr) == n:
            r_1d = np.linalg.norm(np.asarray(c_arr, dtype=float), axis=1)
    if r_1d is None:
        r_1d = cell_radii_from_poly_mesh(case_dir, n)
    if r_1d is None:
        r_1d = np.arange(n, dtype=float) + 0.5
    return {
        "r": np.asarray(r_1d, dtype=float),
        "p": np.asarray(p_arr, dtype=float),
        "T": np.asarray(t_arr, dtype=float),
        "U_mag": np.asarray(u_mag, dtype=float),
        "rho.air": np.asarray(rhoa_arr, dtype=float),
        "rho.c4": np.asarray(rho4_arr, dtype=float),
        "alpha.c4": np.asarray(a4_arr, dtype=float),
    }


def latest_complete_time_dir(case_dir: str) -> Optional[str]:
    for _value, name in reversed(_list_time_dirs(case_dir)):
        if _value <= 0.0:
            continue
        arrays = capture_arrays_from_time_dir(case_dir, name)
        if arrays is not None and arrays["r"].size >= 2:
            return name
    return None


def write_snapshot_from_time_dir(
    case_dir: str,
    time_label: str,
    *,
    physical_time: Optional[float] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    arrays = capture_arrays_from_time_dir(case_dir, time_label)
    if arrays is None:
        return None
    extra = dict(extra_metadata or {})
    extra["captured_from_time_dir"] = str(time_label)
    try:
        t_label = float(time_label)
    except ValueError:
        t_label = physical_time
    return write_snapshot(
        case_dir,
        arrays,
        physical_time=physical_time if physical_time is not None else t_label,
        extra_metadata=extra,
    )


def write_snapshot_after_run(
    case_dir: str,
    completion: Any = None,
    *,
    user_stopped: bool = False,
) -> str:
    """Capture a remap snapshot after a 1D run. Returns a short status string."""
    stop_reason = str(getattr(completion, "stop_reason", "") or "")
    final_t = getattr(completion, "final_solver_time_s", None)
    arrived = bool(getattr(completion, "wave_radius_reached", False))
    if user_stopped:
        time_label = matching_time_dir(case_dir, final_t)
        if not time_label:
            return (
                "Remap snapshot not written: the stopped 1D state has no matching "
                "field dump to capture."
            )
        meta = write_snapshot_from_time_dir(case_dir, time_label, physical_time=final_t)
        if meta is None:
            return "Remap snapshot not written: the stopped 1D field dump is incomplete."
        return (
            "Remap snapshot written from last solver state "
            f"(t={float(meta['source_physical_time']):.6g} s)."
        )
    if stop_reason == "wave_radius_reached" or arrived:
        time_label = matching_time_dir(case_dir, final_t)
        if not time_label:
            return (
                "Remap snapshot unavailable: 1D stopped before writeInterval and no "
                "final field dump was captured. Remap needs the last solver state snapshot."
            )
        meta = write_snapshot_from_time_dir(case_dir, time_label, physical_time=final_t)
        if meta is None:
            return "Remap snapshot unavailable: the final 1D field dump is incomplete."
        return (
            "Remap snapshot written from last solver state "
            f"(t={float(meta['source_physical_time']):.6g} s)."
        )
    if stop_reason == "end_time_reached":
        time_label = matching_time_dir(case_dir, final_t) or latest_complete_time_dir(case_dir)
        if not time_label:
            return "Remap snapshot unavailable: End Time finished without a field dump."
        meta = write_snapshot_from_time_dir(case_dir, time_label, physical_time=final_t)
        if meta is None:
            return "Remap snapshot unavailable: the End Time field dump is incomplete."
        return (
            "Remap snapshot written from last solver state "
            f"(t={float(meta['source_physical_time']):.6g} s)."
        )
    return ""


def resolve_remap_source(case_dir: str) -> RemapSourceResolution:
    """Prefer a valid snapshot; fall back to OpenFOAM only when no snapshot exists."""
    if not case_dir or not os.path.isdir(case_dir):
        return RemapSourceResolution(
            ok=False,
            blocked=False,
            message="The selected 1D source case does not exist.",
        )
    if snapshot_exists(case_dir):
        ok, message = validate_snapshot(case_dir)
        if not ok:
            return RemapSourceResolution(
                ok=False,
                blocked=True,
                source_type=SOURCE_SNAPSHOT,
                message=message,
            )
        meta = read_snapshot_metadata(case_dir) or {}
        arrays = read_snapshot_arrays(case_dir) or {}
        phys = meta.get("source_physical_time")
        try:
            phys_f = float(phys) if phys is not None else None
        except (TypeError, ValueError):
            phys_f = None
        time_label = (
            f"{phys_f:.12g}"
            if phys_f is not None
            else str(meta.get("captured_from_time_dir") or "snapshot")
        )
        names = tuple(str(n) for n in (meta.get("field_names") or list(arrays)))
        return RemapSourceResolution(
            ok=True,
            blocked=False,
            source_type=SOURCE_SNAPSHOT,
            time_label=time_label,
            physical_time=phys_f,
            message=(
                "Remap is available from the last 1D solver state snapshot"
                + (f" (t={phys_f:.6g} s)." if phys_f is not None else ".")
            ),
            field_names=names,
            profile=profile_from_snapshot(arrays),
            metadata=meta,
        )
    latest = latest_complete_time_dir(case_dir)
    if not latest:
        return RemapSourceResolution(
            ok=False,
            blocked=False,
            message=(
                "Remap is unavailable: no valid 1D remap snapshot and no complete "
                "OpenFOAM time directory. The last solver state was not captured. "
                "This is independent of writeInterval."
            ),
        )
    completion = _completion_info(case_dir)
    final_t = completion.get("final_solver_time_s")
    try:
        latest_t = float(latest)
    except ValueError:
        latest_t = None
    if final_t is not None and latest_t is not None and not _times_match(latest_t, float(final_t)):
        return RemapSourceResolution(
            ok=False,
            blocked=False,
            message=(
                "Remap is unavailable: no remap snapshot, and the latest OpenFOAM "
                f"time directory ({latest}) does not match the final solver time "
                f"({float(final_t):.6g} s). It would be a stale writeInterval dump."
            ),
        )
    arrays = capture_arrays_from_time_dir(case_dir, latest) or {}
    phys = latest_t if latest_t is not None else final_t
    return RemapSourceResolution(
        ok=True,
        blocked=False,
        source_type=SOURCE_OPENFOAM,
        time_label=latest,
        physical_time=phys,
        message=(
            "Remap snapshot is missing; using OpenFOAM time directory "
            f"{latest} as a fallback."
        ),
        field_names=tuple(arrays),
        profile=profile_from_snapshot(arrays) if arrays else None,
    )


def availability_for_case(case_dir: str) -> RemapAvailability:
    completion = _completion_info(case_dir)
    solver_done = bool(
        completion.get("stop_reason") in ("wave_radius_reached", "end_time_reached")
        or (
            completion.get("stop_reason") == "user_stopped"
            and completion.get("final_solver_time_s") is not None
        )
    )
    if not case_dir or not os.path.isdir(case_dir):
        return RemapAvailability(
            solver_completed=solver_done,
            status="missing",
            message="No 1D source case is selected.",
        )
    resolved = resolve_remap_source(case_dir)
    if resolved.blocked:
        return RemapAvailability(
            solver_completed=solver_done,
            snapshot_available=False,
            snapshot_invalid=True,
            status="invalid",
            message=resolved.message,
            physical_time=resolved.physical_time,
            source_type=SOURCE_SNAPSHOT,
        )
    if resolved.ok and resolved.source_type == SOURCE_SNAPSHOT:
        return RemapAvailability(
            solver_completed=solver_done,
            snapshot_available=True,
            openfoam_fallback_available=latest_complete_time_dir(case_dir) is not None,
            status="available_snapshot",
            message=resolved.message,
            physical_time=resolved.physical_time,
            source_type=SOURCE_SNAPSHOT,
        )
    if resolved.ok and resolved.source_type == SOURCE_OPENFOAM:
        return RemapAvailability(
            solver_completed=solver_done,
            openfoam_fallback_available=True,
            status="available_openfoam",
            message=resolved.message,
            physical_time=resolved.physical_time,
            source_type=SOURCE_OPENFOAM,
        )
    status = "stale" if "stale" in (resolved.message or "").lower() else "missing"
    return RemapAvailability(
        solver_completed=solver_done,
        status=status,
        message=resolved.message or "Remap is unavailable.",
        physical_time=completion.get("final_solver_time_s"),
    )


def _positive_radius_m(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def declared_remap_radius_m(case_dir: str) -> Optional[float]:
    """Return explicit 1D R_remap only.

    Domain Radius, requested stop radius, and numerical field extent are
    not substitutes. Remap = No sources return None.
    """
    if not case_dir:
        return None
    try:
        from remap_handoff_1d import read_handoff_metadata

        handoff = read_handoff_metadata(case_dir) or {}
    except Exception:
        handoff = {}
    found = _positive_radius_m(handoff.get("remap_radius_m"))
    if found is not None:
        return found
    snap = read_snapshot_metadata(case_dir) or {}
    return _positive_radius_m(snap.get("remap_radius_m"))


def source_field_r_max_m(case_dir: str) -> Optional[float]:
    if not case_dir:
        return None
    snap = read_snapshot_metadata(case_dir) or {}
    found = _positive_radius_m(snap.get("field_r_max_m"))
    if found is not None:
        return found
    try:
        from remap_handoff_1d import read_handoff_metadata

        handoff = read_handoff_metadata(case_dir) or {}
    except Exception:
        handoff = {}
    return _positive_radius_m(handoff.get("field_r_max_m"))


def transfer_limit_notes(case_dir: str, mapped_radius: float) -> List[str]:
    """GUI / preflight notes that keep Domain Radius distinct from R_remap."""
    notes: List[str] = []
    if not case_dir:
        return notes
    try:
        mapped = float(mapped_radius)
    except (TypeError, ValueError):
        mapped = 0.0
    if not math.isfinite(mapped) or mapped < 0.0:
        mapped = 0.0
    completion = _completion_info(case_dir)
    has_completion = False
    try:
        from completion_1d import read_completion_record

        has_completion = read_completion_record(case_dir) is not None
    except Exception:
        has_completion = False
    declared = declared_remap_radius_m(case_dir)
    field_max = source_field_r_max_m(case_dir)
    if has_completion and not bool(completion.get("remap_for_2d")):
        notes.append(
            "This 1D run used Remap = No. Domain Radius is the computational "
            "domain, not a physical R_remap."
        )
        stop_r = _positive_radius_m(completion.get("requested_stop_radius_m"))
        if stop_r is not None:
            notes.append(f"1D Domain / stop radius was {stop_r:g} m.")
    elif declared is not None:
        notes.append(f"1D R_remap = {declared:g} m.")
    if mapped > 0.0:
        notes.append(
            f"2D will transfer source state only within mapped radius {mapped:g} m."
        )
    if field_max is not None and mapped > 0.0 and mapped + 1e-9 < field_max:
        notes.append(
            f"Source field extends to {field_max:.6g} m; state beyond "
            f"{mapped:g} m stays ambient."
        )
    return notes


def display_text(avail: RemapAvailability) -> str:
    """Short GUI text distinguishing solver completion from remap availability."""
    parts: List[str] = []
    if avail.solver_completed:
        parts.append("Solver completed.")
    if avail.snapshot_available:
        parts.append("Remap snapshot available from the last solver state.")
    elif avail.snapshot_invalid:
        parts.append("Remap snapshot is stale, missing, or invalid.")
    elif avail.openfoam_fallback_available:
        parts.append("Remap snapshot missing; OpenFOAM time-directory fallback is available.")
    elif avail.status == "stale":
        parts.append("Remap source is stale.")
    elif avail.status == "missing":
        parts.append("Remap snapshot is missing.")
    if avail.message and avail.message not in " ".join(parts):
        parts.append(avail.message)
    return " ".join(parts).strip()
