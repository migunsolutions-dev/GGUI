import os
import math
import shutil
from typing import Dict, Any, Optional

try:
    import pyvista as pv
except ImportError:
    pv = None

from base_generator import BaseGenerator
from charge_capture import resolve_charge_capture_radius_m
from models import CaseInputs3D
from path_utils import get_latest_time_dir, win_to_wsl_path


def effective_charge_refine(inputs) -> int:
    """Compute the effective charge refinement level.

    If the user-specified level (``charge_refinement_level``) is 0 but the
    charge sphere/cylinder is smaller than the base cell, we auto-raise the
    refinement level so that ``setRefinedFields`` will refine the mesh enough
    to contain at least one cell inside the charge volume.

    Returns the effective level (may be larger than the user value).
    """
    user_level = getattr(inputs, "charge_refinement_level", 0)
    shape = getattr(inputs, "charge_shape", "")
    if shape not in ("Sphere", "Cylinder"):
        return user_level  # Cuboid uses boxToCell – mesh resolution less critical

    cell_size = getattr(inputs, "cell_size", 0.5)
    mass_kg = getattr(inputs, "mass_kg", 1.0)
    rho = getattr(inputs, "rho_charge", 0) or 1600.0
    if rho <= 0:
        rho = 1600.0
    vol = mass_kg / rho

    if shape == "Sphere":
        charge_radius = ((3.0 * vol) / (4.0 * math.pi)) ** (1.0 / 3.0)
    else:  # Cylinder
        charge_radius = getattr(inputs, "cylinder_radius", 0) or 0.05
        if charge_radius <= 0:
            charge_radius = 0.05

    # Need cell_size / 2^level < 2 * charge_radius (diameter) so that at
    # least one cell centre falls inside the charge.  Solve for level:
    #   level > log2(cell_size / (2 * charge_radius))
    if charge_radius > 0 and cell_size > 2 * charge_radius:
        min_level = math.ceil(math.log2(cell_size / (2 * charge_radius)))
        return max(user_level, min_level)

    # Charge is large enough for the mesh → user's own value is fine,
    # but guarantee at least 1 when user asked for refinement.
    return user_level


def check_coarse_mesh_would_fail(inputs) -> Optional[str]:
    """Fail-fast: if no startup charge refinement and base mesh is too coarse to capture
    the charge by center-based selection, return an error message; else return None.
    Used to avoid running init that would fail with 'No cells captured inside the charge volume'.
    """
    enable_dyn = getattr(inputs, "enable_dyn_refine", None)
    charge_level = max(0, getattr(inputs, "charge_refinement_level", 0) or 0)
    startup_refine = enable_dyn and charge_level > 0
    if startup_refine:
        return None
    shape = getattr(inputs, "charge_shape", "")
    if shape not in ("Sphere", "Cylinder"):
        return None
    cell_size = max(1e-9, getattr(inputs, "cell_size", 0.5))
    mass_kg = getattr(inputs, "mass_kg", 1.0)
    rho = getattr(inputs, "rho_charge", 0) or 1600.0
    if shape == "Sphere":
        vol = mass_kg / max(1e-9, rho)
        charge_radius = ((3.0 * vol) / (4.0 * math.pi)) ** (1.0 / 3.0)
    else:
        charge_radius = getattr(inputs, "cylinder_radius", 0) or 0.05
    if charge_radius <= 0:
        return None
    # For at least one cell center to lie inside a sphere of radius charge_radius,
    # we need cell_size < 2*charge_radius/sqrt(3) (nearest cell center at distance ~ 0.5*cell_size*sqrt(3))
    if cell_size >= (2.0 * charge_radius / math.sqrt(3.0)):
        return (
            "Base mesh is too coarse to capture the charge: no cell centers fall inside the charge region. "
            "Either reduce Cell Size, or enable Dyn Mesh (AMR) and set Charge pre-refinement (Inside) > 0 "
            "so that startup mesh refinement runs before charge fill."
        )
    return None


# Maximum startup refineMesh levels (avoids runaway refinement).
STARTUP_REFINEMENT_CAP = 6

def build_charge_capture_impossible_message(
    inputs, dims: Dict[str, float],
) -> tuple:
    """Build user-facing no-capture message and suggested max cell size.
    Returns (message: str, suggested_cell_size_max: Optional[float]).
    Use when charge_capture_possible is False. Message explains what failed, why, and what to change
    (reduce Cell Size or enlarge charge). Does not suggest increasing Inside (capture must be possible on base mesh first)."""
    shape = getattr(inputs, "charge_shape", "")
    cell_size = max(1e-9, getattr(inputs, "cell_size", 0.5))
    explain = (
        "The base mesh is too coarse relative to the charge size: no base-mesh cell centres lie inside the charge. "
        "Charge-interior refinement (Inside) cannot start because the charge region cannot be selected. "
        "At least one base-mesh cell centre must lie inside the charge before charge-interior refinement can run."
    )
    action = (
        "Reduce base Cell Size below the suggested value so that at least one cell centre lies inside the charge, "
        "or enlarge the charge if the geometry was entered incorrectly. "
        "Do not rely on increasing Inside alone to fix this; capture must be possible on the base mesh first."
    )
    suggested = None

    if shape == "Sphere":
        r = float(dims.get("radius", 0.05))
        if r <= 0:
            return "Charge capture check: charge radius is zero or missing.", None
        # At least one cell centre inside sphere: cell_size < 2*R/sqrt(3)
        h_max = (2.0 * r) / math.sqrt(3.0)
        suggested = h_max
        size_info = "Charge shape: Sphere. Charge radius: %.6g m. Current base Cell Size: %.6g m." % (r, cell_size)
        threshold_info = "Reduce Cell Size below approximately %.6g m." % suggested
        msg = "Charge capture failed.\n\n%s\n\n%s\n\n%s %s\n\n%s" % (explain, action, size_info, threshold_info, action)
        return msg, suggested

    if shape == "Cylinder":
        r = float(dims.get("radius", 0.05))
        length = dims.get("length", 0.1)
        if r <= 0:
            return "Charge capture check: cylinder radius is zero or missing.", None
        h_max = (2.0 * r) / math.sqrt(3.0)
        suggested = h_max
        size_info = (
            "Charge shape: Cylinder. Radius: %.6g m, length: %.6g m. Current base Cell Size: %.6g m."
            % (r, length, cell_size)
        )
        threshold_info = "Reduce Cell Size below approximately %.6g m (based on cylinder radius)." % suggested
        msg = "Charge capture failed.\n\n%s\n\n%s\n\n%s %s\n\n%s" % (explain, action, size_info, threshold_info, action)
        return msg, suggested

    if shape == "Cuboid":
        L = dims.get("length", 0.1)
        W = dims.get("width", 0.1)
        H = dims.get("height", 0.1)
        if L <= 0 or W <= 0 or H <= 0:
            return "Charge capture check: cuboid dimensions are zero or missing.", None
        min_side = min(L, W, H)
        h_max = min_side / math.sqrt(3.0)
        suggested = h_max
        size_info = (
            "Charge shape: Cuboid. Dimensions (L x W x H): %.6g x %.6g x %.6g m. Current base Cell Size: %.6g m."
            % (L, W, H, cell_size)
        )
        threshold_info = "Reduce Cell Size below approximately %.6g m (based on smallest side)." % suggested
        msg = "Charge capture failed.\n\n%s\n\n%s\n\n%s %s\n\n%s" % (explain, action, size_info, threshold_info, action)
        return msg, suggested

    size_info = "Current base Cell Size: %.6g m." % cell_size
    msg = "Charge capture failed.\n\n%s\n\n%s\n\n%s" % (explain, action, size_info)
    return msg, None


def charge_capture_possible(inputs, dims: Dict[str, float]) -> tuple:
    """Return (possible: bool, message: Optional[str]). If not possible, message is the rich user-facing reason (no auto-adjustment)."""
    shape = getattr(inputs, "charge_shape", "")
    cell_size = max(1e-9, getattr(inputs, "cell_size", 0.5))
    if shape == "Sphere":
        r = float(dims.get("radius", 0.05))
        if r <= 0:
            return True, None
        h_max = (2.0 * r) / math.sqrt(3.0)
        if cell_size >= h_max:
            msg, _ = build_charge_capture_impossible_message(inputs, dims)
            return False, msg
        return True, None
    if shape == "Cylinder":
        r = float(dims.get("radius", 0.05))
        if r <= 0:
            return True, None
        h_max = (2.0 * r) / math.sqrt(3.0)
        if cell_size >= h_max:
            msg, _ = build_charge_capture_impossible_message(inputs, dims)
            return False, msg
        return True, None
    if shape == "Cuboid":
        L = dims.get("length", 0.1)
        W = dims.get("width", 0.1)
        H = dims.get("height", 0.1)
        if L <= 0 or W <= 0 or H <= 0:
            return True, None
        min_side = min(L, W, H)
        if cell_size >= min_side / math.sqrt(3.0):
            msg, _ = build_charge_capture_impossible_message(inputs, dims)
            return False, msg
        return True, None
    return True, None


def build_inside_insufficient_for_capture_message(
    inside_level: int,
    n_capture_needed: int,
    inputs,
    dims: Dict[str, float],
) -> str:
    """Build user-facing message when Inside < n_capture_needed.

    The criterion is that the user-requested inside_level must be enough to progressively
    refine the mesh until the true charge geometry can be seeded.  This is entirely
    controlled by the user's Inside setting and base Cell Size — no automatic correction
    or geometry inflation is applied.
    """
    cell_size = max(1e-9, getattr(inputs, "cell_size", 0.5))
    shape = getattr(inputs, "charge_shape", "")
    intro = (
        "The current inner refinement settings are insufficient to create a valid refined "
        "inner initialization region for this charge geometry and base mesh.\n\n"
        "Requested Inside level   : %d\n"
        "Minimum Inside required  : %d\n"
        "Base Cell Size           : %.6g m\n\n"
    ) % (inside_level, n_capture_needed, cell_size)
    if shape == "Sphere":
        r = float(dims.get("radius", 0.05))
        intro += "Charge: Sphere, radius %.6g m.\n\n" % r
    elif shape == "Cylinder":
        r = float(dims.get("radius", 0.05))
        length = dims.get("length", 0.1)
        intro += "Charge: Cylinder, radius %.6g m, length %.6g m.\n\n" % (r, length)
    elif shape == "Cuboid":
        L, W, H = dims.get("length", 0.1), dims.get("width", 0.1), dims.get("height", 0.1)
        intro += "Charge: Cuboid %.6g × %.6g × %.6g m.\n\n" % (L, W, H)
    else:
        intro += "Charge shape: %s.\n\n" % (shape or "unknown")
    guidance = (
        "Increase Charge pre-refinement (Inside) to at least %d, or reduce the base Cell Size. "
        "No automatic corrective mesh inflation or hidden capture refinement will be applied."
    ) % n_capture_needed
    return intro + guidance


def _charge_radius_for_capture(inputs, dims: Dict[str, float]) -> float:
    """Effective charge radius [m] used for startup capture criterion (cell center inside region).
    Sphere: geometric radius. Cylinder: radius (radial extent). Cuboid: not used for startup split."""
    shape = getattr(inputs, "charge_shape", "")
    if shape == "Sphere":
        return float(dims.get("radius", 0.05))
    if shape == "Cylinder":
        return float(dims.get("radius", 0.05))
    return 0.0


def compute_startup_refinement_split(
    inputs, dims: Dict[str, float],
) -> Dict[str, Any]:
    """Compute how to split the user's target Inside level into startup (refineMesh) vs remaining (setRefinedFields).

    The user's "Charge pre-refinement (Inside)" is the TARGET FINAL refinement level relative to base cell size.
    We split into:
      - startup_levels: topoSet + refineMesh runs before charge fill (for capture).
      - remaining_inside_levels: level used in setRefinedFields (refineInternal) so total = target.

    Formula for startup_levels_needed (mesh-first capture):
      - For at least one cell center to lie inside a sphere of radius R, we need
        cell size after startup h_after < 2*R/sqrt(3) (worst case: origin at cell corner).
      - h_after = h_base / 2^n  =>  n >= ceil(log2(h_base / (2*R/sqrt(3)))).
    Capped at STARTUP_REFINEMENT_CAP. No backup regions.

    Returns dict with:
      startup_levels_needed, startup_levels, remaining_inside_levels,
      target_inside_level, charge_radius_used, h_base, h_final_expected,
      auto_adjusted (bool), message (str, optional warning).
    """
    target_inside_level = max(0, getattr(inputs, "charge_refinement_level", 0) or 0)
    h_base = max(1e-9, getattr(inputs, "cell_size", 0.5))
    charge_radius = _charge_radius_for_capture(inputs, dims)
    shape = getattr(inputs, "charge_shape", "")

    out = {
        "startup_levels_needed": 0,
        "startup_levels": 0,
        "remaining_inside_levels": target_inside_level,
        "target_inside_level": target_inside_level,
        "charge_radius_used": charge_radius,
        "h_base": h_base,
        "h_final_expected": h_base / (2.0 ** target_inside_level) if target_inside_level > 0 else h_base,
        "auto_adjusted": False,
        "message": None,
        "startup_refinement_capped": False,
    }

    if shape not in ("Sphere", "Cylinder") or target_inside_level <= 0 or charge_radius <= 0:
        out["remaining_inside_levels"] = target_inside_level
        return out

    # Conservative: need h_after <= 2*R/sqrt(3) so at least one cell center inside sphere of radius R
    # n >= log2(h_base / (2*R/sqrt(3)))  =>  startup_levels_needed = ceil(log2(h_base * sqrt(3) / (2*R)))
    h_max_for_capture = (2.0 * charge_radius) / math.sqrt(3.0)
    if h_base <= h_max_for_capture:
        startup_levels_needed = 0
    else:
        startup_levels_needed = math.ceil(math.log2(h_base / h_max_for_capture))
    startup_levels_needed = max(0, min(startup_levels_needed, STARTUP_REFINEMENT_CAP))
    if startup_levels_needed == STARTUP_REFINEMENT_CAP and h_base / (2.0 ** STARTUP_REFINEMENT_CAP) > h_max_for_capture:
        out["startup_refinement_capped"] = True

    out["startup_levels_needed"] = startup_levels_needed

    # Split: startup_levels = min(needed, target); remaining = target - startup_levels
    # If capture needs more than target, use needed for startup and 0 for remaining (with warning)
    if startup_levels_needed <= target_inside_level:
        startup_levels = startup_levels_needed
        remaining_inside_levels = target_inside_level - startup_levels
    else:
        startup_levels = startup_levels_needed
        remaining_inside_levels = 0
        out["auto_adjusted"] = True
        out["message"] = (
            "Requested Inside level (%d) was too low to capture the charge on the current base mesh. "
            "Startup refinement was increased automatically (%d levels) to ensure capture. "
            "To get the requested finer structure consistently, reduce Cell Size or increase Charge pre-refinement (Inside)."
            % (target_inside_level, startup_levels)
        )

    out["startup_levels"] = startup_levels
    out["remaining_inside_levels"] = remaining_inside_levels
    out["h_final_expected"] = h_base / (2.0 ** (startup_levels + remaining_inside_levels)) if (startup_levels + remaining_inside_levels) > 0 else h_base
    return out


def _is_time_like_dir(name: str) -> bool:
    """True if name looks like an OpenFOAM time directory (e.g. 0, 0.001, 1e-4)."""
    if not name or not name.strip():
        return False
    try:
        t = float(name.strip())
        return t >= 0
    except ValueError:
        return False


def _split_source_case_and_time(path: str) -> tuple:
    """
    If path ends with a time-directory segment, return (case_dir, time_dir_name).
    Otherwise return (path, None). Ensures rotateFields gets case root and -sourceTime separately.
    """
    norm = os.path.normpath(path.strip())
    if not norm:
        return "", None
    last = os.path.basename(norm)
    if _is_time_like_dir(last):
        parent = os.path.dirname(norm)
        if parent:
            return parent, last
    return norm, None


_REMAP_RADIAL_SCRIPT = r'''
# High-performance radial remap: 1D -> 3D by distance from Origin (vectorized NumPy I/O).
# Generated by BlastFoam GUI. Run from 3D case root after: blockMesh, cp 0.orig 0, postProcess -func writeCellCentres.
from __future__ import print_function
import os
import sys
import re
import io
import numpy as np

SOURCE_1D_CASE = {source_case}
SOURCE_TIME = {source_time}
ORIGIN = np.array([{origin_x}, {origin_y}, {origin_z}], dtype=np.float64)
# Mode A (post-detonation): write p,U,T,rho,h only to MAP_TIME for rhoCentralFoam; no alpha.c4/rho.c4.
POST_DETONATION = {post_detonation}
# Output directory: MAP_TIME (e.g. t_map) for inert continuation; "0" otherwise.
OUT_DIR = os.environ.get("MAP_TIME", "0")

def _find_latest_time(case_path):
    """Find the latest (largest) numeric time directory in case_path."""
    skip = {{"constant", "system", "0.orig", "postProcessing"}}
    best = None
    try:
        for name in os.listdir(case_path):
            if name in skip or name.startswith("processor"):
                continue
            path = os.path.join(case_path, name)
            if not os.path.isdir(path):
                continue
            try:
                t = float(name)
                if best is None or t > best[0]:
                    best = (t, name)
            except ValueError:
                pass
        return best[1] if best else None
    except OSError:
        return None

def _get_available_time_dirs(case_path):
    """Return list of (float_value, dir_name) for all numeric time directories, sorted by time."""
    skip = {{"constant", "system", "0.orig", "postProcessing"}}
    times = []
    try:
        for name in os.listdir(case_path):
            if name in skip or name.startswith("processor"):
                continue
            path = os.path.join(case_path, name)
            if not os.path.isdir(path):
                continue
            try:
                t = float(name)
                times.append((t, name))
            except ValueError:
                pass
        return sorted(times, key=lambda x: x[0])
    except OSError:
        return []

def _resolve_time_dir(case_path, requested_time, tolerance_rel=1e-9):
    """
    Resolve requested_time to an actual time directory in case_path.
    
    Strategy:
    1. If requested_time == "latest", return the largest time dir.
    2. Try exact string match first (fast path).
    3. Parse requested_time as float and find:
       a) Exact numeric match (abs diff < tolerance_rel * requested_val)
       b) Closest match if no exact found (and print warning)
    
    Returns: (resolved_dir_name: str or None, diagnostics: dict)
    """
    diagnostics = {{
        "requested": requested_time,
        "case_path": case_path,
        "method": None,
        "candidates": [],
        "resolved": None,
        "error": None
    }}
    
    # Handle "latest"
    if str(requested_time).strip().strip("'\\\"").lower() == "latest":
        resolved = _find_latest_time(case_path)
        diagnostics["method"] = "latest"
        diagnostics["resolved"] = resolved
        if resolved:
            return resolved, diagnostics
        diagnostics["error"] = "No time directories found"
        return None, diagnostics
    
    # Get all available time dirs
    available = _get_available_time_dirs(case_path)
    diagnostics["candidates"] = [name for _, name in available[:25]]  # limit for display
    
    if not available:
        diagnostics["error"] = "No time directories in case"
        return None, diagnostics
    
    requested_str = str(requested_time).strip().strip("'\\\"")
    
    # Try exact string match first
    for t_val, t_name in available:
        if t_name == requested_str:
            diagnostics["method"] = "exact_string"
            diagnostics["resolved"] = t_name
            return t_name, diagnostics
    
    # Try numeric matching
    try:
        requested_val = float(requested_str)
    except ValueError:
        diagnostics["error"] = f"Cannot parse requested time as float: {{requested_str}}"
        return None, diagnostics
    
    # Exact numeric match (within tolerance)
    abs_tol = abs(requested_val * tolerance_rel) if requested_val != 0 else tolerance_rel
    for t_val, t_name in available:
        if abs(t_val - requested_val) < abs_tol:
            diagnostics["method"] = "exact_numeric"
            diagnostics["resolved"] = t_name
            diagnostics["requested_val"] = requested_val
            diagnostics["resolved_val"] = t_val
            return t_name, diagnostics
    
    # No exact match - find closest
    closest = min(available, key=lambda x: abs(x[0] - requested_val))
    diagnostics["method"] = "closest"
    diagnostics["resolved"] = closest[1]
    diagnostics["requested_val"] = requested_val
    diagnostics["resolved_val"] = closest[0]
    diagnostics["error"] = f"No exact match; using closest: {{closest[1]}} (requested={{requested_val}}, found={{closest[0]}})"
    return closest[1], diagnostics

def _parse_of_points(path):
    with open(path, "r") as f:
        content = f.read()
    start = content.find("points")
    if start == -1:
        return np.zeros((0, 3))
    pts = []
    for m in re.finditer(r"\(\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*\)", content[start:start+100000]):
        pts.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return np.array(pts) if pts else np.zeros((0, 3))

def _parse_internal_field(path, is_vector=False):
    with open(path, "r") as f:
        text = f.read()
    if "uniform" in text:
        m = re.search(r"internalField\s+uniform\s+([^;]+);", text)
        if m:
            val = m.group(1).strip()
            if is_vector:
                t = tuple(float(x) for x in re.findall(r"[\d.e+-]+", val))
                return None, np.array(t[:3] if len(t) >= 3 else (0, 0, 0))
            return None, float(re.search(r"[\d.e+-]+", val).group())
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(([^)]+(?:\([^)]*\)[^)]*)*)\)\s*;", text, re.DOTALL)
    if not m:
        return None, np.zeros(3) if is_vector else 0.0
    n = int(m.group(1))
    inner = m.group(2)
    if is_vector:
        vals = []
        for triple in re.finditer(r"\(([^)]+)\)", inner):
            parts = triple.group(1).split()
            if len(parts) >= 3:
                vals.append((float(parts[0]), float(parts[1]), float(parts[2])))
        return np.array(vals[:n]) if vals else None, np.zeros(3)
    nums = re.findall(r"[\d.e+-]+", inner)
    return np.array([float(x) for x in nums[:n]]), 0.0

def _read_1d_data(source_case, time_dir):
    mesh_dir = os.path.join(source_case, "constant", "polyMesh")
    points_path = os.path.join(mesh_dir, "points")
    if not os.path.isfile(points_path):
        return None
    points = _parse_of_points(points_path)
    if len(points) == 0:
        return None
    r_pts = np.linalg.norm(points, axis=1)
    r_min, r_max = float(np.min(r_pts)), float(np.max(r_pts))
    time_path = os.path.join(source_case, time_dir)
    def read_field(name, vec=False):
        p = os.path.join(time_path, name)
        if not os.path.isfile(p):
            return None, None
        return _parse_internal_field(p, is_vector=vec)
    p_arr, p_def = read_field("p")
    T_arr, T_def = read_field("T")
    rho4_arr, r4_def = read_field("rho.c4")
    rhoa_arr, ra_def = read_field("rho.air")
    a4_arr, a4_def = read_field("alpha.c4")
    U_arr, U_def = read_field("U", vec=True)
    n = len(p_arr) if p_arr is not None else (len(T_arr) if T_arr is not None else 1)
    n = max(1, n)
    r_1d = np.linspace(r_min + (r_max - r_min) / (2 * n), r_max - (r_max - r_min) / (2 * n), n)
    if p_arr is None: p_arr = np.full(n, p_def)
    if T_arr is None: T_arr = np.full(n, T_def)
    if rho4_arr is None: rho4_arr = np.full(n, r4_def)
    if rhoa_arr is None: rhoa_arr = np.full(n, ra_def)
    if a4_arr is None: a4_arr = np.full(n, a4_def)
    if U_arr is None: U_mag = np.zeros(n)
    else: U_mag = np.linalg.norm(U_arr, axis=1)
    return {{"r": r_1d, "p": p_arr, "T": T_arr, "rho.c4": rho4_arr, "rho.air": rhoa_arr, "alpha.c4": a4_arr, "U_mag": U_mag}}

def _read_3d_cell_centres():
    c_file = os.path.join("0", "C")
    if os.path.isfile(c_file):
        with open(c_file, "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if "internalField" in lines[i] and "nonuniform" in lines[i] and "List<vector>" in lines[i]:
                try:
                    n = int(lines[i + 1].strip().rstrip(";"))
                    i += 2
                    while i < len(lines) and lines[i].strip() != "(":
                        i += 1
                    i += 1
                    pts = []
                    for _ in range(n):
                        if i >= len(lines):
                            break
                        m = re.match(r"\s*\(\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*\)", lines[i])
                        if m:
                            pts.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
                        i += 1
                    if len(pts) == n:
                        return np.array(pts)
                except (ValueError, IndexError):
                    pass
                break
            i += 1
        pts = []
        with open(c_file, "r") as f:
            text = f.read()
        start = text.find("internalField nonuniform List<vector>")
        if start >= 0:
            end = text.find("boundaryField", start)
            if end < 0:
                end = len(text)
            block = text[start:end]
            for m in re.finditer(r"\(\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*\)", block):
                pts.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
        if pts:
            return np.array(pts)
    try:
        import pyvista as pv
        if os.path.isfile("case.foam"):
            mesh = pv.read("case.foam")
            if hasattr(mesh, "__getitem__") and len(mesh) > 0:
                mesh = mesh[0]
            return np.array(mesh.cell_centers().points)
    except Exception:
        pass
    return None

def _read_0_orig_internal(zero_dir, name, n_cells, is_vector=False):
    path = os.path.join(zero_dir, name)
    if not os.path.isfile(path):
        return None
    arr, default = _parse_internal_field(path, is_vector=is_vector)
    if arr is not None and len(arr) == n_cells:
        return np.asarray(arr)
    if is_vector:
        return np.full((n_cells, 3), default)
    return np.full(n_cells, default)

def _read_bc(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    if "boundaryField" not in text:
        return ""
    start = text.find("boundaryField")
    end = text.rfind("}}")
    if start == -1 or end == -1:
        return ""
    return text[start:end+1]

def fast_write(path, name, dim, arr, bc, is_vector=False):
    n = len(arr)
    cls = "volVectorField" if is_vector else "volScalarField"
    with open(path, "w") as f:
        f.write("FoamFile\n{{ version 2.0; format ascii; class " + cls + "; object " + name + "; }}\n\n")
        if is_vector:
            f.write("dimensions [0 1 -1 0 0 0 0];\n\ninternalField nonuniform List<vector>\n%d\n(\n" % n)
            buf = io.StringIO()
            np.savetxt(buf, np.asarray(arr).reshape(-1, 3), fmt=" (%.10e %.10e %.10e)")
            f.write(buf.getvalue())
        else:
            f.write("dimensions " + dim + ";\n\ninternalField nonuniform List<scalar>\n%d\n(\n" % n)
            buf = io.StringIO()
            np.savetxt(buf, np.asarray(arr).reshape(-1, 1), fmt=" %.10e")
            f.write(buf.getvalue())
        f.write(");\n\n")
        f.write(bc + "\n")

def main():
    # Robust time directory resolution
    time_dir, diag = _resolve_time_dir(SOURCE_1D_CASE, SOURCE_TIME)
    
    if not time_dir:
        print("remap_radial: FATAL - 1D time directory not found", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("  SOURCE_1D_CASE:     %s" % SOURCE_1D_CASE, file=sys.stderr)
        print("  SOURCE_TIME (raw):  %s" % repr(SOURCE_TIME), file=sys.stderr)
        print("  Case exists:        %s" % os.path.isdir(SOURCE_1D_CASE), file=sys.stderr)
        if diag.get("candidates"):
            print("  Available times:    %s" % diag["candidates"], file=sys.stderr)
        else:
            print("  Available times:    (none found)", file=sys.stderr)
        if diag.get("error"):
            print("  Error:              %s" % diag["error"], file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("", file=sys.stderr)
        print("RESOLUTION HINTS:", file=sys.stderr)
        print("  1. Ensure the 1D case has been run and time directories exist.", file=sys.stderr)
        print("  2. Check SOURCE_1D_CASE path is accessible (WSL: use /mnt/c/... or /home/...).", file=sys.stderr)
        if diag.get("candidates"):
            print("  3. Try SOURCE_TIME='latest' or one of: %s" % diag["candidates"][:5], file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)
    
    # Success: print resolution info
    if diag["method"] == "exact_string":
        print("remap_radial: Resolved SOURCE_TIME '%s' -> '%s' (exact string match)" % (SOURCE_TIME, time_dir), file=sys.stderr)
    elif diag["method"] == "exact_numeric":
        print("remap_radial: Resolved SOURCE_TIME '%s' (%.6g) -> '%s' (%.6g) [exact numeric]" % (
            SOURCE_TIME, diag["requested_val"], time_dir, diag["resolved_val"]), file=sys.stderr)
    elif diag["method"] == "closest":
        print("remap_radial: WARNING - No exact match for SOURCE_TIME '%s' (%.6g)" % (SOURCE_TIME, diag["requested_val"]), file=sys.stderr)
        print("remap_radial: Using closest time: '%s' (%.6g) [diff=%.3g]" % (
            time_dir, diag["resolved_val"], abs(diag["resolved_val"] - diag["requested_val"])), file=sys.stderr)
    elif diag["method"] == "latest":
        print("remap_radial: Resolved SOURCE_TIME 'latest' -> '%s'" % time_dir, file=sys.stderr)
    data_1d = _read_1d_data(SOURCE_1D_CASE, time_dir)
    if not data_1d:
        print("remap_radial: failed to read 1D data", file=sys.stderr)
        sys.exit(1)
    r_1d = data_1d["r"]
    C = _read_3d_cell_centres()
    if C is None or len(C) == 0:
        print("remap_radial: run postProcess -func writeCellCentres (or provide case.foam + PyVista)", file=sys.stderr)
        sys.exit(1)
    R_vec = C - ORIGIN
    R = np.linalg.norm(R_vec, axis=1)
    R = np.maximum(R, 1e-20)
    n_cells = len(R)
    zero_dir = "0.orig" if os.path.isdir("0.orig") else "0"
    R_remap = float(np.max(r_1d))
    mask = R <= R_remap
    inside_count = int(np.sum(mask))
    print("remap_radial: R_remap=%.6g, r_min=%.6g r_max=%.6g, cells_inside=%d total=%d" % (R_remap, float(np.min(R)), float(np.max(R)), inside_count, n_cells), file=sys.stderr)
    if inside_count >= n_cells:
        print("remap_radial: WARNING all cells inside R_remap; check 1D domain extent", file=sys.stderr)
    if os.environ.get("REMAP_ASSERT_OUTSIDE") and inside_count == n_cells:
        print("remap_radial: REMAP_ASSERT_OUTSIDE set but inside_count==total_cells; abort", file=sys.stderr)
        sys.exit(1)
    order = np.argsort(r_1d)
    r_sorted = np.asarray(r_1d)[order]
    p_mapped = np.interp(R, r_sorted, np.asarray(data_1d["p"])[order])
    T_mapped = np.interp(R, r_sorted, np.asarray(data_1d["T"])[order])
    rho4_mapped = np.interp(R, r_sorted, np.asarray(data_1d["rho.c4"])[order])
    rhoa_mapped = np.interp(R, r_sorted, np.asarray(data_1d["rho.air"])[order])
    a4_mapped = np.interp(R, r_sorted, np.asarray(data_1d["alpha.c4"])[order])
    U_mag_mapped = np.interp(R, r_sorted, np.asarray(data_1d["U_mag"])[order])
    R_hat = R_vec / R[:, np.newaxis]
    U_mapped = U_mag_mapped[:, np.newaxis] * R_hat
    p_orig = _read_0_orig_internal(zero_dir, "p", n_cells)
    if p_orig is None:
        p_orig = np.full(n_cells, 101325.0)
    T_orig = _read_0_orig_internal(zero_dir, "T", n_cells)
    if T_orig is None:
        T_orig = np.full(n_cells, 300.0)
    rho4_orig = _read_0_orig_internal(zero_dir, "rho.c4", n_cells)
    if rho4_orig is None:
        rho4_orig = np.full(n_cells, 1600.0)
    rhoa_orig = _read_0_orig_internal(zero_dir, "rho.air", n_cells)
    if rhoa_orig is None:
        rhoa_orig = np.full(n_cells, 1.225)
    a4_orig = _read_0_orig_internal(zero_dir, "alpha.c4", n_cells)
    if a4_orig is None:
        a4_orig = np.zeros(n_cells)
    U_orig = _read_0_orig_internal(zero_dir, "U", n_cells, is_vector=True)
    if U_orig is None:
        U_orig = np.zeros((n_cells, 3))
    p_3d = np.where(mask, p_mapped, p_orig)
    T_3d = np.where(mask, T_mapped, T_orig)
    rho4_3d = np.where(mask, rho4_mapped, rho4_orig)
    rhoa_3d = np.where(mask, rhoa_mapped, rhoa_orig)
    a4_3d = np.where(mask, a4_mapped, a4_orig)
    U_3d = np.where(mask[:, np.newaxis], U_mapped, U_orig)
    # Always zero explosive phase: with activationModel none the blast wave
    # is carried entirely by p/U/T in the air phase; any non-zero alpha.c4
    # or rho.c4 creates an inconsistent thermodynamic state (sigFpe).
    a4_3d = np.zeros(n_cells)
    rho4_3d = np.zeros(n_cells)
    # Debug: sample cells outside R_remap to confirm they get original values
    outside_idx = np.where(~mask)[0]
    if len(outside_idx) > 0:
        n_sample = min(5, len(outside_idx))
        sample_idx = np.random.choice(outside_idx, size=n_sample, replace=False)
        for idx in sample_idx:
            print("remap_radial: outside sample cell %d r=%.6g R_remap=%.6g p_3d=%.6g p_orig=%.6g (match=%s)" % (
                idx, float(R[idx]), R_remap, float(p_3d[idx]), float(p_orig[idx]),
                "yes" if np.isclose(p_3d[idx], p_orig[idx]) else "NO"), file=sys.stderr)
    os.makedirs(OUT_DIR, exist_ok=True)
    if POST_DETONATION:
        rho_3d = rhoa_3d
        h_3d = 1005.0 * T_3d
        for name, dim, arr in [
            ("p", "[1 -1 -2 0 0 0 0]", p_3d),
            ("T", "[0 0 0 1 0 0 0]", T_3d),
            ("rho", "[1 -3 0 0 0 0 0]", rho_3d),
            ("h", "[0 2 -2 0 0 0 0]", h_3d),
        ]:
            bc_path = os.path.join(zero_dir, name)
            bc = _read_bc(bc_path) if os.path.isfile(bc_path) else ""
            if not bc:
                bc = "boundaryField {{ }}"
            fast_write(os.path.join(OUT_DIR, name), name, dim, arr, bc, is_vector=False)
        u_bc = _read_bc(os.path.join(zero_dir, "U"))
        fast_write(os.path.join(OUT_DIR, "U"), "U", "", U_3d, u_bc, is_vector=True)
        print("remap_radial: wrote %s/p, T, U, rho, h (inside r<=R_remap=%g: %d cells)" % (OUT_DIR, R_remap, inside_count))
    else:
        for name, dim, arr in [
            ("p", "[1 -1 -2 0 0 0 0]", p_3d),
            ("T", "[0 0 0 1 0 0 0]", T_3d),
            ("rho.c4", "[1 -3 0 0 0 0 0]", rho4_3d),
            ("rho.air", "[1 -3 0 0 0 0 0]", rhoa_3d),
            ("alpha.c4", "[0 0 0 0 0 0 0]", a4_3d),
        ]:
            fast_write(os.path.join(OUT_DIR, name), name, dim, arr, _read_bc(os.path.join(zero_dir, name)), is_vector=False)
        fast_write(os.path.join(OUT_DIR, "U"), "U", "", U_3d, _read_bc(os.path.join(zero_dir, "U")), is_vector=True)
        print("remap_radial: wrote %s/p, T, U, rho.c4, rho.air, alpha.c4 (inside r<=R_remap=%g: %d cells)" % (OUT_DIR, R_remap, inside_count))

if __name__ == "__main__":
    main()
'''


class Generator3D(BaseGenerator):
    def __init__(self, base_path: str, openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc"):
        super().__init__(base_path)
        self.openfoam_bashrc = openfoam_bashrc
        self._charge_clipped_by_domain = False  # Set by validate; used for case_init_mode.json
        self._charge_warnings = []  # List of warning strings for Info panel
        self._last_charge_capture_meta: Optional[dict] = None
        self._last_charge_refinement_dict_level: int = 0

    def _validate_inputs(self, inputs: CaseInputs3D) -> None:
        """Validate geometry and physical inputs; raise ValueError with clear message on failure."""
        min_p = inputs.min_point
        max_p = inputs.max_point
        if min_p[0] >= max_p[0]:
            raise ValueError("Domain: Xmin must be less than Xmax.")
        if min_p[1] >= max_p[1]:
            raise ValueError("Domain: Ymin must be less than Ymax.")
        if min_p[2] >= max_p[2]:
            raise ValueError("Domain: Zmin must be less than Zmax.")
        if inputs.cell_size <= 0:
            raise ValueError("Cell size must be positive.")
        if inputs.mass_kg <= 0:
            raise ValueError("Charge mass must be positive.")
        if inputs.rho_charge <= 0:
            raise ValueError("Charge density must be positive.")
        cx, cy, cz = inputs.charge_center
        if not (min_p[0] <= cx <= max_p[0] and min_p[1] <= cy <= max_p[1] and min_p[2] <= cz <= max_p[2]):
            raise ValueError("Charge center must lie inside the domain bounds.")

    def generate(self, case_name: str, inputs: CaseInputs3D) -> str:
        """Generate 3D OpenFOAM case. Returns case_dir.
        
        """
        self._charge_clipped_by_domain = False
        self._charge_warnings = []
        
        try:
            self._validate_inputs(inputs)
        except ValueError as e:
            raise ValueError(f"Invalid 3D inputs: {e}") from e

        try:
            case_dir = self.create_case_dirs(case_name)
            dims = self._calculate_charge_dimensions(inputs)
            obstacle_patch_names = self._get_obstacle_patch_names(inputs)

            remap_enabled = getattr(inputs, "remap_enabled", False)
            mapped_source_dir_linux = None
            mapped_source_time = None
            remap_case_path = getattr(inputs, "remap_case_path", "") or ""
            if remap_enabled and remap_case_path.strip():
                source_case_dir_win, source_time_from_path = _split_source_case_and_time(remap_case_path)
                if source_case_dir_win:
                    mapped_source_dir_linux = win_to_wsl_path(source_case_dir_win)
                    remap_time_mode = getattr(inputs, "remap_time_mode", "latest") or "latest"
                    if remap_time_mode == "latest":
                        latest = get_latest_time_dir(source_case_dir_win)
                        mapped_source_time = latest or source_time_from_path or getattr(inputs, "remap_specific_time", None) or "1e-4"
                    else:
                        mapped_source_time = getattr(inputs, "remap_specific_time", None) or source_time_from_path or "1e-4"

            has_obstacles = bool(getattr(inputs, "obstacles", None))
            if has_obstacles:
                self._setup_obstacles(case_dir, inputs, dims)
                self._write_change_dictionary(case_dir)
            else:
                self._write_snappy_charge_only(case_dir, inputs, dims)
                self._write_change_dictionary(case_dir)

            self._write_block_mesh_3d(case_dir, inputs)
            self._write_initial_conditions_3d(case_dir, inputs, obstacle_patch_names if has_obstacles else None)
            self._write_constant_files_3d(case_dir, inputs, dims)
            remap_start_time = mapped_source_time if remap_enabled else None
            # Single source of truth for startup charge-region refinement: Charge pre-refinement UI (charge_refinement_level).
            # Only in Dyn Mesh mode and when charge pre-refinement is requested (level > 0).
            enable_dyn = getattr(inputs, "enable_dyn_refine", None)
            charge_level = getattr(inputs, "charge_refinement_level", 0) or 0
            startup_charge_refine = (
                not remap_enabled
                and enable_dyn
                and charge_level > 0
            )
            use_set_refined = (
                startup_charge_refine
                and getattr(inputs, "charge_shape", "") in ("Sphere", "Cylinder", "Cuboid")
            )
            self._write_system_files_3d(case_dir, inputs, dims, remap_start_time=remap_start_time, use_set_refined_allrun=use_set_refined, use_seed_bubble=False)

            if remap_enabled and mapped_source_dir_linux and mapped_source_time:
                remap_origin = getattr(inputs, "remap_origin", (0.0, 0.0, 0.0))
                self._write_remap_radial_script(case_dir, mapped_source_dir_linux, mapped_source_time, remap_origin)

            solver_app = "blastFoam"
            if not remap_enabled:
                self._write_charge_region_files(case_dir, inputs, dims)
            self._write_check_internal_patch_sh(case_dir)
            self.write_scripts(
                case_dir,
                self.openfoam_bashrc,
                use_snappy=True,
                cores=inputs.cores,
                init_from_1d=remap_enabled and bool(mapped_source_dir_linux and mapped_source_time),
                mapped_source_dir_linux=mapped_source_dir_linux,
                mapped_source_time=mapped_source_time,
                solver_app=solver_app,
                use_set_refined=use_set_refined,
                use_seed_bubble=False,
                startup_refinement_levels=0,
                remaining_inside_levels=0,
                # Charge-region post-init check (writeCellCentres + check_charge_region.py)
                # is a verification step. In fast_run_mode it is skipped to save ~1-2 s
                # of postProcess + Python work; check_alpha_c4.sh still gates the solver.
                use_charge_region_check=(not remap_enabled) and (not getattr(inputs, "fast_run_mode", True)),
                # Native setRefinedFields path: no manual topoSet/refineMesh
                # capture/charge stages → use_charge_interior_refinement=False
                # ensures Allrun runs ``setRefinedFields`` (not ``-noRefine``).
                use_charge_interior_refinement=False,
                inside_levels=getattr(inputs, "charge_refinement_level", 0) or 0,
                capture_levels=0,
                charge_levels=0,
                outside_levels=0,
                charge_capture_impossible_message=None,
                envelope_empty_message=None,
                charge_region_empty_message=None,
                placement_use_setfields=use_set_refined and getattr(inputs, "charge_shape", "") == "Cuboid",
                fast_run_mode=getattr(inputs, "fast_run_mode", True),
            )

            # 3D non-remap only: export init mode and effective values for GUI Info panel
            if not remap_enabled:
                import json
                mode_path = os.path.join(case_dir, "case_init_mode.json")
                ign_rad = getattr(self, "_last_initiation_radius_effective", 0.05)
                ign_rad_req = getattr(self, "_last_initiation_radius_requested", 0.05)
                obs_refine = inputs.obstacles[0].refinement_level if inputs.obstacles else 0
                placement_use_setfields = use_set_refined and getattr(inputs, "charge_shape", "") == "Cuboid"
                cr_req = int(getattr(inputs, "charge_refinement_level", 0) or 0)
                cr_dict = int(getattr(self, "_last_charge_refinement_dict_level", cr_req))
                mode = {
                    "set_cmd": "setFields" if placement_use_setfields else ("setRefinedFields" if use_set_refined else "setFields"),
                    "fallback_reason": None,
                    "initiation_radius_requested": ign_rad_req,
                    "initiation_radius_effective": ign_rad,
                    "smallest_cell_near_charge": getattr(self, "_last_smallest_cell_near_charge", None),
                    "charge_refinement_requested": cr_req,
                    "charge_refinement_effective": cr_dict if use_set_refined else 0,
                    "obstacle_refinement": obs_refine,
                    "snappy_refinement": has_obstacles,
                    "charge_clipped_by_domain": getattr(self, "_charge_clipped_by_domain", False),
                    "charge_warnings": getattr(self, "_charge_warnings", []),
                    "cells_inside_charge": None,
                    "cells_in_ignition_region": None,
                    "expected_eMesh": self._expected_emesh_list(case_dir),
                    "retries_used": 0,
                }
                # Native setRefinedFields path: refineInternal handles inner
                # charge refinement; no manual topoSet/refineMesh stages.
                mode["startup_refinement_levels"] = 0
                mode["remaining_inside_levels"] = 0
                mode["use_charge_interior_refinement"] = False
                mode["inside_levels"] = getattr(inputs, "charge_refinement_level", 0) or 0
                mode["capture_levels"] = 0
                mode["charge_levels"] = 0
                mode["outside_levels"] = 0
                mode["charge_capture_impossible_message"] = None
                mode["envelope_empty_message"] = None
                mode["charge_region_empty_message"] = None
                mode["base_cell_size"] = float(getattr(inputs, "cell_size", 0.5))
                mode["charge_shape"] = getattr(inputs, "charge_shape", "Sphere")
                mode["charge_size_info"] = self._charge_size_info(inputs, dims)
                mode["user_requested_inside"] = cr_req
                mode["charge_capture"] = getattr(self, "_last_charge_capture_meta", None) or {}
                mode["snappy_charge_pre_level"] = 0
                mode["internal_capture_levels"] = 0
                mode["internal_charge_levels"] = 0
                mode["attempted_refine_iterations"] = 0
                mode["realized_refinement_level"] = None
                mode["smallest_cell_near_charge"] = None
                mode["cells_inside_charge_post_refine"] = None
                mode["cells_in_ignition_region"] = None
                mode["realization_status"] = "not_run"
                mode["realization_message"] = None
                self._write_text(mode_path, json.dumps(mode, indent=2))

            # Create case.foam for ParaView compatibility
            import pathlib
            pathlib.Path(case_dir, "case.foam").touch()
            
            return case_dir
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write case files: {e}") from e
        except ValueError as e:
            raise
        except Exception as e:
            raise RuntimeError(f"3D case generation failed: {e}") from e

    def _write_remap_radial_script(
        self, case_dir: str, source_case_linux: str, source_time: str, origin: tuple,
    ) -> None:
        """Write remap_radial.py into the case root for Autodyn-style radial remap from 1D."""
        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
        script = _REMAP_RADIAL_SCRIPT.format(
            source_case=repr(source_case_linux),
            source_time=repr(source_time),
            origin_x=ox, origin_y=oy, origin_z=oz,
            post_detonation=repr(False),
        )
        self._write_text(os.path.join(case_dir, "remap_radial.py"), script)

    def _write_change_dictionary(self, case_dir: str):
        content = self._foam_header("changeDictionaryDict", "dictionary", "system")
        # OF9: after addEmptyPatch, 0/ must have boundaryField for ALL patches including internalPatch.
        # Patch type "internal" (addEmptyPatch) requires patchField type "internal", not "calculated".
        content += """
U { boundaryField { internalPatch { type internal; } "obs.*" { type fixedValue; value uniform (0 0 0); } } }
T { boundaryField { internalPatch { type internal; } "obs.*" { type zeroGradient; } } }
p { boundaryField { internalPatch { type internal; } "obs.*" { type zeroGradient; } } }
"alpha.c4" { boundaryField { internalPatch { type internal; } "obs.*" { type zeroGradient; } } }
"rho.c4" { boundaryField { internalPatch { type internal; } "obs.*" { type zeroGradient; } } }
"rho.air" { boundaryField { internalPatch { type internal; } "obs.*" { type zeroGradient; } } }
"""
        self._write_text(os.path.join(case_dir, "system", "changeDictionaryDict"), content)

    def _write_surface_features_dict(self, case_dir: str, stl_names: list, inputs=None):
        # Filename MUST be surfaceFeaturesDict (pipeline calls surfaceFeatures without -dict). FoamFile object for OF9 compatibility.
        content = self._foam_header("surfaceFeatureExtractDict", "dictionary", "system")
        content += "\nsurfaces\n(\n"
        for name in stl_names:
            content += f'    "{name}"\n'
        content += ");\n\n"
        inc_angle = getattr(inputs, "mesh_included_angle", None) or getattr(inputs, "obstacle_feature_angle", 120) if inputs else 120
        content += f"includedAngle   {inc_angle};\n\n"
        content += "subsetFeatures\n{\n    nonManifoldEdges yes;\n    openEdges yes;\n}\n"
        self._write_text(os.path.join(case_dir, "system", "surfaceFeaturesDict"), content)

    def _snappy_charge_pre_level(self, inputs: CaseInputs3D, dims: Optional[Dict[str, float]] = None) -> int:
        """Return snappy inside pre-level for charge geometry.

        We keep this at 0 so inner charge refinement is handled only by explicit
        topoSet/refineMesh stages. This avoids mixed inside levels caused by combining
        snappy inside refinement with capture/charge loops.
        """
        return 0

    def _snappy_charge_geometry_block(self, inputs: CaseInputs3D, dims: Dict[str, float]) -> str:
        """Return a snappy geometry sub-block (geometry{} section) for the charge refinement
        region, named 'chargeRegionSnappy'.  Used by refinementRegions pre-refinement.

        Uses the TRUE charge geometry (no envelope inflation).  Capturability on the base mesh
        must be validated by _is_charge_capturable_on_base_mesh() before calling this.
        """
        shape = getattr(inputs, "charge_shape", "Sphere")
        cx, cy, cz = inputs.charge_center
        if shape == "Sphere":
            r = float(dims.get("radius", 0.05))
            return (f"    chargeRegionSnappy\n    {{\n"
                    f"        type searchableSphere;\n"
                    f"        centre ({cx:.6g} {cy:.6g} {cz:.6g});\n"
                    f"        radius {r:.6g};\n"
                    f"    }}\n")
        if shape == "Cylinder":
            r = float(dims.get("radius", 0.05))
            length = float(dims.get("length", 0.1))
            half_l = length / 2.0
            p1 = [cx, cy, cz]
            p2 = [cx, cy, cz]
            idx = {"X": 0, "Y": 1, "Z": 2}.get(getattr(inputs, "cylinder_axis", "Z"), 2)
            p1[idx] -= half_l
            p2[idx] += half_l
            return (f"    chargeRegionSnappy\n    {{\n"
                    f"        type searchableCylinder;\n"
                    f"        point1 ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g});\n"
                    f"        point2 ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g});\n"
                    f"        radius {r:.6g};\n"
                    f"    }}\n")
        if shape == "Cuboid":
            if "length" in dims and "width" in dims and "height" in dims:
                hx = dims["length"] / 2.0
                hy = dims["width"] / 2.0
                hz = dims["height"] / 2.0
            else:
                side = dims.get("side", 0.1)
                hx = hy = hz = side / 2.0
            return (f"    chargeRegionSnappy\n    {{\n"
                    f"        type searchableBox;\n"
                    f"        min ({cx-hx:.6g} {cy-hy:.6g} {cz-hz:.6g});\n"
                    f"        max ({cx+hx:.6g} {cy+hy:.6g} {cz+hz:.6g});\n"
                    f"    }}\n")
        # Fallback: sphere with true radius
        r = float(dims.get("radius", 0.05))
        return (f"    chargeRegionSnappy\n    {{\n"
                f"        type searchableSphere;\n"
                f"        centre ({cx:.6g} {cy:.6g} {cz:.6g});\n"
                f"        radius {r:.6g};\n"
                f"    }}\n")

    def _charge_outer_refine_levels(self, inputs: CaseInputs3D) -> tuple:
        """Return (add_charge_outer: bool, level_str: Optional[str]).

        Computes outer transition refinement levels independently of the inner (Inside) setting.
        When Inside > 0, both chargeRegionSnappy (inner) and chargeRefineOuter (outer) can
        coexist in snappy refinementRegions — snappy takes the per-cell maximum, so the inner
        region reaches inside_level and the surrounding band reaches outer_max.
        Fixed Mesh: no charge outer refinement.
        """
        if getattr(inputs, "enable_dyn_refine", None) is False:
            return False, None
        enable = getattr(inputs, "charge_outer_refine_enable", None)
        if enable is False:
            return False, None
        rmin_outer = getattr(inputs, "charge_outer_refine_min", None)
        rmax_outer = getattr(inputs, "charge_outer_refine_max", None)
        # Use charge_outer_* when explicitly set (None = fallback to global refine_min/refine_max); 0 is valid
        rmin = rmin_outer if rmin_outer is not None else getattr(inputs, "refine_min", 2)
        rmax = rmax_outer if rmax_outer is not None else getattr(inputs, "refine_max", 3)
        if rmin == 0 and rmax == 0:
            return False, None
        if not getattr(inputs, "enable_local_refinement", True):
            return True, "2 2"
        rmax = max(rmin, rmax)
        return True, f"{rmin} {rmax}"

    def _charge_outer_geometry(self, inputs: CaseInputs3D, dims: Dict[str, float]) -> tuple:
        """Return (cx, cy, cz, outer_radius) for snappy charge refinement region.

        The inner seed uses ``bubble_radius_factor * R_charge`` only for the **outer transition
        band** (chargeRefineOuter). This is not the setRefinedFields charge capture radius.
        """
        cx, cy, cz = inputs.charge_center
        r = dims.get("radius", 0.05)
        factor = max(0.5, min(5.0, getattr(inputs, "bubble_radius_factor", 1.5)))
        seed_radius = r * factor
        cell_size = max(getattr(inputs, "cell_size", 0.1), 1e-9)
        n_cbl = max(1, min(10, getattr(inputs, "transition_cells", 2)))
        # Outside levels (for band width scaling)
        rmin_outer = getattr(inputs, "charge_outer_refine_min", None)
        rmax_outer = getattr(inputs, "charge_outer_refine_max", None)
        rmin = rmin_outer if rmin_outer is not None else getattr(inputs, "refine_min", 2)
        rmax = rmax_outer if rmax_outer is not None else getattr(inputs, "refine_max", 3)
        level_span = max(1, rmax - rmin)
        # outer_radius = seed_radius + transition_cells * level_span * cell_size (deterministic, no plateau mismatch)
        outer_radius = seed_radius + n_cbl * level_span * cell_size
        return (cx, cy, cz, outer_radius)

    def _location_in_mesh_point(self, inputs: CaseInputs3D) -> tuple:
        """Return a point strictly inside the fluid domain for locationInMesh (snappyHexMesh).
        Nudges charge_center inward if it lies on any domain boundary to avoid empty-mesh failures."""
        lx, ly, lz = inputs.charge_center
        min_x, min_y, min_z = inputs.min_point
        max_x, max_y, max_z = inputs.max_point
        dx = max(getattr(inputs, "cell_size", 0.1), 1e-9)
        margin = dx * 1.1
        if abs(lz - min_z) < 1e-5:
            lz += margin
        elif abs(lz - max_z) < 1e-5:
            lz -= margin
        if abs(lx - min_x) < 1e-5:
            lx += margin
        elif abs(lx - max_x) < 1e-5:
            lx -= margin
        if abs(ly - min_y) < 1e-5:
            ly += margin
        elif abs(ly - max_y) < 1e-5:
            ly -= margin
        return (lx, ly, lz)

    def _setup_obstacles(self, case_dir: str, inputs: CaseInputs3D, dims: Optional[Dict[str, float]] = None):
        tri_surface_dir = os.path.join(case_dir, "constant", "triSurface")
        os.makedirs(tri_surface_dir, exist_ok=True)
        
        stl_names = []
        stl_base_names = []
        
        for i, obs in enumerate(inputs.obstacles):
            safe_name = f"obs{i}_{os.path.splitext(os.path.basename(obs.stl_path))[0]}.stl"
            base_name = safe_name[:-4]
            dst = os.path.join(tri_surface_dir, safe_name)
            try:
                if pv:
                    try:
                        mesh = pv.read(obs.stl_path)
                        if abs(obs.scale - 1.0) > 1e-6: mesh.scale([obs.scale]*3, inplace=True)
                        mesh.translate([obs.offset_x, obs.offset_y, obs.offset_z], inplace=True)
                        mesh.save(dst, binary=True)
                    except Exception as pv_err:
                        print(f"PyVista failed for {obs.stl_path}: {pv_err}; falling back to file copy.")
                        shutil.copy2(obs.stl_path, dst)
                else:
                    shutil.copy2(obs.stl_path, dst)
                stl_names.append(safe_name)
                stl_base_names.append(base_name)
            except Exception as e:
                raise RuntimeError(f"Cannot copy obstacle STL {obs.stl_path}: {e}") from e

        if not stl_names: return

        self._write_surface_features_dict(case_dir, stl_names, inputs)
        # Canonical path: constant/extendedFeatureEdgeMesh/<base>.extendedFeatureEdgeMesh
        # (OF9 surfaceFeatures outputs .extendedFeatureEdgeMesh, not .eMesh)
        # Force Unix line endings so bash 'while read' in Allrun does not pick up \r from CRLF.
        expected = os.path.join(case_dir, "system", "expectedFeatureEdges.txt")
        with open(expected, "w", encoding="utf-8", newline="\n") as f:
            for base_name in stl_base_names:
                f.write("constant/extendedFeatureEdgeMesh/" + base_name + ".extendedFeatureEdgeMesh\n")

        loc_pt = self._location_in_mesh_point(inputs)
        prov = getattr(inputs, "provenance", {}) or {}
        enable_obs = getattr(inputs, "enable_obstacle_refine", None)
        if prov.get("enable_obstacle_refine") == "UNSET":
            enable_obs = False  # do not add refinement when loaded case did not set it
        elif enable_obs is None:
            enable_obs = getattr(inputs, "enable_local_refinement", True)
        if enable_obs:
            rmin = getattr(inputs, "obstacle_refine_min", None) or getattr(inputs, "refine_min", 2)
            rmax = getattr(inputs, "obstacle_refine_max", None) or getattr(inputs, "refine_max", 3)
            rmax = max(rmin, rmax)
            ref_level = f"{rmin} {rmax}"
            feat_level = str(rmin)
        else:
            ref_level = "0 0"
            feat_level = "0"
        resolve_angle = getattr(inputs, "mesh_resolve_feature_angle", None) or getattr(inputs, "obstacle_feature_angle", 30)

        snappy = self._foam_header("snappyHexMeshDict", "dictionary", "system") + "\n"
        snappy += "castellatedMesh on;\nsnap on;\naddLayers off;\n\n"
        
        snappy += "geometry\n{\n"
        for idx, name in enumerate(stl_names):
            clean_name = name.replace(".", "_")
            # If STL was not pre-scaled by PyVista, add scale keyword for snappy
            obs = inputs.obstacles[idx] if idx < len(inputs.obstacles) else None
            obs_scale = obs.scale if obs and abs(obs.scale - 1.0) > 1e-6 else None
            if obs_scale is not None and not pv:
                # PyVista not available → snappy must apply the scale
                snappy += f'    "{clean_name}" {{ type triSurfaceMesh; file "{name}"; scale {obs_scale}; }}\n'
            else:
                snappy += f'    "{clean_name}" {{ type triSurfaceMesh; file "{name}"; }}\n'
        # Inner charge geometry (Inside > 0) and outer transition sphere are independent and additive.
        # snappy takes the per-cell maximum across all refinementRegions, so inner cells reach
        # inside_level and cells in the surrounding band reach outer_max.
        _snappy_pre_lv = self._snappy_charge_pre_level(inputs, dims) if dims else 0
        add_charge_outer, level_str = self._charge_outer_refine_levels(inputs)
        if _snappy_pre_lv > 0 and dims:
            snappy += self._snappy_charge_geometry_block(inputs, dims)
        if add_charge_outer and level_str and dims:
            cx, cy, cz, outer_rad = self._charge_outer_geometry(inputs, dims)
            snappy += f'    chargeRefineOuter {{ type searchableSphere; centre ({cx:.6g} {cy:.6g} {cz:.6g}); radius {outer_rad:.6g}; }}\n'
        snappy += "};\n\n"

        n_cbl = max(1, min(10, getattr(inputs, "transition_cells", 2)))
        snappy += "castellatedMeshControls\n{\n"
        snappy += f"    maxLocalCells 10000000;\n    maxGlobalCells 200000000;\n    minRefinementCells 2;\n    maxLoadUnBalance 0.1;\n    nCellsBetweenLevels {n_cbl};\n"
        
        # OF9 snappyHexMesh searches for feature files in constant/extendedFeatureEdgeMesh/;
        # use just the filename (matches building3D reference format).
        snappy += "    features\n    (\n"
        for base_name in stl_base_names:
            snappy += f'        {{ file "{base_name}.extendedFeatureEdgeMesh"; level {feat_level}; }}\n'
        snappy += "    );\n"
        
        snappy += "    refinementSurfaces\n    {\n"
        for name in stl_names:
            clean_name = name.replace(".", "_")
            snappy += f'        "{clean_name}" {{ level ({ref_level}); patchInfo {{ type wall; }} }}\n'
        snappy += "    }\n"
        
        _ref_regions_obs = ""
        if _snappy_pre_lv > 0:
            _ref_regions_obs += f"        chargeRegionSnappy {{ mode inside; levels (({_snappy_pre_lv} {_snappy_pre_lv})); }}\n"
        if add_charge_outer and level_str:
            parts = level_str.strip().split()
            if len(parts) >= 2:
                rmin_r, rmax_r = int(parts[0]), int(parts[1])
                _ref_regions_obs += f"        chargeRefineOuter {{ mode inside; levels (({rmin_r} {rmax_r})); }}\n"
        if _ref_regions_obs:
            snappy += f"    refinementRegions\n    {{\n{_ref_regions_obs}    }}\n"
        else:
            snappy += "    refinementRegions {\n    }\n"
        snappy += f"    resolveFeatureAngle {resolve_angle};\n    locationInMesh ({loc_pt[0]:.6f} {loc_pt[1]:.6f} {loc_pt[2]:.6f});\n    allowFreeStandingZoneFaces false;\n}}\n\n"
        n_solve = getattr(inputs, "mesh_n_solve_iter", None) or getattr(inputs, "obstacle_snap_iter", 100)
        n_feat_snap = getattr(inputs, "mesh_n_feature_snap_iter", None) or getattr(inputs, "obstacle_feature_snap_iter", 15)
        n_smooth_patch = getattr(inputs, "mesh_n_smooth_patch", None) if getattr(inputs, "mesh_n_smooth_patch", None) is not None else 3
        snap_tol = getattr(inputs, "mesh_snap_tolerance", None) if getattr(inputs, "mesh_snap_tolerance", None) is not None else 1.0
        n_relax = getattr(inputs, "mesh_n_relax_iter", None) if getattr(inputs, "mesh_n_relax_iter", None) is not None else 10
        expl = getattr(inputs, "mesh_explicit_feature_snap", None)
        expl = expl if expl is not None else False
        impl = getattr(inputs, "mesh_implicit_feature_snap", None)
        impl = impl if impl is not None else True
        multi = getattr(inputs, "mesh_multi_region_feature_snap", None)
        multi = multi if multi is not None else False
        snappy += ("snapControls\n{\n"
                   f"    nSmoothPatch {n_smooth_patch};\n"
                   f"    tolerance {snap_tol};\n"
                   f"    nSolveIter {n_solve};\n"
                   f"    nRelaxIter {n_relax};\n"
                   f"    nFeatureSnapIter {n_feat_snap};\n"
                   f"    implicitFeatureSnap {'true' if impl else 'false'};\n"
                   f"    explicitFeatureSnap {'true' if expl else 'false'};\n"
                   f"    multiRegionFeatureSnap {'true' if multi else 'false'};\n"
                   "}\n\n")
        
        snappy += ("addLayersControls\n{\n"
                   "    featureAngle 100;\n"          # building3D: 100 (was 130)
                   "    slipFeatureAngle 30;\n"
                   "    nLayerIter 50;\n"
                   "    nRelaxedIter 20;\n"           # building3D: 20 (was missing)
                   "    nRelaxIter 5;\n"              # building3D: 5 (was 3)
                   "    nGrow 0;\n"
                   "    nSmoothSurfaceNormals 1;\n"
                   "    nSmoothNormals 3;\n"
                   "    nSmoothThickness 10;\n"
                   "    maxFaceThicknessRatio 0.5;\n"
                   "    maxThicknessToMedialRatio 0.3;\n"
                   "    minMedialAxisAngle 90;\n"
                   "    nBufferCellsNoExtrude 0;\n"
                   "    layers {}\n"
                   "    relativeSizes true;\n"
                   "    expansionRatio 1.2;\n"
                   "    finalLayerThickness 0.5;\n"
                   "    minThickness 1e-3;\n"
                   "}\n")
        
        def _mq(k: str, default):
            v = getattr(inputs, k, None)
            return v if v is not None else default
        snappy += ("meshQualityControls\n{\n"
                   f"    maxNonOrtho {_mq('mesh_max_non_ortho', 65)};\n"
                   f"    maxBoundarySkewness {_mq('mesh_max_boundary_skewness', 20)};\n"
                   f"    maxInternalSkewness {_mq('mesh_max_internal_skewness', 4)};\n"
                   f"    maxConcave {_mq('mesh_max_concave', 80)};\n"
                   f"    minVol {_mq('mesh_min_vol', 1e-13)};\n"
                   f"    minTetQuality {_mq('mesh_min_tet_quality', 1e-15)};\n"
                   "    minArea -1;\n"
                   f"    minTwist {_mq('mesh_min_twist', 0.02)};\n"
                   f"    minDeterminant {_mq('mesh_min_determinant', 0.001)};\n"
                   f"    minFaceWeight {_mq('mesh_min_face_weight', 0.05)};\n"
                   f"    minVolRatio {_mq('mesh_min_vol_ratio', 0.01)};\n"
                   "    minTriangleTwist -1;\n"
                   f"    nSmoothScale {_mq('mesh_n_smooth_scale', 4)};\n"
                   f"    errorReduction {_mq('mesh_error_reduction', 0.75)};\n"
                   f"    relaxed {{ maxNonOrtho {_mq('mesh_relaxed_max_non_ortho', 75)}; }}\n"
                   "}\n"
                   "mergeTolerance 1e-6;\n")

        self._write_text(os.path.join(case_dir, "system", "snappyHexMeshDict"), snappy)

    def _write_snappy_charge_only(self, case_dir: str, inputs: CaseInputs3D, dims: Dict[str, float]) -> None:
        """Write snappyHexMeshDict for 3D with no obstacles. Charge outer refinement (if enabled) via refinementRegions."""
        content = self._foam_header("surfaceFeatureExtractDict", "dictionary", "system")
        content += "\nsurfaces\n(\n);\n\nincludedAngle 120;\n\nsubsetFeatures\n{\n    nonManifoldEdges yes;\n    openEdges yes;\n}\n"
        self._write_text(os.path.join(case_dir, "system", "surfaceFeaturesDict"), content)
        loc_pt = self._location_in_mesh_point(inputs)
        _snappy_pre_lv = self._snappy_charge_pre_level(inputs, dims)
        add_charge_outer, level_str = self._charge_outer_refine_levels(inputs)
        snappy = self._foam_header("snappyHexMeshDict", "dictionary", "system") + "\n"
        snappy += "castellatedMesh on;\nsnap on;\naddLayers off;\n\n"
        snappy += "geometry\n{\n"
        if _snappy_pre_lv > 0:
            snappy += self._snappy_charge_geometry_block(inputs, dims)
        if add_charge_outer and level_str:
            cx, cy, cz, outer_rad = self._charge_outer_geometry(inputs, dims)
            snappy += f'    chargeRefineOuter {{ type searchableSphere; centre ({cx:.6g} {cy:.6g} {cz:.6g}); radius {outer_rad:.6g}; }}\n'
        snappy += "};\n\n"
        n_cbl = max(1, min(10, getattr(inputs, "transition_cells", 2)))
        snappy += "castellatedMeshControls\n{\n"
        snappy += f"    maxLocalCells 10000000;\n    maxGlobalCells 200000000;\n    minRefinementCells 2;\n    maxLoadUnBalance 0.1;\n    nCellsBetweenLevels {n_cbl};\n"
        snappy += "    features ();\n"
        snappy += "    refinementSurfaces\n    {\n    }\n"
        _ref_regions_no_obs = ""
        if _snappy_pre_lv > 0:
            _ref_regions_no_obs += f"        chargeRegionSnappy {{ mode inside; levels (({_snappy_pre_lv} {_snappy_pre_lv})); }}\n"
        if add_charge_outer and level_str:
            parts = level_str.strip().split()
            if len(parts) >= 2:
                rmin_r, rmax_r = int(parts[0]), int(parts[1])
                _ref_regions_no_obs += f"        chargeRefineOuter {{ mode inside; levels (({rmin_r} {rmax_r})); }}\n"
        if _ref_regions_no_obs:
            snappy += f"    refinementRegions\n    {{\n{_ref_regions_no_obs}    }}\n"
        else:
            snappy += "    refinementRegions {\n    }\n"
        snappy += "    resolveFeatureAngle 30;\n"
        snappy += f"    locationInMesh ({loc_pt[0]:.6f} {loc_pt[1]:.6f} {loc_pt[2]:.6f});\n    allowFreeStandingZoneFaces false;\n}}\n\n"
        snappy += ("snapControls\n{\n"
                   "    nSmoothPatch 3;\n    tolerance 1.0;\n    nSolveIter 100;\n    nRelaxIter 10;\n"
                   "    nFeatureSnapIter 15;\n    implicitFeatureSnap true;\n    explicitFeatureSnap false;\n    multiRegionFeatureSnap false;\n}\n\n")
        snappy += ("addLayersControls\n{\n"
                   "    featureAngle 100;\n    slipFeatureAngle 30;\n    nLayerIter 50;\n    nRelaxedIter 20;\n    nRelaxIter 5;\n"
                   "    nGrow 0;\n    nSmoothSurfaceNormals 1;\n    nSmoothNormals 3;\n    nSmoothThickness 10;\n"
                   "    maxFaceThicknessRatio 0.5;\n    maxThicknessToMedialRatio 0.3;\n    minMedialAxisAngle 90;\n"
                   "    nBufferCellsNoExtrude 0;\n    layers {}\n    relativeSizes true;\n    expansionRatio 1.2;\n"
                   "    finalLayerThickness 0.5;\n    minThickness 1e-3;\n}\n")
        snappy += ("meshQualityControls\n{\n"
                   "    maxNonOrtho 65;\n    maxBoundarySkewness 20;\n    maxInternalSkewness 4;\n    maxConcave 80;\n"
                   "    minVol 1e-13;\n    minTetQuality 1e-15;\n    minArea -1;\n    minTwist 0.02;\n    minDeterminant 0.001;\n"
                   "    minFaceWeight 0.05;\n    minVolRatio 0.01;\n    minTriangleTwist -1;\n    nSmoothScale 4;\n"
                   "    errorReduction 0.75;\n    relaxed { maxNonOrtho 75; }\n}\nmergeTolerance 1e-6;\n")
        self._write_text(os.path.join(case_dir, "system", "snappyHexMeshDict"), snappy)

    def _get_obstacle_patch_names(self, inputs: CaseInputs3D) -> list:
        """Return patch names that snappyHexMesh will create for obstacles (same as in snappyHexMeshDict)."""
        names = []
        for i, obs in enumerate(inputs.obstacles):
            base = os.path.splitext(os.path.basename(obs.stl_path))[0]
            safe_name = f"obs{i}_{base}.stl"
            clean_name = safe_name.replace(".", "_")
            names.append(clean_name)
        return names

    def _calculate_charge_dimensions(self, inputs: CaseInputs3D) -> Dict[str, float]:
        mass = inputs.mass_kg
        rho = inputs.rho_charge if inputs.rho_charge > 0 else 1600.0
        dims = {}
        vol = mass / rho
        if inputs.charge_shape == "Sphere":
            r = ((3.0 * vol) / (4.0 * math.pi)) ** (1.0 / 3.0)
            dims["radius"] = r
        elif inputs.charge_shape == "Cuboid":
            L = getattr(inputs, "charge_length", 0) or 0
            W = getattr(inputs, "charge_width", 0) or 0
            H = getattr(inputs, "charge_height", 0) or 0
            if L > 1e-9 and W > 1e-9 and H > 1e-9 and abs(L * W * H - vol) <= 0.02 * vol:
                dims["length"] = L
                dims["width"] = W
                dims["height"] = H
            else:
                side = vol ** (1.0 / 3.0)
                dims["side"] = side
        else:  # Cylinder
            r = inputs.cylinder_radius if inputs.cylinder_radius > 0 else 0.05
            # Unified length: explicit > aspect > volume-driven (matches setRefinedFields behavior)
            if hasattr(inputs, "charge_length") and inputs.charge_length > 1e-9:
                length = inputs.charge_length
            elif hasattr(inputs, "charge_aspect") and inputs.charge_aspect > 1e-9:
                length = 2.0 * r * inputs.charge_aspect
            else:
                # Fallback: compute from volume (mass-driven geometry)
                length = vol / (math.pi * r * r)
            dims["radius"] = r
            dims["length"] = length
        return dims
    
    def _validate_charge_position(
        self,
        inputs: CaseInputs3D,
        charge_radius: float,
        half_extent_axis: Optional[float] = None,
    ) -> None:
        """Validate charge position: do NOT adjust. Record if charge is outside domain for Info panel."""
        cx, cy, cz = inputs.charge_center
        min_x, min_y, min_z = inputs.min_point
        max_x, max_y, max_z = inputs.max_point
        center_inside = (min_x <= cx <= max_x and min_y <= cy <= max_y and min_z <= cz <= max_z)
        # Check if charge extent (center ± radius or half_length) goes outside
        extent = charge_radius if half_extent_axis is None else max(charge_radius, half_extent_axis)
        outside = (
            cx - extent < min_x or cx + extent > max_x
            or cy - extent < min_y or cy + extent > max_y
            or cz - extent < min_z or cz + extent > max_z
        )
        if outside or not center_inside:
            self._charge_clipped_by_domain = True
            msg = "Charge center or geometry extends outside domain."
            if not center_inside:
                msg = "Charge center is outside domain."
            self._charge_warnings.append(msg)
            import sys
            print(f"\n⚠️  3D WARNING: {msg} Proceeding with user-defined position.", file=sys.stderr)

    def _write_block_mesh_3d(self, case_dir: str, inputs: CaseInputs3D) -> None:
        sys_dir = os.path.join(case_dir, "system")
        min_p = inputs.min_point
        max_p = inputs.max_point

        sz = max(1e-6, inputs.cell_size)
        nx = max(1, int(round(abs(max_p[0] - min_p[0]) / sz)))
        ny = max(1, int(round(abs(max_p[1] - min_p[1]) / sz)))
        nz = max(1, int(round(abs(max_p[2] - min_p[2]) / sz)))

        def get_type(key: str) -> str:
            val = inputs.boundaries.get(key, "Transmitting")
            return "wall" if val == "Reflecting" else "patch"

        # Vertices: 0-3 bottom (zmin), 4-7 top (zmax). Standard OpenFOAM hex order.
        lines = [
            self._foam_header("blockMeshDict", "dictionary", location="system"),
            "convertToMeters 1;",
            "",
            "vertices",
            "(",
            f"    ({min_p[0]} {min_p[1]} {min_p[2]})",
            f"    ({max_p[0]} {min_p[1]} {min_p[2]})",
            f"    ({max_p[0]} {max_p[1]} {min_p[2]})",
            f"    ({min_p[0]} {max_p[1]} {min_p[2]})",
            f"    ({min_p[0]} {min_p[1]} {max_p[2]})",
            f"    ({max_p[0]} {min_p[1]} {max_p[2]})",
            f"    ({max_p[0]} {max_p[1]} {max_p[2]})",
            f"    ({min_p[0]} {max_p[1]} {max_p[2]})",
            ");",
            "",
            "blocks",
            "(",
            f"    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)",
            ");",
            "",
            "edges",
            "(",
            ");",
            "",
            "boundary",
            "(",
        ]
        # Boundary patches: one per face, multi-line format like reference.
        patches = [
            ("minX", (0, 4, 7, 3), get_type("minX")),
            ("maxX", (1, 2, 6, 5), get_type("maxX")),
            ("minY", (0, 1, 5, 4), get_type("minY")),
            ("maxY", (3, 7, 6, 2), get_type("maxY")),
            ("minZ", (0, 3, 2, 1), get_type("minZ")),
            ("maxZ", (4, 5, 6, 7), get_type("maxZ")),
        ]
        for name, face_verts, btype in patches:
            v0, v1, v2, v3 = face_verts
            lines.append(f"    {name}")
            lines.append("    {")
            lines.append(f"        type {btype};")
            lines.append("        faces")
            lines.append("        (")
            lines.append(f"            ({v0} {v1} {v2} {v3})")
            lines.append("        );")
            lines.append("    }")
        lines.append(");")
        lines.append("")
        lines.append("mergePatchPairs")
        lines.append("(")
        lines.append(");")
        lines.append("")
        self._write_text(os.path.join(sys_dir, "blockMeshDict"), "\n".join(lines))

    def _write_initial_conditions_3d(
        self, case_dir: str, inputs: CaseInputs3D, obstacle_patch_names: Optional[list] = None
    ) -> None:
        # Write template to 0.orig/ with names p, U, T, ... (no .orig suffix).
        # Allrun does cp -r 0.orig 0, so 0/ will contain p, U, ... and blastFoam can read them.
        # When obstacles exist, snappyHexMesh adds wall patches (e.g. obs0_L_Wall_stl). We must
        # include boundaryField entries for those so 0/ matches the mesh after snappy.
        zero_dir = os.path.join(case_dir, "0.orig")
        obstacle_patch_names = obstacle_patch_names or []

        def get_boundary_block(val_str: str, is_vector: bool = False) -> str:
            lines = ["boundaryField", "{"]
            default_bounds = {k: "Transmitting" for k in ["minX", "maxX", "minY", "maxY", "minZ", "maxZ"]}
            user_bounds = {**default_bounds, **inputs.boundaries}
            for face_name, b_type in user_bounds.items():
                lines.append(f"    {face_name}")
                lines.append("    {")
                if b_type == "Reflecting":
                    lines.append("        type            slip;" if is_vector else "        type            zeroGradient;")
                else:
                    lines.append("        type            inletOutlet;")
                    lines.append(f"        inletValue      uniform {val_str};")
                    lines.append(f"        value           uniform {val_str};")
                lines.append("    }")
            for patch_name in obstacle_patch_names:
                lines.append(f"    {patch_name}")
                lines.append("    {")
                lines.append("        type            slip;" if is_vector else "        type            zeroGradient;")
                lines.append("    }")
            lines.append("}")
            return "\n".join(lines)

        def write_field(filename: str, object_name: str, cls: str, dim: str, val: Any) -> None:
            bc_text = get_boundary_block(str(val), is_vector=False)
            content = (
                self._foam_header(object_name, cls, "0")
                + f"\ndimensions      {dim};\n\n"
                + f"internalField   uniform {val};\n\n"
                + f"{bc_text}\n"
            )
            self._write_text(os.path.join(zero_dir, filename), content)

        write_field("p", "p", "volScalarField", "[1 -1 -2 0 0 0 0]", inputs.p_atm)
        write_field("T", "T", "volScalarField", "[0 0 0 1 0 0 0]", inputs.t_atm)
        write_field("alpha.c4", "alpha.c4", "volScalarField", "[0 0 0 0 0 0 0]", 0)
        write_field("rho.c4", "rho.c4", "volScalarField", "[1 -3 0 0 0 0 0]", inputs.rho_charge)
        write_field("rho.air", "rho.air", "volScalarField", "[1 -3 0 0 0 0 0]", 1.225)

        u_val = "(0 0 0)"
        bc_text_u = get_boundary_block(u_val, is_vector=True)
        u_content = (
            self._foam_header("U", "volVectorField", "0")
            + "\ndimensions      [0 1 -1 0 0 0 0];\n\n"
            + f"internalField   uniform {u_val};\n\n"
            + f"{bc_text_u}\n"
        )
        self._write_text(os.path.join(zero_dir, "U"), u_content)

    def _write_constant_files_3d(self, case_dir: str, inputs: CaseInputs3D, dims: Dict[str, float]) -> None:
        const_dir = os.path.join(case_dir, "constant")
        self._write_text(os.path.join(const_dir, "turbulenceProperties"), 
                         self._foam_header("turbulenceProperties", "dictionary", "constant") + "simulationType laminar;\n")
        
        # Dynamic mesh (AMR) only when enable_dyn_refine is explicitly True and not UNSET (no legacy fallback).
        prov = getattr(inputs, "provenance", {}) or {}
        use_dyn = (
            getattr(inputs, "enable_dyn_refine", None) is True
            and prov.get("enable_dyn_refine") != "UNSET"
        )
        if use_dyn:
            # building3D uses maxRefinement 1 (moving refinement front). Default 1 unless user/loaded set.
            outer_lvl = getattr(inputs, "dyn_refine_max", None) or getattr(inputs, "charge_outer_refine_max", None)
            outer_lvl = max(1, int(outer_lvl) if outer_lvl is not None else 1)
            ref_int = getattr(inputs, "refine_interval", 3)
            lower_ref = getattr(inputs, "lower_refine_threshold", 0.1)
            unref = getattr(inputs, "unrefine_threshold", 0.1)
            n_buf_dyn = getattr(inputs, "n_buffer_layers_dynamic", 2)
            enable_bal = getattr(inputs, "enable_balancing", False)
            max_cells = getattr(inputs, "dynamic_max_cells", 200000000)
            err_est = getattr(inputs, "refine_indicator_field", "densityGradient") or "densityGradient"
            # Only write maxCells / enableBalancing when they differ from defaults so the generated
            # dict matches the building3D reference (which omits both keys).
            optional_keys = ""
            if max_cells != 200000000:
                optional_keys += f"maxCells       {max_cells};\n"
            if enable_bal:
                optional_keys += f"enableBalancing {str(enable_bal).lower()};\n"
            mesh_content = f"""
dynamicFvMesh   adaptiveFvMesh;
errorEstimator  {err_est};
refineInterval  {ref_int};
lowerRefineLevel {lower_ref};
unrefineLevel   {unref};
nBufferLayers   {n_buf_dyn};
maxRefinement   {outer_lvl};
dumpLevel      true;
{optional_keys}"""
        else:
            mesh_content = "dynamicFvMesh staticFvMesh;\n"

        self._write_text(os.path.join(const_dir, "dynamicMeshDict"),
                         self._foam_header("dynamicMeshDict", "dictionary", "constant") + mesh_content)

        # Single initiation point: user initiation_point if provided else charge_center (non-remap and remap both use points)
        init_pt = getattr(inputs, "initiation_point", None)
        if init_pt is None or (isinstance(init_pt, (list, tuple)) and len(init_pt) < 3):
            init_pt = inputs.charge_center
        # JWL parameters from literature / tab_1d materials library; Custom from material_props
        jwl_lib = {
            "TNT":  {"A": 373.77e9, "B": 3.7471e9, "R1": 4.15, "R2": 0.90, "omega": 0.35, "E0": 4.29e9, "CvCoeffs": (413.15, 2.1538)},
            "C4":   {"A": 609.77e9, "B": 12.95e9, "R1": 4.50, "R2": 1.40, "omega": 0.25, "E0": 9.0e9, "CvCoeffs": (413.15, 2.1538)},
            "PETN": {"A": 617.0e9,  "B": 16.9e9,  "R1": 4.40, "R2": 1.20, "omega": 0.25, "E0": 6.11e9, "CvCoeffs": (413.15, 2.1538)},
            "ANFO": {"A": 49.46e9,  "B": 1.89e9,  "R1": 3.90, "R2": 1.10, "omega": 0.33, "E0": 3.79e9, "CvCoeffs": (413.15, 2.1538)},
        }
        if inputs.material_name == "Custom" and getattr(inputs, "material_props", None):
            mp = inputs.material_props
            if isinstance(mp, dict) and all(k in mp for k in ("A", "B", "R1", "R2", "omega")):
                j = {
                    "A": float(mp["A"]), "B": float(mp["B"]),
                    "R1": float(mp["R1"]), "R2": float(mp["R2"]), "omega": float(mp["omega"]),
                    "E0": float(mp.get("E0", mp.get("energy", 9.0e9))),
                    "CvCoeffs": mp.get("CvCoeffs", (413.15, 2.1538)),
                }
            else:
                j = jwl_lib["C4"]
        else:
            j = jwl_lib.get(inputs.material_name, jwl_lib["C4"])
        eos = f"equationOfState {{ rho0 {inputs.rho_charge}; A {j['A']:.4g}; B {j['B']:.4g}; R1 {j['R1']}; R2 {j['R2']}; omega {j['omega']}; }}"
        cv_coeffs = j.get("CvCoeffs", (413.15, 2.1538))
        thermo = f"thermodynamics {{ CvCoeffs<8> ({cv_coeffs[0]} {cv_coeffs[1]} 0 0 0 0 0 0); Sf 0.0; Hf 0.0; }}"
        e0_val = j["E0"]

        remap_enabled = getattr(inputs, "remap_enabled", False)
        R_charge = dims.get("radius", 0.05)
        cell_size = max(getattr(inputs, "cell_size", 0.1), 1e-9)
        refine_max = max(0, getattr(inputs, "refine_max", 0))
        # In Fixed Mesh (no AMR) the mesh is not refined, so smallest cell = cell_size.
        # For Dyn Mesh, the AMR max level is controlled solely by refine_max (the AMR field
        # threshold setting), independent of the static charge initialization refinement.
        enable_dyn = getattr(inputs, "enable_dyn_refine", None)
        effective_refine = refine_max if enable_dyn else 0
        smallest_cell = cell_size / (2.0 ** effective_refine)
        self._last_smallest_cell_near_charge = smallest_cell
        user_ign = getattr(inputs, "ignition_radius", None)
        if user_ign is None or user_ign <= 0:
            user_ign = min(0.05, max(0.01, 0.2 * R_charge))
            ign_radius = max(user_ign, 3.0 * smallest_cell)
        else:
            # Explicit loaded/user value gets full priority.
            user_ign = float(user_ign)
            ign_radius = user_ign
        ignition_mode = getattr(inputs, "ignition_mode", "Center of Charge")
        use_com = (ignition_mode == "Center of Charge") and not remap_enabled
        if remap_enabled:
            remap_origin = getattr(inputs, "remap_origin", (0.0, 0.0, 0.0))
            ox, oy, oz = float(remap_origin[0]), float(remap_origin[1]), float(remap_origin[2])
            if ox == 0 and oy == 0 and oz == 0:
                ox = 0.001
            ign_radius = 0.05
            use_com = False
        else:
            ox, oy, oz = float(init_pt[0]), float(init_pt[1]), float(init_pt[2])
            # Clamp initiation point to domain interior (partial-charge / ground cases)
            min_x, min_y, min_z = inputs.min_point
            max_x, max_y, max_z = inputs.max_point
            margin = 1e-6
            ox = max(min_x + margin, min(max_x - margin, ox))
            oy = max(min_y + margin, min(max_y - margin, oy))
            oz = max(min_z + margin, min(max_z - margin, oz))
        if use_com:
            initiation_block = f"""        useCOM yes;
        radius {ign_radius:.6g};
        vDet 7850;"""
        else:
            initiation_block = f"""        useCOM no;
        points (({ox:.6g} {oy:.6g} {oz:.6g}));
        radius {ign_radius:.6g};
        vDet 7850;"""
        self._last_initiation_radius_effective = ign_radius
        self._last_initiation_radius_requested = user_ign

        pp_content = self._foam_header("phaseProperties", "dictionary", location="constant") + f"""
phases (c4 air);
c4
{{
    type detonating;
    reactants
    {{
        thermoType {{ transport const; thermo eConst; equationOfState BirchMurnaghan3; }}
        equationOfState {{ rho0 {inputs.rho_charge}; Gamma 0.25; pRef 101298; K0 8.04e9; K0Prime 7.97; }}
        specie {{ molWeight 55.0; }}
        transport {{ mu 0; Pr 1; }}
        thermodynamics {{ Cv 1400; Hf 0.0; }}
    }}
    products
    {{
        thermoType {{ transport const; thermo {getattr(inputs, "thermo_model", None) or "ePolynomial"}; equationOfState {getattr(inputs, "eos_model", None) or "JWL"}; }}
        {eos}
        specie {{ molWeight 55.0; }}
        transport {{ mu 0; Pr 1; }}
        {thermo}
    }}
    activationModel {"none" if remap_enabled else (getattr(inputs, "activation_model_ui", None) or "pressureBased")};
    initiation
    {{
        E0 {e0_val}; I 4.0e6; a 0.0367; b 0.667; x 7.0; maxLambdaI 0.022;
        G1 1.4997e-7; c 0.667; d 0.33; y 2.0; minLambda1 0.022;
        G2 0.0; e 0.667; f 0.667; z 3.0; minLambda2 0.022;
        pMin {inputs.p_atm};
{initiation_block}
    }}
    residualRho 1e-6; residualAlpha 1e-6;
}}
air
{{
    type basic;
    thermoType {{ transport const; thermo {getattr(inputs, "thermo_model_air", None) or "eConst"}; equationOfState idealGas; }}
    equationOfState {{ gamma 1.4; }}
    specie {{ molWeight 28.97; }}
    transport {{ mu 0; Pr 1; }}
    thermodynamics {{ type {getattr(inputs, "thermo_model_air", None) or "eConst"}; Cv 718; Hf 0; }}
    residualRho 1e-6; residualAlpha 1e-6;
}}
"""
        self._write_text(os.path.join(const_dir, "phaseProperties"), pp_content)

    def _build_set_fields_dict_3d(
        self, inputs: CaseInputs3D, dims: Dict[str, float],
        override_charge_refinement_level: Optional[int] = None,
        remaining_inside_levels: Optional[int] = None,
        target_inside_level: Optional[int] = None,
        charge_refinement_by_topo_set: bool = False,
    ) -> str:
        """Build setFieldsDict content for 3D non-remap.

        Refinement inside the charge is handled natively by ``setRefinedFields`` via
        ``refineInternal yes; level N;`` in the region.  This adaptively subdivides
        the mesh until level ``N`` is reached inside the charge region and then sets
        the field — works regardless of base-mesh coarseness (matches the BlastFoam
        ``building3D`` reference tutorial).

        ``override_charge_refinement_level`` replaces the user level when set (e.g. one-off dict rewrite).
        ``charge_refinement_by_topo_set`` is kept for backward compatibility but
        ignored: the native path is always used now.
        """
        _ = charge_refinement_by_topo_set  # accepted for compatibility, not used
        self._last_charge_capture_meta = None
        remap_enabled = getattr(inputs, "remap_enabled", False)
        n_buf = getattr(inputs, "buffer_layers", 2)
        if remap_enabled:
            return self._foam_header("setFieldsDict", "dictionary", "system") + f"""
fields (alpha.c4);
nBufferLayers {n_buf};

defaultFieldValues ( volScalarFieldValue alpha.c4 0 );
regions ( );
"""
        cx, cy, cz = inputs.charge_center
        if override_charge_refinement_level is not None:
            charge_refine = max(0, min(8, override_charge_refinement_level))
        elif target_inside_level is not None:
            charge_refine = max(0, min(8, target_inside_level))
        elif remaining_inside_levels is not None:
            charge_refine = max(0, min(8, remaining_inside_levels))
        else:
            charge_refine = max(0, min(8, getattr(inputs, "charge_refinement_level", 0)))
        self._last_charge_refinement_dict_level = int(charge_refine)
        apply_refine_in_region = charge_refine > 0
        use_refined = charge_refine > 0 and inputs.charge_shape in ("Sphere", "Cylinder")
        rho = inputs.rho_charge if inputs.rho_charge > 0 else 1600.0
        mass = inputs.mass_kg  # Always user mass; never inflate

        if override_charge_refinement_level is None:
            if inputs.charge_shape in ("Sphere", "Cylinder"):
                charge_radius = dims.get("radius", 0.1)
                half_l = None
                if inputs.charge_shape == "Cylinder":
                    if getattr(inputs, "charge_length", None) and inputs.charge_length > 1e-9:
                        half_l = inputs.charge_length / 2.0
                    elif getattr(inputs, "charge_aspect", None) and inputs.charge_aspect > 1e-9:
                        half_l = charge_radius * inputs.charge_aspect
                    else:
                        half_l = charge_radius * 2.5
                self._validate_charge_position(inputs, charge_radius, half_extent_axis=half_l)
            elif inputs.charge_shape == "Cuboid":
                if "length" in dims and "width" in dims and "height" in dims:
                    L, W, H = dims["length"], dims["width"], dims["height"]
                    half_max = max(L, W, H) / 2.0
                    self._validate_charge_position(inputs, half_max, half_extent_axis=half_max)
                else:
                    side = dims.get("side", 0.1)
                    self._validate_charge_position(inputs, side / 2.0, half_extent_axis=side / 2.0)

        # Charge capture region (setFieldsDict keyword ``backup``): minimal search volume so
        # setRefinedFields can seed mass on a coarse base mesh. Independent of snappy outer
        # transition (bubble_radius_factor) — see _charge_outer_geometry / topoSet seed paths.
        if use_refined and inputs.charge_shape == "Sphere":
            r = dims["radius"]
            ref_part = f"        refineInternal yes;\n        level {charge_refine};\n        " if apply_refine_in_region else ""
            backup_radius, cap_report = resolve_charge_capture_radius_m(inputs, r)
            self._last_charge_capture_meta = cap_report.as_json_dict()
            for w in cap_report.warnings:
                self._charge_warnings.append(w)
            backup_block = (
                f"        backup\n        {{\n"
                f"            centre ({cx} {cy} {cz});\n"
                f"            radius {backup_radius:.6g};\n"
                f"        }}\n"
            )
            region_str = (
                f"sphericalMassToCell\n    {{\n        rho {rho:.6g};\n        mass {mass:.6g};\n        centre ({cx} {cy} {cz});\n"
                f"{backup_block}"
                f"{ref_part}fieldValues ( volScalarFieldValue alpha.c4 1 );\n    }}"
            )
        elif use_refined and inputs.charge_shape == "Cylinder":
            r = dims["radius"]
            if hasattr(inputs, "charge_length") and inputs.charge_length > 1e-9:
                length = inputs.charge_length
            elif hasattr(inputs, "charge_aspect") and inputs.charge_aspect > 1e-9:
                length = 2.0 * r * inputs.charge_aspect
            else:
                length = 2.0 * r * 2.5
            half_l = length / 2.0
            if hasattr(inputs, "charge_aspect") and inputs.charge_aspect > 1e-9:
                lbyd = float(inputs.charge_aspect)
            else:
                lbyd = (2.0 * half_l) / (2.0 * r) if r > 1e-9 else 2.5
            dir_map = {"X": "(1 0 0)", "Y": "(0 1 0)", "Z": "(0 0 1)"}
            direction = dir_map.get(inputs.cylinder_axis, "(0 0 1)")
            ref_part = f"        refineInternal yes;\n        level {charge_refine};\n        " if apply_refine_in_region else ""
            # Backup cylinder: same axis, larger radius and robust axial reach.
            # Policy:
            #   backup length = max(charge length, charge diameter)
            # This keeps capture robust for short cylinders and follows the agreed
            # automatic logic without exposing extra UI inputs.
            # If a loaded case provided an explicit backup L, preserve it via
            # charge_backup_length_override.
            axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(getattr(inputs, "cylinder_axis", "Z"), 2)
            l_vec = [0.0, 0.0, 0.0]
            loaded_backup_len = getattr(inputs, "charge_backup_length_override", None)
            if loaded_backup_len is not None:
                try:
                    backup_len = float(loaded_backup_len)
                except (TypeError, ValueError):
                    backup_len = max(length, 2.0 * r)
            else:
                backup_len = max(length, 2.0 * r)
            l_vec[axis_idx] = backup_len
            backup_radius, cap_report = resolve_charge_capture_radius_m(inputs, r)
            self._last_charge_capture_meta = cap_report.as_json_dict()
            for w in cap_report.warnings:
                self._charge_warnings.append(w)
            backup_block = (
                f"        backup\n        {{\n"
                f"            centre ({cx} {cy} {cz});\n"
                f"            L ({l_vec[0]:.6g} {l_vec[1]:.6g} {l_vec[2]:.6g});\n"
                f"            radius {backup_radius:.6g};\n"
                f"        }}\n"
            )
            region_str = (
                f"cylindericalMassToCell\n    {{\n        rho {rho:.6g};\n        mass {mass:.6g};\n        centre ({cx} {cy} {cz});\n"
                f"        direction {direction};\n        LbyD {lbyd};\n"
                f"{backup_block}"
                f"{ref_part}fieldValues ( volScalarFieldValue alpha.c4 1 );\n    }}"
            )
        elif inputs.charge_shape == "Sphere":
            r = dims["radius"]
            if apply_refine_in_region:
                region_str = (
                    f"sphereToCell {{ centre ({cx} {cy} {cz}); radius {r:.6g}; "
                    f"refineInternal yes; level {charge_refine}; fieldValues ( volScalarFieldValue alpha.c4 1 ); }}"
                )
            else:
                region_str = f"sphereToCell {{ centre ({cx} {cy} {cz}); radius {r:.6g}; fieldValues ( volScalarFieldValue alpha.c4 1 ); }}"
        elif inputs.charge_shape == "Cuboid":
            if "length" in dims and "width" in dims and "height" in dims:
                L, W, H = dims["length"], dims["width"], dims["height"]
                half_x, half_y, half_z = L / 2.0, W / 2.0, H / 2.0
                x1, x2 = cx - half_x, cx + half_x
                y1, y2 = cy - half_y, cy + half_y
                z1, z2 = cz - half_z, cz + half_z
            else:
                side = dims.get("side", 0.1)
                half = side / 2.0
                x1, x2 = cx - half, cx + half
                y1, y2 = cy - half, cy + half
                z1, z2 = cz - half, cz + half
            if apply_refine_in_region:
                region_str = (
                    f"boxToCell {{ box ({x1:.6g} {y1:.6g} {z1:.6g}) ({x2:.6g} {y2:.6g} {z2:.6g}); "
                    f"refineInternal yes; level {charge_refine}; fieldValues ( volScalarFieldValue alpha.c4 1 ); }}"
                )
            else:
                region_str = f"boxToCell {{ box ({x1:.6g} {y1:.6g} {z1:.6g}) ({x2:.6g} {y2:.6g} {z2:.6g}); fieldValues ( volScalarFieldValue alpha.c4 1 ); }}"
        else:
            r = dims["radius"]
            length = dims.get("length", 0.1)
            half_l = length / 2.0
            p1 = [cx, cy, cz]
            p2 = [cx, cy, cz]
            idx = {"X": 0, "Y": 1, "Z": 2}.get(getattr(inputs, "cylinder_axis", "Z"), 2)
            p1[idx] -= half_l
            p2[idx] += half_l
            if apply_refine_in_region:
                region_str = (
                    f"cylinderToCell {{ p1 ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g}); p2 ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g}); radius {r:.6g}; "
                    f"refineInternal yes; level {charge_refine}; fieldValues ( volScalarFieldValue alpha.c4 1 ); }}"
                )
            else:
                region_str = f"cylinderToCell {{ p1 ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g}); p2 ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g}); radius {r:.6g}; fieldValues ( volScalarFieldValue alpha.c4 1 ); }}"

        n_buf = getattr(inputs, "buffer_layers", 2)
        sf = self._foam_header("setFieldsDict", "dictionary", "system") + f"""
fields (alpha.c4);
nBufferLayers {n_buf};

defaultFieldValues ( volScalarFieldValue alpha.c4 0 );
regions ( {region_str} );
"""
        return sf

    def write_set_fields_dict_only(
        self, case_dir: str, inputs: CaseInputs3D,
        override_charge_refinement_level: Optional[int] = None,
    ) -> None:
        """Write only setFieldsDict (for retry). When override is set, Sphere/Cylinder use MassToCell with that level."""
        dims = self._calculate_charge_dimensions(inputs)
        content = self._build_set_fields_dict_3d(inputs, dims, override_charge_refinement_level)
        self._write_text(os.path.join(case_dir, "system", "setFieldsDict"), content)

    def _write_topo_set_dict_3d(self, case_dir: str, inputs: CaseInputs3D, dims: Dict[str, float]) -> None:
        """Write topoSetDict for optional seed refinement sphere (legacy path). Radius from bubble_radius_factor only — not setRefinedFields capture."""
        cx, cy, cz = inputs.charge_center
        r = dims.get("radius", 0.05)
        factor = max(0.5, min(5.0, getattr(inputs, "bubble_radius_factor", 1.5)))
        rad = r * factor
        content = self._foam_header("topoSetDict", "dictionary", "system") + f"""
actions
(
    {{
        name    chargeBubble;
        type    cellSet;
        action  new;
        source  sphereToCell;
        sourceInfo
        {{
            centre ({cx:.6g} {cy:.6g} {cz:.6g});
            radius {rad:.6g};
        }}
    }}
);
"""
        self._write_text(os.path.join(case_dir, "system", "topoSetDict"), content)

    def _write_refine_mesh_dict_3d(self, case_dir: str) -> None:
        """Write refineMeshDict to refine the chargeBubble cellSet (one level). OF9 format with global coordinateSystem."""
        content = self._foam_header("refineMeshDict", "dictionary", "system") + """
set                 chargeBubble;
coordinateSystem    global;

globalCoeffs
{
    e1              (1 0 0);
    e2              (0 1 0);
    e3              (0 0 1);
}

directions          ( e1 e2 e3 );
useHexTopology      yes;
geometricCut        no;
writeMesh           no;
"""
        self._write_text(os.path.join(case_dir, "system", "refineMeshDict"), content)

    def _write_topo_set_charge_from_field_dict(self, case_dir: str) -> None:
        """Write topoSet_chargeFromFieldDict: cellSet chargeCells = cells where alpha.c4 > 0.5 (for adding remaining levels after setRefinedFields)."""
        content = self._foam_header("topoSet_chargeFromFieldDict", "dictionary", "system") + """
actions
(
    {
        name    chargeCells;
        type    cellSet;
        action  new;
        source  fieldToCell;
        sourceInfo
        {
            field alpha.c4;
            min   0.5;
            max   1.0;
        }
    }
);
"""
        self._write_text(os.path.join(case_dir, "system", "topoSet_chargeFromFieldDict"), content)

    def _write_refine_mesh_charge_dict(self, case_dir: str, set_name: str = "chargeRegion") -> None:
        """Write refineMesh_chargeDict to refine the given cellSet (one level per run). OF9 format with global coordinateSystem."""
        content = self._foam_header("refineMesh_chargeDict", "dictionary", "system") + f"""
set                 {set_name};
coordinateSystem    global;

globalCoeffs
{{
    e1              (1 0 0);
    e2              (0 1 0);
    e3              (0 0 1);
}}

directions          ( e1 e2 e3 );
useHexTopology      yes;
geometricCut        no;
writeMesh           no;
"""
        self._write_text(os.path.join(case_dir, "system", "refineMesh_chargeDict"), content)

    def _charge_size_info(self, inputs: CaseInputs3D, dims: Dict[str, float]) -> str:
        """Return a short human-readable string describing charge dimensions for error messages."""
        shape = getattr(inputs, "charge_shape", "Sphere")
        if shape == "Sphere":
            r = dims.get("radius", 0.0)
            return f"radius={r:.4g} m"
        if shape == "Cylinder":
            r = dims.get("radius", 0.0)
            L = dims.get("length", 0.0)
            ax = getattr(inputs, "cylinder_axis", "Z")
            return f"radius={r:.4g} m, length={L:.4g} m, axis={ax}"
        if shape == "Cuboid":
            Lv = dims.get("length", 0.0)
            W = dims.get("width", 0.0)
            H = dims.get("height", 0.0)
            return f"length={Lv:.4g} m, width={W:.4g} m, height={H:.4g} m"
        return ""

    def _write_check_realization_script(self, case_dir: str, inputs: CaseInputs3D, dims: Dict[str, float]) -> None:
        """Write check_realization.py — post-refinement realization verifier.

        Called from Allrun after all topoSet/refineMesh steps, before setRefinedFields.
        Parses the last refineMesh log for global minimum cell size, reads the final
        topoSet_chargeRegion log for cell count, then verifies that the requested Inside
        level was actually realized in the mesh.  Exits non-zero with a precise
        user-facing message if realization fails.
        """
        cell_size = float(getattr(inputs, "cell_size", 0.5))
        shape = getattr(inputs, "charge_shape", "Sphere")
        size_info = self._charge_size_info(inputs, dims)

        script = f'''\
#!/usr/bin/env python3
"""
Post-refinement realization check.
Verifies that the mesh actually realized the charge pre-refinement level the user
requested (Inside).  Fails fast with a precise message when realization is absent
or insufficient so the Allrun exits before setRefinedFields is reached.
"""
import json, math, os, re, sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODE_PATH = os.path.join(CASE_DIR, "case_init_mode.json")

# ── parameters ──────────────────────────────────────────────────────────────
BASE_CELL_SIZE  = {cell_size!r}
CHARGE_SHAPE    = {shape!r}
CHARGE_SIZE_INFO = {size_info!r}

if not os.path.isfile(MODE_PATH):
    print("check_realization: case_init_mode.json not found, skipping check.")
    sys.exit(0)

with open(MODE_PATH) as _f:
    mode = json.load(_f)

requested_inside = int(mode.get("inside_levels", 0))
capture_levels   = int(mode.get("capture_levels", 0))
charge_levels    = int(mode.get("charge_levels",  0))
total_levels     = capture_levels + charge_levels

if requested_inside == 0 or total_levels == 0:
    print("check_realization: no charge pre-refinement requested, skipping check.")
    sys.exit(0)

expected_min = BASE_CELL_SIZE / (2.0 ** total_levels)

# ── parse global minLen from last refineMesh log ─────────────────────────────
log_file = "log.refineMesh_charge" if charge_levels > 0 else "log.refineMesh_captureEnvelope"
log_path = os.path.join(CASE_DIR, log_file)
min_len = None
if os.path.isfile(log_path):
    with open(log_path, "r", errors="replace") as _f:
        _lines = _f.readlines()
    for _line in reversed(_lines):
        _m = re.search(r\'minLen\\s*:\\s*([\\d.eE+\\-]+)\', _line)
        if _m:
            try:
                min_len = float(_m.group(1))
                break
            except ValueError:
                pass

realized_level = None
if min_len is not None and min_len > 1e-15:
    realized_level = math.floor(math.log2(BASE_CELL_SIZE / min_len))

# ── count cells inside true charge from final topoSet_chargeRegion log ───────
n_charge_cells = 0
topo_log = os.path.join(CASE_DIR, "log.topoSet_chargeRegion")
if os.path.isfile(topo_log):
    with open(topo_log, "r", errors="replace") as _f:
        _content = _f.read()
    for _line in reversed(_content.splitlines()):
        _m = re.search(r\'chargeRegion\\s+now\\s+size\\s+(\\d+)\', _line)
        if _m:
            n_charge_cells = int(_m.group(1))
            break
        _m = re.search(r\'Set\\s+now\\s+size\\s+(\\d+)\', _line)
        if _m:
            n_charge_cells = int(_m.group(1))
            break

# ── evaluate realization ─────────────────────────────────────────────────────
success = True
failure_reason = None

if n_charge_cells == 0:
    success = False
    failure_reason = (
        "No cells found inside the true charge region (chargeRegion cellSet is empty) "
        "after {{}} capture-envelope refinement level(s) and {{}} charge-region refinement level(s).".format(
            capture_levels, charge_levels)
    )
elif realized_level is not None and realized_level < requested_inside - 1:
    # Tolerance of 1 level: very thin charge slices near obstacle boundaries
    # can leave a single-cell-thick inside layer at level N-1; that is acceptable.
    # Below N-1 is a genuine under-refinement failure.
    success = False
    failure_reason = (
        "Global minimum cell size {{:.6g}} m corresponds to realized level {{}}, "
        "but requested Inside level is {{}} (expected minimum cell size <= {{:.6g}} m). "
        "The mesh was under-refined; a stale cellSet may have been used.".format(
            min_len, realized_level, requested_inside, expected_min)
    )

# ── update metadata ──────────────────────────────────────────────────────────
mode["realized_refinement_level"]      = realized_level
mode["smallest_cell_global"]           = min_len
mode["cells_inside_charge_post_refine"] = n_charge_cells
mode["realization_attempted_levels"]   = total_levels

if success:
    msg = (
        "Realization check PASSED. "
        "Realized level: {{}} (requested {{}}). "
        "Global min cell size: {{}} m. "
        "Cells inside charge: {{}}.".format(
            realized_level if realized_level is not None else "N/A",
            requested_inside,
            round(min_len, 8) if min_len is not None else "N/A",
            n_charge_cells,
        )
    )
    mode["realization_status"]  = "success"
    mode["realization_message"] = msg
    print("check_realization: PASS --", msg)
else:
    msg = (
        "Charge pre-refinement realization FAILED.\\n"
        "  Shape            : {{}}\\n"
        "  Charge dimensions: {{}}\\n"
        "  Base Cell Size   : {{:.6g}} m\\n"
        "  Requested Inside : {{}} levels\\n"
        "  Realized level   : {{}}\\n"
        "  Min cell size    : {{}} m\\n"
        "  Expected min     : {{:.6g}} m  (= {{:.6g}} m / 2^{{}})\\n"
        "  Cells in charge  : {{}}\\n"
        "  Reason           : {{}}\\n"
        "  To fix:\\n"
        "    - Reduce base Cell Size to <= {{:.6g}} m\\n"
        "    - OR review charge geometry / location\\n"
        "    - OR increase Inside to {{}} if the mesh cannot be refined further"
    ).format(
        CHARGE_SHAPE,
        CHARGE_SIZE_INFO,
        BASE_CELL_SIZE,
        requested_inside,
        realized_level if realized_level is not None else "N/A",
        round(min_len, 8) if min_len is not None else "N/A",
        expected_min, BASE_CELL_SIZE, total_levels,
        n_charge_cells,
        failure_reason,
        expected_min * 0.5,
        total_levels + 1,
    )
    mode["realization_status"]  = "fail"
    mode["realization_message"] = msg
    with open(MODE_PATH, "w") as _f:
        json.dump(mode, _f, indent=2)
    print("FATAL:", msg)
    sys.exit(1)

with open(MODE_PATH, "w") as _f:
    json.dump(mode, _f, indent=2)
'''
        self._write_text(os.path.join(case_dir, "check_realization.py"), script)

    def _is_charge_capturable_on_base_mesh(self, inputs: CaseInputs3D, dims: Dict[str, float]) -> bool:
        """Return True if snappy can capture the charge geometry on the BASE mesh.

        Snappy's ``refinementRegions mode inside`` needs at least one cell centre inside the
        geometry on the base mesh.  The criterion: smallest half-dimension > cell_size * sqrt(3)/2.

        Used as a routing decision only — not as a generation abort.  When False, the caller
        falls back to the explicit topoSet/refineMesh inner-initialization path (capture
        envelope + refineMesh loops), which can refine the mesh down to the charge level
        regardless of base-mesh coarseness.
        """
        shape = getattr(inputs, "charge_shape", "Sphere")
        cell_size = max(1e-9, getattr(inputs, "cell_size", 0.5))
        threshold = cell_size * math.sqrt(3.0) / 2.0
        if shape == "Sphere":
            r = float(dims.get("radius", 0.05))
            return r > threshold
        if shape == "Cylinder":
            r = float(dims.get("radius", 0.05))
            return r > threshold
        if shape == "Cuboid":
            L = dims.get("length", dims.get("side", 0.1))
            W = dims.get("width", dims.get("side", 0.1))
            H = dims.get("height", dims.get("side", 0.1))
            return min(L, W, H) > cell_size  # cuboid: each side must exceed one cell
        return True  # Unknown shape: allow

    def _capture_levels_needed(self, inputs: CaseInputs3D, dims: Dict[str, float],
                               pre_refined_levels: int = 0) -> int:
        """Minimum additional refineMesh levels (after any snappy pre-refinement) so at least one
        cell centre lies inside the true charge.

        ``pre_refined_levels`` is the number of levels snappy already applied inside the charge
        (via refinementRegions).  After snappy the effective cell size is
        ``cell_size / 2**pre_refined_levels``.  The returned value is the number of
        *further* topoSet+refineMesh passes required.  Capped at STARTUP_REFINEMENT_CAP.
        Returns 0 when the (possibly pre-refined) mesh already captures the charge.
        """
        shape = getattr(inputs, "charge_shape", "")
        base_cell_size = max(1e-9, getattr(inputs, "cell_size", 0.5))
        # Effective cell size after snappy pre-refinement
        cell_size = base_cell_size / (2.0 ** max(0, pre_refined_levels))
        if shape == "Sphere":
            r = float(dims.get("radius", 0.05))
            if r <= 0:
                return 0
            h_max = (2.0 * r) / math.sqrt(3.0)
            if cell_size <= h_max:
                return 0
            n = math.ceil(math.log2(cell_size / h_max))
            return max(0, min(n, STARTUP_REFINEMENT_CAP))
        if shape == "Cylinder":
            r = float(dims.get("radius", 0.05))
            if r <= 0:
                return 0
            h_max = (2.0 * r) / math.sqrt(3.0)
            if cell_size <= h_max:
                return 0
            n = math.ceil(math.log2(cell_size / h_max))
            return max(0, min(n, STARTUP_REFINEMENT_CAP))
        if shape == "Cuboid":
            # Box-axis: at least one cell centre inside box requires cell_size/2^n <= min(L,W,H).
            L = dims.get("length", 0.1)
            W = dims.get("width", 0.1)
            H = dims.get("height", 0.1)
            if L <= 0 or W <= 0 or H <= 0:
                return 0
            min_side = min(L, W, H)
            if cell_size <= min_side:
                return 0
            n = math.ceil(math.log2(cell_size / min_side))
            return max(0, min(n, STARTUP_REFINEMENT_CAP))
        return 0

    def _capture_envelope_dims(self, inputs: CaseInputs3D, dims: Dict[str, float]) -> Dict[str, float]:
        """Compute automatic capture-envelope dimensions so at least one base-mesh cell centre lies inside.
        Envelope may be larger than the true charge; used only for refinement, not for final explosive fill.
        Returns dict with envelope geometry (radius_envelope, length_envelope, or length/width/height_envelope)."""
        cell_size = max(1e-9, getattr(inputs, "cell_size", 0.5))
        # Minimum size so one cell centre can lie inside: for sphere, R >= cell_size * sqrt(3)/2
        min_radius = (cell_size * math.sqrt(3.0) / 2.0) * 1.01
        min_side = (cell_size * math.sqrt(3.0) / 2.0) * 1.01
        shape = getattr(inputs, "charge_shape", "Sphere")
        out = dict(dims)
        if shape == "Sphere":
            r = float(dims.get("radius", 0.05))
            out["radius_envelope"] = max(r, min_radius)
            return out
        if shape == "Cylinder":
            r = float(dims.get("radius", 0.05))
            out["radius_envelope"] = max(r, min_radius)
            # Also inflate length so at least one base-mesh cell centre falls
            # within the cylinder's axial span (needed when charge length <= cell_size).
            out["length_envelope"] = max(float(dims.get("length", 0.1)), cell_size * 1.01)
            return out
        if shape == "Cuboid":
            # Box-axis: for boxToCell at least one base-mesh cell centre must lie inside the box;
            # require each full dimension >= cell_size (half-dim >= cell_size/2). Safety factor 1.01.
            min_side_cuboid = cell_size * 1.01
            L = dims.get("length", 0.1)
            W = dims.get("width", 0.1)
            H = dims.get("height", 0.1)
            out["length_envelope"] = max(L, min_side_cuboid)
            out["width_envelope"] = max(W, min_side_cuboid)
            out["height_envelope"] = max(H, min_side_cuboid)
            return out
        out["radius_envelope"] = max(float(dims.get("radius", 0.05)), min_radius)
        return out

    def _write_topo_set_capture_envelope_dict(self, case_dir: str, inputs: CaseInputs3D, dims: Dict[str, float]) -> None:
        """Write topoSet_captureEnvelopeDict: cellSet captureEnvelope from automatic envelope geometry.
        Envelope may be larger than true charge; used only for refinement. Final fill uses true geometry in setFieldsDict."""
        env = self._capture_envelope_dims(inputs, dims)
        cx, cy, cz = inputs.charge_center
        shape = getattr(inputs, "charge_shape", "Sphere")
        if shape == "Sphere":
            r = env.get("radius_envelope", dims.get("radius", 0.05))
            actions = f"""
    {{
        name    captureEnvelope;
        type    cellSet;
        action  new;
        source  sphereToCell;
        sourceInfo
        {{
            centre ({cx:.6g} {cy:.6g} {cz:.6g});
            radius {r:.6g};
        }}
    }}
"""
        elif shape == "Cylinder":
            r = env.get("radius_envelope", dims.get("radius", 0.05))
            length = env.get("length_envelope", dims.get("length", 0.1))
            half_l = length / 2.0
            p1 = [cx, cy, cz]
            p2 = [cx, cy, cz]
            idx = {"X": 0, "Y": 1, "Z": 2}.get(getattr(inputs, "cylinder_axis", "Z"), 2)
            p1[idx] -= half_l
            p2[idx] += half_l
            actions = f"""
    {{
        name    captureEnvelope;
        type    cellSet;
        action  new;
        source  cylinderToCell;
        sourceInfo
        {{
            p1 ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g});
            p2 ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g});
            radius {r:.6g};
        }}
    }}
"""
        elif shape == "Cuboid":
            L = env.get("length_envelope", dims.get("length", 0.1))
            W = env.get("width_envelope", dims.get("width", 0.1))
            H = env.get("height_envelope", dims.get("height", 0.1))
            half_x, half_y, half_z = L / 2.0, W / 2.0, H / 2.0
            x1, x2 = cx - half_x, cx + half_x
            y1, y2 = cy - half_y, cy + half_y
            z1, z2 = cz - half_z, cz + half_z
            actions = f"""
    {{
        name    captureEnvelope;
        type    cellSet;
        action  new;
        source  boxToCell;
        sourceInfo
        {{
            box ({x1:.6g} {y1:.6g} {z1:.6g}) ({x2:.6g} {y2:.6g} {z2:.6g});
        }}
    }}
"""
        else:
            r = env.get("radius_envelope", dims.get("radius", 0.05))
            actions = f"""
    {{
        name    captureEnvelope;
        type    cellSet;
        action  new;
        source  sphereToCell;
        sourceInfo
        {{
            centre ({cx:.6g} {cy:.6g} {cz:.6g});
            radius {r:.6g};
        }}
    }}
"""
        content = self._foam_header("topoSet_captureEnvelopeDict", "dictionary", "system") + f"""
actions
(
{actions}
);
"""
        self._write_text(os.path.join(case_dir, "system", "topoSet_captureEnvelopeDict"), content)

    def _write_refine_mesh_capture_envelope_dict(self, case_dir: str) -> None:
        """Write refineMesh_captureEnvelopeDict to refine the captureEnvelope cellSet (one level per run). OF9 format with global coordinateSystem."""
        content = self._foam_header("refineMesh_captureEnvelopeDict", "dictionary", "system") + """
set                 captureEnvelope;
coordinateSystem    global;

globalCoeffs
{
    e1              (1 0 0);
    e2              (0 1 0);
    e3              (0 0 1);
}

directions          ( e1 e2 e3 );
useHexTopology      yes;
geometricCut        no;
writeMesh           no;
"""
        self._write_text(os.path.join(case_dir, "system", "refineMesh_captureEnvelopeDict"), content)

    def _write_topo_set_charge_region_dict(self, case_dir: str, inputs: CaseInputs3D, dims: Dict[str, float]) -> None:
        """Write topoSet_chargeRegionDict: cellSet chargeRegion from charge geometry (sphereToCell/cylinderToCell/boxToCell). Used for Inside-only refinement."""
        cx, cy, cz = inputs.charge_center
        shape = getattr(inputs, "charge_shape", "Sphere")
        if shape == "Sphere":
            r = dims.get("radius", 0.05)
            actions = f"""
    {{
        name    chargeRegion;
        type    cellSet;
        action  new;
        source  sphereToCell;
        sourceInfo
        {{
            centre ({cx:.6g} {cy:.6g} {cz:.6g});
            radius {r:.6g};
        }}
    }}
"""
        elif shape == "Cylinder":
            r = dims.get("radius", 0.05)
            length = dims.get("length", 0.1)
            half_l = length / 2.0
            p1 = [cx, cy, cz]
            p2 = [cx, cy, cz]
            idx = {"X": 0, "Y": 1, "Z": 2}.get(getattr(inputs, "cylinder_axis", "Z"), 2)
            p1[idx] -= half_l
            p2[idx] += half_l
            actions = f"""
    {{
        name    chargeRegion;
        type    cellSet;
        action  new;
        source  cylinderToCell;
        sourceInfo
        {{
            p1 ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g});
            p2 ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g});
            radius {r:.6g};
        }}
    }}
"""
        elif shape == "Cuboid":
            if "length" in dims and "width" in dims and "height" in dims:
                L, W, H = dims["length"], dims["width"], dims["height"]
                half_x, half_y, half_z = L / 2.0, W / 2.0, H / 2.0
            else:
                side = dims.get("side", 0.1)
                half_x = half_y = half_z = side / 2.0
            x1, x2 = cx - half_x, cx + half_x
            y1, y2 = cy - half_y, cy + half_y
            z1, z2 = cz - half_z, cz + half_z
            actions = f"""
    {{
        name    chargeRegion;
        type    cellSet;
        action  new;
        source  boxToCell;
        sourceInfo
        {{
            box ({x1:.6g} {y1:.6g} {z1:.6g}) ({x2:.6g} {y2:.6g} {z2:.6g});
        }}
    }}
"""
        else:
            r = dims.get("radius", 0.05)
            actions = f"""
    {{
        name    chargeRegion;
        type    cellSet;
        action  new;
        source  sphereToCell;
        sourceInfo
        {{
            centre ({cx:.6g} {cy:.6g} {cz:.6g});
            radius {r:.6g};
        }}
    }}
"""
        content = self._foam_header("topoSet_chargeRegionDict", "dictionary", "system") + f"""
actions
(
{actions}
);
"""
        self._write_text(os.path.join(case_dir, "system", "topoSet_chargeRegionDict"), content)

    def _write_topo_set_outside_shell_dict(self, case_dir: str, inputs: CaseInputs3D, dims: Dict[str, float]) -> None:
        """Write topoSet_outsideShellDict: outsideShell = outer region minus true charge (shape-consistent: sphere/sphere, cylinder/cylinder, cuboid/box)."""
        cx, cy, cz = inputs.charge_center
        shape = getattr(inputs, "charge_shape", "Sphere")
        _, _, _, outer_radius = self._charge_outer_geometry(inputs, dims)
        cell_size = max(getattr(inputs, "cell_size", 0.1), 1e-9)
        n_cbl = max(1, min(10, getattr(inputs, "transition_cells", 2)))
        rmin_outer = getattr(inputs, "charge_outer_refine_min", None)
        rmax_outer = getattr(inputs, "charge_outer_refine_max", None)
        rmin = rmin_outer if rmin_outer is not None else getattr(inputs, "refine_min", 2)
        rmax = rmax_outer if rmax_outer is not None else getattr(inputs, "refine_max", 3)
        level_span = max(1, rmax - rmin)
        margin = n_cbl * level_span * cell_size

        if shape == "Sphere":
            r_inner = float(dims.get("radius", 0.05))
            content = self._foam_header("topoSet_outsideShellDict", "dictionary", "system") + f"""
actions
(
    {{
        name    outer;
        type    cellSet;
        action  new;
        source  sphereToCell;
        sourceInfo
        {{
            centre ({cx:.6g} {cy:.6g} {cz:.6g});
            radius {outer_radius:.6g};
        }}
    }}
    {{
        name    inner;
        type    cellSet;
        action  new;
        source  sphereToCell;
        sourceInfo
        {{
            centre ({cx:.6g} {cy:.6g} {cz:.6g});
            radius {r_inner:.6g};
        }}
    }}
    {{
        name    outsideShell;
        type    cellSet;
        action  new;
        source  setToCell;
        sourceInfo
        {{
            set outer;
        }}
    }}
    {{
        name    outsideShell;
        type    cellSet;
        action  subtract;
        source  setToCell;
        sourceInfo
        {{
            set inner;
        }}
    }}
);
"""
        elif shape == "Cylinder":
            r_inner = float(dims.get("radius", 0.05))
            length = dims.get("length", 0.1)
            half_l = length / 2.0
            p1 = [cx, cy, cz]
            p2 = [cx, cy, cz]
            idx = {"X": 0, "Y": 1, "Z": 2}.get(getattr(inputs, "cylinder_axis", "Z"), 2)
            p1[idx] -= half_l
            p2[idx] += half_l
            content = self._foam_header("topoSet_outsideShellDict", "dictionary", "system") + f"""
actions
(
    {{
        name    outer;
        type    cellSet;
        action  new;
        source  cylinderToCell;
        sourceInfo
        {{
            p1 ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g});
            p2 ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g});
            radius {outer_radius:.6g};
        }}
    }}
    {{
        name    inner;
        type    cellSet;
        action  new;
        source  cylinderToCell;
        sourceInfo
        {{
            p1 ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g});
            p2 ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g});
            radius {r_inner:.6g};
        }}
    }}
    {{
        name    outsideShell;
        type    cellSet;
        action  new;
        source  setToCell;
        sourceInfo
        {{
            set outer;
        }}
    }}
    {{
        name    outsideShell;
        type    cellSet;
        action  subtract;
        source  setToCell;
        sourceInfo
        {{
            set inner;
        }}
    }}
);
"""
        elif shape == "Cuboid":
            if "length" in dims and "width" in dims and "height" in dims:
                L, W, H = dims["length"], dims["width"], dims["height"]
                half_x, half_y, half_z = L / 2.0, W / 2.0, H / 2.0
            else:
                side = dims.get("side", 0.1)
                half_x = half_y = half_z = side / 2.0
            x1_inner, x2_inner = cx - half_x, cx + half_x
            y1_inner, y2_inner = cy - half_y, cy + half_y
            z1_inner, z2_inner = cz - half_z, cz + half_z
            x1_outer, x2_outer = x1_inner - margin, x2_inner + margin
            y1_outer, y2_outer = y1_inner - margin, y2_inner + margin
            z1_outer, z2_outer = z1_inner - margin, z2_inner + margin
            content = self._foam_header("topoSet_outsideShellDict", "dictionary", "system") + f"""
actions
(
    {{
        name    outer;
        type    cellSet;
        action  new;
        source  boxToCell;
        sourceInfo
        {{
            box ({x1_outer:.6g} {y1_outer:.6g} {z1_outer:.6g}) ({x2_outer:.6g} {y2_outer:.6g} {z2_outer:.6g});
        }}
    }}
    {{
        name    inner;
        type    cellSet;
        action  new;
        source  boxToCell;
        sourceInfo
        {{
            box ({x1_inner:.6g} {y1_inner:.6g} {z1_inner:.6g}) ({x2_inner:.6g} {y2_inner:.6g} {z2_inner:.6g});
        }}
    }}
    {{
        name    outsideShell;
        type    cellSet;
        action  new;
        source  setToCell;
        sourceInfo
        {{
            set outer;
        }}
    }}
    {{
        name    outsideShell;
        type    cellSet;
        action  subtract;
        source  setToCell;
        sourceInfo
        {{
            set inner;
        }}
    }}
);
"""
        else:
            r_inner = float(dims.get("radius", 0.05))
            content = self._foam_header("topoSet_outsideShellDict", "dictionary", "system") + f"""
actions
(
    {{
        name    outer;
        type    cellSet;
        action  new;
        source  sphereToCell;
        sourceInfo
        {{
            centre ({cx:.6g} {cy:.6g} {cz:.6g});
            radius {outer_radius:.6g};
        }}
    }}
    {{
        name    inner;
        type    cellSet;
        action  new;
        source  sphereToCell;
        sourceInfo
        {{
            centre ({cx:.6g} {cy:.6g} {cz:.6g});
            radius {r_inner:.6g};
        }}
    }}
    {{
        name    outsideShell;
        type    cellSet;
        action  new;
        source  setToCell;
        sourceInfo
        {{
            set outer;
        }}
    }}
    {{
        name    outsideShell;
        type    cellSet;
        action  subtract;
        source  setToCell;
        sourceInfo
        {{
            set inner;
        }}
    }}
);
"""
        self._write_text(os.path.join(case_dir, "system", "topoSet_outsideShellDict"), content)

    def _write_refine_mesh_outside_dict(self, case_dir: str) -> None:
        """Write refineMesh_outsideDict to refine the outsideShell cellSet (one level per run). OF9 format with global coordinateSystem."""
        content = self._foam_header("refineMesh_outsideDict", "dictionary", "system") + """
set                 outsideShell;
coordinateSystem    global;

globalCoeffs
{
    e1              (1 0 0);
    e2              (0 1 0);
    e3              (0 0 1);
}

directions          ( e1 e2 e3 );
useHexTopology      yes;
geometricCut        no;
writeMesh           no;
"""
        self._write_text(os.path.join(case_dir, "system", "refineMesh_outsideDict"), content)

    def _write_startup_refinement_log(self, case_dir: str, split: Dict[str, Any]) -> None:
        """Write human-readable startup refinement log for transparency."""
        lines = [
            "Startup refinement split (Charge pre-refinement Inside = target final level)",
            "=" * 60,
            "base cell size [m]          : %s" % split.get("h_base"),
            "charge radius (capture) [m]  : %s" % split.get("charge_radius_used"),
            "target Inside level         : %s" % split.get("target_inside_level"),
            "startup_levels_needed       : %s" % split.get("startup_levels_needed"),
            "startup_levels (used)       : %s" % split.get("startup_levels"),
            "remaining_inside_levels     : %s" % split.get("remaining_inside_levels"),
            "expected final cell size [m] : %s" % split.get("h_final_expected"),
        ]
        if split.get("auto_adjusted"):
            lines.append("")
            lines.append("AUTO-ADJUSTED: " + (split.get("message") or "Startup increased for capture."))
        if split.get("startup_refinement_capped"):
            lines.append("")
            lines.append("WARNING: startup_levels capped at %d; capture may still fail. Reduce Cell Size or increase Inside." % STARTUP_REFINEMENT_CAP)
        self._write_text(os.path.join(case_dir, "startup_refinement.log"), "\n".join(lines))

    def _expected_emesh_list(self, case_dir: str) -> list:
        """Return list of expected feature edge mesh paths (canonical constant/extendedFeatureEdgeMesh/<base>.extendedFeatureEdgeMesh) for GUI info."""
        p = os.path.join(case_dir, "system", "expectedFeatureEdges.txt")
        if not os.path.isfile(p):
            return []
        with open(p, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _write_check_internal_patch_sh(self, case_dir: str) -> None:
        """Write check_internal_patch.sh for OF9 addEmptyPatch/0 consistency (all 3D cases with snappy)."""
        script = r'''#!/usr/bin/env bash
# Preflight: internalPatch must exist in mesh and in all required 0/* boundaryField (OF9 addEmptyPatch consistency)
set -e
BOUNDARY="constant/polyMesh/boundary"
if [ ! -f "$BOUNDARY" ]; then
  echo "FATAL: internalPatch preflight: $BOUNDARY missing."
  exit 1
fi
if ! grep -q "internalPatch" "$BOUNDARY"; then
  echo "FATAL: internalPatch missing in mesh boundary. Run addEmptyPatch before restore 0 / changeDictionary."
  exit 1
fi
for f in 0/U 0/T 0/p 0/alpha.c4 0/rho.c4 0/rho.air; do
  if [ -f "$f" ] && ! grep -q "internalPatch" "$f"; then
    echo "FATAL: $f missing boundaryField internalPatch. Run changeDictionary after addEmptyPatch and 0 restore."
    exit 1
  fi
done
echo "internalPatch preflight: OK"
'''
        self._write_text(os.path.join(case_dir, "check_internal_patch.sh"), script)

    def _write_charge_region_files(self, case_dir: str, inputs: CaseInputs3D, dims: Dict[str, float]) -> None:
        """Write case_charge_region.json and check_charge_region.py for region + ignition validation (non-remap only)."""
        import json
        cx, cy, cz = inputs.charge_center
        init_pt = getattr(inputs, "initiation_point", None) or inputs.charge_center
        ign_c = [float(init_pt[0]), float(init_pt[1]), float(init_pt[2])]
        ign_r = getattr(self, "_last_initiation_radius_effective", 0.05)
        prov = getattr(inputs, "provenance", {}) or {}
        ign_mode = getattr(inputs, "ignition_mode", "Center of Charge")
        run_ignition_check = (
            ign_mode == "Manual"
            or any(prov.get(k) in ("LOADED", "USER") for k in ("ignition_radius", "initiation_point", "ignition_mode"))
        )
        region = {
            "shape": inputs.charge_shape,
            "center": [float(cx), float(cy), float(cz)],
            "ignition_center": ign_c,
            "ignition_radius": float(ign_r),
            "run_ignition_check": run_ignition_check,
        }
        if inputs.charge_shape == "Sphere":
            region["radius"] = float(dims.get("radius", 0.05))
        elif inputs.charge_shape == "Cylinder":
            region["radius"] = float(dims.get("radius", 0.05))
            region["length"] = float(dims.get("length", 0.1))
            region["axis"] = getattr(inputs, "cylinder_axis", "Z")
        elif inputs.charge_shape == "Cuboid":
            if "length" in dims and "width" in dims and "height" in dims:
                L, W, H = dims["length"], dims["width"], dims["height"]
                hx, hy, hz = L / 2.0, W / 2.0, H / 2.0
            else:
                s = dims.get("side", 0.1)
                hx = hy = hz = s / 2.0
            region["box"] = [
                cx - hx, cy - hy, cz - hz,
                cx + hx, cy + hy, cz + hz,
            ]
        path_json = os.path.join(case_dir, "case_charge_region.json")
        self._write_text(path_json, json.dumps(region, indent=2))

        script = r'''# Region-consistency check: charge (alpha.c4>threshold) must be inside intended charge geometry.
# Run after setRefinedFields/setFields and postProcess writeCellCentres. Exit 1 if zero cells in region.
from __future__ import print_function
import json
import os
import re
import subprocess
import sys

THRESHOLD = 0.5
TIME_DIR = "0"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _ensure_cell_centres(case_dir, c_path):
    """If 0/C missing, run postProcess writeCellCentres -time 0. Return True if 0/C exists after."""
    if os.path.isfile(c_path):
        return True
    try:
        subprocess.run(
            ["postProcess", "-func", "writeCellCentres", "-time", "0"],
            cwd=case_dir,
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    return os.path.isfile(c_path)

def _is_binary_format(path):
    """Return True if FoamFile header contains format binary."""
    try:
        with open(path, "rb") as f:
            raw = f.read(2048)
        text = raw.decode("utf-8", errors="replace")
        return "format" in text and "binary" in text
    except Exception:
        return False

def _convert_to_ascii(case_dir, time_dir):
    """Run foamFormatConvert -ascii for the given time directory."""
    try:
        subprocess.run(
            ["foamFormatConvert", "-ascii", "-time", time_dir],
            cwd=case_dir,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass

def parse_scalar_field(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            text = f.read()
    except Exception:
        return None
    m = re.search(r"internalField\s+uniform\s+([\d.eE+\-]+)\s*;", text)
    if m:
        v = float(m.group(1))
        return None, v  # uniform: no per-cell list; caller uses single value
    start = text.find("internalField")
    if start == -1:
        return None
    bf = text.find("boundaryField", start)
    segment = text[start:bf] if bf != -1 else text[start:]
    paren = segment.find("(")
    if paren == -1:
        return None
    rest = segment[paren + 1:]
    m_close = re.search(r"\)\s*;", rest)
    if not m_close:
        return None
    chunk = rest[:m_close.start()]
    vals = []
    for tok in chunk.split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals

def parse_vector_field(path):
    """Parse internalField only (ignore boundaryField). Support uniform (x y z) and nonuniform List<vector> N ( ... )."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            text = f.read()
    except Exception:
        return None
    # Restrict to segment before boundaryField so we do not parse boundary data
    start = text.find("internalField")
    if start == -1:
        return None
    bf = text.find("boundaryField", start)
    segment = text[start:bf] if bf != -1 else text[start:]
    # uniform (x y z);
    m = re.search(r"internalField\s+uniform\s+\(\s*([\d.eE+\-\s]+)\s*\)\s*;", segment)
    if m:
        nums = [float(x) for x in m.group(1).split()]
        if len(nums) >= 3:
            return [(nums[0], nums[1], nums[2])]
        return None
    # nonuniform List<vector> N ( ... ) or nonuniform List<vector> ( ... )
    nu = re.search(r"internalField\s+nonuniform\s+", segment)
    if not nu:
        return None
    paren_start = segment.find("(", nu.end())
    if paren_start == -1:
        return None
    # Find the closing ") ;" (OpenFOAM may write ")\n;" so ");" is not consecutive)
    rest = segment[paren_start + 1:]
    m_close = re.search(r"\)\s*;", rest)
    if not m_close:
        return None
    chunk = rest[:m_close.start()]
    points = []
    for triple in re.findall(r"\(\s*([\d.eE+\-\s]+)\s*\)", chunk):
        nums = [float(x) for x in triple.split()]
        if len(nums) >= 3:
            points.append((nums[0], nums[1], nums[2]))
    return points if points else None

def point_in_region(x, y, z, region):
    shape = region.get("shape", "Sphere")
    cx, cy, cz = region["center"][0], region["center"][1], region["center"][2]
    if shape == "Sphere":
        r = region["radius"]
        return (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r * r * 1.0001
    if shape == "Cuboid":
        b = region["box"]
        return b[0] <= x <= b[3] and b[1] <= y <= b[4] and b[2] <= z <= b[5]
    if shape == "Cylinder":
        r, L = region["radius"], region.get("length", 0.1)
        axis = region.get("axis", "Z")
        half = L / 2.0
        if axis == "X":
            t = x - cx
            d2 = (y - cy)**2 + (z - cz)**2
        elif axis == "Y":
            t = y - cy
            d2 = (x - cx)**2 + (z - cz)**2
        else:
            t = z - cz
            d2 = (x - cx)**2 + (y - cy)**2
        return -half - 1e-9 <= t <= half + 1e-9 and d2 <= r * r * 1.0001
    return False

def main():
    case_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(case_dir, "case_charge_region.json")
    if not os.path.isfile(json_path):
        print("FATAL: case_charge_region.json missing. Cannot run region check.", file=sys.stderr)
        sys.exit(1)
    region = load_json(json_path)
    c_path = os.path.join(case_dir, TIME_DIR, "C")
    a_path = os.path.join(case_dir, TIME_DIR, "alpha.c4")
    if not os.path.isfile(c_path):
        if not _ensure_cell_centres(case_dir, c_path):
            print("FATAL: 0/C missing. Run postProcess -func writeCellCentres -time 0 first.", file=sys.stderr)
            sys.exit(1)
    centres = parse_vector_field(c_path)
    if not centres and os.path.isfile(c_path):
        if _is_binary_format(c_path):
            _convert_to_ascii(case_dir, TIME_DIR)
            centres = parse_vector_field(c_path)
    if not centres:
        if os.path.isfile(c_path):
            try:
                with open(c_path, "r") as f:
                    lines = f.readlines()[:50]
            except Exception:
                lines = ["(binary or unreadable file)\n"]
            print("FATAL: Could not parse 0/C (cell centres). First 50 lines:", file=sys.stderr)
            for L in lines:
                sys.stderr.write(L)
            print("If 0/C is binary, run: foamFormatConvert -ascii -time 0", file=sys.stderr)
        else:
            print("FATAL: Could not parse 0/C (cell centres). File missing or unreadable.", file=sys.stderr)
        sys.exit(1)
    alphas = parse_scalar_field(a_path)
    if alphas is None:
        print("FATAL: Could not parse 0/alpha.c4.", file=sys.stderr)
        sys.exit(1)
    if len(centres) == 1 and not isinstance(alphas, tuple):
        centres = centres * len(alphas)
    elif len(centres) == 1 and isinstance(alphas, tuple):
        print("FATAL: 0/C is uniform (single centre) but alpha.c4 is uniform; cannot infer n_cells. Run postProcess -func writeCellCentres -time 0 to get per-cell centres.", file=sys.stderr)
        sys.exit(1)
    uniform_val = None
    if isinstance(alphas, tuple) and len(alphas) == 2 and alphas[0] is None:
        uniform_val = float(alphas[1])
    if uniform_val is not None:
        in_region = sum(1 for c in centres if point_in_region(c[0], c[1], c[2], region))
        count = in_region if uniform_val > THRESHOLD else 0
    else:
        if len(alphas) != len(centres):
            print("FATAL: alpha.c4 length %d != cell count %d." % (len(alphas), len(centres)), file=sys.stderr)
            sys.exit(1)
        count = sum(1 for i, c in enumerate(centres) if point_in_region(c[0], c[1], c[2], region) and alphas[i] > THRESHOLD)
    if count == 0:
        print("FATAL: N_charge_cells=0. Zero cells with alpha.c4>%g in the charge region. Increase charge refinement, bubble extent, or adjust position." % THRESHOLD, file=sys.stderr)
        sys.exit(1)
    print("charge_region_check: %d cells with alpha.c4>%g in charge region" % (count, THRESHOLD))
    # Ignition intersection check only when ignition is used (Manual or LOADED/USER); skip when useCOM and UNSET
    if region.get("run_ignition_check", True):
        ign_c = region.get("ignition_center", region["center"])
        ign_r = region.get("ignition_radius", 0.05)
        def in_ignition(x, y, z):
            d2 = (x - ign_c[0])**2 + (y - ign_c[1])**2 + (z - ign_c[2])**2
            return d2 <= (ign_r * 1.0001) ** 2
        if uniform_val is not None:
            ign_count = sum(1 for c in centres if in_ignition(c[0], c[1], c[2])) if uniform_val > THRESHOLD else 0
        else:
            ign_count = sum(1 for i, c in enumerate(centres) if in_ignition(c[0], c[1], c[2]) and alphas[i] > THRESHOLD)
        if ign_count == 0:
            print("FATAL: Ignition region does not intersect any charge cells. Increase ignition radius or refine near charge.", file=sys.stderr)
            sys.exit(1)
        print("charge_region_check: %d cells in ignition region" % ign_count)

if __name__ == "__main__":
    main()
'''
        self._write_text(os.path.join(case_dir, "check_charge_region.py"), script)

    def _write_system_files_3d(
        self, case_dir: str, inputs: CaseInputs3D, dims: Dict[str, float],
        remap_start_time: Optional[str] = None,
        use_set_refined_allrun: bool = True,
        use_seed_bubble: bool = False,
    ) -> None:
        sys_dir = os.path.join(case_dir, "system")
        remap_enabled = getattr(inputs, "remap_enabled", False)
        inside_level = max(0, getattr(inputs, "charge_refinement_level", 0) or 0)
        # Two-stage flow: (1) capture envelope refinement (topoSet captureEnvelope + refineMesh) then (2) exact charge fill (setFieldsDict true geometry only).
        self._startup_refinement_levels = 0
        self._remaining_inside_levels = inside_level
        self._startup_refinement_log = None
        self._startup_refinement_message = None
        self._charge_capture_impossible_message = None  # not set for envelope flow; envelope guarantees capture
        self._envelope_empty_message = (
            "Capture envelope is empty. After snappyHexMesh the mesh has no cell centres inside the charge region, "
            "so charge capture cannot proceed. Increase Charge pre-refinement (Inside) or reduce base Cell Size, then try again."
        )
        self._charge_region_empty_message = (
            "True charge region has no cells (no cell centres inside charge after capture refinement). "
            "Reduce base Cell Size or increase Inside so the charge is captured."
        )
        self._charge_outer_refine_max = 0
        self._capture_levels = 0
        self._charge_levels = 0
        self._snappy_pre_level = 0
        if not remap_enabled:
            # Native flow (matches BlastFoam ``building3D`` reference):
            #   setRefinedFields reads setFieldsDict and, for any region with
            #   ``refineInternal yes; level N;``, adaptively refines the mesh until
            #   level ``N`` is reached inside the region AND sets the field — in one
            #   pass.  This works even when the charge is much smaller than a base
            #   cell (no manual topoSet+refineMesh capture loop required).
            sf = self._build_set_fields_dict_3d(
                inputs, dims, override_charge_refinement_level=None,
            )
            self._write_text(os.path.join(sys_dir, "setFieldsDict"), sf)
            # Inner initialization refinement is owned by setRefinedFields → no
            # manual topoSet/refineMesh capture or charge stages, no envelope dicts.
            self._capture_levels = 0
            self._charge_levels = 0
            self._snappy_pre_level = 0
            self._remaining_inside_levels = inside_level
            self._charge_capture_impossible_message = None
            self._envelope_empty_message = None
            self._charge_region_empty_message = None
            # Stage D — outer transition handled by snappy chargeRefineOuter geometry.
            self._charge_outer_refine_max = 0
        else:
            self._write_text(os.path.join(sys_dir, "setFieldsDict"), self._build_set_fields_dict_3d(inputs, dims or {}, None))

        app_name = "blastFoam"
        # Fresh runs: start at time 0. Restart/remap: startFrom latestTime and set startTime in controlDict or use -time.
        if remap_enabled and remap_start_time:
            start_from = "latestTime"
            start_time_val = str(remap_start_time)
        else:
            start_from = "startTime"
            start_time_val = "0"

        cd_lines = [
            self._foam_header("controlDict", "dictionary", "system"),
            f"application     {app_name};",
            "",
            f"startFrom       {start_from};",
            f"startTime       {start_time_val};",
            "stopAt          endTime;",
            f"endTime         {inputs.end_time_s};",
            "",
            f"deltaT          {inputs.delta_t};",
            "adjustTimeStep  yes;",
            f"maxCo            {getattr(inputs, 'cfl_value', 0.5)};",
            "maxDeltaT       1;",
            "",
        ]
        wc_type = getattr(inputs, "write_control_type", "timeStep")
        if wc_type == "adjustableRunTime":
            w_interval = getattr(inputs, "write_interval_time", 5e-5)
            cd_lines.append(f"writeControl    adjustableRunTime;")
            cd_lines.append(f"writeInterval   {w_interval};")
        else:
            cd_lines.append("writeControl    timeStep;")
            cd_lines.append(f"writeInterval   {inputs.write_interval_steps};")
        cycle_w = getattr(inputs, "cycle_write", 0)
        cd_lines += [
            f"cycleWrite      {cycle_w};",
            "purgeWrite      0;",  # Keep all time directories (0 = unlimited)
            "",
            "writeFormat     ascii;",
            "writePrecision  6;",
            "writeCompression off;",
            "",
            "timeFormat      general;",
            "timePrecision   12;",
            "runTimeModifiable true;",
            "",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //",
        ]
        if getattr(inputs, "enable_post_processing", False):
            cd_lines += [
                "functions",
                "{",
                "    impulse",
                "    {",
                "        type            impulse;",
                "        writeControl    writeTime;",
                f"        pRef            {inputs.p_atm};",
                "    }",
                "    overpressure",
                "    {",
                "        type            overpressure;",
                "        writeControl    writeTime;",
                "        store           yes;",
                f"        pRef            {inputs.p_atm};",
                "    }",
                "    maxPImpulse",
                "    {",
                "        type            fieldMinMax;",
                "        writeControl    writeTime;",
                "        fields",
                "        (",
                "            overpressure",
                "            impulse",
                "        );",
                "    }",
                "};",
            ]
        cd_lines += [
            "",
            "// ************************************************************************* //",
        ]
        self._write_text(os.path.join(sys_dir, "controlDict"), "\n".join(cd_lines))

        self._write_text(os.path.join(sys_dir, "fvSolution"),
                self._foam_header("fvSolution", "dictionary", "system") + """
solvers { "(rho|rhoU|rhoE|alpha|.*)" { solver diagonal; } }
""")
        self._write_text(os.path.join(sys_dir, "fvSchemes"),
                self._foam_header("fvSchemes", "dictionary", "system") + """
fluxScheme Kurganov;
ddtSchemes { default Euler; timeIntegrator RK2SSP 3; }
gradSchemes { default cellMDLimited leastSquares 1.0; }
divSchemes { default none; div(alphaRhoPhi.c4,lambda.c4) Riemann; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; reconstruct(alpha) quadraticMUSCL Minmod; reconstruct(rho) quadraticMUSCL Minmod; reconstruct(U) quadraticMUSCL Minmod; reconstruct(e) quadraticMUSCL Minmod; reconstruct(p) quadraticMUSCL Minmod; reconstruct(speedOfSound) quadraticMUSCL Minmod; reconstruct(lambda.c4) quadraticMUSCL Minmod; }
snGradSchemes { default corrected; }
""")
        
        n_cores = max(1, inputs.cores)
        method = getattr(inputs, "decomposition_method", None) or "scotch"
        simple_n = getattr(inputs, "decomposition_simple_n", None)
        if simple_n is None:
            n1 = n2 = n3 = 1
            if n_cores >= 8:
                n1, n2, n3 = 2, 2, n_cores // 4
            elif n_cores >= 4:
                n1, n2, n3 = 2, 2, 1
            elif n_cores >= 2:
                n1, n2, n3 = 2, 1, 1
            if n1 * n2 * n3 != n_cores:
                n3 = max(1, n_cores // (n1 * n2))
            simple_n = (n1, n2, n3)
        delta = getattr(inputs, "decomposition_simple_delta", None)
        if delta is None:
            delta = 0.001
        decomp = (
            self._foam_header("decomposeParDict", "dictionary", "system")
            + f"numberOfSubdomains {n_cores};\n\n"
            + f"method         {method};\n\n"
            + "simpleCoeffs\n{\n"
            + f"    n               ( {simple_n[0]} {simple_n[1]} {simple_n[2]} );\n"
            + f"    delta           {delta};\n"
            + "}\n\n"
            + "distributed     no;\n\n"
            + "roots           ( );\n\n"
            + "// ************************************************************************* //\n"
        )
        self._write_text(os.path.join(sys_dir, "decomposeParDict"), decomp)