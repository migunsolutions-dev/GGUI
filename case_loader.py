"""
case_loader.py — Load an existing BlastFoam/OpenFOAM case folder and extract
parameters that map to the GUI input fields.

Returns a dict with parsed values for keys that have current UI widgets only.
Also returns _load_summary: { filled, not_filled, unsupported } for transparency.
"""

import math
import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Low-level OpenFOAM dict helpers
# ---------------------------------------------------------------------------

def _read_text(path: str) -> Optional[str]:
    """Read a file and strip C/C++ comments.  Returns None if file missing."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    # Strip block comments  /* ... */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Strip line comments  // ...
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _scalar(text: str, key: str) -> Optional[float]:
    """Extract  key  <number> ;  from text."""
    m = re.search(rf"\b{re.escape(key)}\s+([\d\.eE\+\-]+)\s*;", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _int_val(text: str, key: str) -> Optional[int]:
    """Extract  key  <integer> ;  from text."""
    m = re.search(rf"\b{re.escape(key)}\s+(\d+)\s*;", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _string_val(text: str, key: str) -> Optional[str]:
    """Extract  key  <word> ;  from text (unquoted word token)."""
    m = re.search(rf"\b{re.escape(key)}\s+(\S+)\s*;", text)
    return m.group(1) if m else None


def _vector3(text: str, key: str) -> Optional[Tuple[float, float, float]]:
    """Extract  key  ( x y z )  from text."""
    m = re.search(
        rf"\b{re.escape(key)}\s+\(\s*([\d\.eE\+\-]+)\s+([\d\.eE\+\-]+)\s+([\d\.eE\+\-]+)\s*\)",
        text,
    )
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        except ValueError:
            pass
    return None


def _vector3_int(text: str, key: str) -> Optional[Tuple[int, int, int]]:
    """Extract  key  ( n1 n2 n3 )  from text (integers)."""
    m = re.search(
        rf"\b{re.escape(key)}\s+\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)",
        text,
    )
    if m:
        try:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _find_block(text: str, block_name: str) -> Optional[str]:
    """Return the contents between the outermost braces of *block_name { ... }*.

    Handles nested braces correctly.
    """
    pattern = rf"\b{re.escape(block_name)}\s*\{{"
    m = re.search(pattern, text)
    if not m:
        return None
    start = m.end()  # first char after opening '{'
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


# ---------------------------------------------------------------------------
#  Per-file parsers
# ---------------------------------------------------------------------------

def _parse_blockMeshDict(case_dir: str, out: Dict[str, Any]) -> None:
    text = _read_text(os.path.join(case_dir, "system", "blockMeshDict"))
    if text is None:
        return

    # --- vertices ---
    vblock = re.search(r"vertices\s*\((.+?)\)\s*;", text, re.DOTALL)
    if vblock:
        verts = re.findall(
            r"\(\s*([\d\.eE\+\-]+)\s+([\d\.eE\+\-]+)\s+([\d\.eE\+\-]+)\s*\)",
            vblock.group(1),
        )
        if len(verts) >= 8:
            xs = [float(v[0]) for v in verts]
            ys = [float(v[1]) for v in verts]
            zs = [float(v[2]) for v in verts]
            out["min_point"] = (min(xs), min(ys), min(zs))
            out["max_point"] = (max(xs), max(ys), max(zs))

    # --- hex cell counts ---
    m = re.search(
        r"hex\s+\([^)]+\)\s+\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", text
    )
    if m and "min_point" in out and "max_point" in out:
        nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
        dx = (out["max_point"][0] - out["min_point"][0]) / max(nx, 1)
        dy = (out["max_point"][1] - out["min_point"][1]) / max(ny, 1)
        dz = (out["max_point"][2] - out["min_point"][2]) / max(nz, 1)
        # cell_size = average of three directions (usually equal)
        out["cell_size"] = round((dx + dy + dz) / 3.0, 6)

    # --- boundary types ---
    # Strategy: first check if we have the GUI's own 6-face naming (minX..maxZ).
    # If so, read their `type` directly: `type wall` → Reflecting, `type patch` → Transmitting.
    # Otherwise fall back to heuristic name matching for manually-authored cases
    # (e.g. building3D with "ground"/"outlet").
    #
    # blockMeshDict uses  boundary ( ... );  (parentheses, not braces) so
    # _find_block (brace-based) won't work.  Extract the section manually.
    bounds = None
    bm_idx = text.find("boundary")
    if bm_idx >= 0:
        # Find the opening '(' after 'boundary'
        rest = text[bm_idx:]
        open_paren = rest.find("(")
        if open_paren >= 0:
            # Grab everything from '(' to matching ')'
            start = bm_idx + open_paren + 1
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            if depth == 0:
                bounds = text[start:i - 1]
    if bounds:
        gui_face_names = {"minX", "maxX", "minY", "maxY", "minZ", "maxZ"}
        # Parse every patch entry: name { type <word>; ... }
        patch_types: Dict[str, str] = {}
        for m_pt in re.finditer(r"(\w+)\s*\{[^}]*?type\s+(\w+)\s*;", bounds):
            patch_types[m_pt.group(1)] = m_pt.group(2)

        boundaries: Dict[str, str] = {}
        found_gui_names = gui_face_names & set(patch_types.keys())

        if found_gui_names:
            # GUI-generated blockMeshDict: read type directly
            for face_name in gui_face_names:
                pt = patch_types.get(face_name, "wall")
                boundaries[face_name] = "Reflecting" if pt == "wall" else "Transmitting"
        else:
            # Manually-authored case (e.g. building3D): heuristic mapping
            reflecting_names = set()
            transmitting_names = set()
            for patch_name, ptype in patch_types.items():
                low = patch_name.lower()
                # type wall → Reflecting regardless of name
                if ptype == "wall":
                    reflecting_names.add(patch_name)
                elif any(kw in low for kw in ("ground", "slip", "symmetry")):
                    reflecting_names.add(patch_name)
                elif any(kw in low for kw in ("outlet", "open", "far")):
                    transmitting_names.add(patch_name)
                elif ptype == "patch":
                    # Generic "type patch" with no special name → guess Transmitting
                    transmitting_names.add(patch_name)
                else:
                    reflecting_names.add(patch_name)

            # Map to 6-face GUI scheme
            # ground/bottom face → minZ=Reflecting
            if reflecting_names:
                boundaries["minZ"] = "Reflecting"
            # If outlet/transmitting names found → set sides+top to Transmitting
            if transmitting_names:
                for face_key in ("minX", "maxX", "minY", "maxY", "maxZ"):
                    boundaries[face_key] = "Transmitting"

            out["_boundary_reflecting_names"] = reflecting_names
            out["_boundary_transmitting_names"] = transmitting_names

        if boundaries:
            out["boundaries"] = boundaries


def _parse_controlDict(case_dir: str, out: Dict[str, Any]) -> None:
    text = _read_text(os.path.join(case_dir, "system", "controlDict"))
    if text is None:
        return
    v = _scalar(text, "maxCo")
    if v is not None:
        out["cfl_value"] = v
    v = _scalar(text, "endTime")
    if v is not None:
        out["end_time_s"] = v
    v = _scalar(text, "deltaT")
    if v is not None:
        out["delta_t"] = v
    v = _scalar(text, "writeInterval")
    if v is not None:
        out["write_interval_raw"] = v  # raw value; caller decides steps vs time
    wc = _string_val(text, "writeControl")
    if wc is not None:
        # OpenFOAM uses "runTime" for time-based write; GUI uses "adjustableRunTime"
        if wc == "runTime":
            wc = "adjustableRunTime"
        out["write_control_type"] = wc
        if wc == "adjustableRunTime" and v is not None:
            out["write_interval_time"] = v
        elif wc == "timeStep" and v is not None:
            out["write_interval_steps"] = int(round(v))
    iv = _int_val(text, "cycleWrite")
    if iv is not None:
        out["cycle_write"] = iv


def _parse_setFieldsDict(case_dir: str, out: Dict[str, Any]) -> None:
    text = _read_text(os.path.join(case_dir, "system", "setFieldsDict"))
    if text is None:
        return

    # nBufferLayers (top-level in setFieldsDict)
    iv = _int_val(text, "nBufferLayers")
    if iv is not None:
        out["buffer_layers"] = iv

    # Detect shape from keyword (setRefinedFields: mass-based; setFields: geometry-based)
    if "cylindericalMassToCell" in text or "cylindricalMassToCell" in text:
        out["charge_shape"] = "Cylinder"
    elif "sphericalMassToCell" in text:
        out["charge_shape"] = "Sphere"
    elif "sphereToCell" in text:
        out["charge_shape"] = "Sphere"
    elif "cylinderToCell" in text:
        out["charge_shape"] = "Cylinder"
    elif "boxToCell" in text:
        out["charge_shape"] = "Cuboid"

    # Extract within the regions block (first mass-to-cell or geometry region)
    regions = _find_block(text, "regions")
    if regions is None:
        regions = text  # fallback: search whole file

    v = _scalar(regions, "rho")
    if v is not None:
        out["rho_charge"] = v
    v = _scalar(regions, "mass")
    if v is not None:
        out["mass_kg"] = v
    vec = _vector3(regions, "centre")
    if vec is not None:
        out["charge_center"] = vec
    # cylinderToCell has p1 and p2, not centre — use midpoint
    if "charge_center" not in out:
        p1 = _vector3(regions, "p1")
        p2 = _vector3(regions, "p2")
        if p1 is not None and p2 is not None and len(p1) == 3 and len(p2) == 3:
            out["charge_center"] = (
                (p1[0] + p2[0]) / 2.0,
                (p1[1] + p2[1]) / 2.0,
                (p1[2] + p2[2]) / 2.0,
            )
    # Radius from region (setFields: sphereToCell/cylinderToCell have radius directly)
    if "charge_radius" not in out:
        v = _scalar(regions, "radius")
        if v is not None:
            out["charge_radius"] = v
    # Cylinder-specific: LbyD (aspect ratio)
    v = _scalar(regions, "LbyD")
    if v is not None:
        out["charge_lbyd"] = v
    # Cylinder axis from direction vector (1 0 0)->X, (0 1 0)->Y, (0 0 1)->Z
    dvec = _vector3(regions, "direction")
    if dvec is not None and len(dvec) == 3:
        ax = (1.0, 0.0, 0.0)
        ay = (0.0, 1.0, 0.0)
        az = (0.0, 0.0, 1.0)
        t = 0.5
        if abs(dvec[0] - 1) < t and abs(dvec[1]) < t and abs(dvec[2]) < t:
            out["cylinder_axis"] = "X"
        elif abs(dvec[1] - 1) < t and abs(dvec[0]) < t and abs(dvec[2]) < t:
            out["cylinder_axis"] = "Y"
        elif abs(dvec[2] - 1) < t and abs(dvec[0]) < t and abs(dvec[1]) < t:
            out["cylinder_axis"] = "Z"
    # Backup geometry: radius (only use as fallback if main radius not found)
    backup = _find_block(regions, "backup")
    if backup:
        v = _scalar(backup, "radius")
        if v is not None and "charge_radius" not in out:  # <-- FIX: don't overwrite main radius
            out["charge_radius"] = v
        v = _scalar(backup, "L")
        if v is not None:
            out["charge_length"] = v
    # Charge refinement level (setRefinedFields)
    iv = _int_val(regions, "level")
    if iv is not None:
        out["charge_refinement_level"] = iv
    # Backup radius factor: backup_radius / computed_charge_radius (when both available)
    backup_rad = _scalar(backup, "radius") if backup else None
    if backup_rad is not None and backup_rad > 1e-9:
        mass_val = out.get("mass_kg")
        rho_val = out.get("rho_charge")
        if mass_val and rho_val and rho_val > 0:
            vol = mass_val / rho_val
            if out.get("charge_shape") == "Sphere":
                main_r = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
                if main_r > 1e-9:
                    out["charge_backup_radius_factor"] = backup_rad / main_r
                    # Store computed main radius if not already set
                    if "charge_radius" not in out or out["charge_radius"] == backup_rad:
                        out["charge_radius"] = main_r
            elif out.get("charge_shape") == "Cylinder":
                lbyd = out.get("charge_lbyd", 2.5)
                if lbyd and lbyd > 0:
                    main_r = (vol / (2.0 * math.pi * lbyd)) ** (1.0 / 3.0)
                    if main_r > 1e-9:
                        out["charge_backup_radius_factor"] = backup_rad / main_r
                        # Store computed main radius if not already set
                        if "charge_radius" not in out or out["charge_radius"] == backup_rad:
                            out["charge_radius"] = main_r


def _parse_phaseProperties(case_dir: str, out: Dict[str, Any]) -> None:
    text = _read_text(os.path.join(case_dir, "constant", "phaseProperties"))
    if text is None:
        return

    # activationModel
    am = _string_val(text, "activationModel")
    if am is not None:
        out["activation_model"] = am
        out["activation_model_ui"] = am

    # Products: equationOfState model name and thermo (thermoType { equationOfState X; thermo Y; } or equationOfState { ... })
    products = _find_block(text, "products")
    if products:
        thermo_products = _find_block(products, "thermoType")
        if thermo_products:
            eos_name = _string_val(thermo_products, "equationOfState")
            if eos_name is not None:
                out["eos_model"] = eos_name
            thermo_name = _string_val(thermo_products, "thermo")
            if thermo_name is not None:
                out["thermo_model"] = thermo_name
        eos = _find_block(products, "equationOfState")
        if eos:
            for key in ("A", "B", "R1", "R2", "omega", "rho0"):
                v = _scalar(eos, key)
                if v is not None:
                    out[f"jwl_{key}"] = v
    # Reactants rho0 and EOS name
    reactants = _find_block(text, "reactants")
    if reactants:
        eos_r = _find_block(reactants, "equationOfState")
        if eos_r:
            v = _scalar(eos_r, "rho0")
            if v is not None:
                out["reactants_rho0"] = v

    # Air thermodynamics type (thermodynamics { type eConst; } or thermoType thermo eConst)
    air_block = _find_block(text, "air")
    if air_block:
        thermo = _find_block(air_block, "thermodynamics")
        if thermo:
            v = _scalar(thermo, "Cv")
            if v is not None:
                out["air_Cv"] = v
            ttype = _string_val(thermo, "type")
            if ttype is not None:
                out["thermo_model_air"] = ttype
        thermo_air = _find_block(air_block, "thermoType")
        if thermo_air and "thermo_model_air" not in out:
            tname = _string_val(thermo_air, "thermo")
            if tname is not None:
                out["thermo_model_air"] = tname


def _parse_initial_fields(case_dir: str, out: Dict[str, Any]) -> None:
    """Parse 0/*.orig (or 0/*) for atmosphere pressure and temperature."""
    zero_dir = os.path.join(case_dir, "0")
    for fname, out_key in [("p.orig", "p_atm"), ("p", "p_atm"),
                            ("T.orig", "t_atm"), ("T", "t_atm")]:
        if out_key in out:
            continue  # already found from .orig
        text = _read_text(os.path.join(zero_dir, fname))
        if text is None:
            continue
        m = re.search(r"internalField\s+uniform\s+([\d\.eE\+\-]+)\s*;", text)
        if m:
            try:
                out[out_key] = float(m.group(1))
            except ValueError:
                pass


def _parse_decomposeParDict(case_dir: str, out: Dict[str, Any]) -> None:
    text = _read_text(os.path.join(case_dir, "system", "decomposeParDict"))
    if text is None:
        return
    v = _int_val(text, "numberOfSubdomains")
    if v is not None:
        out["cores"] = v
    method = _string_val(text, "method")
    if method is not None:
        out["decomposition_method"] = method
    simple = _find_block(text, "simpleCoeffs")
    if simple:
        n_tuple = _vector3_int(simple, "n")
        if n_tuple is not None:
            out["decomposition_simple_n"] = n_tuple
        delta = _scalar(simple, "delta")
        if delta is not None:
            out["decomposition_simple_delta"] = delta


def _parse_dynamicMeshDict(case_dir: str, out: Dict[str, Any]) -> None:
    path = os.path.join(case_dir, "constant", "dynamicMeshDict")
    text = _read_text(path)
    if text is None:
        return
    if "staticFvMesh" in text:
        out["enable_local_refinement"] = False
        out["enable_dyn_refine"] = False
    else:
        out["enable_local_refinement"] = True
        out["enable_dyn_refine"] = True
    v = _int_val(text, "maxRefinement")
    if v is not None:
        out["refine_max"] = v
        out["charge_outer_refine_max"] = v
        out["dyn_refine_max"] = v
    v = _int_val(text, "refineInterval")
    if v is not None:
        out["refine_interval"] = v
    v = _scalar(text, "lowerRefineLevel")
    if v is not None:
        out["lower_refine_threshold"] = v
    v = _scalar(text, "unrefineLevel")
    if v is not None:
        out["unrefine_threshold"] = v
    v = _int_val(text, "nBufferLayers")
    if v is not None:
        out["n_buffer_layers_dynamic"] = v
    s = _string_val(text, "errorEstimator")
    if s is not None:
        out["refine_indicator_field"] = s
    s = _string_val(text, "enableBalancing")
    if s is not None:
        out["enable_balancing"] = s.lower() == "true"


def _parse_snappyHexMeshDict(case_dir: str, out: Dict[str, Any]) -> None:
    """Extract STL file references from the geometry block."""
    text = _read_text(os.path.join(case_dir, "system", "snappyHexMeshDict"))
    if text is None:
        return
    geom = _find_block(text, "geometry")
    if geom is None:
        return

    stl_files: List[Dict[str, Any]] = []
    # Find all triSurfaceMesh entries:  <name> { type triSurfaceMesh; file "<file>"; ... }
    for m in re.finditer(r"(\w+)\s*\{([^}]*type\s+triSurfaceMesh[^}]*)\}", geom):
        entry_name = m.group(1)
        body = m.group(2)
        # file "X.stl"  or  file X.stl;
        fm = re.search(r'file\s+"?([^";]+)"?\s*;', body)
        if not fm:
            continue
        stl_filename = fm.group(1).strip()
        # scale
        scale = 1.0
        sm = re.search(r"scale\s+([\d\.eE\+\-]+)\s*;", body)
        if sm:
            try:
                scale = float(sm.group(1))
            except ValueError:
                pass
        # Resolve to full path
        stl_path = os.path.join(case_dir, "constant", "triSurface", stl_filename)
        stl_files.append({
            "name": entry_name,
            "file": stl_filename,
            "path": stl_path,
            "scale": scale,
            "exists": os.path.isfile(stl_path),
        })

    if stl_files:
        out["stl_obstacles"] = stl_files

    # Obstacle surface refinement levels from refinementSurfaces (first entry)
    ref_surf = _find_block(text, "refinementSurfaces")
    if ref_surf:
        level_m = re.search(r"level\s*\(\s*(\d+)\s+(\d+)\s*\)", ref_surf)
        if level_m:
            out["obstacle_refine_min"] = int(level_m.group(1))
            out["obstacle_refine_max"] = int(level_m.group(2))
            out["enable_obstacle_refine"] = (int(level_m.group(1)) != 0 or int(level_m.group(2)) != 0)
    # castellatedMeshControls
    cast = _find_block(text, "castellatedMeshControls")
    if cast:
        v = _int_val(cast, "nCellsBetweenLevels")
        if v is not None:
            out["obstacle_cells_between_levels"] = v
            out["mesh_n_cells_between_levels"] = v
        v = _int_val(cast, "resolveFeatureAngle")
        if v is not None:
            out["obstacle_feature_angle"] = v
            out["mesh_resolve_feature_angle"] = v
    # snapControls
    snap = _find_block(text, "snapControls")
    if snap:
        v = _int_val(snap, "nSolveIter")
        if v is not None:
            out["obstacle_snap_iter"] = v
            out["mesh_n_solve_iter"] = v
        v = _int_val(snap, "nFeatureSnapIter")
        if v is not None:
            out["obstacle_feature_snap_iter"] = v
            out["mesh_n_feature_snap_iter"] = v
        v = _int_val(snap, "nSmoothPatch")
        if v is not None:
            out["mesh_n_smooth_patch"] = v
        v = _scalar(snap, "tolerance")
        if v is not None:
            out["mesh_snap_tolerance"] = v
        v = _int_val(snap, "nRelaxIter")
        if v is not None:
            out["mesh_n_relax_iter"] = v
        b = _string_val(snap, "explicitFeatureSnap")
        if b is not None:
            out["mesh_explicit_feature_snap"] = b.lower() == "true"
        b = _string_val(snap, "implicitFeatureSnap")
        if b is not None:
            out["mesh_implicit_feature_snap"] = b.lower() == "true"
        b = _string_val(snap, "multiRegionFeatureSnap")
        if b is not None:
            out["mesh_multi_region_feature_snap"] = b.lower() == "true"
    # meshQualityControls
    mqc = _find_block(text, "meshQualityControls")
    if mqc:
        v = _scalar(mqc, "maxNonOrtho")
        if v is not None:
            out["mesh_max_non_ortho"] = v
        v = _scalar(mqc, "maxBoundarySkewness")
        if v is not None:
            out["mesh_max_boundary_skewness"] = v
        v = _scalar(mqc, "maxInternalSkewness")
        if v is not None:
            out["mesh_max_internal_skewness"] = v
        v = _scalar(mqc, "maxConcave")
        if v is not None:
            out["mesh_max_concave"] = v
        v = _scalar(mqc, "minVol")
        if v is not None:
            out["mesh_min_vol"] = v
        v = _scalar(mqc, "minTetQuality")
        if v is not None:
            out["mesh_min_tet_quality"] = v
        v = _scalar(mqc, "minTwist")
        if v is not None:
            out["mesh_min_twist"] = v
        v = _scalar(mqc, "minDeterminant")
        if v is not None:
            out["mesh_min_determinant"] = v
        v = _scalar(mqc, "minFaceWeight")
        if v is not None:
            out["mesh_min_face_weight"] = v
        v = _scalar(mqc, "minVolRatio")
        if v is not None:
            out["mesh_min_vol_ratio"] = v
        v = _int_val(mqc, "nSmoothScale")
        if v is not None:
            out["mesh_n_smooth_scale"] = v
        v = _scalar(mqc, "errorReduction")
        if v is not None:
            out["mesh_error_reduction"] = v
        relaxed = _find_block(mqc, "relaxed")
        if relaxed:
            vo = _scalar(relaxed, "maxNonOrtho")
            if vo is not None:
                out["mesh_relaxed_max_non_ortho"] = vo


def _parse_surfaceFeaturesDict(case_dir: str, out: Dict[str, Any]) -> None:
    text = _read_text(os.path.join(case_dir, "system", "surfaceFeaturesDict"))
    if text is None:
        return
    v = _int_val(text, "includedAngle")
    if v is not None:
        out["obstacle_feature_angle"] = v
        out["mesh_included_angle"] = v


# Do NOT infer outside_extent from topoSetDict; leave UNSET if not explicitly in case.


def _not_filled_reason(key: str, charge_shape: str, is_remap: bool) -> str:
    """Return reason for not_filled: not applicable for this case vs not in case (left UNSET)."""
    if key in ("charge_width", "charge_height") and charge_shape != "Cuboid":
        return "not applicable for this case"
    if key == "charge_length" and charge_shape == "Sphere":
        return "not applicable for this case"
    if key == "charge_lbyd" and charge_shape != "Cylinder":
        return "not applicable for this case"
    if key == "cylinder_axis" and charge_shape != "Cylinder":
        return "not applicable for this case"
    if key in ("initiation_point", "ignition_mode", "ignition_radius") and is_remap:
        return "not applicable for this case"
    return "not in case (left UNSET)"


# ---------------------------------------------------------------------------
#  Material detection
# ---------------------------------------------------------------------------

# Known materials with their JWL rho0 values (matches materials_db in tab_3d_general.py)
_KNOWN_MATERIALS = {
    "TNT":  {"rho": 1630, "energy": 4.29e6},
    "C4":   {"rho": 1601, "energy": 4.52e6},
    "PETN": {"rho": 1770, "energy": 6.11e6},
    "ANFO": {"rho": 840,  "energy": 3.79e6},
}


def _detect_material(out: Dict[str, Any]) -> None:
    """Try to match parsed JWL/rho0 to a known material name.

    Picks the material whose density is closest (within 5% tolerance).
    """
    rho0 = out.get("reactants_rho0") or out.get("rho_charge")
    if rho0 is None:
        return

    # Find the best match by smallest relative difference
    best_name: Optional[str] = None
    best_diff = float("inf")
    for name, props in _KNOWN_MATERIALS.items():
        diff = abs(rho0 - props["rho"]) / max(props["rho"], 1)
        if diff < 0.05 and diff < best_diff:
            best_diff = diff
            best_name = name

    if best_name is not None:
        out["material_name"] = best_name
        out["energy_j_per_kg"] = _KNOWN_MATERIALS[best_name]["energy"]
        return

    # No match → Custom
    out["material_name"] = "Custom"
    out["custom_material_props"] = {}
    for key in ("A", "B", "R1", "R2", "omega"):
        jkey = f"jwl_{key}"
        if jkey in out:
            out["custom_material_props"][key] = out[jkey]
    if "reactants_rho0" in out:
        out["custom_material_props"]["rho"] = out["reactants_rho0"]
    # energy: estimate from JWL E0 if available, else placeholder
    out["custom_material_props"]["energy"] = 4.5e6


# ---------------------------------------------------------------------------
#  UI field keys and unsupported keys (for Load Summary)
# ---------------------------------------------------------------------------

# Keys that have a current GUI widget and are filled by Open when present in case.
UI_FIELD_KEYS = [
    "min_point", "max_point", "cell_size", "boundaries",
    "cfl_value", "end_time_s", "delta_t", "write_control_type", "write_interval_time", "write_interval_steps", "cycle_write",
    "material_name", "custom_material_props",
    "charge_shape", "mass_kg", "rho_charge", "charge_radius", "charge_lbyd", "charge_length", "charge_width", "charge_height", "charge_center",
    "initiation_point", "ignition_mode",
    "p_atm", "t_atm",
    "refine_max", "refine_min", "enable_local_refinement", "cores",
    "enable_dyn_refine", "dyn_refine_min", "dyn_refine_max",
    "enable_obstacle_refine", "obstacle_refine_min", "obstacle_refine_max",
    "outside_extent", "transition_cells",
    "charge_refinement_level", "charge_outer_refine_min", "charge_outer_refine_max", "charge_outer_refine_enable",
    "cylinder_axis", "charge_backup_radius_factor", "buffer_layers",
    "refine_interval", "lower_refine_threshold", "unrefine_threshold", "n_buffer_layers_dynamic", "refine_indicator_field", "enable_balancing",
    "obstacle_feature_angle", "obstacle_cells_between_levels", "obstacle_snap_iter", "obstacle_feature_snap_iter",
    "activation_model", "activation_model_ui",
    "mesh_included_angle", "mesh_n_smooth_patch", "mesh_snap_tolerance", "mesh_n_solve_iter", "mesh_n_relax_iter",
    "mesh_n_feature_snap_iter", "mesh_explicit_feature_snap", "mesh_implicit_feature_snap", "mesh_multi_region_feature_snap",
    "mesh_n_cells_between_levels", "mesh_resolve_feature_angle",
    "mesh_max_non_ortho", "mesh_max_boundary_skewness", "mesh_max_internal_skewness", "mesh_max_concave",
    "mesh_min_vol", "mesh_min_tet_quality", "mesh_min_twist", "mesh_min_determinant", "mesh_min_face_weight", "mesh_min_vol_ratio",
    "mesh_n_smooth_scale", "mesh_error_reduction", "mesh_relaxed_max_non_ortho",
    "eos_model", "thermo_model", "thermo_model_air",
    "decomposition_method", "decomposition_simple_n", "decomposition_simple_delta",
    "stl_obstacles",
]

# Keys present in these case files that we do NOT map to any UI (reported as "not supported yet").
UNSUPPORTED_KEYS_BY_FILE: Dict[str, List[str]] = {
    "system/blockMeshDict": ["convertToMeters", "blocks", "edges", "patches"],
    "system/controlDict": ["startTime", "startFrom", "purgeWrite", "writeFormat", "writePrecision", "writeCompression", "timeFormat", "runTimeModifiable", "functions"],
    "system/setFieldsDict": ["defaultFieldValues", "regions"],
    "constant/phaseProperties": ["phases", "products", "reactants", "air", "activationModel", "initiationPoints", "equationOfState", "thermodynamics"],
    "system/decomposeParDict": ["method", "numberOfSubdomains", "roots", "hierarchicalCoeffs", "manualCoeffs", "scotchCoeffs"],
    "constant/dynamicMeshDict": ["dynamicFvMesh", "errorEstimator", "refineInterval", "lowerRefineLevel", "unrefineLevel", "nBufferLayers", "maxRefinement", "maxCells", "dumpLevel", "enableBalancing"],
    "system/snappyHexMeshDict": ["castellatedMeshControls", "snapControls", "addLayersControls", "meshQualityControls", "geometry", "features", "refinementSurfaces", "refinementRegions", "locationInMesh", "debug"],
    "0/p": ["internalField", "boundaryField"],
    "0/T": ["internalField", "boundaryField"],
}


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def load_case(case_dir: str) -> Dict[str, Any]:
    """Parse an existing BlastFoam case directory and return a dict of
    parameter values suitable for ``TabGeneral3D.set_case_inputs()``.

    *case_dir* should be the root of the OpenFOAM case (the folder that
    contains ``system/``, ``constant/``, ``0/``).

    Only keys that map to current GUI widgets are parsed and returned.
    The returned dict includes "_load_summary" with:
      - filled: list of keys that were set from the case
      - not_filled: list of (key, reason) for UI fields not set (use GUI default)
      - unsupported: dict file -> list of key names in that file not mapped to UI
    """
    out: Dict[str, Any] = {}
    out["_case_dir"] = case_dir
    out["_found_files"] = []
    out["_missing_files"] = []

    expected_files = [
        ("system/blockMeshDict", _parse_blockMeshDict),
        ("system/controlDict", _parse_controlDict),
        ("system/setFieldsDict", _parse_setFieldsDict),
        ("constant/phaseProperties", _parse_phaseProperties),
        ("system/decomposeParDict", _parse_decomposeParDict),
        ("constant/dynamicMeshDict", _parse_dynamicMeshDict),
        ("system/snappyHexMeshDict", _parse_snappyHexMeshDict),
    ]

    for rel_path, parse_fn in expected_files:
        full = os.path.join(case_dir, rel_path)
        if os.path.isfile(full):
            out["_found_files"].append(rel_path)
        else:
            out["_missing_files"].append(rel_path)
        parse_fn(case_dir, out)

    _parse_surfaceFeaturesDict(case_dir, out)

    # Initial fields (0/p.orig, 0/T.orig)
    _parse_initial_fields(case_dir, out)
    for fname in ("0/p.orig", "0/p", "0/T.orig", "0/T"):
        full = os.path.join(case_dir, fname)
        if os.path.isfile(full) and fname not in out["_found_files"]:
            out["_found_files"].append(fname)

    # Detect material from parsed JWL / rho0
    _detect_material(out)

    # Build load summary: filled (LOADED), not_filled (UNSET) with reason, _provenance for tab
    filled: List[str] = []
    not_filled: List[Tuple[str, str]] = []
    charge_shape = out.get("charge_shape") or "Sphere"
    activation = (out.get("activation_model") or "").lower()
    is_remap = activation == "none"
    for key in UI_FIELD_KEYS:
        if key not in out or out[key] is None:
            reason = _not_filled_reason(key, charge_shape, is_remap)
            not_filled.append((key, reason))
        elif key == "boundaries" and not out.get("boundaries"):
            not_filled.append((key, "not in case (left UNSET)"))
        elif key == "stl_obstacles" and not out.get("stl_obstacles"):
            not_filled.append((key, "no obstacles in case"))
        else:
            filled.append(key)
    # Provenance: LOADED for every key we set from case; tab will set UNSET for rest
    out["_provenance"] = {k: "LOADED" for k in filled}
    unsupported: Dict[str, List[str]] = {}
    for rel_path in out["_found_files"]:
        if rel_path in UNSUPPORTED_KEYS_BY_FILE:
            unsupported[rel_path] = UNSUPPORTED_KEYS_BY_FILE[rel_path]

    out["_load_summary"] = {
        "filled": filled,
        "not_filled": not_filled,
        "unsupported": unsupported,
    }

    log.info("load_case(%s): found=%s, missing=%s, filled=%s",
             case_dir, out["_found_files"], out["_missing_files"], len(filled))

    return out
