#!/usr/bin/env python3
"""
Developer/audit tool: compare manual building3D-style reference vs GGUI-generated case.

Usage (from repo root):
  python tools/compare_building3d_reference.py \\
    building3D/building3D \\
    _audit_building3d_ggui/mimic_building3d

Does not run OpenFOAM. Performs shallow semantic extraction (regex), not full dict parsing.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


def _read(p: str) -> Optional[str]:
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _f(case: str, *parts: str) -> str:
    return os.path.join(case, *parts)


def _scalar(text: str, key: str) -> Optional[str]:
    m = re.search(rf"\b{re.escape(key)}\s+([\d.eE+-]+)\s*;", text)
    return m.group(1) if m else None


def _int_sc(text: str, key: str) -> Optional[int]:
    m = re.search(rf"\b{re.escape(key)}\s+(\d+)\s*;", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _word(text: str, key: str) -> Optional[str]:
    m = re.search(rf"\b{re.escape(key)}\s+(\S+)\s*;", text)
    return m.group(1) if m else None


def extract_block_mesh(case: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _f(case, "system", "blockMeshDict")}
    t = _read(out["path"])
    if not t:
        out["missing"] = True
        return out
    verts = re.findall(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", t)
    if len(verts) >= 8:
        xs = [float(v[0]) for v in verts[:8]]
        ys = [float(v[1]) for v in verts[:8]]
        zs = [float(v[2]) for v in verts[:8]]
        out["domain_min"] = (min(xs), min(ys), min(zs))
        out["domain_max"] = (max(xs), max(ys), max(zs))
    m = re.search(r"hex\s+\([^)]+\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", t)
    if m:
        out["n_cells"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return out


def extract_set_fields(case: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _f(case, "system", "setFieldsDict")}
    t = _read(out["path"])
    if not t:
        out["missing"] = True
        return out
    out["nBufferLayers"] = _int_sc(t, "nBufferLayers")
    if "cylindericalMassToCell" in t:
        out["region_type"] = "cylindericalMassToCell"
    elif "sphericalMassToCell" in t:
        out["region_type"] = "sphericalMassToCell"
    else:
        out["region_type"] = None
    out["mass"] = _scalar(t, "mass")
    out["rho"] = _scalar(t, "rho")
    mc = re.search(r"centre\s+\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", t)
    if mc:
        out["centre"] = (mc.group(1), mc.group(2), mc.group(3))
    out["LbyD"] = _scalar(t, "LbyD")
    out["refineInternal"] = "refineInternal yes" in t
    out["level"] = _int_sc(t, "level")
    br = re.search(r"backup\s*\{[^}]*radius\s+([\d.eE+-]+)\s*;", t, re.DOTALL)
    if br:
        out["backup_radius"] = br.group(1)
    bl = re.search(r"backup\s*\{[^}]*\bL\s+\(\s*([-\d.eE+\s]+)\)\s*;", t, re.DOTALL)
    if bl:
        out["backup_L"] = bl.group(1).strip()
    return out


def extract_dynamic_mesh(case: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _f(case, "constant", "dynamicMeshDict")}
    t = _read(out["path"])
    if not t:
        out["missing"] = True
        return out
    out["dynamicFvMesh"] = _word(t, "dynamicFvMesh")
    out["errorEstimator"] = _word(t, "errorEstimator")
    out["scaledDeltaField"] = _word(t, "scaledDeltaField")
    for k in (
        "refineInterval",
        "lowerRefineLevel",
        "unrefineLevel",
        "nBufferLayers",
        "maxRefinement",
    ):
        v = _scalar(t, k)
        if v is not None:
            out[k] = v
    out["dumpLevel"] = _word(t, "dumpLevel")
    out["maxCells"] = _scalar(t, "maxCells")
    out["beginUnrefine"] = _scalar(t, "beginUnrefine")
    out["has_loadBalance"] = "loadBalance" in t
    return out


def extract_snappy_summary(case: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _f(case, "system", "snappyHexMeshDict")}
    t = _read(out["path"])
    if not t:
        out["missing"] = True
        return out
    geom = re.findall(r"type\s+(triSurfaceMesh|searchableSphere|searchableCylinder|searchableBox)\s*;", t)
    out["geometry_types"] = geom
    out["n_tri_surfaces"] = geom.count("triSurfaceMesh")
    rs = re.findall(r"level\s*\(\s*(\d+)\s+(\d+)\s*\)", t)
    out["refinement_surface_levels"] = rs[:12]
    rr = re.findall(r"levels\s*\(\(\s*(\d+)\s+(\d+)\s*\)\)", t)
    out["refinement_region_levels"] = rr
    lim = re.search(r"locationInMesh\s+\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", t)
    if lim:
        out["locationInMesh"] = (lim.group(1), lim.group(2), lim.group(3))
    out["nCellsBetweenLevels"] = _scalar(
        t.split("castellatedMeshControls")[-1] if "castellatedMeshControls" in t else t, "nCellsBetweenLevels"
    )
    feat = re.findall(r"file\s+\"([^\"]+)\"\s*;\s*level\s+(\d+)\s*;", t)
    out["features"] = feat[:8]
    return out


def extract_surface_features(case: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _f(case, "system", "surfaceFeaturesDict")}
    t = _read(out["path"])
    if not t:
        out["missing"] = True
        return out
    surfs = re.findall(r"\"([^\"]+\.stl)\"", t)
    out["stl_list"] = surfs
    out["includedAngle"] = _scalar(t, "includedAngle")
    return out


def extract_phase_properties(case: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _f(case, "constant", "phaseProperties")}
    t = _read(out["path"])
    if not t:
        out["missing"] = True
        return out
    idx = t.find("\n    initiation")
    block = t[idx:] if idx >= 0 else t
    out["activationModel"] = _word(t, "activationModel")
    out["initiation_E0"] = _scalar(block, "E0")
    out["pMin"] = _scalar(block, "pMin")
    out["useCOM"] = bool(re.search(r"useCOM\s+yes\s*;", block))
    ir = re.search(
        r"initiation\s*\{[^}]*?radius\s+([\d.eE+-]+)\s*;",
        block,
        re.DOTALL,
    )
    out["ignition_radius"] = ir.group(1) if ir else _scalar(block, "radius")
    out["rho0_reactants"] = None
    ri = t.find("reactants")
    pi = t.find("products")
    if ri >= 0 and pi > ri:
        chunk = t[ri:pi]
        m = re.search(r"rho0\s+([\d.eE+-]+)\s*;", chunk)
        if m:
            out["rho0_reactants"] = m.group(1)
    return out


def extract_control_dict(case: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _f(case, "system", "controlDict")}
    t = _read(out["path"])
    if not t:
        out["missing"] = True
        return out
    out["application"] = _word(t, "application")
    out["endTime"] = _scalar(t, "endTime")
    out["deltaT"] = _scalar(t, "deltaT")
    out["writeControl"] = _word(t, "writeControl")
    out["writeInterval"] = _scalar(t, "writeInterval")
    out["maxCo"] = _scalar(t, "maxCo")
    out["has_functions"] = "functions" in t and "impulse" in t
    return out


def extract_decompose(case: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _f(case, "system", "decomposeParDict")}
    t = _read(out["path"])
    if not t:
        out["missing"] = True
        return out
    out["numberOfSubdomains"] = _int_sc(t, "numberOfSubdomains")
    out["method"] = _word(t, "method")
    mn = re.search(
        r"simpleCoeffs[^{]*\{[^}]*\bn\s+\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)",
        t,
        re.DOTALL,
    )
    if mn:
        out["simple_n"] = (int(mn.group(1)), int(mn.group(2)), int(mn.group(3)))
    return out


def extract_allrun_stages(case: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _f(case, "Allrun")}
    t = _read(out["path"])
    if not t:
        out["missing"] = True
        return out
    lines = [ln.strip() for ln in t.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    keywords: List[str] = []
    seen: set = set()

    def add(k: str) -> None:
        if k not in seen:
            seen.add(k)
            keywords.append(k)

    for ln in lines:
        if re.search(r"\bsurfaceFeatures\b", ln):
            add("surfaceFeatures")
        if re.search(r"\bblockMesh\b", ln):
            add("blockMesh")
        if "decomposePar -force" in ln or re.search(r"decomposeParFinal", ln):
            add("decomposePar-force")
        elif re.search(r"\bdecomposePar\b", ln):
            add("decomposePar")
        if "snappyHexMesh" in ln:
            add("snappyHexMesh")
        if "reconstructParMesh" in ln:
            add("reconstructParMesh")
        if "addEmptyPatch" in ln:
            add("addEmptyPatch")
        if "changeDictionary" in ln:
            add("changeDictionary")
        if "setRefinedFields" in ln:
            add("setRefinedFields")
        if re.search(r"\bsetFields\b", ln) and "setRefinedFields" not in ln:
            add("setFields")
        if "blastFoam" in ln or "getApplication" in ln:
            add("blastFoam")
    out["stage_order_deduped"] = keywords
    out["uses_parallel_snappy"] = "snappyHexMesh" in t and (
        "-parallel" in t or "runParallel" in t
    )
    out["uses_setRefinedFields"] = "setRefinedFields" in t
    return out


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, tuple):
        return str(x)
    return str(x)


def _compare_key(label: str, a: Any, b: Any, notes: str = "") -> Tuple[str, str, str]:
    match = "match" if a == b else "diff"
    extra = f"  ({notes})" if notes else ""
    return (label, _fmt(a), _fmt(b) + extra)


def print_report(ref: str, gen: str) -> None:
    print("=" * 72)
    print("building3D audit comparison (semantic extract)")
    print("  REF:", os.path.abspath(ref))
    print("  GEN:", os.path.abspath(gen))
    print("=" * 72)

    bm_r = extract_block_mesh(ref)
    bm_g = extract_block_mesh(gen)
    print("\n--- blockMeshDict ---")
    for lbl, ar, br in (
        _compare_key("domain_min", bm_r.get("domain_min"), bm_g.get("domain_min")),
        _compare_key("domain_max", bm_r.get("domain_max"), bm_g.get("domain_max")),
        _compare_key("n_cells", bm_r.get("n_cells"), bm_g.get("n_cells")),
    ):
        st = "OK" if ar == br.split("  (")[0] else "DIFF"
        print(f"  [{st}] {lbl}: ref={ar}  gen={br}")

    sf_r = extract_set_fields(ref)
    sf_g = extract_set_fields(gen)
    print("\n--- setFieldsDict (charge seed) ---")
    for k in (
        "region_type",
        "mass",
        "rho",
        "centre",
        "LbyD",
        "nBufferLayers",
        "refineInternal",
        "level",
        "backup_radius",
        "backup_L",
    ):
        ar, br = sf_r.get(k), sf_g.get(k)
        st = "OK" if ar == br else "DIFF"
        print(f"  [{st}] {k}: ref={_fmt(ar)}  gen={_fmt(br)}")

    dm_r = extract_dynamic_mesh(ref)
    dm_g = extract_dynamic_mesh(gen)
    print("\n--- dynamicMeshDict ---")
    for k in sorted(set(dm_r.keys()) | set(dm_g.keys())):
        if k == "path":
            continue
        ar, br = dm_r.get(k), dm_g.get(k)
        if ar is None and br is None:
            continue
        st = "OK" if ar == br else "DIFF"
        print(f"  [{st}] {k}: ref={_fmt(ar)}  gen={_fmt(br)}")

    sn_r = extract_snappy_summary(ref)
    sn_g = extract_snappy_summary(gen)
    print("\n--- snappyHexMeshDict (summary) ---")
    print(f"  geometry_types ref={sn_r.get('geometry_types')}")
    print(f"  geometry_types gen={sn_g.get('geometry_types')}")
    print(f"  refinement_surface_levels ref={sn_r.get('refinement_surface_levels')}")
    print(f"  refinement_surface_levels gen={sn_g.get('refinement_surface_levels')}")
    print(f"  refinement_region_levels ref={sn_r.get('refinement_region_levels')}")
    print(f"  refinement_region_levels gen={sn_g.get('refinement_region_levels')}")
    print(f"  locationInMesh ref={sn_r.get('locationInMesh')} gen={sn_g.get('locationInMesh')}")
    print(f"  features ref={sn_r.get('features')} gen={sn_g.get('features')}")

    su_r = extract_surface_features(ref)
    su_g = extract_surface_features(gen)
    print("\n--- surfaceFeaturesDict ---")
    print(f"  stl_list ref={su_r.get('stl_list')} gen={su_g.get('stl_list')}")
    print(f"  includedAngle ref={su_r.get('includedAngle')} gen={su_g.get('includedAngle')}")

    pp_r = extract_phase_properties(ref)
    pp_g = extract_phase_properties(gen)
    print("\n--- phaseProperties (initiation excerpt) ---")
    for k in ("activationModel", "initiation_E0", "pMin", "useCOM", "ignition_radius", "rho0_reactants"):
        ar, br = pp_r.get(k), pp_g.get(k)
        st = "OK" if ar == br else "DIFF"
        print(f"  [{st}] {k}: ref={_fmt(ar)}  gen={_fmt(br)}")

    cd_r = extract_control_dict(ref)
    cd_g = extract_control_dict(gen)
    print("\n--- controlDict ---")
    for k in ("application", "endTime", "deltaT", "writeControl", "writeInterval", "maxCo", "has_functions"):
        ar, br = cd_r.get(k), cd_g.get(k)
        st = "OK" if ar == br else "DIFF"
        print(f"  [{st}] {k}: ref={_fmt(ar)}  gen={_fmt(br)}")

    dc_r = extract_decompose(ref)
    dc_g = extract_decompose(gen)
    print("\n--- decomposeParDict ---")
    for k in ("numberOfSubdomains", "method", "simple_n"):
        ar, br = dc_r.get(k), dc_g.get(k)
        st = "OK" if ar == br else "DIFF"
        print(f"  [{st}] {k}: ref={_fmt(ar)}  gen={_fmt(br)}")

    ar = extract_allrun_stages(ref)
    ag = extract_allrun_stages(gen)
    print("\n--- Allrun (detected stages, order may be approximate) ---")
    print(f"  ref: {ar.get('stage_order_deduped')}")
    print(f"  gen: {ag.get('stage_order_deduped')}")
    print(f"  ref parallel snappy: {ar.get('uses_parallel_snappy')}  setRefinedFields: {ar.get('uses_setRefinedFields')}")
    print(f"  gen parallel snappy: {ag.get('uses_parallel_snappy')}  setRefinedFields: {ag.get('uses_setRefinedFields')}")

    print("\n" + "=" * 72)
    print("Done. Review DIFF lines and snappy geometry lists for intentional GUI design gaps.")
    print("=" * 72)


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="Compare reference building3D case vs GGUI-generated mimic.")
    ap.add_argument(
        "reference_case",
        nargs="?",
        default=os.path.join(repo, "building3D", "building3D"),
        help="Path to reference case root (default: building3D/building3D)",
    )
    ap.add_argument(
        "generated_case",
        nargs="?",
        default=os.path.join(repo, "_audit_building3d_ggui", "mimic_building3d"),
        help="Path to GGUI case root (default: _audit_building3d_ggui/mimic_building3d)",
    )
    args = ap.parse_args()
    ref, gen = args.reference_case, args.generated_case
    if not os.path.isdir(ref):
        print(f"ERROR: reference case not found: {ref}", file=sys.stderr)
        return 1
    if not os.path.isdir(gen):
        print(f"ERROR: generated case not found: {gen}", file=sys.stderr)
        print("  Hint: python tools/generate_building3d_mimic_case.py", file=sys.stderr)
        return 1
    print_report(ref, gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
