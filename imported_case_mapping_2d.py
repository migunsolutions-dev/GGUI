"""Map an imported axisymmetric blastFoam case onto Cylindrical–2D GUI fields.

Provenance is explicit per field. Unknown values are NOT filled with native defaults.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from case_topology import ClassificationResult, classify_case_topology
from material_catalog import materials_copy


class FieldProvenance(str, Enum):
    DIRECT = "DIRECT"
    DERIVED = "DERIVED"
    CASE_DEFINED = "CASE_DEFINED"
    NOT_RECOVERED = "NOT_RECOVERED"


_COMMENT_RE = re.compile(r"/\*.*?\*/|//.*?$", re.DOTALL | re.MULTILINE)


def _strip(text: str) -> str:
    return _COMMENT_RE.sub("", text)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as handle:
        return handle.read()


@dataclass
class MappedField:
    key: str
    displayed_value: Any
    provenance: FieldProvenance
    source_file: str = ""
    editable: bool = False
    write_target: Optional[str] = None  # e.g. "system/controlDict:endTime"
    reason: str = ""
    gui_key: Optional[str] = None  # CaseInputs2D / widget key when applicable


@dataclass
class ImportMappingResult:
    fields: Dict[str, MappedField] = field(default_factory=dict)
    gui_values: Dict[str, Any] = field(default_factory=dict)
    case_defined_keys: Tuple[str, ...] = ()
    not_recovered_keys: Tuple[str, ...] = ()
    editable_keys: Tuple[str, ...] = ()
    read_only_keys: Tuple[str, ...] = ()
    probes: Tuple[Dict[str, Any], ...] = ()
    notes: Tuple[str, ...] = ()

    def get(self, key: str) -> Optional[MappedField]:
        return self.fields.get(key)


# controlDict keys with proven one-to-one writers (working copy only).
EDITABLE_CONTROL_KEYS = {
    "end_time_s": ("system/controlDict", "endTime"),
    "delta_t": ("system/controlDict", "deltaT"),
    "max_co": ("system/controlDict", "maxCo"),
    "write_control_type": ("system/controlDict", "writeControl"),
    "write_interval_time": ("system/controlDict", "writeInterval"),
    "write_interval_steps": ("system/controlDict", "writeInterval"),
}


def _parse_macros(text: str) -> Dict[str, float]:
    macros: Dict[str, float] = {}
    for m in re.finditer(
        r"\b([A-Za-z_]\w*)\s+([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*;",
        text,
    ):
        name = m.group(1)
        # Skip FoamFile / common dict keys that are not geometry macros.
        if name in {
            "version",
            "format",
            "class",
            "object",
            "location",
            "convertToMeters",
        }:
            continue
        macros[name] = float(m.group(2))
    return macros


def _eval_round_div(expr: str, macros: Dict[str, float]) -> Optional[int]:
    """Evaluate simple #calc "round($R / $cellSize)" style expressions."""
    m = re.search(
        r"round\s*\(\s*\$(\w+)\s*/\s*\$(\w+)\s*\)",
        expr,
        re.IGNORECASE,
    )
    if not m:
        return None
    a, b = macros.get(m.group(1)), macros.get(m.group(2))
    if a is None or b is None or b == 0:
        return None
    return int(round(a / b))


def _parse_block_mesh(case_dir: str) -> Dict[str, Any]:
    path = os.path.join(case_dir, "system", "blockMeshDict")
    out: Dict[str, Any] = {"path": path if os.path.isfile(path) else ""}
    if not out["path"]:
        return out
    text = _strip(_read_text(path))
    macros = _parse_macros(text)
    out["macros"] = macros
    if "R" in macros:
        out["radius"] = macros["R"]
    if "H" in macros:
        out["height"] = macros["H"]
    if "cellSize" in macros:
        out["cell_size"] = macros["cellSize"]

    # Explicit nx/ny assignments via #calc or literals.
    for name in ("nx", "ny"):
        m = re.search(rf"\b{name}\s+#calc\s+\"([^\"]+)\"", text)
        if m:
            val = _eval_round_div(m.group(1), macros)
            if val is not None:
                out[name] = val
            else:
                out[f"{name}_expr"] = m.group(1)
        else:
            m2 = re.search(rf"\b{name}\s+(\d+)\s*;", text)
            if m2:
                out[name] = int(m2.group(1))

    block = re.search(
        r"hex\s*\([^)]+\)\s*\(([^)]+)\)\s*simpleGrading\s*\(([^)]+)\)",
        text,
    )
    if block:
        counts = block.group(1).strip().split()
        grading = [g.strip() for g in block.group(2).strip().split()]
        out["block_counts_raw"] = counts
        out["grading"] = grading
        resolved: List[Optional[int]] = []
        for tok in counts:
            if tok.startswith("$"):
                key = tok[1:]
                if key in out and isinstance(out[key], int):
                    resolved.append(out[key])
                elif key in macros:
                    resolved.append(int(round(macros[key])))
                else:
                    resolved.append(None)
            else:
                try:
                    resolved.append(int(tok))
                except ValueError:
                    resolved.append(None)
        out["block_counts"] = resolved
        if len(resolved) >= 2 and resolved[0] and resolved[1]:
            out["radial_cells"] = resolved[0]
            out["vertical_cells"] = resolved[1]
        uniform = (
            len(grading) >= 2
            and all(abs(float(g) - 1.0) < 1e-12 for g in grading[:2])
        )
        out["uniform_grading"] = uniform
    return out


def _parse_control(case_dir: str) -> Dict[str, Any]:
    path = os.path.join(case_dir, "system", "controlDict")
    out: Dict[str, Any] = {"path": path if os.path.isfile(path) else ""}
    if not out["path"]:
        return out
    text = _strip(_read_text(path))
    for key in (
        "application",
        "endTime",
        "deltaT",
        "maxCo",
        "writeControl",
        "writeInterval",
        "adjustTimeStep",
        "purgeWrite",
    ):
        m = re.search(rf"\b{key}\s+([^;]+);", text)
        if m:
            out[key] = m.group(1).strip()
    return out


def _extract_brace_block(text: str, keyword: str) -> Optional[str]:
    """Return the body inside keyword { ... } with nested-brace awareness."""
    m = re.search(rf"\b{re.escape(keyword)}\s*\{{", text)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


def _parse_setfields(case_dir: str) -> Dict[str, Any]:
    path = os.path.join(case_dir, "system", "setFieldsDict")
    out: Dict[str, Any] = {"path": path if os.path.isfile(path) else ""}
    if not out["path"]:
        return out
    text = _strip(_read_text(path))
    m = re.search(r"\bnBufferLayers\s+(\d+)\s*;", text)
    if m:
        out["nBufferLayers"] = int(m.group(1))
    body = _extract_brace_block(text, "sphereToCell")
    if body is not None:
        out["shape"] = "Sphere"
        centre = re.search(r"centre\s*\(([^)]+)\)", body)
        # Prefer primary radius, not backup radius: first radius before 'backup'.
        primary = body.split("backup", 1)[0]
        radius = re.search(r"radius\s+([^;]+);", primary)
        level = re.search(r"level\s+(\d+)\s*;", body)
        if centre:
            parts = [float(x) for x in centre.group(1).split()]
            out["centre"] = tuple(parts)
        if radius:
            out["radius"] = float(radius.group(1).strip())
        if level:
            out["level"] = int(level.group(1))
        fv = re.search(r"volScalarFieldValue\s+(\S+)\s+1", body)
        if fv:
            out["phase_field"] = fv.group(1)
        return out
    body = _extract_brace_block(text, "cylinderToCell")
    if body is not None:
        out["shape"] = "Cylinder"
        p1 = re.search(r"p1\s*\(([^)]+)\)", body)
        p2 = re.search(r"p2\s*\(([^)]+)\)", body)
        radius = re.search(r"radius\s+([^;]+);", body)
        if p1:
            out["p1"] = tuple(float(x) for x in p1.group(1).split())
        if p2:
            out["p2"] = tuple(float(x) for x in p2.group(1).split())
        if radius:
            out["radius"] = float(radius.group(1).strip())
    return out


def _parse_phase_properties(case_dir: str) -> Dict[str, Any]:
    path = os.path.join(case_dir, "constant", "phaseProperties")
    out: Dict[str, Any] = {"path": path if os.path.isfile(path) else ""}
    if not out["path"]:
        return out
    text = _strip(_read_text(path))
    phases = re.search(r"phases\s*\(([^)]+)\)", text)
    if phases:
        out["phases"] = [p.strip() for p in phases.group(1).split()]
    # First detonating / explosive-like phase density.
    rho = re.search(r"rho0\s+([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", text)
    if rho:
        out["rho0"] = float(rho.group(1))
    e0 = re.search(r"\bE0\s+([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", text)
    if e0:
        out["E0"] = float(e0.group(1))
    points = re.search(r"points\s*\(\(([^)]+)\)\)", text)
    if points:
        out["initiation_point"] = tuple(float(x) for x in points.group(1).split())
    return out


def _parse_dynamic_mesh(case_dir: str) -> Dict[str, Any]:
    path = os.path.join(case_dir, "constant", "dynamicMeshDict")
    out: Dict[str, Any] = {"path": path if os.path.isfile(path) else ""}
    if not out["path"]:
        return out
    text = _strip(_read_text(path))
    for key in (
        "dynamicFvMesh",
        "errorEstimator",
        "refineInterval",
        "lowerRefineLevel",
        "unrefineLevel",
        "nBufferLayers",
        "maxRefinement",
        "maxCells",
        "dumpLevel",
    ):
        m = re.search(rf"\b{key}\s+([^;]+);", text)
        if m:
            out[key] = m.group(1).strip()
    return out


def _parse_uniform_field(case_dir: str, name: str) -> Optional[float]:
    for folder in ("0", "0.orig"):
        for fname in (name, f"{name}.orig"):
            path = os.path.join(case_dir, folder, fname)
            if not os.path.isfile(path):
                continue
            text = _strip(_read_text(path))
            m = re.search(
                r"internalField\s+uniform\s+([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",
                text,
            )
            if m:
                return float(m.group(1))
    return None


def _map_material(phase_field: Optional[str], phases: List[str]) -> Tuple[Optional[str], str]:
    """Return (material_name, reason)."""
    catalog = materials_copy()
    candidates: List[str] = []
    if phase_field and phase_field.startswith("alpha."):
        candidates.append(phase_field.split(".", 1)[1])
    candidates.extend(phases or [])
    for cand in candidates:
        key = cand.upper() if cand.lower() == "c4" else cand
        # Prefer exact catalog keys (C4).
        for name in catalog:
            if name.lower() == cand.lower():
                return name, f"phase '{cand}' matched material catalog"
        if key in catalog:
            return key, f"phase '{cand}' matched material catalog"
    if candidates:
        return None, f"phase '{candidates[0]}' has no GGUI catalog entry — case-defined"
    return None, "explosive phase not recovered"


def _add(
    result: ImportMappingResult,
    key: str,
    value: Any,
    provenance: FieldProvenance,
    *,
    source_file: str = "",
    editable: bool = False,
    write_target: Optional[str] = None,
    reason: str = "",
    gui_key: Optional[str] = None,
    apply_gui: bool = True,
) -> None:
    mf = MappedField(
        key=key,
        displayed_value=value,
        provenance=provenance,
        source_file=source_file,
        editable=editable,
        write_target=write_target,
        reason=reason,
        gui_key=gui_key or key,
    )
    result.fields[key] = mf
    if (
        apply_gui
        and provenance in (FieldProvenance.DIRECT, FieldProvenance.DERIVED)
        and value is not None
        and mf.gui_key
    ):
        result.gui_values[mf.gui_key] = value


def map_imported_case_to_gui(
    case_dir: str,
    classification: Optional[ClassificationResult] = None,
) -> ImportMappingResult:
    """Build a typed mapping from an imported working-case directory."""
    case_dir = os.path.normpath(case_dir)
    classification = classification or classify_case_topology(case_dir)
    result = ImportMappingResult()
    notes: List[str] = []

    block = _parse_block_mesh(case_dir)
    control = _parse_control(case_dir)
    setfields = _parse_setfields(case_dir)
    phases = _parse_phase_properties(case_dir)
    dyn = _parse_dynamic_mesh(case_dir)
    ev = classification.evidence

    # --- Domain ---
    radius = block.get("radius", ev.radius_extent_m)
    height = block.get("height", ev.height_extent_m)
    if radius is not None:
        _add(
            result,
            "radius",
            float(radius),
            FieldProvenance.DIRECT if "radius" in block else FieldProvenance.DERIVED,
            source_file=block.get("path") or "topology",
            reason="blockMeshDict R / topology extent",
        )
    else:
        _add(
            result,
            "radius",
            None,
            FieldProvenance.NOT_RECOVERED,
            reason="domain radius not found",
            apply_gui=False,
        )

    if height is not None:
        _add(
            result,
            "height",
            float(height),
            FieldProvenance.DIRECT if "height" in block else FieldProvenance.DERIVED,
            source_file=block.get("path") or "topology",
            reason="blockMeshDict H / topology extent",
        )
    else:
        _add(
            result,
            "height",
            None,
            FieldProvenance.NOT_RECOVERED,
            reason="domain height not found",
            apply_gui=False,
        )

    if ev.symmetry_axis:
        _add(
            result,
            "symmetry_axis",
            ev.symmetry_axis,
            FieldProvenance.DERIVED,
            source_file=ev.source,
            reason="topology wedge axis",
            apply_gui=False,
        )
    if ev.wedge_half_angle_deg is not None:
        _add(
            result,
            "wedge_half_angle_deg",
            float(ev.wedge_half_angle_deg),
            FieldProvenance.DERIVED,
            source_file=ev.source,
            reason="wedge patch geometry",
            apply_gui=False,
        )

    radial = block.get("radial_cells")
    vertical = block.get("vertical_cells")
    if radial is not None:
        _add(
            result,
            "radial_cells",
            int(radial),
            FieldProvenance.DERIVED,
            source_file=block.get("path", ""),
            reason="blockMeshDict hex block / #calc",
            apply_gui=False,
        )
    if vertical is not None:
        _add(
            result,
            "vertical_cells",
            int(vertical),
            FieldProvenance.DERIVED,
            source_file=block.get("path", ""),
            reason="blockMeshDict hex block / #calc",
            apply_gui=False,
        )

    grading_ok = bool(block.get("uniform_grading"))
    cell_size = block.get("cell_size")
    if cell_size is not None and grading_ok and radial and height and radius:
        # Effective sizes when uniform.
        _add(
            result,
            "cell_size",
            float(cell_size),
            FieldProvenance.DIRECT,
            source_file=block.get("path", ""),
            reason="blockMeshDict cellSize with uniform simpleGrading",
        )
        _add(
            result,
            "radial_cell_size",
            float(radius) / float(radial),
            FieldProvenance.DERIVED,
            source_file=block.get("path", ""),
            reason="R / radial_cells",
            apply_gui=False,
        )
        _add(
            result,
            "axial_cell_size",
            float(height) / float(vertical) if vertical else None,
            FieldProvenance.DERIVED,
            source_file=block.get("path", ""),
            reason="H / vertical_cells",
            apply_gui=False,
        )
    elif grading_ok is False or (cell_size is None and (radial or vertical)):
        _add(
            result,
            "cell_size",
            None,
            FieldProvenance.CASE_DEFINED,
            source_file=block.get("path", ""),
            reason="graded or non-scalar base mesh — preserved in blockMeshDict",
            apply_gui=False,
        )
        notes.append("Base Cell Size is case-defined (non-representable as a single scalar).")
    else:
        _add(
            result,
            "cell_size",
            None,
            FieldProvenance.NOT_RECOVERED,
            reason="base cell size not recovered",
            apply_gui=False,
        )

    # --- Charge ---
    shape = setfields.get("shape")
    if shape:
        _add(
            result,
            "charge_shape",
            shape,
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            gui_key="charge_shape",
        )
    else:
        _add(
            result,
            "charge_shape",
            None,
            FieldProvenance.NOT_RECOVERED,
            reason="no sphereToCell/cylinderToCell in setFieldsDict",
            apply_gui=False,
        )

    centre = setfields.get("centre")
    if centre and len(centre) >= 2:
        # Y-axial: height_of_burst = centre_y
        _add(
            result,
            "height_of_burst",
            float(centre[1]),
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            reason=f"sphere centre {centre}",
        )
        _add(
            result,
            "charge_centre",
            centre,
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            apply_gui=False,
        )
    else:
        _add(
            result,
            "height_of_burst",
            None,
            FieldProvenance.NOT_RECOVERED,
            reason="charge centre not recovered",
            apply_gui=False,
        )

    charge_r = setfields.get("radius")
    if charge_r is not None and shape == "Sphere":
        # Native GUI stores mass; radius is derived. For import we store a
        # synthetic mass that reproduces radius ONLY when full-sphere is valid.
        # Sphere at (0,0,0) with ground at y=0 intersects the domain boundary —
        # do not invent mass.
        intersects_boundary = False
        if centre is not None and height is not None and radius is not None:
            cy = float(centre[1])
            if cy - float(charge_r) < -1e-12 or cy + float(charge_r) > float(height) + 1e-12:
                intersects_boundary = True
            if float(centre[0]) ** 2 + (float(centre[2]) if len(centre) > 2 else 0.0) ** 2 > 1e-18:
                intersects_boundary = True
            # Centre on ground plane with positive radius → half outside.
            if abs(cy) < 1e-12 and float(charge_r) > 0:
                intersects_boundary = True
        _add(
            result,
            "charge_radius",
            float(charge_r),
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            apply_gui=False,
            reason="setFieldsDict sphereToCell radius",
        )
        if intersects_boundary:
            _add(
                result,
                "mass_kg",
                None,
                FieldProvenance.NOT_RECOVERED,
                source_file=setfields.get("path", ""),
                reason=(
                    "sphere intersects domain boundary — full-sphere mass formula unsafe"
                ),
                apply_gui=False,
            )
            notes.append(
                f"Charge radius {charge_r} m loaded; mass left unrecovered "
                "(sphere intersects domain boundary)."
            )
        else:
            # Safe full-sphere mass only when density known and fully inside.
            rho = phases.get("rho0")
            if rho:
                mass = (4.0 / 3.0) * math.pi * (float(charge_r) ** 3) * float(rho)
                _add(
                    result,
                    "mass_kg",
                    mass,
                    FieldProvenance.DERIVED,
                    source_file=setfields.get("path", "") + " + phaseProperties",
                    reason="full sphere volume × rho0",
                )
            else:
                _add(
                    result,
                    "mass_kg",
                    None,
                    FieldProvenance.NOT_RECOVERED,
                    reason="density unavailable for mass derivation",
                    apply_gui=False,
                )
    elif charge_r is not None:
        _add(
            result,
            "charge_radius",
            float(charge_r),
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            apply_gui=False,
        )
        _add(
            result,
            "mass_kg",
            None,
            FieldProvenance.NOT_RECOVERED,
            reason="cylinder mass not uniquely determined from setFieldsDict alone",
            apply_gui=False,
        )
    else:
        _add(
            result,
            "mass_kg",
            None,
            FieldProvenance.NOT_RECOVERED,
            reason="charge radius/mass not recovered",
            apply_gui=False,
        )

    phase_field = setfields.get("phase_field")
    material, mat_reason = _map_material(phase_field, phases.get("phases") or [])
    if material:
        _add(
            result,
            "material_name",
            material,
            FieldProvenance.DERIVED,
            source_file=phases.get("path") or setfields.get("path", ""),
            reason=mat_reason,
        )
        catalog = materials_copy()
        props = catalog.get(material, {})
        if "rho" in props:
            _add(
                result,
                "rho_charge",
                float(props["rho"]),
                FieldProvenance.DERIVED,
                source_file="material_catalog",
                reason=f"catalog density for {material}",
            )
        elif phases.get("rho0") is not None:
            _add(
                result,
                "rho_charge",
                float(phases["rho0"]),
                FieldProvenance.DIRECT,
                source_file=phases.get("path", ""),
                reason="phaseProperties rho0",
            )
        if "energy" in props:
            _add(
                result,
                "energy_j_per_kg",
                float(props["energy"]),
                FieldProvenance.DERIVED,
                source_file="material_catalog",
                reason=f"catalog energy for {material}",
            )
    else:
        _add(
            result,
            "material_name",
            None,
            FieldProvenance.CASE_DEFINED if (phase_field or phases.get("phases")) else FieldProvenance.NOT_RECOVERED,
            source_file=phases.get("path") or setfields.get("path", ""),
            reason=mat_reason,
            apply_gui=False,
        )
        if phases.get("rho0") is not None:
            _add(
                result,
                "rho_charge",
                float(phases["rho0"]),
                FieldProvenance.DIRECT,
                source_file=phases.get("path", ""),
                reason="phaseProperties rho0",
            )
        else:
            _add(
                result,
                "rho_charge",
                None,
                FieldProvenance.NOT_RECOVERED,
                reason="charge density not recovered",
                apply_gui=False,
            )
        _add(
            result,
            "energy_j_per_kg",
            None,
            FieldProvenance.CASE_DEFINED if phases.get("E0") is not None else FieldProvenance.NOT_RECOVERED,
            source_file=phases.get("path", ""),
            reason="initiation E0 / EOS preserved in phaseProperties (not GGUI energy spin)",
            apply_gui=False,
        )

    if phases.get("initiation_point") is not None:
        ip = phases["initiation_point"]
        if len(ip) >= 2:
            _add(
                result,
                "detonation_height",
                float(ip[1]),
                FieldProvenance.DIRECT,
                source_file=phases.get("path", ""),
                reason=f"initiation points {ip}",
            )

    _add(
        result,
        "initialization_source",
        "Direct Charge",
        FieldProvenance.DERIVED,
        source_file=setfields.get("path", ""),
        reason="setFieldsDict present",
    )

    # Atmosphere
    p_atm = _parse_uniform_field(case_dir, "p")
    if p_atm is not None:
        _add(
            result,
            "p_atm",
            float(p_atm),
            FieldProvenance.DIRECT,
            source_file="0/p or 0/p.orig",
        )
    else:
        _add(
            result,
            "p_atm",
            None,
            FieldProvenance.NOT_RECOVERED,
            reason="ambient pressure not recovered",
            apply_gui=False,
        )
    t_atm = _parse_uniform_field(case_dir, "T")
    if t_atm is not None:
        _add(
            result,
            "t_atm",
            float(t_atm),
            FieldProvenance.DIRECT,
            source_file="0/T or 0/T.orig",
        )
    else:
        _add(
            result,
            "t_atm",
            None,
            FieldProvenance.NOT_RECOVERED,
            reason="ambient temperature not recovered",
            apply_gui=False,
        )

    # Boundaries — only when proven equivalent (conservative: leave unrecovered).
    for key, label in (
        ("outer_boundary", "outlet/outer"),
        ("top_boundary", "top"),
        ("bottom_boundary", "ground/bottom"),
    ):
        _add(
            result,
            key,
            None,
            FieldProvenance.NOT_RECOVERED,
            reason=f"OpenFOAM BC for {label} has no proven GGUI Open/Reflecting equivalent",
            apply_gui=False,
        )

    # Solver / controlDict — editable when keys exist
    def _float_ctrl(of_key: str, gui_key: str) -> None:
        if of_key not in control:
            _add(
                result,
                gui_key,
                None,
                FieldProvenance.NOT_RECOVERED,
                reason=f"{of_key} missing",
                apply_gui=False,
            )
            return
        try:
            val = float(control[of_key])
        except ValueError:
            _add(
                result,
                gui_key,
                control[of_key],
                FieldProvenance.CASE_DEFINED,
                source_file=control.get("path", ""),
                reason=f"{of_key} not a plain float",
                apply_gui=False,
            )
            return
        target = f"system/controlDict:{of_key}"
        _add(
            result,
            gui_key,
            val,
            FieldProvenance.DIRECT,
            source_file=control.get("path", ""),
            editable=True,
            write_target=target,
        )

    _float_ctrl("endTime", "end_time_s")
    _float_ctrl("deltaT", "delta_t")
    _float_ctrl("maxCo", "max_co")

    if "application" in control:
        _add(
            result,
            "application",
            control["application"],
            FieldProvenance.DIRECT,
            source_file=control.get("path", ""),
            apply_gui=False,
        )
    if "writeControl" in control:
        wc = control["writeControl"]
        # Map only known GUI values
        if wc in ("adjustableRunTime", "timeStep", "runTime", "clockTime", "cpuTime"):
            _add(
                result,
                "write_control_type",
                wc if wc != "runTime" else "adjustableRunTime",
                FieldProvenance.DIRECT,
                source_file=control.get("path", ""),
                editable=True,
                write_target="system/controlDict:writeControl",
            )
        else:
            _add(
                result,
                "write_control_type",
                wc,
                FieldProvenance.CASE_DEFINED,
                source_file=control.get("path", ""),
                reason="writeControl value not in GGUI combo",
                apply_gui=False,
            )
    if "writeInterval" in control:
        try:
            wi = float(control["writeInterval"])
            gui_wc = result.gui_values.get("write_control_type", control.get("writeControl"))
            if gui_wc in ("timeStep",):
                _add(
                    result,
                    "write_interval_steps",
                    int(wi),
                    FieldProvenance.DIRECT,
                    source_file=control.get("path", ""),
                    editable=True,
                    write_target="system/controlDict:writeInterval",
                )
            else:
                _add(
                    result,
                    "write_interval_time",
                    wi,
                    FieldProvenance.DIRECT,
                    source_file=control.get("path", ""),
                    editable=True,
                    write_target="system/controlDict:writeInterval",
                )
        except ValueError:
            _add(
                result,
                "write_interval_time",
                None,
                FieldProvenance.CASE_DEFINED,
                source_file=control.get("path", ""),
                apply_gui=False,
            )
    if "adjustTimeStep" in control:
        adj = str(control["adjustTimeStep"]).lower() in ("yes", "true", "on")
        _add(
            result,
            "adjust_time_step",
            adj,
            FieldProvenance.DIRECT,
            source_file=control.get("path", ""),
            editable=False,
            reason="imported adjustTimeStep — preserved; no dedicated writer yet",
        )

    # Mesh / AMR
    if dyn.get("path"):
        _add(
            result,
            "mesh_mode",
            "Dynamic Mesh (AMR)",
            FieldProvenance.DERIVED,
            source_file=dyn["path"],
            reason="dynamicMeshDict present",
        )
        if "dynamicFvMesh" in dyn:
            _add(
                result,
                "dynamic_mesh_type",
                dyn["dynamicFvMesh"],
                FieldProvenance.DIRECT,
                source_file=dyn["path"],
                apply_gui=False,
            )
        if "errorEstimator" in dyn:
            # Only map if combo knows it; else case-defined
            est = dyn["errorEstimator"]
            if est in ("densityGradient", "delta", "scaledDensityGradient"):
                _add(
                    result,
                    "refine_indicator_field",
                    est,
                    FieldProvenance.DIRECT,
                    source_file=dyn["path"],
                    editable=False,
                    reason="Imported case setting — preserved in working copy",
                )
            else:
                _add(
                    result,
                    "refine_indicator_field",
                    est,
                    FieldProvenance.CASE_DEFINED,
                    source_file=dyn["path"],
                    apply_gui=False,
                    reason="estimator not in GGUI combo — preserved",
                )
        for of_key, gui_key, cast in (
            ("refineInterval", "refine_interval", int),
            ("lowerRefineLevel", "lower_refine_threshold", float),
            ("unrefineLevel", "unrefine_threshold", float),
            ("nBufferLayers", "n_buffer_layers_dynamic", int),
            ("maxRefinement", "dyn_refine_max", int),
            ("maxCells", "dynamic_max_cells", int),
        ):
            if of_key not in dyn:
                continue
            try:
                val = cast(dyn[of_key])
            except ValueError:
                _add(
                    result,
                    gui_key,
                    dyn[of_key],
                    FieldProvenance.CASE_DEFINED,
                    source_file=dyn["path"],
                    apply_gui=False,
                )
                continue
            _add(
                result,
                gui_key,
                val,
                FieldProvenance.DIRECT,
                source_file=dyn["path"],
                editable=False,
                reason="Imported case setting — preserved in working copy",
            )
        if "dumpLevel" in dyn:
            dump = str(dyn["dumpLevel"]).lower() in ("true", "yes", "on", "1")
            _add(
                result,
                "dump_level",
                dump,
                FieldProvenance.DIRECT,
                source_file=dyn["path"],
                editable=False,
                reason="Imported case setting — preserved in working copy",
            )
        # Startup refinement from setFields
        if setfields.get("level") is not None:
            _add(
                result,
                "charge_seed_mode",
                "Manual",
                FieldProvenance.DERIVED,
                source_file=setfields.get("path", ""),
                reason=f"setFieldsDict level {setfields['level']}",
            )
            _add(
                result,
                "charge_refinement_level",
                int(setfields["level"]),
                FieldProvenance.DIRECT,
                source_file=setfields.get("path", ""),
                editable=False,
                reason="Imported case setting — preserved in working copy",
            )
        if setfields.get("nBufferLayers") is not None:
            _add(
                result,
                "buffer_layers",
                int(setfields["nBufferLayers"]),
                FieldProvenance.DIRECT,
                source_file=setfields.get("path", ""),
                editable=False,
                reason="Imported case setting — preserved in working copy",
            )
    else:
        _add(
            result,
            "mesh_mode",
            "Fixed Mesh",
            FieldProvenance.DERIVED,
            reason="no dynamicMeshDict",
        )

    # Probes — empty unless compatible probes exist (tutorial has none in probes dict)
    probes_path = os.path.join(case_dir, "system", "probes")
    # Also check controlDict functions — skip inventing.
    result.probes = ()
    if not os.path.isfile(probes_path):
        notes.append("No compatible probe table found — probe list left empty.")

    # Summaries
    case_defined = []
    not_recovered = []
    editable = []
    read_only = []
    for key, mf in result.fields.items():
        if mf.provenance == FieldProvenance.CASE_DEFINED:
            case_defined.append(key)
        elif mf.provenance == FieldProvenance.NOT_RECOVERED:
            not_recovered.append(key)
        if mf.editable:
            editable.append(key)
        elif mf.provenance in (
            FieldProvenance.DIRECT,
            FieldProvenance.DERIVED,
            FieldProvenance.CASE_DEFINED,
        ):
            read_only.append(key)
    result.case_defined_keys = tuple(sorted(case_defined))
    result.not_recovered_keys = tuple(sorted(not_recovered))
    result.editable_keys = tuple(sorted(editable))
    result.read_only_keys = tuple(sorted(read_only))
    result.notes = tuple(notes)
    return result
