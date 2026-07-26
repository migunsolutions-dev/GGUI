"""Map an imported axisymmetric blastFoam case onto Cylindrical–2D GUI fields.

Provenance is explicit per field. Recovered values populate the editable GGUI model.
Unknown or unsupported values are not invented from native widget defaults; they
remain unrecovered / case-defined so Initialise Model can require an explicit
user value or block on unsupported source representations.
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
from material_validation import REQUIRED_IMPORTED_PHYSICS_KEYS, UNSUPPORTED_IMPORT_KEYS


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
    unsupported_features: Tuple[Dict[str, str], ...] = ()

    def get(self, key: str) -> Optional[MappedField]:
        return self.fields.get(key)


# Legacy controlDict write targets (retained for optional patch helpers).
# Converted imported cases regenerate via generator_2d; all recovered fields
# are treated as editable GGUI model inputs.
EDITABLE_CONTROL_KEYS = {
    "end_time_s": ("system/controlDict", "endTime"),
    "delta_t": ("system/controlDict", "deltaT"),
    "max_co": ("system/controlDict", "maxCo"),
    "write_control_type": ("system/controlDict", "writeControl"),
    "write_interval_time": ("system/controlDict", "writeInterval"),
    "write_interval_steps": ("system/controlDict", "writeInterval"),
}


def full_sphere_mass_kg(radius_m: float, density_kg_m3: float) -> float:
    """VIPER-compatible nominal full-sphere mass from radius and density."""
    r = float(radius_m)
    rho = float(density_kg_m3)
    return (4.0 / 3.0) * math.pi * (r ** 3) * rho


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

    # Generator2D writes explicit numeric vertices instead of R/H macros.
    vertices = re.search(r"\bvertices\s*\((.*?)\)\s*;", text, re.DOTALL)
    if vertices:
        parsed_vertices: List[Tuple[float, float, float]] = []
        for match in re.finditer(r"\(([^()]+)\)", vertices.group(1)):
            parts = match.group(1).split()
            if len(parts) != 3:
                continue
            try:
                parsed_vertices.append(tuple(float(value) for value in parts))
            except ValueError:
                continue
        if parsed_vertices:
            if "radius" not in out:
                out["radius"] = max(
                    math.hypot(point[0], point[2]) for point in parsed_vertices
                )
                out["radius_from_vertices"] = True
            if "height" not in out:
                ys = [point[1] for point in parsed_vertices]
                out["height"] = max(ys) - min(ys)
                out["height_from_vertices"] = True

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
    out["text"] = text
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


def _parse_decompose(case_dir: str) -> Dict[str, Any]:
    path = os.path.join(case_dir, "system", "decomposeParDict")
    out: Dict[str, Any] = {"path": path if os.path.isfile(path) else ""}
    if not out["path"]:
        return out
    text = _strip(_read_text(path))
    match = re.search(r"\bnumberOfSubdomains\s+(\d+)\s*;", text)
    if match:
        out["cores"] = int(match.group(1))
    return out


def _parse_control_probes(
    control: Dict[str, Any],
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[str, ...]]:
    text = str(control.get("text") or "")
    body = _extract_brace_block(text, "probes2d")
    if body is None or not re.search(r"\btype\s+probes\s*;", body):
        return (), ()
    fields_match = re.search(r"\bfields\s*\(([^)]*)\)\s*;", body, re.DOTALL)
    fields = (
        tuple(fields_match.group(1).split())
        if fields_match
        else ()
    )
    locations = re.search(
        r"\bprobeLocations\s*\((.*?)\)\s*;",
        body,
        re.DOTALL,
    )
    probes: List[Dict[str, Any]] = []
    if locations:
        for index, point in enumerate(
            re.finditer(r"\(([^()]+)\)", locations.group(1)), start=1
        ):
            parts = point.group(1).split()
            if len(parts) != 3:
                continue
            try:
                radius, height, _ = (float(value) for value in parts)
            except ValueError:
                continue
            probes.append({"name": f"P{index}", "radius": radius, "height": height})
    return tuple(probes), fields


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
    def _vector(body: str, key: str) -> Optional[Tuple[float, ...]]:
        match = re.search(rf"\b{re.escape(key)}\s*\(([^)]+)\)", body)
        return (
            tuple(float(value) for value in match.group(1).split())
            if match
            else None
        )

    def _scalar(body: str, key: str) -> Optional[float]:
        match = re.search(
            rf"\b{re.escape(key)}\s+"
            r"([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*;",
            body,
        )
        return float(match.group(1)) if match else None

    def _common(body: str) -> None:
        for key in ("rho", "mass", "LbyD"):
            value = _scalar(body, key)
            if value is not None:
                out[key] = value
        centre = _vector(body, "centre")
        if centre is not None:
            out["centre"] = centre
        level = re.search(r"\blevel\s+(\d+)\s*;", body)
        if level:
            out["level"] = int(level.group(1))
        fv = re.search(r"volScalarFieldValue\s+(\S+)\s+1", body)
        if fv:
            out["phase_field"] = fv.group(1)

    # Exact mass-based region names emitted by the installed Generator2D.
    body = _extract_brace_block(text, "sphericalMassToCell")
    if body is not None:
        out["shape"] = "Sphere"
        out["mass_based"] = True
        _common(body)
        return out

    body = _extract_brace_block(text, "cylindericalMassToCell")
    if body is not None:
        out["shape"] = "Cylinder"
        out["mass_based"] = True
        _common(body)
        direction = _vector(body, "direction")
        if direction is not None:
            out["direction"] = direction
        return out

    body = _extract_brace_block(text, "sphereToCell")
    if body is not None:
        out["shape"] = "Sphere"
        _common(body)
        # Prefer primary radius, not backup radius.
        primary = body.split("backup", 1)[0]
        radius = _scalar(primary, "radius")
        if radius is not None:
            out["radius"] = radius
        return out

    body = _extract_brace_block(text, "cylinderToCell")
    if body is not None:
        out["shape"] = "Cylinder"
        _common(body)
        p1 = _vector(body, "p1")
        p2 = _vector(body, "p2")
        radius = _scalar(body, "radius")
        if p1 is not None:
            out["p1"] = p1
        if p2 is not None:
            out["p2"] = p2
        if p1 is not None and p2 is not None:
            out["centre"] = tuple((a + b) / 2.0 for a, b in zip(p1, p2))
            out["length"] = math.sqrt(sum((b - a) ** 2 for a, b in zip(p1, p2)))
        if radius is not None:
            out["radius"] = radius
        if radius and out.get("length") is not None:
            out["LbyD"] = float(out["length"]) / (2.0 * radius)
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
    use_com = re.search(r"\buseCOM\s+(yes|no|true|false)\s*;", text, re.IGNORECASE)
    if use_com:
        out["useCOM"] = use_com.group(1).lower() in ("yes", "true")
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
        "unrefineInterval",
        "lowerRefineLevel",
        "unrefineLevel",
        "nBufferLayers",
        "maxRefinement",
        "maxCells",
        "dumpLevel",
        "refineProbes",
        "beginUnrefine",
        "enableBalancing",
        "balanceInterval",
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
    decompose = _parse_decompose(case_dir)
    control_probes, control_output_fields = _parse_control_probes(control)
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
            (
                FieldProvenance.DERIVED
                if block.get("radius_from_vertices") or "radius" not in block
                else FieldProvenance.DIRECT
            ),
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
            (
                FieldProvenance.DERIVED
                if block.get("height_from_vertices") or "height" not in block
                else FieldProvenance.DIRECT
            ),
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
    if (
        cell_size is None
        and grading_ok
        and radius is not None
        and height is not None
        and radial
        and vertical
    ):
        radial_size = float(radius) / float(radial)
        axial_size = float(height) / float(vertical)
        tolerance = max(1e-10, 1e-8 * max(abs(radial_size), abs(axial_size), 1.0))
        if math.isclose(radial_size, axial_size, rel_tol=1e-8, abs_tol=tolerance):
            cell_size = 0.5 * (radial_size + axial_size)
            block["derived_cell_size"] = True
    if cell_size is not None and grading_ok and radial and height and radius:
        # Effective sizes when uniform.
        _add(
            result,
            "cell_size",
            float(cell_size),
            (
                FieldProvenance.DERIVED
                if block.get("derived_cell_size")
                else FieldProvenance.DIRECT
            ),
            source_file=block.get("path", ""),
            reason=(
                "uniform R/Nr and H/Nz agree"
                if block.get("derived_cell_size")
                else "blockMeshDict cellSize with uniform simpleGrading"
            ),
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
    setfields_mass = setfields.get("mass")
    charge_rho = setfields.get("rho", phases.get("rho0"))
    if setfields_mass is not None and charge_rho and shape == "Sphere":
        # Generator2D sphericalMassToCell: mass/rho/centre are authoritative.
        charge_r = (
            3.0 * float(setfields_mass) / (4.0 * math.pi * float(charge_rho))
        ) ** (1.0 / 3.0)
        _add(
            result,
            "charge_radius",
            charge_r,
            FieldProvenance.DERIVED,
            source_file=setfields.get("path", ""),
            apply_gui=False,
            reason="derived from sphericalMassToCell mass and rho",
        )
        _add(
            result,
            "mass_kg",
            float(setfields_mass),
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            reason="sphericalMassToCell nominal full-charge mass",
            editable=True,
        )
    elif setfields_mass is not None and charge_rho and shape == "Cylinder":
        aspect = setfields.get("LbyD")
        if aspect and float(aspect) > 0.0:
            volume = float(setfields_mass) / float(charge_rho)
            charge_r = (volume / (2.0 * math.pi * float(aspect))) ** (1.0 / 3.0)
            length = 2.0 * charge_r * float(aspect)
            _add(
                result,
                "charge_aspect",
                float(aspect),
                FieldProvenance.DIRECT,
                source_file=setfields.get("path", ""),
                reason="cylindericalMassToCell LbyD",
            )
            _add(
                result,
                "charge_radius",
                charge_r,
                FieldProvenance.DERIVED,
                source_file=setfields.get("path", ""),
                apply_gui=False,
                reason="derived from cylindericalMassToCell mass/rho/LbyD",
            )
            _add(
                result,
                "charge_length",
                length,
                FieldProvenance.DERIVED,
                source_file=setfields.get("path", ""),
                apply_gui=False,
                reason="derived cylinder length",
            )
        _add(
            result,
            "mass_kg",
            float(setfields_mass),
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            reason="cylindericalMassToCell mass",
            editable=True,
        )
    elif charge_r is not None and shape == "Sphere":
        # VIPER convention: Mass is always the nominal complete-sphere mass.
        # HOB=0 on a reflecting bottom is valid — the computational half-domain
        # contains one hemisphere; do not halve mass/radius.
        off_axis = False
        past_top = False
        if centre is not None:
            cy = float(centre[1])
            if float(centre[0]) ** 2 + (float(centre[2]) if len(centre) > 2 else 0.0) ** 2 > 1e-18:
                off_axis = True
            if height is not None and cy + float(charge_r) > float(height) + 1e-12:
                past_top = True
        _add(
            result,
            "charge_radius",
            float(charge_r),
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            apply_gui=False,
            reason="setFieldsDict sphereToCell radius",
        )
        rho = charge_rho
        if off_axis or past_top:
            _add(
                result,
                "mass_kg",
                None,
                FieldProvenance.NOT_RECOVERED,
                source_file=setfields.get("path", ""),
                reason="sphere centre off-axis or past domain top — mass not derived",
                apply_gui=False,
            )
            notes.append(
                f"Charge radius {charge_r} m loaded; mass left unrecovered "
                "(unsafe centre placement for VIPER full-sphere mass)."
            )
        elif rho:
            mass = full_sphere_mass_kg(float(charge_r), float(rho))
            _add(
                result,
                "mass_kg",
                mass,
                FieldProvenance.DERIVED,
                source_file=setfields.get("path", "") + " + phaseProperties",
                reason="VIPER full-sphere mass = (4/3)π r³ ρ (HOB=0 does not halve)",
                editable=True,
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
    elif charge_r is not None and shape == "Cylinder":
        length = setfields.get("length")
        aspect = setfields.get("LbyD")
        _add(
            result,
            "charge_radius",
            float(charge_r),
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            apply_gui=False,
        )
        if aspect is not None:
            _add(
                result,
                "charge_aspect",
                float(aspect),
                FieldProvenance.DERIVED,
                source_file=setfields.get("path", ""),
                reason="cylinderToCell length / diameter",
            )
        if length is not None and charge_rho:
            mass = math.pi * float(charge_r) ** 2 * float(length) * float(charge_rho)
            _add(
                result,
                "mass_kg",
                mass,
                FieldProvenance.DERIVED,
                source_file=setfields.get("path", ""),
                reason="cylinder volume × phase density",
                editable=True,
            )
        else:
            _add(
                result,
                "mass_kg",
                None,
                FieldProvenance.NOT_RECOVERED,
                reason="cylinder length/density unavailable",
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
        if setfields.get("rho") is not None:
            _add(
                result,
                "rho_charge",
                float(setfields["rho"]),
                FieldProvenance.DIRECT,
                source_file=setfields.get("path", ""),
                reason="mass-based setFields region rho",
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
        elif "rho" in props:
            _add(
                result,
                "rho_charge",
                float(props["rho"]),
                FieldProvenance.DERIVED,
                source_file="material_catalog",
                reason=f"catalog density for {material}",
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
    elif phases.get("useCOM") and centre and len(centre) >= 2:
        _add(
            result,
            "detonation_height",
            float(centre[1]),
            FieldProvenance.DERIVED,
            source_file=phases.get("path", ""),
            reason="phaseProperties useCOM yes → charge centre",
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

    # Boundaries: prefer 0/ field BC types (authoritative physics), then blockMeshDict.
    # Official axisymmetricCharge uses ground type patch + U=slip / p=zeroGradient.
    bm_text = ""
    if block.get("path") and os.path.isfile(block["path"]):
        bm_text = _strip(_read_text(block["path"]))

    def _field_boundary_text(*candidates: str) -> Tuple[str, str]:
        for folder in ("0", "0.orig"):
            for fname in candidates:
                path = os.path.join(case_dir, folder, fname)
                if os.path.isfile(path):
                    return _strip(_read_text(path)), path
        return "", ""

    u_text, u_path = _field_boundary_text("U", "U.orig")
    p_text, p_path = _field_boundary_text("p", "p.orig")

    def _bc_type_for_patch(text: str, patch: str) -> Optional[str]:
        if not text:
            return None
        m = re.search(
            rf"\b{re.escape(patch)}\s*\{{[^}}]*?type\s+(\w+)\s*;",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        return m.group(1) if m else None

    def _boundary_from_fields(names: Tuple[str, ...]) -> Tuple[Optional[str], str]:
        reflecting = {
            "slip",
            "symmetry",
            "symmetryplane",
            "wall",
            "noSlip",
            "fixedValue",
        }
        openish = {
            "pressureWaveTransmissive",
            "waveTransmissive",
            "inletOutlet",
            "zeroGradient",
            "freestream",
        }
        for name in names:
            ut = _bc_type_for_patch(u_text, name)
            pt = _bc_type_for_patch(p_text, name)
            if ut and ut.lower() in {x.lower() for x in reflecting}:
                # zeroGradient on U is not reflecting; slip/wall/symmetry are.
                if ut.lower() == "zerogradient":
                    pass
                else:
                    src = u_path or "0/U"
                    return "Reflecting slip wall", f"{src}:{name} type {ut}"
            if pt and pt in openish:
                src = p_path or "0/p"
                return "Open", f"{src}:{name} type {pt}"
            if ut and ut in openish:
                src = u_path or "0/U"
                return "Open", f"{src}:{name} type {ut}"
        return None, ""

    def _boundary_from_patch(names: Tuple[str, ...], default: Optional[str] = None) -> Optional[str]:
        for name in names:
            m = re.search(
                rf"\b{name}\s*\{{[^}}]*?type\s+(\w+)\s*;",
                bm_text,
                re.IGNORECASE | re.DOTALL,
            )
            if not m:
                continue
            ptype = m.group(1).lower()
            if ptype in ("wall", "symmetryplane", "symmetry"):
                return "Reflecting slip wall"
            if ptype in ("patch", "inletoutlet", "wavetransmissive", "zerogradient"):
                # Ambiguous: many reflecting grounds are still type patch.
                return None
        return default

    bottom_names = ("ground", "bottom", "floor")
    outer_names = ("outer", "outerRadius", "outlet", "sides")
    top_names = ("top", "atmosphere", "sky")

    bottom_bc, bottom_reason = _boundary_from_fields(bottom_names)
    if bottom_bc is None:
        bottom_bc = _boundary_from_patch(bottom_names, "Reflecting slip wall")
        bottom_reason = "blockMeshDict patch type → GGUI Reflecting (default/ground)"
        if bottom_bc == "Open":
            bottom_reason = "blockMeshDict patch type → GGUI Open"

    outer_bc, outer_reason = _boundary_from_fields(outer_names)
    if outer_bc is None:
        outer_bc = _boundary_from_patch(outer_names, "Open") or "Open"
        outer_reason = "blockMeshDict / default → Open"

    top_bc, top_reason = _boundary_from_fields(top_names)
    if top_bc is None:
        # Tutorial often merges top into outlet — inherit Open when unrecovered.
        mapped_top = _boundary_from_patch(top_names, None)
        top_bc = mapped_top if mapped_top is not None else "Open"
        top_reason = "blockMeshDict / default → Open (top may share outlet)"

    for key, val, label, reason in (
        ("bottom_boundary", bottom_bc, "ground/bottom", bottom_reason),
        ("outer_boundary", outer_bc, "outlet/outer", outer_reason),
        ("top_boundary", top_bc, "top", top_reason),
    ):
        if val is None:
            _add(
                result,
                key,
                None,
                FieldProvenance.NOT_RECOVERED,
                reason=f"OpenFOAM BC for {label} not mapped",
                apply_gui=False,
            )
        else:
            _add(
                result,
                key,
                val,
                FieldProvenance.DERIVED,
                source_file=(u_path or p_path or block.get("path", "")),
                editable=True,
                reason=reason or f"boundary mapping for {label}",
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
            editable=True,
            reason="Recovered for editable GGUI model",
        )
    if "purgeWrite" in control:
        try:
            _add(
                result,
                "cycle_write",
                int(float(control["purgeWrite"])),
                FieldProvenance.DIRECT,
                source_file=control.get("path", ""),
                editable=True,
                reason="controlDict purgeWrite",
            )
        except ValueError:
            pass
    if decompose.get("cores") is not None:
        _add(
            result,
            "cores",
            int(decompose["cores"]),
            FieldProvenance.DIRECT,
            source_file=decompose.get("path", ""),
            editable=True,
            reason="decomposeParDict numberOfSubdomains",
        )

    # Mesh / AMR
    dynamic_mesh = bool(
        dyn.get("path") and dyn.get("dynamicFvMesh") != "staticFvMesh"
    )
    if dynamic_mesh:
        _add(
            result,
            "mesh_mode",
            "Dynamic Mesh (AMR)",
            FieldProvenance.DERIVED,
            source_file=dyn["path"],
            reason="adaptive dynamicMeshDict",
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
            ("unrefineInterval", "unrefine_interval", int),
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
                editable=True,
                reason="Recovered for editable GGUI model",
            )
        if "dumpLevel" in dyn:
            dump = str(dyn["dumpLevel"]).lower() in ("true", "yes", "on", "1")
            _add(
                result,
                "dump_level",
                dump,
                FieldProvenance.DIRECT,
                source_file=dyn["path"],
                editable=True,
                reason="Recovered for editable GGUI model",
            )
        if "refineProbes" in dyn:
            rp = str(dyn["refineProbes"]).lower() in ("true", "yes", "on", "1")
            _add(
                result,
                "refine_probes",
                rp,
                FieldProvenance.DIRECT,
                source_file=dyn["path"],
                editable=True,
                reason="dynamicMeshDict Switch refineProbes",
            )
        if "beginUnrefine" in dyn:
            try:
                _add(
                    result,
                    "begin_unrefine",
                    float(dyn["beginUnrefine"]),
                    FieldProvenance.DIRECT,
                    source_file=dyn["path"],
                    editable=True,
                    reason="Recovered for editable GGUI model",
                )
            except ValueError:
                pass
        if "enableBalancing" in dyn or "balance" in dyn:
            raw_bal = dyn.get("enableBalancing", dyn.get("balance"))
            if raw_bal is not None:
                bal = str(raw_bal).lower() in ("true", "yes", "on", "1")
                _add(
                    result,
                    "enable_balancing",
                    bal,
                    FieldProvenance.DIRECT,
                    source_file=dyn["path"],
                    editable=True,
                    reason="Recovered for editable GGUI model",
                )
        if "balanceInterval" in dyn:
            try:
                _add(
                    result,
                    "balance_interval",
                    int(dyn["balanceInterval"]),
                    FieldProvenance.DIRECT,
                    source_file=dyn["path"],
                    editable=True,
                    reason="Recovered for editable GGUI model",
                )
            except ValueError:
                pass
        # Startup refinement from setFields
        if setfields.get("level") is not None:
            _add(
                result,
                "charge_seed_mode",
                "Manual",
                FieldProvenance.DERIVED,
                source_file=setfields.get("path", ""),
                reason=f"setFieldsDict level {setfields['level']}",
                editable=True,
            )
            _add(
                result,
                "charge_refinement_level",
                int(setfields["level"]),
                FieldProvenance.DIRECT,
                source_file=setfields.get("path", ""),
                editable=True,
                reason="Recovered for editable GGUI model",
            )
    else:
        _add(
            result,
            "mesh_mode",
            "Fixed Mesh",
            FieldProvenance.DERIVED,
            source_file=dyn.get("path", ""),
            reason=(
                "dynamicFvMesh staticFvMesh"
                if dyn.get("path")
                else "no dynamicMeshDict"
            ),
        )

    if shape and setfields.get("level") is None:
        _add(
            result,
            "charge_seed_mode",
            "Off",
            FieldProvenance.DERIVED,
            source_file=setfields.get("path", ""),
            editable=True,
            reason="setFields region has no refineInternal level",
        )
    if setfields.get("nBufferLayers") is not None:
        _add(
            result,
            "buffer_layers",
            int(setfields["nBufferLayers"]),
            FieldProvenance.DIRECT,
            source_file=setfields.get("path", ""),
            editable=True,
            reason="setFieldsDict nBufferLayers",
        )

    # Generator2D writes native probes in controlDict/functions/probes2d.
    result.probes = control_probes
    if control_probes:
        result.gui_values["probes"] = list(control_probes)
    if control_output_fields:
        result.gui_values["output_fields"] = tuple(control_output_fields)
    if not control_probes:
        notes.append("No compatible probe table found — probe list left empty.")

    # Converted imported cases are fully editable GGUI models: mark every
    # recovered DIRECT/DERIVED field that applies to the GUI as editable.
    for key, mf in list(result.fields.items()):
        if (
            mf.provenance in (FieldProvenance.DIRECT, FieldProvenance.DERIVED)
            and mf.displayed_value is not None
            and (mf.gui_key or key) in result.gui_values
        ):
            mf.editable = True

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
        elif mf.provenance == FieldProvenance.CASE_DEFINED:
            read_only.append(key)
    result.case_defined_keys = tuple(sorted(case_defined))
    result.not_recovered_keys = tuple(sorted(not_recovered))
    result.editable_keys = tuple(sorted(editable))
    result.read_only_keys = tuple(sorted(read_only))

    unsupported: List[Dict[str, str]] = []
    for key, mf in result.fields.items():
        gui_key = mf.gui_key or key
        if mf.provenance != FieldProvenance.CASE_DEFINED:
            continue
        if gui_key not in UNSUPPORTED_IMPORT_KEYS and key not in UNSUPPORTED_IMPORT_KEYS:
            continue
        affected = (
            "Direct Charge physics used by Initialise Model"
            if gui_key in REQUIRED_IMPORTED_PHYSICS_KEYS or key in REQUIRED_IMPORTED_PHYSICS_KEYS
            else "the regenerated mesh / boundary model"
        )
        unsupported.append(
            {
                "field": gui_key,
                "source_feature": (
                    f"{gui_key} in the imported blastFoam case"
                    + (f" ({mf.source_file})" if mf.source_file else "")
                ),
                "reason": mf.reason
                or "the source value cannot be represented identically in GGUI",
                "affected": affected,
            }
        )
    result.unsupported_features = tuple(unsupported)

    notes.append(
        "Converted editable GGUI model: Initialise Model regenerates a fresh case "
        "via generator_2d (source remains read-only)."
    )
    result.notes = tuple(notes)
    return result
