"""
Context-aware Load Summary classification and presentation.

Report-only: does not alter CaseInputs3D values, generator output, or load_case
model keys. Builds classification metadata for the Load Summary popup/copy text.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Internal classifications (replace Filled / Not filled / Not mapped).
LOADED_EXACT = "LOADED_EXACT"
LOADED_CONVERTED = "LOADED_CONVERTED"
DERIVED = "DERIVED"
ALTERNATIVE_MAPPING = "ALTERNATIVE_MAPPING"
SOLVER_DEFAULT = "SOLVER_DEFAULT"
PRESERVED_UNCHANGED = "PRESERVED_UNCHANGED"
CONFLICT_AMBIGUOUS = "CONFLICT_AMBIGUOUS"
UNSUPPORTED_LOST = "UNSUPPORTED_LOST"
NOT_APPLICABLE = "NOT_APPLICABLE"

ALL_CLASSIFICATIONS = (
    LOADED_EXACT,
    LOADED_CONVERTED,
    DERIVED,
    ALTERNATIVE_MAPPING,
    SOLVER_DEFAULT,
    PRESERVED_UNCHANGED,
    CONFLICT_AMBIGUOUS,
    UNSUPPORTED_LOST,
    NOT_APPLICABLE,
)

_COUNT_LABELS = {
    LOADED_EXACT: "Loaded exactly",
    LOADED_CONVERTED: "Converted",
    DERIVED: "Derived",
    ALTERNATIVE_MAPPING: "Alternative mappings",
    SOLVER_DEFAULT: "Using solver defaults",
    PRESERVED_UNCHANGED: "Preserved unchanged",
    CONFLICT_AMBIGUOUS: "Conflicts / ambiguous",
    UNSUPPORTED_LOST: "Unsupported / lost",
}

# Verified blastFoam 6.2.0 (commit 4e6ee07a) defaults from
# src/errorEstimators/errorEstimator/errorEstimator.C:
#   upperRefine_  = dict.lookupOrDefault("upperRefineLevel", great);
#   upperUnrefine_ = dict.lookupOrDefault("upperUnrefineLevel", great);
_BLASTFOAM_620_UPPER_REFINE_DEFAULT = "Foam::great (lookupOrDefault)"
_BLASTFOAM_620_COMMIT = "4e6ee07a0c1fc4629ee7206804f4f1fe802ec64c"

# Mirror of generator_3d.jwl_lib (report-only comparison; do not mutate DB).
_BUILTIN_JWL = {
    "TNT": {"A": 373.77e9, "B": 3.7471e9, "R1": 4.15, "R2": 0.90, "omega": 0.35},
    "C4": {"A": 609.77e9, "B": 12.95e9, "R1": 4.50, "R2": 1.40, "omega": 0.25},
    "PETN": {"A": 617.0e9, "B": 16.9e9, "R1": 4.40, "R2": 1.20, "omega": 0.25},
    "ANFO": {"A": 49.46e9, "B": 1.89e9, "R1": 3.90, "R2": 1.10, "omega": 0.33},
}

_AMR_ONLY_KEYS = frozenset(
    {
        "refine_interval",
        "lower_refine_threshold",
        "unrefine_threshold",
        "n_buffer_layers_dynamic",
        "refine_indicator_field",
        "enable_balancing",
        "dynamic_max_cells",
        "begin_unrefine",
        "upper_refine_level",
        "upper_unrefine_level",
        "balance_interval",
        "dyn_refine_min",
        "dyn_refine_max",
    }
)

_OUTER_INACTIVE_KEYS = frozenset(
    {
        "outside_extent",
        "transition_cells",
        "charge_outer_refine_level",
        "charge_outer_refine_min",
        "charge_outer_refine_max",
    }
)

_SOURCE_HINTS: Dict[str, Tuple[str, str]] = {
    "min_point": ("system/blockMeshDict", "vertices"),
    "max_point": ("system/blockMeshDict", "vertices"),
    "cell_size": ("system/blockMeshDict", "hex division / vertices"),
    "boundaries": ("system/blockMeshDict", "boundary/*/type"),
    "cfl_value": ("system/controlDict", "maxCo"),
    "end_time_s": ("system/controlDict", "endTime"),
    "delta_t": ("system/controlDict", "deltaT"),
    "write_control_type": ("system/controlDict", "writeControl"),
    "write_interval_time": ("system/controlDict", "writeInterval"),
    "write_interval_steps": ("system/controlDict", "writeInterval"),
    "cycle_write": ("system/controlDict", "cycleWrite"),
    "enable_post_processing": ("system/controlDict", "functions"),
    "mass_kg": ("system/setFieldsDict", "regions/*/mass"),
    "rho_charge": ("system/setFieldsDict", "regions/*/rho"),
    "charge_center": ("system/setFieldsDict", "regions/*/centre"),
    "charge_shape": ("system/setFieldsDict", "regions/* (MassToCell/ToCell type)"),
    "charge_lbyd": ("system/setFieldsDict", "regions/*/LbyD"),
    "cylinder_axis": ("system/setFieldsDict", "regions/*/direction"),
    "buffer_layers": ("system/setFieldsDict", "nBufferLayers"),
    "charge_refinement_level": ("system/setFieldsDict", "regions/*/level"),
    "charge_seed_mode": ("system/setFieldsDict", "regions/*/refineInternal"),
    "charge_capture_radius": ("system/setFieldsDict", "regions/*/backup/radius"),
    "charge_backup_radius_override": ("system/setFieldsDict", "regions/*/backup/radius"),
    "charge_backup_radius_factor": ("system/setFieldsDict", "backup/radius vs computed charge radius"),
    "charge_radius": ("system/setFieldsDict", "derived from mass/rho/(LbyD) or region radius"),
    "activation_model": ("constant/phaseProperties", "activationModel"),
    "activation_model_ui": ("constant/phaseProperties", "activationModel"),
    "ignition_radius": ("constant/phaseProperties", "initiation/radius"),
    "ignition_mode": ("constant/phaseProperties", "initiation/useCOM"),
    "initiation_point": ("constant/phaseProperties", "initiation/useCOM or initiationPoints"),
    "eos_model": ("constant/phaseProperties", "products/thermoType/equationOfState"),
    "thermo_model": ("constant/phaseProperties", "products/thermoType/thermo"),
    "thermo_model_air": ("constant/phaseProperties", "air/thermodynamics|thermoType"),
    "material_name": ("constant/phaseProperties", "reactants/equationOfState/rho0 (matched)"),
    "custom_material_props": ("constant/phaseProperties", "products/equationOfState/*"),
    "p_atm": ("0/p.orig|0/p", "internalField"),
    "t_atm": ("0/T.orig|0/T", "internalField"),
    "cores": ("system/decomposeParDict", "numberOfSubdomains"),
    "decomposition_method": ("system/decomposeParDict", "method"),
    "decomposition_simple_n": ("system/decomposeParDict", "simpleCoeffs/n"),
    "decomposition_simple_delta": ("system/decomposeParDict", "simpleCoeffs/delta"),
    "enable_dyn_refine": ("constant/dynamicMeshDict", "dynamicFvMesh"),
    "enable_local_refinement": ("constant/dynamicMeshDict", "dynamicFvMesh"),
    "refine_max": ("constant/dynamicMeshDict", "maxRefinement"),
    "dyn_refine_max": ("constant/dynamicMeshDict", "maxRefinement"),
    "refine_interval": ("constant/dynamicMeshDict", "refineInterval"),
    "lower_refine_threshold": ("constant/dynamicMeshDict", "lowerRefineLevel"),
    "unrefine_threshold": ("constant/dynamicMeshDict", "unrefineLevel"),
    "n_buffer_layers_dynamic": ("constant/dynamicMeshDict", "nBufferLayers"),
    "refine_indicator_field": ("constant/dynamicMeshDict", "errorEstimator"),
    "dynamic_max_cells": ("constant/dynamicMeshDict", "maxCells"),
    "begin_unrefine": ("constant/dynamicMeshDict", "beginUnrefine"),
    "upper_refine_level": ("constant/dynamicMeshDict", "upperRefineLevel"),
    "upper_unrefine_level": ("constant/dynamicMeshDict", "upperUnrefineLevel"),
    "enable_balancing": ("constant/dynamicMeshDict", "enableBalancing"),
    "balance_interval": ("constant/dynamicMeshDict", "loadBalance/balanceInterval"),
    "dyn_refine_min": ("(no solver key)", "legacy GUI only"),
    "refine_min": ("(no adaptiveFvMesh key)", "legacy GUI only"),
    "stl_obstacles": ("system/snappyHexMeshDict", "geometry/*/triSurfaceMesh"),
    "enable_obstacle_refine": ("system/snappyHexMeshDict", "refinementSurfaces"),
    "obstacle_refine_min": ("system/snappyHexMeshDict", "refinementSurfaces/*/level"),
    "obstacle_refine_max": ("system/snappyHexMeshDict", "refinementSurfaces/*/level"),
    "outside_extent": ("system/snappyHexMeshDict", "geometry/chargeRefineOuter"),
    "charge_outer_refine_enable": ("system/snappyHexMeshDict", "refinementRegions/chargeRefineOuter"),
    "charge_outer_refine_level": ("system/snappyHexMeshDict", "refinementRegions/chargeRefineOuter/levels"),
    "charge_outer_refine_min": ("system/snappyHexMeshDict", "refinementRegions/chargeRefineOuter/levels"),
    "charge_outer_refine_max": ("system/snappyHexMeshDict", "refinementRegions/chargeRefineOuter/levels"),
    "charge_length": ("system/setFieldsDict", "derived from mass, rho, LbyD"),
    "charge_width": ("system/setFieldsDict", "boxToCell / cuboid dims"),
    "charge_height": ("system/setFieldsDict", "boxToCell / cuboid dims"),
    "charge_capture_factor": ("system/setFieldsDict", "backup radius factor (auto)"),
    "transition_cells": ("(GUI policy)", "outer transition cells"),
}

# Keys inferred from dictionary structure / presence (not leaf scalar copy).
_STRUCTURE_INFERRED = frozenset(
    {
        "enable_dyn_refine",
        "enable_local_refinement",
        "charge_seed_mode",
        "charge_shape",
        "enable_post_processing",
        "enable_obstacle_refine",
        "charge_outer_refine_enable",
        "charge_capture_mode",
    }
)


@dataclass
class FieldClassification:
    gui_key: str
    classification: str
    source_file: str = ""
    source_path: str = ""
    source_value: Any = None
    imported_value: Any = None
    regenerated_value: Any = None
    rule: str = ""
    regenerated_destination: str = ""
    notes: str = ""
    applicable: bool = True
    difference_kind: str = ""  # "numeric" | "textual_formatting" | ""
    value_origin: str = ""  # loader_payload | report_interpretation | gui_default | regen_proof

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoadSummaryReport:
    case_dir: str
    fields: List[FieldClassification] = field(default_factory=list)
    filled: List[str] = field(default_factory=list)
    not_filled: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    lost_leaves: List[Dict[str, Any]] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        c = {k: 0 for k in ALL_CLASSIFICATIONS}
        for f in self.fields:
            if f.classification == NOT_APPLICABLE:
                continue
            c[f.classification] = c.get(f.classification, 0) + 1
        return c

    def visible_fields(self) -> List[FieldClassification]:
        return [f for f in self.fields if f.classification != NOT_APPLICABLE]

    def to_load_summary_dict(self) -> Dict[str, Any]:
        counts = self.counts()
        return {
            "filled": list(self.filled),
            "not_filled": [list(t) for t in self.not_filled],
            "notes": list(self.notes),
            "schema_version": 3,
            "classifications": [f.to_dict() for f in self.fields],
            "counts": {k: counts.get(k, 0) for k in ALL_CLASSIFICATIONS if k != NOT_APPLICABLE},
            "lost_leaves": list(self.lost_leaves),
            "provenance": list(self.provenance),
            "unsupported": {},
        }


def _read_text(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _string_val(text: str, key: str) -> Optional[str]:
    m = re.search(rf"\b{re.escape(key)}\s+(\S+)\s*;", text)
    return m.group(1) if m else None


def _scalar(text: str, key: str) -> Optional[float]:
    m = re.search(rf"\b{re.escape(key)}\s+([\d\.eE\+\-]+)\s*;", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _scalar_raw(text: str, key: str) -> Optional[str]:
    m = re.search(rf"\b{re.escape(key)}\s+([\d\.eE\+\-]+)\s*;", text)
    return m.group(1) if m else None


def _key_present(text: str, key: str) -> bool:
    """True if ``key <token>;`` appears as an OpenFOAM leaf assignment."""
    return re.search(rf"\b{re.escape(key)}\s+\S+\s*;", text) is not None


def _find_block(text: str, block_name: str) -> Optional[str]:
    pattern = rf"\b{re.escape(block_name)}\s*\{{"
    m = re.search(pattern, text)
    if not m:
        return None
    start = m.end()
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


def _cylinder_dims_from_mass(mass: float, rho: float, lbyd: float) -> Tuple[float, float]:
    vol = mass / rho
    r = (vol / (2.0 * math.pi * lbyd)) ** (1.0 / 3.0)
    length = 2.0 * r * lbyd
    return r, length


def _sphere_radius_from_mass(mass: float, rho: float) -> float:
    vol = mass / rho
    return (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)


def _nearly_equal(a: Any, b: Any, rel: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return abs(fa - fb) <= max(abs_tol, rel * max(abs(fa), abs(fb), 1.0))


def _numeric_differs(a: Any, b: Any) -> bool:
    try:
        return not _nearly_equal(a, b)
    except Exception:
        return str(a) != str(b)


def _jwl_emit_4g(val: float) -> str:
    """Match generator_3d emission: ``A {j['A']:.4g}`` (OpenFOAM-ish text)."""
    return f"{float(val):.4g}".replace("e+", "e").replace("E+", "e")


def _legacy_not_filled_reason(key: str, charge_shape: str, is_remap: bool) -> str:
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


def _source_for(key: str) -> Tuple[str, str]:
    return _SOURCE_HINTS.get(key, ("", ""))


def _case_context(out: Dict[str, Any], case_dir: str) -> Dict[str, Any]:
    shape = out.get("charge_shape") or "Sphere"
    activation = (out.get("activation_model") or "").lower()
    is_remap = activation == "none"
    dyn = bool(out.get("enable_dyn_refine") or out.get("enable_local_refinement"))
    write_ctl = out.get("write_control_type") or ""
    material = out.get("material_name") or ""
    outer_on = bool(out.get("charge_outer_refine_enable"))
    has_obstacles = bool(out.get("stl_obstacles"))
    decomp = (out.get("decomposition_method") or "").lower()

    use_com: Optional[bool] = None
    pmin_raw: Optional[str] = None
    pmin_val: Optional[float] = None
    jwl_a_raw: Optional[str] = None
    jwl_a_val: Optional[float] = None
    write_control_raw: Optional[str] = None
    feature_levels: List[int] = []
    has_initiation_points = False

    phase = _read_text(os.path.join(case_dir, "constant", "phaseProperties"))
    if phase:
        init = _find_block(phase, "initiation")
        if init:
            raw = _string_val(init, "useCOM")
            if raw is not None:
                use_com = raw.lower() in ("yes", "true", "on")
            pmin_raw = _scalar_raw(init, "pMin")
            pmin_val = _scalar(init, "pMin")
            if re.search(r"\bpoints\s*\(", init) or re.search(r"\binitiationPoints\b", init):
                has_initiation_points = True
        products = _find_block(phase, "products")
        if products:
            eos = _find_block(products, "equationOfState")
            if eos:
                jwl_a_raw = _scalar_raw(eos, "A")
                jwl_a_val = _scalar(eos, "A")

    ctrl = _read_text(os.path.join(case_dir, "system", "controlDict"))
    if ctrl:
        write_control_raw = _string_val(ctrl, "writeControl")

    # Optional AMR key presence in original dynamicMeshDict (for omission-parity proof).
    dm_keys_present: Dict[str, bool] = {
        "maxCells": False,
        "beginUnrefine": False,
        "enableBalancing": False,
        "balanceInterval": False,
    }
    source_enable_balancing: Optional[str] = None
    dm = _read_text(os.path.join(case_dir, "constant", "dynamicMeshDict"))
    if dm:
        dm_keys_present["maxCells"] = _key_present(dm, "maxCells")
        dm_keys_present["beginUnrefine"] = _key_present(dm, "beginUnrefine")
        source_enable_balancing = _string_val(dm, "enableBalancing")
        dm_keys_present["enableBalancing"] = source_enable_balancing is not None
        lb = _find_block(dm, "loadBalance")
        if lb and _key_present(lb, "balanceInterval"):
            dm_keys_present["balanceInterval"] = True
        elif _key_present(dm, "balanceInterval"):
            dm_keys_present["balanceInterval"] = True

    # Balancing is on only when loader set True, or source enableBalancing is true.
    balancing_on = (
        "enable_balancing" in out
        and out.get("enable_balancing") is not None
        and bool(out.get("enable_balancing"))
    )
    if source_enable_balancing is not None and str(source_enable_balancing).lower() in (
        "true",
        "yes",
        "on",
    ):
        balancing_on = True

    orphan_surfaces: List[Dict[str, Any]] = []
    snappy = _read_text(os.path.join(case_dir, "system", "snappyHexMeshDict"))
    if snappy:
        geom = _find_block(snappy, "geometry") or ""
        ref_surf = _find_block(snappy, "refinementSurfaces") or ""
        for m in re.finditer(
            r"(\w+)\s*\{([^}]*type\s+searchable\w+[^}]*)\}",
            geom,
            re.DOTALL,
        ):
            name = m.group(1)
            body = m.group(2)
            if name == "chargeRefineOuter":
                continue
            gtype_m = re.search(r"type\s+(searchable\w+)\s*;", body)
            gtype = gtype_m.group(1) if gtype_m else "searchable"
            level_m = re.search(
                rf"\b{re.escape(name)}\s*\{{[^}}]*level\s*\(\s*(\d+)\s+(\d+)\s*\)",
                ref_surf,
                re.DOTALL,
            )
            if level_m:
                orphan_surfaces.append(
                    {
                        "name": name,
                        "type": gtype,
                        "level": (int(level_m.group(1)), int(level_m.group(2))),
                        "geometry_body": body.strip(),
                    }
                )
        for fm in re.finditer(
            r"features\s*\((.*?)\)\s*;",
            snappy,
            re.DOTALL,
        ):
            for lm in re.finditer(r"\blevel\s+(\d+)\s*;", fm.group(1)):
                feature_levels.append(int(lm.group(1)))

    return {
        "charge_shape": shape,
        "is_remap": is_remap,
        "dyn_mesh": dyn,
        "write_control_type": write_ctl,
        "write_control_raw": write_control_raw,
        "material_name": material,
        "outer_on": outer_on,
        "has_obstacles": has_obstacles,
        "decomposition_method": decomp,
        "use_com": use_com,
        "orphan_surfaces": orphan_surfaces,
        "pmin_raw": pmin_raw,
        "pmin_val": pmin_val,
        "jwl_a_raw": jwl_a_raw,
        "jwl_a_val": jwl_a_val,
        "feature_levels": feature_levels,
        "has_initiation_points": has_initiation_points,
        "manual_capture": (out.get("charge_capture_mode") or "") == "manual",
        "dm_keys_present": dm_keys_present,
        "balancing_on": balancing_on,
        "has_case_dir": bool(case_dir) and os.path.isdir(case_dir),
    }


def _is_applicable(key: str, ctx: Dict[str, Any]) -> Tuple[bool, str]:
    shape = ctx["charge_shape"]
    if key in ("charge_width", "charge_height") and shape != "Cuboid":
        return False, f"Cuboid-only dimension; charge_shape={shape}"
    if key == "charge_length" and shape == "Sphere":
        return False, f"Not used for Sphere; charge_shape={shape}"
    if key == "charge_lbyd" and shape != "Cylinder":
        return False, f"Cylinder-only; charge_shape={shape}"
    if key == "cylinder_axis" and shape != "Cylinder":
        return False, f"Cylinder-only; charge_shape={shape}"
    if key in ("initiation_point", "ignition_mode", "ignition_radius") and ctx["is_remap"]:
        return False, "Remap / activationModel=none: ignition path not used"
    if key in _AMR_ONLY_KEYS and not ctx["dyn_mesh"]:
        return False, "Fixed Mesh: runtime AMR fields not applicable"
    if key == "write_interval_steps" and ctx["write_control_type"] in (
        "adjustableRunTime",
        "runTime",
    ):
        return False, "writeControl is time-based; write_interval_steps not applicable"
    if key == "write_interval_time" and ctx["write_control_type"] == "timeStep":
        return False, "writeControl=timeStep; write_interval_time not applicable"
    if key == "charge_capture_factor" and ctx.get("manual_capture"):
        return False, "Manual capture radius present; auto capture factor not used"
    # Outer inactive: hide extent/transition/level fields; keep enable flag.
    if not ctx["outer_on"] and key in _OUTER_INACTIVE_KEYS:
        return False, "Outer band disabled/absent: inactive outer field not applicable"
    if key in (
        "stl_obstacles",
        "obstacle_refine_min",
        "obstacle_refine_max",
        "obstacle_feature_angle",
        "obstacle_cells_between_levels",
        "obstacle_snap_iter",
        "obstacle_feature_snap_iter",
    ) and not ctx["has_obstacles"]:
        return False, "No obstacles in case"
    # balanceInterval only applies when balancing is explicitly enabled.
    if key == "balance_interval" and not ctx.get("balancing_on"):
        return False, "Balancing disabled/absent: balance_interval not applicable"
    return True, ""


def _append_prov(
    provenance: List[Dict[str, Any]],
    *,
    gui_key: str,
    source_file: str,
    source_path: str,
    source_value: Any,
    mapping_type: str,
    imported_value: Any,
    transformation: Optional[str] = None,
) -> None:
    provenance.append(
        {
            "gui_key": gui_key,
            "source_file": source_file,
            "source_dictionary_path": source_path,
            "source_value": _jsonable(source_value),
            "mapping_type": mapping_type,
            "transformation": transformation,
            "imported_value": _jsonable(imported_value),
        }
    )


def _add_parity_leafs(
    fields: List[FieldClassification],
    lost_leaves: List[Dict[str, Any]],
    provenance: List[Dict[str, Any]],
    out: Dict[str, Any],
    ctx: Dict[str, Any],
) -> None:
    """Leaf-level original→regenerated differences (report-only)."""

    # --- feature edge level 0 -> obstacle_refine_min (typically 1) ---
    feat_levels = ctx.get("feature_levels") or []
    if feat_levels and ctx.get("has_obstacles"):
        orig = feat_levels[0]
        # Generator: when obstacle refine enabled, feat_level = obstacle_refine_min
        regen = out.get("obstacle_refine_min")
        if regen is None:
            regen = 1
        if int(orig) != int(regen):
            fields.append(
                FieldClassification(
                    gui_key="[parity] snappyHexMeshDict/features/*/level",
                    classification=ALTERNATIVE_MAPPING,
                    source_file="system/snappyHexMeshDict",
                    source_path="castellatedMeshControls/features/*/level",
                    source_value=orig,
                    imported_value=regen,
                    regenerated_value=int(regen),
                    rule=(
                        "Feature edge level is not a dedicated GUI field; when obstacle "
                        "refinement is enabled the generator sets features level = "
                        "obstacle_refine_min (from refinementSurfaces), replacing the "
                        "original features level."
                    ),
                    regenerated_destination="system/snappyHexMeshDict/features/*/level",
                    notes="Genuine numeric change (not textual formatting).",
                    difference_kind="numeric",
                    value_origin="report_interpretation",
                )
            )
            _append_prov(
                provenance,
                gui_key="[parity] snappyHexMeshDict/features/*/level",
                source_file="system/snappyHexMeshDict",
                source_path="features/*/level",
                source_value=orig,
                mapping_type=ALTERNATIVE_MAPPING,
                imported_value=regen,
                transformation="features level <- obstacle_refine_min",
            )

    # --- initiation/pMin <- p_atm from 0/p ---
    if ctx.get("pmin_val") is not None:
        p_atm = out.get("p_atm")
        if p_atm is not None and _numeric_differs(ctx["pmin_val"], p_atm):
            fields.append(
                FieldClassification(
                    gui_key="[parity] phaseProperties/initiation/pMin",
                    classification=ALTERNATIVE_MAPPING,
                    source_file="constant/phaseProperties",
                    source_path="initiation/pMin",
                    source_value=ctx.get("pmin_raw") or ctx["pmin_val"],
                    imported_value=p_atm,
                    regenerated_value=p_atm,
                    rule=(
                        "Generator writes initiation/pMin from CaseInputs3D.p_atm "
                        "(loaded from 0/p internalField), not from the original "
                        "phaseProperties/initiation/pMin leaf."
                    ),
                    regenerated_destination="constant/phaseProperties/initiation/pMin",
                    notes=(
                        f"Original pMin={ctx.get('pmin_raw')}; loader p_atm={p_atm} "
                        f"from 0/p; regenerated pMin uses p_atm. Genuine numeric change."
                    ),
                    difference_kind="numeric",
                    value_origin="loader_payload",
                )
            )
            _append_prov(
                provenance,
                gui_key="[parity] phaseProperties/initiation/pMin",
                source_file="constant/phaseProperties",
                source_path="initiation/pMin",
                source_value=ctx.get("pmin_raw"),
                mapping_type=ALTERNATIVE_MAPPING,
                imported_value=p_atm,
                transformation="pMin <- p_atm (0/p internalField)",
            )

    # --- JWL A via built-in material + .4g emission ---
    mat = ctx.get("material_name") or ""
    if mat in _BUILTIN_JWL and ctx.get("jwl_a_val") is not None:
        builtin_a = _BUILTIN_JWL[mat]["A"]
        orig_a = ctx["jwl_a_val"]
        regen_text = _jwl_emit_4g(builtin_a)
        try:
            regen_num = float(regen_text)
        except ValueError:
            regen_num = builtin_a
        # Always surface when regenerated numeric text differs from original.
        if _numeric_differs(orig_a, regen_num) or (ctx.get("jwl_a_raw") and ctx["jwl_a_raw"] != regen_text):
            kind = "numeric" if _numeric_differs(orig_a, regen_num) else "textual_formatting"
            if kind == "numeric" or (ctx.get("jwl_a_raw") and ctx["jwl_a_raw"] != regen_text and kind == "textual_formatting"):
                # Prefer reporting when regenerated value string/number differs.
                fields.append(
                    FieldClassification(
                        gui_key="[parity] phaseProperties/products/equationOfState/A",
                        classification=ALTERNATIVE_MAPPING,
                        source_file="constant/phaseProperties",
                        source_path="products/equationOfState/A",
                        source_value=ctx.get("jwl_a_raw") or orig_a,
                        imported_value=out.get("jwl_A", orig_a),
                        regenerated_value=regen_text,
                        rule=(
                            f"Material detected as built-in {mat}; generator emits "
                            f"jwl_lib['{mat}']['A'] with format .4g "
                            f"({builtin_a} -> '{regen_text}') instead of preserving "
                            "the original dictionary token."
                        ),
                        regenerated_destination="constant/phaseProperties/products/equationOfState/A",
                        notes=(
                            f"Built-in {mat} A={builtin_a}; original token="
                            f"{ctx.get('jwl_a_raw')}; regenerated={regen_text}. "
                            + (
                                "Genuine numeric change after .4g rounding."
                                if kind == "numeric"
                                else "Textual formatting only."
                            )
                        ),
                        difference_kind=kind,
                        value_origin="report_interpretation",
                    )
                )
                _append_prov(
                    provenance,
                    gui_key="[parity] phaseProperties/products/equationOfState/A",
                    source_file="constant/phaseProperties",
                    source_path="products/equationOfState/A",
                    source_value=ctx.get("jwl_a_raw"),
                    mapping_type=ALTERNATIVE_MAPPING,
                    imported_value=out.get("jwl_A", orig_a),
                    transformation=f"built-in {mat} A emitted with .4g",
                )

    # Orphan searchable surfaces (unsupported/lost)
    for surf in ctx.get("orphan_surfaces") or []:
        lost_leaves.append(
            {
                "leaf": f"geometry/{surf['name']} + refinementSurfaces/{surf['name']}/level",
                "source_file": "system/snappyHexMeshDict",
                "source_value": {
                    "type": surf["type"],
                    "level": surf["level"],
                },
                "classification": UNSUPPORTED_LOST,
                "notes": (
                    "Non-STL searchable surface used for snappy refinementSurfaces is not "
                    "mapped to a GUI obstacle/charge field and is not regenerated."
                ),
            }
        )
        fields.append(
            FieldClassification(
                gui_key=f"[lost] snappy:{surf['name']}",
                classification=UNSUPPORTED_LOST,
                source_file="system/snappyHexMeshDict",
                source_path=f"geometry/{surf['name']}; refinementSurfaces/{surf['name']}/level",
                source_value=surf["level"],
                regenerated_value=None,
                rule="Searchable surface refinement not represented in GUI; not regenerated",
                notes=f"level={surf['level']}",
                difference_kind="numeric",
                value_origin="report_interpretation",
            )
        )


def _material_props_classification(
    out: Dict[str, Any],
    ctx: Dict[str, Any],
) -> FieldClassification:
    """Classify custom_material_props by comparing source JWL to built-in DB."""
    mat = ctx.get("material_name") or ""
    src_file, src_path = _source_for("custom_material_props")
    if mat not in _BUILTIN_JWL:
        if out.get("custom_material_props"):
            return FieldClassification(
                gui_key="custom_material_props",
                classification=LOADED_EXACT,
                source_file=src_file,
                source_path=src_path,
                imported_value=out.get("custom_material_props"),
                rule="Custom material props present in loader payload",
                value_origin="loader_payload",
            )
        return FieldClassification(
            gui_key="custom_material_props",
            classification=NOT_APPLICABLE,
            rule="No custom material props for this configuration",
            applicable=False,
        )

    builtin = _BUILTIN_JWL[mat]
    mismatches: List[str] = []
    details: Dict[str, Any] = {}
    for key in ("A", "B", "R1", "R2", "omega"):
        src = out.get(f"jwl_{key}")
        if src is None:
            continue
        dbv = builtin[key]
        details[key] = {"original": src, "builtin": dbv}
        if _numeric_differs(src, dbv):
            mismatches.append(key)

    if not mismatches:
        # Floats match built-in; still may have .4g regen text change (reported as parity leaf).
        return FieldClassification(
            gui_key="custom_material_props",
            classification=NOT_APPLICABLE,
            source_file=src_file,
            source_path=src_path,
            rule=(
                f"Built-in {mat}: source JWL floats match material library; "
                "custom_material_props not required. See parity leaf for any "
                ".4g emission rounding of A."
            ),
            applicable=False,
            notes=str(details.get("A")),
        )

    # Source differs from what generator will use from built-in DB.
    return FieldClassification(
        gui_key="custom_material_props",
        classification=ALTERNATIVE_MAPPING,
        source_file=src_file,
        source_path=src_path,
        source_value={k: details[k]["original"] for k in mismatches},
        imported_value={k: details[k]["builtin"] for k in mismatches},
        regenerated_value={k: _jwl_emit_4g(details[k]["builtin"]) for k in mismatches},
        rule=(
            f"Source JWL properties differ from built-in {mat} library values that "
            "the generator will emit; built-in mapping replaces source leaves."
        ),
        regenerated_destination="constant/phaseProperties/products/equationOfState",
        notes=f"Mismatched keys: {', '.join(mismatches)}",
        difference_kind="numeric",
        value_origin="report_interpretation",
    )


def classify_load_fields(
    out: Dict[str, Any],
    ui_field_keys: Sequence[str],
    *,
    preserved_keys: Optional[Sequence[str]] = None,
    ambiguous_keys: Optional[Sequence[str]] = None,
    regen_proof: Optional[Dict[str, Dict[str, Any]]] = None,
) -> LoadSummaryReport:
    """Build context-aware classifications for UI fields and parity leaves.

    ``preserved_keys`` / ``regen_proof`` enable PRESERVED_UNCHANGED when regeneration
    has proven a leaf unchanged. ``ambiguous_keys`` forces CONFLICT_AMBIGUOUS.
    Does not mutate model values in ``out``.
    """
    case_dir = str(out.get("_case_dir") or "")
    ctx = _case_context(out, case_dir)
    preserved = set(preserved_keys or [])
    ambiguous = set(ambiguous_keys or [])
    regen_proof = dict(regen_proof or {})

    # Auto-ambiguous: useCOM yes together with explicit initiation points.
    if ctx.get("use_com") is True and ctx.get("has_initiation_points"):
        ambiguous.add("ignition_mode")
        ambiguous.add("initiation_point")

    filled: List[str] = []
    not_filled: List[Tuple[str, str]] = []
    fields: List[FieldClassification] = []
    provenance: List[Dict[str, Any]] = []
    lost_leaves: List[Dict[str, Any]] = []

    shape = ctx["charge_shape"]
    is_remap = ctx["is_remap"]

    for key in ui_field_keys:
        present = key in out and out[key] is not None
        if key == "boundaries" and not out.get("boundaries"):
            present = False
        if key == "stl_obstacles" and not out.get("stl_obstacles"):
            present = False

        if present:
            filled.append(key)
        else:
            not_filled.append((key, _legacy_not_filled_reason(key, shape, is_remap)))

        applicable, na_reason = _is_applicable(key, ctx)
        src_file, src_path = _source_for(key)

        if key in ambiguous:
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=CONFLICT_AMBIGUOUS,
                    source_file=src_file,
                    source_path=src_path,
                    source_value=out.get(key),
                    imported_value=out.get(key),
                    rule=(
                        "More than one interpretation is possible or parity cannot be "
                        "established confidently for this field."
                    ),
                    notes="Report-only conflict flag; loader payload unchanged.",
                    value_origin="report_interpretation",
                )
            )
            continue

        if key in preserved or (
            key in regen_proof and regen_proof[key].get("unchanged") is True
        ):
            proof = regen_proof.get(key) or {}
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=PRESERVED_UNCHANGED,
                    source_file=src_file or str(proof.get("source_file") or ""),
                    source_path=src_path or str(proof.get("source_path") or ""),
                    source_value=proof.get("original", out.get(key)),
                    imported_value=out.get(key),
                    regenerated_value=proof.get("regenerated", proof.get("original")),
                    rule="Regeneration proved this leaf is re-emitted unchanged.",
                    regenerated_destination=str(proof.get("destination") or src_file),
                    value_origin="regen_proof",
                )
            )
            continue

        if not applicable:
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=NOT_APPLICABLE,
                    source_file=src_file,
                    source_path=src_path,
                    rule=na_reason,
                    notes=na_reason,
                    applicable=False,
                )
            )
            continue

        # --- custom material ---
        if key == "custom_material_props":
            fields.append(_material_props_classification(out, ctx))
            continue

        # --- charge_length derived (not in loader payload) ---
        if key == "charge_length" and shape == "Cylinder" and not present:
            mass, rho, lbyd = out.get("mass_kg"), out.get("rho_charge"), out.get("charge_lbyd")
            if mass and rho and lbyd and float(rho) > 0 and float(lbyd) > 0:
                r, length = _cylinder_dims_from_mass(float(mass), float(rho), float(lbyd))
                fields.append(
                    FieldClassification(
                        gui_key=key,
                        classification=DERIVED,
                        source_file="system/setFieldsDict",
                        source_path="regions/*/mass,rho,LbyD",
                        source_value={"mass": mass, "density": rho, "L/D": lbyd},
                        imported_value=length,
                        rule=(
                            "length = 2*r*L/D with r=(V/(2*pi*L/D))^(1/3), V=mass/density"
                        ),
                        regenerated_destination="GUI charge_length (widget derivation)",
                        notes=(
                            "Not set in loader payload. Report derives length; GUI widgets "
                            f"also compute length≈{length:.6g} m from mass/rho/L/D "
                            f"(r≈{r:.6g} m). value_origin=report_interpretation."
                        ),
                        value_origin="report_interpretation",
                    )
                )
                _append_prov(
                    provenance,
                    gui_key=key,
                    source_file="system/setFieldsDict",
                    source_path="regions/*/mass,rho,LbyD",
                    source_value={"mass": mass, "rho": rho, "LbyD": lbyd},
                    mapping_type=DERIVED,
                    imported_value=length,
                    transformation="cylinder length from mass/density/LbyD",
                )
                continue

        # --- charge_radius: DERIVED when reconstructed from mass/rho/(LbyD) ---
        if key == "charge_radius" and present:
            mass, rho = out.get("mass_kg"), out.get("rho_charge")
            derived_r = None
            rule = ""
            if mass and rho and float(rho) > 0:
                if shape == "Cylinder":
                    lbyd = out.get("charge_lbyd") or 2.5
                    derived_r, _ = _cylinder_dims_from_mass(float(mass), float(rho), float(lbyd))
                    rule = "r = (V/(2*pi*L/D))^(1/3) from mass, density, L/D"
                elif shape == "Sphere":
                    derived_r = _sphere_radius_from_mass(float(mass), float(rho))
                    rule = "r = (3V/(4*pi))^(1/3) from mass, density"
            if derived_r is not None and _nearly_equal(out[key], derived_r, rel=1e-6, abs_tol=1e-9):
                fields.append(
                    FieldClassification(
                        gui_key=key,
                        classification=DERIVED,
                        source_file="system/setFieldsDict",
                        source_path="regions/*/mass,rho,LbyD",
                        source_value={"mass": mass, "density": rho, "L/D": out.get("charge_lbyd")},
                        imported_value=out[key],
                        rule=rule,
                        regenerated_destination="system/setFieldsDict (via mass/rho geometry)",
                        notes="Loader payload contains derived charge_radius (not a raw region radius leaf).",
                        value_origin="loader_payload",
                    )
                )
                _append_prov(
                    provenance,
                    gui_key=key,
                    source_file="system/setFieldsDict",
                    source_path="mass,rho,LbyD",
                    source_value={"mass": mass, "rho": rho},
                    mapping_type=DERIVED,
                    imported_value=out[key],
                    transformation=rule,
                )
                continue

        # --- charge_backup_radius_factor DERIVED ---
        if key == "charge_backup_radius_factor" and present:
            backup = out.get("charge_backup_radius_override") or out.get("charge_capture_radius")
            main_r = out.get("charge_radius")
            if backup and main_r and float(main_r) > 0:
                expected = float(backup) / float(main_r)
                if _nearly_equal(out[key], expected, rel=1e-6):
                    fields.append(
                        FieldClassification(
                            gui_key=key,
                            classification=DERIVED,
                            source_file="system/setFieldsDict",
                            source_path="regions/*/backup/radius / charge_radius",
                            source_value={"backup_radius": backup, "charge_radius": main_r},
                            imported_value=out[key],
                            rule="factor = backup_radius / charge_radius",
                            regenerated_destination="system/setFieldsDict backup radius path",
                            notes="Derived in loader from backup radius and computed charge radius.",
                            value_origin="loader_payload",
                        )
                    )
                    _append_prov(
                        provenance,
                        gui_key=key,
                        source_file="system/setFieldsDict",
                        source_path="backup/radius",
                        source_value={"backup_radius": backup, "charge_radius": main_r},
                        mapping_type=DERIVED,
                        imported_value=out[key],
                        transformation="backup_radius / charge_radius",
                    )
                    continue

        # --- write_control_type: EXACT vs CONVERTED ---
        if key == "write_control_type" and present:
            raw = ctx.get("write_control_raw")
            imported = out[key]
            if raw == "runTime" and imported == "adjustableRunTime":
                cls = LOADED_CONVERTED
                rule = "OpenFOAM writeControl runTime normalized to GUI adjustableRunTime"
            elif raw == imported:
                cls = LOADED_EXACT
                rule = f"Loaded writeControl={imported} with no normalization"
            elif raw is None:
                cls = LOADED_CONVERTED
                rule = "writeControl inferred/normalized into GUI token"
            else:
                cls = LOADED_CONVERTED
                rule = f"writeControl {raw} -> {imported}"
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=cls,
                    source_file="system/controlDict",
                    source_path="writeControl",
                    source_value=raw,
                    imported_value=imported,
                    rule=rule,
                    regenerated_destination="system/controlDict/writeControl",
                    value_origin="loader_payload",
                )
            )
            _append_prov(
                provenance,
                gui_key=key,
                source_file="system/controlDict",
                source_path="writeControl",
                source_value=raw,
                mapping_type=cls,
                imported_value=imported,
                transformation=rule if cls == LOADED_CONVERTED else None,
            )
            continue

        # --- write_interval_time alternative ---
        if key == "write_interval_time" and present:
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=ALTERNATIVE_MAPPING
                    if ctx["write_control_type"] in ("adjustableRunTime", "runTime")
                    else LOADED_EXACT,
                    source_file="system/controlDict",
                    source_path="writeInterval",
                    source_value=out.get("write_interval_raw", out.get(key)),
                    imported_value=out.get(key),
                    rule=(
                        "writeControl is time-based -> loader sets write_interval_time "
                        "from writeInterval; write_interval_steps is not applicable"
                    ),
                    regenerated_destination="system/controlDict/writeInterval",
                    notes="value_origin=loader_payload (key is present in load_case output).",
                    value_origin="loader_payload",
                )
            )
            _append_prov(
                provenance,
                gui_key=key,
                source_file="system/controlDict",
                source_path="writeInterval",
                source_value=out.get("write_interval_raw", out.get(key)),
                mapping_type=ALTERNATIVE_MAPPING,
                imported_value=out.get(key),
                transformation="time-based writeInterval -> write_interval_time",
            )
            continue

        if key == "write_interval_steps" and present:
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=ALTERNATIVE_MAPPING
                    if ctx["write_control_type"] == "timeStep"
                    else LOADED_EXACT,
                    source_file="system/controlDict",
                    source_path="writeInterval",
                    source_value=out.get("write_interval_raw", out.get(key)),
                    imported_value=out.get(key),
                    rule="writeControl=timeStep -> loader sets write_interval_steps from writeInterval",
                    regenerated_destination="system/controlDict/writeInterval",
                    value_origin="loader_payload",
                )
            )
            continue

        # --- ignition_mode / initiation_point (report interpretation + GUI default) ---
        if key == "ignition_mode":
            if ctx["use_com"] is True:
                fields.append(
                    FieldClassification(
                        gui_key=key,
                        classification=ALTERNATIVE_MAPPING,
                        source_file="constant/phaseProperties",
                        source_path="initiation/useCOM",
                        source_value="yes",
                        imported_value="Center of Charge",
                        regenerated_value="useCOM yes",
                        rule="useCOM yes maps semantically to ignition_mode=Center of Charge",
                        regenerated_destination="constant/phaseProperties/initiation/useCOM",
                        notes=(
                            "Loader does NOT set ignition_mode in the payload (UNSET). "
                            "This is a report-level interpretation of useCOM. The GUI "
                            "widget default 'Center of Charge' happens to regenerate "
                            "useCOM yes."
                        ),
                        value_origin="report_interpretation",
                    )
                )
                continue
            if ctx["use_com"] is False:
                fields.append(
                    FieldClassification(
                        gui_key=key,
                        classification=ALTERNATIVE_MAPPING,
                        source_file="constant/phaseProperties",
                        source_path="initiation/useCOM",
                        source_value="no",
                        imported_value="Manual",
                        rule="useCOM no maps to ignition_mode=Manual",
                        value_origin="report_interpretation",
                    )
                )
                continue
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=CONFLICT_AMBIGUOUS,
                    source_file="constant/phaseProperties",
                    source_path="initiation/useCOM",
                    rule="useCOM absent; cannot verify effective ignition mode without solver default proof",
                    notes="Not classified as SOLVER_DEFAULT: default not verified from blastFoam source.",
                    value_origin="report_interpretation",
                )
            )
            continue

        if key == "initiation_point":
            if ctx["use_com"] is True:
                fields.append(
                    FieldClassification(
                        gui_key=key,
                        classification=ALTERNATIVE_MAPPING,
                        source_file="constant/phaseProperties",
                        source_path="initiation/useCOM",
                        source_value="yes (COM)",
                        imported_value=out.get("charge_center"),
                        rule="useCOM yes -> initiation at charge COM; initiation_point unused",
                        regenerated_destination="constant/phaseProperties/initiation/useCOM",
                        notes=(
                            "Loader does NOT set initiation_point. Report maps COM to "
                            "charge_center. GUI leaves initiation_point UNSET under "
                            "Center of Charge."
                        ),
                        value_origin="report_interpretation",
                    )
                )
                continue
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=CONFLICT_AMBIGUOUS,
                    source_file="constant/phaseProperties",
                    source_path="initiation",
                    rule="No verified initiationPoints/useCOM mapping for this case",
                    value_origin="report_interpretation",
                )
            )
            continue

        # --- legacy GUI mins: NOT solver defaults ---
        if key in ("dyn_refine_min", "refine_min") and ctx["dyn_mesh"] and not present:
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=NOT_APPLICABLE,
                    source_file="constant/dynamicMeshDict",
                    source_path="(no corresponding adaptiveFvMesh integer-min key)",
                    rule=(
                        "Legacy GUI field with no corresponding blastFoam/OpenFOAM "
                        "adaptiveFvMesh key. Not a verified solver default."
                    ),
                    notes=(
                        "Provenance UNSET; generator does not emit an integer min level. "
                        "Classified NOT_APPLICABLE (legacy-only), not SOLVER_DEFAULT."
                    ),
                    applicable=False,
                    value_origin="report_interpretation",
                )
            )
            continue

        # --- upperRefineLevel / upperUnrefineLevel: verified Foam::great ---
        if key in ("upper_refine_level", "upper_unrefine_level") and ctx["dyn_mesh"] and not present:
            of_key = (
                "upperRefineLevel" if key == "upper_refine_level" else "upperUnrefineLevel"
            )
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=SOLVER_DEFAULT,
                    source_file="constant/dynamicMeshDict",
                    source_path=of_key,
                    source_value=None,
                    imported_value=None,
                    regenerated_value=f"(omitted; solver default {_BLASTFOAM_620_UPPER_REFINE_DEFAULT})",
                    rule=(
                        f"blastFoam 6.2.0 ({_BLASTFOAM_620_COMMIT}) "
                        f"errorEstimator::read uses lookupOrDefault(\"{of_key}\", great). "
                        "GGUI leaves UNSET and omits the key on regeneration."
                    ),
                    regenerated_destination=f"constant/dynamicMeshDict/{of_key} (omitted)",
                    notes="Verified implementation default: Foam::great when key absent.",
                    value_origin="report_interpretation",
                )
            )
            continue

        # Optional AMR keys: omission-parity PRESERVED_UNCHANGED when source absent,
        # loader UNSET, and generator omits the same key. Do not claim solver defaults.
        # balance_interval when balancing inactive is NOT_APPLICABLE (handled above).
        if key in (
            "dynamic_max_cells",
            "begin_unrefine",
            "enable_balancing",
            "balance_interval",
        ) and ctx["dyn_mesh"] and not present:
            src_file, src_path = _source_for(key)
            of_key_map = {
                "dynamic_max_cells": "maxCells",
                "begin_unrefine": "beginUnrefine",
                "enable_balancing": "enableBalancing",
                "balance_interval": "balanceInterval",
            }
            of_key = of_key_map[key]
            source_absent = not (ctx.get("dm_keys_present") or {}).get(of_key, False)

            if key == "balance_interval":
                # Balancing is on (applicability already filtered inactive case).
                fields.append(
                    FieldClassification(
                        gui_key=key,
                        classification=CONFLICT_AMBIGUOUS,
                        source_file=src_file,
                        source_path=src_path,
                        source_value=None,
                        rule=(
                            "Balancing is enabled but balanceInterval is absent and no "
                            "verified blastFoam default was established for this report."
                        ),
                        regenerated_destination=f"{src_file}/{src_path}",
                        notes="Not classified as NOT_APPLICABLE because balancing is active.",
                        value_origin="report_interpretation",
                    )
                )
                continue

            # Generator omission policy (generator_3d._write_constant_files_3d):
            # - maxCells written only when value != 200000000
            # - enableBalancing / loadBalance written only when enable_balancing is True
            # - beginUnrefine written only when begin_unrefine is not None
            # With UNSET fields, these keys are omitted from regenerated dynamicMeshDict.
            omission_parity = (
                bool(ctx.get("has_case_dir"))
                and source_absent
                and (not present)
            )
            if omission_parity:
                fields.append(
                    FieldClassification(
                        gui_key=key,
                        classification=PRESERVED_UNCHANGED,
                        source_file=src_file,
                        source_path=src_path,
                        source_value=None,
                        imported_value=None,
                        regenerated_value="(omitted)",
                        rule=(
                            f"Omission parity: original {src_path} absent -> loader leaves "
                            f"{key} UNSET -> generator omits {src_path} -> absent state preserved. "
                            "Not a verified solver default."
                        ),
                        regenerated_destination=f"{src_file}/{src_path} (omitted)",
                        notes=(
                            "Report-only classification. Legacy filled/not_filled/_provenance "
                            "unchanged (field remains UNSET for set_case_inputs)."
                        ),
                        value_origin="regen_proof",
                        difference_kind="",
                    )
                )
            else:
                fields.append(
                    FieldClassification(
                        gui_key=key,
                        classification=CONFLICT_AMBIGUOUS,
                        source_file=src_file,
                        source_path=src_path,
                        rule=(
                            "Cannot establish omission-parity proof (missing case dir or "
                            "source-key absence confirmation). Absence alone is not enough "
                            "for PRESERVED_UNCHANGED."
                        ),
                        value_origin="report_interpretation",
                    )
                )
            continue

        if key == "charge_capture_factor" and not present:
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=NOT_APPLICABLE,
                    rule="Auto capture factor not used for this case",
                    applicable=False,
                )
            )
            continue

        # --- generic present keys ---
        if present:
            val = out.get(key)
            if key in _STRUCTURE_INFERRED:
                cls = LOADED_CONVERTED
                rule = "Inferred from dictionary structure/presence (not a direct leaf copy)"
            elif key in (
                "cell_size",
                "boundaries",
                "material_name",
                "refine_indicator_field",
            ):
                cls = LOADED_CONVERTED
                rule = "Loaded with structural or semantic conversion/normalization"
            else:
                cls = LOADED_EXACT
                rule = "Loaded directly from a case leaf value"
            fields.append(
                FieldClassification(
                    gui_key=key,
                    classification=cls,
                    source_file=src_file,
                    source_path=src_path,
                    source_value=val if cls == LOADED_EXACT else None,
                    imported_value=val,
                    rule=rule,
                    regenerated_destination=src_file or "",
                    value_origin="loader_payload",
                )
            )
            _append_prov(
                provenance,
                gui_key=key,
                source_file=src_file,
                source_path=src_path,
                source_value=val,
                mapping_type=cls,
                imported_value=val,
                transformation=rule if cls == LOADED_CONVERTED else None,
            )
            continue

        # Remaining absent applicable keys — do not claim unverified solver defaults.
        fields.append(
            FieldClassification(
                gui_key=key,
                classification=CONFLICT_AMBIGUOUS,
                source_file=src_file,
                source_path=src_path,
                rule=(
                    "Absent in case; effective solver/GUI default not verified for "
                    "this report. Loader leaves UNSET."
                ),
                value_origin="report_interpretation",
            )
        )

    _add_parity_leafs(fields, lost_leaves, provenance, out, ctx)

    # PRESERVED_UNCHANGED for non-GUI regen_proof entries
    for leaf_key, proof in regen_proof.items():
        if leaf_key in {f.gui_key for f in fields}:
            continue
        if proof.get("unchanged") is True:
            fields.append(
                FieldClassification(
                    gui_key=leaf_key,
                    classification=PRESERVED_UNCHANGED,
                    source_file=str(proof.get("source_file") or ""),
                    source_path=str(proof.get("source_path") or ""),
                    source_value=proof.get("original"),
                    imported_value=proof.get("original"),
                    regenerated_value=proof.get("regenerated", proof.get("original")),
                    rule="Regeneration proved this leaf is re-emitted unchanged.",
                    regenerated_destination=str(proof.get("destination") or ""),
                    value_origin="regen_proof",
                )
            )

    return LoadSummaryReport(
        case_dir=case_dir,
        fields=fields,
        filled=filled,
        not_filled=not_filled,
        notes=list(out.get("_load_notes") or []),
        lost_leaves=lost_leaves,
        provenance=provenance,
    )


def _jsonable(val: Any) -> Any:
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, set):
        return sorted(_jsonable(x) for x in val)
    if isinstance(val, dict):
        return {k: _jsonable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_jsonable(v) for v in val]
    return val


def format_load_summary_text(
    summary: Dict[str, Any],
    case_dir: str,
    *,
    include_technical: bool = False,
    include_not_applicable: bool = False,
) -> str:
    """Format popup / clipboard text from a v2+ load summary dict."""
    lines: List[str] = [
        f"Loaded: {case_dir}",
        "",
    ]

    classifications = summary.get("classifications") or []
    counts = summary.get("counts") or {}
    if classifications:
        n_resolved = sum(
            1
            for c in classifications
            if c.get("classification")
            in (
                LOADED_EXACT,
                LOADED_CONVERTED,
                DERIVED,
                ALTERNATIVE_MAPPING,
                PRESERVED_UNCHANGED,
            )
        )
        n_conflict = int(counts.get(CONFLICT_AMBIGUOUS, 0))
        n_lost = int(counts.get(UNSUPPORTED_LOST, 0))
        n_attention = n_conflict + n_lost

        lines.append(f"Resolved field entries (exact/converted/derived/alternative/preserved): {n_resolved}")
        lines.append("")
        lines.append("Summary counts:")
        lines.append(f"  Loaded exactly: {int(counts.get(LOADED_EXACT, 0))}")
        lines.append(f"  Converted: {int(counts.get(LOADED_CONVERTED, 0))}")
        lines.append(f"  Derived: {int(counts.get(DERIVED, 0))}")
        lines.append(f"  Alternative mappings: {int(counts.get(ALTERNATIVE_MAPPING, 0))}")
        lines.append(f"  Using solver defaults: {int(counts.get(SOLVER_DEFAULT, 0))}")
        lines.append(f"  Preserved unchanged: {int(counts.get(PRESERVED_UNCHANGED, 0))}")
        lines.append(f"  Conflicts / ambiguous: {n_conflict}")
        lines.append(f"  Unsupported / lost: {n_lost}")
        lines.append(f"  Total needing attention: {n_attention}")
        lines.append("")

        needs = [
            c
            for c in classifications
            if c.get("classification") in (CONFLICT_AMBIGUOUS, UNSUPPORTED_LOST)
        ]
        lines.append("Needs attention (review required):")
        if not needs:
            lines.append("  (none)")
        else:
            for c in sorted(needs, key=lambda x: x.get("gui_key", "")):
                sev = (
                    "AMBIGUOUS"
                    if c.get("classification") == CONFLICT_AMBIGUOUS
                    else "UNSUPPORTED / LOST"
                )
                lines.append(f"  [{sev}] {c.get('gui_key')}")
                _append_entry_lines(lines, c)
        lines.append("")

        # Default view: derived + alternative (including parity replacements)
        derived_alt = [
            c
            for c in classifications
            if c.get("classification") in (DERIVED, ALTERNATIVE_MAPPING)
        ]
        lines.append("Derived / alternative mappings:")
        if not derived_alt:
            lines.append("  (none)")
        else:
            for c in sorted(derived_alt, key=lambda x: x.get("gui_key", "")):
                lines.append(f"  {c.get('gui_key')}: {c.get('classification')}")
                _append_entry_lines(lines, c)
        lines.append("")

        if include_technical:
            lines.append("Technical details:")
            for cls in (
                LOADED_EXACT,
                LOADED_CONVERTED,
                SOLVER_DEFAULT,
                PRESERVED_UNCHANGED,
            ):
                group = [c for c in classifications if c.get("classification") == cls]
                lines.append(f"  [{_COUNT_LABELS[cls]}] ({len(group)})")
                for c in sorted(group, key=lambda x: x.get("gui_key", "")):
                    src = ""
                    if c.get("source_file"):
                        src = f" <- {c.get('source_file')}/{c.get('source_path')}"
                    val = c.get("imported_value")
                    val_s = f" = {val}" if val is not None else ""
                    lines.append(f"    {c.get('gui_key')}{val_s}{src}")
                    if c.get("rule"):
                        lines.append(f"      {c.get('rule')}")
                    if c.get("value_origin"):
                        lines.append(f"      origin: {c.get('value_origin')}")
            if include_not_applicable:
                na = [c for c in classifications if c.get("classification") == NOT_APPLICABLE]
                lines.append(f"  [Not applicable] ({len(na)})")
                for c in sorted(na, key=lambda x: x.get("gui_key", "")):
                    lines.append(f"    {c.get('gui_key')}: {c.get('rule')}")
            lines.append("")
    else:
        filled = summary.get("filled", [])
        not_filled = summary.get("not_filled", [])
        lines.append(f"Fields filled from case (LOADED): {len(filled)}")
        lines.append(f"Fields not filled (UNSET): {len(not_filled)}")
        lines.append("")

    notes = summary.get("notes") or []
    if notes:
        lines.append("Load notes:")
        for note in notes:
            lines.append(f"  - {note}")
        lines.append("")

    lines.append("Pre-run parity: To verify generated files match this case (no edits), run:")
    lines.append(f"  python tools/roundtrip_building3d_case.py \"{case_dir}\"")
    return "\n".join(lines)


def _append_entry_lines(lines: List[str], c: Dict[str, Any]) -> None:
    if c.get("source_file"):
        lines.append(f"    Source: {c.get('source_file')}/{c.get('source_path')}")
    if c.get("source_value") is not None:
        lines.append(f"    Original: {c.get('source_value')}")
    if c.get("imported_value") is not None:
        lines.append(f"    Imported/selected/derived: {c.get('imported_value')}")
    if c.get("regenerated_value") is not None:
        lines.append(f"    Regenerated value: {c.get('regenerated_value')}")
    if c.get("regenerated_destination"):
        lines.append(f"    Regenerated path: {c.get('regenerated_destination')}")
    if c.get("rule"):
        lines.append(f"    Rule: {c.get('rule')}")
    if c.get("difference_kind"):
        lines.append(f"    Difference kind: {c.get('difference_kind')}")
    if c.get("value_origin"):
        lines.append(f"    Origin: {c.get('value_origin')}")
    if c.get("notes"):
        lines.append(f"    Note: {c.get('notes')}")


def strip_report_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of load_case data with report-only keys removed."""
    import copy

    skip = {"_load_summary", "_field_classifications", "_report_provenance"}
    return {k: copy.deepcopy(v) for k, v in data.items() if k not in skip}
