"""Versioned, human-readable GGUI project persistence."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import MISSING, asdict, fields
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Protocol

from charge_seed_plan import charge_dims_from_inputs, migrate_case_inputs_seed_fields
from models import CaseInputs1D, CaseInputs3D, ObstacleData
from models_2d import CaseInputs2D, MappingSource2D, ProbePoint2D
from output_options import OutputFileOptions

SCHEMA_VERSION = 2
PROJECT_SUFFIX = ".ggui.json"

# Display-only 2D fields: persisted for UI restore, never consumed by generation.
_DISPLAY_ONLY_2D_FIELDS = (
    "mirrored_view",
    "show_mesh",
    "show_probes",
    "log_scale",
)


class _SupportsProjectCapture(Protocol):
    def get_case_inputs(self) -> CaseInputs3D: ...
    def set_case_inputs(self, data: dict, load_summary: dict = None) -> None: ...
    def load_project_gui_state(self, state: dict) -> None: ...
    def _refresh_table(self) -> None: ...
    sections: Any
    obstacles: Any


class _SupportsProbesDict(Protocol):
    def to_dict(self) -> Dict[str, Any]: ...
    def load_dict(self, data: Dict[str, Any]) -> None: ...


class _SupportsProjectCapture2D(Protocol):
    def get_case_inputs(self) -> CaseInputs2D: ...
    def set_case_inputs(self, data: dict) -> None: ...


class _SupportsProjectCapture1D(Protocol):
    def get_case_inputs(self) -> CaseInputs1D: ...
    def set_case_inputs(self, data: dict) -> None: ...


class ProjectFormatError(ValueError):
    pass


def _normalize_v2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a schema-v2 payload has the nested dimensions layout."""
    out = dict(payload)
    out["schema_version"] = SCHEMA_VERSION
    dimensions = out.get("dimensions")
    if not isinstance(dimensions, dict):
        dimensions = {}
    else:
        dimensions = dict(dimensions)

    case_inputs = out.get("case_inputs")
    if "3D" not in dimensions and isinstance(case_inputs, dict):
        dimensions["3D"] = {
            "case_inputs": case_inputs,
            "input_fields": sorted(case_inputs.keys()),
            "display_only_fields": [],
        }
    elif "3D" in dimensions and isinstance(dimensions["3D"], dict):
        section = dict(dimensions["3D"])
        nested = section.get("case_inputs")
        if isinstance(nested, dict):
            out["case_inputs"] = nested
            section.setdefault("input_fields", sorted(nested.keys()))
            section.setdefault("display_only_fields", [])
        dimensions["3D"] = section

    section_2d = dimensions.get("2D")
    if isinstance(section_2d, dict):
        section = dict(section_2d)
        nested_2d = section.get("case_inputs")
        if isinstance(nested_2d, dict):
            undefined = nested_2d.get("undefined_keys") or section.get("undefined_keys") or []
            section["undefined_keys"] = list(undefined)
            imported_meta = section.get("imported_case_metadata")
            if not isinstance(imported_meta, dict):
                mapping = nested_2d.get("mapping") or {}
                imported_meta = {
                    "mapping_source": mapping if isinstance(mapping, dict) else {},
                    "undefined_keys": list(undefined),
                }
            section["imported_case_metadata"] = imported_meta
            section.setdefault(
                "input_fields",
                sorted(
                    k for k in nested_2d.keys() if k not in _DISPLAY_ONLY_2D_FIELDS
                ),
            )
            section.setdefault("display_only_fields", list(_DISPLAY_ONLY_2D_FIELDS))
        dimensions["2D"] = section

    section_1d = dimensions.get("1D")
    if isinstance(section_1d, dict):
        section = dict(section_1d)
        nested_1d = section.get("case_inputs")
        if isinstance(nested_1d, dict):
            section.setdefault("input_fields", sorted(nested_1d.keys()))
            section.setdefault("display_only_fields", [])
        dimensions["1D"] = section

    out["dimensions"] = dimensions
    available = out.get("dimensions_available")
    if not isinstance(available, list) or not available:
        available = ["3D"]
        if "1D" in dimensions:
            available.insert(0, "1D")
        if "2D" in dimensions:
            available.append("2D")
        out["dimensions_available"] = available
    gui_state = out.get("gui_state") if isinstance(out.get("gui_state"), dict) else {}
    out.setdefault(
        "active_tab",
        gui_state.get("selected_primary_tab") or out.get("project_dimension") or "General 3D",
    )
    out.setdefault("project_dimension", "3D")
    provenance = out.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance.setdefault("format", "explicit-json")
    provenance["contains_runtime_results"] = False
    out["provenance"] = provenance
    return out


def _migrate_v1_to_v2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic schema 1 → 2 migration. Preserves undefined 2D keys as undefined."""
    out = dict(payload)
    # Legacy v1 kept 3D inputs at the top level and optional dimensions.2D.
    return _normalize_v2(out)


def _migrate_v2(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_v2(payload)


_MIGRATIONS = {1: _migrate_v1_to_v2, 2: _migrate_v2}


def build_project(
    inputs: CaseInputs3D,
    *,
    probes: Dict[str, Any],
    gui_state: Dict[str, Any],
    inputs_1d: CaseInputs1D | None = None,
    inputs_2d: CaseInputs2D | None = None,
    simulation_state_2d: str | None = None,
) -> Dict[str, Any]:
    case_inputs_3d = asdict(inputs)
    dimensions: Dict[str, Any] = {
        "3D": {
            "case_inputs": case_inputs_3d,
            "input_fields": sorted(case_inputs_3d.keys()),
            "display_only_fields": [],
        }
    }
    dimensions_available = ["3D"]
    if inputs_1d is not None:
        case_inputs_1d = asdict(inputs_1d)
        dimensions["1D"] = {
            "model": "spherical-1d",
            "case_inputs": case_inputs_1d,
            "input_fields": sorted(case_inputs_1d.keys()),
            "display_only_fields": [],
        }
        dimensions_available.insert(0, "1D")
    if inputs_2d is not None:
        case_inputs_2d = asdict(inputs_2d)
        undefined = list(getattr(inputs_2d, "undefined_keys", ()) or ())
        mapping = case_inputs_2d.get("mapping") or {}
        dimensions["2D"] = {
            "model": "axisymmetric-rz-wedge",
            "case_inputs": case_inputs_2d,
            "undefined_keys": undefined,
            "imported_case_metadata": {
                "mapping_source": mapping if isinstance(mapping, dict) else {},
                "undefined_keys": undefined,
            },
            "input_fields": sorted(
                k for k in case_inputs_2d.keys() if k not in _DISPLAY_ONLY_2D_FIELDS
            ),
            "display_only_fields": list(_DISPLAY_ONLY_2D_FIELDS),
            "simulation_state": simulation_state_2d,
        }
        dimensions_available.append("2D")
    active_tab = (
        gui_state.get("selected_primary_tab")
        if isinstance(gui_state, dict)
        else None
    ) or "General 3D"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "application": {
            "name": "GGUI",
            "version": "4.0",
            "saved_utc": datetime.now(timezone.utc).isoformat(),
        },
        "dimensions_available": dimensions_available,
        "active_tab": active_tab,
        "project_dimension": "3D",
        "case_inputs": case_inputs_3d,
        "dimensions": dimensions,
        "probes": probes,
        "gui_state": gui_state,
        "provenance": {
            "format": "explicit-json",
            "contains_runtime_results": False,
        },
    }
    return payload


def capture_project_payload(
    tab: _SupportsProjectCapture,
    probes_model: _SupportsProbesDict,
    tab_2d: _SupportsProjectCapture2D | None = None,
    tab_1d: _SupportsProjectCapture1D | None = None,
    selected_primary_tab: str = "General 3D",
    output_file_options: OutputFileOptions | None = None,
) -> Dict[str, Any]:
    """Capture dialog-independent project JSON from the live 3D GUI state."""
    gui_state = {
        "selected_primary_tab": selected_primary_tab,
        "sections": [asdict(section) for section in tab.sections],
        "obstacles": [asdict(obstacle) for obstacle in tab.obstacles],
    }
    if output_file_options is not None:
        gui_state["output_file_options"] = asdict(output_file_options)
    return build_project(
        tab.get_case_inputs(),
        probes=probes_model.to_dict(),
        inputs_1d=tab_1d.get_case_inputs() if tab_1d is not None else None,
        inputs_2d=tab_2d.get_case_inputs() if tab_2d is not None else None,
        gui_state=gui_state,
    )


def apply_project_payload(
    tab: _SupportsProjectCapture,
    probes_model: _SupportsProbesDict,
    project: Dict[str, Any],
    tab_2d: _SupportsProjectCapture2D | None = None,
    tab_1d: _SupportsProjectCapture1D | None = None,
) -> None:
    """Apply a read_project() result to the 3D tab without file dialogs.

    A GGUI project is authoritative: set_case_inputs is called without load_summary
    so stale OpenFOAM case-loader provenance is cleared.
    """
    from viewer_widget import ObstacleItem

    inputs = project["inputs"]
    data = asdict(inputs)
    data["charge_radius"] = inputs.cylinder_radius
    tab.set_case_inputs(data)
    saved_obstacles = project["gui_state"].get("obstacles")
    if isinstance(saved_obstacles, list):
        tab.obstacles = [
            ObstacleItem(
                bool(item.get("enabled", True)),
                str(item["path"]),
                float(item.get("scale", 1.0)),
                float(item.get("ox", 0.0)),
                float(item.get("oy", 0.0)),
                float(item.get("oz", 0.0)),
            )
            for item in saved_obstacles
            if isinstance(item, dict) and item.get("path")
        ]
    else:
        tab.obstacles = [
            ObstacleItem(
                True,
                obstacle.stl_path,
                obstacle.scale,
                obstacle.offset_x,
                obstacle.offset_y,
                obstacle.offset_z,
            )
            for obstacle in inputs.obstacles
        ]
    tab._refresh_table()
    probes_model.load_dict(project["probes"])
    tab.load_project_gui_state(project["gui_state"])
    if tab_2d is not None and project.get("inputs_2d") is not None:
        tab_2d.set_case_inputs(asdict(project["inputs_2d"]))
    if tab_1d is not None and project.get("inputs_1d") is not None:
        tab_1d.set_case_inputs(asdict(project["inputs_1d"]))


def write_project_atomic(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".ggui-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _characteristic_charge_extent_m_from_data(data: Dict[str, Any]) -> float:
    """Length scale matching Generator3D._characteristic_charge_extent_m for migration."""
    shape = str(data.get("charge_shape") or "Sphere")
    dims = charge_dims_from_inputs(
        SimpleNamespace(
            charge_shape=shape,
            mass_kg=float(data.get("mass_kg") or 1.0),
            rho_charge=float(data.get("rho_charge") or 1600.0),
            charge_length=float(data.get("charge_length") or 0.0),
            charge_width=float(data.get("charge_width") or 0.0),
            charge_height=float(data.get("charge_height") or 0.0),
            cylinder_radius=float(data.get("cylinder_radius") or 0.05),
            charge_aspect=float(data.get("charge_aspect") or 0.0),
        )
    )
    if shape == "Cuboid":
        if "length" in dims and "width" in dims and "height" in dims:
            return max(float(dims["length"]), float(dims["width"]), float(dims["height"])) / 2.0
        return float(dims.get("side", 0.1)) / 2.0
    if shape == "Cylinder":
        return max(float(dims.get("radius", 0.05)), float(dims.get("length", 0.1)) / 2.0)
    return float(dims.get("radius", 0.05))


def _legacy_auto_outside_extent_m(data: Dict[str, Any]) -> float:
    """OLD auto outside_extent: bubble_radius_factor shell + transition_cells×level_span×dx."""
    cs = max(1e-9, float(data.get("cell_size") or 0.1))
    r_char = _characteristic_charge_extent_m_from_data(data)
    factor = max(0.5, min(5.0, float(data.get("bubble_radius_factor") or 1.5)))
    seed_radius = r_char * factor
    n_cbl = max(1, min(10, int(data.get("transition_cells") or 2)))
    rmin = data.get("charge_outer_refine_min")
    rmax = data.get("charge_outer_refine_max")
    if rmin is None:
        rmin = data.get("refine_min", 2)
    if rmax is None:
        rmax = data.get("refine_max", 3)
    try:
        level_span = max(1, int(rmax) - int(rmin))
    except (TypeError, ValueError):
        level_span = 1
    legacy_outer_sphere = seed_radius + n_cbl * level_span * cs
    return max(0.0, legacy_outer_sphere - r_char)


def _bake_legacy_outside_extent_if_needed(data: Dict[str, Any]) -> Dict[str, Any]:
    """One-time bake of explicit outside_extent for legacy outer-band projects.

    When outer band is enabled and outside_extent is missing/0, store the old
    auto formula result so mesh geometry does not silently change under the
    new bubble_radius_factor-only auto policy.

    Never bake over an extent recovered from snappy chargeRefineOuter geometry
    (charge_outer_geometry present), and never invent metres that replace a
    preserved searchable* radius/points/box.
    """
    out = dict(data)
    if out.get("charge_outer_refine_enable") is False:
        return out
    if out.get("charge_outer_geometry"):
        # Geometry already loaded from case — do not replace with formula bake.
        return out
    oe_raw = out.get("outside_extent")
    try:
        oe_f = float(oe_raw) if oe_raw is not None else 0.0
    except (TypeError, ValueError):
        oe_f = 0.0
    if oe_f > 0.0:
        return out
    if out.get("charge_outer_legacy_migration_warning"):
        return out
    extent = _legacy_auto_outside_extent_m(out)
    out["outside_extent"] = extent
    out["charge_outer_legacy_migration_warning"] = (
        f"Legacy project: baked explicit outside_extent={extent:.6g} m using the "
        f"previous auto formula (bubble_radius_factor + transition_cells×level_span×cell_size) "
        f"so the mesh does not silently change under the new bubble_radius_factor-only policy."
    )
    return out


def _case_inputs_from_dict(data: Dict[str, Any]) -> CaseInputs3D:
    if not isinstance(data, dict):
        raise ProjectFormatError("case_inputs must be a JSON object")
    # True legacy projects lack an explicit seed-mode key (do not bake new projects).
    is_legacy_seed_project = (
        "charge_seed_mode" not in data or data.get("charge_seed_mode") in (None, "")
    )
    # Migrate seed/outer fields before field validation so new keys are known.
    from charge_seed_plan import SeedPolicyError

    try:
        data = migrate_case_inputs_seed_fields(data)
    except SeedPolicyError as exc:
        raise ProjectFormatError(f"Invalid charge seed policy: {exc}") from exc
    if is_legacy_seed_project:
        data = _bake_legacy_outside_extent_if_needed(data)
    allowed = {f.name: f for f in fields(CaseInputs3D)}
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ProjectFormatError(f"Unknown CaseInputs3D field(s): {', '.join(unknown)}")
    required = [
        name
        for name, f in allowed.items()
        if f.default is MISSING and f.default_factory is MISSING
    ]
    missing = [name for name in required if name not in data]
    if missing:
        raise ProjectFormatError(f"Missing required project field(s): {', '.join(missing)}")
    values = dict(data)
    for key in (
        "min_point",
        "max_point",
        "charge_center",
        "initiation_point",
        "remap_origin",
        "decomposition_simple_n",
    ):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    for key in ("probe_fields", "section_fields", "obstacle_fields", "volume_fields"):
        if key in values:
            values[key] = tuple(values[key])
    for key in ("probe_points", "surface_planes"):
        if key in values:
            values[key] = tuple(
                tuple(item) if isinstance(item, list) else item
                for item in values[key]
            )
    obstacles = values.get("obstacles", [])
    if not isinstance(obstacles, list):
        raise ProjectFormatError("case_inputs.obstacles must be a list")
    values["obstacles"] = [
        item if isinstance(item, ObstacleData) else ObstacleData(**item)
        for item in obstacles
    ]
    try:
        return CaseInputs3D(**values)
    except (TypeError, ValueError) as exc:
        raise ProjectFormatError(f"Invalid CaseInputs3D data: {exc}") from exc


def _case_inputs_2d_from_dict(data: Dict[str, Any]) -> CaseInputs2D:
    if not isinstance(data, dict):
        raise ProjectFormatError("dimensions.2D.case_inputs must be a JSON object")
    allowed = {f.name for f in fields(CaseInputs2D)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ProjectFormatError(f"Unknown CaseInputs2D field(s): {', '.join(unknown)}")
    values = dict(data)
    mapping = values.get("mapping", {})
    if isinstance(mapping, dict):
        values["mapping"] = MappingSource2D(**mapping)
    probes = values.get("probes", [])
    if not isinstance(probes, (list, tuple)):
        raise ProjectFormatError("dimensions.2D.case_inputs.probes must be a list")
    values["probes"] = tuple(
        item if isinstance(item, ProbePoint2D) else ProbePoint2D(**item)
        for item in probes
    )
    if "output_fields" in values:
        values["output_fields"] = tuple(values["output_fields"])
    if "vtk_fields" in values:
        values["vtk_fields"] = tuple(values["vtk_fields"])
    if "undefined_keys" in values:
        keys = values.get("undefined_keys") or ()
        if not isinstance(keys, (list, tuple)):
            raise ProjectFormatError(
                "dimensions.2D.case_inputs.undefined_keys must be a list"
            )
        values["undefined_keys"] = tuple(str(item) for item in keys)
    try:
        return CaseInputs2D(**values)
    except (TypeError, ValueError) as exc:
        raise ProjectFormatError(f"Invalid CaseInputs2D data: {exc}") from exc


def _case_inputs_1d_from_dict(data: Dict[str, Any]) -> CaseInputs1D:
    if not isinstance(data, dict):
        raise ProjectFormatError("dimensions.1D.case_inputs must be a JSON object")
    allowed = {f.name: f for f in fields(CaseInputs1D)}
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ProjectFormatError(f"Unknown CaseInputs1D field(s): {', '.join(unknown)}")
    required = [
        name
        for name, field_info in allowed.items()
        if field_info.default is MISSING and field_info.default_factory is MISSING
    ]
    missing = [name for name in required if name not in data]
    if missing:
        raise ProjectFormatError(
            f"Missing required 1D project field(s): {', '.join(missing)}"
        )
    values = dict(data)
    for key in ("probe_fields", "gauge_locations"):
        if key in values:
            values[key] = tuple(
                tuple(item) if isinstance(item, list) else item
                for item in values[key]
            )
    try:
        return CaseInputs1D(**values)
    except (TypeError, ValueError) as exc:
        raise ProjectFormatError(f"Invalid CaseInputs1D data: {exc}") from exc


def read_project(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectFormatError(f"Could not read project: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProjectFormatError("Project root must be a JSON object")
    version = payload.get("schema_version")
    migration = _MIGRATIONS.get(version)
    if migration is None:
        raise ProjectFormatError(
            f"Unsupported schema_version {version!r}; supported versions: "
            f"{', '.join(str(v) for v in sorted(_MIGRATIONS))}."
        )
    payload = migration(payload)
    available = payload.get("dimensions_available") or []
    if payload.get("project_dimension") != "3D" and "3D" not in available:
        raise ProjectFormatError("Only 3D GGUI projects are supported by this project format")
    case_inputs_raw = payload.get("case_inputs")
    dimensions = payload.get("dimensions", {})
    if dimensions is None:
        dimensions = {}
    if not isinstance(dimensions, dict):
        raise ProjectFormatError("dimensions must be a JSON object")
    section_3d = dimensions.get("3D")
    if isinstance(section_3d, dict) and isinstance(section_3d.get("case_inputs"), dict):
        case_inputs_raw = section_3d["case_inputs"]
    inputs = _case_inputs_from_dict(case_inputs_raw)
    section_1d = dimensions.get("1D")
    inputs_1d = None
    if section_1d is not None:
        if not isinstance(section_1d, dict):
            raise ProjectFormatError("dimensions.1D must be a JSON object")
        if section_1d.get("model") != "spherical-1d":
            raise ProjectFormatError("Unsupported dimensions.1D model")
        inputs_1d = _case_inputs_1d_from_dict(section_1d.get("case_inputs"))
    section_2d = dimensions.get("2D")
    inputs_2d = None
    if section_2d is not None:
        if not isinstance(section_2d, dict):
            raise ProjectFormatError("dimensions.2D must be a JSON object")
        if section_2d.get("model") != "axisymmetric-rz-wedge":
            raise ProjectFormatError("Unsupported dimensions.2D model")
        inputs_2d = _case_inputs_2d_from_dict(section_2d.get("case_inputs"))
    probes = payload.get("probes", {"probes": []})
    gui_state = payload.get("gui_state", {})
    if not isinstance(probes, dict) or not isinstance(gui_state, dict):
        raise ProjectFormatError("probes and gui_state must be JSON objects")
    # Never persist solver results; reject if a file claims otherwise after migration.
    provenance = payload.get("provenance") or {}
    if provenance.get("contains_runtime_results"):
        raise ProjectFormatError(
            "Project files must not contain solver results or large run outputs."
        )
    return {
        "payload": payload,
        "inputs": inputs,
        "inputs_1d": inputs_1d,
        "inputs_2d": inputs_2d,
        "probes": probes,
        "gui_state": gui_state,
    }
