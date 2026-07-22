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
from models import CaseInputs3D, ObstacleData

SCHEMA_VERSION = 1
PROJECT_SUFFIX = ".ggui.json"


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


class ProjectFormatError(ValueError):
    pass


def _migrate_v1(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload


_MIGRATIONS = {1: _migrate_v1}


def build_project(
    inputs: CaseInputs3D,
    *,
    probes: Dict[str, Any],
    gui_state: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "application": {
            "name": "GGUI",
            "version": "4.0",
            "saved_utc": datetime.now(timezone.utc).isoformat(),
        },
        "project_dimension": "3D",
        "case_inputs": asdict(inputs),
        "probes": probes,
        "gui_state": gui_state,
        "provenance": {
            "format": "explicit-json",
            "contains_runtime_results": False,
        },
    }


def capture_project_payload(tab: _SupportsProjectCapture, probes_model: _SupportsProbesDict) -> Dict[str, Any]:
    """Capture dialog-independent project JSON from the live 3D GUI state."""
    return build_project(
        tab.get_case_inputs(),
        probes=probes_model.to_dict(),
        gui_state={
            "selected_primary_tab": "General 3D",
            "sections": [asdict(section) for section in tab.sections],
            "obstacles": [asdict(obstacle) for obstacle in tab.obstacles],
        },
    )


def apply_project_payload(
    tab: _SupportsProjectCapture,
    probes_model: _SupportsProbesDict,
    project: Dict[str, Any],
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
    if payload.get("project_dimension") != "3D":
        raise ProjectFormatError("Only 3D GGUI projects are supported by this project format")
    inputs = _case_inputs_from_dict(payload.get("case_inputs"))
    probes = payload.get("probes", {"probes": []})
    gui_state = payload.get("gui_state", {})
    if not isinstance(probes, dict) or not isinstance(gui_state, dict):
        raise ProjectFormatError("probes and gui_state must be JSON objects")
    return {
        "payload": payload,
        "inputs": inputs,
        "probes": probes,
        "gui_state": gui_state,
    }
