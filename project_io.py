"""Versioned, human-readable GGUI project persistence."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import MISSING, asdict, fields
from datetime import datetime, timezone
from typing import Any, Dict

from models import CaseInputs3D, ObstacleData

SCHEMA_VERSION = 1
PROJECT_SUFFIX = ".ggui.json"


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


def _case_inputs_from_dict(data: Dict[str, Any]) -> CaseInputs3D:
    if not isinstance(data, dict):
        raise ProjectFormatError("case_inputs must be a JSON object")
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
