"""Shared strict native-field readers for generated 2D and 3D remap."""
import os
import re
import numpy as np

def _parse_internal_field(path, is_vector=False):
    from remap_snapshot_1d import _parse_internal_field as strict_parse
    values, uniform = strict_parse(path, is_vector=is_vector)
    if values is None and uniform is None:
        raise ValueError("Missing or invalid remap field: %s" % path)
    return values, uniform

def _read_1d_data(source_case, time_dir):
    from remap_snapshot_1d import capture_arrays_from_time_dir
    data = capture_arrays_from_time_dir(source_case, time_dir)
    if data is None:
        raise ValueError("1D remap source has incomplete fields or coordinates.")
    return data

def _read_3d_cell_centres():
    values, uniform = _parse_internal_field(os.path.join("0", "C"), is_vector=True)
    if values is None or len(values) == 0:
        raise ValueError("Remap requires actual nonuniform 0/C cell centres; run writeCellCentres.")
    return values

def _read_0_orig_internal(zero_dir, name, n_cells, is_vector=False):
    path = os.path.join(zero_dir, name)
    values, uniform = _parse_internal_field(path, is_vector=is_vector)
    _read_bc(path)  # Validate all receiver boundaries before any field is replaced.
    if values is not None:
        if len(values) != n_cells:
            raise ValueError("Receiver field %s has the wrong cell count." % name)
        return np.asarray(values)
    if is_vector:
        return np.tile(uniform, (n_cells, 1))
    return np.full(n_cells, uniform)

def _read_bc(filepath):
    with open(filepath, encoding="utf-8") as handle:
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", handle.read(), flags=re.S)
    match = re.search(r"\bboundaryField\s*\{", text)
    if match is None:
        raise ValueError("Missing boundaryField in %s" % filepath)
    start = text.find("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{": depth += 1
        elif text[index] == "}": depth -= 1
        if depth == 0: return text[match.start():index+1]
    raise ValueError("Truncated boundaryField in %s" % filepath)


def verified_remap_source(source_case, source_time):
    from decimal import Decimal, InvalidOperation
    from remap_snapshot_1d import resolve_remap_source
    resolved = resolve_remap_source(source_case)
    if not resolved.ok or resolved.blocked or resolved.profile is None:
        raise ValueError(resolved.message or "Remap source is unavailable.")
    if str(source_time).lower() not in ("latest", "latesttime"):
        try:
            requested = Decimal(str(source_time))
            selected = Decimal(str(resolved.physical_time))
        except InvalidOperation as exc:
            raise ValueError("Invalid specific remap time.") from exc
        if not requested.is_finite() or requested != selected:
            raise ValueError("Specific remap time does not match the verified source time.")
    return resolved


def read_handoff_radius(source_case):
    """Absent legacy metadata uses full extent; present metadata must be valid."""
    import json
    import math
    path = os.path.join(source_case, "ggui_remap_handoff.json")
    try:
        with open(path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except FileNotFoundError:
        return 0.0
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read remap handoff radius metadata: %s" % path) from exc
    if not isinstance(metadata, dict):
        raise ValueError("Remap handoff metadata must be an object.")
    raw = metadata.get("remap_radius_m")
    if isinstance(raw, bool):
        raise ValueError("Remap handoff radius must be a positive finite length.")
    try:
        radius = float(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Remap handoff radius is missing or invalid.") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Remap handoff radius must be a positive finite length.")
    return radius
