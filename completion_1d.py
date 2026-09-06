"""1D run modes: Terminate (stop on verified arrival) vs Reflect (run to End Time).

Qt-free. Used by generation, the solver runner, and result classification.
User-facing ``endTime`` is always written; its meaning depends on the mode.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from models import (
    SOURCE_MODEL_JWL,
    SOURCE_MODEL_SCHEMA_VERSION,
    normalize_source_model,
)
from validation.map_1d import KIND_EXACT, map_radius
from validation.metrics import is_finite_number
from validation.probes import (
    latest_probe_field_file,
    parse_probe_history,
    radial_distance,
    radii_from_locations,
    series_for_index,
    xyz_from_location,
)

RUN_MODE_TERMINATE = "terminate"
RUN_MODE_REFLECT = "reflect"
RUN_MODE_OPTIONS = (RUN_MODE_TERMINATE, RUN_MODE_REFLECT)

# Shared GUI / Allrun-harness poll while waiting for the arrival probe.
# 1D remap steps ~300/s; 0.25 s of latency lets the front travel several cells
# past R_handoff before writeNow is published.
WATCHDOG_POLL_S = 0.10

# Legacy aliases kept so older tests/projects still import a name.
STOP_MODE_TERMINATE = RUN_MODE_TERMINATE
STOP_MODE_REFLECT = RUN_MODE_REFLECT
STOP_MODE_WAVE_RADIUS = RUN_MODE_TERMINATE
STOP_MODE_END_TIME = RUN_MODE_REFLECT

STOP_REASON_WAVE_RADIUS_REACHED = "wave_radius_reached"
STOP_REASON_NO_ARRIVAL = "no_verified_arrival"
STOP_REASON_END_TIME_WITHOUT_ARRIVAL = "end_time_reached_without_wave_arrival"
STOP_REASON_END_TIME_REACHED = "end_time_reached"
STOP_REASON_USER_STOPPED = "user_stopped"
STOP_REASON_USER_INTERRUPT = STOP_REASON_USER_STOPPED
STOP_REASON_SOLVER_ERROR = "solver_error"

COMPLETION_FILENAME = "ggui_1d_run_completion.json"
ARRIVAL_OVERPRESSURE_PA = 8000.0
ARRIVAL_CRITERION = (
    "overpressure_above_ambient: first probe sample with "
    f"(p - p_atm) >= {ARRIVAL_OVERPRESSURE_PA:.0f} Pa"
)
WATCHDOG_FO = "watchdog_probe"
PROBES1D_FO = "probes1d"
RIGHT_BOUNDARY_TERMINATE = "Terminate"
RIGHT_BOUNDARY_REFLECT = "Reflect"


@dataclass
class CompletionRecord:
    mode: str = RUN_MODE_TERMINATE
    stop_mode: str = RUN_MODE_TERMINATE
    right_boundary: str = RIGHT_BOUNDARY_TERMINATE
    requested_stop_radius_m: Optional[float] = None
    end_time_s: Optional[float] = None
    p_atm: float = 101325.0
    threshold_overpressure_pa: float = ARRIVAL_OVERPRESSURE_PA
    criterion: str = ARRIVAL_CRITERION
    wave_radius_reached: bool = False
    detected_arrival_time_s: Optional[float] = None
    probe_function_object: str = ""
    probe_index: Optional[int] = None
    probe_location: str = ""
    probe_radius_m: Optional[float] = None
    final_solver_time_s: Optional[float] = None
    stop_reason: str = ""
    return_code: Optional[int] = None
    remap_for_2d: bool = False
    remap_radius_m: Optional[float] = None
    dr_1d_m: Optional[float] = None
    remap_front_buffer_cells: Optional[int] = None
    handoff_radius_m: Optional[float] = None
    # Which blast source produced this run. Records written before the IG feature
    # have no such key and resolve to JWL, which is what they were.
    source_model: str = SOURCE_MODEL_JWL
    source_model_schema_version: int = SOURCE_MODEL_SCHEMA_VERSION

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode
        data["stop_mode"] = self.mode
        return data


def completion_path(case_dir: str) -> str:
    return os.path.join(case_dir or "", COMPLETION_FILENAME)


def normalize_run_mode(
    value: Optional[str] = None,
    right_boundary: Optional[str] = None,
) -> str:
    """Resolve Terminate vs Reflect. An explicit right-boundary label wins."""
    rb = str(right_boundary or "").strip().lower()
    if rb in ("reflect", "reflecting"):
        return RUN_MODE_REFLECT
    if rb in ("terminate", "transmit", "transmissive", "outflow"):
        return RUN_MODE_TERMINATE
    text = str(value or "").strip().lower()
    if text in ("reflect", "reflecting", "end_time"):
        return RUN_MODE_REFLECT
    return RUN_MODE_TERMINATE


def normalize_stop_mode(
    value: Optional[str] = None,
    right_boundary: Optional[str] = None,
) -> str:
    return normalize_run_mode(value, right_boundary)


def right_boundary_for_mode(mode: str) -> str:
    return (
        RIGHT_BOUNDARY_REFLECT
        if normalize_run_mode(mode) == RUN_MODE_REFLECT
        else RIGHT_BOUNDARY_TERMINATE
    )


def is_terminate_mode(record: CompletionRecord) -> bool:
    return normalize_run_mode(record.mode or record.stop_mode, record.right_boundary) == (
        RUN_MODE_TERMINATE
    )


def is_reflect_mode(record: CompletionRecord) -> bool:
    return not is_terminate_mode(record)


def write_completion_record(case_dir: str, record: CompletionRecord) -> str:
    path = completion_path(case_dir)
    payload = record.as_dict()
    payload["requested_radius_m"] = record.requested_stop_radius_m
    payload["endTime"] = record.end_time_s
    payload["arrival_event"] = bool(record.wave_radius_reached)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def read_completion_record(case_dir: str) -> Optional[CompletionRecord]:
    path = completion_path(case_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return CompletionRecord(
            mode=normalize_run_mode(
                data.get("mode") or data.get("stop_mode"),
                data.get("right_boundary"),
            ),
            stop_mode=normalize_run_mode(
                data.get("mode") or data.get("stop_mode"),
                data.get("right_boundary"),
            ),
            right_boundary=str(
                data.get("right_boundary")
                or right_boundary_for_mode(
                    data.get("mode") or data.get("stop_mode")
                )
            ),
            requested_stop_radius_m=_opt_float(
                data.get("requested_radius_m", data.get("requested_stop_radius_m"))
            ),
            end_time_s=_opt_float(data.get("end_time_s", data.get("endTime"))),
            p_atm=float(data.get("p_atm") or 101325.0),
            threshold_overpressure_pa=float(
                data.get("threshold_overpressure_pa") or ARRIVAL_OVERPRESSURE_PA
            ),
            criterion=str(data.get("criterion") or ARRIVAL_CRITERION),
            wave_radius_reached=bool(data.get("wave_radius_reached")),
            detected_arrival_time_s=_opt_float(data.get("detected_arrival_time_s")),
            probe_function_object=str(data.get("probe_function_object") or ""),
            probe_index=None if data.get("probe_index") is None else int(data["probe_index"]),
            probe_location=str(data.get("probe_location") or ""),
            probe_radius_m=_opt_float(data.get("probe_radius_m")),
            final_solver_time_s=_opt_float(data.get("final_solver_time_s")),
            stop_reason=str(data.get("stop_reason") or ""),
            return_code=None if data.get("return_code") is None else int(data["return_code"]),
            remap_for_2d=bool(data.get("remap_for_2d")),
            remap_radius_m=_opt_float(data.get("remap_radius_m")),
            dr_1d_m=_opt_float(data.get("dr_1d_m")),
            remap_front_buffer_cells=_opt_int(data.get("remap_front_buffer_cells")),
            handoff_radius_m=_opt_float(data.get("handoff_radius_m")),
            source_model=normalize_source_model(data.get("source_model")),
            source_model_schema_version=_opt_int(
                data.get("source_model_schema_version")
            ) or SOURCE_MODEL_SCHEMA_VERSION,
        )
    except (TypeError, ValueError):
        return None


def initial_completion_record(
    *,
    stop_mode: Optional[str] = None,
    mode: Optional[str] = None,
    requested_stop_radius_m: Optional[float],
    p_atm: float = 101325.0,
    right_boundary: Optional[str] = None,
    end_time_s: Optional[float] = None,
    remap_for_2d: bool = False,
    remap_radius_m: Optional[float] = None,
    dr_1d_m: Optional[float] = None,
    remap_front_buffer_cells: Optional[int] = None,
    handoff_radius_m: Optional[float] = None,
    criterion: Optional[str] = None,
    source_model: Optional[str] = None,
) -> CompletionRecord:
    resolved = normalize_run_mode(mode or stop_mode, right_boundary)
    remap = bool(remap_for_2d)
    if criterion is not None:
        resolved_criterion = str(criterion)
    elif remap:
        from remap_handoff_1d import HANDOFF_CRITERION

        resolved_criterion = HANDOFF_CRITERION
    else:
        resolved_criterion = ARRIVAL_CRITERION
    return CompletionRecord(
        mode=resolved,
        stop_mode=resolved,
        right_boundary=str(right_boundary or right_boundary_for_mode(resolved)),
        requested_stop_radius_m=requested_stop_radius_m,
        end_time_s=_opt_float(end_time_s),
        p_atm=float(p_atm) if is_finite_number(p_atm) else 101325.0,
        threshold_overpressure_pa=ARRIVAL_OVERPRESSURE_PA,
        criterion=resolved_criterion,
        wave_radius_reached=False,
        detected_arrival_time_s=None,
        probe_function_object="",
        probe_index=None,
        probe_location="",
        probe_radius_m=None,
        final_solver_time_s=None,
        stop_reason="",
        return_code=None,
        remap_for_2d=remap,
        remap_radius_m=_opt_float(remap_radius_m),
        dr_1d_m=_opt_float(dr_1d_m),
        remap_front_buffer_cells=_opt_int(remap_front_buffer_cells),
        handoff_radius_m=_opt_float(handoff_radius_m),
        source_model=normalize_source_model(source_model),
        source_model_schema_version=SOURCE_MODEL_SCHEMA_VERSION,
    )


def reset_completion_for_new_run(case_dir: str) -> CompletionRecord:
    """Clear stale arrival/final-time evidence at the start of a run."""
    existing = read_completion_record(case_dir)
    if existing is None:
        record = initial_completion_record(
            mode=RUN_MODE_TERMINATE, requested_stop_radius_m=None
        )
    else:
        record = initial_completion_record(
            mode=existing.mode,
            requested_stop_radius_m=existing.requested_stop_radius_m,
            p_atm=existing.p_atm,
            right_boundary=existing.right_boundary,
            end_time_s=existing.end_time_s,
            remap_for_2d=existing.remap_for_2d,
            remap_radius_m=existing.remap_radius_m,
            dr_1d_m=existing.dr_1d_m,
            remap_front_buffer_cells=existing.remap_front_buffer_cells,
            handoff_radius_m=existing.handoff_radius_m,
            criterion=existing.criterion,
            # Carried, not re-derived: a re-run of an existing case must not silently
            # switch source model just because the arrival evidence was cleared.
            source_model=existing.source_model,
        )
        record.threshold_overpressure_pa = existing.threshold_overpressure_pa
    write_completion_record(case_dir, record)
    try:
        from remap_snapshot_1d import invalidate_snapshot

        invalidate_snapshot(case_dir)
    except Exception:
        pass
    return record


def stop_mode_for_case(case_dir: str) -> str:
    record = read_completion_record(case_dir)
    if record is not None:
        return record.mode
    return RUN_MODE_TERMINATE


def arrival_time_from_history(
    times: Sequence[float],
    pressures: Sequence[float],
    *,
    p_atm: float,
    threshold_pa: float = ARRIVAL_OVERPRESSURE_PA,
) -> Optional[float]:
    """First time overpressure at the probe is at least ``threshold_pa`` above ambient."""
    if not is_finite_number(p_atm) or not is_finite_number(threshold_pa):
        return None
    atm = float(p_atm)
    thr = float(threshold_pa)
    for raw_t, raw_p in zip(times, pressures):
        if not is_finite_number(raw_t) or not is_finite_number(raw_p):
            continue
        if float(raw_p) - atm >= thr:
            return float(raw_t)
    return None


def overpressure_arrived(
    pressure: float,
    *,
    p_atm: float,
    threshold_pa: float = ARRIVAL_OVERPRESSURE_PA,
) -> bool:
    if not is_finite_number(pressure) or not is_finite_number(p_atm):
        return False
    return float(pressure) - float(p_atm) >= float(threshold_pa)


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _opt_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _probe_ref_from_file(
    case_dir: str,
    fo_name: str,
    requested_radius: float,
) -> Optional[Tuple[int, str, float]]:
    path = latest_probe_field_file(case_dir, fo_name, "p")
    if not path:
        return None
    locations, _times, _cols = parse_probe_history(path)
    radii = radii_from_locations(locations, dim="1d")
    mapping = map_radius(radii, requested_radius)
    index = None
    radius = None
    if mapping.ok and mapping.kind == KIND_EXACT and mapping.index_lo is not None:
        index = mapping.index_lo
        radius = mapping.r_lo
    else:
        best_i = None
        best_d = None
        for i, r in enumerate(radii):
            if not is_finite_number(r):
                continue
            d = abs(float(r) - float(requested_radius))
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        if best_i is None or best_d is None:
            return None
        scale = max(abs(float(requested_radius)), 1.0e-6)
        if best_d > max(1.0e-4, 0.02 * scale):
            return None
        index = best_i
        radius = float(radii[best_i]) if is_finite_number(radii[best_i]) else None
    if index is None or index >= len(locations):
        return None
    loc = locations[index]
    xyz = xyz_from_location(loc)
    if radius is None and xyz is not None:
        radius = radial_distance(*xyz)
    return index, str(loc), float(radius) if radius is not None else float(requested_radius)


def resolve_arrival_probe(
    case_dir: str, requested_radius: float
) -> Optional[Tuple[str, int, str, float]]:
    """Return (function_object, index, location, radius_m) at/near the stop radius."""
    if not is_finite_number(requested_radius) or float(requested_radius) <= 0.0:
        return None
    target = float(requested_radius)
    for fo in (WATCHDOG_FO, PROBES1D_FO):
        found = _probe_ref_from_file(case_dir, fo, target)
        if found is not None:
            index, loc, radius = found
            return fo, index, loc, radius
    return None


def detect_arrival_in_case(
    case_dir: str,
    record: CompletionRecord,
) -> CompletionRecord:
    """Fill arrival fields from probe histories. Does not decide endTime success."""
    updated = CompletionRecord(**asdict(record))
    already = bool(
        updated.wave_radius_reached
        and is_finite_number(updated.detected_arrival_time_s)
    )
    radius = updated.requested_stop_radius_m
    if not is_finite_number(radius) or float(radius) <= 0.0:
        return updated
    resolved = resolve_arrival_probe(case_dir, float(radius))
    if resolved is None:
        return updated
    fo, index, loc, probe_r = resolved
    updated.probe_function_object = fo
    updated.probe_index = index
    updated.probe_location = loc
    updated.probe_radius_m = probe_r
    path = latest_probe_field_file(case_dir, fo, "p")
    if not path:
        return updated
    _locs, times, cols = parse_probe_history(path)
    _t, pressures = series_for_index(times, cols, index)
    if updated.remap_for_2d:
        from remap_handoff_1d import primary_shock_arrival_time

        arrived_at = primary_shock_arrival_time(_t, pressures, updated.p_atm)
    else:
        arrived_at = arrival_time_from_history(
            _t,
            pressures,
            p_atm=updated.p_atm,
            threshold_pa=updated.threshold_overpressure_pa,
        )
    if arrived_at is not None:
        updated.wave_radius_reached = True
        if not is_finite_number(updated.detected_arrival_time_s):
            updated.detected_arrival_time_s = arrived_at
    elif already:
        updated.wave_radius_reached = True
    return updated


def finalize_completion_record(
    case_dir: str,
    *,
    return_code: Optional[int],
    user_stopped: bool,
    final_solver_time_s: Optional[float],
    reached_end_time: bool,
    foam_fatal: bool = False,
    end_time_s: Optional[float] = None,
) -> CompletionRecord:
    """Persist stop reason and arrival evidence after the solver process exits."""
    record = read_completion_record(case_dir) or initial_completion_record(
        mode=RUN_MODE_TERMINATE, requested_stop_radius_m=None
    )
    record.return_code = return_code
    if is_finite_number(end_time_s):
        record.end_time_s = float(end_time_s)
    if is_finite_number(final_solver_time_s):
        record.final_solver_time_s = float(final_solver_time_s)
    record = detect_arrival_in_case(case_dir, record)
    if user_stopped:
        record.stop_reason = STOP_REASON_USER_STOPPED
        write_completion_record(case_dir, record)
        return record
    if foam_fatal:
        record.stop_reason = STOP_REASON_SOLVER_ERROR
        write_completion_record(case_dir, record)
        return record
    if is_terminate_mode(record):
        if record.wave_radius_reached:
            record.stop_reason = STOP_REASON_WAVE_RADIUS_REACHED
        elif reached_end_time:
            record.stop_reason = STOP_REASON_END_TIME_WITHOUT_ARRIVAL
        else:
            record.stop_reason = STOP_REASON_NO_ARRIVAL
    else:
        if reached_end_time:
            record.stop_reason = STOP_REASON_END_TIME_REACHED
        else:
            record.stop_reason = STOP_REASON_SOLVER_ERROR
    write_completion_record(case_dir, record)
    return record


def wave_radius_stop_is_success(record: CompletionRecord) -> bool:
    return bool(
        is_terminate_mode(record)
        and record.wave_radius_reached
        and record.stop_reason == STOP_REASON_WAVE_RADIUS_REACHED
        and is_finite_number(record.detected_arrival_time_s)
    )


def reflect_end_time_is_success(record: CompletionRecord) -> bool:
    return bool(
        is_reflect_mode(record)
        and record.stop_reason == STOP_REASON_END_TIME_REACHED
        and is_finite_number(record.final_solver_time_s)
    )
