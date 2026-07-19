"""Collect runtime AMR metrics from solver logs and case_init_mode.json."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.amr_tuning_sweep import parse_blastfoam_log  # noqa: E402
from tools.validation_harness.io_case import find_blastfoam_log, load_case_init_mode
from tools.validation_harness.models import RuntimeMetrics


def collect_runtime_metrics(case_dir: Path, mode: Optional[Dict[str, Any]] = None) -> RuntimeMetrics:
    mode = mode if mode is not None else load_case_init_mode(case_dir)
    amr = mode.get("amr_written") or {}

    indicator = None
    if isinstance(amr, dict):
        line = (amr.get("errorEstimator_line") or "").strip()
        if "densityGradient" in line:
            indicator = "densityGradient"
        elif "scaledDelta" in line:
            if amr.get("uses_scaled_delta_pressure"):
                indicator = "scaledDelta_p"
            else:
                indicator = "scaledDelta"
        else:
            indicator = line or None

    max_ref = amr.get("maxRefinement") if isinstance(amr, dict) else None

    log_path = find_blastfoam_log(case_dir)
    if not log_path:
        return RuntimeMetrics(
            amr_indicator=indicator,
            max_refinement=_i(max_ref),
            extra={"log_found": False},
        )

    parsed = parse_blastfoam_log(log_path.read_text(encoding="utf-8", errors="replace"))
    peak = parsed.get("peak_cell_count")
    final = parsed.get("final_cell_count_from_log")
    release_ratio = None
    if peak and final and peak > 0:
        release_ratio = float(final) / float(peak)

    series_raw = parsed.get("cell_series") or []
    cell_series: List[Dict[str, Any]] = []
    for item in series_raw:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            cell_series.append(
                {"kind": item[0], "from_cells": item[1], "to_cells": item[2]}
            )

    persistence = None
    if parsed.get("unrefine_events", 0) > 0:
        persistence = parsed.get("unrefine_after_first_refine") is False

    return RuntimeMetrics(
        peak_cell_count=_i(peak),
        final_cell_count=_i(final),
        release_ratio=release_ratio,
        refine_events=_i(parsed.get("refine_events")),
        unrefine_events=_i(parsed.get("unrefine_events")),
        refinement_persistence=persistence,
        amr_indicator=indicator,
        max_refinement=_i(max_ref),
        final_sim_time_s=_f(parsed.get("final_time")),
        wall_execution_time_s=_f(parsed.get("last_execution_time_s")),
        cell_series=cell_series,
        extra={
            "foam_fatal": parsed.get("foam_fatal"),
            "blastfoam_started": parsed.get("blastfoam_started"),
            "initial_cell_count_from_log": parsed.get("initial_cell_count_from_log"),
        },
    )


def _f(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
