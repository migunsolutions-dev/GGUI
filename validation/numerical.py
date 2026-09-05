"""Numerical quality diagnostics from logs, checkMesh, and output options.

Does not invent PASS for missing data. Does not treat disabled outputs as errors.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from foam_dictionary import read_top_level_entries
from output_options import REMAP_2D_FILENAME, OutputFileOptions
from validation.metrics import is_finite_number

_TIME = re.compile(r"\bTime\s*=\s*([0-9.eE+-]+)")
_DELTAT = re.compile(r"\bdeltaT\s*=\s*([0-9.eE+-]+)")
_COURANT = re.compile(
    r"Courant\s+Number\s+mean:\s*([0-9.eE+-]+)\s+max:\s*([0-9.eE+-]+)",
    re.I,
)
_COURANT_MEAN_MAX = re.compile(
    r"Courant\s+Number\s+Mean/Max\s*=\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)",
    re.I,
)
_FPE_CRASH = re.compile(
    r"(caught\s+floating\s+point\s+exception|"
    r"sigfpe\s*:\s*caught|"
    r"floating\s+point\s+exception(?!\s+trapping))",
    re.I,
)
_REFINE = re.compile(r"Refined\s+from\s+(\d+)\s+to\s+(\d+)\s+cells")
_UNREFINE = re.compile(r"Unrefined\s+from\s+(\d+)\s+to\s+(\d+)\s+cells")
_EXEC = re.compile(r"ExecutionTime\s*=\s*([0-9.]+)\s*s")
_CLOCK = re.compile(r"ClockTime\s*=\s*([0-9.]+)\s*s")
_NCELLS = re.compile(r"\bnCells:\s*(\d+)")
_NONORTHO = re.compile(r"Max\s+non-orthogonality\s*=\s*([0-9.eE+-]+)", re.I)
_SKEW = re.compile(r"Max\s+skewness\s*=\s*([0-9.eE+-]+)", re.I)


@dataclass
class TimeSeries:
    time: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)


@dataclass
class NumericalReport:
    dimension: str = ""
    solver: str = "blastFoam"
    run_status: str = "N/A"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    n_steps: Optional[int] = None
    n_cores: Optional[int] = None
    cpu_time_s: Optional[float] = None
    wall_time_s: Optional[float] = None
    max_co_configured: Optional[float] = None
    courant: TimeSeries = field(default_factory=TimeSeries)
    delta_t: TimeSeries = field(default_factory=TimeSeries)
    cells: TimeSeries = field(default_factory=TimeSeries)
    refine_events: int = 0
    unrefine_events: int = 0
    checkmesh_ok: Optional[bool] = None
    n_cells: Optional[int] = None
    max_nonortho: Optional[float] = None
    max_skewness: Optional[float] = None
    foam_fatal: Optional[bool] = None
    foam_error: Optional[bool] = None
    fpe: Optional[bool] = None
    completed: Optional[bool] = None
    reconstruct_ok: Optional[bool] = None
    completeness: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def parse_solver_log(text: str) -> Dict[str, Any]:
    times = [float(x) for x in _TIME.findall(text)]
    delta = [float(x) for x in _DELTAT.findall(text)]
    courant_max = []
    courant_t = []
    for match in list(_COURANT.finditer(text)) + list(_COURANT_MEAN_MAX.finditer(text)):
        try:
            courant_max.append(float(match.group(2)))
        except ValueError:
            continue
    refine = _REFINE.findall(text)
    unrefine = _UNREFINE.findall(text)
    exec_times = [float(x) for x in _EXEC.findall(text)]
    clock_times = [float(x) for x in _CLOCK.findall(text)]
    cells_series: List[Tuple[float, int]] = []
    for index, pair in enumerate(refine):
        t = times[min(index, len(times) - 1)] if times else float(index)
        cells_series.append((t, int(pair[1])))
    low = text.lower()
    return {
        "times": times,
        "delta_t": delta,
        "courant_max": courant_max,
        "refine_events": len(refine),
        "unrefine_events": len(unrefine),
        "cells_series": cells_series,
        "cpu_time_s": exec_times[-1] if exec_times else None,
        "wall_time_s": clock_times[-1] if clock_times else None,
        "foam_fatal": bool(re.search(r"FOAM\s+FATAL", text, re.I)),
        "foam_error": "foam error" in low,
        "fpe": bool(_FPE_CRASH.search(text)),
        "completed": ("end" in low[-1200:] or "finalising" in low[-1200:] or bool(exec_times)),
    }


def parse_checkmesh(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {"ok": None}
    ncells = _NCELLS.search(text)
    nonortho = _NONORTHO.search(text)
    skew = _SKEW.search(text)
    ok = bool(re.search(r"\bMesh OK\b", text))
    failed = bool(re.search(r"Failed\s+\d+\s+mesh checks", text, re.I))
    return {
        "ok": True if ok and not failed else False if (failed or "FOAM FATAL" in text) else None,
        "n_cells": int(ncells.group(1)) if ncells else None,
        "max_nonortho": float(nonortho.group(1)) if nonortho else None,
        "max_skewness": float(skew.group(1)) if skew else None,
    }


def _control_max_co(case_dir: str) -> Optional[float]:
    path = os.path.join(case_dir, "system", "controlDict")
    text = _read(path)
    if not text:
        return None
    values = read_top_level_entries(text, ("maxCo", "endTime", "startTime"))
    raw = values.get("maxCo")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _present(path: str) -> bool:
    return os.path.isfile(path) or os.path.isdir(path)


def completeness(
    case_dir: str,
    *,
    dim: str,
    options: Optional[OutputFileOptions],
    keep_openfoam_time_folders: bool,
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    if not case_dir or not os.path.isdir(case_dir):
        return [("case", "N/A")]
    dim_key = dim.lower()
    fo = {"1d": "gauges1d", "2d": "probes2d", "3d": "probes3d"}.get(dim_key, "")
    if fo:
        gauges_requested = True
        if options is not None:
            if dim_key == "1d":
                gauges_requested = True
            elif dim_key == "2d":
                gauges_requested = True
        probe_root = os.path.join(case_dir, "postProcessing", fo)
        if gauges_requested:
            rows.append(("Gauge output", "present" if os.path.isdir(probe_root) else "missing"))
        else:
            rows.append(("Gauge output", "not requested"))
    if dim_key == "2d" and options is not None:
        vtk_on = any(
            getattr(options.dim2d.vtk, key)
            for key in ("pressure", "density", "velocity", "mass_fractions", "temperature", "energy")
        )
        vtk_dir = os.path.join(case_dir, "VTK")
        if vtk_on:
            rows.append(("VTK frames", "present" if os.path.isdir(vtk_dir) else "missing"))
        else:
            rows.append(("VTK frames", "not requested"))
        if options.dim2d.output_remap_data:
            remap = os.path.join(case_dir, REMAP_2D_FILENAME)
            rows.append(("Remap output", "present" if os.path.isfile(remap) else "missing"))
        else:
            rows.append(("Remap output", "not requested"))
        if keep_openfoam_time_folders:
            from openfoam_times_2d import list_numeric_time_entries

            times = list_numeric_time_entries(case_dir)
            rows.append(("OpenFOAM time folders", "present" if times else "missing"))
        else:
            rows.append(("OpenFOAM time folders", "not requested (Keep OpenFOAM time folders = Off)"))
    if dim_key == "3d" and options is not None:
        if options.dim3d.write_volumes:
            vtk_dir = os.path.join(case_dir, "VTK")
            rows.append(("Volume VTK", "present" if os.path.isdir(vtk_dir) else "missing"))
        else:
            rows.append(("Volume VTK", "not requested"))
        if options.dim3d.write_surfaces:
            surf = os.path.join(case_dir, "postProcessing", "sectionsVTK")
            rows.append(("Sections", "present" if os.path.isdir(surf) else "missing"))
        else:
            rows.append(("Sections", "not requested"))
        if keep_openfoam_time_folders:
            from openfoam_times_2d import list_numeric_time_entries

            times = list_numeric_time_entries(case_dir)
            rows.append(("OpenFOAM time folders", "present" if times else "missing"))
        else:
            rows.append(("OpenFOAM time folders", "not requested (Keep OpenFOAM time folders = Off)"))
    return rows


def build_report(
    case_dir: str,
    *,
    dim: str = "",
    options: Optional[OutputFileOptions] = None,
    keep_openfoam_time_folders: bool = False,
) -> NumericalReport:
    report = NumericalReport(dimension=dim or "N/A")
    if not case_dir or not os.path.isdir(case_dir):
        report.notes.append("Current run does not contain the required validation data.")
        return report
    log = _read(os.path.join(case_dir, "log.blastFoam"))
    parsed = parse_solver_log(log) if log else {}
    if not log:
        report.notes.append("log.blastFoam is not available.")
    report.foam_fatal = parsed.get("foam_fatal") if log else None
    report.foam_error = parsed.get("foam_error") if log else None
    report.fpe = parsed.get("fpe") if log else None
    report.completed = parsed.get("completed") if log else None
    if report.foam_fatal:
        report.run_status = "FOAM FATAL"
    elif report.completed:
        report.run_status = "completed"
    elif log:
        report.run_status = "log present"
    times = list(parsed.get("times") or [])
    report.start_time = times[0] if times else None
    report.end_time = times[-1] if times else None
    report.n_steps = len(times) if times else None
    report.cpu_time_s = parsed.get("cpu_time_s")
    report.wall_time_s = parsed.get("wall_time_s")
    report.refine_events = int(parsed.get("refine_events") or 0)
    report.unrefine_events = int(parsed.get("unrefine_events") or 0)
    dt = list(parsed.get("delta_t") or [])
    n = min(len(times), len(dt))
    report.delta_t = TimeSeries(time=times[:n], values=dt[:n])
    cm = list(parsed.get("courant_max") or [])
    nc = min(len(times), len(cm))
    report.courant = TimeSeries(time=times[:nc], values=cm[:nc])
    cells = parsed.get("cells_series") or []
    report.cells = TimeSeries(time=[c[0] for c in cells], values=[float(c[1]) for c in cells])
    report.max_co_configured = _control_max_co(case_dir)
    check = parse_checkmesh(_read(os.path.join(case_dir, "log.checkMesh")))
    report.checkmesh_ok = check.get("ok")
    report.n_cells = check.get("n_cells")
    report.max_nonortho = check.get("max_nonortho")
    report.max_skewness = check.get("max_skewness")
    if not report.cells.values:
        n_fixed = report.n_cells
        if n_fixed is None:
            owner = _read(os.path.join(case_dir, "constant", "polyMesh", "owner"))
            listed = re.search(r"nCells:\s*(\d+)", owner)
            if listed:
                n_fixed = int(listed.group(1))
            else:
                count_match = re.search(r"\n\s*(\d+)\s*\n\s*\(", owner)
                if count_match:
                    n_fixed = int(count_match.group(1))
        if n_fixed is not None:
            if report.n_cells is None:
                report.n_cells = n_fixed
            if times:
                report.cells = TimeSeries(
                    time=[times[0], times[-1]],
                    values=[float(n_fixed), float(n_fixed)],
                )
                report.notes.append("Cell count is constant (no AMR refine lines in the log).")
    recon = _read(os.path.join(case_dir, "log.reconstructPar")) or _read(
        os.path.join(case_dir, "log.reconstructFinal")
    )
    if recon:
        report.reconstruct_ok = "FOAM FATAL" not in recon.upper()
    decomp = os.path.join(case_dir, "system", "decomposeParDict")
    if os.path.isfile(decomp):
        text = _read(decomp)
        entries = read_top_level_entries(text, ("numberOfSubdomains",))
        try:
            report.n_cores = int(float(entries.get("numberOfSubdomains") or "0")) or None
        except (TypeError, ValueError):
            report.n_cores = None
    report.completeness = completeness(
        case_dir,
        dim=dim,
        options=options,
        keep_openfoam_time_folders=keep_openfoam_time_folders,
    )
    return report
