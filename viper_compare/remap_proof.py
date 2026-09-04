"""Prove VIPER actually propagated remapped 2D, not only initialized it."""
from __future__ import annotations

import os
import re
from typing import List, Optional

_STEP_LINE = re.compile(
    r"Step=\s*(\d+)\s+TT=\s*([0-9.eE+-]+).*?Step=\s*(\d+)\s+TT=\s*([0-9.eE+-]+)"
)


def vtk_2d_frames(case_dir: str) -> List[str]:
    names = []
    try:
        listing = os.listdir(case_dir)
    except OSError:
        return []
    for name in listing:
        if re.fullmatch(r"viper2d_\d+\.vtk", name, flags=re.I):
            names.append(name)
    return sorted(names, key=lambda n: int(re.search(r"(\d+)", n).group(1)))


def run_summary_has_2d(case_dir: str) -> bool:
    path = os.path.join(case_dir, "viper_RunSummary.txt")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    return "2D Simulation" in text


def _echo_path(case_dir: str) -> str:
    for name in ("vprt.txt", "status_echo.txt"):
        path = os.path.join(case_dir, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    return ""


def max_2d_step_from_echo(case_dir: str) -> Optional[int]:
    path = _echo_path(case_dir)
    if not path:
        return None
    max_step = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            match = _STEP_LINE.search(raw)
            if not match:
                continue
            step_2d = int(match.group(3))
            tt_2d = float(match.group(4))
            if max_step is None or step_2d > max_step:
                max_step = step_2d
            # remap init copies 1D time into 2D at step 0; that is not propagation
            _ = tt_2d
    return max_step


def remap_propagation_report(case_dir: str) -> dict:
    frames = vtk_2d_frames(case_dir)
    max_step = max_2d_step_from_echo(case_dir)
    has_2d = run_summary_has_2d(case_dir)
    th_p = os.path.join(case_dir, "viper2d_th_overpressure.txt")
    th_bytes = os.path.getsize(th_p) if os.path.isfile(th_p) else 0
    return {
        "case_dir": case_dir,
        "vtk_frames": frames,
        "n_vtk": len(frames),
        "max_2d_step": max_step,
        "run_summary_has_2d": has_2d,
        "th_2d_bytes": th_bytes,
        "initialized_only": (
            (max_step is None or max_step <= 0)
            and len(frames) <= 1
            and not has_2d
        ),
    }


def require_remap_propagation(case_dir: str) -> dict:
    report = remap_propagation_report(case_dir)
    reasons = []
    if report["n_vtk"] < 2:
        reasons.append(f"need >=2 VTK frames, got {report['vtk_frames']}")
    if report["max_2d_step"] is None or report["max_2d_step"] <= 0:
        reasons.append(f"2D step never advanced (max={report['max_2d_step']})")
    if not report["run_summary_has_2d"]:
        reasons.append("viper_RunSummary.txt has no 2D Simulation section")
    if report["th_2d_bytes"] <= 0:
        reasons.append("viper2d_th_overpressure.txt is empty/missing")
    if reasons:
        raise RuntimeError("VIPER remap did not propagate 2D: " + "; ".join(reasons))
    return report
