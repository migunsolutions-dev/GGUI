"""Run VIPER nogui in a writable directory. Never write under Program Files."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Sequence

from viper_compare.cli import viper_argv
from viper_compare.extract import histories_empty

VIPER_EXE = r"C:\Program Files\viperblast_1.31\viperblast.exe"
TH_1D_P = "viper1d_th_overpressure.txt"
TH_1D_I = "viper1d_th_impulse.txt"
TH_2D_P = "viper2d_th_overpressure.txt"
TH_2D_I = "viper2d_th_impulse.txt"


def run_viper(
    *,
    case_dir: str,
    vip_path: str,
    json_path: str,
    stages: Sequence[str],
    exe: str = VIPER_EXE,
    timeout_s: float | None = None,
) -> dict:
    Path(case_dir).mkdir(parents=True, exist_ok=True)
    argv = viper_argv(exe, vip_path, json_path, stages)
    cmd_path = os.path.join(case_dir, "command.json")
    with open(cmd_path, "w", encoding="utf-8") as handle:
        json.dump({"argv": argv, "cwd": case_dir}, handle, indent=2)
    t0 = time.perf_counter()
    proc = subprocess.run(
        argv,
        cwd=case_dir,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    wall = time.perf_counter() - t0
    stdout_path = os.path.join(case_dir, "stdout.txt")
    stderr_path = os.path.join(case_dir, "stderr.txt")
    Path(stdout_path).write_text(proc.stdout or "", encoding="utf-8")
    Path(stderr_path).write_text(proc.stderr or "", encoding="utf-8")
    files = sorted(os.listdir(case_dir))
    vprt = os.path.join(case_dir, "vprt.txt")
    result = {
        "argv": argv,
        "cwd": case_dir,
        "returncode": proc.returncode,
        "wall_s": wall,
        "files": files,
        "vprt_exists": os.path.isfile(vprt),
        "vprt_bytes": os.path.getsize(vprt) if os.path.isfile(vprt) else 0,
        "th_1d_p": os.path.join(case_dir, TH_1D_P),
        "th_1d_i": os.path.join(case_dir, TH_1D_I),
        "th_2d_p": os.path.join(case_dir, TH_2D_P),
        "th_2d_i": os.path.join(case_dir, TH_2D_I),
        "th_1d_p_empty": histories_empty(os.path.join(case_dir, TH_1D_P)),
        "th_2d_p_empty": histories_empty(os.path.join(case_dir, TH_2D_P)),
    }
    with open(os.path.join(case_dir, "run_result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def require_nonempty_th(result: dict, *, need_1d: bool, need_2d: bool) -> None:
    if need_1d and result.get("th_1d_p_empty", True):
        raise RuntimeError(
            f"VIPER 1D overpressure history is empty: {result.get('th_1d_p')}"
        )
    if need_2d and result.get("th_2d_p_empty", True):
        raise RuntimeError(
            f"VIPER 2D overpressure history is empty: {result.get('th_2d_p')}"
        )
