"""Final numbers for the extra CFL points, JWL RK2, and shock thickness."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "viper_1d"))
sys.path.insert(0, str(ROOT / "validation" / "jwl_internal"))
sys.path.insert(0, str(ROOT / "validation" / "rarefaction_osc"))

import run_compare as rc
import run_models as jm
from diagnose_raw import load
from post_ab import choose_folder, gauge_rows, oscillation

P_ATM = 101325.0
WORK = Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work")
CASES = {
    "IG_cfl025": WORK / "OscIG_cfl025_20261003_152933",
    "IG_cfl030": WORK / "OscIG_cfl030_20261003_152957",
    "IG_dx1mm": WORK / "OscIG_dx1mm_20261003_152640",
    "IG_rk2": WORK / "OscIG_rk2_20261003_152729",
    "IG_cfl01": WORK / "OscIG_cfl01_20261003_152739",
    "IG_upwind": WORK / "OscIG_upwind_20261003_152713",
    "JWL_rk2": WORK / "OscJWL_rk2",
    "JWL_upwind": WORK / "OscJWL_upwind",
    "JWL_tadmor": WORK / "JwlFlux_Tadmor",
}


def rise_cells(data) -> dict:
    p = data["p"]
    r = data["r"]
    outer = int(np.where(p > 2 * P_ATM)[0][-1]) if np.any(p > 2 * P_ATM) else int(np.argmax(p))
    behind = p[max(0, outer - 12) : max(1, outer - 4)]
    level = float(np.median(behind)) if behind.size else float(p[outer])
    hi = P_ATM + 0.9 * (level - P_ATM)
    lo = P_ATM + 0.1 * (level - P_ATM)
    i_hi = i_lo = None
    for i in range(min(p.size - 1, outer + 2), max(0, outer - 30), -1):
        if i_hi is None and p[i] >= hi:
            i_hi = i
        if i_hi is not None and p[i] <= lo:
            i_lo = i
            break
    if i_hi is None or i_lo is None:
        return {"rise_cells": None}
    return {
        "rise_cells": int(i_hi - i_lo),
        "rise_m": float(r[i_hi] - r[i_lo]),
        "level_pa": level,
    }


def tadmor_time_cost() -> None:
    text = (WORK / "JwlFlux_Tadmor" / "log.blastFoam").read_text(encoding="utf-8", errors="replace")
    times = [float(v) for v in re.findall(r"^Time = ([0-9.eE+-]+)", text, re.M)]
    costs = [float(v) for v in re.findall(r"ExecutionTime = ([0-9.]+) s", text)]
    if not times or not costs:
        print("tadmor log parse", len(times), len(costs))
        return
    n = min(len(times), len(costs))
    idx = int(np.argmin(np.abs(np.asarray(times[:n]) - 0.00025)))
    print(f"Tadmor Euler at t={times[idx]:.6g} execution={costs[idx]:.2f}s step_index={idx}")


def main() -> None:
    tadmor_time_cost()
    for name, case in CASES.items():
        folder = choose_folder(case)
        data = load(case, folder)
        row = oscillation(data)
        thick = rise_cells(data)
        log = jm._log_stats(case)
        gauges = gauge_rows(case)
        g = next(item for item in gauges if abs(item["gauge_m"] - 0.25) < 1e-6)
        print(
            f"{name:12} dp/p={row['neighbor_dp_rms_over_p']:.4f} resid/p={row['residual_rms_over_p']:.4f} "
            f"period={row['autocorr_period_cells']} front={row['front_r_m']:.4f} "
            f"rise_cells={thick.get('rise_cells')} rise_mm={None if thick.get('rise_m') is None else round(thick['rise_m']*1e3, 2)} "
            f"solver={log.get('solver_execution_s')} steps={log.get('n_steps')} "
            f"arr_us={None if g['arrival_s'] is None else round(g['arrival_s']*1e6, 2)} "
            f"peak_kPa={None if g['peak_pa'] is None else round(g['peak_pa']/1e3, 1)} "
            f"I={None if g['impulse_pas'] is None else round(g['impulse_pas'], 2)} {g['duration_note']}"
        )


if __name__ == "__main__":
    main()
