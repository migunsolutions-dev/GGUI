"""Post-shock window metrics on existing volume fields. No new solves."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation" / "jwl_internal"))
sys.path.insert(0, str(ROOT / "validation" / "rarefaction_osc"))

import run_models as jm
from diagnose_raw import load, times

WORK = Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work")
P_ATM = 101325.0
OUT = Path(__file__).resolve().parent / "results"


def shock_outer(p: np.ndarray) -> int:
    for i in range(p.size - 2, 4, -1):
        if p[i] > 2.0 * P_ATM and p[i + 1] < 2.0 * P_ATM:
            return i
    return int(np.argmax(p))


def window_metrics(data: dict, behind: int = 80, skip: int = 6) -> dict:
    outer = shock_outer(data["p"])
    hi = outer - skip
    lo = max(0, hi - behind)
    sl = slice(lo, hi)
    p = data["p"][sl]
    r = data["r"][sl]
    # Remove the local monotone trend with a 17-cell moving average.
    k = 17
    pad = k // 2
    padded = np.pad(p, (pad, pad), mode="edge")
    trend = np.array([np.mean(padded[i : i + k]) for i in range(p.size)])
    resid = p - trend
    dp = np.diff(p)
    turns = 0
    gaps = []
    last = None
    for i in range(1, resid.size - 1):
        swing = min(abs(resid[i] - resid[i - 1]), abs(resid[i] - resid[i + 1]))
        if (resid[i] - resid[i - 1]) * (resid[i + 1] - resid[i]) < 0 and swing > 0.02 * np.median(p):
            turns += 1
            if last is not None:
                gaps.append(i - last)
            last = i
    rho_corr = None
    u_corr = None
    if "rho" in data:
        rr = data["rho"][sl]
        rtrend = np.array([np.mean(np.pad(rr, (pad, pad), mode="edge")[i : i + k]) for i in range(rr.size)])
        rres = rr - rtrend
        denom = np.linalg.norm(resid) * np.linalg.norm(rres)
        rho_corr = float(np.dot(resid, rres) / denom) if denom else 0.0
    if "ur" in data:
        uu = data["ur"][sl]
        utrend = np.array([np.mean(np.pad(uu, (pad, pad), mode="edge")[i : i + k]) for i in range(uu.size)])
        ures = uu - utrend
        denom = np.linalg.norm(resid) * np.linalg.norm(ures)
        u_corr = float(np.dot(resid, ures) / denom) if denom else 0.0
    return {
        "front_r_m": float(data["r"][outer]),
        "window_r": [float(r[0]), float(r[-1])],
        "window_cells": int(p.size),
        "local_median_p_pa": float(np.median(p)),
        "neighbor_dp_rms_pa": float(np.sqrt(np.mean(dp ** 2))),
        "neighbor_dp_rms_over_p": float(np.sqrt(np.mean(dp ** 2)) / max(np.median(p), 1.0)),
        "residual_rms_pa": float(np.sqrt(np.mean(resid ** 2))),
        "residual_peak_to_trough_pa": float(resid.max() - resid.min()),
        "residual_rms_over_p": float(np.sqrt(np.mean(resid ** 2)) / max(np.median(p), 1.0)),
        "extrema": turns,
        "median_cells_between_extrema": None if not gaps else float(np.median(gaps)),
        "p_rho_residual_corr": rho_corr,
        "p_ur_residual_corr": u_corr,
    }


def main() -> None:
    cases = {
        "JWL_Tadmor": WORK / "JwlFlux_Tadmor",
        "JWL_Kurganov": WORK / "JwlFlux_Kurganov",
        "JWL_HLL": WORK / "JwlFlux_HLL",
        "JWL_HLLC": WORK / "JwlFlux_HLLC",
        "JWL_AUSM+": WORK / "JwlFlux_AUSMp",
        "JWL_AUSM+up": WORK / "JwlFlux_AUSMpup",
        "IG_1kg": WORK / "ViperCmp_ig_1kg_r1_20261003_132248",
    }
    rows = []
    for name, case in cases.items():
        available = times(case)
        chosen = None
        for value, folder in available:
            if abs(value - 0.00025) < 2e-5:
                chosen = (value, folder)
                break
        if chosen is None:
            chosen = available[-1]
        data = load(case, chosen[1])
        row = window_metrics(data)
        row["case"] = name
        row["time_s"] = chosen[0]
        rows.append(row)
        if name in ("JWL_Tadmor", "IG_1kg"):
            outer = shock_outer(data["p"])
            print("\nSLICE", name, "front", data["r"][outer])
            print(f"{'i':>5} {'r':>8} {'p_kPa':>10} {'rho':>8} {'ur':>8}")
            for i in range(outer - 24, outer + 4):
                rho = data["rho"][i] if "rho" in data else float("nan")
                ur = data["ur"][i] if "ur" in data else float("nan")
                print(f"{i:5d} {data['r'][i]:8.4f} {data['p'][i]/1e3:10.2f} {rho:8.3f} {ur:8.1f}")
    print("\nMETRICS")
    print(json.dumps(rows, indent=2))
    (OUT / "window_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
