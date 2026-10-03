"""Metrics and overlay plots for the rarefaction oscillation study. No new solves."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "viper_1d"))
sys.path.insert(0, str(ROOT / "validation" / "jwl_internal"))
sys.path.insert(0, str(ROOT / "validation" / "rarefaction_osc"))

from viper_compare.extract import as_overpressure
import run_compare as rc
import run_models as jm
from diagnose_raw import load, times

OUT = Path(__file__).resolve().parent / "results"
P_ATM = 101325.0
TARGET = 0.00025
GAUGES = (0.25, 0.50, 0.75)


def shock_outer(p: np.ndarray) -> int:
    for i in range(p.size - 2, 4, -1):
        if p[i] > 2.0 * P_ATM and p[i + 1] < 2.0 * P_ATM:
            return i
    return int(np.argmax(p))


def _poly_residual(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    if r.size < 8:
        return p - np.median(p)
    degree = 3 if r.size >= 12 else 2
    coeff = np.polyfit(r, p, degree)
    return p - np.polyval(coeff, r)


def _period_cells(resid: np.ndarray) -> float | None:
    if resid.size < 16:
        return None
    signal = resid - np.mean(resid)
    corr = np.correlate(signal, signal, mode="full")
    corr = corr[corr.size // 2 :]
    if corr[0] <= 0:
        return None
    neg = np.where(corr[1:] < 0)[0]
    if neg.size == 0:
        return None
    start = int(neg[0]) + 1
    for lag in range(max(2, start), min(corr.size - 1, 80)):
        if corr[lag] > corr[lag - 1] and corr[lag] >= corr[lag + 1] and corr[lag] > 0.15 * corr[0]:
            return float(lag)
    return None


def oscillation(data: dict) -> dict:
    p = data["p"]
    r = data["r"]
    outer = shock_outer(p)
    hi = max(6, outer - 4)
    lo = hi
    while lo > 0 and (r[hi] - r[lo]) < 0.08:
        lo -= 1
    seg_p = p[lo:hi]
    seg_r = r[lo:hi]
    dp = np.diff(seg_p)
    resid = _poly_residual(seg_r, seg_p)
    turns = 0
    gaps = []
    last = None
    floor = 0.03 * max(float(np.median(seg_p)), 1.0)
    for i in range(1, resid.size - 1):
        swing = min(abs(resid[i] - resid[i - 1]), abs(resid[i] - resid[i + 1]))
        turned = (resid[i] - resid[i - 1]) * (resid[i + 1] - resid[i]) < 0 and swing >= floor
        if turned:
            turns += 1
            if last is not None:
                gaps.append(i - last)
            last = i
    width = None
    if 0 < outer < p.size - 1:
        width = float(r[min(p.size - 1, outer + 1)] - r[max(0, outer - 1)])
    row = {
        "front_r_m": float(r[outer]),
        "front_index": int(outer),
        "shock_span_m": width,
        "shock_span_cells": None if width is None else float(width / max(np.median(np.diff(r)), 1e-12)),
        "postshock_p_pa": float(np.median(p[max(0, outer - 8) : max(1, outer - 3)])),
        "profile_peak_pa": float(np.max(p)),
        "window_cells": int(seg_p.size),
        "window_m": float(seg_r[-1] - seg_r[0]) if seg_r.size else 0.0,
        "neighbor_dp_rms_pa": float(np.sqrt(np.mean(dp ** 2))) if dp.size else None,
        "neighbor_dp_rms_over_p": float(np.sqrt(np.mean(dp ** 2)) / max(np.median(seg_p), 1.0)) if dp.size else None,
        "residual_rms_pa": float(np.sqrt(np.mean(resid ** 2))),
        "residual_peak_to_trough_pa": float(np.max(resid) - np.min(resid)),
        "residual_rms_over_p": float(np.sqrt(np.mean(resid ** 2)) / max(np.median(seg_p), 1.0)),
        "extrema": turns,
        "median_cells_between_extrema": None if not gaps else float(np.median(gaps)),
        "autocorr_period_cells": _period_cells(resid),
        "median_dr_m": float(np.median(np.diff(r))),
        "n_cells": int(p.size),
        "r_min_m": float(r[0]),
    }
    if row["autocorr_period_cells"] is not None:
        row["autocorr_period_m"] = row["autocorr_period_cells"] * row["median_dr_m"]
    if "rho" in data:
        rho = data["rho"][lo:hi]
        rres = _poly_residual(seg_r, rho)
        denom = np.linalg.norm(resid) * np.linalg.norm(rres)
        row["p_rho_corr"] = float(np.dot(resid, rres) / denom) if denom else 0.0
    if "ur" in data:
        ur = data["ur"][lo:hi]
        ures = _poly_residual(seg_r, ur)
        denom = np.linalg.norm(resid) * np.linalg.norm(ures)
        row["p_ur_corr"] = float(np.dot(resid, ures) / denom) if denom else 0.0
    return row


def gauge_rows(case: Path) -> list[dict]:
    radii, series_t, values = rc._read_ggui_gauge(str(case), "p")
    rows = []
    if values.size == 0:
        return rows
    for radius in GAUGES:
        column = int(np.argmin(np.abs(np.asarray(radii) - radius)))
        metrics = rc._series_metrics(series_t, values[:, column], P_ATM)
        metrics["gauge_m"] = radius
        metrics["actual_r_m"] = float(radii[column])
        rows.append(metrics)
    return rows


def conservation(case: Path) -> dict:
    available = times(case)
    if len(available) < 1:
        return {}
    earliest = available[0][1]
    latest = available[-1][1]

    def budget(folder: Path) -> dict:
        rho = jm._foam_scalar(folder / "rho")
        energy = jm._foam_scalar(folder / "rhoE") if (folder / "rhoE").is_file() else None
        radius = jm._cell_radii(case, rho.size)
        dvol = jm._shell_volumes(radius, 1.0)
        out = {"mass_kg": float(np.sum(rho * dvol))}
        if energy is not None and energy.size == rho.size:
            out["energy_j"] = float(np.sum(energy * dvol))
        return out

    initial = budget(earliest)
    final = budget(latest)
    row = {
        "t0": available[0][0],
        "t1": available[-1][0],
        "mass0_kg": initial["mass_kg"],
        "mass1_kg": final["mass_kg"],
        "mass_change": (final["mass_kg"] - initial["mass_kg"]) / max(abs(initial["mass_kg"]), 1e-12),
    }
    if "energy_j" in initial and "energy_j" in final:
        row["energy0_j"] = initial["energy_j"]
        row["energy1_j"] = final["energy_j"]
        row["energy_change"] = (final["energy_j"] - initial["energy_j"]) / max(abs(initial["energy_j"]), 1.0)
    return row


def choose_folder(case: Path) -> Path | None:
    available = times(case)
    best = None
    best_err = None
    for value, folder in available:
        if not (folder / "p").is_file():
            continue
        err = abs(value - TARGET)
        if best_err is None or err < best_err:
            best, best_err = folder, err
    return best


def main() -> None:
    mapping = {}
    for line in (OUT / "case_paths.txt").read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        mapping[key] = Path(value)
    rows = []
    profiles = {}
    for key, case in mapping.items():
        folder = choose_folder(case)
        row = {"case": key, "path": str(case)}
        if folder is None:
            row["missing"] = True
            rows.append(row)
            continue
        data = load(case, folder)
        row.update(oscillation(data))
        row["time_s"] = float(folder.name)
        try:
            row["gauges"] = gauge_rows(case)
        except Exception as exc:
            row["gauge_error"] = str(exc)
        try:
            row["conservation"] = conservation(case)
        except Exception as exc:
            row["conservation_error"] = str(exc)
        try:
            row["log"] = jm._log_stats(case)
        except Exception as exc:
            row["log_error"] = str(exc)
        rows.append(row)
        profiles[key] = data
        print(
            f"{key:22} front={row['front_r_m']:.4f} "
            f"dp/p={row['neighbor_dp_rms_over_p']:.4f} "
            f"resid/p={row['residual_rms_over_p']:.4f} "
            f"period={row.get('autocorr_period_cells')} "
            f"span_cells={row['shock_span_cells']:.2f}"
        )
    (OUT / "ab_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    groups = (
        ("Ideal gas, grid", [k for k in profiles if k.startswith("IG_dx")]),
        ("Ideal gas, one change", [k for k in profiles if k.startswith("IG_") and not k.startswith("IG_dx")]),
        ("JWL", [k for k in profiles if k.startswith("JWL_")]),
    )
    # Pressure overlays use the same axes per family, split across the figure.
    ax_map = {
        "Ideal gas, grid": axes[0, 0],
        "Ideal gas, one change": axes[0, 1],
        "JWL": axes[1, 0],
    }
    zoom = axes[1, 1]
    for title, keys in groups:
        ax = ax_map[title]
        for key in keys:
            data = profiles[key]
            ax.plot(data["r"], data["p"] / 1e6, lw=1.0, label=key)
            outer = shock_outer(data["p"])
            lo = max(0, outer - 40)
            zoom.plot(data["r"][lo : outer + 3], data["p"][lo : outer + 3] / 1e6, lw=1.0, label=key)
        ax.set_title(title)
        ax.set_ylabel("p (MPa)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    zoom.set_title("40 cells behind each shock")
    zoom.set_ylabel("p (MPa)")
    zoom.set_xlabel("r (m)")
    zoom.grid(True, alpha=0.3)
    zoom.legend(fontsize=6, frameon=False, ncol=2)
    axes[1, 0].set_xlabel("r (m)")
    fig.tight_layout()
    fig.savefig(OUT / "ab_profiles.png", dpi=140)
    print("wrote", OUT / "ab_metrics.json")


if __name__ == "__main__":
    main()
