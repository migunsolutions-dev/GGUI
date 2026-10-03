"""Confirm whether rarefaction pressure jumps are in the raw cell fields.

Reads finished cases only. Does not run the solver and does not change production.
"""
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
sys.path.insert(0, str(ROOT / "validation" / "jwl_internal"))

import run_models as jm

OUT = Path(__file__).resolve().parent / "results"
WORK = Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work")
P_ATM = 101325.0


def times(case: Path) -> list[tuple[float, Path]]:
    found = []
    for child in case.iterdir():
        if not child.is_dir():
            continue
        try:
            found.append((float(child.name), child))
        except ValueError:
            continue
    return sorted(found)


def load(case: Path, folder: Path) -> dict:
    p = jm._foam_scalar(folder / "p")
    rho = jm._foam_scalar(folder / "rho") if (folder / "rho").is_file() else None
    energy = None
    for name in ("e", "T", "rhoE"):
        if (folder / name).is_file():
            energy = (name, jm._foam_scalar(folder / name))
            break
    ufile = folder / "U"
    vel = None
    if ufile.is_file():
        text = ufile.read_text(encoding="utf-8", errors="replace")
        body = text[text.index("internalField") :]
        header, _, rest = body.partition("(")
        end = rest.rfind(")")
        nums = []
        for token in rest[:end].replace("(", " ").replace(")", " ").split():
            try:
                nums.append(float(token))
            except ValueError:
                continue
        if "nonuniform" not in header:
            vel = np.asarray(nums[:3], dtype=float).reshape(1, 3)
        else:
            vel = np.asarray(nums, dtype=float).reshape(-1, 3)
    radius = jm._cell_radii(case, p.size)
    order = np.argsort(radius)
    r = radius[order]
    out = {
        "r": r,
        "p": p[order],
        "dr": np.diff(r),
        "monotonic": bool(np.all(np.diff(radius[order]) > 0)),
        "n": int(p.size),
        "duplicate_r": int(np.sum(np.diff(r) <= 1e-12)),
    }
    if rho is not None and rho.size == p.size:
        out["rho"] = rho[order]
    if energy is not None and energy[1].size == p.size:
        out[energy[0]] = energy[1][order]
    if vel is not None and vel.shape[0] == p.size:
        ur = np.sum(vel[order] * r[:, None], axis=1) / np.maximum(r, 1e-12)
        out["ur"] = ur
    return out


def rarefaction_mask(data: dict) -> np.ndarray:
    p = data["p"]
    r = data["r"]
    outer = None
    for i in range(p.size - 2, 1, -1):
        if p[i] > 2.0 * P_ATM and p[i + 1] < 2.0 * P_ATM:
            outer = i
            break
    if outer is None:
        outer = int(np.argmax(p))
    # Behind the front, outside the near-origin cavity.
    lo = max(8, int(np.searchsorted(r, 0.05)))
    hi = max(lo + 5, outer - 4)
    return np.arange(lo, hi), outer


def extrema(y: np.ndarray, threshold: float) -> int:
    count = 0
    for i in range(1, y.size - 1):
        swing = min(abs(y[i] - y[i - 1]), abs(y[i] - y[i + 1]))
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0 and swing >= threshold:
            count += 1
    return count


def period_cells(dp: np.ndarray) -> float | None:
    if dp.size < 16:
        return None
    signal = dp - np.mean(dp)
    corr = np.correlate(signal, signal, mode="full")
    corr = corr[corr.size // 2 :]
    if corr[0] <= 0:
        return None
    # First lag where correlation returns positive after a negative dip.
    neg = np.where(corr[1:] < 0)[0]
    if neg.size == 0:
        return None
    start = int(neg[0]) + 1
    for lag in range(start + 1, min(corr.size - 1, 40)):
        if corr[lag] > corr[lag - 1] and corr[lag] > corr[lag + 1] and corr[lag] > 0.2 * corr[0]:
            return float(lag)
    return None


def summarise(name: str, data: dict) -> dict:
    idx, outer = rarefaction_mask(data)
    p = data["p"]
    seg = p[idx]
    dp = np.diff(seg)
    # Local envelope: 11-cell median, so a 2-cell wiggle is the residual.
    kernel = 11
    pad = kernel // 2
    padded = np.pad(seg, (pad, pad), mode="edge")
    env = np.array([np.median(padded[i : i + kernel]) for i in range(seg.size)])
    resid = seg - env
    span = float(data["r"][idx[-1]] - data["r"][idx[0]]) if idx.size else 0.0
    n_ext = extrema(seg, threshold=0.02 * max(float(np.median(np.abs(dp))), 1.0))
    row = {
        "case": name,
        "n_cells": data["n"],
        "r_min": float(data["r"][0]),
        "r_max": float(data["r"][-1]),
        "monotonic_radius": data["monotonic"],
        "duplicate_radii": data["duplicate_r"],
        "median_dr_m": float(np.median(data["dr"])),
        "dr_cv": float(np.std(data["dr"]) / max(np.median(data["dr"]), 1e-12)),
        "front_index": int(outer),
        "front_r_m": float(data["r"][outer]),
        "rarefaction_cells": int(idx.size),
        "rarefaction_span_m": span,
        "median_abs_neighbor_dp_pa": float(np.median(np.abs(dp))),
        "p95_abs_neighbor_dp_pa": float(np.percentile(np.abs(dp), 95)),
        "max_abs_neighbor_dp_pa": float(np.max(np.abs(dp))),
        "residual_peak_to_trough_pa": float(np.max(resid) - np.min(resid)),
        "residual_rms_pa": float(np.sqrt(np.mean(resid ** 2))),
        "residual_rms_over_local_p": float(
            np.sqrt(np.mean(resid ** 2)) / max(float(np.median(np.abs(seg))), 1.0)
        ),
        "extrema_in_rarefaction": n_ext,
        "cells_per_extremum": float(idx.size / max(n_ext, 1)),
        "autocorr_period_cells": period_cells(dp),
        "sign_change_fraction": float(np.mean(dp[1:] * dp[:-1] < 0)),
    }
    if "rho" in data:
        d_rho = np.diff(data["rho"][idx])
        row["median_abs_neighbor_drho"] = float(np.median(np.abs(d_rho)))
        row["rho_sign_change_fraction"] = float(np.mean(d_rho[1:] * d_rho[:-1] < 0))
    if "ur" in data:
        d_u = np.diff(data["ur"][idx])
        row["median_abs_neighbor_du"] = float(np.median(np.abs(d_u)))
        row["ur_sign_change_fraction"] = float(np.mean(d_u[1:] * d_u[:-1] < 0))
    # Odd-even: correlation of residual with a (+1,-1) pattern.
    pattern = np.resize([1.0, -1.0], resid.size)
    pattern -= np.mean(pattern)
    denom = np.linalg.norm(resid - np.mean(resid)) * np.linalg.norm(pattern)
    row["odd_even_correlation"] = float(
        np.dot(resid - np.mean(resid), pattern) / denom
    ) if denom else 0.0
    return row, idx, resid


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = {
        "JWL_Tadmor": WORK / "JwlFlux_Tadmor",
        "IG_1kg": WORK / "ViperCmp_ig_1kg_r1_20261003_132248",
    }
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for name, case in targets.items():
        if not case.is_dir():
            rows.append({"case": name, "missing": str(case)})
            continue
        available = times(case)
        # Prefer a mid-propagation dump near 0.25 ms; otherwise the latest solved time.
        chosen = None
        for value, folder in available:
            if 0.0002 <= value <= 0.0003:
                chosen = (value, folder)
                break
        if chosen is None and available:
            chosen = available[-1]
        if chosen is None or not (chosen[1] / "p").is_file():
            rows.append({"case": name, "times": [v for v, _ in available], "no_pressure": True})
            continue
        data = load(case, chosen[1])
        row, idx, resid = summarise(name, data)
        row["time_s"] = chosen[0]
        row["time_name"] = chosen[1].name
        rows.append(row)
        r = data["r"]
        axes[0, 0].plot(r, data["p"] / 1e6, lw=1.0, label=name)
        axes[0, 1].plot(r[idx], data["p"][idx] / 1e6, lw=1.0, label=name)
        axes[1, 0].plot(r[idx][:-1], np.diff(data["p"][idx]) / 1e6, lw=0.8, label=name)
        axes[1, 1].plot(r[idx], resid / 1e6, lw=0.8, label=name)
        # Probe comparison when a probes1d file exists.
        probe_dirs = list((case / "postProcessing" / "probes1d").glob("*/p")) if (case / "postProcessing").is_dir() else []
        if probe_dirs:
            text = probe_dirs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
            coords = []
            for line in text:
                if line.startswith("# Probe"):
                    inside = line[line.index("(") + 1 : line.index(")")]
                    xyz = [float(v) for v in inside.split()]
                    coords.append((xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2) ** 0.5)
            samples = [ln for ln in text if ln and not ln.startswith("#")]
            if samples and coords:
                vals = [float(v) for v in samples[-1].split()[1:]]
                n = min(len(coords), len(vals))
                order = np.argsort(coords[:n])
                pr = np.asarray(coords[:n], dtype=float)[order]
                pp = np.asarray(vals[:n], dtype=float)[order]
                # Nearest cell for each probe.
                nearest = np.abs(r[:, None] - pr[None, :]).argmin(axis=0)
                cell_p = data["p"][nearest]
                row["probe_count"] = int(n)
                row["probe_vs_cell_max_abs_pa"] = float(np.max(np.abs(pp - cell_p)))
                row["probe_vs_cell_median_abs_pa"] = float(np.median(np.abs(pp - cell_p)))
                row["probe_radii_monotonic"] = bool(np.all(np.diff(pr) > 0))
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].set_title("Raw cell pressure")
    axes[0, 0].set_ylabel("p (MPa)")
    axes[0, 1].set_title("Rarefaction zoom")
    axes[1, 0].set_title("Neighbor pressure jump")
    axes[1, 0].set_ylabel("dp (MPa)")
    axes[1, 0].set_xlabel("r (m)")
    axes[1, 1].set_title("Residual after 11-cell median")
    axes[1, 1].set_xlabel("r (m)")
    fig.tight_layout()
    fig.savefig(OUT / "raw_fields.png", dpi=140)
    (OUT / "raw_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
