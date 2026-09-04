"""Comparison plots. Every X coordinate is the actual physical location."""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np

from viper_compare.extract import as_overpressure


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    fig.clf()


def plot_pt_overlays(
    series: Sequence[dict],
    out_dir: str,
    p_atm: float,
) -> List[str]:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written = []
    by_r: Dict[float, list] = {}
    for item in series:
        by_r.setdefault(round(float(item["R_m"]), 6), []).append(item)
    for r, items in sorted(by_r.items()):
        if len(items) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for item in items:
            over = as_overpressure(np.asarray(item["p"]), p_atm)
            ax.plot(item["t"], over / 1000.0, label=item["label"], lw=1.2)
        ax.set_xlabel("Physical time [s]")
        ax.set_ylabel("Overpressure [kPa]")
        ax.set_title(f"P(t) at R = {r:.2f} m (no time shift)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f"pt_R{r:.2f}.png")
        _save(fig, path)
        plt.close(fig)
        written.append(path)
    return written


def plot_peak_vs_r(rows: Sequence[dict], out_path: str, *, x_key: str, xlabel: str) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    groups: Dict[str, list] = {}
    for row in rows:
        key = f"{row['solver']} {row['configuration']}"
        groups.setdefault(key, []).append(row)
    for key, items in groups.items():
        xs = [i[x_key] for i in items if i.get(x_key) is not None]
        ys = [i["peak_pressure_pa"] / 1000.0 for i in items if i.get(x_key) is not None]
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], "o-", label=key, ms=4)
    kb = [(r["R_m"] if x_key == "R_m" else r["Z"], r["kb_peak_pressure_pa"]) for r in rows]
    kb = [(x, y) for x, y in kb if x is not None and y is not None]
    if kb:
        uniq = {}
        for x, y in kb:
            uniq[round(float(x), 6)] = y
        xs = sorted(uniq)
        ax.plot(xs, [uniq[x] / 1000.0 for x in xs], "k--", label="KB free-air spherical")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak overpressure [kPa]")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _save(fig, out_path)
    plt.close(fig)
    return out_path


def plot_impulse_vs_r(rows: Sequence[dict], out_path: str) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    groups: Dict[str, list] = {}
    for row in rows:
        key = f"{row['solver']} {row['configuration']}"
        groups.setdefault(key, []).append(row)
    for key, items in groups.items():
        xs = [i["R_m"] for i in items]
        ys = [i["derived_impulse_pa_s"] for i in items]
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], "o-", label=key + " derived", ms=4)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Derived positive impulse [Pa s]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _save(fig, out_path)
    plt.close(fig)
    return out_path


def plot_error_vs_r(errors: Sequence[dict], out_path: str) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups: Dict[str, list] = {}
    for row in errors:
        groups.setdefault(row.get("pair", "GGUI vs VIPER"), []).append(row)
    for key, items in groups.items():
        xs = [i["R_m"] for i in items]
        ys = [i["peak_pressure_error_pct"] for i in items]
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], "o-", label=key, ms=4)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Peak pressure error (GGUI-VIPER)/VIPER [%]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _save(fig, out_path)
    plt.close(fig)
    return out_path
