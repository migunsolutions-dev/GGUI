"""Zoomed overlays without the failed SuperBee scale."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation" / "rarefaction_osc"))
from diagnose_raw import load
from post_ab import choose_folder, shock_outer

OUT = Path(__file__).resolve().parent / "results"
CASES = {
    "IG 2 mm": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscIG_dx2mm_20261003_152625"),
    "IG 1 mm": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscIG_dx1mm_20261003_152640"),
    "IG 0.5 mm": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscIG_dx0p5mm_20261003_152645"),
    "IG upwind": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscIG_upwind_20261003_152713"),
    "IG Minmod": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscIG_minmod_20261003_152716"),
    "IG RK2": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscIG_rk2_20261003_152729"),
    "IG CFL 0.1": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscIG_cfl01_20261003_152739"),
    "IG CFL 0.25": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscIG_cfl025_20261003_152933"),
    "IG CFL 0.30": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscIG_cfl030_20261003_152957"),
    "JWL Tadmor/Euler": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\JwlFlux_Tadmor"),
    "JWL RK2": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscJWL_rk2"),
    "JWL upwind": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscJWL_upwind"),
    "JWL Minmod": Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\OscJWL_minmod"),
}


def series(case: Path):
    folder = choose_folder(case)
    data = load(case, folder)
    return data["r"], data["p"] / 1e6


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ig = ["IG 2 mm", "IG 1 mm", "IG 0.5 mm", "IG upwind", "IG Minmod", "IG RK2", "IG CFL 0.1", "IG CFL 0.25", "IG CFL 0.30"]
    jwl = ["JWL Tadmor/Euler", "JWL RK2", "JWL upwind", "JWL Minmod"]
    for label in ig:
        r, p = series(CASES[label])
        axes[0].plot(r, p, lw=1.15, label=label)
    for label in jwl:
        r, p = series(CASES[label])
        axes[1].plot(r, p, lw=1.15, label=label)
    axes[0].set_xlim(0.35, 0.75)
    axes[0].set_ylim(0.0, 4.2)
    axes[1].set_xlim(0.50, 0.90)
    axes[1].set_ylim(0.0, 4.2)
    for ax, title in zip(axes, ("Ideal gas at 0.25 ms", "JWL at 0.25 ms")):
        ax.set_title(title)
        ax.set_xlabel("Radius (m)")
        ax.set_ylabel("Pressure (MPa)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "zoom_profiles.png", dpi=140)
    print("wrote zoom")


if __name__ == "__main__":
    main()
