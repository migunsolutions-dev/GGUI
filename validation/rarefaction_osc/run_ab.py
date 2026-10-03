"""Grid study and one-change tests for the 1D rarefaction oscillation.

Does not edit production generators. Each case is a generated copy with one patch.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "viper_1d"))
sys.path.insert(0, str(ROOT / "validation" / "jwl_internal"))

from models import SOURCE_MODEL_IG, SOURCE_MODEL_JWL, CaseInputs1D
from viper_compare.ggui_run import generate_case, run_allrun_with_optional_watchdog
import run_models as jm

OUT = Path(__file__).resolve().parent / "results"
WORK = Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work")
BASE_JWL = WORK / "JwlActBase_20261003_143646"
P_ATM = 101325.0
TNT = {
    "rho": 1630.0,
    "A": 371.2e9,
    "B": 3.23e9,
    "R1": 4.15,
    "R2": 0.95,
    "omega": 0.30,
    "E0": 4.29e6,
}
END = 0.00025
GAUGES = (0.25, 0.50, 0.75)


def _inputs(dx: float, source: str) -> CaseInputs1D:
    return CaseInputs1D(
        radius=1.0,
        cell_size=dx,
        p_atm=P_ATM,
        t_atm=288.0,
        mass_kg=1.0,
        rho_charge=TNT["rho"],
        energy_j_per_kg=TNT["E0"],
        material_props=dict(TNT),
        max_cfl=0.5,
        end_time_s=END,
        write_interval_s=END,
        n_probes=40,
        probe_write_interval_steps=1,
        gauge_locations=tuple((r, f"G{r:.2f}") for r in GAUGES),
        material_name="TNT",
        source_model=source,
        stop_mode="terminate",
        stop_radius_m=1.0,
        remap_for_2d=False,
    )


def _patch_text(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{old!r} not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def _keep_times(case: Path) -> None:
    _patch_text(case / "system" / "controlDict", "purgeWrite      1;", "purgeWrite      0;")


def _patch_limiter(case: Path, limiter: str) -> None:
    if limiter == "vanLeer":
        return
    _patch_text(case / "system" / "fvSchemes", "vanLeer", limiter)


def _patch_time(case: Path, integrator: str) -> None:
    if integrator == "Euler":
        return
    _patch_text(
        case / "system" / "fvSchemes",
        "timeIntegrator Euler",
        f"timeIntegrator {integrator}",
    )


def _patch_cfl(case: Path, cfl: float) -> None:
    if abs(cfl - 0.5) < 1e-12:
        return
    _patch_text(case / "system" / "controlDict", "maxCo           0.5;", f"maxCo           {cfl:.3g};")


def _run(case: Path, prefix: str) -> dict:
    print(f"RUN {prefix} {case}", flush=True)
    t0 = time.perf_counter()
    ran = run_allrun_with_optional_watchdog(
        case_dir=str(case),
        log_dir=str(OUT),
        prefix=prefix,
        watchdog=False,
    )
    ran["wall_s"] = time.perf_counter() - t0
    try:
        ran["log"] = jm._log_stats(case)
    except Exception as exc:
        ran["log_error"] = str(exc)
    print(
        f"CASE_DONE {prefix} rc={ran.get('returncode')} wall={ran['wall_s']:.1f}s "
        f"solver={ran.get('log', {}).get('solver_execution_s')} fatal={ran.get('log', {}).get('fatal')}",
        flush=True,
    )
    return ran


def _ig(name: str, dx: float, limiter: str, integrator: str, cfl: float) -> Path:
    made = generate_case(prefix=name, inputs=_inputs(dx, SOURCE_MODEL_IG))
    case = Path(made["case_dir"])
    _keep_times(case)
    _patch_limiter(case, limiter)
    _patch_time(case, integrator)
    _patch_cfl(case, cfl)
    _run(case, name)
    return case


def _jwl(name: str, limiter: str) -> Path:
    dest = WORK / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(BASE_JWL, dest)
    control = (dest / "system" / "controlDict").read_text(encoding="utf-8")
    control = control.replace("endTime         0.005;", f"endTime         {END:.8g};")
    control = control.replace("writeInterval   0.005;", f"writeInterval   {END:.8g};")
    control = control.replace("purgeWrite      1;", "purgeWrite      0;")
    (dest / "system" / "controlDict").write_text(control, encoding="utf-8", newline="\n")
    _patch_limiter(dest, limiter)
    _run(dest, name)
    return dest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {}
    cases["IG_dx2mm"] = str(_ig("OscIG_dx2mm", 0.002, "vanLeer", "Euler", 0.5))
    cases["IG_dx1mm"] = str(_ig("OscIG_dx1mm", 0.001, "vanLeer", "Euler", 0.5))
    cases["IG_dx0p5mm"] = str(_ig("OscIG_dx0p5mm", 0.0005, "vanLeer", "Euler", 0.5))
    cases["IG_upwind"] = str(_ig("OscIG_upwind", 0.001, "upwind", "Euler", 0.5))
    cases["IG_minmod"] = str(_ig("OscIG_minmod", 0.001, "Minmod", "Euler", 0.5))
    cases["IG_superbee"] = str(_ig("OscIG_superbee", 0.001, "SuperBee", "Euler", 0.5))
    cases["IG_rk2"] = str(_ig("OscIG_rk2", 0.001, "vanLeer", "RK2SSP", 0.5))
    cases["IG_cfl01"] = str(_ig("OscIG_cfl01", 0.001, "vanLeer", "Euler", 0.1))
    cases["JWL_upwind"] = str(_jwl("OscJWL_upwind", "upwind"))
    cases["JWL_minmod"] = str(_jwl("OscJWL_minmod", "Minmod"))
    cases["JWL_tadmor_existing"] = str(WORK / "JwlFlux_Tadmor")
    (OUT / "case_paths.txt").write_text(
        "\n".join(f"{key}\t{value}" for key, value in cases.items()) + "\n",
        encoding="utf-8",
    )
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
