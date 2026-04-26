"""
Harness: load each relevant blastFoam tutorial via case_loader, regenerate
through the GUI's Generator3D into /home/naor/.../run/Work/<name>_GUI/, and
record diffs against the original tutorial.

DOES NOT modify anything in run/blastfoam/tutorials/.

Usage (Windows / WSL):
    python _tutorials_harness.py
"""
from __future__ import annotations
import os
import sys
import json
import shutil
import subprocess
from typing import Any, Dict, List, Optional

# Ensure GUI modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case_loader import load_case
from models import CaseInputs3D, ObstacleData
from generator_3d import Generator3D


# Network UNC paths from Windows to the WSL filesystem
WSL_PREFIX_WIN = r"\\wsl.localhost\Ubuntu-20.04"
TUTS_LIN = "/home/naor/OpenFOAM/naor-9/run/blastfoam/tutorials/blastFoam"
WORK_LIN = "/home/naor/OpenFOAM/naor-9/run/Work"

TUTS_WIN = WSL_PREFIX_WIN + TUTS_LIN.replace("/", "\\")
WORK_WIN = WSL_PREFIX_WIN + WORK_LIN.replace("/", "\\")

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tutorials_report")
os.makedirs(REPORT_DIR, exist_ok=True)


def _safe_get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def build_inputs_from_loaded(loaded: Dict[str, Any], original_case_dir_win: str) -> CaseInputs3D:
    """Convert loaded dict (case_loader output) into a CaseInputs3D.

    For fields that case_loader doesn't return (because they're not in the
    tutorial file), pick GUI defaults that won't trigger generator validation.
    """
    obstacles: List[ObstacleData] = []
    for s in loaded.get("stl_obstacles", []) or []:
        # case_loader stores the resolved Windows-side STL path
        if s.get("exists"):
            obstacles.append(
                ObstacleData(
                    stl_path=s["path"],
                    name=s["name"],
                    scale=s.get("scale", 1.0),
                    refinement_level=int(loaded.get("obstacle_refine_max", 1) or 1),
                )
            )

    charge_shape = loaded.get("charge_shape") or "Sphere"
    charge_radius = float(loaded.get("charge_radius") or 0.1)
    cylinder_axis = loaded.get("cylinder_axis") or "Z"

    cell_size = float(loaded.get("cell_size") or 0.5)
    min_p = tuple(loaded.get("min_point") or (-5.0, -5.0, 0.0))
    max_p = tuple(loaded.get("max_point") or (5.0, 5.0, 5.0))
    charge_center = tuple(loaded.get("charge_center") or (0.0, 0.0, 0.5))

    end_t = float(loaded.get("end_time_s") or 0.0025)
    delta_t = float(loaded.get("delta_t") or 1e-7)

    write_ctrl = loaded.get("write_control_type") or "adjustableRunTime"
    write_int_t = float(loaded.get("write_interval_time") or 5e-5)
    write_int_steps = int(loaded.get("write_interval_steps") or 100)

    # AMR (loaded keys come from dynamicMeshDict)
    refine_max = int(loaded.get("dyn_refine_max") or loaded.get("refine_max") or 2)
    refine_min = int(loaded.get("dyn_refine_min") or loaded.get("refine_min") or 1)

    # setRefinedFields charge level (level X in setFieldsDict regions)
    charge_lvl = int(loaded.get("charge_refinement_level") or 3)
    backup_factor = float(loaded.get("charge_backup_radius_factor") or 1.5)

    inputs = CaseInputs3D(
        # Geometry & Grid
        min_point=min_p,
        max_point=max_p,
        cell_size=cell_size,
        # Charge
        charge_center=charge_center,
        charge_shape=charge_shape,
        mass_kg=float(loaded.get("mass_kg") or 25.0),
        cylinder_radius=charge_radius,
        cylinder_axis=cylinder_axis,
        # Material
        material_name=loaded.get("material_name") or "C4",
        rho_charge=float(loaded.get("rho_charge") or 1601.0),
        energy_j_per_kg=float(loaded.get("energy_j_per_kg") or 4.5e6),
        # Physics
        p_atm=float(loaded.get("p_atm") or 101298.0),
        t_atm=float(loaded.get("t_atm") or 300.0),
        # Simulation control
        end_time_s=end_t,
        delta_t=delta_t,
        write_interval_steps=write_int_steps,
        cores=4,
        cfl_value=float(loaded.get("cfl_value") or 0.5),
        write_control_type=write_ctrl,
        write_interval_time=write_int_t,
        cycle_write=int(loaded.get("cycle_write") or 0),
        # Obstacles & boundaries
        obstacles=obstacles,
        boundaries=loaded.get("boundaries", {}),
        # Mesh refinement
        enable_local_refinement=True,
        enable_dyn_refine=True,
        refine_min=refine_min,
        refine_max=refine_max,
        dyn_refine_min=refine_min,
        dyn_refine_max=refine_max,
        enable_obstacle_refine=bool(obstacles),
        obstacle_refine_min=int(loaded.get("obstacle_refine_min") or 1),
        obstacle_refine_max=int(loaded.get("obstacle_refine_max") or 2),
        charge_refinement_level=charge_lvl,
        charge_backup_radius_factor=backup_factor,
        buffer_layers=int(loaded.get("buffer_layers") or 2),
        # AMR advanced
        refine_interval=int(loaded.get("refine_interval") or 3),
        lower_refine_threshold=float(loaded.get("lower_refine_threshold") or 0.1),
        unrefine_threshold=float(loaded.get("unrefine_threshold") or 0.1),
        n_buffer_layers_dynamic=int(loaded.get("n_buffer_layers_dynamic") or 2),
        enable_balancing=bool(loaded.get("enable_balancing") or False),
        # Run mode (default fast)
        enable_post_processing=bool(loaded.get("enable_post_processing") or False),
        fast_run_mode=True,
        # Charge geometry
        charge_aspect=float(loaded.get("charge_lbyd") or 2.5),
        charge_length=float(loaded.get("charge_length") or 0.0),
    )
    return inputs


def win_to_lin(p: str) -> str:
    r"""Convert \\wsl.localhost\Ubuntu-20.04\<lin path> back to /<lin path>."""
    p = p.strip().replace("\\", "/")
    pref = WSL_PREFIX_WIN.replace("\\", "/")
    if p.startswith(pref):
        return p[len(pref):]
    return p


def diff_via_wsl(orig_lin: str, new_lin: str, rel_files: List[str]) -> Dict[str, str]:
    """For each rel file, run `diff -u` inside WSL and capture the diff.

    Returns dict: rel -> diff text (empty string if identical).
    """
    cmds = []
    for rel in rel_files:
        cmds.append(
            f'echo "=== {rel} ==="; '
            f'if [ -f "{orig_lin}/{rel}" ] && [ -f "{new_lin}/{rel}" ]; then '
            f'  diff -u "{orig_lin}/{rel}" "{new_lin}/{rel}" || true; '
            f'else '
            f'  echo "(missing on one side: orig={orig_lin}/{rel} new={new_lin}/{rel})"; '
            f'fi'
        )
    bash = " ; ".join(cmds)
    try:
        r = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu-20.04", "-e", "bash", "-lc", bash],
            capture_output=True, text=True, timeout=120,
        )
        return {"output": r.stdout + ("\nSTDERR:\n" + r.stderr if r.stderr.strip() else "")}
    except Exception as e:
        return {"output": f"WSL diff failed: {e}"}


def process_tutorial(name: str, tutorial_subpath: str) -> Dict[str, Any]:
    """Load + regenerate one tutorial.  Returns a per-case report record."""
    print(f"\n=== Processing {name} ({tutorial_subpath}) ===")
    src_win = os.path.join(TUTS_WIN, tutorial_subpath.replace("/", "\\"))
    src_lin = TUTS_LIN + "/" + tutorial_subpath
    if not os.path.isdir(src_win):
        return {"name": name, "ok": False, "error": f"source not found: {src_win}"}

    rec: Dict[str, Any] = {
        "name": name,
        "src_win": src_win,
        "src_lin": src_lin,
    }

    # 1. Load through case_loader
    try:
        loaded = load_case(src_win)
    except Exception as e:
        rec.update({"ok": False, "error": f"load_case failed: {e}"})
        return rec

    rec["loaded_keys"] = sorted([k for k in loaded.keys() if not k.startswith("_")])
    rec["loaded_summary"] = loaded.get("_load_summary", {})

    # 2. Build inputs and regenerate
    try:
        inputs = build_inputs_from_loaded(loaded, src_win)
    except Exception as e:
        rec.update({"ok": False, "error": f"build_inputs failed: {e}", "loaded": loaded})
        return rec

    case_name = name + "_GUI"
    gen = Generator3D(WORK_WIN)
    try:
        case_dir = gen.generate(case_name, inputs)
    except Exception as e:
        rec.update({"ok": False, "error": f"generate failed: {e}"})
        return rec

    case_dir_lin = WORK_LIN + "/" + case_name
    rec["new_case_win"] = case_dir
    rec["new_case_lin"] = case_dir_lin
    rec["ok"] = True

    # 3. Diff key files vs original tutorial
    diff_targets = [
        "system/blockMeshDict",
        "system/controlDict",
        "system/setFieldsDict",
        "system/fvSchemes",
        "system/fvSolution",
        "system/decomposeParDict",
        "system/snappyHexMeshDict",
        "system/surfaceFeaturesDict",
        "constant/dynamicMeshDict",
        "constant/phaseProperties",
        "0/p.orig",
        "0/T.orig",
        "0/U.orig",
        "0/alpha.c4.orig",
        "0/rho.c4.orig",
        "0/rho.air.orig",
    ]
    diff = diff_via_wsl(src_lin, case_dir_lin, diff_targets)
    rec["diff"] = diff["output"]
    return rec


def main():
    # Ensure WORK_LIN dir exists in WSL (mkdir -p; harmless if present)
    subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-20.04", "-e", "bash", "-lc", f"mkdir -p {WORK_LIN}"],
        capture_output=True, text=True,
    )

    # Curated list of relevant tutorials (3D, single charge, single block, no
    # baffles / topoSet / wedge / multi-charge — i.e. supported by the GUI).
    candidates = [
        ("building3D", "building3D"),
        ("building3DWorkshop", "building3DWorkshop"),
        ("freeField", "freeField"),
    ]

    results: List[Dict[str, Any]] = []
    for name, sub in candidates:
        rec = process_tutorial(name, sub)
        results.append(rec)

    summary_path = os.path.join(REPORT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Per-case textual diff dumps
    for r in results:
        if not r.get("ok"):
            continue
        diff_path = os.path.join(REPORT_DIR, f"diff_{r['name']}.txt")
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(f"=== Original  : {r['src_lin']}\n")
            f.write(f"=== GUI rebuild: {r['new_case_lin']}\n\n")
            f.write(r.get("diff", ""))

    print("\n=== Summary ===")
    for r in results:
        ok = "OK" if r.get("ok") else "FAIL"
        print(f"  [{ok}] {r['name']}: {r.get('new_case_lin') or r.get('error')}")
    print(f"\nFull report: {summary_path}")
    print(f"Diff files : {REPORT_DIR}/diff_<name>.txt")


if __name__ == "__main__":
    main()
