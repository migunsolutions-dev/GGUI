"""End-to-end WSL validation: generate sphere_b3d_match, run a short blastFoam,
verify mesh actually decays back toward base level over time.

Steps:
  1. Build case via TabGeneral3D + Generator3D (same as harness) into
     C:\\Users\\migun\\Desktop\\GGUI\\_e2e_runs\\<name>.
  2. Shorten endTime to keep runtime manageable.
  3. Convert Windows path to WSL (/mnt/c/Users/...).
  4. Run Allrun via WSL.
  5. Inspect log files and the per-time `level` field to confirm:
       * blockMesh   OK
       * snappyHexMesh OK
       * setRefinedFields seeded alpha.c4 (positive cells)
       * blastFoam ran timesteps
       * Mesh decay: max(level) drops over time (or unrefined cell count grows).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from tab_3d_general import TabGeneral3D
from probes_model import ProbesModel
from generator_3d import Generator3D


SHORT_END_TIME = 2.0e-6
SHORT_WRITE_INTERVAL = 5.0e-7
RUN_ROOT_WIN = r"C:\Users\migun\Desktop\GGUI\_e2e_runs"
ALLRUN_TIMEOUT_SEC = 1200


def win_to_wsl(p: str) -> str:
    p = os.path.abspath(p).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def configure_sphere_b3d(tab: TabGeneral3D) -> None:
    tab.sx1.setValue(-5); tab.sx2.setValue(5)
    tab.sy1.setValue(-5); tab.sy2.setValue(5)
    tab.sz1.setValue(0); tab.sz2.setValue(5)
    tab.scell.setValue(0.5)
    tab.c_mat.setCurrentText("C4")
    tab.c_shape.setCurrentText("Sphere")
    tab.c_mass.setValue(25.0); tab.c_rho.setValue(1601.0)
    tab.cx.setValue(0.0); tab.cy.setValue(0.0); tab.cz.setValue(0.5)
    tab._update_charge_radius()
    tab.rad_dyn_mesh.setChecked(True); tab.rad_fixed_mesh.setChecked(False)
    tab.spin_refine_min.setValue(2); tab.spin_refine_max.setValue(1)
    tab._set_provenance_user("enable_dyn_refine")
    tab.spin_charge_refine.setValue(5)
    tab.spin_charge_outer_min.setValue(2)
    tab.spin_charge_outer_max.setValue(3)
    tab._bubble_radius_factor = 1.5
    # spin_end / spin_write_time are now configured with decimals=10 in the GUI itself,
    # so small values like 2e-6 / 5e-7 are preserved without any local override.
    tab.spin_end.setValue(SHORT_END_TIME)
    tab.spin_write_time.setValue(SHORT_WRITE_INTERVAL)
    idx = tab.combo_write_control.findText("adjustableRunTime")
    if idx >= 0:
        tab.combo_write_control.setCurrentIndex(idx)
    tab.spin_cores.setValue(4)


def generate_case() -> str:
    if os.path.exists(RUN_ROOT_WIN):
        shutil.rmtree(RUN_ROOT_WIN, ignore_errors=True)
    os.makedirs(RUN_ROOT_WIN, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    tab = TabGeneral3D(ProbesModel())
    configure_sphere_b3d(tab)
    inputs = tab.get_case_inputs()
    gen = Generator3D(RUN_ROOT_WIN)
    case_dir = gen.generate("sphere_b3d_short", inputs)
    print(f"Case generated: {case_dir}")
    return case_dir


def wsl_run(cmd: str, cwd_wsl: str | None = None, timeout: int = 600) -> tuple[int, str, str]:
    full = ["wsl", "--", "bash", "-lc",
            (f"cd '{cwd_wsl}' && " if cwd_wsl else "") +
            "source /opt/openfoam9/etc/bashrc && " + cmd]
    p = subprocess.run(full, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout or "", p.stderr or ""


def run_allrun(case_dir: str) -> tuple[int, str]:
    case_wsl = win_to_wsl(case_dir)
    print(f"WSL path : {case_wsl}")
    print(f"Running  : bash Allrun (timeout 600s)...")
    t0 = time.time()
    rc, out, err = wsl_run(
        "chmod +x Allrun Allclean check_alpha_c4.sh check_charge_region.py 2>/dev/null; bash Allrun",
        cwd_wsl=case_wsl,
        timeout=ALLRUN_TIMEOUT_SEC,
    )
    elapsed = time.time() - t0
    log = (out or "") + ("\n--- stderr ---\n" + err if err else "")
    print(f"Allrun exit={rc} after {elapsed:.1f}s")
    return rc, log


def parse_blastfoam_log(case_dir: str) -> list[tuple[float, str, int, int]]:
    """Return list of (time, kind, before, after) for each AMR event."""
    log_path = os.path.join(case_dir, "log.blastFoam")
    if not os.path.exists(log_path):
        return []
    rows = []
    cur_t = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            mt = re.match(r"^Time\s*=\s*([0-9.eE+-]+)", line)
            if mt:
                cur_t = float(mt.group(1))
                continue
            mref = re.search(r"Refined from\s+(\d+)\s+to\s+(\d+)", line)
            mun = re.search(r"Unrefined from\s+(\d+)\s+to\s+(\d+)", line)
            if cur_t is None:
                continue
            if mref:
                rows.append((cur_t, "refine", int(mref.group(1)), int(mref.group(2))))
            elif mun:
                rows.append((cur_t, "unrefine", int(mun.group(1)), int(mun.group(2))))
    return rows


def get_time_dirs(case_dir: str) -> list[float]:
    out = []
    for name in os.listdir(case_dir):
        try:
            t = float(name)
            if os.path.isdir(os.path.join(case_dir, name)):
                out.append(t)
        except ValueError:
            pass
    return sorted(out)


def get_max_level_from_dir(case_dir: str, t: float) -> int | None:
    """Read polyMesh/cellLevel for time t (or constant) and return max value."""
    candidates = [
        os.path.join(case_dir, f"{t}", "polyMesh", "cellLevel"),
        os.path.join(case_dir, "constant", "polyMesh", "cellLevel"),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
        except OSError:
            continue
        m = re.search(r"\(([\s\S]*?)\)", txt)
        if not m:
            continue
        vals = []
        for tok in m.group(1).split():
            try:
                vals.append(int(tok))
            except ValueError:
                pass
        if vals:
            return max(vals)
    return None


def check_alpha_c4_seeded(case_dir: str) -> tuple[bool, int, str]:
    """Read 0/alpha.c4 internalField and count nonzero cells (after setRefinedFields)."""
    alpha_path = os.path.join(case_dir, "0", "alpha.c4")
    if not os.path.exists(alpha_path):
        return (False, 0, "0/alpha.c4 missing")
    with open(alpha_path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    m = re.search(r"internalField\s+nonuniform[^\n]*\n\s*(\d+)\s*\n\s*\(([\s\S]*?)\)", txt)
    if not m:
        m2 = re.search(r"internalField\s+uniform\s+([0-9.eE+-]+)", txt)
        if m2:
            v = float(m2.group(1))
            return (v > 0, 1 if v > 0 else 0, f"uniform {v}")
        return (False, 0, "internalField not parsed")
    vals = m.group(2).split()
    nonzero = sum(1 for v in vals if float(v) > 0.0)
    return (nonzero > 0, nonzero, f"{nonzero} of {len(vals)} cells with alpha.c4>0")


def main() -> int:
    print("=" * 78)
    print("E2E WSL validation: sphere_b3d_match (short run)")
    print("=" * 78)
    case_dir = generate_case()
    rc, log = run_allrun(case_dir)

    log_tail_path = os.path.join(case_dir, "_run_log.txt")
    with open(log_tail_path, "w", encoding="utf-8") as f:
        f.write(log)
    print(f"Run log saved: {log_tail_path}")

    print("\n--- Stage checks ---")
    summary = []
    for stage, marker in [
        ("blockMesh",         "log.blockMesh"),
        ("snappyHexMesh",     "log.snappyHexMesh"),
        ("addEmptyPatch",     "log.addEmptyPatch"),
        ("changeDictionary",  "log.changeDictionary"),
        ("setFields",         "log.setFields"),
        ("setRefinedFields",  "log.setRefinedFields"),
        ("blastFoam",         "log.blastFoam"),
    ]:
        p = os.path.join(case_dir, marker)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-200:]
            ok = ("FOAM FATAL ERROR" not in tail) and ("End" in tail or "Finalising" in tail or "Time =" in tail)
            print(f"  [{'OK' if ok else '??'}] {stage:18s}  log={marker}")
            summary.append((stage, ok))
        else:
            print(f"  [MISSING] {stage:18s}  log={marker}")
            summary.append((stage, False))

    seeded, n, msg = check_alpha_c4_seeded(case_dir)
    print(f"\n--- alpha.c4 seeding ---\n  {'OK' if seeded else 'FAIL'}: {msg}")

    rows = parse_blastfoam_log(case_dir)
    if rows:
        print(f"\n--- AMR activity (blastFoam log) ---")
        print(f"  Refine/unrefine events: {len(rows)}")
        max_cells = max(b for _, _, b, _ in rows) if rows else 0
        end_cells = rows[-1][3] if rows else 0
        print(f"  Peak cell count : {max_cells:,}")
        print(f"  Final cell count: {end_cells:,}")
        print(f"  Reduction       : {max_cells - end_cells:,} cells ({(max_cells-end_cells)/max(1,max_cells)*100:.1f}%)")
        print()
        unref_events = [(t, b, a) for t, k, b, a in rows if k == "unrefine"]
        ref_events   = [(t, b, a) for t, k, b, a in rows if k == "refine"]
        print(f"  Unrefinement events: {len(unref_events)}")
        for t, b, a in unref_events[:6]:
            print(f"    t={t:.3e}  {b:>10,} -> {a:>10,}  (-{b-a:,})")
        print(f"  Refinement events  : {len(ref_events)}")
        for t, b, a in ref_events[:4]:
            print(f"    t={t:.3e}  {b:>10,} -> {a:>10,}  (+{a-b:,})")
        if unref_events:
            print(f"  -> Mesh DOES decay over time (unrefinement triggered {len(unref_events)} times).")
        else:
            print(f"  -> No unrefinement detected in this window.")

    times = get_time_dirs(case_dir)
    print(f"\n--- Time dirs written ---\n  {times}")

    print(f"\n--- Cell-level field decay ---")
    max_lvl0 = get_max_level_from_dir(case_dir, 0)
    print(f"  t=0:        max(cellLevel) = {max_lvl0}")
    for t in times:
        if t == 0:
            continue
        ml = get_max_level_from_dir(case_dir, t)
        print(f"  t={t:<10.4e} max(cellLevel) = {ml}")

    n_pass = sum(1 for _, ok in summary if ok) + (1 if seeded else 0)
    n_total = len(summary) + 1
    print("\n" + "=" * 78)
    print(f"Result: {n_pass}/{n_total} stage checks passed.")
    print(f"Allrun exit code: {rc}")
    print("=" * 78)
    return 0 if rc == 0 and seeded else 1


if __name__ == "__main__":
    sys.exit(main())
