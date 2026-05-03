"""Headless scenario tests for the 3D GUI -> Generator pipeline.

For each scenario:
  1. Build a TabGeneral3D, drive its widgets (Mesh mode, charge, AMR, bubble factor, ...).
  2. Call get_case_inputs() to obtain a CaseInputs3D.
  3. Run Generator3D.generate(case_name, inputs) into a temp dir.
  4. Assert on the generated dictionary files.

This does NOT run blastFoam itself (requires WSL/OpenFOAM); it validates that the
generated case matches what the user configured in the GUI.
"""
from __future__ import annotations

import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, List

from PyQt5.QtWidgets import QApplication

from tab_3d_general import TabGeneral3D
from probes_model import ProbesModel
from generator_3d import Generator3D


@dataclass
class Result:
    name: str
    passed: bool
    notes: List[str]
    case_dir: str = ""

    def add(self, ok: bool, msg: str) -> None:
        prefix = "OK  " if ok else "FAIL"
        self.notes.append(f"  {prefix} {msg}")
        if not ok:
            self.passed = False


def read(case_dir: str, rel: str) -> str:
    p = os.path.join(case_dir, rel)
    if not os.path.exists(p):
        return ""
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def _rebuild_radius(tab: TabGeneral3D) -> None:
    tab._update_charge_radius()


def _select_dyn(tab: TabGeneral3D, refine_max: int = 1, refine_min: int = 2) -> None:
    tab.rad_dyn_mesh.setChecked(True)
    tab.rad_fixed_mesh.setChecked(False)
    tab.spin_refine_min.setValue(refine_min)
    tab.spin_refine_max.setValue(refine_max)
    tab._set_provenance_user("enable_dyn_refine")


def _select_fixed(tab: TabGeneral3D) -> None:
    tab.rad_fixed_mesh.setChecked(True)
    tab.rad_dyn_mesh.setChecked(False)
    tab._set_provenance_user("enable_dyn_refine")


def _set_domain(tab: TabGeneral3D, x: tuple, y: tuple, z: tuple, cell: float) -> None:
    tab.sx1.setValue(x[0]); tab.sx2.setValue(x[1])
    tab.sy1.setValue(y[0]); tab.sy2.setValue(y[1])
    tab.sz1.setValue(z[0]); tab.sz2.setValue(z[1])
    tab.scell.setValue(cell)


def _set_charge(tab: TabGeneral3D, shape: str, mass: float, rho: float,
                center: tuple, *, lbyd: float = 2.5, axis: str = "Z") -> None:
    tab.c_shape.setCurrentText(shape)
    tab.c_mass.setValue(mass)
    tab.c_rho.setValue(rho)
    tab.cx.setValue(center[0]); tab.cy.setValue(center[1]); tab.cz.setValue(center[2])
    if shape == "Cylinder":
        tab.c_aspect.setValue(lbyd)
        idx = tab.c_cylinder_axis.findText(axis)
        if idx >= 0:
            tab.c_cylinder_axis.setCurrentIndex(idx)
    _rebuild_radius(tab)


def _set_charge_refine(tab: TabGeneral3D, inside: int, outer_min: int, outer_max: int) -> None:
    tab.spin_charge_refine.setValue(inside)
    tab.spin_charge_outer_min.setValue(outer_min)
    tab.spin_charge_outer_max.setValue(outer_max)


def run_scenario(name: str, configure: Callable[[TabGeneral3D], None],
                 expectations: Callable[[Result, str], None],
                 base_path: str) -> Result:
    res = Result(name=name, passed=True, notes=[])
    tab = TabGeneral3D(ProbesModel())
    try:
        configure(tab)
    except Exception as e:
        res.add(False, f"GUI configuration raised {type(e).__name__}: {e}")
        return res

    try:
        inputs = tab.get_case_inputs()
    except Exception as e:
        res.add(False, f"get_case_inputs() raised {type(e).__name__}: {e}")
        return res

    gen = Generator3D(base_path)
    try:
        case_dir = gen.generate(name, inputs)
        res.case_dir = case_dir
    except Exception as e:
        res.add(False, f"Generator3D.generate raised {type(e).__name__}: {e}")
        return res

    try:
        expectations(res, case_dir)
    except AssertionError as e:
        res.add(False, f"Assertion failed: {e}")
    except Exception as e:
        res.add(False, f"Expectation raised {type(e).__name__}: {e}")
    return res


def assert_in(res: Result, haystack: str, needle: str, label: str) -> None:
    res.add(needle in haystack, f"{label}: '{needle}' present" if needle in haystack else f"{label}: missing '{needle}'")


def assert_not_in(res: Result, haystack: str, needle: str, label: str) -> None:
    res.add(needle not in haystack, f"{label}: '{needle}' absent" if needle not in haystack else f"{label}: should NOT contain '{needle}'")


_CMD_INVOKE_RE = r"^[ \t]*{cmd}(\s|$)"


def assert_runs_cmd(res: Result, allrun: str, cmd: str, label: str) -> None:
    pat = _CMD_INVOKE_RE.format(cmd=re.escape(cmd))
    m = re.search(pat, allrun, re.MULTILINE)
    res.add(bool(m), f"{label}: command '{cmd}' invoked" if m else f"{label}: command '{cmd}' is NOT invoked")


def assert_not_runs_cmd(res: Result, allrun: str, cmd: str, label: str) -> None:
    pat = _CMD_INVOKE_RE.format(cmd=re.escape(cmd))
    m = re.search(pat, allrun, re.MULTILINE)
    res.add(not m, f"{label}: command '{cmd}' is NOT invoked" if not m else f"{label}: command '{cmd}' is invoked but should not be")


def assert_match(res: Result, haystack: str, pattern: str, label: str) -> None:
    m = re.search(pattern, haystack)
    res.add(bool(m), f"{label}: pattern '{pattern}' " + ("matched: " + m.group(0) if m else "NOT matched"))


SCENARIOS: list[tuple[str, Callable[[TabGeneral3D], None], Callable[[Result, str], None]]] = []


def scenario_sphere_b3d(tab: TabGeneral3D) -> None:
    _set_domain(tab, (-5, 5), (-5, 5), (0, 5), 0.5)
    tab.c_mat.setCurrentText("C4")
    _set_charge(tab, "Sphere", 25.0, 1601.0, (0.0, 0.0, 0.5))
    _select_dyn(tab, refine_max=1, refine_min=2)
    _set_charge_refine(tab, inside=5, outer_min=2, outer_max=3)
    tab._bubble_radius_factor = 1.5  # snappy transition seed; charge capture is separate (auto)

def expect_sphere_b3d(res: Result, case_dir: str) -> None:
    sf = read(case_dir, "system/setFieldsDict")
    dm = read(case_dir, "constant/dynamicMeshDict")
    allrun = read(case_dir, "Allrun")
    assert_in(res, sf, "sphericalMassToCell", "setFieldsDict has sphericalMassToCell")
    assert_in(res, sf, "refineInternal yes", "setFieldsDict uses native refineInternal")
    assert_in(res, sf, "level 5", "setFieldsDict level=5")
    assert_in(res, sf, "backup", "setFieldsDict has backup region")
    # Auto capture: max(1.05*R, 0.5*sqrt(3)*dx*factor) with dx=0.5, factor=1, R~0.155 m → ~0.433 m
    assert_match(res, sf, r"radius\s+0\.43[0-9]", "backup.radius matches auto charge capture policy")
    assert_in(res, dm, "adaptiveFvMesh", "dynamicMeshDict adaptive")
    assert_in(res, dm, "maxRefinement   1", "dynamicMeshDict maxRefinement=1")
    assert_in(res, dm, "dumpLevel      true", "dumpLevel true (allows level field decay visualization)")
    assert_runs_cmd(res, allrun, "setRefinedFields", "Allrun calls setRefinedFields")
    assert_not_in(res, allrun, "setRefinedFields -noRefine", "Allrun does NOT use -noRefine")
    assert_not_runs_cmd(res, allrun, "topoSet", "Allrun does NOT have manual topoSet stages")
SCENARIOS.append(("sphere_b3d_match", scenario_sphere_b3d, expect_sphere_b3d))


def scenario_sphere_fine_no_inside(tab: TabGeneral3D) -> None:
    _set_domain(tab, (-2, 2), (-2, 2), (-2, 2), 0.05)
    tab.c_mat.setCurrentText("C4")
    _set_charge(tab, "Sphere", 1.0, 1601.0, (0.0, 0.0, 0.0))
    _select_dyn(tab, refine_max=1, refine_min=2)
    _set_charge_refine(tab, inside=0, outer_min=2, outer_max=2)

def expect_sphere_fine_no_inside(res: Result, case_dir: str) -> None:
    sf = read(case_dir, "system/setFieldsDict")
    dm = read(case_dir, "constant/dynamicMeshDict")
    allrun = read(case_dir, "Allrun")
    has_sphere_region = ("sphericalMassToCell" in sf) or ("sphereToCell" in sf)
    res.add(has_sphere_region, "setFieldsDict has spherical-style region (sphericalMassToCell or sphereToCell)")
    assert_not_in(res, sf, "refineInternal", "no refineInternal when Inside=0")
    assert_not_in(res, sf, "backup", "no backup region when Inside=0")
    assert_in(res, dm, "adaptiveFvMesh", "AMR enabled")
    assert_runs_cmd(res, allrun, "setFields", "Allrun runs setFields")
    assert_not_runs_cmd(res, allrun, "setRefinedFields", "Allrun does NOT call setRefinedFields when Inside=0")
SCENARIOS.append(("sphere_fine_no_inside", scenario_sphere_fine_no_inside, expect_sphere_fine_no_inside))


def scenario_sphere_fixed(tab: TabGeneral3D) -> None:
    _set_domain(tab, (-5, 5), (-5, 5), (0, 5), 0.5)
    tab.c_mat.setCurrentText("C4")
    _set_charge(tab, "Sphere", 25.0, 1601.0, (0.0, 0.0, 0.5))
    _select_fixed(tab)
    _set_charge_refine(tab, inside=5, outer_min=2, outer_max=3)

def expect_sphere_fixed(res: Result, case_dir: str) -> None:
    sf = read(case_dir, "system/setFieldsDict")
    dm = read(case_dir, "constant/dynamicMeshDict")
    allrun = read(case_dir, "Allrun")
    assert_in(res, dm, "staticFvMesh", "dynamicMeshDict static (no AMR)")
    assert_not_in(res, dm, "adaptiveFvMesh", "no adaptive when Fixed Mesh")
    assert_not_in(res, sf, "refineInternal", "Inside refinement skipped when Fixed Mesh")
    assert_not_runs_cmd(res, allrun, "setRefinedFields", "Allrun has no setRefinedFields when Fixed Mesh")
SCENARIOS.append(("sphere_fixed_mesh", scenario_sphere_fixed, expect_sphere_fixed))


def scenario_cylinder_b3d(tab: TabGeneral3D) -> None:
    _set_domain(tab, (-5, 5), (-5, 5), (0, 5), 0.5)
    tab.c_mat.setCurrentText("C4")
    _set_charge(tab, "Cylinder", 25.0, 1601.0, (0.0, 0.0, 0.5), lbyd=2.5, axis="Z")
    _select_dyn(tab, refine_max=1, refine_min=2)
    _set_charge_refine(tab, inside=5, outer_min=2, outer_max=3)
    tab._bubble_radius_factor = 1.5

def expect_cylinder_b3d(res: Result, case_dir: str) -> None:
    sf = read(case_dir, "system/setFieldsDict")
    dm = read(case_dir, "constant/dynamicMeshDict")
    allrun = read(case_dir, "Allrun")
    assert_in(res, sf, "cylindericalMassToCell", "setFieldsDict has cylindericalMassToCell (BlastFoam spelling)")
    assert_in(res, sf, "refineInternal yes", "setFieldsDict uses native refineInternal")
    assert_in(res, sf, "level 5", "setFieldsDict level=5")
    assert_in(res, sf, "backup", "setFieldsDict has backup region")
    assert_in(res, dm, "adaptiveFvMesh", "AMR enabled")
    assert_runs_cmd(res, allrun, "setRefinedFields", "Allrun calls setRefinedFields")
SCENARIOS.append(("cylinder_b3d_match", scenario_cylinder_b3d, expect_cylinder_b3d))


def scenario_cuboid(tab: TabGeneral3D) -> None:
    _set_domain(tab, (-5, 5), (-5, 5), (0, 5), 0.5)
    tab.c_mat.setCurrentText("C4")
    tab.c_shape.setCurrentText("Cuboid")
    tab.c_mass.setValue(25.0); tab.c_rho.setValue(1601.0)
    tab.c_length.setValue(0.5); tab.c_width.setValue(0.5); tab.c_height.setValue(0.5)
    tab.cx.setValue(0.0); tab.cy.setValue(0.0); tab.cz.setValue(0.5)
    _select_dyn(tab, refine_max=1, refine_min=2)
    _set_charge_refine(tab, inside=0, outer_min=2, outer_max=2)

def expect_cuboid(res: Result, case_dir: str) -> None:
    sf = read(case_dir, "system/setFieldsDict")
    dm = read(case_dir, "constant/dynamicMeshDict")
    assert_in(res, sf, "boxToCell", "setFieldsDict has boxToCell for Cuboid")
    assert_in(res, dm, "adaptiveFvMesh", "AMR enabled")
SCENARIOS.append(("cuboid_standard", scenario_cuboid, expect_cuboid))


def scenario_sphere_factor3(tab: TabGeneral3D) -> None:
    _set_domain(tab, (-5, 5), (-5, 5), (0, 5), 0.5)
    tab.c_mat.setCurrentText("C4")
    _set_charge(tab, "Sphere", 25.0, 1601.0, (0.0, 0.0, 0.5))
    _select_dyn(tab, refine_max=1, refine_min=2)
    _set_charge_refine(tab, inside=5, outer_min=2, outer_max=3)
    tab._bubble_radius_factor = 1.5
    vol = 25.0 / 1601.0
    r_phys = ((3.0 * vol) / (4.0 * math.pi)) ** (1.0 / 3.0)
    tab._charge_capture_mode = "manual"
    tab._charge_capture_radius_manual = 3.0 * r_phys
    tab._charge_backup_radius_override = 3.0 * r_phys

def expect_sphere_factor3(res: Result, case_dir: str) -> None:
    sf = read(case_dir, "system/setFieldsDict")
    # 0.155 m * 3.0 = 0.465 m
    assert_match(res, sf, r"radius\s+0\.46[0-9]+", "backup.radius ~= 3.0 x R = 0.465 m")
SCENARIOS.append(("sphere_bubble_factor_3.0", scenario_sphere_factor3, expect_sphere_factor3))


def scenario_amr_aggressive(tab: TabGeneral3D) -> None:
    _set_domain(tab, (-5, 5), (-5, 5), (0, 5), 0.5)
    tab.c_mat.setCurrentText("C4")
    _set_charge(tab, "Sphere", 25.0, 1601.0, (0.0, 0.0, 0.5))
    _select_dyn(tab, refine_max=2, refine_min=2)
    _set_charge_refine(tab, inside=5, outer_min=2, outer_max=3)
    tab._refine_interval = 1
    tab._unrefine_threshold = 0.5
    tab._n_buffer_layers_dynamic = 1

def expect_amr_aggressive(res: Result, case_dir: str) -> None:
    dm = read(case_dir, "constant/dynamicMeshDict")
    assert_in(res, dm, "refineInterval  1", "refineInterval=1 (per-step AMR)")
    assert_in(res, dm, "unrefineLevel   0.5", "unrefineLevel=0.5 (aggressive decay)")
    assert_in(res, dm, "nBufferLayers   1", "nBufferLayers=1")
    assert_in(res, dm, "maxRefinement   2", "maxRefinement=2")
SCENARIOS.append(("sphere_amr_aggressive_decay", scenario_amr_aggressive, expect_amr_aggressive))


def scenario_amr_max5(tab: TabGeneral3D) -> None:
    _set_domain(tab, (-5, 5), (-5, 5), (0, 5), 0.5)
    tab.c_mat.setCurrentText("C4")
    _set_charge(tab, "Sphere", 25.0, 1601.0, (0.0, 0.0, 0.5))
    _select_dyn(tab, refine_max=5, refine_min=2)
    _set_charge_refine(tab, inside=5, outer_min=2, outer_max=3)

def expect_amr_max5(res: Result, case_dir: str) -> None:
    dm = read(case_dir, "constant/dynamicMeshDict")
    assert_in(res, dm, "maxRefinement   5", "maxRefinement=5 (AMR can refine up to 5)")
SCENARIOS.append(("sphere_amr_max5", scenario_amr_max5, expect_amr_max5))


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    base = tempfile.mkdtemp(prefix="ggui_scenarios_")
    print(f"Base path: {base}\n")
    print("=" * 78)
    results: List[Result] = []
    for name, conf, exp in SCENARIOS:
        r = run_scenario(name, conf, exp, base)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {name}")
        for line in r.notes:
            print(line)
        print("-" * 78)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"\nSummary: {passed} passed, {failed} failed (of {len(results)})")
    print(f"Cases written under: {base}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
