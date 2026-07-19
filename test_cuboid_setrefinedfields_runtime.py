"""Gated WSL runtime smoke: Cuboid setRefinedFields initialization only.

Skipped cleanly when WSL/OpenFOAM/blastFoam is unavailable. Does not run a
long blastFoam solve.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest

from case_init_mode import record_set_cmd_actual
from generator_3d import Generator3D
from models import CaseInputs3D
from path_utils import win_to_wsl_path


def _wsl_blastfoam_available() -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [
                "wsl",
                "bash",
                "-lc",
                "source /opt/openfoam9/etc/bashrc 2>/dev/null; which blastFoam || true",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"WSL probe failed: {exc}"
    if completed.returncode == 127:
        return False, "WSL unavailable (exit 127)"
    path = (completed.stdout or "").strip().splitlines()
    blast = path[-1] if path else ""
    if blast and "/blastFoam" in blast:
        return True, blast
    return False, "blastFoam not found after sourcing OpenFOAM 9"


def _alpha_has_explosive_cells(case_dir: str) -> tuple[bool, str]:
    alpha_path = os.path.join(case_dir, "0", "alpha.c4")
    if not os.path.isfile(alpha_path):
        return False, "0/alpha.c4 missing"
    with open(alpha_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    m = re.search(r"internalField\s+uniform\s+([-+.eE0-9]+)\s*;", text)
    if m:
        try:
            v = float(m.group(1))
        except ValueError:
            return False, "uniform alpha parse failed"
        return (v > 1e-12), f"uniform {v}"
    if "nonuniform" in text:
        chunk = text[text.find("nonuniform") :]
        vals = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", chunk)
        for token in vals[:5000]:
            try:
                if float(token) > 0.5:
                    return True, "nonuniform alpha>0.5 present"
            except ValueError:
                continue
        return False, "nonuniform alpha has no alpha>0.5 sample"
    return False, "alpha.c4 present but no internalField detected"


def _cuboid_seed2_inputs() -> CaseInputs3D:
    return CaseInputs3D(
        min_point=(0.0, 0.0, 0.0),
        max_point=(1.0, 1.0, 1.0),
        cell_size=0.25,
        charge_center=(0.5, 0.5, 0.5),
        charge_shape="Cuboid",
        mass_kg=0.2 * 0.2 * 0.2 * 1601.0,
        cylinder_radius=0.05,
        cylinder_axis="Z",
        material_name="C4",
        rho_charge=1601.0,
        energy_j_per_kg=4.5e6,
        p_atm=101325.0,
        t_atm=288.0,
        end_time_s=1e-4,
        delta_t=1e-7,
        write_interval_steps=10,
        cores=1,
        enable_dyn_refine=True,
        charge_refinement_level=2,
        charge_outer_refine_enable=False,
        charge_length=0.2,
        charge_width=0.2,
        charge_height=0.2,
        fast_run_mode=True,
    )


class CuboidSetRefinedFieldsRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wsl_ok, cls.wsl_detail = _wsl_blastfoam_available()

    def test_cuboid_setfieldsdict_contains_box_backup_and_level(self):
        """String-level compatibility: Cuboid seed>0 writes boxToCell + backup + level."""
        inputs = _cuboid_seed2_inputs()
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("cuboid_dict", inputs)
            with open(os.path.join(case_dir, "system", "setFieldsDict"), encoding="utf-8") as f:
                text = f.read()
            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                mode = json.load(f)
        self.assertIn("boxToCell", text)
        self.assertIn("backup", text)
        self.assertIn("box (", text)
        self.assertIn("refineInternal yes", text)
        self.assertIn("level 2", text)
        self.assertEqual(mode.get("set_cmd"), "setRefinedFields")
        self.assertIsNone(mode.get("set_cmd_actual"))

    def test_record_set_cmd_actual_helper_matches_production(self):
        """Production helper must persist set_cmd_actual after successful init."""
        inputs = _cuboid_seed2_inputs()
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("cuboid_meta", inputs)
            mode_path = os.path.join(case_dir, "case_init_mode.json")
            with open(mode_path, encoding="utf-8") as f:
                before = json.load(f)
            self.assertEqual(before.get("set_cmd"), "setRefinedFields")
            self.assertIsNone(before.get("set_cmd_actual"))

            updated = record_set_cmd_actual(
                case_dir,
                "setRefinedFields",
                retries_used=0,
                cells_inside_charge=64,
            )
            with open(mode_path, encoding="utf-8") as f:
                after = json.load(f)
        self.assertEqual(updated.get("set_cmd_actual"), "setRefinedFields")
        self.assertEqual(after.get("set_cmd"), "setRefinedFields")
        self.assertEqual(after.get("set_cmd_actual"), "setRefinedFields")
        self.assertEqual(after.get("cells_inside_charge"), 64)
        self.assertEqual(after.get("retries_used"), 0)

    def test_cuboid_init_only_runtime_smoke(self):
        if not self.wsl_ok:
            self.skipTest(
                f"WSL/blastFoam unavailable — runtime Cuboid verification not claimed. "
                f"Detail: {self.wsl_detail}"
            )

        inputs = _cuboid_seed2_inputs()
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("cuboid_smoke", inputs)
            linux_dir = win_to_wsl_path(case_dir)
            # Initialization only — stop before blastFoam. Do not use set -e around
            # OpenFOAM bashrc source (it can return non-zero while still exporting PATH).
            init_script = (
                "set +e; "
                "source /opt/openfoam9/etc/bashrc >/dev/null 2>&1 || true; "
                f"cd '{linux_dir}' || {{ echo CD_FAIL; exit 2; }}; "
                "chmod +x Allrun check_alpha_c4.sh 2>/dev/null || true; "
                "sed -i 's/\\r$//' Allrun check_alpha_c4.sh 2>/dev/null || true; "
                "blockMesh > log.blockMesh 2>&1; echo BLOCKMESH:$?; "
                "snappyHexMesh -overwrite > log.snappyHexMesh 2>&1; echo SNAPPY:$?; "
                "addEmptyPatch internalPatch internal -overwrite > log.addEmptyPatch 2>&1; "
                "echo ADDPATCH:$?; "
                "rm -rf 0; cp -r 0.orig 0; "
                "changeDictionary > log.changeDictionary 2>&1; echo CHANGEDICT:$?; "
                "setRefinedFields > log.setRefinedFields 2>&1; echo SETREFINED:$?; "
                "bash ./check_alpha_c4.sh; echo ALPHA_CHECK:$?"
            )
            completed = subprocess.run(
                ["wsl", "bash", "-lc", init_script],
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8",
                errors="replace",
            )
            combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
            ctx = combined[-2500:]
            for marker in (
                "BLOCKMESH:0",
                "SNAPPY:0",
                "ADDPATCH:0",
                "CHANGEDICT:0",
                "SETREFINED:0",
                "ALPHA_CHECK:0",
            ):
                self.assertIn(marker, combined, f"missing {marker}\n{ctx}")

            ok, detail = _alpha_has_explosive_cells(case_dir)
            self.assertTrue(ok, f"{detail}\n{ctx}")

            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                mode = json.load(f)
            self.assertEqual(
                mode.get("set_cmd"),
                "setRefinedFields",
                f"generated metadata set_cmd mismatch\n{ctx}",
            )


if __name__ == "__main__":
    unittest.main()
