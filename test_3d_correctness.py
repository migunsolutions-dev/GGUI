from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace

from generator_3d import Generator3D
from initialization_plan import build_initialization_plan
from models import CaseInputs3D, ObstacleData
from project_io import (
    ProjectFormatError,
    build_project,
    read_project,
    write_project_atomic,
)
from solver_runner import (
    ExecutionIntent,
    ExecutionPreparationError,
    build_execution_plan,
)
from startup_capture_guard import evaluate_unsafe_capture


def case(**overrides) -> CaseInputs3D:
    values = dict(
        min_point=(0.0, 0.0, 0.0),
        max_point=(2.0, 2.0, 2.0),
        cell_size=0.25,
        charge_center=(1.0, 1.0, 1.0),
        charge_shape="Sphere",
        mass_kg=1.0,
        cylinder_radius=0.08,
        cylinder_axis="Z",
        material_name="C4",
        rho_charge=1601.0,
        energy_j_per_kg=4.5e6,
        p_atm=101325.0,
        t_atm=288.0,
        end_time_s=1e-3,
        delta_t=1e-7,
        write_interval_steps=10,
        cores=1,
        enable_dyn_refine=True,
        dyn_refine_max=1,
        charge_refinement_level=0,
        charge_outer_refine_enable=True,
        charge_outer_refine_min=2,
        charge_outer_refine_max=3,
    )
    values.update(overrides)
    return CaseInputs3D(**values)


class ProtectedDefaultsTests(unittest.TestCase):
    def test_protected_model_defaults(self):
        value = case()
        self.assertEqual(value.charge_refinement_level, 0)
        self.assertEqual(value.dyn_refine_max, 1)
        self.assertEqual(value.refine_indicator_field, "densityGradient")
        self.assertEqual(value.refine_interval, 3)
        self.assertEqual(value.lower_refine_threshold, 0.1)
        self.assertEqual(value.unrefine_threshold, 0.1)


class InitializationPlanTests(unittest.TestCase):
    def test_all_shapes_metadata_and_allrun_agree(self):
        for shape in ("Sphere", "Cylinder", "Cuboid"):
            for seed in (0, 2):
                with self.subTest(shape=shape, seed=seed), tempfile.TemporaryDirectory() as td:
                    inputs = case(
                        charge_shape=shape,
                        charge_refinement_level=seed,
                        charge_length=0.1,
                        charge_width=0.1,
                        charge_height=0.1,
                    )
                    expected = build_initialization_plan(inputs)
                    case_dir = Generator3D(td).generate("case", inputs)
                    with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                        metadata = json.load(f)
                    with open(os.path.join(case_dir, "Allrun"), encoding="utf-8") as f:
                        allrun = f.read()
                    self.assertEqual(metadata["set_cmd"], expected.command)
                    self.assertEqual(
                        metadata["startup_mesh_metadata"]["startup_mesh"]["uses_set_refined_fields"],
                        expected.uses_set_refined_fields,
                    )
                    command_line = f"{expected.command} > log.{expected.command}"
                    self.assertIn(command_line, allrun)
                    other = "setFields" if expected.command == "setRefinedFields" else "setRefinedFields"
                    self.assertNotIn(f"{other} > log.{other}", allrun)


class RunnerIntentTests(unittest.TestCase):
    def _initialized_case(self, root: str) -> None:
        os.makedirs(os.path.join(root, "0"), exist_ok=True)
        os.makedirs(os.path.join(root, "system"), exist_ok=True)
        open(os.path.join(root, "0", "p"), "w").close()

    def test_initialized_serial_never_allrun(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            plan = build_execution_plan(td, 1, ExecutionIntent.INITIALIZED_SOLVER_RUN)
            self.assertIn("blastFoam", plan.command)
            self.assertNotIn("Allrun", plan.command)
            self.assertNotIn("Allclean", plan.command)
            fresh = build_execution_plan(td, 1, ExecutionIntent.FRESH_FULL_PIPELINE)
            self.assertEqual(fresh.command, "bash ./Allrun")

    def test_resume_requires_saved_state(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            with self.assertRaises(ExecutionPreparationError):
                build_execution_plan(td, 1, ExecutionIntent.RESUME)

    def test_serial_and_parallel_resume_preserve_directories(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "0.1"))
            os.makedirs(os.path.join(td, "processor0", "0.1"))
            os.makedirs(os.path.join(td, "processor1", "0.1"))
            serial = build_execution_plan(td, 1, ExecutionIntent.RESUME)
            parallel = build_execution_plan(td, 2, ExecutionIntent.RESUME)
            self.assertIn("latestTime", serial.command)
            self.assertIn("mpirun -np 2", parallel.command)
            self.assertNotIn("rm -rf", parallel.command)
            self.assertTrue(os.path.isdir(os.path.join(td, "0.1")))
            self.assertTrue(os.path.isdir(os.path.join(td, "processor0", "0.1")))

    def test_new_parallel_run_decomposes_without_cleaning(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            plan = build_execution_plan(td, 4, ExecutionIntent.INITIALIZED_SOLVER_RUN)
            self.assertIn("decomposePar -force", plan.command)
            self.assertIn("mpirun -np 4", plan.command)
            self.assertNotIn("rm -rf", plan.command)
            self.assertNotIn("Allclean", plan.command)

    def test_parallel_processor_count_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "processor0", "0.1"))
            os.makedirs(os.path.join(td, "processor1", "0.1"))
            with self.assertRaisesRegex(ExecutionPreparationError, "2 directories"):
                build_execution_plan(td, 4, ExecutionIntent.RESUME)

    def test_serial_resume_reconstructs_newer_parallel_state(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "processor0", "0.2"))
            os.makedirs(os.path.join(td, "processor1", "0.2"))
            plan = build_execution_plan(td, 1, ExecutionIntent.RESUME)
            self.assertIn("reconstructPar -latestTime", plan.command)
            self.assertNotIn("Allclean", plan.command)

    def test_inconsistent_parallel_times_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "processor0", "0.2"))
            os.makedirs(os.path.join(td, "processor1", "0.1"))
            with self.assertRaisesRegex(ExecutionPreparationError, "consistent latest"):
                build_execution_plan(td, 2, ExecutionIntent.RESUME)

    def test_one_step_accepts_initialized_zero(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            plan = build_execution_plan(td, 1, ExecutionIntent.ONE_STEP_RESUME)
            self.assertEqual(plan.latest_time, 0.0)


class CaptureGuardTests(unittest.TestCase):
    def test_unsafe_seed_zero_no_band(self):
        inputs = case(
            min_point=(0, 0, 0),
            max_point=(1, 1, 1),
            cell_size=0.2,
            charge_center=(0.4, 0.4, 0.4),
            mass_kg=0.001,
            charge_outer_refine_enable=False,
        )
        self.assertFalse(evaluate_unsafe_capture(inputs).safe)

    def test_subcell_charge_can_contain_cell_centre(self):
        inputs = case(
            min_point=(0, 0, 0),
            max_point=(1, 1, 1),
            cell_size=0.2,
            charge_center=(0.5, 0.5, 0.5),
            mass_kg=0.02,
            charge_outer_refine_enable=False,
        )
        self.assertTrue(evaluate_unsafe_capture(inputs).safe)

    def test_seed_or_band_protects_all_shapes(self):
        for shape in ("Sphere", "Cylinder", "Cuboid"):
            base = case(charge_shape=shape, mass_kg=0.001)
            self.assertTrue(evaluate_unsafe_capture(replace(base, charge_refinement_level=2)).safe)
            self.assertTrue(evaluate_unsafe_capture(replace(base, charge_outer_refine_enable=True)).safe)


class ProjectPersistenceTests(unittest.TestCase):
    def test_project_round_trip(self):
        original = case(
            dyn_refine_max=4,
            begin_unrefine=2e-4,
            enable_balancing=True,
            obstacles=[ObstacleData("wall.stl", "wall", 0.001, 1, 2, 3, 2)],
            charge_capture_mode="manual",
            charge_capture_radius=0.12,
            write_control_type="adjustableRunTime",
            write_interval_time=2e-5,
        )
        payload = build_project(
            original,
            probes={"probes": [{"name": "P1", "x": 1, "y": 2, "z": 3}]},
            gui_state={
                "sections": [{"name": "cut"}],
                "obstacles": [
                    {
                        "enabled": False,
                        "path": "disabled-wall.stl",
                        "scale": 0.001,
                        "ox": 1.0,
                        "oy": 2.0,
                        "oz": 3.0,
                    }
                ],
            },
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.ggui.json")
            write_project_atomic(path, payload)
            loaded = read_project(path)
        self.assertEqual(loaded["inputs"], original)
        self.assertEqual(loaded["probes"], payload["probes"])
        self.assertEqual(loaded["gui_state"], payload["gui_state"])

    def test_loaded_project_regenerates_solver_dictionaries(self):
        with tempfile.TemporaryDirectory() as td:
            stl = os.path.join(td, "wall.stl")
            with open(stl, "w", encoding="ascii") as f:
                f.write(
                    "solid wall\nfacet normal 0 0 1\nouter loop\n"
                    "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
                    "endloop\nendfacet\nendsolid wall\n"
                )
            original = case(
                obstacles=[ObstacleData(stl, "wall", 1.0, 0, 0, 0, 2)],
                dyn_refine_max=2,
                upper_refine_level=0.2,
                balance_interval=5,
            )
            path = os.path.join(td, "model.ggui.json")
            write_project_atomic(
                path,
                build_project(original, probes={"probes": []}, gui_state={"sections": []}),
            )
            loaded = read_project(path)["inputs"]
            first = Generator3D(td).generate("first", original)
            second = Generator3D(td).generate("second", loaded)
            for rel in (
                "constant/dynamicMeshDict",
                "system/controlDict",
                "system/setFieldsDict",
                "system/snappyHexMeshDict",
            ):
                with open(os.path.join(first, rel), encoding="utf-8") as f:
                    a = f.read()
                with open(os.path.join(second, rel), encoding="utf-8") as f:
                    b = f.read()
                self.assertEqual(a, b, rel)

    def test_schema_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.ggui.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"schema_version": 99, "project_dimension": "3D"}, f)
            with self.assertRaises(ProjectFormatError):
                read_project(path)


if __name__ == "__main__":
    unittest.main()
