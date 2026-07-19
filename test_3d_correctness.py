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
    FINAL_RECONSTRUCT_CMD,
    ExecutionIntent,
    ExecutionPreparationError,
    build_execution_plan,
)
from startup_capture_guard import UNSAFE_CAPTURE_MESSAGE, evaluate_unsafe_capture, require_safe_capture


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

    def _assert_non_destructive(self, command: str) -> None:
        self.assertNotIn("rm -rf", command)
        self.assertNotIn("Allclean", command)
        self.assertNotIn("Allrun", command)

    def test_parallel_resume_redecomposes_newer_serial_state(self):
        """Serial 0.3 ahead of processor 0.2 must re-decompose; latest_time is 0.3."""
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "0.3"))
            os.makedirs(os.path.join(td, "processor0", "0.2"))
            os.makedirs(os.path.join(td, "processor1", "0.2"))
            plan = build_execution_plan(td, 2, ExecutionIntent.RESUME)
            self.assertEqual(plan.latest_time, 0.3)
            self.assertIn("decomposePar -force -latestTime", plan.command)
            self.assertIn("mpirun -np 2", plan.command)
            self._assert_non_destructive(plan.command)

    def test_parallel_resume_uses_newer_processor_state(self):
        """Processor 0.3 ahead of serial 0.2 must reuse processors; latest_time is 0.3."""
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "0.2"))
            os.makedirs(os.path.join(td, "processor0", "0.3"))
            os.makedirs(os.path.join(td, "processor1", "0.3"))
            plan = build_execution_plan(td, 2, ExecutionIntent.RESUME)
            self.assertEqual(plan.latest_time, 0.3)
            self.assertNotIn("decomposePar", plan.command)
            self.assertIn("mpirun -np 2", plan.command)
            self._assert_non_destructive(plan.command)

    def test_parallel_resume_equal_times_reuse_processors(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "0.3"))
            os.makedirs(os.path.join(td, "processor0", "0.3"))
            os.makedirs(os.path.join(td, "processor1", "0.3"))
            plan = build_execution_plan(td, 2, ExecutionIntent.RESUME)
            self.assertEqual(plan.latest_time, 0.3)
            self.assertNotIn("decomposePar", plan.command)
            self._assert_non_destructive(plan.command)

    def test_one_step_resume_selects_actual_serial_source(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "0.3"))
            os.makedirs(os.path.join(td, "processor0", "0.2"))
            os.makedirs(os.path.join(td, "processor1", "0.2"))
            plan = build_execution_plan(td, 2, ExecutionIntent.ONE_STEP_RESUME)
            self.assertEqual(plan.latest_time, 0.3)
            self.assertIn("decomposePar -force -latestTime", plan.command)
            self._assert_non_destructive(plan.command)

    def test_one_step_resume_selects_actual_processor_source(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "0.2"))
            os.makedirs(os.path.join(td, "processor0", "0.3"))
            os.makedirs(os.path.join(td, "processor1", "0.3"))
            plan = build_execution_plan(td, 2, ExecutionIntent.ONE_STEP_RESUME)
            self.assertEqual(plan.latest_time, 0.3)
            self.assertNotIn("decomposePar", plan.command)
            self._assert_non_destructive(plan.command)

    def test_serial_resume_latest_time_matches_reconstructed_source(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "0.2"))
            os.makedirs(os.path.join(td, "processor0", "0.3"))
            os.makedirs(os.path.join(td, "processor1", "0.3"))
            plan = build_execution_plan(td, 1, ExecutionIntent.RESUME)
            self.assertEqual(plan.latest_time, 0.3)
            self.assertIn("reconstructPar -latestTime", plan.command)
            self._assert_non_destructive(plan.command)

    def test_final_reconstruct_command_is_latest_time(self):
        self.assertIn("reconstructPar -latestTime", FINAL_RECONSTRUCT_CMD)
        self.assertNotIn("rm -rf", FINAL_RECONSTRUCT_CMD)
        self.assertNotIn("Allclean", FINAL_RECONSTRUCT_CMD)

    def test_parallel_success_runs_final_reconstruct(self):
        """Successful parallel SolverRunner completion must invoke final reconstructPar."""
        from unittest.mock import MagicMock, patch
        from solver_runner import SolverRunner

        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "0.1"))
            os.makedirs(os.path.join(td, "processor0", "0.1"))
            os.makedirs(os.path.join(td, "processor1", "0.1"))
            runner = SolverRunner(
                win_case_dir=td,
                openfoam_bashrc="/opt/openfoam9/etc/bashrc",
                project_root=td,
                cores=2,
                intent=ExecutionIntent.RESUME,
            )
            poll_returns = [None, 0, 0]

            def poll_side_effect():
                return poll_returns.pop(0) if poll_returns else 0

            mock_proc = MagicMock()
            mock_proc.poll.side_effect = poll_side_effect
            final_mock = MagicMock(return_value=0)
            finished = []

            with patch.object(runner, "_build_wsl_cmd", return_value=["true"]), \
                 patch("solver_runner.subprocess.Popen", return_value=mock_proc), \
                 patch.object(runner, "_final_reconstruct_latest", final_mock), \
                 patch.object(runner, "_maybe_reconstruct_new_times"), \
                 patch.object(runner, "_check_watchdog_trigger"), \
                 patch.object(runner, "_maybe_stop_after_watchdog"), \
                 patch.object(runner, "_discover_probe_file", return_value=None), \
                 patch.object(runner, "_find_control_dict_end_time"), \
                 patch.object(runner, "finished_signal") as fin_sig, \
                 patch.object(runner, "status_signal"), \
                 patch.object(runner, "progress_signal"), \
                 patch.object(runner, "data_signal"):
                fin_sig.emit.side_effect = lambda ok: finished.append(ok)
                runner.run()

            final_mock.assert_called_once()
            self.assertEqual(finished, [True])


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
        with self.assertRaises(ValueError) as ctx:
            require_safe_capture(inputs)
        self.assertIn("Initialization is blocked", str(ctx.exception))

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
        require_safe_capture(inputs)  # must not raise

    def test_seed_or_band_protects_all_shapes(self):
        for shape in ("Sphere", "Cylinder", "Cuboid"):
            base = case(charge_shape=shape, mass_kg=0.001)
            self.assertTrue(evaluate_unsafe_capture(replace(base, charge_refinement_level=2)).safe)
            self.assertTrue(evaluate_unsafe_capture(replace(base, charge_outer_refine_enable=True)).safe)

    def test_generator_blocks_unsafe_before_writing_case(self):
        inputs = case(
            min_point=(0, 0, 0),
            max_point=(1, 1, 1),
            cell_size=0.2,
            charge_center=(0.4, 0.4, 0.4),
            mass_kg=0.001,
            charge_outer_refine_enable=False,
            charge_refinement_level=0,
        )
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                Generator3D(td).generate("unsafe", inputs)
            self.assertEqual(str(ctx.exception), UNSAFE_CAPTURE_MESSAGE)
            self.assertFalse(os.path.isdir(os.path.join(td, "unsafe")))

    def test_generator_allows_safe_centre_inside(self):
        inputs = case(
            min_point=(0, 0, 0),
            max_point=(1, 1, 1),
            cell_size=0.2,
            charge_center=(0.5, 0.5, 0.5),
            mass_kg=0.02,
            charge_outer_refine_enable=False,
            charge_refinement_level=0,
        )
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("safe", inputs)
            self.assertTrue(os.path.isdir(case_dir))

    def test_generator_remap_bypasses_capture_guard(self):
        inputs = case(
            min_point=(0, 0, 0),
            max_point=(1, 1, 1),
            cell_size=0.2,
            charge_center=(0.4, 0.4, 0.4),
            mass_kg=0.001,
            charge_outer_refine_enable=False,
            charge_refinement_level=0,
            remap_enabled=True,
            remap_case_path="",
        )
        require_safe_capture(inputs)  # remap must not raise
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("remap", inputs)
            self.assertTrue(os.path.isdir(case_dir))


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
