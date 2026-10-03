from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from result_storage import (
    ResultStoragePolicy,
    cleanup_native_time_folders,
    ensure_remap_snapshot,
    run_reached_configured_end,
    solver_run_succeeded,
)


def _mkdir(root: str, relative: str) -> str:
    path = os.path.join(root, relative)
    os.makedirs(path, exist_ok=True)
    return path


class ResultStorageTests(unittest.TestCase):
    def test_retention_on_keeps_native_and_parallel_times(self):
        with tempfile.TemporaryDirectory() as case:
            _mkdir(case, "0")
            _mkdir(case, "0.1")
            _mkdir(case, "processor0/0.1")
            report = cleanup_native_time_folders(
                case, ResultStoragePolicy(keep_openfoam_time_folders=True)
            )
            self.assertTrue(os.path.isdir(os.path.join(case, "0.1")))
            self.assertTrue(os.path.isdir(os.path.join(case, "processor0", "0.1")))
            self.assertEqual(report.removed, [])

    def test_retention_off_removes_native_history_after_selected_outputs(self):
        with tempfile.TemporaryDirectory() as case:
            for name in ("0", "0.1", "0.2", "postProcessing/probes2d/0", "VTK"):
                _mkdir(case, name)
            _mkdir(case, "processor0/0.2")
            report = cleanup_native_time_folders(case, ResultStoragePolicy())
            self.assertTrue(os.path.isdir(os.path.join(case, "0")))
            self.assertFalse(os.path.exists(os.path.join(case, "0.1")))
            self.assertFalse(os.path.exists(os.path.join(case, "0.2")))
            self.assertFalse(os.path.exists(os.path.join(case, "processor0")))
            self.assertTrue(os.path.isdir(os.path.join(case, "VTK")))
            self.assertTrue(
                os.path.isdir(os.path.join(case, "postProcessing", "probes2d", "0"))
            )
            self.assertFalse(report.failures)

    def test_remap_keeps_exact_latest_native_snapshot_only(self):
        with tempfile.TemporaryDirectory() as case:
            for name in ("0", "0.1", "0.2", "processor0/0.2"):
                _mkdir(case, name)
            for field in ("p", "rho", "U", "T", "alpha.c4"):
                open(os.path.join(case, "0.2", field), "w").close()
            self.assertTrue(ensure_remap_snapshot(case))
            report = cleanup_native_time_folders(
                case, ResultStoragePolicy(preserve_remap_data=True)
            )
            self.assertFalse(os.path.exists(os.path.join(case, "0.1")))
            self.assertTrue(os.path.isdir(os.path.join(case, "0.2")))
            self.assertFalse(os.path.exists(os.path.join(case, "processor0")))
            self.assertIn("0.2", report.preserved)

    def test_parallel_cleanup_aborts_if_remap_has_no_serial_result(self):
        with tempfile.TemporaryDirectory() as case:
            _mkdir(case, "0")
            _mkdir(case, "processor0/0.2")
            report = cleanup_native_time_folders(
                case, ResultStoragePolicy(preserve_remap_data=True)
            )
            self.assertTrue(os.path.isdir(os.path.join(case, "processor0", "0.2")))
            self.assertIn("no reconstructed serial time", report.skipped_reason)

    def test_parallel_cleanup_preserves_unknown_processor_outputs(self):
        with tempfile.TemporaryDirectory() as case:
            _mkdir(case, "0")
            _mkdir(case, "0.1")
            _mkdir(case, "processor0/0.1")
            custom = _mkdir(case, "processor0/customOutput")
            with open(os.path.join(custom, "keep.txt"), "w", encoding="utf-8") as stream:
                stream.write("keep")
            report = cleanup_native_time_folders(case, ResultStoragePolicy())
            self.assertFalse(os.path.exists(os.path.join(case, "processor0", "0.1")))
            self.assertTrue(
                os.path.isfile(
                    os.path.join(case, "processor0", "customOutput", "keep.txt")
                )
            )
            self.assertIn("processor0", report.preserved)

    def test_selected_vtk_command_uses_only_requested_fields(self):
        policy = ResultStoragePolicy(
            vtk_fields=("p", "rho", "p", "bad field"), terminal_run=True
        )
        self.assertEqual(
            policy.foam_to_vtk_command(),
            "foamToVTK -fields '(p rho)' > log.foamToVTK 2>&1",
        )

    def test_non_terminal_exact_one_does_not_finalize_outputs(self):
        policy = ResultStoragePolicy(vtk_fields=("p",), terminal_run=False)
        self.assertFalse(policy.needs_serial_results)
        self.assertEqual(policy.foam_to_vtk_command(), "")

    def test_cleanup_eligibility_requires_configured_end_time(self):
        with tempfile.TemporaryDirectory() as case:
            _mkdir(case, "system")
            with open(
                os.path.join(case, "system", "controlDict"), "w", encoding="utf-8"
            ) as stream:
                stream.write("endTime 0.2;\n")
            _mkdir(case, "0.1")
            self.assertFalse(run_reached_configured_end(case))
            _mkdir(case, "0.2")
            self.assertTrue(run_reached_configured_end(case))

    def test_1d_without_completion_json_is_never_done(self):
        with tempfile.TemporaryDirectory() as case:
            _mkdir(case, "system")
            with open(
                os.path.join(case, "system", "controlDict"), "w", encoding="utf-8"
            ) as stream:
                stream.write(
                    "endTime         0.03;\n"
                    "functions\n{\n    probes1d { writeInterval 25; }\n}\n"
                )
            with open(os.path.join(case, ".watchdog_target_radius"), "w") as stream:
                stream.write("1.0\n")
            with open(os.path.join(case, "log.blastFoam"), "w", encoding="utf-8") as stream:
                stream.write("Time = 0.005776\nEnd\n")
            self.assertFalse(solver_run_succeeded(case, 0))
            with open(os.path.join(case, "log.blastFoam"), "w", encoding="utf-8") as stream:
                stream.write("Time = 0.0299999\nEnd\n")
            self.assertFalse(solver_run_succeeded(case, 0))
            with open(os.path.join(case, "log.blastFoam"), "w", encoding="utf-8") as stream:
                stream.write("Time = 0.03\nEnd\n")
            self.assertFalse(solver_run_succeeded(case, 0))
            self.assertFalse(solver_run_succeeded(case, 0, user_stopped=True))
            with open(os.path.join(case, "log.blastFoam"), "w", encoding="utf-8") as stream:
                stream.write(
                    "--> FOAM FATAL ERROR: \n"
                    "No cells will be activated using the detonation point (0.0105 0 0)\n"
                )
            self.assertFalse(solver_run_succeeded(case, 0))
            self.assertFalse(solver_run_succeeded(case, 1))

    def test_1d_terminate_mode_is_not_done_without_verified_arrival(self):
        from completion_1d import (
            RUN_MODE_TERMINATE,
            finalize_completion_record,
            initial_completion_record,
            write_completion_record,
        )

        with tempfile.TemporaryDirectory() as case:
            _mkdir(case, "system")
            with open(
                os.path.join(case, "system", "controlDict"), "w", encoding="utf-8"
            ) as stream:
                stream.write(
                    "endTime         0.03;\n"
                    "functions\n{\n    probes1d { writeInterval 25; }\n}\n"
                )
            write_completion_record(
                case,
                initial_completion_record(
                    mode=RUN_MODE_TERMINATE,
                    requested_stop_radius_m=1.0,
                ),
            )
            with open(os.path.join(case, "log.blastFoam"), "w", encoding="utf-8") as stream:
                stream.write("Time = 0.005776\nEnd\n")
            finalize_completion_record(
                case,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.005776,
                reached_end_time=False,
            )
            self.assertFalse(solver_run_succeeded(case, 0))

    def test_vtk_export_must_materialize_durable_directory(self):
        from solver_runner import SolverRunner

        with tempfile.TemporaryDirectory() as case:
            runner = SolverRunner(
                case,
                result_storage_policy=ResultStoragePolicy(
                    vtk_fields=("p",), terminal_run=True
                ),
            )
            with mock.patch.object(runner, "_build_wsl_cmd", return_value=["true"]), mock.patch(
                "solver_runner.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ):
                self.assertEqual(runner._run_result_export(), 1)
                _mkdir(case, "VTK")
                self.assertEqual(runner._run_result_export(), 0)


    def test_normal_last_step_counts_as_end_time(self):
        from result_storage import case_reached_configured_end, logged_time_reached_end

        with tempfile.TemporaryDirectory() as case:
            _mkdir(case, "system")
            with open(os.path.join(case, "system", "controlDict"), "w", encoding="utf-8") as stream:
                stream.write("endTime         0.0001;\n")
            with open(os.path.join(case, "log.blastFoam"), "w", encoding="utf-8") as stream:
                stream.write("deltaT = 8.07118e-07\nTime = 9.96146e-05\nEnd\n")
            self.assertTrue(case_reached_configured_end(case))
            self.assertTrue(logged_time_reached_end(case))
            self.assertTrue(run_reached_configured_end(case))

    def test_logged_solver_progress_reads_the_final_step(self):
        from result_storage import logged_solver_progress

        with tempfile.TemporaryDirectory() as case:
            with open(os.path.join(case, "log.blastFoam"), "w", encoding="utf-8") as stream:
                stream.write("Time = 0.001\ndeltaT = 1e-06\nTime = 0.002\ndeltaT = 2e-07\n")
            step, last_t, last_dt = logged_solver_progress(case)
            self.assertEqual(step, 2)
            self.assertAlmostEqual(last_t, 0.002)
            self.assertAlmostEqual(last_dt, 2e-07)

    def test_early_stop_is_not_end_time(self):
        from result_storage import case_reached_configured_end

        with tempfile.TemporaryDirectory() as case:
            _mkdir(case, "system")
            with open(os.path.join(case, "system", "controlDict"), "w", encoding="utf-8") as stream:
                stream.write("endTime         0.002;\n")
            with open(os.path.join(case, "log.blastFoam"), "w", encoding="utf-8") as stream:
                stream.write("deltaT = 1.0e-06\nTime = 0.001\n")
            self.assertFalse(case_reached_configured_end(case))


if __name__ == "__main__":
    unittest.main()
