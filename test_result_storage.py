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


if __name__ == "__main__":
    unittest.main()
