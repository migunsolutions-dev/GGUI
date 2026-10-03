"""Parallel interrupt must reconstruct case-root times after the solver exits."""
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from solver_runner import SolverRunner


class _Done:
    def __init__(self):
        self.returncode = 0

    def poll(self):
        return 0


class ParallelInterruptReconstructTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_stop_flag_blocks_reconstruct_until_the_solver_has_exited(self):
        with tempfile.TemporaryDirectory() as root:
            runner = SolverRunner(root, cores=2)
            runner._stop_requested = True
            with mock.patch("solver_runner.subprocess.Popen") as popen:
                code = runner._final_reconstruct_latest()
            self.assertEqual(code, 1)
            popen.assert_not_called()

    def test_interrupt_reconstruct_runs_after_stop_was_requested(self):
        with tempfile.TemporaryDirectory() as root:
            runner = SolverRunner(root, cores=2)
            runner._stop_requested = True
            with mock.patch("solver_runner.subprocess.Popen", return_value=_Done()) as popen:
                code = runner._final_reconstruct_latest(allow_after_stop=True)
            self.assertEqual(code, 0)
            command = popen.call_args.args[0]
            joined = " ".join(command)
            self.assertIn("reconstructPar -latestTime", joined)
            self.assertNotIn("reconstructParMesh", joined)

    def test_adaptive_mesh_merges_the_refined_mesh_before_fields(self):
        with tempfile.TemporaryDirectory() as root:
            system = os.path.join(root, "constant")
            os.makedirs(system)
            with open(os.path.join(system, "dynamicMeshDict"), "w", encoding="utf-8") as handle:
                handle.write("dynamicFvMesh adaptiveFvMesh;\n")
            runner = SolverRunner(root, cores=2)
            runner._stop_requested = True
            with mock.patch("solver_runner.subprocess.Popen", return_value=_Done()) as popen:
                code = runner._final_reconstruct_latest(allow_after_stop=True)
            self.assertEqual(code, 0)
            joined = " ".join(popen.call_args.args[0])
            self.assertIn("reconstructParMesh -latestTime", joined)
            self.assertIn("reconstructPar -latestTime", joined)
            self.assertNotIn("-newTimes", joined)

    def test_live_adaptive_reconstruct_pins_one_processor_time(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "constant"))
            with open(os.path.join(root, "constant", "dynamicMeshDict"), "w", encoding="utf-8") as handle:
                handle.write("dynamicFvMesh adaptiveFvMesh;\n")
            proc = os.path.join(root, "processor0", "0.00048", "uniform")
            os.makedirs(proc)
            with open(os.path.join(proc, "time"), "w", encoding="utf-8") as handle:
                handle.write("0.00048\n")
            runner = SolverRunner(root, cores=2)
            with mock.patch("solver_runner.subprocess.Popen", return_value=_Done()) as popen:
                runner._maybe_reconstruct_new_times()
            joined = " ".join(popen.call_args.args[0])
            self.assertIn("reconstructParMesh -time 0.00048", joined)
            self.assertIn("reconstructPar -time 0.00048", joined)
            self.assertNotIn("-latestTime", joined)


if __name__ == "__main__":
    unittest.main()
