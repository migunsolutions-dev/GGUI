"""Tailing log.blastFoam after Resume must not freeze on a truncated/old log."""
import os
import tempfile
import unittest

from execution_plan import ExecutionIntent, build_execution_plan
from solver_runner import live_log_read_position


class LiveLogReadPositionTests(unittest.TestCase):
    def test_fresh_run_starts_at_beginning(self):
        pos, armed, truncated = live_log_read_position(
            0, 4096, armed=False, skip_existing=False
        )
        self.assertEqual(pos, 0)
        self.assertTrue(armed)
        self.assertFalse(truncated)

    def test_resume_arms_at_eof_and_skips_history(self):
        pos, armed, truncated = live_log_read_position(
            0, 4096, armed=False, skip_existing=True
        )
        self.assertEqual(pos, 4096)
        self.assertTrue(armed)
        self.assertFalse(truncated)

    def test_truncate_rewinds_to_zero(self):
        pos, armed, truncated = live_log_read_position(
            8000, 120, armed=True, skip_existing=True
        )
        self.assertEqual(pos, 0)
        self.assertTrue(armed)
        self.assertTrue(truncated)

    def test_growing_file_keeps_offset(self):
        pos, armed, truncated = live_log_read_position(
            8000, 9000, armed=True, skip_existing=True
        )
        self.assertEqual(pos, 8000)
        self.assertTrue(armed)
        self.assertFalse(truncated)

    def test_equal_size_is_not_a_truncate(self):
        pos, armed, truncated = live_log_read_position(
            120, 120, armed=True, skip_existing=False
        )
        self.assertEqual(pos, 120)
        self.assertTrue(armed)
        self.assertFalse(truncated)


class ResumeTeeAppendTests(unittest.TestCase):
    def _initialized_case(self, root: str) -> None:
        os.makedirs(os.path.join(root, "0"), exist_ok=True)
        os.makedirs(os.path.join(root, "system"), exist_ok=True)
        open(os.path.join(root, "0", "p"), "w").close()

    def test_resume_appends_solver_log(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            os.makedirs(os.path.join(td, "0.1"))
            serial = build_execution_plan(td, 1, ExecutionIntent.RESUME)
            parallel = build_execution_plan(td, 2, ExecutionIntent.RESUME)
            self.assertIn("tee -a log.blastFoam", serial.command)
            self.assertIn("tee -a log.blastFoam", parallel.command)

    def test_fresh_initialized_overwrites_solver_log(self):
        with tempfile.TemporaryDirectory() as td:
            self._initialized_case(td)
            serial = build_execution_plan(
                td, 1, ExecutionIntent.INITIALIZED_SOLVER_RUN
            )
            parallel = build_execution_plan(
                td, 2, ExecutionIntent.INITIALIZED_SOLVER_RUN
            )
            self.assertIn("| tee log.blastFoam", serial.command)
            self.assertIn("| tee log.blastFoam", parallel.command)
            self.assertNotIn("tee -a", serial.command)
            self.assertNotIn("tee -a", parallel.command)


if __name__ == "__main__":
    unittest.main()
