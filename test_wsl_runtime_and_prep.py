"""Checkpoint 2: centralized WSL runtime and preparation worker."""
from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from execution_plan import ExecutionIntent, build_execution_plan
from preparation_worker_qt import PreparationResult, PreparationStep, PreparationWorker
from wsl_runtime import (
    WslCancelToken,
    build_case_command_argv,
    build_wsl_argv,
    quote_shell,
    run_wsl_command,
    to_wsl_path_and_distro,
    win_to_wsl_path,
)

app = QApplication.instance() or QApplication([])


def _module_imports_pyqt(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("PyQt"):
                    return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("PyQt"):
                return True
    return False


class WslRuntimePathTests(unittest.TestCase):
    def test_windows_drive_path(self):
        path = to_wsl_path_and_distro(r"C:\Users\migun\case")
        self.assertIsNone(path.distro)
        self.assertEqual(path.linux_path, "/mnt/c/Users/migun/case")

    def test_path_with_spaces(self):
        path = to_wsl_path_and_distro(r"C:\Users\migun\My Cases\blast")
        self.assertIn("My Cases", path.linux_path)
        argv, _, safe = build_case_command_argv(
            r"C:\Users\migun\My Cases\blast", "blockMesh"
        )
        self.assertTrue(any("My Cases" in part or "My\\ Cases" in part or "'/mnt" in part for part in argv) or "My Cases" in safe)
        self.assertIn(quote_shell("/mnt/c/Users/migun/My Cases/blast"), " ".join(argv))

    def test_hebrew_characters(self):
        path = to_wsl_path_and_distro(r"C:\Users\migun\תיקיה\case")
        self.assertIn("תיקיה", path.linux_path)

    def test_single_quotes_and_parentheses(self):
        quoted = quote_shell("it's (test) & more")
        self.assertNotIn("it's (test) & more", [quoted])  # must be escaped
        self.assertTrue(quoted.startswith("'") or "\\" in quoted)

    def test_ampersand_quoted(self):
        self.assertNotEqual(quote_shell("a&b"), "a&b")

    def test_wsl_unc_path(self):
        path = to_wsl_path_and_distro(
            r"\\wsl.localhost\Ubuntu-20.04\home\naor\case"
        )
        self.assertEqual(path.distro, "Ubuntu-20.04")
        self.assertEqual(path.linux_path, "/home/naor/case")

    def test_plain_linux_path(self):
        path = to_wsl_path_and_distro("/home/naor/case")
        self.assertIsNone(path.distro)
        self.assertEqual(path.linux_path, "/home/naor/case")

    def test_bashrc_and_case_quoted_in_argv(self):
        argv, _, _ = build_case_command_argv(
            r"C:\Users\migun\case dir",
            "blockMesh -dict system/blockMeshDict",
            openfoam_bashrc="/opt/openfoam9/etc/bashrc",
        )
        script = argv[-1]
        self.assertIn(quote_shell("/opt/openfoam9/etc/bashrc"), script)
        self.assertIn(quote_shell("/mnt/c/Users/migun/case dir"), script)
        self.assertIn("blockMesh -dict system/blockMeshDict", script)

    def test_win_to_wsl_path_wrapper(self):
        self.assertEqual(
            win_to_wsl_path(r"D:\data\case"),
            "/mnt/d/data/case",
        )


class WslRuntimeExecutionTests(unittest.TestCase):
    def test_timeout_result(self):
        with mock.patch("wsl_runtime.subprocess.Popen") as popen:
            proc = mock.Mock()
            proc.communicate.side_effect = [
                __import__("subprocess").TimeoutExpired(cmd="x", timeout=0.2),
                ("", ""),
            ]
            proc.returncode = -9
            popen.return_value = proc
            result = run_wsl_command("/tmp/case", "sleep 10", timeout_s=0.01)
            self.assertTrue(result.timed_out)
            self.assertFalse(result.ok)

    def test_cancellation_result(self):
        token = WslCancelToken()
        token.cancel()
        result = run_wsl_command("/tmp/case", "blockMesh", cancel_token=token)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.ok)

    def test_nonzero_exit_and_capture(self):
        with mock.patch("wsl_runtime.subprocess.Popen") as popen:
            proc = mock.Mock()
            proc.communicate.return_value = ("out-text", "err-text")
            proc.returncode = 7
            popen.return_value = proc
            result = run_wsl_command("/tmp/case", "false")
            self.assertFalse(result.ok)
            self.assertEqual(result.exit_code, 7)
            self.assertEqual(result.stdout, "out-text")
            self.assertEqual(result.stderr, "err-text")

    def test_argument_preservation_in_safe_command(self):
        _, _, safe = build_case_command_argv(
            "/home/case",
            "blockMesh -dict system/blockMeshDict.custom",
        )
        self.assertIn("-dict system/blockMeshDict.custom", safe)


class PureModuleImportTests(unittest.TestCase):
    def test_execution_plan_has_no_pyqt(self):
        path = Path(__file__).resolve().parent / "execution_plan.py"
        self.assertFalse(_module_imports_pyqt(path))

    def test_wsl_runtime_has_no_pyqt(self):
        path = Path(__file__).resolve().parent / "wsl_runtime.py"
        self.assertFalse(_module_imports_pyqt(path))

    def test_execution_plan_still_builds(self):
        with tempfile.TemporaryDirectory() as td:
            plan = build_execution_plan(td, 1, ExecutionIntent.FRESH_FULL_PIPELINE)
            self.assertEqual(plan.command, "bash ./Allrun")


class PreparationWorkerTests(unittest.TestCase):
    def test_success_failure_cancellation_and_buttons(self):
        outcomes = []

        def ok_step(token):
            return PreparationResult(ok=True, payload={"v": 1})

        def fail_step(token):
            return PreparationResult(ok=False, error="boom")

        def cancel_step(token):
            token.cancel()
            return PreparationResult(ok=False, cancelled=True, error="Cancelled")

        worker = PreparationWorker([PreparationStep("ok", ok_step)])
        worker.finished_ok.connect(lambda r: outcomes.append(("ok", r.ok)))
        worker.run()
        self.assertEqual(outcomes[-1], ("ok", True))

        worker = PreparationWorker([PreparationStep("fail", fail_step)])
        worker.finished_error.connect(lambda r: outcomes.append(("err", r.error)))
        worker.run()
        self.assertEqual(outcomes[-1], ("err", "boom"))

        worker = PreparationWorker([PreparationStep("cancel", cancel_step)])
        worker.finished_cancelled.connect(lambda r: outcomes.append(("cancel", r.cancelled)))
        worker.run()
        self.assertEqual(outcomes[-1], ("cancel", True))

    def test_failed_prep_does_not_mark_initialized(self):
        from main_new import BlastFoamApp
        from tab_2d import Tab2D

        win = BlastFoamApp.__new__(BlastFoamApp)
        win.tab_2d = Tab2D()
        win.status_bar = mock.Mock()
        win.active_case_initialized_2d = False
        win._set_preparation_controls_enabled = mock.Mock()
        result = PreparationResult(ok=False, error="fail")
        with mock.patch("PyQt5.QtWidgets.QMessageBox.critical"):
            win._on_imported_2d_prep_failed(result)
        self.assertFalse(win.active_case_initialized_2d)

    def test_cancelled_prep_does_not_mark_initialized(self):
        from main_new import BlastFoamApp
        from tab_2d import Tab2D

        win = BlastFoamApp.__new__(BlastFoamApp)
        win.tab_2d = Tab2D()
        win.status_bar = mock.Mock()
        win.active_case_initialized_2d = True  # should be cleared
        win._set_preparation_controls_enabled = mock.Mock()
        win._on_imported_2d_prep_cancelled(PreparationResult(ok=False, cancelled=True))
        self.assertFalse(win.active_case_initialized_2d)


if __name__ == "__main__":
    unittest.main()
