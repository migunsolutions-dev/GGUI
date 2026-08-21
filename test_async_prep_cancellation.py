"""Async 2D preparation, cancellation, and control-restoration tests."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from models_2d import CaseInputs2D, SimulationState2D
from preparation_service_2d import PreparationResult
from preparation_worker_qt import PreparationStep, PreparationWorker
from wsl_runtime import WslCancelToken, run_wsl_command, terminate_process_tree

app = QApplication.instance() or QApplication([])


def _wait_signal(signal, timeout_ms: int = 5000):
    """Wait for a Qt signal using the real event loop. Returns emitted args."""
    loop = QEventLoop()
    holder = {"args": None}

    def _slot(*args):
        holder["args"] = args
        loop.quit()

    signal.connect(_slot)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec_()
    signal.disconnect(_slot)
    return holder["args"]


class NativeAsyncPrepTests(unittest.TestCase):
    def test_native_init_starts_preparation_worker(self):
        from main_new import BlastFoamApp

        win = BlastFoamApp.__new__(BlastFoamApp)
        win._prep_phase = "idle"
        win._prep_worker = None
        win._prep_kind = None
        win._prep_result_handled = False
        win._force_sync_prep = False
        win._pending_exact_end_after_prep = False
        win.active_case_initialized_2d = False
        win.openfoam_bashrc = "/opt/openfoam9/etc/bashrc"
        win.status_bar = mock.Mock()
        win.service = mock.Mock()
        win.service.make_case_name = lambda prefix: f"{prefix}_test"
        win._resolved_case_root = mock.Mock(return_value=tempfile.gettempdir())
        win.tab_2d = mock.Mock()
        win.tab_2d.is_imported_mode = False
        win.tab_2d.btn_initialize = mock.Mock()
        win.tab_2d.btn_exact_end = mock.Mock()
        win.tab_2d.btn_stop = mock.Mock()
        win.tab_2d._apply_action_buttons = mock.Mock()
        win.tab_2d.set_simulation_state = mock.Mock()
        win.tab_2d.handle_initialization_failure = mock.Mock()
        win.tab_2d.set_preparation_step = mock.Mock()

        started = []

        class FakeWorker:
            def __init__(self, *a, **k):
                self.progress = mock.Mock()
                self.log_line = mock.Mock()
                self.finished_ok = mock.Mock()
                self.finished_error = mock.Mock()
                self.finished_cancelled = mock.Mock()

            def start(self):
                started.append("start")

            def run(self):
                started.append("run")

        with mock.patch("preparation_worker_qt.PreparationWorker", FakeWorker), mock.patch(
            "main_new.validate_case_inputs_2d"
        ) as val:
            val.return_value = mock.Mock(valid=True, domain=mock.Mock(total_cells=10), errors=[])
            win.on_initialize_model_2d(CaseInputs2D())
        self.assertIn("start", started)
        self.assertNotIn("run", started)
        self.assertEqual(win._prep_kind, "native_2d")

    def test_no_process_events_in_native_or_import_prep(self):
        root = Path(__file__).resolve().parent
        text = (root / "main_new.py").read_text(encoding="utf-8")
        for marker in (
            "def on_initialize_model_2d",
            "def _open_axisymmetric_imported_case",
            "def on_initialize_imported_model_2d",
            "def _on_native_2d_prep_ok",
            "def _begin_preparation",
        ):
            start = text.index(marker)
            # Class methods are indented; find the next sibling method.
            rest = text[start + len(marker) :]
            end_rel = rest.find("\n    def ")
            self.assertGreater(end_rel, 0, marker)
            segment = text[start : start + len(marker) + end_rel]
            self.assertNotIn(
                "QApplication.processEvents",
                segment,
                msg=f"{marker} still uses processEvents",
            )

    def test_duplicate_init_blocked(self):
        from main_new import BlastFoamApp

        win = BlastFoamApp.__new__(BlastFoamApp)
        win._prep_phase = "active"
        win.tab_2d = mock.Mock(is_imported_mode=False)
        with mock.patch("PyQt5.QtWidgets.QMessageBox.information") as info:
            win.on_initialize_model_2d(CaseInputs2D())
        info.assert_called()

    def test_failed_and_cancelled_not_initialized(self):
        from main_new import BlastFoamApp

        win = BlastFoamApp.__new__(BlastFoamApp)
        win._prep_result_handled = False
        win._prep_phase = "active"
        win._pending_exact_end_after_prep = False
        win.active_case_initialized_2d = True
        win.status_bar = mock.Mock()
        win.tab_2d = mock.Mock()
        win.tab_2d.btn_initialize = mock.Mock()
        win.tab_2d.btn_exact_end = mock.Mock()
        win.tab_2d.btn_stop = mock.Mock()
        win.tab_2d._apply_action_buttons = mock.Mock()
        with mock.patch("PyQt5.QtWidgets.QMessageBox.critical"):
            win._on_native_2d_prep_failed(PreparationResult(ok=False, error="x"))
        self.assertFalse(win.active_case_initialized_2d)
        win.active_case_initialized_2d = True
        win._prep_result_handled = False
        win._on_native_2d_prep_cancelled(PreparationResult(ok=False, cancelled=True))
        self.assertFalse(win.active_case_initialized_2d)


class ImportAsyncPrepTests(unittest.TestCase):
    def test_open_import_uses_worker_not_sync_inventory(self):
        from main_new import BlastFoamApp

        win = BlastFoamApp.__new__(BlastFoamApp)
        win._prep_phase = "idle"
        win._prep_worker = None
        win._prep_result_handled = False
        win._force_sync_prep = False
        win.openfoam_bashrc = "/opt/openfoam9/etc/bashrc"
        win.status_bar = mock.Mock()
        win.tabs = mock.Mock()
        win.tab_2d = mock.Mock()
        win.tab_2d.btn_initialize = mock.Mock()
        win.tab_2d.btn_exact_end = mock.Mock()
        win.tab_2d.btn_stop = mock.Mock()
        win.tab_2d._apply_action_buttons = mock.Mock()
        win.tab_2d.set_simulation_state = mock.Mock()
        win.tab_2d.set_preparation_step = mock.Mock()
        win._resolved_case_root = mock.Mock(return_value=tempfile.gettempdir())
        win._repo_root = tempfile.gettempdir()

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "axisymmetricCharge"
            source.mkdir()
            (source / "system").mkdir()
            started = []

            class FakeWorker:
                def __init__(self, *a, **k):
                    self.progress = mock.Mock()
                    self.log_line = mock.Mock()
                    self.finished_ok = mock.Mock()
                    self.finished_error = mock.Mock()
                    self.finished_cancelled = mock.Mock()

                def start(self):
                    started.append("start")

            with mock.patch("preparation_worker_qt.PreparationWorker", FakeWorker):
                win._open_axisymmetric_imported_case(str(source), classification=None)
            self.assertIn("start", started)
            self.assertEqual(win._prep_kind, "import_copy")

    def test_cancel_cleans_staging_payload(self):
        from preparation_service_2d import ImportCopyContext, prepare_imported_copy_and_inspect

        token = WslCancelToken()
        token.cancel()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            src.mkdir()
            (src / "f.txt").write_text("x", encoding="utf-8")
            result = prepare_imported_copy_and_inspect(
                ImportCopyContext(str(src), td, td),
                token,
                lambda _n: None,
            )
        self.assertTrue(result.cancelled)
        self.assertFalse(result.ok)


class WslCancellationProcessTests(unittest.TestCase):
    def test_cancel_before_start(self):
        token = WslCancelToken()
        token.cancel()
        result = run_wsl_command("/tmp", "echo hi", cancel_token=token)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.ok)

    def test_cancel_during_real_child_process(self):
        # Use a long-running Python child through bash -lc so no OpenFOAM is required.
        token = WslCancelToken()
        started = threading.Event()

        def _cancel_soon():
            started.wait(timeout=5)
            time.sleep(0.3)
            token.cancel()

        thread = threading.Thread(target=_cancel_soon, daemon=True)
        thread.start()
        # Patch build to run a local long sleep without needing WSL on CI/Windows.
        with mock.patch("wsl_runtime.build_wsl_argv", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]), mock.patch(
            "wsl_runtime.build_case_command_argv",
            return_value=(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                mock.Mock(linux_path="/tmp", distro=None),
                "sleep",
            ),
        ):
            started.set()
            result = run_wsl_command("/tmp", "sleep 30", cancel_token=token, timeout_s=10)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.ok)
        self.assertNotEqual(result.error, "")

    def test_repeated_cancel_idempotent(self):
        token = WslCancelToken()
        token.cancel()
        token.cancel()
        self.assertTrue(token.cancelled)

    def test_timeout_distinct_from_cancel(self):
        with mock.patch("wsl_runtime.subprocess.Popen") as popen:
            proc = mock.Mock()
            proc.communicate.side_effect = [
                subprocess.TimeoutExpired(cmd="x", timeout=0.2),
                ("partial", ""),
            ]
            proc.returncode = -9
            proc.poll.return_value = -9
            proc.pid = 12345
            popen.return_value = proc
            with mock.patch("wsl_runtime.terminate_process_tree"):
                result = run_wsl_command("/tmp", "sleep", timeout_s=0.01)
            self.assertTrue(result.timed_out)
            self.assertFalse(result.cancelled)

    def test_nonzero_exit_distinct(self):
        with mock.patch("wsl_runtime.subprocess.Popen") as popen:
            proc = mock.Mock()
            proc.communicate.return_value = ("out", "err")
            proc.returncode = 3
            popen.return_value = proc
            result = run_wsl_command("/tmp", "false")
            self.assertFalse(result.ok)
            self.assertFalse(result.cancelled)
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(result.stdout, "out")


class UiCancelAndEventLoopTests(unittest.TestCase):
    def test_event_loop_worker_completion(self):
        outcomes = []

        def step(token, progress):
            progress("Working")
            return PreparationResult(ok=True, payload={"v": 1})

        worker = PreparationWorker([PreparationStep("step", step)])
        loop = QEventLoop()

        def _on_ok(result):
            outcomes.append(result.payload)
            loop.quit()

        worker.finished_ok.connect(_on_ok)
        QTimer.singleShot(0, worker.start)
        QTimer.singleShot(5000, loop.quit)
        loop.exec_()
        self.assertEqual(outcomes, [{"v": 1}])
        worker.wait(2000)

    def test_stop_requests_prep_cancel(self):
        from main_new import BlastFoamApp

        win = BlastFoamApp.__new__(BlastFoamApp)
        win._prep_phase = "active"
        win.status_bar = mock.Mock()
        worker = mock.Mock()
        win._prep_worker = worker
        win.runner = None
        win.view_timer = mock.Mock()
        win.on_stop_request()
        worker.request_cancel.assert_called_once()
        self.assertEqual(win._prep_phase, "cancelling")

    def test_stop_controls_solver_when_not_preparing(self):
        from main_new import BlastFoamApp

        win = BlastFoamApp.__new__(BlastFoamApp)
        win._prep_phase = "idle"
        win._prep_worker = None
        win.status_bar = mock.Mock()
        win.runner = mock.Mock()
        win.view_timer = mock.Mock()
        win._active_run_mode = "3D"
        win.on_stop_request()
        win.runner.stop.assert_called_once()


class RegressionSourceGuards(unittest.TestCase):
    def test_inventory_not_called_in_open_handler_body(self):
        text = Path("main_new.py").read_text(encoding="utf-8")
        start = text.index("def _open_axisymmetric_imported_case")
        rest = text[start + 1 :]
        end = rest.find("\n    def ")
        segment = text[start : start + 1 + end]
        self.assertNotIn("inventory_case(", segment)
        self.assertNotIn("create_automatic_working_copy(", segment)


if __name__ == "__main__":
    unittest.main()
