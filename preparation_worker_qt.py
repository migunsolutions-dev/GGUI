"""Qt worker for long-running case preparation (separate from SolverRunner)."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence

from PyQt5.QtCore import QThread, pyqtSignal

from preparation_service_2d import PreparationResult
from wsl_runtime import WslCancelToken, WslRunResult, run_wsl_command


class PrepWorkerPhase(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    CANCELLING = "cancelling"
    FINISHED = "finished"


@dataclass
class PreparationStep:
    """One named preparation step executed by the worker."""

    name: str
    # Either a pure callable or a WSL shell command string.
    action: Any
    kind: str = "callable"  # "callable" | "wsl"


class PreparationWorker(QThread):
    """Run generation / mesh utilities / inventory off the GUI thread."""

    progress = pyqtSignal(str)
    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(object)  # PreparationResult
    finished_error = pyqtSignal(object)  # PreparationResult
    finished_cancelled = pyqtSignal(object)  # PreparationResult

    def __init__(
        self,
        steps: Sequence[PreparationStep],
        *,
        case_dir: str = "",
        openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc",
        parent=None,
    ):
        super().__init__(parent)
        self.steps = list(steps)
        self.case_dir = case_dir
        self.openfoam_bashrc = openfoam_bashrc
        self._cancel = WslCancelToken()
        self._payload: Any = None
        self._phase = PrepWorkerPhase.STARTING
        self._finished_emitted = False

    @property
    def cancel_token(self) -> WslCancelToken:
        return self._cancel

    @property
    def phase(self) -> PrepWorkerPhase:
        return self._phase

    def request_cancel(self) -> None:
        self._phase = PrepWorkerPhase.CANCELLING
        self._cancel.cancel()

    def set_payload(self, payload: Any) -> None:
        self._payload = payload

    def _emit_once(self, signal, result: PreparationResult) -> None:
        if self._finished_emitted:
            return
        self._finished_emitted = True
        self._phase = PrepWorkerPhase.FINISHED
        signal.emit(result)

    def _progress(self, name: str) -> None:
        self.progress.emit(name)
        self.log_line.emit(f"[Prepare] {name}")

    def run(self) -> None:
        self._phase = PrepWorkerPhase.ACTIVE
        step_results: List[WslRunResult] = []
        try:
            for step in self.steps:
                if self._cancel.cancelled:
                    result = PreparationResult(
                        ok=False,
                        cancelled=True,
                        failed_step=step.name,
                        error="Cancelled",
                        step_results=step_results,
                        payload=self._payload,
                    )
                    self._emit_once(self.finished_cancelled, result)
                    return
                self._progress(step.name)
                if step.kind == "wsl":
                    wsl_result = run_wsl_command(
                        self.case_dir,
                        str(step.action),
                        openfoam_bashrc=self.openfoam_bashrc,
                        cancel_token=self._cancel,
                    )
                    step_results.append(wsl_result)
                    if wsl_result.stdout:
                        for line in wsl_result.stdout.splitlines()[-40:]:
                            self.log_line.emit(line)
                    if wsl_result.stderr:
                        for line in wsl_result.stderr.splitlines()[-20:]:
                            self.log_line.emit(line)
                    if wsl_result.cancelled:
                        result = PreparationResult(
                            ok=False,
                            cancelled=True,
                            failed_step=step.name,
                            error="Cancelled",
                            step_results=step_results,
                            payload=self._payload,
                        )
                        self._emit_once(self.finished_cancelled, result)
                        return
                    if not wsl_result.ok:
                        result = PreparationResult(
                            ok=False,
                            failed_step=step.name,
                            error=wsl_result.error
                            or f"{step.name} failed with exit {wsl_result.exit_code}",
                            step_results=step_results,
                            payload=self._payload,
                        )
                        self._emit_once(self.finished_error, result)
                        return
                else:
                    fn: Callable = step.action
                    # Prefer (token, progress) signature when supported.
                    try:
                        params = list(inspect.signature(fn).parameters)
                    except (TypeError, ValueError):
                        params = []
                    if len(params) >= 2:
                        fn_result = fn(self._cancel, self._progress)
                    else:
                        fn_result = fn(self._cancel)
                    if self._cancel.cancelled:
                        result = PreparationResult(
                            ok=False,
                            cancelled=True,
                            failed_step=step.name,
                            error="Cancelled",
                            step_results=step_results,
                            payload=fn_result if fn_result is not None else self._payload,
                        )
                        if isinstance(fn_result, PreparationResult) and fn_result.cancelled:
                            result = fn_result
                        self._emit_once(self.finished_cancelled, result)
                        return
                    if isinstance(fn_result, PreparationResult):
                        if not fn_result.ok:
                            if fn_result.cancelled:
                                self._emit_once(self.finished_cancelled, fn_result)
                            else:
                                self._emit_once(self.finished_error, fn_result)
                            return
                        self._payload = fn_result.payload
                    else:
                        self._payload = fn_result
            result = PreparationResult(
                ok=True,
                step_results=step_results,
                payload=self._payload,
            )
            self._emit_once(self.finished_ok, result)
        except Exception as exc:
            result = PreparationResult(
                ok=False,
                failed_step="worker",
                error=str(exc),
                step_results=step_results,
                payload=self._payload,
            )
            self._emit_once(self.finished_error, result)
