"""Qt worker for long-running case preparation (separate from SolverRunner)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

from PyQt5.QtCore import QThread, pyqtSignal

from wsl_runtime import WslCancelToken, WslRunResult, run_wsl_command


@dataclass
class PreparationStep:
    """One named preparation step executed by the worker."""

    name: str
    # Either a pure callable or a WSL shell command string.
    action: Any
    kind: str = "callable"  # "callable" | "wsl"


@dataclass
class PreparationResult:
    ok: bool
    cancelled: bool = False
    failed_step: str = ""
    error: str = ""
    step_results: List[WslRunResult] = field(default_factory=list)
    payload: Any = None


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

    def request_cancel(self) -> None:
        self._cancel.cancel()

    def set_payload(self, payload: Any) -> None:
        self._payload = payload

    def run(self) -> None:
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
                    self.finished_cancelled.emit(result)
                    return
                self.progress.emit(step.name)
                self.log_line.emit(f"[Prepare] starting {step.name}")
                if step.kind == "wsl":
                    wsl_result = run_wsl_command(
                        self.case_dir,
                        str(step.action),
                        openfoam_bashrc=self.openfoam_bashrc,
                        cancel_token=self._cancel,
                    )
                    step_results.append(wsl_result)
                    if wsl_result.cancelled:
                        result = PreparationResult(
                            ok=False,
                            cancelled=True,
                            failed_step=step.name,
                            error="Cancelled",
                            step_results=step_results,
                            payload=self._payload,
                        )
                        self.finished_cancelled.emit(result)
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
                        self.finished_error.emit(result)
                        return
                else:
                    fn: Callable = step.action
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
                        self.finished_cancelled.emit(result)
                        return
                    if isinstance(fn_result, PreparationResult):
                        if not fn_result.ok:
                            if fn_result.cancelled:
                                self.finished_cancelled.emit(fn_result)
                            else:
                                self.finished_error.emit(fn_result)
                            return
                        self._payload = fn_result.payload
                    else:
                        self._payload = fn_result
            result = PreparationResult(
                ok=True,
                step_results=step_results,
                payload=self._payload,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            result = PreparationResult(
                ok=False,
                failed_step="worker",
                error=str(exc),
                step_results=step_results,
                payload=self._payload,
            )
            self.finished_error.emit(result)
