"""Central non-Qt WSL / OpenFOAM command runtime.

Single supported location for path conversion, argv construction, quoting,
timeouts, cancellation, and result capture. No PyQt imports.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WslPath:
    distro: Optional[str]
    linux_path: str


@dataclass
class WslRunResult:
    ok: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    argv: Tuple[str, ...] = ()
    cancelled: bool = False
    timed_out: bool = False
    error: str = ""
    safe_command: str = ""
    launch_failed: bool = False


@dataclass
class WslCancelToken:
    """Thread-safe cancellation flag that can terminate an attached process tree.

    ``cancel()`` is idempotent: repeated calls only re-attempt termination of the
    currently attached process (if any).
    """

    _event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            proc = self._proc
        if proc is not None:
            terminate_process_tree(proc)

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def attach_process(self, proc: subprocess.Popen) -> None:
        """Register the active child so cancel() can kill it mid-wait."""
        with self._lock:
            self._proc = proc
        if self.cancelled:
            terminate_process_tree(proc)

    def detach_process(self, proc: Optional[subprocess.Popen] = None) -> None:
        with self._lock:
            if proc is None or self._proc is proc:
                self._proc = None


def terminate_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of ``proc`` and its descendants.

    Strategy:
    * POSIX: processes are started in a new session (``start_new_session=True``).
      Cancellation sends ``SIGTERM`` / ``SIGKILL`` to the process group so the
      shell and OpenFOAM utility started under it are included.
    * Windows: ``wsl.exe`` is started in a new process group. Cancellation uses
      ``taskkill /F /T /PID`` to terminate the host process tree. This stops the
      ``wsl.exe`` bridge; the Linux-side utility typically receives SIGHUP when
      the WSL session side closes. Completely guaranteeing no orphan inside a
      shared WSL distro is environment-dependent — see module notes in tests.

    Always waits briefly for the host process to exit so the PID is reaped.
    """
    if proc.poll() is not None:
        return
    pid = proc.pid
    try:
        if os.name == "nt":
            # /T = terminate child processes of the host wsl.exe tree.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.terminate()
                except OSError:
                    pass
            deadline = time.time() + 2.0
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            if proc.poll() is None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass
    # Reap host process.
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def to_wsl_path_and_distro(win_path: str) -> WslPath:
    """Convert Windows / WSL-UNC / Linux paths to (distro, linux_path)."""
    p = (win_path or "").strip()
    if p.startswith("/") and not p.startswith("//") and not p.startswith("\\\\"):
        return WslPath(None, p)
    if p.startswith("\\\\") or p.startswith("//"):
        parts = [x for x in p.replace("/", "\\").split("\\") if x]
        if len(parts) >= 3 and parts[0].lower() in ("wsl.localhost", "wsl$"):
            return WslPath(parts[1], "/" + "/".join(parts[2:]))
        return WslPath(None, p.replace("\\", "/"))
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/").lstrip("/")
        linux = f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}/"
        return WslPath(None, linux)
    return WslPath(None, p.replace("\\", "/"))


def win_to_wsl_path(win_path: str) -> str:
    """Compatibility helper returning only the Linux path."""
    return to_wsl_path_and_distro(win_path).linux_path


def build_openfoam_script(
    *,
    case_linux_path: str,
    command: str,
    openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc",
    quiet_source: bool = True,
) -> str:
    """Build a bash -lc script that sources OpenFOAM, cds, and runs command."""
    src = shlex.quote(openfoam_bashrc)
    cdir = shlex.quote(case_linux_path)
    source_bit = (
        f"source {src} >/dev/null 2>&1 || true; "
        if quiet_source
        else f"source {src}; "
    )
    return (
        'set +u; '
        'export ZSH_NAME="${ZSH_NAME:-}"; '
        f"{source_bit}"
        f"cd {cdir} && {command}"
    )


def build_wsl_argv(
    script: str,
    *,
    distro: Optional[str] = None,
) -> List[str]:
    """Build the host argv for running ``script`` under bash (via wsl on Windows)."""
    if os.name == "nt":
        if distro:
            return ["wsl.exe", "-d", distro, "--", "bash", "-lc", script]
        return ["wsl.exe", "bash", "-lc", script]
    return ["bash", "-lc", script]


def build_case_command_argv(
    case_dir: str,
    command: str,
    *,
    openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc",
    quiet_source: bool = True,
) -> Tuple[List[str], WslPath, str]:
    """Return (argv, path_info, safe_command_description)."""
    path = to_wsl_path_and_distro(case_dir)
    script = build_openfoam_script(
        case_linux_path=path.linux_path,
        command=command,
        openfoam_bashrc=openfoam_bashrc,
        quiet_source=quiet_source,
    )
    argv = build_wsl_argv(script, distro=path.distro)
    safe = f"cd {path.linux_path} && {command}"
    return argv, path, safe


def popen_group_kwargs() -> dict:
    """Start children in a new session/group so cancel can target the tree."""
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP lets taskkill /T walk the host tree.
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _popen_kwargs() -> dict:
    """Backward-compatible internal alias for existing tests and callers."""
    return popen_group_kwargs()


def run_wsl_command(
    case_dir: str,
    command: str,
    *,
    openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc",
    timeout_s: Optional[float] = None,
    cancel_token: Optional[WslCancelToken] = None,
    quiet_source: bool = True,
) -> WslRunResult:
    """Run an approved command sequence in the case directory via WSL/bash.

    Cancellation is checked before start and continuously while waiting. When
    the cancel token fires, ``terminate_process_tree`` is used — not only a
    flag checked between steps.
    """
    argv, path, safe = build_case_command_argv(
        case_dir,
        command,
        openfoam_bashrc=openfoam_bashrc,
        quiet_source=quiet_source,
    )
    if cancel_token is not None and cancel_token.cancelled:
        return WslRunResult(
            ok=False,
            exit_code=-1,
            argv=tuple(argv),
            cancelled=True,
            error="Cancelled before start",
            safe_command=safe,
        )
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **_popen_kwargs(),
        )
    except OSError as exc:
        return WslRunResult(
            ok=False,
            exit_code=-1,
            argv=tuple(argv),
            error=str(exc),
            safe_command=safe,
            launch_failed=True,
        )

    if cancel_token is not None:
        cancel_token.attach_process(proc)

    timed_out = False
    cancelled = False
    stdout = ""
    stderr = ""
    try:
        if cancel_token is None and timeout_s is None:
            stdout, stderr = proc.communicate()
        else:
            elapsed = 0.0
            slice_s = 0.2
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=slice_s)
                    break
                except subprocess.TimeoutExpired:
                    elapsed += slice_s
                    if cancel_token is not None and cancel_token.cancelled:
                        cancelled = True
                        terminate_process_tree(proc)
                        try:
                            out_err = proc.communicate(timeout=5)
                            stdout, stderr = out_err[0] or "", out_err[1] or ""
                        except Exception:
                            stdout, stderr = "", ""
                        break
                    if timeout_s is not None and elapsed >= timeout_s:
                        timed_out = True
                        terminate_process_tree(proc)
                        try:
                            out_err = proc.communicate(timeout=5)
                            stdout, stderr = out_err[0] or "", out_err[1] or ""
                        except Exception:
                            stdout, stderr = "", ""
                        break
    except Exception as exc:
        terminate_process_tree(proc)
        if cancel_token is not None:
            cancel_token.detach_process(proc)
        return WslRunResult(
            ok=False,
            exit_code=-1,
            argv=tuple(argv),
            error=str(exc),
            safe_command=safe,
            cancelled=bool(cancel_token and cancel_token.cancelled),
        )
    finally:
        if cancel_token is not None:
            cancel_token.detach_process(proc)

    # If cancel raced with a natural exit, prefer cancelled when the token is set
    # and the process was terminated by us (non-zero / signal-like codes).
    if cancel_token is not None and cancel_token.cancelled:
        cancelled = True

    code = int(proc.returncode if proc.returncode is not None else -1)
    return WslRunResult(
        ok=(code == 0 and not cancelled and not timed_out),
        exit_code=code,
        stdout=stdout or "",
        stderr=stderr or "",
        argv=tuple(argv),
        cancelled=cancelled,
        timed_out=timed_out,
        error=(
            "Cancelled"
            if cancelled
            else ("Timed out" if timed_out else "")
        ),
        safe_command=safe,
    )


def run_wsl_script(
    script: str,
    *,
    distro: Optional[str] = None,
    timeout_s: Optional[float] = None,
    cancel_token: Optional[WslCancelToken] = None,
) -> WslRunResult:
    """Run a raw bash script (already fully constructed by the caller)."""
    argv = build_wsl_argv(script, distro=distro)
    if cancel_token is not None and cancel_token.cancelled:
        return WslRunResult(
            ok=False,
            exit_code=-1,
            argv=tuple(argv),
            cancelled=True,
            error="Cancelled before start",
            safe_command="<raw script>",
        )
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **_popen_kwargs(),
        )
    except OSError as exc:
        return WslRunResult(
            ok=False,
            exit_code=-1,
            argv=tuple(argv),
            error=str(exc),
            safe_command="<raw script>",
            launch_failed=True,
        )
    if cancel_token is not None:
        cancel_token.attach_process(proc)
    timed_out = False
    cancelled = False
    stdout = ""
    stderr = ""
    try:
        if cancel_token is None and timeout_s is None:
            stdout, stderr = proc.communicate()
        else:
            elapsed = 0.0
            slice_s = 0.2
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=slice_s)
                    break
                except subprocess.TimeoutExpired:
                    elapsed += slice_s
                    if cancel_token is not None and cancel_token.cancelled:
                        cancelled = True
                        terminate_process_tree(proc)
                        try:
                            out_err = proc.communicate(timeout=5)
                            stdout, stderr = out_err[0] or "", out_err[1] or ""
                        except Exception:
                            stdout, stderr = "", ""
                        break
                    if timeout_s is not None and elapsed >= timeout_s:
                        timed_out = True
                        terminate_process_tree(proc)
                        try:
                            out_err = proc.communicate(timeout=5)
                            stdout, stderr = out_err[0] or "", out_err[1] or ""
                        except Exception:
                            stdout, stderr = "", ""
                        break
    finally:
        if cancel_token is not None:
            cancel_token.detach_process(proc)
    if cancel_token is not None and cancel_token.cancelled:
        cancelled = True
    return WslRunResult(
        ok=(proc.returncode == 0 and not cancelled and not timed_out),
        exit_code=int(proc.returncode if proc.returncode is not None else -1),
        stdout=stdout or "",
        stderr=stderr or "",
        argv=tuple(argv),
        cancelled=cancelled,
        timed_out=timed_out,
        error=(
            "Cancelled"
            if cancelled
            else ("Timed out" if timed_out else "")
        ),
        safe_command="<raw script>",
    )


def quote_shell(value: str) -> str:
    """Quote a value for safe interpolation into a bash -lc script."""
    return shlex.quote(value)
