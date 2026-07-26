"""Central non-Qt WSL / OpenFOAM command runtime.

Single supported location for path conversion, argv construction, quoting,
timeouts, cancellation, and result capture. No PyQt imports.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import threading
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


@dataclass
class WslCancelToken:
    """Thread-safe cancellation flag for long-running WSL operations."""

    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


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


def run_wsl_command(
    case_dir: str,
    command: str,
    *,
    openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc",
    timeout_s: Optional[float] = None,
    cancel_token: Optional[WslCancelToken] = None,
    quiet_source: bool = True,
) -> WslRunResult:
    """Run an approved command sequence in the case directory via WSL/bash."""
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
        )
    except OSError as exc:
        return WslRunResult(
            ok=False,
            exit_code=-1,
            argv=tuple(argv),
            error=str(exc),
            safe_command=safe,
        )

    timed_out = False
    cancelled = False
    try:
        if cancel_token is None and timeout_s is None:
            stdout, stderr = proc.communicate()
        else:
            # Poll so cancellation can interrupt waits.
            deadline = None if timeout_s is None else (timeout_s)
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
                        proc.kill()
                        stdout, stderr = proc.communicate()
                        break
                    if deadline is not None and elapsed >= deadline:
                        timed_out = True
                        proc.kill()
                        stdout, stderr = proc.communicate()
                        break
    except Exception as exc:
        try:
            proc.kill()
        except OSError:
            pass
        return WslRunResult(
            ok=False,
            exit_code=-1,
            argv=tuple(argv),
            error=str(exc),
            safe_command=safe,
        )

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
) -> WslRunResult:
    """Run a raw bash script (already fully constructed by the caller)."""
    argv = build_wsl_argv(script, distro=distro)
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return WslRunResult(
            ok=False,
            exit_code=-1,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            argv=tuple(argv),
            timed_out=True,
            error="Timed out",
            safe_command="<raw script>",
        )
    except OSError as exc:
        return WslRunResult(
            ok=False,
            exit_code=-1,
            argv=tuple(argv),
            error=str(exc),
            safe_command="<raw script>",
        )
    return WslRunResult(
        ok=completed.returncode == 0,
        exit_code=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        argv=tuple(argv),
        safe_command="<raw script>",
    )


def quote_shell(value: str) -> str:
    """Quote a value for safe interpolation into a bash -lc script."""
    return shlex.quote(value)
