import glob
import os
import re
import time
import subprocess
import threading
from typing import Optional, Tuple, List

from PyQt5.QtCore import QThread, pyqtSignal

from execution_plan import (  # noqa: F401 — re-export for existing imports
    FINAL_RECONSTRUCT_CMD,
    ExecutionIntent,
    ExecutionPlan,
    ExecutionPreparationError,
    build_execution_plan,
)
from wsl_runtime import (
    build_case_command_argv,
    popen_group_kwargs,
    terminate_process_tree,
    to_wsl_path_and_distro,
)

# Number of tail lines to capture from each log file for debug_summary.txt
DEBUG_TAIL_LINES = 50
DEFAULT_PROBE_WRITE_INTERVAL_STEPS = 25
WATCHDOG_STRONG_P_PA = 1.5e5
WATCHDOG_ARRIVAL_OVER_PA = 8.0e3
WATCHDOG_DROP_FRAC = 0.10


class WatchdogState:
    """Ambient and peak overpressure at the 1D target-radius probe."""

    __slots__ = ("p_ref", "peak_over")

    def __init__(self) -> None:
        self.p_ref: Optional[float] = None
        self.peak_over: float = 0.0


def watchdog_should_stop(pressure: float, state: WatchdogState) -> bool:
    """True when the shock has reached the target radius.

    A hard 150 kPa cut misses weak arrivals (e.g. ~149 kPa at 4 m). Stop on a
    strong jump, or after overpressure at the target has peaked and fallen.
    """
    if pressure != pressure or pressure <= 0.0:
        return False
    if state.p_ref is None:
        state.p_ref = float(pressure)
        return False
    over = float(pressure) - float(state.p_ref)
    if over > state.peak_over:
        state.peak_over = over
    if float(pressure) >= WATCHDOG_STRONG_P_PA:
        return True
    arrived = state.peak_over >= WATCHDOG_ARRIVAL_OVER_PA
    dropped = over <= (1.0 - WATCHDOG_DROP_FRAC) * state.peak_over
    return bool(arrived and dropped)


def probe_write_interval_from_control_dict(
    text: str, default: int = DEFAULT_PROBE_WRITE_INTERVAL_STEPS
) -> int:
    """Read probes1d writeInterval from a controlDict (GUI refresh cadence)."""
    match = re.search(r"probes1d\s*\{.*?writeInterval\s+(\d+)", text, re.DOTALL)
    if not match:
        return max(1, int(default))
    try:
        return max(1, int(match.group(1)))
    except (TypeError, ValueError):
        return max(1, int(default))


def request_solver_write_and_stop(case_dir: str) -> bool:
    """Ask a running OpenFOAM solver to dump the current time and exit.

    1D field dumps are sparse (one interval at endTime). The watchdog must not
    wait for that write; ``stopAt writeNow`` is reread because controlDict is
    ``runTimeModifiable``.
    """
    cd_path = os.path.join(case_dir, "system", "controlDict")
    try:
        with open(cd_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        new_text, n_sub = re.subn(
            r"(?m)^stopAt\s+\S+\s*;",
            "stopAt          writeNow;",
            text,
            count=1,
        )
        if n_sub:
            with open(cd_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(new_text)
            return True
    except OSError:
        return False
    return False


def complete_probe_chunk(raw: bytes) -> Tuple[bytes, int]:
    """Keep only newline-terminated probe text; leftover bytes stay unread."""
    if not raw:
        return b"", 0
    last_nl = raw.rfind(b"\n")
    if last_nl < 0:
        return b"", len(raw)
    complete = raw[: last_nl + 1]
    return complete, len(raw) - len(complete)


def parse_last_probe_pressures(text: str) -> Optional[Tuple[float, List[float], int]]:
    """Return (time, pressures, data_line_count) from complete probe text."""
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        return None
    parts = lines[-1].split()
    if len(parts) < 2:
        return None
    try:
        t = float(parts[0])
        pressures = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if not pressures:
        return None
    return t, pressures, len(lines)


class SolverRunner(QThread):
    """
    Run the case via Allrun in WSL and stream probes data live.
    
    UPDATES:
    - Calculates Step Number from probes1d writeInterval (GUI refresh freq.).
    - Calculates Avg DeltaT (based on time difference).
    - Emits (pressures, time, step, dt).
    - On failure: aggregates last N lines of all log.* into debug_summary.txt at project_root.
    """

    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    # MODIFIED SIGNAL: pressures, time [s], step [int], dt [s]
    data_signal = pyqtSignal(list, float, int, float)
    finished_signal = pyqtSignal(bool)

    def __init__(
        self,
        win_case_dir: str,
        openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc",
        project_root: Optional[str] = None,
        cores: int = 1,
        intent: ExecutionIntent = ExecutionIntent.FRESH_FULL_PIPELINE,
    ):
        super().__init__()
        self.win_case_dir = win_case_dir
        self.openfoam_bashrc = openfoam_bashrc
        self.project_root = project_root
        self.cores = max(1, int(cores))
        self.intent = ExecutionIntent(intent)
        self.keep_running = True
        self._stop_requested = False
        self._process_lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None

        self._probe_file: Optional[str] = None
        self._probe_pos: int = 0
        self._end_time_s: Optional[float] = None
        self._probe_write_interval_steps: int = DEFAULT_PROBE_WRITE_INTERVAL_STEPS
        
        # Stats tracking
        self._total_lines_read = 0
        self._last_time_val = 0.0

        # On-the-fly reconstruction (parallel only): tail solver log, spawn reconstructPar -newTimes
        self._log_blastfoam_pos: int = 0
        self._reconstruct_proc: Optional[subprocess.Popen] = None
        self._last_reconstructed_time: Optional[float] = None

        # 1D watchdog: trigger stop when shock reaches target radius (only once)
        self._watchdog_triggered: bool = False
        self._watchdog_stop_requested_time: Optional[float] = None  # time.time() when writeNow was requested
        self._watchdog_grace_seconds: float = 3.0  # wait before forcing process stop if solver ignores writeNow
        self._watchdog_state = WatchdogState()

        path = to_wsl_path_and_distro(win_case_dir)
        self._wsl_distro, self._linux_case_dir = path.distro, path.linux_path

    def stop(self) -> None:
        self.keep_running = False
        self._stop_requested = True
        with self._process_lock:
            solver = self._proc
            reconstruct = self._reconstruct_proc
        if solver and solver.poll() is None:
            self.status_signal.emit("Stopping solver...")
            terminate_process_tree(solver)
        if (
            reconstruct is not None
            and reconstruct is not solver
            and reconstruct.poll() is None
        ):
            terminate_process_tree(reconstruct)
        with self._process_lock:
            if self._reconstruct_proc is reconstruct:
                self._reconstruct_proc = None

    @staticmethod
    def _win_unc_to_wsl_path_and_distro(win_path: str) -> Tuple[Optional[str], str]:
        """Compatibility wrapper around ``wsl_runtime.to_wsl_path_and_distro``."""
        path = to_wsl_path_and_distro(win_path)
        return path.distro, path.linux_path

    def _build_wsl_cmd(self, linux_dir: str, cmd: str) -> List[str]:
        """Build WSL/bash argv via the central ``wsl_runtime`` module."""
        # linux_dir is already converted; pass a synthetic case dir only for quoting
        # of bashrc + command. Distro comes from the runner instance.
        argv, _, _ = build_case_command_argv(
            linux_dir if linux_dir.startswith("/") else self.win_case_dir,
            cmd,
            openfoam_bashrc=self.openfoam_bashrc,
            quiet_source=True,
        )
        # Rebuild with the exact linux_dir + known distro to avoid re-detect drift.
        from wsl_runtime import build_openfoam_script, build_wsl_argv

        script = build_openfoam_script(
            case_linux_path=linux_dir,
            command=cmd,
            openfoam_bashrc=self.openfoam_bashrc,
            quiet_source=True,
        )
        return build_wsl_argv(script, distro=self._wsl_distro)

    def _run_simple(self, linux_dir: str, cmd: str) -> None:
        try:
            args = self._build_wsl_cmd(linux_dir, cmd)
            subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _find_control_dict_end_time(self) -> None:
        try:
            p = os.path.join(self.win_case_dir, "system", "controlDict")
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            self._end_time_s = None
            return
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("endTime"):
                tokens = s.replace(";", "").split()
                if len(tokens) >= 2:
                    try:
                        self._end_time_s = float(tokens[1])
                    except ValueError:
                        self._end_time_s = None
                    break
        self._probe_write_interval_steps = probe_write_interval_from_control_dict(text)

    def _discover_probe_file(self) -> Optional[str]:
        try:
            base = os.path.join(self.win_case_dir, "postProcessing", "probes1d")
            if not os.path.isdir(base):
                return None
            candidate = os.path.join(base, "0", "p")
            if os.path.isfile(candidate):
                return candidate
            paths = glob.glob(os.path.join(base, "*", "p"))
            if not paths:
                return None

            def time_key(path: str) -> float:
                try:
                    tdir = os.path.basename(os.path.dirname(path))
                    return float(tdir)
                except Exception:
                    return -1.0

            return sorted(paths, key=time_key)[-1]
        except Exception:
            return None

    def _aggregate_log_errors(self, exit_code: int) -> str:
        """Collect last DEBUG_TAIL_LINES from each log.* in case dir. Return full summary text."""
        lines = [
            "=== Simulation failure: automatic error summary ===",
            "",
            f"Case directory: {self.win_case_dir}",
            f"Execution intent: {self.intent.value}",
            f"Process exit code: {exit_code}",
            "",
            "--- Last {} lines of each log file (newest first) ---".format(DEBUG_TAIL_LINES),
            "",
        ]
        pattern = os.path.join(self.win_case_dir, "log.*")
        log_paths = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p) if os.path.isfile(p) else 0, reverse=True)
        if not log_paths:
            lines.append("(No log.* files found in case directory.)")
            return "\n".join(lines)
        for path in log_paths:
            name = os.path.basename(path)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                lines.append(f"=== {name} (read error: {e}) ===")
                lines.append("")
                continue
            file_lines = content.splitlines()
            tail = file_lines[-DEBUG_TAIL_LINES:] if len(file_lines) > DEBUG_TAIL_LINES else file_lines
            lines.append(f"=== {name} (last {len(tail)} lines) ===")
            lines.append("")
            lines.extend(tail)
            lines.append("")
        return "\n".join(lines)

    def _write_debug_summary(self, exit_code: int) -> None:
        """On failure, write aggregated log tail to project_root/debug_summary.txt."""
        root = self.project_root
        if not root:
            root = os.path.dirname(self.win_case_dir)
        root = os.path.abspath(root)
        os.makedirs(root, exist_ok=True)
        out_path = os.path.join(root, "debug_summary.txt")
        try:
            content = self._aggregate_log_errors(exit_code)
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
        except Exception:
            pass

    def _maybe_reconstruct_new_times(self) -> None:
        """If parallel (cores > 1), tail solver log and run reconstructPar -newTimes when a write is detected (non-blocking).
        Uses shell redirection so Linux creates log.reconstructPar in the case directory."""
        if self.cores <= 1:
            return
        if self._reconstruct_proc is not None and self._reconstruct_proc.poll() is None:
            return
        log_path = os.path.join(self.win_case_dir, "log.blastFoam")
        if not os.path.isfile(log_path):
            return
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(0, 2)
                end_pos = f.tell()
                if self._log_blastfoam_pos == 0:
                    self._log_blastfoam_pos = end_pos
                    return
                f.seek(self._log_blastfoam_pos)
                new_content = f.read()
                self._log_blastfoam_pos = f.tell()
        except Exception:
            return
        if not new_content:
            return
        if "Time =" not in new_content and "Writing" not in new_content:
            return

        # Extract latest time string from log (Time = 0.001 or Writing time 0.001)
        time_re = re.compile(r"(?:Time\s*=\s*|Writing\s+time\s+)([\d\.eE\+\-]+)")
        matches = time_re.findall(new_content)
        if not matches:
            return
        time_str = matches[-1]
        try:
            time_val = float(time_str)
        except ValueError:
            return
        if self._last_reconstructed_time is not None and time_val <= self._last_reconstructed_time:
            return

        # Poll for processor0/<time> to exist and have content (avoids race with solver write)
        proc0_dir = os.path.join(self.win_case_dir, "processor0")
        time_dir = os.path.join(proc0_dir, time_str)
        marker_file = os.path.join(time_dir, "uniform", "time")
        max_retries = 20
        interval = 0.2
        found = False
        for _ in range(max_retries):
            if os.path.isdir(time_dir):
                if os.path.isfile(marker_file):
                    found = True
                    break
                try:
                    if os.listdir(time_dir):
                        found = True
                        break
                except OSError:
                    pass
            time.sleep(interval)

        self._last_reconstructed_time = time_val
        cmd = "reconstructPar -newTimes > log.reconstructPar 2>&1"
        try:
            args = self._build_wsl_cmd(self._linux_case_dir, cmd)
            with self._process_lock:
                if self._stop_requested or not self.keep_running:
                    return
                self._reconstruct_proc = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **popen_group_kwargs(),
                )
        except Exception:
            with self._process_lock:
                self._reconstruct_proc = None

    def _wait_for_inflight_reconstruction(self, timeout_s: float = 120.0) -> bool:
        """Wait for any in-flight reconstructPar -newTimes process.

        Returns True if there is no in-flight process or it exits before timeout.
        On timeout: terminate (then kill if needed) the owned child and return False.
        Never leaves a reconstruction child running when False is returned and kill succeeds.
        """
        proc = self._reconstruct_proc
        if proc is None or proc.poll() is not None:
            return True
        deadline = time.time() + timeout_s
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if proc.poll() is not None:
            return True

        # Timeout: stop the owned child before any final reconstructPar -latestTime.
        try:
            proc.terminate()
        except Exception:
            pass
        stop_deadline = time.time() + 2.0
        while proc.poll() is None and time.time() < stop_deadline:
            time.sleep(0.05)
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
            kill_deadline = time.time() + 1.0
            while proc.poll() is None and time.time() < kill_deadline:
                time.sleep(0.05)
        if proc.poll() is not None:
            self._reconstruct_proc = None
        return False

    def _final_reconstruct_latest(self) -> int:
        """Deterministic post-parallel reconstruct so the serial case has the latest result.

        Returns the subprocess exit code (0 on success). Does not launch a second
        reconstructPar while an earlier reconstruction is still active or after a
        wait timeout.
        """
        if not self._wait_for_inflight_reconstruction():
            self.status_signal.emit(
                "Final reconstruction skipped: in-flight reconstructPar did not finish "
                "and was stopped after timeout. Check log.reconstructPar / debug_summary.txt."
            )
            return 1
        self.status_signal.emit("Final reconstruction (reconstructPar -latestTime)...")
        proc = None
        try:
            args = self._build_wsl_cmd(self._linux_case_dir, FINAL_RECONSTRUCT_CMD)
            with self._process_lock:
                if self._stop_requested or not self.keep_running:
                    return 1
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **popen_group_kwargs(),
                )
                self._reconstruct_proc = proc
            while proc.poll() is None:
                time.sleep(0.05)
            return int(proc.returncode)
        except Exception:
            return 1
        finally:
            with self._process_lock:
                if self._reconstruct_proc is proc:
                    self._reconstruct_proc = None

    def _check_watchdog_trigger(self, case_dir: str) -> None:
        """Stop when the shock has arrived at the target radius (peak-and-fall or strong p)."""
        if self._watchdog_triggered:
            return
        base = os.path.join(case_dir, "postProcessing", "watchdog_probe")
        if not os.path.isdir(base):
            return
        # Find latest p file (OpenFOAM may write 0/p or <time>/p)
        p_files = glob.glob(os.path.join(base, "*", "p"))
        if not p_files:
            return
        def mtime_key(p: str) -> float:
            try:
                return os.path.getmtime(p)
            except OSError:
                return 0.0
        p_path = max(p_files, key=mtime_key)
        try:
            with open(p_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]
        except OSError:
            return
        if not lines:
            return
        last = lines[-1]
        parts = last.split()
        if len(parts) < 2:
            return
        try:
            pressure = float(parts[1])
        except ValueError:
            return
        if not watchdog_should_stop(pressure, self._watchdog_state):
            return
        self._watchdog_triggered = True
        radius_str = "?"
        try:
            r_path = os.path.join(case_dir, ".watchdog_target_radius")
            if os.path.isfile(r_path):
                with open(r_path, "r", encoding="utf-8") as f:
                    radius_str = f.read().strip() or "?"
        except OSError:
            pass
        self.status_signal.emit(f"Shockwave reached target radius ({radius_str}m). Stopping simulation.")
        request_solver_write_and_stop(case_dir)
        self._watchdog_stop_requested_time = time.time()

    def _maybe_stop_after_watchdog(self) -> None:
        """If watchdog requested writeNow and grace period elapsed, terminate process so run actually stops."""
        if not self._watchdog_triggered or self._watchdog_stop_requested_time is None:
            return
        if self._proc is None or self._proc.poll() is not None:
            return
        elapsed = time.time() - self._watchdog_stop_requested_time
        if elapsed < self._watchdog_grace_seconds:
            return
        self.keep_running = False
        terminate_process_tree(self._proc)

    def _read_new_probe_lines(self) -> Optional[Tuple[float, List[float], int, float]]:
        if not self._probe_file:
            return None

        try:
            with open(self._probe_file, "rb") as f:
                f.seek(self._probe_pos)
                raw = f.read()
                complete, leftover = complete_probe_chunk(raw)
                self._probe_pos = f.tell() - leftover
        except Exception:
            return None

        if not complete:
            return None

        parsed = parse_last_probe_pressures(complete.decode("utf-8", "ignore"))
        if parsed is None:
            return None

        try:
            t, ps, count_new = parsed
            self._total_lines_read += count_new
            interval = max(1, int(self._probe_write_interval_steps))
            current_step = self._total_lines_read * interval
            dt_est = 0.0
            if count_new > 0 and self._total_lines_read > 1:
                time_diff = t - self._last_time_val
                steps_diff = count_new * interval
                if steps_diff > 0:
                    dt_est = time_diff / steps_diff
            self._last_time_val = t
            return t, ps, current_step, dt_est
        except Exception:
            return None

    def run(self) -> None:
        linux_dir = self._linux_case_dir
        self._find_control_dict_end_time()

        try:
            execution = build_execution_plan(
                self.win_case_dir, self.cores, self.intent
            )
        except ExecutionPreparationError as exc:
            self.status_signal.emit(str(exc))
            self.finished_signal.emit(False)
            return
        if self.intent == ExecutionIntent.FRESH_FULL_PIPELINE:
            self.status_signal.emit("Preparing fresh-run scripts...")
            self._run_simple(linux_dir, r"sed -i 's/\r$//' Allrun Allclean 2>/dev/null || true")
            self._run_simple(linux_dir, "chmod +x Allrun Allclean 2>/dev/null || true")
        self.status_signal.emit(f"Running: {self.intent.value}")
        args = self._build_wsl_cmd(linux_dir, execution.command)
        try:
            with self._process_lock:
                if self._stop_requested or not self.keep_running:
                    self.finished_signal.emit(False)
                    return
                self._proc = subprocess.Popen(args, **popen_group_kwargs())
        except Exception as e:
            self.status_signal.emit(
                f"Failed command `{execution.command}`: {e}. Log: {execution.log_name}"
            )
            self.finished_signal.emit(False)
            return

        self._probe_file = None
        self._probe_pos = 0
        self._total_lines_read = 0
        self._last_time_val = 0.0
        self._log_blastfoam_pos = 0
        self._log_data_pos = 0          # separate pos for data parsing from log
        self._log_step_count = 0        # step counter from log parsing
        self._reconstruct_proc = None
        self._last_reconstructed_time = None

        self._watchdog_triggered = False
        self._watchdog_stop_requested_time = None
        self._watchdog_state = WatchdogState()
        _re_time = re.compile(r"^Time\s*=\s*([\d\.eE\+\-]+)", re.MULTILINE)
        _re_dt = re.compile(r"^deltaT\s*=\s*([\d\.eE\+\-]+)", re.MULTILINE)
        _re_courant = re.compile(r"^Courant Number.*$", re.MULTILINE)
        while self.keep_running and self._proc.poll() is None:
            self._maybe_reconstruct_new_times()
            self._check_watchdog_trigger(self.win_case_dir)
            self._maybe_stop_after_watchdog()
            if self._probe_file is None:
                self._probe_file = self._discover_probe_file()
                if self._probe_file:
                    self._probe_pos = 0
                    rel = os.path.relpath(self._probe_file, self.win_case_dir)
                    self.status_signal.emit(f"Streaming: {rel}")

            latest = self._read_new_probe_lines()
            if latest is not None:
                t_s, pressures, step_n, dt_val = latest
                self.data_signal.emit(pressures, t_s, step_n, dt_val)
            elif self._probe_file is None:
                # No probe file (3D): parse log.blastFoam for step/time/dt
                log_path = os.path.join(self.win_case_dir, "log.blastFoam")
                if os.path.isfile(log_path):
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                            f.seek(self._log_data_pos)
                            new_text = f.read()
                            self._log_data_pos = f.tell()
                        if new_text:
                            times = _re_time.findall(new_text)
                            dts = _re_dt.findall(new_text)
                            if times:
                                self._log_step_count += len(times)
                                try:
                                    t_s = float(times[-1])
                                    dt_val = float(dts[-1]) if dts else 0.0
                                    self.data_signal.emit([], t_s, self._log_step_count, dt_val)
                                except ValueError:
                                    pass
                    except Exception:
                        pass

            time.sleep(0.10)

        rc = self._proc.poll() if self._proc else 1
        if not self.keep_running:
            self.status_signal.emit("Stopped.")
            self.finished_signal.emit(False)
            return

        if rc == 0:
            if self.cores > 1 and self.intent != ExecutionIntent.FRESH_FULL_PIPELINE:
                recon_rc = self._final_reconstruct_latest()
                if recon_rc != 0:
                    self._write_debug_summary(recon_rc)
                    self.status_signal.emit(
                        f"Solver finished but final reconstructPar -latestTime failed "
                        f"(rc={recon_rc}). Log: log.reconstructFinal, debug_summary.txt."
                    )
                    self.finished_signal.emit(False)
                    return
            self.progress_signal.emit(100)
            self.status_signal.emit("Finished.")
            self.finished_signal.emit(True)
        else:
            self._write_debug_summary(rc)
            self.status_signal.emit(
                f"Failed command `{execution.command}` (rc={rc}). "
                f"Logs: {execution.log_name}, debug_summary.txt."
            )
            self.finished_signal.emit(False)