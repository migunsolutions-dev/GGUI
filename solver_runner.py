import os
import re
import glob
import time
import shlex
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

from PyQt5.QtCore import QThread, pyqtSignal

# Number of tail lines to capture from each log file for debug_summary.txt
DEBUG_TAIL_LINES = 50


class ExecutionIntent(str, Enum):
    FRESH_FULL_PIPELINE = "fresh_full_pipeline"
    INITIALIZED_SOLVER_RUN = "initialized_solver_run"
    RESUME = "resume"
    ONE_STEP_RESUME = "one_step_resume"


class ExecutionPreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionPlan:
    intent: ExecutionIntent
    command: str
    latest_time: Optional[float]
    log_name: str = "log.blastFoam"


def _numeric_time_dirs(path: str) -> List[float]:
    values: List[float] = []
    if not os.path.isdir(path):
        return values
    for name in os.listdir(path):
        try:
            value = float(name)
        except ValueError:
            continue
        if value > 0 and os.path.isdir(os.path.join(path, name)):
            values.append(value)
    return values


def build_execution_plan(
    case_dir: str,
    cores: int,
    intent: ExecutionIntent,
) -> ExecutionPlan:
    """Build a non-destructive solver command for an already generated case."""
    cores = max(1, int(cores))
    if intent == ExecutionIntent.FRESH_FULL_PIPELINE:
        return ExecutionPlan(intent, "bash ./Allrun", None, "log.Allrun")

    serial_times = _numeric_time_dirs(case_dir)
    serial_latest = max(serial_times) if serial_times else None
    processor_dirs = sorted(
        path
        for path in glob.glob(os.path.join(case_dir, "processor[0-9]*"))
        if os.path.isdir(path)
    )
    per_processor_times = [_numeric_time_dirs(path) for path in processor_dirs]
    resume = intent in (ExecutionIntent.RESUME, ExecutionIntent.ONE_STEP_RESUME)

    # Consistent processor latest only when every rank reports the same max time.
    processor_latest: Optional[float] = None
    if processor_dirs:
        latest_values = [max(times) if times else None for times in per_processor_times]
        if cores > 1:
            expected = {f"processor{i}" for i in range(cores)}
            actual = {os.path.basename(path) for path in processor_dirs}
            if actual != expected:
                raise ExecutionPreparationError(
                    f"Processor state has {len(processor_dirs)} directories but the GUI requests {cores} cores."
                )
            if resume and len(set(latest_values)) != 1:
                raise ExecutionPreparationError(
                    "Processor directories do not share one consistent latest saved time; "
                    "reconstruct or repair the case before resuming."
                )
            if latest_values and latest_values[0] is not None and len(set(latest_values)) == 1:
                processor_latest = latest_values[0]
        else:
            # Serial resume: use max across ranks (reconstructPar -latestTime will unify).
            rank_maxes = [t for t in latest_values if t is not None]
            processor_latest = max(rank_maxes) if rank_maxes else None

    latest: Optional[float] = None
    if resume:
        # latest_time must match the state the solver will actually use.
        if cores == 1:
            if processor_latest is not None and (
                serial_latest is None or processor_latest > serial_latest
            ):
                latest = processor_latest
            else:
                latest = serial_latest
        else:
            if serial_latest is not None and (
                processor_latest is None or serial_latest > processor_latest
            ):
                latest = serial_latest
            elif processor_latest is not None:
                latest = processor_latest
            else:
                latest = serial_latest
    else:
        candidates = [t for t in (serial_latest, processor_latest) if t is not None]
        latest = max(candidates) if candidates else None

    if intent == ExecutionIntent.ONE_STEP_RESUME and latest is None:
        zero_dir = os.path.join(processor_dirs[0] if processor_dirs else case_dir, "0")
        if os.path.isdir(zero_dir) and os.path.isfile(os.path.join(zero_dir, "p")):
            latest = 0.0
    if resume and latest is None:
        raise ExecutionPreparationError(
            "No resumable saved time exists. Initialize and run the case before Resume/Exact 1."
        )

    if cores == 1:
        start_mode = "latestTime" if resume else "startTime"
        prep = ""
        if resume and processor_dirs:
            if processor_latest is not None and (
                serial_latest is None or processor_latest > serial_latest
            ):
                prep = "reconstructPar -latestTime > log.reconstructResume 2>&1 && "
        cmd = (
            f"set -o pipefail; foamDictionary system/controlDict -entry startFrom -set {start_mode} "
            f"> log.prepareSolver 2>&1 && {prep}blastFoam 2>&1 | tee log.blastFoam"
        )
        return ExecutionPlan(intent, cmd, latest)

    prep = ""
    if resume and processor_dirs:
        if serial_latest is not None and (
            processor_latest is None or serial_latest > processor_latest
        ):
            # Serial state is ahead: re-decompose latest serial time into processors.
            prep = "decomposePar -force -latestTime > log.decomposeParSolver 2>&1 && "
        # else: consistent processor state is newer or equal — reuse it.
    elif not processor_dirs:
        # Decompose the initialized time 0 for a new run, or only the latest
        # reconstructed serial state for resume. Neither path cleans the case.
        decompose_opt = "-force -latestTime" if resume else "-force"
        prep = f"decomposePar {decompose_opt} > log.decomposeParSolver 2>&1 && "
    start_mode = "latestTime" if resume else "startTime"
    cmd = (
        f"set -o pipefail; foamDictionary system/controlDict -entry startFrom -set {start_mode} "
        f"> log.prepareSolver 2>&1 && {prep}"
        f"mpirun -np {cores} blastFoam -parallel 2>&1 | tee log.blastFoam"
    )
    return ExecutionPlan(intent, cmd, latest)


FINAL_RECONSTRUCT_CMD = "reconstructPar -latestTime > log.reconstructFinal 2>&1"


class SolverRunner(QThread):
    """
    Run the case via Allrun in WSL and stream probes data live.
    
    UPDATES:
    - Calculates Step Number (based on writeInterval=100).
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
        self._proc: Optional[subprocess.Popen] = None

        self._probe_file: Optional[str] = None
        self._probe_pos: int = 0
        self._end_time_s: Optional[float] = None
        
        # Stats tracking
        self._total_lines_read = 0
        self._last_time_val = 0.0

        # On-the-fly reconstruction (parallel only): tail solver log, spawn reconstructPar -newTimes
        self._log_blastfoam_pos: int = 0
        self._reconstruct_proc: Optional[subprocess.Popen] = None
        self._last_reconstructed_time: Optional[float] = None

        # 1D watchdog: trigger stop when shock reaches target radius (only once)
        self._watchdog_triggered: bool = False
        self._watchdog_stop_requested_time: Optional[float] = None  # time.time() when we created "stop"
        self._watchdog_grace_seconds: float = 3.0  # wait before forcing process stop if solver ignores "stop"

        self._wsl_distro, self._linux_case_dir = self._win_unc_to_wsl_path_and_distro(win_case_dir)

    def stop(self) -> None:
        self.keep_running = False
        if self._proc and self._proc.poll() is None:
            try:
                self.status_signal.emit("Stopping solver...")
                self._proc.terminate()
                for _ in range(40):
                    if self._proc.poll() is not None:
                        break
                    time.sleep(0.05)
                if self._proc.poll() is None:
                    self._proc.kill()
            except Exception:
                pass

    @staticmethod
    def _win_unc_to_wsl_path_and_distro(win_path: str) -> Tuple[Optional[str], str]:
        p = (win_path or "").strip()
        if p.startswith("/"):
            return None, p
        if p.startswith("\\\\"):
            parts = [x for x in p.split("\\") if x]
            if len(parts) >= 3 and parts[0].lower() in ("wsl.localhost", "wsl$"):
                distro = parts[1]
                linux_parts = parts[2:]
                return distro, "/" + "/".join(linux_parts)
            return None, p.replace("\\", "/")
        # Windows absolute path (e.g. C:\...): WSL needs /mnt/c/...
        if len(p) >= 2 and p[1] == ":":
            drive = p[0].lower()
            rest = p[2:].replace("\\", "/").lstrip("/")
            return None, f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}/"
        return None, p.replace("\\", "/")

    def _build_wsl_cmd(self, linux_dir: str, cmd: str) -> List[str]:
        """Build WSL/bash command that sources OpenFOAM in the same shell then runs cmd.
        Source must run in the current shell (not a subshell) so PATH and env persist for cmd.
        Redirection (e.g. > log.reconstructPar) in cmd is interpreted by bash -lc."""
        src = shlex.quote(self.openfoam_bashrc)
        cdir = shlex.quote(linux_dir)
        script = (
            'set +u; '
            'export ZSH_NAME="${ZSH_NAME:-}"; '
            f'source {src} >/dev/null 2>&1 || true; '
            f'cd {cdir} && {cmd}'
        )
        if os.name == "nt":
            if self._wsl_distro:
                return ["wsl", "-d", self._wsl_distro, "--", "bash", "-lc", script]
            return ["wsl", "bash", "-lc", script]
        return ["bash", "-lc", script]

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
                for line in f:
                    s = line.strip()
                    if s.startswith("endTime"):
                        tokens = s.replace(";", "").split()
                        if len(tokens) >= 2:
                            self._end_time_s = float(tokens[1])
                            return
        except Exception:
            self._end_time_s = None

    def _discover_probe_file(self) -> Optional[str]:
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
            self._reconstruct_proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self._reconstruct_proc = None

    def _wait_for_inflight_reconstruction(self, timeout_s: float = 120.0) -> None:
        """Block until any in-flight reconstructPar -newTimes process exits."""
        proc = self._reconstruct_proc
        if proc is None or proc.poll() is not None:
            return
        deadline = time.time() + timeout_s
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)

    def _final_reconstruct_latest(self) -> int:
        """Deterministic post-parallel reconstruct so the serial case has the latest result.

        Returns the subprocess exit code (0 on success).
        """
        self._wait_for_inflight_reconstruction()
        self.status_signal.emit("Final reconstruction (reconstructPar -latestTime)...")
        try:
            args = self._build_wsl_cmd(self._linux_case_dir, FINAL_RECONSTRUCT_CMD)
            completed = subprocess.run(
                args,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return int(completed.returncode)
        except Exception:
            return 1

    def _check_watchdog_trigger(self, case_dir: str) -> None:
        """If 1D case has watchdog_probe and pressure at target radius > 1.5e5 Pa, create 'stop' for graceful exit."""
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
        if pressure <= 1.5e5:
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
        try:
            stop_path = os.path.join(case_dir, "stop")
            with open(stop_path, "w", encoding="utf-8") as f:
                pass
        except OSError:
            pass
        self._watchdog_stop_requested_time = time.time()

    def _maybe_stop_after_watchdog(self) -> None:
        """If watchdog requested stop and grace period elapsed, terminate process so run actually stops."""
        if not self._watchdog_triggered or self._watchdog_stop_requested_time is None:
            return
        if self._proc is None or self._proc.poll() is not None:
            return
        elapsed = time.time() - self._watchdog_stop_requested_time
        if elapsed < self._watchdog_grace_seconds:
            return
        self.keep_running = False
        try:
            self._proc.terminate()
            for _ in range(40):
                if self._proc.poll() is not None:
                    break
                time.sleep(0.05)
            if self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass

    def _read_new_probe_lines(self) -> Optional[Tuple[float, List[float], int, float]]:
        if not self._probe_file:
            return None

        try:
            with open(self._probe_file, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(self._probe_pos)
                new = f.read()
                self._probe_pos = f.tell()
        except Exception:
            return None

        if not new:
            return None

        lines = [ln.strip() for ln in new.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not lines:
            return None

        # Parse the last line
        last = lines[-1]
        parts = last.split()
        if len(parts) < 2:
            return None

        try:
            t = float(parts[0])
            ps = [float(x) for x in parts[1:]]
            
            # --- CALCULATE STATS ---
            # 1. Update total lines read
            count_new = len(lines)
            self._total_lines_read += count_new
            
            # 2. Estimated Step (We know writeInterval is 100 from Generator)
            current_step = self._total_lines_read * 100
            
            # 3. Estimated dt (Time difference / steps elapsed)
            # Avoid division by zero
            dt_est = 0.0
            if count_new > 0 and self._total_lines_read > 1:
                time_diff = t - self._last_time_val
                # Each line represents 100 steps
                steps_diff = count_new * 100 
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
            self._proc = subprocess.Popen(args)
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