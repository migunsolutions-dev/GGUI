"""Post-run handling for selected outputs and native OpenFOAM time folders."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import shutil
from typing import Iterable, Optional, Tuple

from output_options import REMAP_2D_FILENAME


_FOAM_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_REMAP_TIME = re.compile(r"^\s*time\s+([^;]+);", re.MULTILINE)


@dataclass(frozen=True)
class ResultStoragePolicy:
    keep_openfoam_time_folders: bool = False
    vtk_fields: Tuple[str, ...] = ()
    preserve_remap_data: bool = False
    terminal_run: bool = False

    @property
    def needs_serial_results(self) -> bool:
        return self.terminal_run and bool(
            self.vtk_fields or self.preserve_remap_data
        )

    def foam_to_vtk_command(self) -> str:
        if not self.terminal_run:
            return ""
        fields = tuple(
            dict.fromkeys(
                str(name) for name in self.vtk_fields if _FOAM_FIELD.match(str(name))
            )
        )
        if not fields:
            return ""
        return "foamToVTK -fields '({})' > log.foamToVTK 2>&1".format(
            " ".join(fields)
        )


@dataclass
class CleanupReport:
    removed: list[str] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    skipped_reason: str = ""


def ensure_remap_snapshot(case_dir: str) -> bool:
    """Write metadata for the latest reconstructed native remap snapshot."""
    entries = _numeric_time_entries(case_dir)
    non_initial = [entry for entry in entries if entry[0] > 0]
    if not non_initial:
        return False
    time_name = non_initial[-1][1]
    time_dir = non_initial[-1][2]
    required_fields = ("p", "rho", "U", "T", "alpha.c4")
    if any(not os.path.isfile(os.path.join(time_dir, name)) for name in required_fields):
        return False
    path = os.path.join(case_dir, REMAP_2D_FILENAME)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(f"/* GGUI 2D remap snapshot ({REMAP_2D_FILENAME}) */\n")
            stream.write(f"time            {time_name};\n")
            stream.write('sourceCase      ".";\n')
            stream.write("fields          (p rho U T alpha.c4);\n")
        return True
    except OSError:
        return False


def _numeric_time_entries(root: str) -> list[tuple[float, str, str]]:
    entries: list[tuple[float, str, str]] = []
    try:
        names = os.listdir(root)
    except OSError:
        return entries
    for name in names:
        path = os.path.join(root, name)
        if os.path.islink(path) or not os.path.isdir(path):
            continue
        try:
            value = float(name)
        except ValueError:
            continue
        if value < 0:
            continue
        entries.append((value, name, path))
    return sorted(entries)


_RE_TIME_EQ = re.compile(r"(?m)^Time\s*=\s*([0-9.eE+-]+)")
_RE_ROOT_END_TIME = re.compile(r"(?m)^\s*endTime\s+([0-9.eE+-]+)\s*;")


def control_dict_root_text(case_dir: str) -> str:
    """controlDict text before the ``functions`` block (root keys only)."""
    control_path = os.path.join(case_dir, "system", "controlDict")
    with open(control_path, encoding="utf-8", errors="ignore") as stream:
        text = stream.read()
    idx = text.find("\nfunctions")
    if idx < 0:
        idx = text.find("functions")
    return text if idx < 0 else text[:idx]


def control_dict_root_end_time(case_dir: str) -> Optional[float]:
    """Root ``endTime``; never a nested function-object value."""
    try:
        match = _RE_ROOT_END_TIME.search(control_dict_root_text(case_dir))
    except OSError:
        return None
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def last_blastfoam_logged_time(
    case_dir: str, log_text: Optional[str] = None
) -> Optional[float]:
    """Last ``Time =`` value from log.blastFoam."""
    if log_text is None:
        log_path = os.path.join(case_dir, "log.blastFoam")
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as stream:
                log_text = stream.read()
        except OSError:
            return None
    times = _RE_TIME_EQ.findall(log_text)
    if not times:
        return None
    try:
        return float(times[-1])
    except ValueError:
        return None


def is_generated_1d_case(case_dir: str) -> bool:
    if os.path.isfile(os.path.join(case_dir, ".watchdog_target_radius")):
        return True
    try:
        with open(
            os.path.join(case_dir, "system", "controlDict"),
            encoding="utf-8",
            errors="ignore",
        ) as stream:
            return "probes1d" in stream.read()
    except OSError:
        return False


def solver_run_succeeded(
    case_dir: str,
    return_code: Optional[int],
    *,
    user_stopped: bool = False,
) -> bool:
    """True only for a normal solver finish that honours configured endTime on 1D.

    A process exiting 0 is not enough: ``blastFoam | tee`` used to hide FOAM
    abort, and 1D ``stopAt writeNow`` (watchdog) also exits 0 before endTime.
    """
    if user_stopped:
        return False
    if return_code != 0:
        return False
    log_path = os.path.join(case_dir, "log.blastFoam")
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as stream:
            log_text = stream.read()
    except OSError:
        log_text = ""
    if "FOAM FATAL" in log_text:
        return False
    if not is_generated_1d_case(case_dir):
        return True
    end_time = control_dict_root_end_time(case_dir)
    last_t = last_blastfoam_logged_time(case_dir, log_text)
    if last_t is None:
        entries = [entry for entry in _numeric_time_entries(case_dir) if entry[0] > 0]
        last_t = entries[-1][0] if entries else None
    if end_time is None or last_t is None:
        return False
    tolerance = max(1.0e-9, abs(end_time) * 1.0e-4)
    return last_t >= end_time - tolerance


def run_reached_configured_end(case_dir: str) -> bool:
    """Confirm that a successful terminal run materialized its configured end."""
    end_time = control_dict_root_end_time(case_dir)
    entries = [entry for entry in _numeric_time_entries(case_dir) if entry[0] > 0]
    if end_time is None or not entries:
        return False
    latest = entries[-1][0]
    tolerance = max(1.0e-12, abs(end_time) * 1.0e-8)
    return latest >= end_time - tolerance


def _safe_remove_tree(case_dir: str, path: str) -> None:
    case_real = os.path.realpath(case_dir)
    path_real = os.path.realpath(path)
    if os.path.islink(path) or os.path.commonpath((case_real, path_real)) != case_real:
        raise OSError("refusing to remove symlink or path outside case")
    shutil.rmtree(path)


def _remap_time_name(case_dir: str, entries: Iterable[tuple[float, str, str]]) -> str:
    entries = list(entries)
    path = os.path.join(case_dir, REMAP_2D_FILENAME)
    token = ""
    try:
        with open(path, encoding="utf-8", errors="ignore") as stream:
            match = _REMAP_TIME.search(stream.read())
            token = match.group(1).strip() if match else ""
    except OSError:
        pass
    if token.lower() == "latest" or not token:
        return entries[-1][1] if entries else ""
    try:
        requested = float(token)
    except ValueError:
        return ""
    for value, name, _path in entries:
        if value == requested:
            return name
    return ""


def cleanup_native_time_folders(
    case_dir: str,
    policy: ResultStoragePolicy,
) -> CleanupReport:
    """Remove completed-run native history after all consumers have finished.

    Initial time ``0`` and case inputs are always retained.  A requested 2D
    remap snapshot keeps exactly its referenced serial time because the current
    remap consumer reads native OpenFOAM fields rather than VTK.
    """
    report = CleanupReport()
    if policy.keep_openfoam_time_folders:
        report.skipped_reason = "native time-folder retention enabled"
        return report
    if not case_dir or not os.path.isdir(case_dir):
        report.skipped_reason = "case directory unavailable"
        return report

    entries = _numeric_time_entries(case_dir)
    preserve = {"0"}
    if policy.preserve_remap_data:
        remap_time = _remap_time_name(case_dir, entries)
        if not remap_time or remap_time == "0":
            report.skipped_reason = "remap snapshot has no reconstructed serial time"
            return report
        preserve.add(remap_time)

    for value, name, path in entries:
        rel = name
        if value == 0 or name in preserve:
            report.preserved.append(rel)
            continue
        try:
            _safe_remove_tree(case_dir, path)
            report.removed.append(rel)
        except OSError as exc:
            report.failures.append(f"{rel}: {exc}")

    # Parallel partitions are temporary only after the runner has completed
    # reconstruction/VTK export.  Never remove them if remap reconstruction
    # could not be verified above.
    try:
        processor_names = [
            name
            for name in os.listdir(case_dir)
            if re.fullmatch(r"processor\d+", name)
            and os.path.isdir(os.path.join(case_dir, name))
        ]
    except OSError:
        processor_names = []
    for name in processor_names:
        processor_dir = os.path.join(case_dir, name)
        processor_entries = _numeric_time_entries(processor_dir)
        for value, time_name, time_path in processor_entries:
            if value <= 0:
                continue
            rel = os.path.join(name, time_name)
            try:
                _safe_remove_tree(case_dir, time_path)
                report.removed.append(rel)
            except OSError as exc:
                report.failures.append(f"{rel}: {exc}")
        try:
            remaining = set(os.listdir(processor_dir))
            if remaining.issubset({"0", "constant"}):
                _safe_remove_tree(case_dir, processor_dir)
                report.removed.append(name)
            elif remaining:
                report.preserved.append(name)
        except OSError as exc:
            report.failures.append(f"{name}: {exc}")
    return report
