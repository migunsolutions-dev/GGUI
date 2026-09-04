"""Generate and run GGUI/blastFoam cases for the VIPER comparison."""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Sequence, Tuple

from axisymmetric_2d import BOUNDARY_OPEN, DIRECT_SOURCE, FIXED_MESH, REMAP_SOURCE
from completion_1d import overpressure_arrived, read_completion_record
from remap_handoff_1d import primary_shock_at_probe
from foam_dictionary import update_top_level_entries
from models import BOUNDARY_1D_TERMINATE, CaseInputs1D, RUN_MODE_TERMINATE
from models_2d import CaseInputs2D, MappingSource2D, ProbePoint2D
from simulation_service import SimulationService
from wsl_runtime import build_case_command_argv
from viper_compare.physics import TestDefinition

WORK = r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work"
BASHRC = "/opt/openfoam9/etc/bashrc"
ALLRUN = (
    "set -o pipefail; sed -i 's/\\r$//' Allrun Allclean 2>/dev/null || true; "
    "chmod +x Allrun Allclean 2>/dev/null || true; bash ./Allrun"
)


def _gauges_1d(spec: TestDefinition, radii: Sequence[float]) -> Tuple[Tuple[float, str], ...]:
    return tuple((r, f"G1D_R{r:.2f}") for r in radii)


def inputs_1d(
    spec: TestDefinition,
    *,
    remap_for_2d: bool,
    radius: float,
    gauges: Sequence[float],
) -> CaseInputs1D:
    return CaseInputs1D(
        radius=float(radius),
        cell_size=spec.dx_1d,
        p_atm=spec.p_atm,
        t_atm=spec.t_atm,
        mass_kg=spec.mass_kg,
        rho_charge=spec.rho_kg_m3,
        energy_j_per_kg=spec.energy_j_kg,
        material_props=spec.material_props(),
        max_cfl=spec.cfl_1d,
        end_time_s=spec.end_time_1d_s,
        write_interval_s=0.0,
        n_probes=50,
        probe_write_interval_steps=1,
        right_boundary=BOUNDARY_1D_TERMINATE,
        probe_fields=("p", "impulse"),
        enable_impulse=True,
        gauge_locations=_gauges_1d(spec, gauges),
        material_name="Custom",
        stop_mode=RUN_MODE_TERMINATE,
        stop_radius_m=float(radius),
        remap_for_2d=remap_for_2d,
    )


def inputs_2d(
    spec: TestDefinition,
    *,
    remapped: bool,
    source_1d: str = "",
    cores: int = 4,
) -> CaseInputs2D:
    probes = tuple(
        ProbePoint2D(name=f"G2D_R{r:.2f}", radius=r, height=spec.hob_m)
        for r in spec.r_gauges_2d
    )
    mapping = MappingSource2D()
    source = DIRECT_SOURCE
    if remapped:
        source = REMAP_SOURCE
        mapping = MappingSource2D(
            case_path=source_1d,
            time_mode="latest",
            mapped_radius=spec.r_remap_m,
        )
    return CaseInputs2D(
        radius=spec.domain_2d_r_m,
        height=spec.domain_2d_h_m,
        cell_size=spec.dx_2d,
        initialization_source=source,
        charge_shape="Sphere",
        charge_center_r=0.0,
        height_of_burst=spec.hob_m,
        detonation_radius=0.0,
        detonation_height=spec.hob_m,
        mass_kg=spec.mass_kg,
        material_name="Custom",
        rho_charge=spec.rho_kg_m3,
        energy_j_per_kg=spec.energy_j_kg,
        material_props=spec.material_props(),
        p_atm=spec.p_atm,
        t_atm=spec.t_atm,
        outer_boundary=BOUNDARY_OPEN,
        top_boundary=BOUNDARY_OPEN,
        bottom_boundary=BOUNDARY_OPEN,
        max_co=spec.cfl_2d,
        end_time_s=spec.end_time_2d_s,
        delta_t=1.0e-8,
        adjust_time_step=True,
        write_control_type="timeStep",
        write_interval_steps=100000,
        cycle_write=0,
        keep_openfoam_time_folders=False,
        cores=int(cores),
        mesh_mode=FIXED_MESH,
        mapping=mapping,
        probes=probes,
        output_fields=("p", "impulse"),
        enable_impulse=True,
    )


def request_write_now(case_dir: str) -> bool:
    """Qt-free copy of solver_runner.request_solver_write_and_stop."""
    import tempfile

    cd_path = os.path.join(case_dir, "system", "controlDict")
    try:
        with open(cd_path, encoding="utf-8") as handle:
            text = handle.read()
        new_text, _changed = update_top_level_entries(text, {"stopAt": "writeNow"})
        sys_dir = os.path.dirname(cd_path)
        fd, temp_path = tempfile.mkstemp(prefix=".ggui-cd-", suffix=".tmp", dir=sys_dir)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(new_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, cd_path)
        return True
    except (OSError, KeyError):
        return False


def _watchdog_pressure_path(case_dir: str) -> str:
    root = os.path.join(case_dir, "postProcessing", "watchdog_probe")
    if not os.path.isdir(root):
        return ""
    best = ""
    best_t = None
    for name in os.listdir(root):
        path = os.path.join(root, name, "p")
        if not os.path.isfile(path):
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        if best_t is None or t >= best_t:
            best_t = t
            best = path
    return best


def _last_watchdog_sample(path: str):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    for raw in reversed(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            continue
    return None


def run_allrun_with_optional_watchdog(
    *,
    case_dir: str,
    log_dir: str,
    prefix: str,
    watchdog: bool,
) -> dict:
    argv, mapped, safe = build_case_command_argv(
        case_dir, ALLRUN, openfoam_bashrc=BASHRC
    )
    os.makedirs(log_dir, exist_ok=True)
    cmd_path = os.path.join(log_dir, f"{prefix}_command.json")
    with open(cmd_path, "w", encoding="utf-8") as handle:
        json.dump({"case_dir": case_dir, "argv": argv, "safe": safe, "watchdog": watchdog}, handle, indent=2)
    out_path = os.path.join(log_dir, f"{prefix}_allrun.log")
    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT)
        triggered = False
        record = read_completion_record(case_dir) if watchdog else None
        p_atm = float(getattr(record, "p_atm", 101325.0) or 101325.0)
        remap_handoff = bool(getattr(record, "remap_for_2d", False))
        threshold = float(getattr(record, "threshold_overpressure_pa", 8000.0) or 8000.0)
        while proc.poll() is None:
            if watchdog and not triggered:
                path = _watchdog_pressure_path(case_dir)
                sample = _last_watchdog_sample(path) if path else None
                if sample:
                    if remap_handoff:
                        reached = primary_shock_at_probe(sample[1], p_atm)
                    else:
                        reached = overpressure_arrived(
                            sample[1], p_atm=p_atm, threshold_pa=threshold
                        )
                    if reached:
                        request_write_now(case_dir)
                        triggered = True
            time.sleep(0.25)
        wall = time.perf_counter() - t0
    text = ""
    try:
        text = open(out_path, encoding="utf-8", errors="replace").read()[-4000:]
    except OSError:
        pass
    crashed = "Floating point exception" in text or "FOAM FATAL" in text
    return {
        "case_dir": case_dir,
        "linux_path": mapped.linux_path,
        "returncode": 1 if crashed else proc.returncode,
        "wall_s": wall,
        "safe": safe,
        "watchdog_triggered": triggered,
        "crashed": crashed,
    }


def generate_case(
    *,
    prefix: str,
    inputs,
) -> dict:
    svc = SimulationService(base_projects_path=WORK, openfoam_bashrc=BASHRC)
    name = svc.make_case_name(prefix)
    t_gen = time.perf_counter()
    case_dir = svc.generate_case(name, inputs)
    return {
        "name": name,
        "case_dir": case_dir,
        "generate_s": time.perf_counter() - t_gen,
    }


def run_case_command(
    *,
    case_dir: str,
    log_dir: str,
    prefix: str,
    command: str,
) -> dict:
    argv, mapped, safe = build_case_command_argv(
        case_dir, command, openfoam_bashrc=BASHRC
    )
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"{prefix}_command.log")
    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT)
    text = ""
    try:
        text = open(out_path, encoding="utf-8", errors="replace").read()[-4000:]
    except OSError:
        pass
    crashed = "Floating point exception" in text or "FOAM FATAL" in text
    return {
        "case_dir": case_dir,
        "linux_path": mapped.linux_path,
        "returncode": 1 if crashed else proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "safe": safe,
        "crashed": crashed,
    }


def generate_and_run(
    *,
    prefix: str,
    inputs,
    log_dir: str,
    watchdog: bool | None = None,
) -> dict:
    svc = SimulationService(base_projects_path=WORK, openfoam_bashrc=BASHRC)
    name = svc.make_case_name(prefix)
    t_gen = time.perf_counter()
    case_dir = svc.generate_case(name, inputs)
    gen_s = time.perf_counter() - t_gen
    use_watchdog = bool(getattr(inputs, "remap_for_2d", False)) if watchdog is None else bool(watchdog)
    run = run_allrun_with_optional_watchdog(
        case_dir=case_dir,
        log_dir=log_dir,
        prefix=prefix,
        watchdog=use_watchdog,
    )
    if bool(getattr(inputs, "remap_for_2d", False)) and run.get("returncode") == 0:
        from remap_snapshot_1d import latest_complete_time_dir, write_snapshot_from_time_dir

        time_label = latest_complete_time_dir(case_dir)
        if time_label:
            try:
                physical = float(time_label)
            except ValueError:
                physical = None
            write_snapshot_from_time_dir(
                case_dir, time_label, physical_time=physical
            )
    payload = {
        "name": name,
        "generate_s": gen_s,
        **run,
    }
    with open(os.path.join(log_dir, f"{prefix}_result.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
