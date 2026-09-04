"""Pure solver execution planning (no PyQt).

Builds non-destructive blastFoam command plans for fresh, initialized, and
resume intents. Callers supply an already-generated case directory.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


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


FINAL_RECONSTRUCT_CMD = "reconstructPar -latestTime > log.reconstructFinal 2>&1"


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
        if resume:
            if len(set(latest_values)) != 1:
                raise ExecutionPreparationError(
                    "Processor directories do not share one consistent latest saved time; "
                    "reconstruct or repair the case before resuming."
                )
            if latest_values and latest_values[0] is not None:
                processor_latest = latest_values[0]
        elif latest_values and latest_values[0] is not None and len(set(latest_values)) == 1:
            processor_latest = latest_values[0]

    latest: Optional[float] = None
    if resume:
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
            prep = "decomposePar -force -latestTime > log.decomposeParSolver 2>&1 && "
    elif not processor_dirs:
        decompose_opt = "-force -latestTime" if resume else "-force"
        prep = f"decomposePar {decompose_opt} > log.decomposeParSolver 2>&1 && "
    start_mode = "latestTime" if resume else "startTime"
    cmd = (
        f"set -o pipefail; foamDictionary system/controlDict -entry startFrom -set {start_mode} "
        f"> log.prepareSolver 2>&1 && {prep}"
        f"mpirun -np {cores} blastFoam -parallel 2>&1 | tee log.blastFoam"
    )
    return ExecutionPlan(intent, cmd, latest)
