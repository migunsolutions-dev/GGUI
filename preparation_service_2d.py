"""Pure (non-Qt) preparation helpers for Cylindrical–2D workflows.

Heavy work used by ``PreparationWorker`` lives here so the GUI only
orchestrates workers and applies results on the GUI thread.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from case_loader_2d import inspect_imported_axisymmetric_case
from external_case_workflow_2d import (
    CopyVerificationError,
    ImportMode2D,
    create_automatic_working_copy,
    inventory_case,
)
from generator_2d import Generator2D
from models_2d import CaseInputs2D
from wsl_runtime import WslCancelToken, WslRunResult, run_wsl_command


class PreparationCancelled(Exception):
    """Raised when a cancel token is observed during long preparation work."""


@dataclass
class PreparationResult:
    ok: bool
    cancelled: bool = False
    failed_step: str = ""
    error: str = ""
    step_results: List[WslRunResult] = field(default_factory=list)
    payload: Any = None


def _check_cancel(token: Optional[WslCancelToken]) -> None:
    if token is not None and token.cancelled:
        raise PreparationCancelled("Cancelled")


def _safe_rmtree(path: Optional[str], *, source_dir: Optional[str] = None) -> None:
    """Remove a generated/staging directory without touching the source case."""
    if not path:
        return
    try:
        target = os.path.normpath(path)
        if source_dir and os.path.normpath(source_dir) == target:
            return
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        elif os.path.isfile(target):
            os.unlink(target)
    except OSError:
        # Cleanup best-effort; caller already failed or cancelled.
        pass


def write_initialize_log(
    case_dir: str,
    command: str,
    result: WslRunResult,
    *,
    linux_path: str = "",
) -> str:
    """Write ``log.initialize`` from a WSL result; returns log path."""
    log_file = os.path.join(case_dir, "log.initialize")
    with open(log_file, "w", encoding="utf-8") as log:
        log.write("=" * 60 + "\n")
        log.write("Initialize Command Log\n")
        log.write("=" * 60 + "\n")
        if linux_path:
            log.write(f"Directory: {linux_path}\n")
        log.write(f"Command: {command}\n")
        log.write(f"Safe: {result.safe_command}\n")
        log.write("=" * 60 + "\n\n")
        if result.stdout:
            log.write("STDOUT:\n")
            log.write(result.stdout)
            log.write("\n\n")
        if result.stderr:
            log.write("STDERR:\n")
            log.write(result.stderr)
            log.write("\n\n")
        log.write("=" * 60 + "\n")
        if result.cancelled:
            outcome = "CANCELLED"
        elif result.timed_out:
            outcome = "TIMED OUT"
        elif result.ok:
            outcome = "SUCCESS"
        else:
            outcome = f"FAILED with exit code {result.exit_code}"
        log.write(f"Result: {outcome}\n")
        log.write("=" * 60 + "\n")
    return log_file


def run_initialization_wsl(
    case_dir: str,
    command: str,
    *,
    openfoam_bashrc: str,
    cancel_token: Optional[WslCancelToken] = None,
) -> WslRunResult:
    """Run the initialization utility chain with cancellation support."""
    from wsl_runtime import to_wsl_path_and_distro

    result = run_wsl_command(
        case_dir,
        str(command),
        openfoam_bashrc=openfoam_bashrc,
        cancel_token=cancel_token,
        quiet_source=False,
    )
    path = to_wsl_path_and_distro(case_dir)
    write_initialize_log(case_dir, command, result, linux_path=path.linux_path)
    return result


def mesh_ok_from_logs(case_dir: str) -> bool:
    for log_name in ("log.checkMesh", "log.initialize"):
        log_path = os.path.join(case_dir, log_name)
        if not os.path.isfile(log_path):
            continue
        with open(log_path, encoding="utf-8", errors="ignore") as handle:
            if "Mesh OK" in handle.read():
                return True
    return False


@dataclass
class NativePrepContext:
    inputs: CaseInputs2D
    case_root: str
    openfoam_bashrc: str
    make_case_name: Callable[[str], str]
    expected_cells: Optional[int] = None
    mapping_report: Any = None


def prepare_native_2d_case(
    ctx: NativePrepContext,
    cancel_token: WslCancelToken,
    progress: Callable[[str], None],
) -> PreparationResult:
    """Generate and initialize a native 2D case (no Qt)."""
    case_dir: Optional[str] = None
    try:
        _check_cancel(cancel_token)
        progress("Generating 2D case")
        case_name = ctx.make_case_name("Case_2D")
        generator = Generator2D(ctx.case_root, ctx.openfoam_bashrc)
        case_dir = generator.generate(case_name, ctx.inputs)
        _check_cancel(cancel_token)

        command = generator.initialization_command(ctx.inputs)
        # Surface the dominant utility name for status while the chain runs.
        if "blockMesh" in command:
            progress("Running blockMesh")
        if "setRefinedFields" in command:
            progress("Running setRefinedFields")
        elif "setFields" in command:
            progress("Running setFields")
        if "checkMesh" in command:
            progress("Running checkMesh")

        wsl_result = run_initialization_wsl(
            case_dir,
            command,
            openfoam_bashrc=ctx.openfoam_bashrc,
            cancel_token=cancel_token,
        )
        if wsl_result.cancelled:
            _safe_rmtree(case_dir)
            return PreparationResult(
                ok=False,
                cancelled=True,
                failed_step="initialize",
                error="Cancelled",
                payload={"case_dir": case_dir, "cleaned": True},
            )
        if not wsl_result.ok:
            return PreparationResult(
                ok=False,
                failed_step="initialize",
                error=(
                    "blockMesh / charge initialization / checkMesh failed. "
                    "See log.initialize."
                ),
                payload={"case_dir": case_dir, "command": command},
            )

        progress("Verifying mesh")
        check_ok = mesh_ok_from_logs(case_dir)
        if not check_ok and "checkMesh" in command:
            return PreparationResult(
                ok=False,
                failed_step="checkMesh",
                error="checkMesh did not report Mesh OK. See log.checkMesh / log.initialize.",
                payload={"case_dir": case_dir, "command": command},
            )

        return PreparationResult(
            ok=True,
            payload={
                "case_dir": case_dir,
                "command": command,
                "inputs": ctx.inputs,
                "check_ok": check_ok,
                "expected_cells": ctx.expected_cells,
                "mapping_report": ctx.mapping_report,
                "kind": "native_2d",
            },
        )
    except PreparationCancelled:
        _safe_rmtree(case_dir)
        return PreparationResult(
            ok=False,
            cancelled=True,
            error="Cancelled",
            payload={"case_dir": case_dir, "cleaned": True},
        )
    except Exception as exc:
        return PreparationResult(
            ok=False,
            failed_step="generate_or_initialize",
            error=str(exc),
            payload={"case_dir": case_dir},
        )


@dataclass
class ImportCopyContext:
    source_dir: str
    case_root: str
    repo_root: str


def prepare_imported_copy_and_inspect(
    ctx: ImportCopyContext,
    cancel_token: WslCancelToken,
    progress: Callable[[str], None],
) -> PreparationResult:
    """Inventory, copy, verify, and inspect an imported axisymmetric case."""
    staging_or_working: Optional[str] = None
    source = os.path.normpath(ctx.source_dir)
    try:
        _check_cancel(cancel_token)
        progress("Hashing source")
        before = inventory_case(source, cancel_token=cancel_token)
        _check_cancel(cancel_token)

        progress("Creating working copy")
        paths = create_automatic_working_copy(
            source,
            ctx.case_root,
            ctx.repo_root,
            cancel_token=cancel_token,
        )
        staging_or_working = paths.working_copy_dir
        _check_cancel(cancel_token)

        progress("Hashing working copy")
        # Destination verification already ran inside create_working_copy;
        # re-check source integrity after the copy transaction.
        progress("Verifying copy")
        after = inventory_case(source, cancel_token=cancel_token)
        if before != after:
            _safe_rmtree(paths.working_copy_dir, source_dir=source)
            return PreparationResult(
                ok=False,
                failed_step="verify_source_unchanged",
                error="Source case hashes changed during copy — aborting.",
                payload={"source_dir": source},
            )
        _check_cancel(cancel_token)

        progress("Inspecting imported case")
        state = inspect_imported_axisymmetric_case(
            paths.working_copy_dir,
            source_dir=paths.source_dir,
            working_copy_dir=paths.working_copy_dir,
            mode=ImportMode2D.IMPORTED_2D_UNINITIALIZED,
        )
        _check_cancel(cancel_token)

        return PreparationResult(
            ok=True,
            payload={
                "kind": "import_copy",
                "source_dir": paths.source_dir,
                "working_copy_dir": paths.working_copy_dir,
                "copy_method": paths.copy_method,
                "distro": paths.distro,
                "source_linux": paths.source_linux,
                "dest_linux": paths.dest_linux,
                "state": state,
                "source_inventory": before,
            },
        )
    except PreparationCancelled:
        _safe_rmtree(staging_or_working, source_dir=source)
        # Also clean any leftover .incomplete staging next to case_root.
        try:
            parent = os.path.normpath(ctx.case_root)
            if os.path.isdir(parent):
                for name in os.listdir(parent):
                    if ".incomplete" in name:
                        _safe_rmtree(os.path.join(parent, name), source_dir=source)
        except OSError:
            pass
        return PreparationResult(
            ok=False,
            cancelled=True,
            error="Cancelled",
            payload={"source_dir": source, "cleaned": True},
        )
    except CopyVerificationError as exc:
        _safe_rmtree(staging_or_working, source_dir=source)
        return PreparationResult(
            ok=False,
            failed_step="copy_verify",
            error=str(exc),
            payload={"source_dir": source, "case_root": ctx.case_root},
        )
    except Exception as exc:
        _safe_rmtree(staging_or_working, source_dir=source)
        return PreparationResult(
            ok=False,
            failed_step="import_copy",
            error=str(exc),
            payload={"source_dir": source, "case_root": ctx.case_root},
        )


@dataclass
class ImportedInitContext:
    source: str
    inputs: CaseInputs2D
    case_root: str
    openfoam_bashrc: str
    make_case_name: Callable[[str], str]


def prepare_imported_model_generation(
    ctx: ImportedInitContext,
    cancel_token: WslCancelToken,
    progress: Callable[[str], None],
) -> PreparationResult:
    """Regenerate a fresh GGUI case from an editable imported model."""
    case_dir: Optional[str] = None
    source = os.path.normpath(ctx.source)
    try:
        _check_cancel(cancel_token)
        progress("Generating 2D case")
        src_base = os.path.basename(source.rstrip("\\/")) or "imported"
        case_name = ctx.make_case_name(f"Case_2D_from_{src_base}")
        generator = Generator2D(ctx.case_root, ctx.openfoam_bashrc)
        case_dir = generator.generate(case_name, ctx.inputs)
        if os.path.normpath(case_dir) == source:
            raise RuntimeError("Refusing to generate into the source directory.")
        _check_cancel(cancel_token)

        command = generator.initialization_command(ctx.inputs)
        if "blockMesh" in command:
            progress("Running blockMesh")
        if "setRefinedFields" in command:
            progress("Running setRefinedFields")
        elif "setFields" in command:
            progress("Running setFields")
        if "checkMesh" in command:
            progress("Running checkMesh")

        wsl_result = run_initialization_wsl(
            case_dir,
            command,
            openfoam_bashrc=ctx.openfoam_bashrc,
            cancel_token=cancel_token,
        )
        if wsl_result.cancelled:
            _safe_rmtree(case_dir, source_dir=source)
            return PreparationResult(
                ok=False,
                cancelled=True,
                failed_step="initialize",
                error="Cancelled",
                payload={"case_dir": case_dir, "source": source, "cleaned": True},
            )
        if not wsl_result.ok:
            return PreparationResult(
                ok=False,
                failed_step="initialize",
                error=(
                    "blockMesh / setRefinedFields / checkMesh failed in the "
                    "generated GGUI case. See log.initialize. Source was not modified."
                ),
                payload={"case_dir": case_dir, "command": command, "source": source},
            )

        progress("Verifying mesh")
        check_ok = mesh_ok_from_logs(case_dir)
        if not check_ok:
            return PreparationResult(
                ok=False,
                failed_step="checkMesh",
                error=(
                    "checkMesh did not report Mesh OK in the generated GGUI case.\n"
                    "Source was not modified. See log.checkMesh / log.initialize."
                ),
                payload={"case_dir": case_dir, "command": command, "source": source},
            )

        return PreparationResult(
            ok=True,
            payload={
                "kind": "imported_init",
                "case_dir": case_dir,
                "command": command,
                "source": source,
                "inputs": ctx.inputs,
                "check_ok": check_ok,
            },
        )
    except PreparationCancelled:
        _safe_rmtree(case_dir, source_dir=source)
        return PreparationResult(
            ok=False,
            cancelled=True,
            error="Cancelled",
            payload={"case_dir": case_dir, "source": source, "cleaned": True},
        )
    except Exception as exc:
        return PreparationResult(
            ok=False,
            failed_step="imported_init",
            error=str(exc),
            payload={"case_dir": case_dir, "source": source},
        )
