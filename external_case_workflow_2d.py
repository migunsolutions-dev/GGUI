"""Imported axisymmetric working-case workflow for Cylindrical–2D.

Automatic working-case creation, whitelisted preparation, and proven controlDict
writers. Isolated from generator_2d / native CaseInputs2D generation.

Working-case copies into a WSL/OpenFOAM Work root reuse the same path conversion
and WSL process invocation pattern as General 3D (`SolverRunner._win_unc_to_wsl_*`).
"""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from case_topology import CaseDimension, classify_case_topology


class ImportMode2D(str, Enum):
    NATIVE_GGUI_2D = "NATIVE_GGUI_2D"
    IMPORTED_2D_UNINITIALIZED = "IMPORTED_2D_UNINITIALIZED"
    IMPORTED_2D_INITIALIZING = "IMPORTED_2D_INITIALIZING"
    IMPORTED_2D_READY = "IMPORTED_2D_READY"
    IMPORTED_2D_RUNNING = "IMPORTED_2D_RUNNING"
    IMPORTED_2D_FAILED = "IMPORTED_2D_FAILED"


# Backward-compatible aliases used by older call sites during transition.
ExternalLifecycle = ImportMode2D  # type: ignore[misc,assignment]


WHITELISTED_UTILITIES = frozenset(
    {"blockMesh", "setFields", "setRefinedFields", "checkMesh"}
)

_FORBIDDEN_ALLRUN_TOKENS = frozenset(
    {
        "blastFoam",
        "paraFoam",
        "mpirun",
        "decomposePar",
        "reconstructPar",
        "Allrun",
        "bash",
        "sh",
        "rm",
        "curl",
        "wget",
        "python",
        "perl",
    }
)

REQUIRED_POLYMESH_FILES = ("points", "faces", "owner", "neighbour", "boundary")

SUPPORTED_CONTROL_WRITERS = frozenset(
    {"endTime", "deltaT", "maxCo", "writeControl", "writeInterval"}
)

# Windows Mark-of-the-Web ADS may materialize on 9P as sibling files; not case content.
_ZONE_ID_MARKERS = ("Zone.Identifier", "\uf03aZone.Identifier", ":Zone.Identifier")

RunUtilityFn = Callable[[str, str], Tuple[int, str]]


@dataclass(frozen=True)
class InventoryEntry:
    """Semantic content identity for one relative path (no timestamps/ACL/owners)."""

    rel: str
    kind: str  # "file" | "dir" | "symlink"
    size: Optional[int] = None
    sha256: Optional[str] = None
    link_target: Optional[str] = None


@dataclass
class CaseInventory:
    files: Dict[str, str]  # relative posix path -> sha256 (compat)
    dirs: Tuple[str, ...]
    entries: Dict[str, InventoryEntry] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CaseInventory):
            return NotImplemented
        if self.entries and other.entries:
            return self.entries == other.entries
        return self.files == other.files and self.dirs == other.dirs


@dataclass
class InventoryMismatch:
    category: str
    relative_path: str
    detail: str
    source_value: str = ""
    destination_value: str = ""


@dataclass
class InventoryComparison:
    ok: bool
    mismatches: Tuple[InventoryMismatch, ...] = ()
    report_path: str = ""

    @property
    def first(self) -> Optional[InventoryMismatch]:
        return self.mismatches[0] if self.mismatches else None


class CopyVerificationError(RuntimeError):
    """Raised when source/destination inventories differ after copy."""

    def __init__(self, message: str, comparison: InventoryComparison):
        super().__init__(message)
        self.comparison = comparison


@dataclass
class UtilityResult:
    name: str
    exit_code: int
    log_path: str
    log_excerpt: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass
class PrepareResult:
    ok: bool
    mode: ImportMode2D
    commands: Tuple[str, ...]
    results: Tuple[UtilityResult, ...] = ()
    reason: str = ""
    cell_count: Optional[int] = None
    cell_count_source: str = "none"
    check_mesh_ok: bool = False
    wedge_patch_names: Tuple[str, ...] = ()
    wedge_half_angle_deg: Optional[float] = None
    classification: str = ""
    charge_cell_count: Optional[int] = None
    fields_verified: Tuple[str, ...] = ()
    mesh_owner_path: str = ""

    @property
    def lifecycle(self) -> ImportMode2D:
        return self.mode


@dataclass
class WorkingCopyPaths:
    source_dir: str
    working_copy_dir: str
    copy_method: str = ""
    source_linux: str = ""
    dest_linux: str = ""
    distro: str = ""


@dataclass
class ControlDictWriteResult:
    ok: bool
    changed: Tuple[str, ...] = ()
    readback: Dict[str, str] = field(default_factory=dict)
    reason: str = ""


def _is_zone_identifier_rel(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return any(marker in name for marker in _ZONE_ID_MARKERS)


def _is_wsl_unc_path(path: str) -> bool:
    p = (path or "").replace("/", "\\")
    if not p.startswith("\\\\"):
        return False
    parts = [x for x in p.split("\\") if x]
    return len(parts) >= 2 and parts[0].lower() in ("wsl.localhost", "wsl$")


def inventory_case(case_dir: str) -> CaseInventory:
    """Semantic inventory: relative paths, types, sizes, SHA-256 (no metadata)."""
    root = Path(case_dir)
    files: Dict[str, str] = {}
    dirs: List[str] = []
    entries: Dict[str, InventoryEntry] = {}
    if not root.is_dir():
        return CaseInventory(files={}, dirs=(), entries={})
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if _is_zone_identifier_rel(rel):
            continue
        try:
            if path.is_symlink():
                target = os.readlink(path)
                entries[rel] = InventoryEntry(
                    rel=rel, kind="symlink", link_target=str(target)
                )
                continue
            if path.is_dir():
                dirs.append(rel + "/")
                entries[rel + "/"] = InventoryEntry(rel=rel + "/", kind="dir")
            elif path.is_file():
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                files[rel] = digest
                entries[rel] = InventoryEntry(
                    rel=rel, kind="file", size=len(data), sha256=digest
                )
        except OSError:
            continue
    return CaseInventory(files=files, dirs=tuple(dirs), entries=entries)


def compare_inventories(
    source: CaseInventory,
    destination: CaseInventory,
    *,
    write_report: bool = True,
    report_dir: Optional[str] = None,
) -> InventoryComparison:
    """Compare semantic inventories; optionally write a full mismatch report."""
    mismatches: List[InventoryMismatch] = []
    src_keys = set(source.entries) if source.entries else set(source.files) | set(source.dirs)
    dst_keys = (
        set(destination.entries)
        if destination.entries
        else set(destination.files) | set(destination.dirs)
    )

    for rel in sorted(src_keys - dst_keys):
        mismatches.append(
            InventoryMismatch(
                category="missing_from_destination",
                relative_path=rel,
                detail="Present in source, absent in destination",
                source_value=str(source.entries.get(rel) or source.files.get(rel, "")),
            )
        )
    for rel in sorted(dst_keys - src_keys):
        mismatches.append(
            InventoryMismatch(
                category="extra_in_destination",
                relative_path=rel,
                detail="Present in destination, absent in source",
                destination_value=str(
                    destination.entries.get(rel) or destination.files.get(rel, "")
                ),
            )
        )
    for rel in sorted(src_keys & dst_keys):
        if source.entries and destination.entries:
            a = source.entries[rel]
            b = destination.entries[rel]
            if a.kind != b.kind:
                mismatches.append(
                    InventoryMismatch(
                        "item_type_change",
                        rel,
                        f"Kind changed: {a.kind} → {b.kind}",
                        a.kind,
                        b.kind,
                    )
                )
                continue
            if a.kind == "file":
                if a.size != b.size:
                    mismatches.append(
                        InventoryMismatch(
                            "size_mismatch",
                            rel,
                            f"Byte size differs: {a.size} → {b.size}",
                            str(a.size),
                            str(b.size),
                        )
                    )
                elif a.sha256 != b.sha256:
                    mismatches.append(
                        InventoryMismatch(
                            "sha256_mismatch",
                            rel,
                            "SHA-256 content hash differs",
                            a.sha256 or "",
                            b.sha256 or "",
                        )
                    )
            elif a.kind == "symlink" and a.link_target != b.link_target:
                mismatches.append(
                    InventoryMismatch(
                        "symlink_target_mismatch",
                        rel,
                        "Symlink target differs",
                        str(a.link_target),
                        str(b.link_target),
                    )
                )
        elif rel in source.files and rel in destination.files:
            if source.files[rel] != destination.files[rel]:
                mismatches.append(
                    InventoryMismatch(
                        "sha256_mismatch",
                        rel,
                        "SHA-256 content hash differs",
                        source.files[rel],
                        destination.files[rel],
                    )
                )

    report_path = ""
    if write_report and mismatches:
        report_dir = report_dir or tempfile.gettempdir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"ggui_import_copy_mismatch_{ts}.txt")
        lines = [
            "GGUI imported working-case copy verification failed",
            f"Mismatch count: {len(mismatches)}",
            "",
        ]
        for item in mismatches:
            lines.append(f"[{item.category}] {item.relative_path}: {item.detail}")
            if item.source_value:
                lines.append(f"  source: {item.source_value}")
            if item.destination_value:
                lines.append(f"  destination: {item.destination_value}")
        Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return InventoryComparison(
        ok=not mismatches,
        mismatches=tuple(mismatches),
        report_path=report_path,
    )


def format_copy_verification_error(comparison: InventoryComparison) -> str:
    first = comparison.first
    if first is None:
        return "Working-copy verification failed (no details)."
    lines = [
        "Working-copy verification failed.",
        f"First mismatch: {first.relative_path}",
        f"Reason: {first.detail}",
    ]
    if first.source_value or first.destination_value:
        lines.append(f"Expected: {first.source_value}")
        lines.append(f"Actual:   {first.destination_value}")
    if comparison.report_path:
        lines.append(f"Full report: {comparison.report_path}")
    return "\n".join(lines)


def sanitize_case_stem(name: str) -> str:
    stem = re.sub(r"[^\w.\-]+", "_", name.strip()) or "case"
    return stem[:80]


def make_imported_working_case_name(source_dir: str, when: Optional[datetime] = None) -> str:
    """Production naming: Case_2D_imported_<stem>_YYYYMMDD_HHMMSS."""
    stem = sanitize_case_stem(Path(source_dir).name)
    ts = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"Case_2D_imported_{stem}_{ts}"


def parse_allrun_preprocess_sequence(case_dir: str) -> Tuple[str, ...]:
    """Extract whitelisted preprocessing utilities from Allrun, in order."""
    allrun = Path(case_dir) / "Allrun"
    if not allrun.is_file():
        raise ValueError(f"No Allrun found in {case_dir}")
    text = allrun.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    body = "\n".join(lines)

    for token in _FORBIDDEN_ALLRUN_TOKENS:
        if token == "blastFoam":
            continue
        if re.search(rf"(?<![\w.]){re.escape(token)}(?![\w.])", body):
            if token == "paraFoam":
                continue
            if token in {"bash", "sh"} and "bash -" not in body and "./" not in body:
                continue

    sequence: List[str] = []
    for match in re.finditer(r"runApplication\s+(.+)", body):
        args = match.group(1).strip()
        if not args:
            continue
        first = args.split()[0]
        if first.startswith("$") or first in {"getApplication", "blastFoam"}:
            continue
        if first == "paraFoam":
            continue
        if first in WHITELISTED_UTILITIES:
            if first not in sequence:
                sequence.append(first)
            continue
        raise ValueError(
            f"Allrun references unrecognized utility '{first}'. "
            f"Only {sorted(WHITELISTED_UTILITIES)} are allowed."
        )

    if not sequence:
        raise ValueError(
            "Allrun contains no whitelisted preprocessing utilities "
            f"(expected one of {sorted(WHITELISTED_UTILITIES)})"
        )
    return tuple(sequence)


def preparation_commands_for_case(case_dir: str) -> Tuple[str, ...]:
    """Proven preprocess sequence from Allrun, plus checkMesh validation."""
    seq = list(parse_allrun_preprocess_sequence(case_dir))
    if "checkMesh" not in seq:
        seq.append("checkMesh")
    return tuple(seq)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_working_copy_destination(
    source_dir: str,
    dest_dir: str,
    repo_root: str,
) -> None:
    source = Path(source_dir)
    dest = Path(dest_dir)
    repo = Path(repo_root)

    try:
        source_r = source.resolve()
    except OSError:
        source_r = Path(os.path.normpath(str(source)))
    try:
        dest_r = dest.resolve()
    except OSError:
        dest_r = Path(os.path.normpath(str(dest)))
    try:
        repo_r = repo.resolve()
    except OSError:
        repo_r = Path(os.path.normpath(str(repo)))

    if not source_r.is_dir():
        raise ValueError(f"Source case does not exist: {source_r}")
    if dest_r == source_r:
        raise ValueError("Destination cannot be the source case directory.")
    if _is_relative_to(dest_r, source_r):
        raise ValueError("Destination cannot be inside the source case.")
    if _is_relative_to(dest_r, repo_r) or dest_r == repo_r:
        raise ValueError(
            "Destination cannot be inside the Git repository "
            f"({repo_r}). Choose a folder outside the worktree."
        )
    if dest.exists():
        if dest.is_file():
            raise ValueError(f"Destination exists as a file: {dest}")
        remaining = list(dest.iterdir())
        if remaining:
            raise ValueError(
                f"Destination already exists and is not empty: {dest}"
            )
    parent = dest.parent
    if not parent.is_dir():
        raise ValueError(f"Destination parent does not exist: {parent}")


def _wsl_path_and_distro(win_path: str) -> Tuple[Optional[str], str]:
    """Reuse General 3D / SolverRunner UNC→Linux conversion."""
    from solver_runner import SolverRunner

    return SolverRunner._win_unc_to_wsl_path_and_distro(win_path)


def _run_wsl_argv(distro: Optional[str], script: str) -> subprocess.CompletedProcess:
    """Invoke WSL with bash -lc (same pattern as General 3D)."""
    if os.name == "nt":
        if distro:
            args = ["wsl", "-d", distro, "--", "bash", "-lc", script]
        else:
            args = ["wsl", "bash", "-lc", script]
    else:
        args = ["bash", "-lc", script]
    return subprocess.run(args, check=False, capture_output=True, text=True)


def _copy_tree_via_wsl(
    source_dir: str,
    staging_dir: str,
    dest_dir: str,
) -> Tuple[str, str, str]:
    """Copy with Linux-side cp -a; returns (distro, source_linux, dest_linux)."""
    src_distro, src_linux = _wsl_path_and_distro(source_dir)
    dest_distro, dest_linux = _wsl_path_and_distro(dest_dir)
    _, staging_linux = _wsl_path_and_distro(staging_dir)
    distro = dest_distro or src_distro
    if not distro and os.name == "nt" and _is_wsl_unc_path(dest_dir):
        raise RuntimeError(
            f"Could not resolve WSL distro from destination {dest_dir!r}"
        )

    parent_linux = os.path.dirname(staging_linux.rstrip("/")) or "/"
    script = (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(parent_linux)}; "
        f"rm -rf {shlex.quote(staging_linux)}; "
        f"cp -a {shlex.quote(src_linux)} {shlex.quote(staging_linux)}; "
        f"test -d {shlex.quote(staging_linux)}"
    )
    completed = _run_wsl_argv(distro, script)
    if completed.returncode != 0:
        raise RuntimeError(
            "WSL cp -a failed "
            f"(distro={distro!r}, src={src_linux!r}, staging={staging_linux!r}):\n"
            f"{(completed.stderr or completed.stdout or '').strip()}"
        )
    return distro or "", src_linux, dest_linux


def _finalize_via_wsl(
    staging_dir: str,
    dest_dir: str,
    distro: Optional[str],
) -> None:
    _, staging_linux = _wsl_path_and_distro(staging_dir)
    _, dest_linux = _wsl_path_and_distro(dest_dir)
    script = (
        "set -euo pipefail; "
        f"rm -rf {shlex.quote(dest_linux)}; "
        f"mv {shlex.quote(staging_linux)} {shlex.quote(dest_linux)}; "
        f"test -d {shlex.quote(dest_linux)}"
    )
    completed = _run_wsl_argv(distro, script)
    if completed.returncode != 0:
        raise RuntimeError(
            "WSL mv finalize failed:\n"
            f"{(completed.stderr or completed.stdout or '').strip()}"
        )


def _remove_via_wsl(path: str, distro: Optional[str]) -> None:
    if not path:
        return
    _, linux = _wsl_path_and_distro(path)
    _run_wsl_argv(distro, f"rm -rf {shlex.quote(linux)}")


def create_working_copy(
    source_dir: str,
    dest_dir: str,
    repo_root: str,
    *,
    diagnostic_dir: Optional[str] = None,
) -> WorkingCopyPaths:
    """Transactional copy: stage → verify → finalize; never touch the source.

    Destinations under ``\\\\wsl.localhost\\...`` / ``\\\\wsl$\\...`` use Linux-side
    ``cp -a`` via the same SolverRunner path conversion as General 3D. Windows
    ``shutil.copytree`` into that UNC root materializes Mark-of-the-Web
    ``Zone.Identifier`` ADS as extra files and fails verification.
    """
    validate_working_copy_destination(source_dir, dest_dir, repo_root)
    source = Path(os.path.normpath(source_dir))
    dest = Path(os.path.normpath(dest_dir))
    parent = dest.parent
    staging = parent / (dest.name + ".incomplete")
    n = 0
    while staging.exists():
        n += 1
        staging = parent / (dest.name + f".incomplete{n}")

    use_wsl = _is_wsl_unc_path(str(dest)) or _is_wsl_unc_path(str(parent))
    created_staging = False
    distro = ""
    src_linux = ""
    dest_linux = ""
    copy_method = ""
    try:
        if dest.exists() and dest.is_dir() and not any(dest.iterdir()):
            dest.rmdir()
        if dest.exists():
            raise ValueError(f"Destination already exists: {dest}")

        if use_wsl:
            copy_method = "wsl_cp"
            distro, src_linux, dest_linux = _copy_tree_via_wsl(
                str(source), str(staging), str(dest)
            )
            created_staging = True
        else:
            copy_method = "windows_copytree"
            shutil.copytree(str(source), str(staging))
            created_staging = True

        # Verify immediately after copy — before *.orig restore or OF mutation.
        src_inv = inventory_case(str(source))
        dst_inv = inventory_case(str(staging))
        comparison = compare_inventories(
            src_inv,
            dst_inv,
            write_report=True,
            report_dir=diagnostic_dir or tempfile.gettempdir(),
        )
        if not comparison.ok:
            raise CopyVerificationError(
                format_copy_verification_error(comparison),
                comparison,
            )

        if use_wsl:
            _finalize_via_wsl(str(staging), str(dest), distro or None)
        else:
            os.rename(str(staging), str(dest))
        created_staging = False
        return WorkingCopyPaths(
            source_dir=str(source),
            working_copy_dir=str(dest),
            copy_method=copy_method,
            source_linux=src_linux,
            dest_linux=dest_linux,
            distro=distro,
        )
    except Exception:
        if created_staging and staging.exists():
            if use_wsl:
                try:
                    _remove_via_wsl(str(staging), distro or None)
                except Exception:
                    shutil.rmtree(str(staging), ignore_errors=True)
            else:
                try:
                    resolved = staging.resolve()
                    if str(resolved) != str(source.resolve()) and not _is_relative_to(
                        source.resolve(), resolved
                    ):
                        shutil.rmtree(resolved, ignore_errors=False)
                except OSError:
                    shutil.rmtree(str(staging), ignore_errors=True)
        raise


def create_automatic_working_copy(
    source_dir: str,
    case_root: str,
    repo_root: str,
    *,
    when: Optional[datetime] = None,
    diagnostic_dir: Optional[str] = None,
) -> WorkingCopyPaths:
    """Create a unique persistent working case under the production case root."""
    if not case_root:
        raise ValueError("case_root is empty")
    try:
        if not os.path.isdir(case_root):
            if _is_wsl_unc_path(case_root):
                distro, linux = _wsl_path_and_distro(case_root)
                completed = _run_wsl_argv(distro, f"mkdir -p {shlex.quote(linux)}")
                if completed.returncode != 0 or not os.path.isdir(case_root):
                    raise OSError(
                        completed.stderr or f"mkdir failed for {case_root}"
                    )
            else:
                os.makedirs(case_root, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Cannot create/access case root {case_root!r}: {exc}"
        ) from exc
    base_name = make_imported_working_case_name(source_dir, when=when)
    dest = os.path.join(case_root, base_name)
    n = 0
    while os.path.exists(dest):
        n += 1
        dest = os.path.join(case_root, f"{base_name}_{n}")
    return create_working_copy(
        source_dir, dest, repo_root, diagnostic_dir=diagnostic_dir
    )


def restore_zero_orig_fields(case_dir: str) -> List[str]:
    """Python-side restore of 0/*.orig → 0/* when working field files are missing."""
    zero = Path(case_dir) / "0"
    restored: List[str] = []
    if not zero.is_dir():
        return restored
    for path in sorted(zero.glob("*.orig")):
        target = zero / path.name[: -len(".orig")]
        if target.exists():
            continue
        shutil.copy2(path, target)
        restored.append(target.name)
    return restored


def _polymesh_complete(case_dir: str) -> bool:
    mesh = Path(case_dir) / "constant" / "polyMesh"
    if not mesh.is_dir():
        return False
    return all((mesh / name).is_file() for name in REQUIRED_POLYMESH_FILES)


def _newest_mesh_owner(case_dir: str) -> Tuple[Optional[Path], str]:
    """Return (owner_path, source_tag) preferring newest time polyMesh."""
    best_t = None
    best_owner: Optional[Path] = None
    root = Path(case_dir)
    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                tval = float(child.name)
            except ValueError:
                continue
            owner = child / "polyMesh" / "owner"
            if owner.is_file() and (best_t is None or tval >= best_t):
                best_t = tval
                best_owner = owner
    except OSError:
        pass
    if best_owner is not None:
        return best_owner, "time_polyMesh"
    const = root / "constant" / "polyMesh" / "owner"
    if const.is_file():
        return const, "constant_polyMesh"
    return None, "none"


def _count_owner_cells(case_dir: str) -> Tuple[Optional[int], str, str]:
    """Authoritative cell count from newest mesh owner. Returns (n, source, path)."""
    owner, source = _newest_mesh_owner(case_dir)
    if owner is None:
        return None, "none", ""
    try:
        text = owner.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, source, str(owner)
    note_cells = re.search(r"\bnCells:\s*(\d+)", text)
    if note_cells:
        return int(note_cells.group(1)), source, str(owner)
    from axisymmetric_viewer import AxisymmetricViewerWidget

    counted = AxisymmetricViewerWidget.count_owner_cells(str(owner.parent))
    return counted, source, str(owner)


def _parse_charge_cells_from_setrefined_log(case_dir: str) -> Optional[int]:
    log_path = Path(case_dir) / "log.setRefinedFields"
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    matches = re.findall(r"Selected\s+(\d+)\s+cells,\s+\d+\s+faces", text)
    if matches:
        return int(matches[-1])
    return None


def _check_mesh_ok(log_text: str) -> bool:
    return bool(re.search(r"\bMesh OK\b", log_text))


def _count_nonzero_alpha(case_dir: str, time_name: str = "0") -> Optional[int]:
    for name in ("alpha.c4", "alpha.c4.orig"):
        path = Path(case_dir) / time_name / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        m = re.search(
            r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)",
            text,
            re.DOTALL,
        )
        if not m:
            if re.search(r"internalField\s+uniform\s+0\s*;", text):
                return 0
            return None
        n = int(m.group(1))
        body = m.group(2)
        values = re.findall(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", body)
        if len(values) < n:
            return None
        return sum(1 for v in values[:n] if float(v) > 1e-12)
    return None


def prepare_working_copy(
    working_copy_dir: str,
    source_dir: str,
    run_utility: RunUtilityFn,
    commands: Optional[Sequence[str]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> PrepareResult:
    """Run whitelisted preprocess utilities only inside the working copy."""
    wc = os.path.normpath(working_copy_dir)
    src = os.path.normpath(source_dir)
    if os.path.normpath(wc) == os.path.normpath(src):
        return PrepareResult(
            ok=False,
            mode=ImportMode2D.IMPORTED_2D_FAILED,
            commands=(),
            reason="Refusing to prepare the source case in place.",
        )

    try:
        commands = tuple(commands or preparation_commands_for_case(wc))
    except ValueError as exc:
        return PrepareResult(
            ok=False,
            mode=ImportMode2D.IMPORTED_2D_FAILED,
            commands=(),
            reason=str(exc),
        )

    for name in commands:
        if name not in WHITELISTED_UTILITIES:
            return PrepareResult(
                ok=False,
                mode=ImportMode2D.IMPORTED_2D_FAILED,
                commands=commands,
                reason=f"Refusing non-whitelisted utility: {name}",
            )

    restore_zero_orig_fields(wc)

    results: List[UtilityResult] = []
    for name in commands:
        if progress:
            progress(name)
        code, log_text = run_utility(wc, name)
        log_path = os.path.join(wc, f"log.{name}")
        results.append(
            UtilityResult(
                name=name,
                exit_code=code,
                log_path=log_path,
                log_excerpt=(log_text or "")[-4000:],
            )
        )
        if code != 0:
            return PrepareResult(
                ok=False,
                mode=ImportMode2D.IMPORTED_2D_FAILED,
                commands=commands,
                results=tuple(results),
                reason=f"{name} failed with exit code {code}",
            )

        if name == "blockMesh":
            if not _polymesh_complete(wc):
                return PrepareResult(
                    ok=False,
                    mode=ImportMode2D.IMPORTED_2D_FAILED,
                    commands=commands,
                    results=tuple(results),
                    reason="blockMesh finished but constant/polyMesh is incomplete",
                )
            classification = classify_case_topology(wc)
            if classification.classification != CaseDimension.AXISYMMETRIC_WEDGE:
                return PrepareResult(
                    ok=False,
                    mode=ImportMode2D.IMPORTED_2D_FAILED,
                    commands=commands,
                    results=tuple(results),
                    reason=(
                        "Post-mesh topology is not AXISYMMETRIC_WEDGE: "
                        f"{classification.classification.value} ({classification.reason})"
                    ),
                    classification=classification.classification.value,
                )
            if classification.evidence.source not in ("polyMesh/boundary", "both"):
                return PrepareResult(
                    ok=False,
                    mode=ImportMode2D.IMPORTED_2D_FAILED,
                    commands=commands,
                    results=tuple(results),
                    reason="Expected polyMesh/boundary evidence after blockMesh",
                    classification=classification.classification.value,
                )

    if not _polymesh_complete(wc):
        return PrepareResult(
            ok=False,
            mode=ImportMode2D.IMPORTED_2D_FAILED,
            commands=commands,
            results=tuple(results),
            reason="polyMesh missing after preparation",
        )

    classification = classify_case_topology(wc)
    if classification.classification != CaseDimension.AXISYMMETRIC_WEDGE:
        return PrepareResult(
            ok=False,
            mode=ImportMode2D.IMPORTED_2D_FAILED,
            commands=commands,
            results=tuple(results),
            reason=f"Final classification failed: {classification.reason}",
            classification=classification.classification.value,
        )

    check_logs = [r for r in results if r.name == "checkMesh"]
    check_ok = bool(check_logs) and check_logs[-1].ok and _check_mesh_ok(
        check_logs[-1].log_excerpt
        or (
            Path(check_logs[-1].log_path).read_text(encoding="utf-8", errors="ignore")
            if os.path.isfile(check_logs[-1].log_path)
            else ""
        )
    )
    if not check_ok:
        if check_logs and os.path.isfile(check_logs[-1].log_path):
            full = Path(check_logs[-1].log_path).read_text(
                encoding="utf-8", errors="ignore"
            )
            check_ok = check_logs[-1].ok and _check_mesh_ok(full)
        if not check_ok:
            return PrepareResult(
                ok=False,
                mode=ImportMode2D.IMPORTED_2D_FAILED,
                commands=commands,
                results=tuple(results),
                reason="checkMesh did not report Mesh OK",
                classification=classification.classification.value,
            )

    cell_count, cell_source, owner_path = _count_owner_cells(wc)
    charge_cells = _count_nonzero_alpha(wc, "0")
    if charge_cells is None:
        charge_cells = _parse_charge_cells_from_setrefined_log(wc)
    fields: List[str] = []
    zero = Path(wc) / "0"
    if zero.is_dir():
        for p in zero.iterdir():
            if p.is_file() and not p.name.startswith("."):
                base = p.name[: -len(".orig")] if p.name.endswith(".orig") else p.name
                if base not in fields:
                    fields.append(base)

    return PrepareResult(
        ok=True,
        mode=ImportMode2D.IMPORTED_2D_READY,
        commands=commands,
        results=tuple(results),
        cell_count=cell_count,
        cell_count_source=cell_source,
        check_mesh_ok=True,
        wedge_patch_names=tuple(classification.evidence.wedge_patch_names),
        wedge_half_angle_deg=classification.evidence.wedge_half_angle_deg,
        classification=classification.classification.value,
        charge_cell_count=charge_cells,
        fields_verified=tuple(sorted(fields)),
        mesh_owner_path=owner_path,
    )


def read_control_dict_entries(case_dir: str, keys: Iterable[str]) -> Dict[str, str]:
    path = Path(case_dir) / "system" / "controlDict"
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Strip comments loosely for matching.
    stripped = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    stripped = re.sub(r"//.*?$", "", stripped, flags=re.MULTILINE)
    for key in keys:
        m = re.search(rf"\b{re.escape(key)}\s+([^;]+);", stripped)
        if m:
            out[key] = m.group(1).strip()
    return out


def write_control_dict_entries(
    case_dir: str,
    updates: Dict[str, object],
) -> ControlDictWriteResult:
    """Rewrite proven controlDict keys only; validate by read-back."""
    unsupported = [k for k in updates if k not in SUPPORTED_CONTROL_WRITERS]
    if unsupported:
        return ControlDictWriteResult(
            ok=False,
            reason=f"Unsupported controlDict keys: {unsupported}",
        )
    path = Path(case_dir) / "system" / "controlDict"
    if not path.is_file():
        return ControlDictWriteResult(ok=False, reason="controlDict missing")

    original = path.read_text(encoding="utf-8", errors="ignore")
    lines = original.splitlines(keepends=True)
    changed: List[str] = []
    remaining = dict(updates)

    def _fmt(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.12g}"
        if isinstance(value, int):
            return str(value)
        return str(value)

    new_lines: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        matched = False
        for key in list(remaining.keys()):
            if stripped.startswith(key) and (
                len(stripped) == len(key)
                or stripped[len(key) : len(key) + 1].isspace()
            ):
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}{key} {_fmt(remaining.pop(key))};\n")
                changed.append(key)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    if remaining:
        return ControlDictWriteResult(
            ok=False,
            reason=f"Keys not found in controlDict: {sorted(remaining)}",
        )

    path.write_text("".join(new_lines), encoding="utf-8")
    readback = read_control_dict_entries(case_dir, changed)
    for key in changed:
        expected = _fmt(updates[key])
        got = readback.get(key, "")
        # Numeric compare when both parse as floats.
        try:
            if abs(float(expected) - float(got)) > 1e-15 * max(1.0, abs(float(expected))):
                return ControlDictWriteResult(
                    ok=False,
                    changed=tuple(changed),
                    readback=readback,
                    reason=f"Read-back mismatch for {key}: wrote {expected!r} got {got!r}",
                )
        except ValueError:
            if got != expected:
                return ControlDictWriteResult(
                    ok=False,
                    changed=tuple(changed),
                    readback=readback,
                    reason=f"Read-back mismatch for {key}: wrote {expected!r} got {got!r}",
                )
    return ControlDictWriteResult(
        ok=True, changed=tuple(changed), readback=readback
    )


def gui_values_to_control_updates(gui_values: Dict[str, object]) -> Dict[str, object]:
    """Map GUI keys to controlDict entries for supported writers only."""
    updates: Dict[str, object] = {}
    if "end_time_s" in gui_values and gui_values["end_time_s"] is not None:
        updates["endTime"] = gui_values["end_time_s"]
    if "delta_t" in gui_values and gui_values["delta_t"] is not None:
        updates["deltaT"] = gui_values["delta_t"]
    if "max_co" in gui_values and gui_values["max_co"] is not None:
        updates["maxCo"] = gui_values["max_co"]
    if "write_control_type" in gui_values and gui_values["write_control_type"] is not None:
        updates["writeControl"] = gui_values["write_control_type"]
    wc = gui_values.get("write_control_type")
    if wc == "timeStep" and gui_values.get("write_interval_steps") is not None:
        updates["writeInterval"] = gui_values["write_interval_steps"]
    elif gui_values.get("write_interval_time") is not None:
        updates["writeInterval"] = gui_values["write_interval_time"]
    return updates


def import_mode_label(mode: ImportMode2D) -> str:
    return {
        ImportMode2D.NATIVE_GGUI_2D: "Native GGUI 2D",
        ImportMode2D.IMPORTED_2D_UNINITIALIZED: "Imported case — ready to initialise",
        ImportMode2D.IMPORTED_2D_INITIALIZING: "Initialising imported case",
        ImportMode2D.IMPORTED_2D_READY: "Imported case — initialized",
        ImportMode2D.IMPORTED_2D_RUNNING: "Running imported case",
        ImportMode2D.IMPORTED_2D_FAILED: "Imported case — failed",
    }[mode]


def lifecycle_state_label(lifecycle: ImportMode2D) -> str:
    """Compatibility wrapper."""
    return import_mode_label(lifecycle)
