"""Numeric OpenFOAM time-directory helpers for the Cylindrical–2D viewer."""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Iterable, List, Optional, Sequence, Tuple

_SKIP_DIR_NAMES = frozenset({"constant", "system", "postProcessing"})
LIVE_FOLLOW_LABEL = "Live"
TIME_ZERO_LABEL = "0"
_VIEW_LINK_NAMES = ("constant", "system")


def list_numeric_time_entries(case_dir: str) -> List[Tuple[float, str]]:
    """Return ``(numeric_value, directory_label)`` sorted numerically ascending.

    Skips ``constant``, ``system``, ``processor*``, ``postProcessing``, and any
    non-numeric directory names. Labels keep the on-disk spelling.
    """
    times: List[Tuple[float, str]] = []
    try:
        for name in os.listdir(case_dir):
            path = os.path.join(case_dir, name)
            if not os.path.isdir(path):
                continue
            if name in _SKIP_DIR_NAMES or name.startswith("processor"):
                continue
            try:
                tval = float(name)
            except ValueError:
                continue
            times.append((tval, name))
    except OSError:
        return []
    times.sort(key=lambda item: item[0])
    return times


def list_numeric_time_labels(case_dir: str) -> List[str]:
    return [label for _, label in list_numeric_time_entries(case_dir)]


def pick_opening_time(entries: Sequence[Tuple[float, str]]) -> Tuple[str, float]:
    """Default viewer selection on case open: always prefer time ``0``."""
    for tval, label in entries:
        if tval == 0.0 or label == TIME_ZERO_LABEL:
            return label, float(tval)
    return TIME_ZERO_LABEL, 0.0


def opening_time_entry() -> Tuple[float, str]:
    """Hard-coded initial selection. Does not inspect the case directory."""
    return 0.0, TIME_ZERO_LABEL


def time_zero_dir(case_dir: str) -> str:
    return os.path.join(case_dir, TIME_ZERO_LABEL)


def poly_mesh_dir_for_time_zero(case_dir: str) -> Optional[str]:
    """Locate the mesh for time 0 without listing other time directories."""
    owner_zero = os.path.join(case_dir, TIME_ZERO_LABEL, "polyMesh", "owner")
    if os.path.isfile(owner_zero):
        return os.path.join(case_dir, TIME_ZERO_LABEL, "polyMesh")
    const_owner = os.path.join(case_dir, "constant", "polyMesh", "owner")
    if os.path.isfile(const_owner):
        return os.path.join(case_dir, "constant", "polyMesh")
    return None


def _link_directory(source: str, dest: str) -> None:
    """Point ``dest`` at ``source`` without copying. Original files are untouched."""
    source = os.path.abspath(source)
    dest = os.path.abspath(dest)
    if os.path.lexists(dest):
        return
    try:
        os.symlink(source, dest, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            raise
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", dest, source],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0 or not os.path.isdir(dest):
        detail = (completed.stderr or completed.stdout or "").strip()
        raise OSError(detail or f"failed to link {source!r} -> {dest!r}")


def make_single_time_case_view(case_dir: str, time_label: str) -> str:
    """Temp case root that exposes only ``constant``, ``system``, and one time dir.

    VTK/OpenFOAM readers otherwise enumerate every saved time directory. Linking
    a single time keeps initial load independent of how many results exist.
    The original case is not copied or modified.
    """
    root = tempfile.mkdtemp(prefix="ggui_of_tview_")
    try:
        for name in _VIEW_LINK_NAMES:
            source = os.path.join(case_dir, name)
            if os.path.isdir(source):
                _link_directory(source, os.path.join(root, name))
        label = str(time_label or TIME_ZERO_LABEL)
        source_time = os.path.join(case_dir, label)
        if os.path.isdir(source_time):
            _link_directory(source_time, os.path.join(root, label))
        with open(os.path.join(root, "case.foam"), "w", encoding="utf-8") as handle:
            handle.write("")
    except Exception:
        remove_single_time_case_view(root)
        raise
    return root


def remove_single_time_case_view(view_root: Optional[str]) -> None:
    """Drop junctions/symlinks only. Never recurse into the original case."""
    if not view_root or not os.path.isdir(view_root):
        return
    try:
        for name in os.listdir(view_root):
            path = os.path.join(view_root, name)
            try:
                if os.path.isdir(path):
                    os.rmdir(path)
                else:
                    os.unlink(path)
            except OSError:
                pass
        os.rmdir(view_root)
    except OSError:
        pass


def match_reader_time_value(
    time_values: Iterable[float],
    target: float,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-15,
) -> Optional[float]:
    """Map a selected numeric time onto a PyVista/OpenFOAM reader time value."""
    values = [float(v) for v in time_values]
    if not values:
        return None
    best = min(values, key=lambda v: abs(v - float(target)))
    tol = max(abs_tol, abs(float(target)) * rel_tol, abs(best) * rel_tol)
    if abs(best - float(target)) <= tol:
        return best
    return None


def poly_mesh_dir_at_or_before(case_dir: str, time_value: float) -> Optional[str]:
    """Latest ``polyMesh`` at or before ``time_value``, else ``constant/polyMesh``."""
    best_time: Optional[float] = None
    best_path: Optional[str] = None
    try:
        for tval, name in list_numeric_time_entries(case_dir):
            if tval > float(time_value) + abs_tol_for(time_value):
                continue
            owner = os.path.join(case_dir, name, "polyMesh", "owner")
            if os.path.isfile(owner) and (best_time is None or tval >= best_time):
                best_time = tval
                best_path = os.path.join(case_dir, name, "polyMesh")
    except OSError:
        best_path = None
    if best_path is not None:
        return best_path
    const_owner = os.path.join(case_dir, "constant", "polyMesh", "owner")
    if os.path.isfile(const_owner):
        return os.path.join(case_dir, "constant", "polyMesh")
    return None


def abs_tol_for(time_value: float) -> float:
    return max(1e-15, abs(float(time_value)) * 1e-12)
