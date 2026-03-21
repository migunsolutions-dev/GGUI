"""
Path and case-directory utilities for the BlastFoam GUI.
- Convert Windows paths to WSL Linux paths for rotateFields and Allrun.
- Resolve "latest" time directory in OpenFOAM case dirs without fragile shell sorting.
"""
from __future__ import annotations

import os
import re
from typing import Optional

# Names that are never OpenFOAM time directories (exclude from latest-time scan)
_NON_TIME_DIRS = frozenset({
    "constant", "system", "postProcessing", "0.orig",
})


def win_to_wsl_path(win_path: str) -> str:
    """
    Convert a Windows or UNC path to the path seen inside WSL.
    - C:\\Users\\... -> /mnt/c/Users/...
    - \\wsl.localhost\\Ubuntu\\home\\... -> /home/...
    - Already Linux (starts with /) -> returned as-is.
    """
    p = (win_path or "").strip().replace("\\", "/")
    if not p:
        return ""
    if p.startswith("/") and not p.startswith("//"):
        return p
    # UNC: \\wsl.localhost\Ubuntu\home\user\case -> /home/user/case
    if p.startswith("//"):
        parts = [x for x in p.split("/") if x]
        if len(parts) >= 3 and parts[0].lower() in ("wsl.localhost", "wsl$"):
            return "/" + "/".join(parts[2:])
    # Windows: C:/Users/... -> /mnt/c/Users/...
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    return p


def get_latest_time_dir(case_dir: str) -> Optional[str]:
    """
    Find the latest time directory in an OpenFOAM case directory by scanning
    subdirs, excluding known non-time names, and parsing the rest as floats.
    Returns the time directory name (e.g. "0.001" or "1e-4") with the maximum
    numeric value, or None if no valid time dirs exist.
    Do not rely on shell 'ls | sort' — alphanumeric sort gives wrong order
    (e.g. "10" before "2") and can include postProcessing etc.
    """
    if not case_dir or not os.path.isdir(case_dir):
        return None
    try:
        candidates: list[tuple[float, str]] = []
        for name in os.listdir(case_dir):
            if name in _NON_TIME_DIRS:
                continue
            if name.startswith("processor"):
                continue
            path = os.path.join(case_dir, name)
            if not os.path.isdir(path):
                continue
            try:
                t = float(name)
                candidates.append((t, name))
            except ValueError:
                continue
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    except OSError:
        return None
