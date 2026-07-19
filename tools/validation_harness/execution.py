"""Optional WSL execution helpers (init-only or full Allrun). Does not patch cases."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from path_utils import win_to_wsl_path  # noqa: E402

INIT_ONLY_SCRIPT = r"""#!/usr/bin/env bash
set +e
export WM_PROJECT_DIR="${WM_PROJECT_DIR:-/opt/openfoam9}"
set +o pipefail
source "${WM_PROJECT_DIR}/etc/bashrc" 2>/dev/null || true
cd "$CASE_DIR" || exit 1
rm -rf processor* postProcessing 0 log.* 2>/dev/null
[ -d 0.orig ] && cp -r 0.orig 0
[ -f system/surfaceFeaturesDict ] && surfaceFeatures > log.surfaceFeatures 2>&1
blockMesh > log.blockMesh 2>&1
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
addEmptyPatch internalPatch internal -overwrite > log.addEmptyPatch 2>&1
rm -rf 0; [ -d 0.orig ] && cp -r 0.orig 0
changeDictionary > log.changeDictionary 2>&1
if grep -qE "^setRefinedFields" Allrun 2>/dev/null; then
  setRefinedFields > log.setRefinedFields 2>&1
else
  setFields > log.setFields 2>&1
fi
postProcess -func writeCellVolumes -time 0 > log.cellVolumes 2>&1
echo "init done"
"""


def run_wsl_bash(script_body: str, case_dir: Path, timeout: int) -> Tuple[int, str]:
    wsl_case = win_to_wsl_path(str(case_dir.resolve()))
    inner = script_body.replace("$CASE_DIR", wsl_case)
    try:
        proc = subprocess.run(
            ["wsl", "bash", "-lc", inner],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, out


def run_init_only(case_dir: Path, timeout: int = 3600) -> Tuple[int, str]:
    """Mesh + setFields/setRefinedFields only (no blastFoam)."""
    return run_wsl_bash(INIT_ONLY_SCRIPT, case_dir, timeout)


def run_allrun(case_dir: Path, timeout: int = 7200) -> Tuple[int, str]:
    """Run case Allrun in WSL."""
    wsl_case = win_to_wsl_path(str(case_dir.resolve()))
    inner = (
        f'source /opt/openfoam9/etc/bashrc 2>/dev/null; '
        f'cd "{wsl_case}" && chmod +x Allrun 2>/dev/null && ./Allrun'
    )
    try:
        proc = subprocess.run(
            ["wsl", "bash", "-lc", inner],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, out
