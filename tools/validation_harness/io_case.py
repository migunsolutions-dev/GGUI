"""Read case directories and OpenFOAM fields (read-only)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

ScalarList = Union[List[float], Tuple[str, float]]


def load_case_init_mode(case_dir: Path) -> Dict[str, Any]:
    p = case_dir / "case_init_mode.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_load_error": str(exc)}


def read_scalar_internal_field(path: Path) -> Optional[ScalarList]:
    """Parse ASCII OpenFOAM volScalarField internalField (uniform or nonuniform list)."""
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "format      binary" in raw[:3000]:
        return "binary"
    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\(",
        raw,
    )
    if m:
        start = raw.find("(", m.end() - 1)
        end = raw.find(")", start)
        if start < 0 or end < 0:
            return None
        vals = [float(t) for t in raw[start + 1 : end].split() if t.strip()]
        return vals
    m2 = re.search(r"internalField\s+uniform\s+([0-9.eE+-]+)", raw)
    if m2:
        return ("uniform", float(m2.group(1)))
    return None


def captured_mass_kg(
    alpha_path: Path,
    volume_path: Path,
    rho: float,
) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Return (captured_mass_kg, total_cells, charge_cells with alpha>=0.5).
    Mass = sum(alpha * V) * rho for mass-conserving fields.
    """
    av = read_scalar_internal_field(alpha_path)
    vv = read_scalar_internal_field(volume_path)
    if av == "binary" or vv == "binary":
        return None, None, None
    if isinstance(av, tuple) and av[0] == "uniform":
        n = len(vv) if isinstance(vv, list) else None
        return 0.0, n, 0
    if not isinstance(av, list):
        return None, None, None
    n = len(av)
    charge_cells = sum(1 for a in av if a >= 0.5)
    if isinstance(vv, list) and len(vv) == n:
        mass = sum(a * v for a, v in zip(av, vv)) * float(rho)
        return mass, n, charge_cells
    return None, n, charge_cells


def find_blastfoam_log(case_dir: Path) -> Optional[Path]:
    for name in ("log.blastFoam", "log.blastfoam"):
        p = case_dir / name
        if p.is_file():
            return p
    return None


def get_nested(d: Dict[str, Any], dot_path: str) -> Any:
    cur: Any = d
    for part in dot_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur
