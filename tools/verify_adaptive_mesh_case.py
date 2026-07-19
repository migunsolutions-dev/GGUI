#!/usr/bin/env python3
"""Verify an adaptive-mesh 3D blast case in two modes (Phase 6).

Mode 1 (default): dictionary-only validation.
    Walks a generated case and checks that the OpenFOAM/blastFoam dictionaries
    referenced by the GGUI 3D adaptive-mesh workflow exist and are internally
    consistent: blockMeshDict, setFieldsDict / setRefinedFields, dynamicMeshDict
    (when AMR is enabled), 0.orig fields, Allrun staging, case_init_mode.json
    metadata, and dumpLevel.

Mode 2 (--run-wsl): runtime validation via WSL OpenFOAM 9 / blastFoam.
    Sources /opt/openfoam9/etc/bashrc, confirms blastFoam is available, runs
    ``./Allrun`` for a short window (caller is expected to have set a small
    ``endTime`` in system/controlDict for fast verification), and checks that
    time directories were created, that alpha.c4 mass is nonzero, that
    cellLevel is dumped when ``dumpLevel true`` is configured, and reports
    refine/unrefine events from log.blastFoam.

The script never modifies GUI defaults or test cases. It is read-only on the
target case unless ``--run-wsl`` is specified (which executes Allrun).

Exit codes:
    0   all required checks passed (in the chosen mode)
    1   one or more required checks failed
    2   bad arguments / missing case

Usage:
    python tools/verify_adaptive_mesh_case.py --case path/to/case
    python tools/verify_adaptive_mesh_case.py --case path/to/case --run-wsl \
        --wsl-timeout 1800
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from path_utils import win_to_wsl_path
except ImportError:  # pragma: no cover - path_utils is always present in this repo
    def win_to_wsl_path(p: str) -> str:
        return p


# --------------------------------------------------------------------------- #
# Check infrastructure
# --------------------------------------------------------------------------- #

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    severity: str = "required"  # "required" | "optional"


@dataclass
class Report:
    case_dir: Path
    mode: str
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", *, optional: bool = False) -> None:
        self.checks.append(CheckResult(name, ok, detail, "optional" if optional else "required"))

    def required_failed(self) -> List[CheckResult]:
        return [c for c in self.checks if c.severity == "required" and not c.ok]

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_dir": str(self.case_dir),
            "mode": self.mode,
            "checks": [c.__dict__ for c in self.checks],
            "required_failures": len(self.required_failed()),
        }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# Mode 1 – dictionary-only checks
# --------------------------------------------------------------------------- #

def _has_amr_enabled(case_dir: Path) -> bool:
    dm = case_dir / "constant" / "dynamicMeshDict"
    if not dm.is_file():
        return False
    return "adaptiveFvMesh" in _read_text(dm)


def _set_refined_fields_referenced(case_dir: Path) -> bool:
    setf = case_dir / "system" / "setFieldsDict"
    if not setf.is_file():
        return False
    txt = _read_text(setf)
    return "refineInternal" in txt or "sphericalMassToCell" in txt or "cylindericalMassToCell" in txt or "boxToCell" in txt


def check_dictionaries(case_dir: Path) -> Report:
    rep = Report(case_dir=case_dir, mode="dictionary")

    # --- Mesh & geometry ---
    bm = case_dir / "system" / "blockMeshDict"
    rep.add("blockMeshDict exists", bm.is_file(), str(bm))
    snappy = case_dir / "system" / "snappyHexMeshDict"
    rep.add("snappyHexMeshDict exists", snappy.is_file(), str(snappy), optional=True)

    # --- AMR (dynamicMeshDict) ---
    dm = case_dir / "constant" / "dynamicMeshDict"
    amr_enabled = _has_amr_enabled(case_dir)
    rep.add("dynamicMeshDict exists", dm.is_file(), str(dm))
    if dm.is_file():
        text = _read_text(dm)
        rep.add(
            "AMR errorEstimator present",
            "errorEstimator" in text,
            "Expected `errorEstimator <name>;` in dynamicMeshDict",
        )
        # scaledDelta must use new blastFoam keyword, not legacy deltaCoeffs block.
        if "scaledDelta" in text and "deltaCoeffs" in text and "scaledDeltaField" not in text:
            rep.add(
                "scaledDelta uses scaledDeltaField (not deprecated deltaCoeffs)",
                False,
                "Found `deltaCoeffs` block without `scaledDeltaField`; blastFoam expects `scaledDeltaField <name>;`.",
            )
        else:
            rep.add(
                "scaledDelta uses scaledDeltaField (not deprecated deltaCoeffs)",
                True,
                "OK (either not scaledDelta, or scaledDeltaField present)",
            )
        rep.add(
            "dumpLevel directive present",
            re.search(r"\bdumpLevel\s+(true|false)\s*;", text) is not None,
            "Required for ParaView cellLevel verification",
            optional=True,
        )
        rep.add(
            "maxRefinement present",
            "maxRefinement" in text,
            "AMR must declare maxRefinement",
        )
    elif amr_enabled:
        rep.add("AMR dictionary readable", False, "dynamicMeshDict reported AMR but file missing")

    # --- Charge / fields seeding ---
    set_dict = case_dir / "system" / "setFieldsDict"
    set_refined = _set_refined_fields_referenced(case_dir)
    rep.add(
        "setFieldsDict / setRefinedFields inputs exist",
        set_dict.is_file(),
        str(set_dict),
    )
    rep.add(
        "Charge seeding region declared",
        set_refined,
        "Expected refineInternal/sphericalMassToCell/cylindericalMassToCell/boxToCell",
    )
    rep.add(
        "0.orig/alpha.c4 present (seeded by Allrun)",
        (case_dir / "0.orig" / "alpha.c4").is_file(),
        "alpha.c4 must exist in 0.orig before setRefinedFields",
    )
    rep.add(
        "0.orig/p present",
        (case_dir / "0.orig" / "p").is_file(),
        "Required by blastFoam.",
    )
    rep.add(
        "0.orig/U present",
        (case_dir / "0.orig" / "U").is_file(),
        "Required by blastFoam.",
    )
    rep.add(
        "0.orig/T present",
        (case_dir / "0.orig" / "T").is_file(),
        "Required by blastFoam.",
    )

    # --- Allrun staging ---
    allrun = case_dir / "Allrun"
    rep.add("Allrun present", allrun.is_file(), str(allrun))
    if allrun.is_file():
        atxt = _read_text(allrun)
        rep.add(
            "Allrun copies 0.orig -> 0",
            ("cp -r 0.orig 0" in atxt) or ("0.orig" in atxt and "rm -rf 0" in atxt),
            "Allrun should ensure 0/ exists before field initialization",
        )
        rep.add(
            "Allrun calls blockMesh",
            "blockMesh" in atxt,
        )
        rep.add(
            "Allrun calls setRefinedFields when inside-refinement is requested",
            (not set_refined) or ("setRefinedFields" in atxt) or ("setFields" in atxt),
            "If setFieldsDict has refineInternal/regions, Allrun must invoke setRefinedFields/setFields",
        )

    # --- Metadata ---
    meta = case_dir / "case_init_mode.json"
    rep.add("case_init_mode.json present", meta.is_file(), str(meta))
    if meta.is_file():
        try:
            blob = json.loads(_read_text(meta))
        except json.JSONDecodeError as exc:
            rep.add("case_init_mode.json parseable", False, f"JSON error: {exc}")
            blob = {}
        else:
            rep.add("case_init_mode.json parseable", True)
        # Soft cross-checks
        if amr_enabled:
            amr_meta = blob.get("amr_written") or blob.get("amr") or {}
            rep.add(
                "case_init_mode.json: amr metadata present",
                bool(amr_meta),
                "AMR is enabled but metadata block is missing",
                optional=True,
            )
            est = amr_meta.get("errorEstimator_line") or amr_meta.get("errorEstimator")
            rep.add(
                "case_init_mode.json: AMR indicator matches dynamicMeshDict",
                est is None or any(k in (est or "") for k in ("densityGradient", "scaledDelta")),
                f"errorEstimator(_line)={est!r}",
                optional=True,
            )
        dom = blob.get("domain_alignment") or {}
        if dom:
            rep.add(
                "case_init_mode.json: actual_cell_size_m present",
                dom.get("actual_cell_size_m") is not None,
                optional=True,
            )
        cap = blob.get("charge_capture") or {}
        if cap:
            rep.add(
                "case_init_mode.json: charge_capture metadata present",
                "mode" in cap,
                optional=True,
            )

    # --- Validation helpers shipped per case ---
    rep.add(
        "check_charge_region.py present",
        (case_dir / "check_charge_region.py").is_file(),
        "Per-case post-init validator (alpha.c4 in charge region)",
        optional=True,
    )
    rep.add(
        "case_charge_region.json present",
        (case_dir / "case_charge_region.json").is_file(),
        "Charge region geometry used by check_charge_region.py",
        optional=True,
    )

    return rep


# --------------------------------------------------------------------------- #
# Mode 2 – WSL runtime validation
# --------------------------------------------------------------------------- #

def _wsl_bash(inner: str, timeout: int) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            ["wsl", "bash", "-lc", inner],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, "wsl not found in PATH"
    except subprocess.TimeoutExpired:
        return 124, "wsl bash timed out"


def _alpha_c4_has_nonzero_mass(case_dir: Path) -> Tuple[bool, str]:
    """Best-effort check that alpha.c4 in time directory 0 is not uniform zero.

    Walks ``case_dir/0`` first, falls back to ``processor0/0`` for parallel runs.
    """
    candidates: List[Path] = []
    for sub in ("0", "0/uniform", "processor0/0"):
        p = case_dir / sub / "alpha.c4"
        if p.is_file():
            candidates.append(p)
    if not candidates:
        return False, "0/alpha.c4 missing (Allrun not run yet or wrong case?)"
    for p in candidates:
        text = _read_text(p)
        if not text:
            continue
        m = re.search(r"internalField\s+uniform\s+([-+.eE0-9]+)\s*;", text)
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if v > 1e-12:
                return True, f"{p}: uniform {v}"
            continue
        # nonuniform list – sample tokens and check max > 0
        if "nonuniform" in text:
            chunk = text[text.find("nonuniform"):]
            vals = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", chunk)
            for v in vals[:5000]:
                try:
                    if float(v) > 0.5:
                        return True, f"{p}: nonuniform field has alpha>0.5 sample"
                except ValueError:
                    continue
            return False, f"{p}: nonuniform field but no alpha>0.5 token seen in first 5000"
    return False, "alpha.c4 present but no nonzero values detected"


def check_wsl_runtime(case_dir: Path, *, timeout: int, skip_solver: bool) -> Report:
    rep = Report(case_dir=case_dir, mode="wsl-runtime")

    code, out = _wsl_bash("source /opt/openfoam9/etc/bashrc 2>/dev/null; which blastFoam || true", 30)
    rep.add("WSL available", code != 127, out.strip()[:200])
    if code == 127:
        return rep
    blast_path = out.strip().splitlines()[-1] if out.strip() else ""
    rep.add(
        "blastFoam in PATH after sourcing OpenFOAM 9",
        bool(blast_path) and "/blastFoam" in blast_path,
        blast_path or "blastFoam not found",
    )
    code2, out2 = _wsl_bash(
        "source /opt/openfoam9/etc/bashrc 2>/dev/null; blastFoam -help 2>&1 | head -n 5 || true",
        30,
    )
    rep.add(
        "blastFoam -help reachable",
        code2 == 0,
        out2.strip().splitlines()[0][:200] if out2.strip() else "no output",
    )

    if skip_solver:
        return rep

    # Run Allrun in the case directory. Caller is expected to have set a small
    # endTime in system/controlDict beforehand if they want a short verification.
    wsl_case = win_to_wsl_path(str(case_dir.resolve()))
    inner = (
        "source /opt/openfoam9/etc/bashrc 2>/dev/null; "
        f'cd "{wsl_case}" && '
        "chmod +x Allrun 2>/dev/null; "
        "./Allrun"
    )
    code, out = _wsl_bash(inner, timeout)
    rep.add(
        "Allrun completed without timeout",
        code != 124,
        f"exit={code} (124=timeout)",
    )
    rep.add(
        "Allrun exited 0",
        code == 0,
        f"exit={code}, tail: {out[-500:].strip() if out else ''}",
    )

    # Time directories created?
    time_dirs: List[str] = []
    try:
        for name in os.listdir(case_dir):
            try:
                float(name)
            except ValueError:
                continue
            if name in ("0",):
                continue
            time_dirs.append(name)
    except OSError:
        pass
    proc0 = case_dir / "processor0"
    if proc0.is_dir():
        try:
            for name in os.listdir(proc0):
                try:
                    float(name)
                except ValueError:
                    continue
                if name in ("0",):
                    continue
                time_dirs.append(f"processor0/{name}")
        except OSError:
            pass
    rep.add(
        "Solver produced at least one new time directory",
        len(time_dirs) > 0,
        f"time dirs: {sorted(time_dirs)[:10]}",
    )

    # cellLevel when dumpLevel true
    dm = case_dir / "constant" / "dynamicMeshDict"
    dump_level_requested = False
    if dm.is_file():
        m = re.search(r"\bdumpLevel\s+(true|false)\s*;", _read_text(dm))
        dump_level_requested = bool(m and m.group(1) == "true")
    if dump_level_requested:
        any_celllevel = False
        sample_path = ""
        roots: List[Path] = [case_dir]
        if proc0.is_dir():
            roots.append(proc0)
        for root in roots:
            try:
                for name in os.listdir(root):
                    try:
                        float(name)
                    except ValueError:
                        continue
                    if name == "0":
                        continue
                    cl = root / name / "cellLevel"
                    if cl.is_file():
                        any_celllevel = True
                        sample_path = str(cl)
                        break
            except OSError:
                continue
            if any_celllevel:
                break
        rep.add(
            "cellLevel field present (dumpLevel true requested)",
            any_celllevel,
            sample_path or "no cellLevel found under any time directory",
        )

    # alpha.c4 mass nonzero
    ok_alpha, detail = _alpha_c4_has_nonzero_mass(case_dir)
    rep.add("alpha.c4 mass nonzero in 0/ after setRefinedFields", ok_alpha, detail)

    # AMR refine/unrefine events
    bf_log = _read_text(case_dir / "log.blastFoam")
    refine_n = len(re.findall(r"Refined\s+from\s+\d+\s+to\s+\d+\s+cells", bf_log))
    unrefine_n = len(re.findall(r"Unrefined\s+from\s+\d+\s+to\s+\d+\s+cells", bf_log))
    if dump_level_requested or _has_amr_enabled(case_dir):
        rep.add(
            "blastFoam log shows refine events",
            refine_n > 0 or unrefine_n > 0,
            f"refine={refine_n} unrefine={unrefine_n}",
            optional=True,
        )

    return rep


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Verify an adaptive-mesh blast case (Phase 6).")
    ap.add_argument("--case", type=Path, required=True, help="Path to the case directory")
    ap.add_argument(
        "--run-wsl",
        action="store_true",
        help="Also run WSL/OpenFOAM/blastFoam runtime checks (executes ./Allrun).",
    )
    ap.add_argument(
        "--wsl-timeout",
        type=int,
        default=1800,
        help="Per-run timeout for WSL Allrun (seconds, default 1800).",
    )
    ap.add_argument(
        "--skip-solver",
        action="store_true",
        help="In WSL mode, only check that blastFoam is reachable; do not run Allrun.",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the structured report as JSON.",
    )
    args = ap.parse_args()

    case_dir = args.case.resolve()
    if not case_dir.is_dir():
        print(f"ERROR: case directory not found: {case_dir}", file=sys.stderr)
        return 2

    dict_report = check_dictionaries(case_dir)
    reports = [dict_report]
    if args.run_wsl:
        reports.append(check_wsl_runtime(case_dir, timeout=args.wsl_timeout, skip_solver=args.skip_solver))

    overall_failed = False
    for rep in reports:
        print(f"\n=== {rep.mode} checks ===")
        for c in rep.checks:
            tag = "OK  " if c.ok else ("FAIL" if c.severity == "required" else "WARN")
            tail = f" -- {c.detail}" if c.detail else ""
            print(f"  [{tag}] {c.name}{tail}")
        required_failed = rep.required_failed()
        if required_failed:
            overall_failed = True
            print(f"  -> {len(required_failed)} required failure(s)")

    if args.json:
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8"
            )
            print(f"\nWrote JSON report to {args.json}")
        except OSError as exc:
            print(f"ERROR writing JSON report: {exc}", file=sys.stderr)

    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
