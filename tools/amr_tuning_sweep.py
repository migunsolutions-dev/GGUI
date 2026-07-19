#!/usr/bin/env python3
"""
Developer-only AMR tuning sweep for 3D blastFoam cases.

Does not change GUI defaults. Generates or clones a baseline case, creates
variants with patched constant/dynamicMeshDict, optionally runs Allrun in WSL,
and writes CSV + Markdown reports with best-effort metrics.

Usage:
  python tools/amr_tuning_sweep.py --out _amr_tuning_runs --dry-run
  python tools/amr_tuning_sweep.py --out _amr_tuning_runs --run-wsl --wsl-timeout 3600
  python tools/amr_tuning_sweep.py --base-case path/to/case --out _amr_tuning_runs --dry-run

See also: run_all.sh written under --out for manual WSL execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generator_3d import Generator3D  # noqa: E402
from models import CaseInputs3D  # noqa: E402
from path_utils import win_to_wsl_path  # noqa: E402


# ---------------------------------------------------------------------------
# Variant matrix (Task 2): small controlled sweep — densityGradient + scaledDelta_p
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AMRVariantSpec:
    variant_id: str
    estimator: str  # "densityGradient" | "scaledDelta_p"
    n_buffer_layers: int
    lower_refine_level: float
    unrefine_level: float


DEFAULT_VARIANTS: Tuple[AMRVariantSpec, ...] = (
    AMRVariantSpec("dg_nb2_lr010_ur010", "densityGradient", 2, 0.10, 0.10),
    AMRVariantSpec("dg_nb1_lr010_ur010", "densityGradient", 1, 0.10, 0.10),
    AMRVariantSpec("dg_nb1_lr020_ur010", "densityGradient", 1, 0.20, 0.10),
    AMRVariantSpec("dg_nb0_lr020_ur010", "densityGradient", 0, 0.20, 0.10),
    AMRVariantSpec("sdp_nb2_lr010_ur010", "scaledDelta_p", 2, 0.10, 0.10),
    AMRVariantSpec("sdp_nb1_lr010_ur010", "scaledDelta_p", 1, 0.10, 0.10),
    AMRVariantSpec("sdp_nb1_lr020_ur010", "scaledDelta_p", 1, 0.20, 0.10),
    AMRVariantSpec("sdp_nb0_lr020_ur010", "scaledDelta_p", 0, 0.20, 0.10),
)

# Held constant across first sweep (Task 2)
SWEEP_REFINE_INTERVAL = 3
SWEEP_MAX_REFINEMENT = 3


def baseline_case_inputs() -> CaseInputs3D:
    """3D spherical free-field baseline: short runtime, explicit outside_extent, auto capture."""
    return CaseInputs3D(
        min_point=(-2.0, -2.0, -2.0),
        max_point=(2.0, 2.0, 2.0),
        cell_size=0.2,
        charge_center=(0.0, 0.0, 0.0),
        charge_shape="Sphere",
        mass_kg=25.0,
        cylinder_radius=0.1,
        cylinder_axis="Z",
        material_name="C4",
        rho_charge=1601.0,
        energy_j_per_kg=4.5e6,
        p_atm=101325.0,
        t_atm=300.0,
        end_time_s=0.001,
        delta_t=1e-7,
        write_interval_steps=50,
        cores=2,
        cfl_value=0.5,
        obstacles=[],
        enable_dyn_refine=True,
        enable_local_refinement=True,
        provenance={"enable_dyn_refine": "USER"},
        charge_refinement_level=3,
        dyn_refine_max=SWEEP_MAX_REFINEMENT,
        refine_interval=SWEEP_REFINE_INTERVAL,
        lower_refine_threshold=0.1,
        unrefine_threshold=0.1,
        n_buffer_layers_dynamic=2,
        refine_indicator_field="densityGradient",
        charge_capture_mode="auto",
        charge_capture_factor=1.0,
        outside_extent=0.35,
        charge_outer_refine_min=2,
        charge_outer_refine_max=2,
        buffer_layers=3,
        write_control_type="adjustableRunTime",
        write_interval_time=2e-5,
        fast_run_mode=True,
        enable_post_processing=False,
        decomposition_method="simple",
        decomposition_simple_n=(1, 2, 1),
        decomposition_simple_delta=0.001,
    )


def patch_dynamic_mesh_dict(
    path: Path,
    *,
    estimator: str,
    n_buffer_layers: int,
    lower_refine_level: float,
    unrefine_level: float,
    refine_interval: int = SWEEP_REFINE_INTERVAL,
    max_refinement: int = SWEEP_MAX_REFINEMENT,
    dump_level: bool = True,
    begin_unrefine: Optional[float] = None,
    max_cells: Optional[int] = None,
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    key = estimator.strip().lower().replace(" ", "")
    if key in ("densitygradient", "dg"):
        err_block = "errorEstimator  densityGradient;\n"
    elif key in ("scaleddeltap", "scaleddelta_p", "sdp"):
        err_block = "errorEstimator  scaledDelta;\nscaledDeltaField p;\n\n"
    else:
        raise ValueError(f"Unknown estimator {estimator!r}")

    i = text.find("dynamicFvMesh")
    if i < 0:
        raise ValueError(f"dynamicFvMesh not found in {path}")
    header = text[:i]
    opt = ""
    if begin_unrefine is not None:
        opt += f"beginUnrefine   {begin_unrefine:g};\n"
    if max_cells is not None:
        opt += f"maxCells       {max_cells};\n"
    dl = "true" if dump_level else "false"
    lr = _fmt_float(lower_refine_level)
    ur = _fmt_float(unrefine_level)
    body = (
        f"dynamicFvMesh   adaptiveFvMesh;\n"
        f"{err_block}"
        f"refineInterval  {int(refine_interval)};\n"
        f"lowerRefineLevel {lr};\n"
        f"unrefineLevel   {ur};\n"
        f"nBufferLayers   {int(n_buffer_layers)};\n"
        f"maxRefinement   {int(max_refinement)};\n"
        f"dumpLevel      {dl};\n"
        f"{opt}"
    )
    path.write_text(header + body, encoding="utf-8", newline="\n")


def _fmt_float(x: float) -> str:
    s = f"{x:.10g}"
    if s.endswith(".0"):
        return s[:-2]
    return s


def extract_dynamic_mesh_review_fields(dm_path: Path) -> Dict[str, str]:
    """Key/value strings for manual review (Task 7)."""
    if not dm_path.is_file():
        return {}
    t = dm_path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, str] = {}
    for name in (
        "errorEstimator",
        "scaledDeltaField",
        "refineInterval",
        "lowerRefineLevel",
        "unrefineLevel",
        "nBufferLayers",
        "maxRefinement",
        "dumpLevel",
        "beginUnrefine",
        "maxCells",
        "upperRefineLevel",
        "upperUnrefineLevel",
        "enableBalancing",
    ):
        m = re.search(rf"^{name}\s+([^;]+);", t, re.MULTILINE)
        if m:
            out[name] = m.group(1).strip()
    return out


def parse_max_refinement_from_dict(case_dir: Path) -> Optional[int]:
    p = case_dir / "constant" / "dynamicMeshDict"
    if not p.is_file():
        return None
    m = re.search(r"maxRefinement\s+(\d+)\s*;", p.read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def parse_blockmesh_cell_count(case_dir: Path) -> Optional[int]:
    p = case_dir / "system" / "blockMeshDict"
    if not p.is_file():
        return None
    t = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", t, re.DOTALL)
    if not m:
        return None
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return a * b * c


def _read_text_if(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def log_ok_finished(log_text: str) -> bool:
    if not log_text.strip():
        return False
    low = log_text.lower()
    if "foam fatal" in low or "foam aborting" in low:
        return False
    return "end" in low[-800:].lower() or "finished meshing" in low.lower() or "clocktime" in low.lower()


def parse_blastfoam_log(log_text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "foam_fatal": bool(re.search(r"FOAM\s+FATAL", log_text, re.I)),
        "blastfoam_started": "blastFoam" in log_text or "RK2SSP" in log_text or "compressibleSystem" in log_text,
        "refine_events": len(re.findall(r"Refined\s+from\s+\d+\s+to\s+\d+\s+cells", log_text)),
        "unrefine_events": len(re.findall(r"Unrefined\s+from\s+\d+\s+to\s+\d+\s+cells", log_text)),
        "final_time": None,
        "times": [],
        "cell_after_refine": [],
        "cell_series": [],
    }
    # \bTime ensures we do not also match inside ExecutionTime / ClockTime (those
    # carry the wall-clock seconds, not the simulation time the user expects).
    for m in re.finditer(r"\bTime\s*=\s*([0-9.eE+-]+)", log_text):
        try:
            out["times"].append(float(m.group(1)))
        except ValueError:
            pass
    if out["times"]:
        out["final_time"] = out["times"][-1]

    series: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"Refined\s+from\s+(\d+)\s+to\s+(\d+)\s+cells", log_text):
        series.append(("refine", int(m.group(1)), int(m.group(2))))
        out["cell_after_refine"].append(int(m.group(2)))
    for m in re.finditer(r"Unrefined\s+from\s+(\d+)\s+to\s+(\d+)\s+cells", log_text):
        series.append(("unrefine", int(m.group(1)), int(m.group(2))))
        out["cell_after_refine"].append(int(m.group(2)))
    out["cell_series"] = series

    # Initial / peak / final cell counts from AMR lines
    nums: List[int] = []
    for kind, a, b in series:
        nums.extend([a, b])
    if nums:
        out["peak_cell_count"] = max(nums)
        out["final_cell_count_from_log"] = nums[-1]
        out["initial_cell_count_from_log"] = nums[0]
    else:
        out["peak_cell_count"] = None
        out["final_cell_count_from_log"] = None
        out["initial_cell_count_from_log"] = None

    # First refine / unrefine order
    first_refine_idx = next((i for i, s in enumerate(series) if s[0] == "refine"), None)
    first_unref_idx = next((i for i, s in enumerate(series) if s[0] == "unrefine"), None)
    out["first_refine_index"] = first_refine_idx
    out["first_unrefine_index"] = first_unref_idx
    out["unrefine_after_first_refine"] = None
    if first_refine_idx is not None and first_unref_idx is not None:
        out["unrefine_after_first_refine"] = first_unref_idx > first_refine_idx

    # ExecutionTime last
    exec_times = [float(x) for x in re.findall(r"ExecutionTime\s*=\s*([0-9.]+)\s*s", log_text)]
    out["last_execution_time_s"] = exec_times[-1] if exec_times else None

    return out


def find_celllevel_path(case_dir: Path, time_name: Optional[str] = None) -> Optional[Path]:
    """Best-effort: serial time/cellLevel or processor0/time/cellLevel."""
    if time_name is None:
        # latest numeric time at case root or under processor0
        candidates: List[Tuple[float, Path]] = []
        proc0 = case_dir / "processor0"
        roots: List[Path] = [case_dir]
        if proc0.is_dir():
            roots.append(proc0)
        for root in roots:
            try:
                for name in os.listdir(root):
                    if name in ("0.orig", "constant", "system", "postProcessing"):
                        continue
                    if name.startswith("processor"):
                        continue
                    p = root / name
                    if not p.is_dir():
                        continue
                    try:
                        tf = float(name)
                    except ValueError:
                        continue
                    cl = p / "cellLevel"
                    if cl.is_file():
                        candidates.append((tf, cl))
            except OSError:
                continue
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    for root in (case_dir, case_dir / "processor0"):
        cl = root / time_name / "cellLevel"
        if cl.is_file():
            return cl
    return None


def parse_celllevel_histogram(path: Path) -> Optional[Dict[int, int]]:
    """ASCII volScalarField only; returns level -> count or None."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "format      binary" in raw[:2500]:
        return None
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\(", raw)
    if not m:
        m2 = re.search(r"internalField\s+uniform\s+([0-9.eE+-]+)", raw)
        if m2:
            # Uniform field: level is constant but total cell count is not in file
            v = int(float(m2.group(1)))
            return {v: -1}
        return None
    n = int(m.group(1))
    start = raw.find("(", m.end() - 1)
    end = raw.find(")", start)
    if start < 0 or end < 0:
        return None
    chunk = raw[start + 1 : end]
    vals: List[int] = []
    for tok in re.split(r"[\s\n]+", chunk.strip()):
        if not tok:
            continue
        try:
            vals.append(int(round(float(tok))))
        except ValueError:
            continue
    if len(vals) != n and len(vals) < max(1, n // 2):
        return None
    hist: Dict[int, int] = {}
    for v in vals:
        hist[v] = hist.get(v, 0) + 1
    return hist


def classify_variant(row: Dict[str, Any]) -> str:
    if row.get("foam_fatal") or row.get("stage_failed"):
        return "failed"
    re_n = int(row.get("refine_events") or 0)
    ue_n = int(row.get("unrefine_events") or 0)
    if re_n == 0 and ue_n == 0:
        return "no_AMR_activity"
    if re_n > 0 and ue_n == 0:
        return "refine_only"
    if re_n == 0 and ue_n > 0:
        return "unrefine_only_anomaly"

    peak = row.get("peak_cell_count")
    final = row.get("final_cell_count_effective")
    base = row.get("base_cell_count")
    ratio_pf = None
    if peak and final and final > 0:
        ratio_pf = float(peak) / float(final)
    ratio_fb = None
    if final and base and base > 0:
        ratio_fb = float(final) / float(base)

    too_big = False
    if peak and peak > 2_000_000:
        too_big = True
    if final and final > 800_000:
        too_big = True

    if too_big:
        return "too_expensive"

    good_unref = row.get("unrefine_after_first_refine") is True
    if good_unref and ratio_pf and ratio_pf > 1.05 and ratio_fb and ratio_fb < 80:
        return "good_release_candidate"

    if re_n > 0 and ue_n > 0:
        return "refine_and_unrefine"

    return "unknown"


def collect_variant_metrics(variant_dir: Path, base_cells: Optional[int]) -> Dict[str, Any]:
    max_ref_target = parse_max_refinement_from_dict(variant_dir) or SWEEP_MAX_REFINEMENT
    row: Dict[str, Any] = {
        "variant_dir": str(variant_dir),
        "base_cell_count": base_cells or parse_blockmesh_cell_count(variant_dir),
    }
    # Stages
    for name, key_ok in [
        ("log.blockMesh", "blockmesh_ok"),
        ("log.surfaceFeatures", "surfacefeatures_ok"),
        ("log.snappyHexMesh", "snappy_ok"),
        ("log.setRefinedFields", "setrefinedfields_ok"),
    ]:
        lt = _read_text_if(variant_dir / name)
        if not lt.strip():
            row[key_ok] = "NA"
        else:
            low = lt.lower()
            row[key_ok] = not ("foam fatal" in low or "foam aborting" in low) and (
                "end" in low[-400:] or "finished" in low or "clocktime" in low
            )
    # setFields fallback
    if row.get("setrefinedfields_ok") == "NA":
        sf = _read_text_if(variant_dir / "log.setFields")
        if sf.strip():
            low = sf.lower()
            row["setrefinedfields_ok"] = "foam fatal" not in low

    bf = _read_text_if(variant_dir / "log.blastFoam")
    if not bf.strip():
        # tee may write only blastFoam; try log.blastFoam from alternate
        for alt in variant_dir.glob("log.*"):
            if "blast" in alt.name.lower():
                bf = _read_text_if(alt)
                if bf.strip():
                    break

    parsed = parse_blastfoam_log(bf)
    row.update(
        {
            "foam_fatal": parsed["foam_fatal"],
            "blastfoam_started": parsed["blastfoam_started"],
            "refine_events": parsed["refine_events"],
            "unrefine_events": parsed["unrefine_events"],
            "final_time": parsed["final_time"],
            "peak_cell_count": parsed.get("peak_cell_count"),
            "final_cell_count_from_log": parsed.get("final_cell_count_from_log"),
            "initial_cell_count_from_log": parsed.get("initial_cell_count_from_log"),
            "unrefine_after_first_refine": parsed.get("unrefine_after_first_refine"),
            "last_execution_time_s": parsed.get("last_execution_time_s"),
        }
    )
    row["celllevel_path_exists"] = find_celllevel_path(variant_dir) is not None
    cl_path = find_celllevel_path(variant_dir)
    hist_final = parse_celllevel_histogram(cl_path) if cl_path else None
    row["celllevel_histogram_final"] = json.dumps(hist_final) if hist_final else "NA"
    if hist_final:
        row["max_celllevel_final"] = max(hist_final.keys())
        hl = sum(c for lv, c in hist_final.items() if c >= 0 and lv >= max_ref_target)
        tot = sum(c for c in hist_final.values() if c >= 0)
        row["high_level_cell_count_final"] = hl if tot > 0 else "NA"
        row["fraction_high_level_final"] = (hl / tot) if tot > 0 else None
    else:
        row["max_celllevel_final"] = "NA"
        row["high_level_cell_count_final"] = "NA"
        row["fraction_high_level_final"] = "NA"

    bc = row.get("base_cell_count")
    peak = row.get("peak_cell_count")
    final = row.get("final_cell_count_from_log")
    row["final_cell_count_effective"] = final
    if bc and peak and final:
        row["peak_over_final"] = float(peak) / float(final) if final else None
        row["final_over_base"] = float(final) / float(bc) if bc else None
    else:
        row["peak_over_final"] = None
        row["final_over_base"] = None

    row["stage_failed"] = False
    if row.get("blockmesh_ok") is False:
        row["stage_failed"] = True
    if row.get("snappy_ok") is False:
        row["stage_failed"] = True
    if row.get("setrefinedfields_ok") is False:
        row["stage_failed"] = True

    row["classification"] = classify_variant(row)
    return row


def write_run_all_sh(out_root: Path, variant_ids: Iterable[str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "# Developer-only: run all AMR sweep variants sequentially in WSL.",
        "# Continues after a variant fails; see log.* inside each variant dir.",
        "set +e",
        f'ROOT="$(cd "$(dirname "$0")" && pwd)"',
        'export WM_PROJECT_DIR="${WM_PROJECT_DIR:-/opt/openfoam9}"',
        "set +o pipefail  # OF bashrc can trip pipefail under strict modes",
        'source "${WM_PROJECT_DIR}/etc/bashrc" 2>/dev/null || true',
        "failures=0",
        "",
    ]
    for vid in variant_ids:
        lines.append(f'echo "========== {vid} =========="')
        lines.append(f'if [ -d "$ROOT/variants/{vid}" ]; then')
        lines.append(f'  (cd "$ROOT/variants/{vid}" && chmod +x Allrun 2>/dev/null || true && ./Allrun)')
        lines.append(f'  ec=$?')
        lines.append(f'  if [ "$ec" -eq 0 ]; then echo "[OK] {vid}"; else echo "[FAIL] {vid} exit $ec" >&2; failures=$((failures+1)); fi')
        lines.append("else")
        lines.append(f'  echo "Missing $ROOT/variants/{vid}" >&2; failures=$((failures+1))')
        lines.append("fi")
        lines.append("")
    lines.extend(
        [
            'echo "========== SWEEP DONE: failures=$failures =========="',
            "exit 0",
        ]
    )
    script = "\n".join(lines) + "\n"
    p = out_root / "run_all.sh"
    p.write_text(script, encoding="utf-8", newline="\n")


def run_wsl_allrun(case_dir: Path, timeout: int) -> Tuple[int, str]:
    wsl_case = win_to_wsl_path(str(case_dir.resolve()))
    # Allrun uses set -e internally; keep this subshell permissive so OF bashrc cannot abort early.
    inner = f'source /opt/openfoam9/etc/bashrc 2>/dev/null; cd "{wsl_case}" && chmod +x Allrun 2>/dev/null && ./Allrun'
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


def _ignore_processor_post(dir: str, names: List[str]) -> List[str]:
    return [n for n in names if n.startswith("processor") or n == "postProcessing"]


def ensure_baseline(out_root: Path, base_case: Optional[Path], dry_run: bool) -> Path:
    baseline = out_root / "baseline_case"
    if base_case and base_case.is_dir():
        if not dry_run:
            if baseline.exists():
                shutil.rmtree(baseline)
            shutil.copytree(base_case, baseline, ignore=_ignore_processor_post)
        return baseline
    if dry_run and not baseline.exists():
        return baseline
    inputs = baseline_case_inputs()
    gen = Generator3D(str(out_root))
    if not dry_run:
        gen.generate("baseline_case", inputs)
    return baseline


def write_variant_metadata(vdir: Path, spec: AMRVariantSpec, dm_text: str) -> None:
    meta = {
        **asdict(spec),
        "refine_interval": SWEEP_REFINE_INTERVAL,
        "max_refinement": SWEEP_MAX_REFINEMENT,
        "dynamicMeshDict_excerpt": dm_text[-1200:] if len(dm_text) > 1200 else dm_text,
    }
    (vdir / "amr_sweep_variant.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def extract_cellcount_vs_time(log_text: str) -> List[Tuple[float, int]]:
    """Best-effort: pair last 'Time = t' before each Refine/Unrefine line with new cell count."""
    current_time: Optional[float] = None
    series: List[Tuple[float, int]] = []
    for line in log_text.splitlines():
        mt = re.search(r"^\s*\bTime\s*=\s*([0-9.eE+-]+)\s*$", line)
        if mt:
            try:
                current_time = float(mt.group(1))
            except ValueError:
                current_time = None
            continue
        mr = re.search(r"(?:Refined|Unrefined)\s+from\s+\d+\s+to\s+(\d+)\s+cells", line)
        if mr and current_time is not None:
            series.append((current_time, int(mr.group(1))))
    return series


def maybe_plot_series(out_root: Path, variant_id: str, series: List[Tuple[str, int, int]], log_text: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if series:
        xs = list(range(len(series)))
        ys = [t[2] for t in series]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, "o-", label="cells after event")
        plt.xlabel("event index (refine/unrefine order)")
        plt.ylabel("cell count")
        plt.title(f"{variant_id}: cell count through AMR events")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{variant_id}_cells_by_event.png", dpi=120)
        plt.close()
    tv = extract_cellcount_vs_time(log_text) if log_text else []
    if len(tv) >= 2:
        plt.figure(figsize=(8, 4))
        plt.plot([t[0] for t in tv], [t[1] for t in tv], "s-", markersize=3)
        plt.xlabel("simulation time")
        plt.ylabel("cell count (after event)")
        plt.title(f"{variant_id}: cell count vs time (log-ordered)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{variant_id}_cells_vs_time.png", dpi=120)
        plt.close()


def write_reports(
    out_root: Path,
    rows: List[Dict[str, Any]],
    variants: Tuple[AMRVariantSpec, ...],
) -> None:
    csv_path = out_root / "amr_tuning_summary.csv"
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)
    else:
        csv_path.write_text(
            "# No metric rows yet. After WSL runs use: python tools/amr_tuning_sweep.py --out <this_dir> --collect-only\n",
            encoding="utf-8",
        )

    md_lines = [
        "# AMR tuning sweep report (developer-only)",
        "",
        "This study does **not** change GUI defaults. Recommendations are informational.",
        "",
        "## Sweep parameters (held constant)",
        "",
        f"- refineInterval: {SWEEP_REFINE_INTERVAL}",
        f"- maxRefinement: {SWEEP_MAX_REFINEMENT}",
        "- outside_extent / charge seed: see baseline `case_init_mode.json` (not varied in this sweep).",
        "",
        "## Variants",
        "",
        "| id | estimator | nBufferLayers | lowerRefineLevel | unrefineLevel | classification |",
        "|----|-----------|---------------|------------------|---------------|----------------|",
    ]
    by_id = {r.get("variant_id"): r for r in rows}
    for spec in variants:
        r = by_id.get(spec.variant_id, {})
        md_lines.append(
            f"| {spec.variant_id} | {spec.estimator} | {spec.n_buffer_layers} | {spec.lower_refine_level} | {spec.unrefine_level} | {r.get('classification', 'NA')} |"
        )
    md_lines.extend(
        [
            "",
            "## dynamicMeshDict paths",
            "",
        ]
    )
    for spec in variants:
        md_lines.append(f"- `{spec.variant_id}`: `variants/{spec.variant_id}/constant/dynamicMeshDict`")
    md_lines.extend(["", "## Parsed AMR keys (manual review)", ""])
    md_lines.append("| variant | estimator | nBuffer | lowerRef | unref | refineInt | maxRef | beginUnref | maxCells |")
    md_lines.append("|---------|-----------|---------|----------|-------|-----------|--------|------------|----------|")
    for spec in variants:
        dm = out_root / "variants" / spec.variant_id / "constant" / "dynamicMeshDict"
        fld = extract_dynamic_mesh_review_fields(dm)
        est = fld.get("errorEstimator", spec.estimator)
        if fld.get("scaledDeltaField"):
            est = f"scaledDelta ({fld.get('scaledDeltaField')})"
        md_lines.append(
            "| {vid} | {est} | {nb} | {lr} | {ur} | {ri} | {mr} | {bu} | {mc} |".format(
                vid=spec.variant_id,
                est=est,
                nb=fld.get("nBufferLayers", str(spec.n_buffer_layers)),
                lr=fld.get("lowerRefineLevel", str(spec.lower_refine_level)),
                ur=fld.get("unrefineLevel", str(spec.unrefine_level)),
                ri=fld.get("refineInterval", str(SWEEP_REFINE_INTERVAL)),
                mr=fld.get("maxRefinement", str(SWEEP_MAX_REFINEMENT)),
                bu=fld.get("beginUnrefine", "—"),
                mc=fld.get("maxCells", "—"),
            )
        )
    md_lines.extend(["", "## Full dynamicMeshDict per variant", ""])
    for spec in variants:
        dm = out_root / "variants" / spec.variant_id / "constant" / "dynamicMeshDict"
        md_lines.append(f"### {spec.variant_id}")
        md_lines.append(f"- File: `{dm.as_posix()}`")
        if dm.is_file():
            md_lines.append("```")
            md_lines.append(dm.read_text(encoding="utf-8", errors="replace").strip())
            md_lines.append("```")
        md_lines.append("")

    md_lines.extend(["## Recommendations (next sweep)", ""])
    good = [r for r in rows if r.get("classification") == "good_release_candidate"]
    if good:
        md_lines.append("**Good release candidates (heuristic):** " + ", ".join(str(r.get("variant_id")) for r in good))
    else:
        md_lines.append("- No automatic `good_release_candidate` labels; inspect `refine_and_unrefine` rows manually.")
    md_lines.extend(
        [
            "- Consider narrowing thresholds around the best `refine_and_unrefine` variants.",
            "- Add beginUnrefine / maxCells variants only after baseline behavior is understood.",
            "",
            "## Limitations",
            "",
            "- Metrics are best-effort from logs; parallel `cellLevel` may be missing if times were not reconstructed.",
            "- Short endTime may miss late-stage unrefinement.",
            "",
        ]
    )
    (out_root / "amr_tuning_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Developer AMR tuning sweep (3D blastFoam).")
    ap.add_argument("--out", type=Path, required=True, help="Output root directory")
    ap.add_argument("--base-case", type=Path, default=None, help="Existing case to clone; default: generate spherical baseline")
    ap.add_argument("--dry-run", action="store_true", help="Only plan: print actions, optionally write scripts without full copy")
    ap.add_argument("--run-wsl", action="store_true", help="Run ./Allrun per variant via WSL (sequential)")
    ap.add_argument("--wsl-timeout", type=int, default=3600, help="Per-variant WSL timeout (seconds)")
    ap.add_argument("--collect-only", action="store_true", help="Only rescan variants/*/ and refresh CSV/MD")
    args = ap.parse_args()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.collect_only:
        rows: List[Dict[str, Any]] = []
        vroot = out_root / "variants"
        base_cells = parse_blockmesh_cell_count(out_root / "baseline_case")
        if vroot.is_dir():
            for sub in sorted(vroot.iterdir()):
                if not sub.is_dir():
                    continue
                meta_p = sub / "amr_sweep_variant.json"
                vid = sub.name
                mrow = collect_variant_metrics(sub, base_cells)
                mrow["variant_id"] = vid
                if meta_p.is_file():
                    try:
                        mrow["variant_spec_json"] = meta_p.read_text(encoding="utf-8")
                    except OSError:
                        pass
                # plots from log
                bf = _read_text_if(sub / "log.blastFoam")
                ser = parse_blastfoam_log(bf).get("cell_series") or []
                maybe_plot_series(out_root, vid, ser, bf)
                rows.append(mrow)
        write_reports(out_root, rows, DEFAULT_VARIANTS)
        print(f"Collected {len(rows)} variants -> {out_root / 'amr_tuning_summary.csv'}")
        return 0

    print(f"AMR sweep out_root={out_root}")
    baseline = ensure_baseline(out_root, args.base_case, args.dry_run)
    print(f"Baseline: {baseline}")

    variant_rows: List[Dict[str, Any]] = []
    if not args.dry_run:
        base_cells = parse_blockmesh_cell_count(baseline)
        variants_root = out_root / "variants"
        variants_root.mkdir(parents=True, exist_ok=True)
        for spec in DEFAULT_VARIANTS:
            vdir = variants_root / spec.variant_id
            if vdir.exists():
                shutil.rmtree(vdir)
            shutil.copytree(baseline, vdir)
            dm = vdir / "constant" / "dynamicMeshDict"
            patch_dynamic_mesh_dict(
                dm,
                estimator=spec.estimator,
                n_buffer_layers=spec.n_buffer_layers,
                lower_refine_level=spec.lower_refine_level,
                unrefine_level=spec.unrefine_level,
                refine_interval=SWEEP_REFINE_INTERVAL,
                max_refinement=SWEEP_MAX_REFINEMENT,
                dump_level=True,
            )
            dm_text = dm.read_text(encoding="utf-8")
            write_variant_metadata(vdir, spec, dm_text)
            print(f"  wrote variant {spec.variant_id}")

        write_run_all_sh(out_root, [v.variant_id for v in DEFAULT_VARIANTS])

        if args.run_wsl:
            for spec in DEFAULT_VARIANTS:
                vdir = variants_root / spec.variant_id
                print(f"  WSL Allrun {spec.variant_id} (timeout {args.wsl_timeout}s) ...")
                code, tail = run_wsl_allrun(vdir, args.wsl_timeout)
                log_extra = vdir / "log.amr_sweep_wsl.txt"
                log_extra.write_text(tail[-20000:], encoding="utf-8")
                print(f"    exit {code}")
                mrow = collect_variant_metrics(vdir, base_cells)
                mrow["variant_id"] = spec.variant_id
                mrow["wsl_exit_code"] = code
                bf = _read_text_if(vdir / "log.blastFoam")
                ser = parse_blastfoam_log(bf).get("cell_series") or []
                maybe_plot_series(out_root, spec.variant_id, ser, bf)
                variant_rows.append(mrow)
        else:
            for spec in DEFAULT_VARIANTS:
                vdir = variants_root / spec.variant_id
                variant_rows.append(
                    {
                        "variant_id": spec.variant_id,
                        "note": "generated_not_run",
                        **asdict(spec),
                    }
                )
    else:
        print("Dry-run: would generate baseline +", len(DEFAULT_VARIANTS), "variants")
        for spec in DEFAULT_VARIANTS:
            print(f"  {spec.variant_id}: {spec.estimator} nBL={spec.n_buffer_layers} lr={spec.lower_refine_level} ur={spec.unrefine_level}")
        write_run_all_sh(out_root, [v.variant_id for v in DEFAULT_VARIANTS])
        print(f"Wrote {out_root / 'run_all.sh'} (paths only)")

    write_reports(out_root, variant_rows, DEFAULT_VARIANTS)
    print(f"Report: {out_root / 'amr_tuning_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
