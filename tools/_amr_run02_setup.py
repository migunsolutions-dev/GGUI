#!/usr/bin/env python3
"""Developer one-shot: set up the run02 AMR validation sweep.

Generates `_amr_tuning_run02/` with a fresh 3D spherical baseline case and
a customised variant matrix targeted at endTime = 5e-4 s, writeInterval =
1e-4 s, parallel decomposition 2x2x2. The variant matrix is a strict subset
of (and extension to) `tools/amr_tuning_sweep.py::DEFAULT_VARIANTS`:

  Required (production-relevant):
    1) dg_nb1_lr020_ur010
    2) dg_nb2_lr010_ur010
    3) sdp_nb1_lr020_ur010
    4) sdp_nb2_lr010_ur010

  Optional diagnostic / sensitivity:
    5) sdp_nb0_lr020_ur010    (nBufferLayers=0 — diagnostic only)
    6) dg_nb0_lr020_ur010     (nBufferLayers=0 — diagnostic only)
    7) dg_nb1_lr005_ur010     (densityGradient, lowerRefineLevel=0.05)
    8) dg_nb2_lr005_ur010     (densityGradient, lowerRefineLevel=0.05)

The script does not change GUI defaults or modify any other workflow. It is
intended for the run02 validation only.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass, asdict, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse helpers from the existing sweep tool. The module name is sanitized.
import importlib.util

_sweep_path = _REPO_ROOT / "tools" / "amr_tuning_sweep.py"
_sweep_name = "ggui_amr_tuning_sweep_run02"
_spec = importlib.util.spec_from_file_location(_sweep_name, _sweep_path)
_sweep_mod = importlib.util.module_from_spec(_spec)
sys.modules[_sweep_name] = _sweep_mod
assert _spec.loader is not None
_spec.loader.exec_module(_sweep_mod)

from generator_3d import Generator3D  # noqa: E402
from models import CaseInputs3D  # noqa: E402


@dataclass(frozen=True)
class Run02Variant:
    variant_id: str
    estimator: str  # "densityGradient" | "scaledDelta_p"
    n_buffer_layers: int
    lower_refine_level: float
    unrefine_level: float
    role: str  # "required" | "diagnostic" | "sensitivity"


REQUIRED_VARIANTS = (
    Run02Variant("dg_nb1_lr020_ur010", "densityGradient", 1, 0.20, 0.10, "required"),
    Run02Variant("dg_nb2_lr010_ur010", "densityGradient", 2, 0.10, 0.10, "required"),
    Run02Variant("sdp_nb1_lr020_ur010", "scaledDelta_p", 1, 0.20, 0.10, "required"),
    Run02Variant("sdp_nb2_lr010_ur010", "scaledDelta_p", 2, 0.10, 0.10, "required"),
)
OPTIONAL_VARIANTS = (
    Run02Variant("sdp_nb0_lr020_ur010", "scaledDelta_p", 0, 0.20, 0.10, "diagnostic"),
    Run02Variant("dg_nb0_lr020_ur010", "densityGradient", 0, 0.20, 0.10, "diagnostic"),
    Run02Variant("dg_nb1_lr005_ur010", "densityGradient", 1, 0.05, 0.10, "sensitivity"),
    Run02Variant("dg_nb2_lr005_ur010", "densityGradient", 2, 0.05, 0.10, "sensitivity"),
)

SWEEP_REFINE_INTERVAL = 3
SWEEP_MAX_REFINEMENT = 3


def baseline_case_inputs_run02() -> CaseInputs3D:
    """Same baseline geometry as run01 but at endTime 5e-4 s, parallel 2x2x2."""
    base = _sweep_mod.baseline_case_inputs()
    return replace(
        base,
        end_time_s=5.0e-4,
        write_control_type="adjustableRunTime",
        write_interval_time=1.0e-4,
        cores=8,
        decomposition_method="simple",
        decomposition_simple_n=(2, 2, 2),
        decomposition_simple_delta=0.001,
    )


def patch_control_dict(case_dir: Path, *, end_time: float, write_interval: float) -> None:
    """Idempotently force endTime/writeInterval on an already-generated controlDict."""
    p = case_dir / "system" / "controlDict"
    if not p.is_file():
        raise FileNotFoundError(p)
    txt = p.read_text(encoding="utf-8", errors="replace")
    # endTime
    txt = re.sub(r"^(\s*endTime\s+)[^;]+;", lambda m: f"{m.group(1)}{end_time:g};", txt, count=1, flags=re.MULTILINE)
    # writeControl + writeInterval (use adjustableRunTime + time-based interval for cellLevel snapshots)
    if re.search(r"^\s*writeControl\s+", txt, re.MULTILINE):
        txt = re.sub(r"^(\s*writeControl\s+)[^;]+;", r"\1adjustableRunTime;", txt, count=1, flags=re.MULTILINE)
    if re.search(r"^\s*writeInterval\s+", txt, re.MULTILINE):
        txt = re.sub(
            r"^(\s*writeInterval\s+)[^;]+;",
            lambda m: f"{m.group(1)}{write_interval:g};",
            txt,
            count=1,
            flags=re.MULTILINE,
        )
    # Make sure the output format is ascii so cellLevel histograms can be parsed
    if re.search(r"^\s*writeFormat\s+", txt, re.MULTILINE):
        txt = re.sub(r"^(\s*writeFormat\s+)[^;]+;", r"\1ascii;", txt, count=1, flags=re.MULTILINE)
    p.write_text(txt, encoding="utf-8", newline="\n")


def write_run02_run_all_sh(out_root: Path, variant_ids):
    """Variant of write_run_all_sh: forwards each Allrun stdout to the consolidated log.

    Identical to amr_tuning_sweep.write_run_all_sh except we explicitly tee Allrun
    output through `tee -a` so that even if a per-case log.blastFoam gets clobbered
    by a later run, the consolidated log retains every blastFoam stream.
    """
    lines = [
        "#!/usr/bin/env bash",
        "# Developer-only: run02 AMR sweep variants sequentially in WSL.",
        "# Continues after a variant fails; see log.* inside each variant dir.",
        "set +e",
        'ROOT="$(cd "$(dirname "$0")" && pwd)"',
        'export WM_PROJECT_DIR="${WM_PROJECT_DIR:-/opt/openfoam9}"',
        "set +o pipefail",
        'source "${WM_PROJECT_DIR}/etc/bashrc" 2>/dev/null || true',
        'CONS="$ROOT/sweep_console.log"',
        ': > "$CONS"',
        "failures=0",
        "",
    ]
    for vid in variant_ids:
        lines.append(f'echo "========== {vid} ==========" | tee -a "$CONS"')
        lines.append(f'date | tee -a "$CONS"')
        lines.append(f'if [ -d "$ROOT/variants/{vid}" ]; then')
        lines.append(
            f'  (cd "$ROOT/variants/{vid}" && chmod +x Allrun 2>/dev/null || true && ./Allrun 2>&1) | tee -a "$CONS"'
        )
        lines.append("  ec=${PIPESTATUS[0]}")
        lines.append(
            f'  if [ "$ec" -eq 0 ]; then echo "[OK] {vid}" | tee -a "$CONS"; '
            f'else echo "[FAIL] {vid} exit $ec" | tee -a "$CONS"; failures=$((failures+1)); fi'
        )
        lines.append("else")
        lines.append(f'  echo "Missing $ROOT/variants/{vid}" | tee -a "$CONS"; failures=$((failures+1))')
        lines.append("fi")
        lines.append("")
    lines.extend(
        [
            'echo "========== SWEEP DONE: failures=$failures ==========" | tee -a "$CONS"',
            "exit 0",
        ]
    )
    script = "\n".join(lines) + "\n"
    (out_root / "run_all.sh").write_text(script, encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "_amr_tuning_run02")
    ap.add_argument(
        "--include-optional",
        action="store_true",
        help="Also generate the 4 optional diagnostic/sensitivity variants.",
    )
    ap.add_argument(
        "--end-time",
        type=float,
        default=5.0e-4,
        help="endTime in seconds (default 5e-4 = primary target).",
    )
    ap.add_argument(
        "--write-interval",
        type=float,
        default=1.0e-4,
        help="writeInterval (s) for cellLevel snapshots (default 1e-4 = 5 saves).",
    )
    args = ap.parse_args()

    out_root: Path = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    variants_root = out_root / "variants"
    variants_root.mkdir(parents=True, exist_ok=True)

    inputs = baseline_case_inputs_run02()
    print(f"out_root           : {out_root}")
    print(f"endTime            : {args.end_time:g} s")
    print(f"writeInterval      : {args.write_interval:g} s")
    print(f"cores / decompose  : {inputs.cores} / simple {inputs.decomposition_simple_n}")
    print(f"maxRefinement      : {SWEEP_MAX_REFINEMENT}")
    print(f"refineInterval     : {SWEEP_REFINE_INTERVAL}")
    print()

    # Generate baseline once
    baseline = out_root / "baseline_case"
    if baseline.exists():
        shutil.rmtree(baseline)
    gen = Generator3D(str(out_root))
    gen.generate("baseline_case", inputs)
    patch_control_dict(baseline, end_time=args.end_time, write_interval=args.write_interval)

    # Copy + patch dynamicMeshDict per variant
    chosen = list(REQUIRED_VARIANTS)
    if args.include_optional:
        chosen.extend(OPTIONAL_VARIANTS)
    for spec in chosen:
        vdir = variants_root / spec.variant_id
        if vdir.exists():
            shutil.rmtree(vdir)
        shutil.copytree(baseline, vdir)
        dm = vdir / "constant" / "dynamicMeshDict"
        _sweep_mod.patch_dynamic_mesh_dict(
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
        # Reuse the existing variant-metadata writer; cast Run02Variant -> AMRVariantSpec dataclass dict
        _sweep_mod.write_variant_metadata(
            vdir,
            _sweep_mod.AMRVariantSpec(
                variant_id=spec.variant_id,
                estimator=spec.estimator,
                n_buffer_layers=spec.n_buffer_layers,
                lower_refine_level=spec.lower_refine_level,
                unrefine_level=spec.unrefine_level,
            ),
            dm_text,
        )
        # Persist the run02 role in a separate sidecar for the report.
        (vdir / "amr_sweep_variant_run02.json").write_text(
            __import__("json").dumps(asdict(spec), indent=2), encoding="utf-8"
        )
        print(f"  prepared {spec.variant_id} (role={spec.role})")

    # run_all.sh with consolidated log capture
    write_run02_run_all_sh(out_root, [v.variant_id for v in chosen])
    (out_root / "VARIANT_MATRIX.json").write_text(
        __import__("json").dumps([asdict(v) for v in chosen], indent=2), encoding="utf-8"
    )
    print(f"\nrun_all.sh         : {out_root / 'run_all.sh'}")
    print(f"VARIANT_MATRIX.json: {out_root / 'VARIANT_MATRIX.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
