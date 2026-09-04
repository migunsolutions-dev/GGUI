"""Re-run GGUI 1D handoff + 2D remap using the GUI watchdog stop."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from remap_fields_2d import _parse_internal_field
from viper_compare.ggui_run import (
    generate_and_run,
    generate_case,
    inputs_1d,
    inputs_2d,
    run_allrun_with_optional_watchdog,
    run_case_command,
)
from viper_compare.physics import default_test

RUN = r"C:\VIPER_COMPARE\20260904_222616"
INIT_ONLY = (
    "set -o pipefail; sed -i 's/\\r$//' Allrun Allclean remap_2d.py 2>/dev/null || true; "
    "chmod +x Allrun Allclean 2>/dev/null || true; "
    "rm -rf 0 && cp -r 0.orig 0 && blockMesh && "
    "postProcess -func writeCellCentres && python3 remap_2d.py && checkMesh"
)
MIN_AIR_DENSITY = 0.05


def _load(path, vector=False):
    arr, default = _parse_internal_field(path, is_vector=vector)
    if arr is None:
        return None, default
    return np.asarray(arr, dtype=float), default


def inspect_zero_fields(case_dir: str) -> dict:
    zero = os.path.join(case_dir, "0")
    names = sorted(
        n
        for n in os.listdir(zero)
        if os.path.isfile(os.path.join(zero, n)) and not n.startswith(".")
    ) if os.path.isdir(zero) else []
    report = {"dir": zero, "fields": names, "stats": {}}
    loaded = {}
    for name in names:
        path = os.path.join(zero, name)
        arr, default = _load(path, vector=name == "U")
        if arr is None:
            report["stats"][name] = {"uniform": default if not hasattr(default, "tolist") else default.tolist()}
            loaded[name] = default
            continue
        a = np.asarray(arr, dtype=float)
        entry = {
            "n": int(a.shape[0] if a.ndim else a.size),
            "min": float(np.nanmin(a)),
            "max": float(np.nanmax(a)),
            "nan": int(np.isnan(a).sum()),
            "inf": int(np.isinf(a).sum()),
        }
        if a.ndim == 1:
            entry["neg"] = int((a < 0).sum())
            entry["near_zero"] = int((np.abs(a) < 1e-6).sum())
        report["stats"][name] = entry
        loaded[name] = a
    ra = loaded.get("rho.air")
    rc = loaded.get("rho.c4")
    al = loaded.get("alpha.c4")
    if isinstance(ra, np.ndarray) and isinstance(rc, np.ndarray) and isinstance(al, np.ndarray):
        mix = np.asarray(al) * np.asarray(rc) + (1.0 - np.asarray(al)) * np.asarray(ra)
        report["rho_mix"] = {
            "min": float(np.nanmin(mix)),
            "max": float(np.nanmax(mix)),
            "near_zero": int((mix < 1e-4).sum()),
        }
        report["alpha.c4_range"] = [float(np.min(al)), float(np.max(al))]
    report["invalid"] = False
    reasons = []
    if isinstance(ra, np.ndarray):
        if float(np.nanmin(ra)) < MIN_AIR_DENSITY:
            report["invalid"] = True
            reasons.append(f"rho.air min {float(np.nanmin(ra)):.3e} < {MIN_AIR_DENSITY}")
        if int(np.isnan(ra).sum()) or int(np.isinf(ra).sum()):
            report["invalid"] = True
            reasons.append("rho.air has NaN/Inf")
    for name in ("p", "T", "U"):
        arr = loaded.get(name)
        if isinstance(arr, np.ndarray) and (np.isnan(arr).any() or np.isinf(arr).any()):
            report["invalid"] = True
            reasons.append(f"{name} has NaN/Inf")
    report["reasons"] = reasons
    return report


def main() -> int:
    spec = default_test()
    log_dir = os.path.join(RUN, "ggui", "1d_to_2d_massfix")
    os.makedirs(log_dir, exist_ok=True)
    print("Generating 1D remap precursor with SolverRunner-equivalent watchdog ...", flush=True)
    g1 = generate_and_run(
        prefix="Case_1D_viperHandoff",
        inputs=inputs_1d(
            spec,
            remap_for_2d=True,
            radius=spec.r_remap_m,
            gauges=[r for r in spec.r_gauges_1d if r < spec.r_remap_m],
        ),
        log_dir=log_dir,
        watchdog=True,
    )
    print(json.dumps({k: g1[k] for k in g1 if k != "safe"}, indent=2), flush=True)
    if g1["returncode"] != 0:
        return 1

    print("Generating 2D remap case (init only first) ...", flush=True)
    gen2 = generate_case(
        prefix="Case_2D_viperHandoff",
        inputs=inputs_2d(spec, remapped=True, source_1d=g1["case_dir"], cores=4),
    )
    print(json.dumps(gen2, indent=2), flush=True)
    init2 = run_case_command(
        case_dir=gen2["case_dir"],
        log_dir=log_dir,
        prefix="Case_2D_viperHandoff_init",
        command=INIT_ONLY,
    )
    print(json.dumps({k: init2[k] for k in init2 if k != "safe"}, indent=2), flush=True)
    zero = inspect_zero_fields(gen2["case_dir"])
    with open(os.path.join(log_dir, "init_0_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(zero, handle, indent=2)
    print(json.dumps(zero, indent=2), flush=True)
    if init2["returncode"] != 0 or init2.get("crashed") or zero.get("invalid"):
        payload = {"1d": g1, "2d_generate": gen2, "2d_init": init2, "init_0": zero}
        with open(os.path.join(log_dir, "recover_result.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return 2

    print("Initialized 0/ is finite; starting 2D Allrun ...", flush=True)
    g2 = run_allrun_with_optional_watchdog(
        case_dir=gen2["case_dir"],
        log_dir=log_dir,
        prefix="Case_2D_viperHandoff",
        watchdog=False,
    )
    g2 = {"name": gen2["name"], "generate_s": gen2["generate_s"], **g2}
    with open(os.path.join(log_dir, "Case_2D_viperHandoff_result.json"), "w", encoding="utf-8") as handle:
        json.dump(g2, handle, indent=2)
    print(json.dumps({k: g2[k] for k in g2 if k != "safe"}, indent=2), flush=True)
    payload = {"1d": g1, "2d_generate": gen2, "2d_init": init2, "init_0": zero, "2d": g2}
    with open(os.path.join(log_dir, "recover_result.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return 0 if g2["returncode"] == 0 and not g2.get("crashed") else 3


if __name__ == "__main__":
    raise SystemExit(main())
