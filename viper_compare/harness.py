"""Orchestrate the VIPER vs GGUI/blastFoam free-air comparison.

Never edits TEST2.vip in place. Never writes under Program Files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from remap_handoff_1d import HANDOFF_CRITERION, HANDOFF_RULE, REMAP_FRONT_BUFFER_CELLS_1D
from validation.kb_propagation import copied_1d2d_radius_m
from viper_compare.analyze import gauge_row, pair_error, remap_eligible_rows
from viper_compare.extract import load_ggui_probe, parse_viper_th
from viper_compare.ggui_run import generate_and_run, inputs_1d, inputs_2d
from viper_compare.physics import TestDefinition, default_test
from viper_compare.plots import plot_error_vs_r, plot_impulse_vs_r, plot_peak_vs_r, plot_pt_overlays
from viper_compare.remap_proof import remap_propagation_report, require_remap_propagation
from viper_compare.schema import extract_gauge_schema, schema_report
from viper_compare.vip_diff import assert_remap_identity
from viper_compare.vip_gauges import build_model
from viper_compare.viper_run import require_nonempty_th, run_viper

NO_REMAP_VIP = r"C:\VIPER_COMPARE\templates\12_pair\1\No_Remap.vip"
NO_REMAP_JSON = r"C:\VIPER_COMPARE\templates\12_pair\1\No_Remap.json"
REMAP_VIP = r"C:\VIPER_COMPARE\templates\12_pair\2\Remap.vip"
REMAP_JSON = r"C:\VIPER_COMPARE\templates\12_pair\2\Remap.json"
VIPER_EXE = r"C:\Program Files\viperblast_1.31\viperblast.exe"
WORK_ROOT = r"C:\VIPER_COMPARE"
COST_LIMIT_S = 90 * 60


class StopHarness(RuntimeError):
    """Authoritative stop condition."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict:
    def run(args):
        proc = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, check=False
        )
        return (proc.stdout or "").strip()

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "revision": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def _write_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _make_run_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(WORK_ROOT, ts)
    for sub in (
        "config",
        "viper/1d",
        "viper/2d_direct",
        "viper/1d_to_2d",
        "ggui/1d",
        "ggui/2d_direct",
        "ggui/1d_to_2d",
        "extracted",
        "plots",
        "report",
    ):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root


def _write_vip_json(
    spec: TestDefinition, dest: str, domain_1d_m: float, template_json: str
) -> str:
    payload = spec.viper_json(template_json, domain_1d_m=domain_1d_m)
    if "shape" in payload.get("params_2d", {}):
        # Keep the template identity; physics.viper_json must not rewrite it.
        pass
    _write_json(dest, payload)
    return dest


def _histories_to_rows(
    *,
    solver: str,
    configuration: str,
    dimension: str,
    remapped: bool,
    radii: list,
    labels: list,
    times: np.ndarray,
    pressure: np.ndarray,
    impulse: np.ndarray | None,
    spec: TestDefinition,
    receive_r_max: float | None,
    source_case: str,
    source_file: str,
) -> list:
    rows = []
    n = min(len(radii), pressure.shape[1] if pressure.size else 0)
    for i in range(n):
        native = None
        if impulse is not None and impulse.size and i < impulse.shape[1]:
            native = impulse[:, i]
        rows.append(
            gauge_row(
                solver=solver,
                configuration=configuration,
                dimension=dimension,
                remapped=remapped,
                gauge_label=labels[i] if i < len(labels) else f"g{i}",
                r_m=float(radii[i]),
                mass_kg=spec.mass_kg,
                times=times,
                pressure=pressure[:, i],
                native_impulse=native,
                p_atm=spec.p_atm,
                receive_r_max=receive_r_max,
                dx_2d=spec.dx_2d,
                source_case=source_case,
                source_file=source_file,
            )
        )
    return rows


def _load_viper_case(case_dir: str, need_1d: bool, need_2d: bool):
    p1, i1, p2, i2 = None, None, None, None
    if need_1d:
        _, t, p1 = parse_viper_th(os.path.join(case_dir, "viper1d_th_overpressure.txt"))
        _, _, i1 = parse_viper_th(os.path.join(case_dir, "viper1d_th_impulse.txt"))
        p1 = (t, p1)
        i1 = (t, i1)
    if need_2d:
        _, t2, p2 = parse_viper_th(os.path.join(case_dir, "viper2d_th_overpressure.txt"))
        _, _, i2 = parse_viper_th(os.path.join(case_dir, "viper2d_th_impulse.txt"))
        p2 = (t2, p2)
        i2 = (t2, i2)
    return p1, i1, p2, i2


def main() -> int:
    for path in (NO_REMAP_VIP, NO_REMAP_JSON, REMAP_VIP, REMAP_JSON, VIPER_EXE):
        if not os.path.isfile(path):
            raise StopHarness(f"Missing template or executable {path}")

    spec = default_test()
    no_remap_hash = _sha256(NO_REMAP_VIP)
    remap_hash = _sha256(REMAP_VIP)
    run_dir = _make_run_dir()
    log_path = os.path.join(run_dir, "report", "harness.log")

    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("No_Remap.vip sha256 " + no_remap_hash)
    log("Remap.vip sha256 " + remap_hash)
    log("\n" + schema_report())
    schema = extract_gauge_schema(NO_REMAP_VIP)
    _write_json(
        os.path.join(run_dir, "config", "test2_gauge_schema.json"),
        {
            "numthloc_1d": schema.numthloc_1d,
            "thlocx_1d": schema.thlocx_1d,
            "labels_1d": schema.labels_1d,
            "numthloc_2d": schema.numthloc_2d,
            "thlocx_2d": schema.thlocx_2d,
            "thlocy_2d": schema.thlocy_2d,
            "labels_2d": schema.labels_2d,
            "pressure_1d": schema.pressure_1d,
            "impulse_1d": schema.impulse_1d,
            "pressure_2d": schema.pressure_2d,
            "impulse_2d": schema.impulse_2d,
        },
    )
    log(spec.cost_report())
    match_rows = [r.__dict__ for r in spec.match_table()]
    _write_json(os.path.join(run_dir, "config", "parameter_match.json"), match_rows)
    _write_json(os.path.join(run_dir, "config", "test_definition.json"), spec.__dict__)
    git = _git_info()
    _write_json(os.path.join(run_dir, "config", "ggui_revision.json"), git)
    log(f"GGUI {git['branch']} {git['revision'][:12]}")

    gauges_1d_full = spec.gauges_1d()
    gauges_1d_remap = [(r, lab) for r, lab in gauges_1d_full if r < spec.r_remap_m - 1e-12]
    gauges_2d = spec.gauges_2d()

    vip_full = os.path.join(run_dir, "config", "compare_full.vip")
    vip_remap = os.path.join(run_dir, "config", "compare_remap.vip")
    build_model(NO_REMAP_VIP, vip_full, gauges_1d=gauges_1d_full, gauges_2d=gauges_2d)
    build_model(REMAP_VIP, vip_remap, gauges_1d=gauges_1d_remap, gauges_2d=gauges_2d)
    ident_direct = assert_remap_identity(vip_full, remapflag=0, shape=1)
    ident_remap = assert_remap_identity(vip_remap, remapflag=1, shape=0)
    _write_json(os.path.join(run_dir, "config", "generated_direct_identity.json"), ident_direct)
    _write_json(os.path.join(run_dir, "config", "generated_remap_identity.json"), ident_remap)
    if _sha256(NO_REMAP_VIP) != no_remap_hash or _sha256(REMAP_VIP) != remap_hash:
        raise StopHarness("Source VIP templates were modified; aborting.")
    log(
        f"Generated VIP copies. DIRECT remapflag={ident_direct['remapflag']} "
        f"shape={ident_direct['shape']}; REMAP remapflag={ident_remap['remapflag']} "
        f"shape={ident_remap['shape']}."
    )

    json_1d = _write_vip_json(
        spec, os.path.join(run_dir, "config", "mods_1d.json"), spec.domain_1d_m, NO_REMAP_JSON
    )
    json_2d = _write_vip_json(
        spec, os.path.join(run_dir, "config", "mods_2d.json"), spec.domain_1d_m, NO_REMAP_JSON
    )
    json_remap = _write_vip_json(
        spec, os.path.join(run_dir, "config", "mods_1d_to_2d.json"), spec.r_remap_m, REMAP_JSON
    )
    for path, expected_shape in ((json_2d, 1), (json_remap, 0)):
        with open(path, encoding="utf-8") as handle:
            shape = json.load(handle)["params_2d"]["shape"]
        if int(shape) != expected_shape:
            raise StopHarness(f"{path} wrote shape={shape}, expected {expected_shape}")

    # --- Smoke: VIPER 1D ---
    smoke_dir = os.path.join(run_dir, "viper", "1d")
    shutil.copy2(vip_full, os.path.join(smoke_dir, "model.vip"))
    shutil.copy2(json_1d, os.path.join(smoke_dir, "mods.json"))
    log("SMOKE VIPER 1D ...")
    smoke = run_viper(
        case_dir=smoke_dir,
        vip_path=os.path.join(smoke_dir, "model.vip"),
        json_path=os.path.join(smoke_dir, "mods.json"),
        stages=["1d"],
    )
    log(f"VIPER 1D rc={smoke['returncode']} wall={smoke['wall_s']:.1f}s vprt={smoke['vprt_bytes']}B")
    try:
        require_nonempty_th(smoke, need_1d=True, need_2d=False)
    except RuntimeError as exc:
        raise StopHarness(str(exc)) from exc
    log("SMOKE OK: viper1d_th_overpressure.txt is non-empty")

    runtimes = {"VIPER_1D": smoke["wall_s"]}
    commands = {"VIPER_1D": smoke["argv"]}
    ggui_cases = {}

    # --- VIPER 2D direct ---
    v2d_dir = os.path.join(run_dir, "viper", "2d_direct")
    shutil.copy2(vip_full, os.path.join(v2d_dir, "model.vip"))
    shutil.copy2(json_2d, os.path.join(v2d_dir, "mods.json"))
    log("VIPER 2D DIRECT ...")
    v2d = run_viper(
        case_dir=v2d_dir,
        vip_path=os.path.join(v2d_dir, "model.vip"),
        json_path=os.path.join(v2d_dir, "mods.json"),
        stages=["2d"],
    )
    log(f"VIPER 2D DIRECT rc={v2d['returncode']} wall={v2d['wall_s']:.1f}s")
    try:
        require_nonempty_th(v2d, need_1d=False, need_2d=True)
    except RuntimeError as exc:
        raise StopHarness(str(exc)) from exc
    runtimes["VIPER_2D_DIRECT"] = v2d["wall_s"]
    commands["VIPER_2D_DIRECT"] = v2d["argv"]

    # --- VIPER 1D->2D ---
    v12_dir = os.path.join(run_dir, "viper", "1d_to_2d")
    shutil.copy2(vip_remap, os.path.join(v12_dir, "model.vip"))
    shutil.copy2(json_remap, os.path.join(v12_dir, "mods.json"))
    log("VIPER 1D->2D ...")
    v12 = run_viper(
        case_dir=v12_dir,
        vip_path=os.path.join(v12_dir, "model.vip"),
        json_path=os.path.join(v12_dir, "mods.json"),
        stages=["1d", "2d"],
    )
    log(f"VIPER 1D->2D rc={v12['returncode']} wall={v12['wall_s']:.1f}s")
    try:
        require_nonempty_th(v12, need_1d=True, need_2d=True)
        v12_prop = require_remap_propagation(v12_dir)
    except RuntimeError as exc:
        raise StopHarness(str(exc)) from exc
    runtimes["VIPER_1D_TO_2D"] = v12["wall_s"]
    commands["VIPER_1D_TO_2D"] = v12["argv"]
    _write_json(os.path.join(run_dir, "config", "viper_remap_propagation.json"), v12_prop)
    log(
        f"VIPER 1D->2D propagated: vtk={v12_prop['n_vtk']} "
        f"max_2d_step={v12_prop['max_2d_step']} "
        f"summary_2d={v12_prop['run_summary_has_2d']}"
    )

    # --- GGUI 1D standalone ---
    log("GGUI 1D standalone ...")
    g1 = generate_and_run(
        prefix="Case_1D_viperCmp",
        inputs=inputs_1d(
            spec,
            remap_for_2d=False,
            radius=spec.domain_1d_m,
            gauges=spec.r_gauges_1d,
        ),
        log_dir=os.path.join(run_dir, "ggui", "1d"),
    )
    log(f"GGUI 1D {g1['name']} rc={g1['returncode']} wall={g1['wall_s']:.1f}s")
    if g1["returncode"] != 0:
        raise StopHarness("GGUI 1D Allrun failed")
    runtimes["GGUI_BF_1D"] = g1["wall_s"]
    ggui_cases["GGUI_BF_1D"] = g1

    # --- GGUI 1D remap precursor ---
    log("GGUI 1D remap precursor ...")
    g1r = generate_and_run(
        prefix="Case_1D_viperRemap",
        inputs=inputs_1d(
            spec,
            remap_for_2d=True,
            radius=spec.r_remap_m,
            gauges=[r for r in spec.r_gauges_1d if r < spec.r_remap_m],
        ),
        log_dir=os.path.join(run_dir, "ggui", "1d_to_2d"),
    )
    log(f"GGUI 1D remap {g1r['name']} rc={g1r['returncode']} wall={g1r['wall_s']:.1f}s")
    if g1r["returncode"] != 0:
        raise StopHarness("GGUI 1D remap precursor Allrun failed")
    runtimes["GGUI_BF_1D_REMAP_SOURCE"] = g1r["wall_s"]
    ggui_cases["GGUI_BF_1D_REMAP_SOURCE"] = g1r

    # --- GGUI 2D direct with cost awareness ---
    log("GGUI 2D DIRECT generate+run (40k cells, 4 cores). If impractical, harness stops.")
    g2 = generate_and_run(
        prefix="Case_2D_viperDirect",
        inputs=inputs_2d(spec, remapped=False),
        log_dir=os.path.join(run_dir, "ggui", "2d_direct"),
    )
    log(f"GGUI 2D DIRECT {g2['name']} rc={g2['returncode']} wall={g2['wall_s']:.1f}s")
    if g2["wall_s"] > COST_LIMIT_S:
        raise StopHarness(
            f"Direct 2D blastFoam wall time {g2['wall_s']:.0f}s exceeds "
            f"{COST_LIMIT_S}s. Resolution was not changed."
        )
    if g2["returncode"] != 0:
        raise StopHarness("GGUI 2D direct Allrun failed")
    runtimes["GGUI_BF_2D_DIRECT"] = g2["wall_s"]
    ggui_cases["GGUI_BF_2D_DIRECT"] = g2

    log("GGUI 1D->2D remap target ...")
    g2r = generate_and_run(
        prefix="Case_2D_viperRemap",
        inputs=inputs_2d(spec, remapped=True, source_1d=g1r["case_dir"]),
        log_dir=os.path.join(run_dir, "ggui", "1d_to_2d"),
    )
    log(f"GGUI 2D REMAP {g2r['name']} rc={g2r['returncode']} wall={g2r['wall_s']:.1f}s")
    if g2r["returncode"] != 0:
        raise StopHarness("GGUI 2D remap Allrun failed")
    runtimes["GGUI_BF_1D_TO_2D"] = g2r["wall_s"]
    ggui_cases["GGUI_BF_1D_TO_2D"] = g2r

    receive_ggui = copied_1d2d_radius_m(
        target_2d_case=g2r["case_dir"],
        source_1d_case=g1r["case_dir"],
        widget_mapped_radius=spec.r_remap_m,
    )
    log(
        f"GGUI remap: R_remap requested={spec.r_remap_m} "
        f"R_handoff={spec.r_handoff_ggui_m:.4f} "
        f"actual copied={receive_ggui} "
        f"rule={HANDOFF_RULE} buffer={REMAP_FRONT_BUFFER_CELLS_1D} "
        f"criterion={HANDOFF_CRITERION}"
    )
    provided_remap_ref = remap_propagation_report(
        r"C:\VIPER_COMPARE\templates\12_pair\2"
    )
    _write_json(
        os.path.join(run_dir, "config", "provided_remap_initialized_only.json"),
        provided_remap_ref,
    )
    log(
        "VIPER remap identity: generated Remap.vip keeps remapflag=1, shape=0. "
        f"Provided 12.zip Remap snapshot is initialized-only "
        f"(vtk={provided_remap_ref['n_vtk']}, max_2d_step={provided_remap_ref['max_2d_step']}). "
        f"Case C 1D domain_radius_od={spec.r_remap_m} m."
    )

    # Extract
    all_rows = []
    p1, i1, _, _ = _load_viper_case(smoke_dir, True, False)
    all_rows.extend(
        _histories_to_rows(
            solver="VIPER",
            configuration="1d",
            dimension="1d",
            remapped=False,
            radii=list(spec.r_gauges_1d),
            labels=[g[1] for g in gauges_1d_full],
            times=p1[0],
            pressure=p1[1],
            impulse=None if i1 is None else i1[1],
            spec=spec,
            receive_r_max=None,
            source_case=smoke_dir,
            source_file=os.path.join(smoke_dir, "viper1d_th_overpressure.txt"),
        )
    )
    _, _, p2, i2 = _load_viper_case(v2d_dir, False, True)
    all_rows.extend(
        _histories_to_rows(
            solver="VIPER",
            configuration="2d_direct",
            dimension="2d",
            remapped=False,
            radii=list(spec.r_gauges_2d),
            labels=[g[2] for g in gauges_2d],
            times=p2[0],
            pressure=p2[1],
            impulse=None if i2 is None else i2[1],
            spec=spec,
            receive_r_max=None,
            source_case=v2d_dir,
            source_file=os.path.join(v2d_dir, "viper2d_th_overpressure.txt"),
        )
    )
    _, _, p12, i12 = _load_viper_case(v12_dir, False, True)
    all_rows.extend(
        _histories_to_rows(
            solver="VIPER",
            configuration="2d_remap",
            dimension="2d",
            remapped=True,
            radii=list(spec.r_gauges_2d),
            labels=[g[2] for g in gauges_2d],
            times=p12[0],
            pressure=p12[1],
            impulse=None if i12 is None else i12[1],
            spec=spec,
            receive_r_max=spec.r_remap_m,
            source_case=v12_dir,
            source_file=os.path.join(v12_dir, "viper2d_th_overpressure.txt"),
        )
    )

    def add_ggui(case, dim, configuration, remapped, radii, labels, receive):
        locs, t, p = load_ggui_probe(case["case_dir"], dim, "p")
        _, _, imp = load_ggui_probe(case["case_dir"], dim, "impulse")
        if t.size == 0:
            raise StopHarness(f"Empty GGUI {dim} pressure probes in {case['case_dir']}")
        n = min(len(radii), p.shape[1])
        all_rows.extend(
            _histories_to_rows(
                solver="GGUI_BF",
                configuration=configuration,
                dimension=dim,
                remapped=remapped,
                radii=list(radii)[:n],
                labels=list(labels)[:n],
                times=t,
                pressure=p,
                impulse=None if imp.size == 0 else imp,
                spec=spec,
                receive_r_max=receive,
                source_case=case["case_dir"],
                source_file=case["case_dir"],
            )
        )
        return t, p

    add_ggui(
        g1, "1d", "1d", False, spec.r_gauges_1d, [g[1] for g in gauges_1d_full], None
    )
    t_g2, p_g2 = add_ggui(
        g2, "2d", "2d_direct", False, spec.r_gauges_2d, [g[2] for g in gauges_2d], None
    )
    add_ggui(
        g2r,
        "2d",
        "2d_remap",
        True,
        spec.r_gauges_2d,
        [g[2] for g in gauges_2d],
        receive_ggui,
    )

    extracted = os.path.join(run_dir, "extracted", "comparison_table.json")
    _write_json(extracted, all_rows)
    csv_path = os.path.join(run_dir, "extracted", "comparison_table.csv")
    fields = [
        "solver",
        "configuration",
        "dimension",
        "remapped",
        "gauge_label",
        "R_m",
        "Z",
        "inside_remap",
        "independent_2d",
        "peak_pressure_pa",
        "native_impulse_pa_s",
        "derived_impulse_pa_s",
        "peak_time_s",
        "arrival_time_s_if_available",
        "kb_peak_pressure_pa",
        "kb_impulse_pa_s",
        "source_case",
        "source_file",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    def pick(solver, configuration):
        return [r for r in all_rows if r["solver"] == solver and r["configuration"] == configuration]

    def errors(a, b, pair):
        by_r = {round(r["R_m"], 6): r for r in b}
        out = []
        for ra in a:
            rb = by_r.get(round(ra["R_m"], 6))
            if rb is None:
                continue
            if ra.get("remapped") and ra.get("dimension") == "2d":
                if not ra.get("independent_2d") or not rb.get("independent_2d"):
                    continue
            item = pair_error(ra, rb)
            item["pair"] = pair
            out.append(item)
        return out

    err_1d = errors(pick("GGUI_BF", "1d"), pick("VIPER", "1d"), "1D GGUI vs VIPER")
    err_2d = errors(
        pick("GGUI_BF", "2d_direct"), pick("VIPER", "2d_direct"), "2D direct GGUI vs VIPER"
    )
    err_rm = errors(
        pick("GGUI_BF", "2d_remap"), pick("VIPER", "2d_remap"), "2D remap GGUI vs VIPER"
    )
    _write_json(os.path.join(run_dir, "extracted", "errors.json"), err_1d + err_2d + err_rm)

    # Waveform overlays at R=0.40, 0.70, 1.00
    overlays = []
    v1p = parse_viper_th(os.path.join(smoke_dir, "viper1d_th_overpressure.txt"))
    g1_locs, g1_t, g1_p = load_ggui_probe(g1["case_dir"], "1d", "p")
    for r in (0.40, 0.70, 1.00):
        series = []
        if r in spec.r_gauges_1d:
            i = list(spec.r_gauges_1d).index(r)
            series.append(
                {
                    "R_m": r,
                    "label": "VIPER 1D",
                    "t": v1p[1],
                    "p": v1p[2][:, i] if v1p[2].shape[1] > i else v1p[2][:, 0],
                }
            )
            if g1_p.size and g1_p.shape[1] > i:
                series.append({"R_m": r, "label": "GGUI 1D", "t": g1_t, "p": g1_p[:, i]})
        if r in spec.r_gauges_2d:
            j = list(spec.r_gauges_2d).index(r)
            vp = parse_viper_th(os.path.join(v2d_dir, "viper2d_th_overpressure.txt"))
            series.append({"R_m": r, "label": "VIPER 2D direct", "t": vp[1], "p": vp[2][:, j]})
            g2_locs, g2t, g2p = load_ggui_probe(g2["case_dir"], "2d", "p")
            if g2p.size and g2p.shape[1] > j:
                series.append({"R_m": r, "label": "GGUI 2D direct", "t": g2t, "p": g2p[:, j]})
            vr = parse_viper_th(os.path.join(v12_dir, "viper2d_th_overpressure.txt"))
            series.append({"R_m": r, "label": "VIPER 2D remap", "t": vr[1], "p": vr[2][:, j]})
            gr_locs, grt, grp = load_ggui_probe(g2r["case_dir"], "2d", "p")
            if grp.size and grp.shape[1] > j:
                series.append({"R_m": r, "label": "GGUI 2D remap", "t": grt, "p": grp[:, j]})
        overlays.extend(series)
    plot_pt_overlays(overlays, os.path.join(run_dir, "plots"), spec.p_atm)

    vis_rows = []
    for row in all_rows:
        if row["dimension"] == "2d" and row["remapped"] and not row.get("independent_2d"):
            continue
        vis_rows.append(row)
    plot_peak_vs_r(
        vis_rows,
        os.path.join(run_dir, "plots", "peak_vs_R.png"),
        x_key="R_m",
        xlabel="R [m] (actual physical location)",
    )
    plot_peak_vs_r(
        vis_rows,
        os.path.join(run_dir, "plots", "peak_vs_Z.png"),
        x_key="Z",
        xlabel="Z = R / W^(1/3) [m/kg^(1/3)]",
    )
    plot_impulse_vs_r(vis_rows, os.path.join(run_dir, "plots", "impulse_vs_R.png"))
    plot_error_vs_r(
        err_1d + err_2d + err_rm, os.path.join(run_dir, "plots", "error_vs_R.png")
    )

    report_md = os.path.join(run_dir, "report", "report.md")
    with open(report_md, "w", encoding="utf-8") as handle:
        handle.write("# VIPER vs GGUI/blastFoam free-air comparison\n\n")
        handle.write(f"Run directory: `{run_dir}`\n\n")
        handle.write("## Physical test\n\n")
        handle.write(spec.cost_report() + "\n\n")
        handle.write("| parameter | VIPER | GGUI/BF | exact | reason |\n|---|---|---|---|---|\n")
        for row in spec.match_table():
            handle.write(
                f"| {row.parameter} | {row.viper} | {row.ggui} | {row.exact} | {row.reason} |\n"
            )
        handle.write("\n## Executables\n\n")
        handle.write(f"- VIPER: `{VIPER_EXE}`\n")
        handle.write(f"- GGUI branch `{git['branch']}` revision `{git['revision']}`\n")
        handle.write("\n## VIPER HDF5 gauge schema\n\n```\n" + schema_report() + "\n```\n")
        handle.write(
            "\nVIPER remap identity (authoritative 12.zip pair): "
            "direct remapflag=0,shape=1; remap remapflag=1,shape=0. "
            "twodremapoption=1 is not a discriminator.\n"
        )
        handle.write("\n## Gauges\n\n")
        handle.write(f"1D R = {list(spec.r_gauges_1d)}\n\n")
        handle.write(f"2D R = {list(spec.r_gauges_2d)} at z = HOB = {spec.hob_m} m\n\n")
        handle.write("\n## Commands\n\n")
        for name, argv in commands.items():
            handle.write(f"- {name}: `{argv}`\n")
        handle.write("\n## GGUI cases\n\n")
        for name, case in ggui_cases.items():
            handle.write(f"- {name}: `{case['name']}` `{case['case_dir']}` wall={case['wall_s']:.1f}s\n")
        handle.write("\n## Runtimes\n\n")
        for k, v in runtimes.items():
            handle.write(f"- {k}: {v:.1f} s\n")
        handle.write("\n## Remap exclusion\n\n")
        handle.write(
            f"GGUI: requested R_remap={spec.r_remap_m} m, "
            f"R_handoff={spec.r_handoff_ggui_m:.4f} m "
            f"({REMAP_FRONT_BUFFER_CELLS_1D} cells), actual copied={receive_ggui} m.\n"
            "Independent 2D gauges require R > copied + dx_2D.\n"
            "VIPER Case C uses Remap.vip (remapflag=1, shape=0), not a TEST2 copy.\n"
        )
        handle.write("\n## Peak table (independent gauges only)\n\n")
        handle.write("| solver | config | R | peak Pa | derived I | t_peak | remap |\n|---|---|---|---|---|---|---|\n")
        for row in remap_eligible_rows(all_rows):
            handle.write(
                f"| {row['solver']} | {row['configuration']} | {row['R_m']:.2f} | "
                f"{row['peak_pressure_pa']:.4g} | {row['derived_impulse_pa_s']:.4g} | "
                f"{row['peak_time_s']:.4g} | {row['inside_remap']} |\n"
            )
        handle.write("\n## Cross-solver peak errors\n\n")
        for item in err_1d + err_2d + err_rm:
            handle.write(
                f"- {item['pair']} R={item['R_m']:.2f}: "
                f"{item['peak_pressure_error_pct']}%\n"
            )
        handle.write(
            "\nDo not interpret every difference as a GGUI defect. "
            "VIPER Method 1 / jwlflag=0 is not blastFoam JWL detonation.\n"
        )
        handle.write(
            f"\nSource No_Remap.vip sha256 still {no_remap_hash}; "
            f"Remap.vip sha256 still {remap_hash}\n"
        )

    _write_json(
        os.path.join(run_dir, "report", "summary.json"),
        {
            "run_dir": run_dir,
            "runtimes": runtimes,
            "ggui_cases": ggui_cases,
            "commands": commands,
            "receive_ggui": receive_ggui,
            "no_remap_vip_sha256": no_remap_hash,
            "remap_vip_sha256": remap_hash,
            "git": git,
        },
    )
    log(f"Report written {report_md}")
    log(f"DONE run_dir={run_dir}")
    print(run_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StopHarness as exc:
        print("STOP:", exc, file=sys.stderr)
        raise SystemExit(2)
