"""Dedicated 1D remap snapshot: capture, validation, 2D/3D source selection."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from axisymmetric_2d import REMAP_SOURCE, validate_mapping_source
from completion_1d import (
    RUN_MODE_REFLECT,
    RUN_MODE_TERMINATE,
    STOP_REASON_END_TIME_REACHED,
    STOP_REASON_WAVE_RADIUS_REACHED,
    CompletionRecord,
    reset_completion_for_new_run,
    write_completion_record,
)
from generator_2d import Generator2D
from generator_3d import Generator3D
from models_2d import CaseInputs2D, MappingSource2D
from remap_fields_2d import map_fields_to_2d_cells, source_radius_rz
from remap_snapshot_1d import (
    SCHEMA_VERSION,
    SNAPSHOT_JSON,
    SNAPSHOT_NPZ,
    SOURCE_OPENFOAM,
    SOURCE_SNAPSHOT,
    availability_for_case,
    canonical_case_path,
    load_profile_for_remap,
    read_snapshot_arrays,
    read_snapshot_metadata,
    resolve_remap_source,
    same_source_case,
    snapshot_exists,
    validate_snapshot,
    write_snapshot,
    write_snapshot_after_run,
)

FINAL_T = 0.000639296
WRITE_INTERVAL = 0.001


def _write_scalar(path: str, name: str, values) -> None:
    vals = [float(v) for v in values]
    inner = "\n".join(f"{v:.10e}" for v in vals)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"FoamFile {{ version 2.0; format ascii; class volScalarField; object {name}; }}\n"
            "dimensions [0 0 0 0 0 0 0];\n"
            f"internalField nonuniform List<scalar>\n{len(vals)}\n(\n{inner}\n);\n"
            "boundaryField {}\n"
        )


def _write_vector(path: str, name: str, values) -> None:
    rows = [float(v) for v in values]
    inner = "\n".join(f"({v:.10e} 0 0)" for v in rows)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"FoamFile {{ version 2.0; format ascii; class volVectorField; object {name}; }}\n"
            "dimensions [0 1 -1 0 0 0 0];\n"
            f"internalField nonuniform List<vector>\n{len(rows)}\n(\n{inner}\n);\n"
            "boundaryField {}\n"
        )


def _write_time_dir(case_dir: str, label: str, n: int = 8, p_peak: float = 5.0e6) -> None:
    tdir = os.path.join(case_dir, label)
    os.makedirs(tdir, exist_ok=True)
    r = np.linspace(0.01, 1.0, n)
    p = 101325.0 + (p_peak - 101325.0) * np.exp(-(r / 0.15) ** 2)
    t = np.full(n, 350.0)
    u = np.linspace(0.0, 400.0, n)
    rho_air = np.full(n, 1.2)
    rho_c4 = np.linspace(1600.0, 0.0, n)
    alpha = np.linspace(1.0, 0.0, n)
    _write_scalar(os.path.join(tdir, "p"), "p", p)
    _write_scalar(os.path.join(tdir, "T"), "T", t)
    _write_vector(os.path.join(tdir, "U"), "U", u)
    _write_scalar(os.path.join(tdir, "rho.air"), "rho.air", rho_air)
    _write_scalar(os.path.join(tdir, "rho.c4"), "rho.c4", rho_c4)
    _write_scalar(os.path.join(tdir, "alpha.c4"), "alpha.c4", alpha)


def _write_control(case_dir: str, write_interval: float = WRITE_INTERVAL) -> None:
    sys_dir = os.path.join(case_dir, "system")
    os.makedirs(sys_dir, exist_ok=True)
    with open(os.path.join(sys_dir, "controlDict"), "w", encoding="utf-8") as handle:
        handle.write(
            "endTime         1;\n"
            f"writeInterval   {write_interval};\n"
        )
    with open(os.path.join(sys_dir, "blockMeshDict"), "w", encoding="utf-8") as handle:
        handle.write(
            "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n"
            "boundary\n(\n"
            "    wedge0 { type wedge; faces ((0 1 2 3)); }\n"
            "    wedge1 { type wedge; faces ((0 4 5 3)); }\n"
            ");\n"
        )
    const = os.path.join(case_dir, "constant")
    os.makedirs(const, exist_ok=True)
    with open(os.path.join(const, "phaseProperties"), "w", encoding="utf-8") as handle:
        handle.write("phases (c4 air);\nequationOfState JWL;\nrho0 1630;\n")


def _arrived_completion(case_dir: str, final_t: float = FINAL_T) -> CompletionRecord:
    record = CompletionRecord(
        mode=RUN_MODE_TERMINATE,
        stop_mode=RUN_MODE_TERMINATE,
        stop_reason=STOP_REASON_WAVE_RADIUS_REACHED,
        wave_radius_reached=True,
        final_solver_time_s=final_t,
        requested_stop_radius_m=0.8,
        detected_arrival_time_s=final_t,
        criterion="overpressure_above_ambient",
        p_atm=101325.0,
    )
    write_completion_record(case_dir, record)
    return record


def _mapping_inputs(case_dir: str) -> CaseInputs2D:
    from dataclasses import replace

    return replace(
        CaseInputs2D(),
        initialization_source=REMAP_SOURCE,
        mapping=MappingSource2D(
            case_path=case_dir,
            time_mode="latest",
            mapped_radius=0.5,
            source_resolution=0.01,
        ),
    )


class RemapSnapshot1DTests(unittest.TestCase):
    def test_windows_unc_and_linux_paths_are_the_same_1d_case(self):
        linux = "/home/naor/OpenFOAM/naor-9/run/Work/Case_1D_20260904_184200"
        unc = (
            r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM"
            r"\naor-9\run\Work\Case_1D_20260904_184200"
        )
        lowered = (
            r"\\wsl.localhost\ubuntu-20.04\home\naor\openfoam"
            r"\naor-9\run\work\case_1d_20260904_184200"
        )
        self.assertTrue(same_source_case(linux, unc))
        self.assertTrue(same_source_case(linux, lowered))
        from remap_snapshot_1d import identity_fingerprint

        kwargs = dict(
            physical_time=FINAL_T,
            stop_reason="wave_radius_reached",
            mode="terminate",
            wave_radius_reached=True,
        )
        self.assertEqual(
            identity_fingerprint(linux, **kwargs),
            identity_fingerprint(unc, **kwargs),
        )
        self.assertEqual(
            identity_fingerprint(linux, **kwargs),
            identity_fingerprint(lowered, **kwargs),
        )

    def test_terminate_before_write_interval_writes_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _write_time_dir(td, f"{FINAL_T:.9f}".rstrip("0").rstrip(".") or f"{FINAL_T:g}")
            # Use the actual directory name OpenFOAM would write for writeNow.
            label = next(name for name in os.listdir(td) if name not in {"system", "constant"})
            self.assertAlmostEqual(float(label), FINAL_T, places=9)
            self.assertFalse(os.path.isdir(os.path.join(td, f"{WRITE_INTERVAL:g}")))
            record = _arrived_completion(td, float(label))
            status = write_snapshot_after_run(td, record)
            self.assertIn("Remap snapshot written", status)
            self.assertTrue(snapshot_exists(td))
            ok, message = validate_snapshot(td)
            self.assertTrue(ok, message)
            meta = read_snapshot_metadata(td)
            self.assertEqual(meta["schema_version"], SCHEMA_VERSION)
            self.assertEqual(meta["source_dimension"], "1D")
            self.assertEqual(meta["remap_source_type"], SOURCE_SNAPSHOT)
            self.assertAlmostEqual(meta["source_physical_time"], float(label))
            self.assertTrue(meta["wave_radius_reached"])
            self.assertEqual(meta["completion_mode"], RUN_MODE_TERMINATE)

    def test_poly_mesh_cell_radii_match_owner_cells_not_a_linspace(self):
        from remap_snapshot_1d import capture_arrays_from_time_dir, cell_radii_from_poly_mesh

        with tempfile.TemporaryDirectory() as td:
            mesh = os.path.join(td, "constant", "polyMesh")
            os.makedirs(mesh)
            with open(os.path.join(mesh, "points"), "w", encoding="utf-8") as handle:
                handle.write(
                    "8\n(\n"
                    "(0.10 0 0)\n(0.10 0.01 0)\n(0.10 0.01 0.01)\n(0.10 0 0.01)\n"
                    "(1.50 0 0)\n(1.50 0.01 0)\n(1.50 0.01 0.01)\n(1.50 0 0.01)\n"
                    ")\n"
                )
            with open(os.path.join(mesh, "faces"), "w", encoding="utf-8") as handle:
                handle.write("2\n(\n4(4 5 6 7)\n4(0 1 2 3)\n)\n")
            with open(os.path.join(mesh, "owner"), "w", encoding="utf-8") as handle:
                handle.write("2\n(\n0\n1\n)\n")
            radii = cell_radii_from_poly_mesh(td, 2)
            self.assertIsNotNone(radii)
            self.assertGreater(float(radii[0]), 1.49)
            self.assertLess(float(radii[1]), 0.12)
            self.assertGreater(float(radii[0]), float(radii[1]))
            _write_time_dir(td, "0.001", n=2, p_peak=2.0e6)
            arrays = capture_arrays_from_time_dir(td, "0.001")
            self.assertIsNotNone(arrays)
            self.assertGreater(float(arrays["r"][0]), 1.49)
            self.assertLess(float(arrays["r"][1]), 0.12)

    def test_2d_remap_from_snapshot_without_final_time_directory(self):
        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _write_time_dir(td, f"{FINAL_T:g}")
            record = _arrived_completion(td)
            write_snapshot_after_run(td, record)
            label = next(
                name
                for name in os.listdir(td)
                if os.path.isdir(os.path.join(td, name)) and name not in {"system", "constant"}
            )
            shutil.rmtree(os.path.join(td, label))
            self.assertFalse(os.path.isdir(os.path.join(td, label)))
            profile, error = load_profile_for_remap(td)
            self.assertIsNone(error)
            self.assertIsNotNone(profile)
            resolved = resolve_remap_source(td)
            self.assertTrue(resolved.ok)
            self.assertFalse(resolved.blocked)
            self.assertEqual(resolved.source_type, SOURCE_SNAPSHOT)
            hob = 1.25
            r_2d = np.array([0.0, 0.0, 0.4])
            z_2d = np.array([0.0, hob, hob])
            mapped = map_fields_to_2d_cells(
                r_2d,
                z_2d,
                hob,
                profile["r"],
                profile,
            )
            self.assertGreater(mapped["p"][1], mapped["p"][0])
            self.assertAlmostEqual(
                float(source_radius_rz(0.0, hob, hob)),
                0.0,
            )
            report = validate_mapping_source(_mapping_inputs(td))
            self.assertTrue(report.valid, report.errors)
            self.assertEqual(report.remap_source_type, SOURCE_SNAPSHOT)
            self.assertAlmostEqual(report.source_physical_time, FINAL_T)
            payload = report.to_dict()
            self.assertEqual(payload["remap_source_type"], SOURCE_SNAPSHOT)

    def test_snapshot_validation_rejects_stale_mismatched_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _write_time_dir(td, f"{FINAL_T:g}")
            _arrived_completion(td)
            write_snapshot(
                td,
                {
                    "r": np.linspace(0.0, 1.0, 6),
                    "p": np.full(6, 2.0e5),
                    "T": np.full(6, 300.0),
                    "U_mag": np.zeros(6),
                },
                physical_time=FINAL_T,
            )
            ok, _ = validate_snapshot(td)
            self.assertTrue(ok)

            meta_path = os.path.join(td, SNAPSHOT_JSON)
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            meta["schema_version"] = 99
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle)
            ok, message = validate_snapshot(td)
            self.assertFalse(ok)
            self.assertIn("schema", message.lower())
            meta["schema_version"] = SCHEMA_VERSION
            meta["source_case_path"] = os.path.abspath(os.path.join(td, "other"))
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle)
            ok, message = validate_snapshot(td)
            self.assertFalse(ok)
            self.assertIn("different source case", message.lower())

        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _arrived_completion(td)
            write_snapshot(
                td,
                {
                    "r": np.linspace(0.0, 1.0, 6),
                    "p": np.full(6, 2.0e5),
                    "T": np.full(6, 300.0),
                    "U_mag": np.zeros(6),
                },
                physical_time=FINAL_T,
            )
            linux = "/home/naor/OpenFOAM/naor-9/run/Work/Case_1D_20260904_184200"
            unc = (
                r"\\wsl.localhost\ubuntu-20.04\home\naor\openfoam"
                r"\naor-9\run\work\case_1d_20260904_184200"
            )
            self.assertTrue(same_source_case(linux, unc))
            meta_path = os.path.join(td, SNAPSHOT_JSON)
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            meta["source_case_path"] = unc
            meta["source_case_path_canonical"] = linux
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle)
            arrays = read_snapshot_arrays(td)
            ok, message = validate_snapshot(linux, metadata=meta, arrays=arrays)
            self.assertTrue(ok, message)

        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _arrived_completion(td)
            write_snapshot(
                td,
                {
                    "r": np.linspace(0.0, 1.0, 6),
                    "p": np.full(6, 2.0e5),
                    "T": np.full(6, 300.0),
                    "U_mag": np.zeros(6),
                },
                physical_time=FINAL_T,
            )
            arrays = read_snapshot_arrays(td)
            del arrays["T"]
            np.savez_compressed(os.path.join(td, SNAPSHOT_NPZ), **arrays)
            ok, message = validate_snapshot(td)
            self.assertFalse(ok)
            self.assertIn("missing required field", message.lower())

        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _arrived_completion(td)
            write_snapshot(
                td,
                {
                    "r": np.linspace(0.0, 1.0, 6),
                    "p": np.full(6, 2.0e5),
                    "T": np.full(6, 300.0),
                    "U_mag": np.zeros(6),
                },
                physical_time=FINAL_T,
            )
            arrays = read_snapshot_arrays(td)
            arrays["p"] = np.asarray(arrays["p"]) + 1.0
            np.savez_compressed(os.path.join(td, SNAPSHOT_NPZ), **arrays)
            ok, message = validate_snapshot(td)
            self.assertFalse(ok)
            self.assertIn("checksum", message.lower())

        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _arrived_completion(td)
            write_snapshot(
                td,
                {
                    "r": np.linspace(0.0, 1.0, 6),
                    "p": np.full(6, 2.0e5),
                    "T": np.full(6, 300.0),
                    "U_mag": np.zeros(6),
                },
                physical_time=FINAL_T,
            )
            write_completion_record(
                td,
                CompletionRecord(
                    mode=RUN_MODE_TERMINATE,
                    stop_reason=STOP_REASON_END_TIME_REACHED,
                    wave_radius_reached=False,
                    final_solver_time_s=0.05,
                ),
            )
            ok, message = validate_snapshot(td)
            self.assertFalse(ok)
            self.assertIn("stale", message.lower())

    def test_invalid_snapshot_blocks_and_does_not_fall_back_to_openfoam(self):
        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _write_time_dir(td, f"{FINAL_T:g}")
            record = _arrived_completion(td)
            write_snapshot_after_run(td, record)
            _write_time_dir(td, f"{WRITE_INTERVAL:g}", p_peak=9.0e7)
            write_completion_record(
                td,
                CompletionRecord(
                    mode=RUN_MODE_TERMINATE,
                    stop_reason=STOP_REASON_WAVE_RADIUS_REACHED,
                    wave_radius_reached=True,
                    final_solver_time_s=WRITE_INTERVAL,
                ),
            )
            resolved = resolve_remap_source(td)
            self.assertTrue(resolved.blocked)
            self.assertFalse(resolved.ok)
            self.assertEqual(resolved.source_type, SOURCE_SNAPSHOT)
            profile, error = load_profile_for_remap(td)
            self.assertIsNone(profile)
            self.assertIsNotNone(error)
            report = validate_mapping_source(_mapping_inputs(td))
            self.assertFalse(report.valid)
            self.assertTrue(any("stale" in err.lower() or "match" in err.lower() for err in report.errors))

    def test_openfoam_fallback_only_when_no_snapshot_and_time_matches(self):
        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _write_time_dir(td, f"{WRITE_INTERVAL:g}")
            write_completion_record(
                td,
                CompletionRecord(
                    mode=RUN_MODE_REFLECT,
                    stop_reason=STOP_REASON_END_TIME_REACHED,
                    wave_radius_reached=False,
                    final_solver_time_s=WRITE_INTERVAL,
                ),
            )
            self.assertFalse(snapshot_exists(td))
            resolved = resolve_remap_source(td)
            self.assertTrue(resolved.ok)
            self.assertFalse(resolved.blocked)
            self.assertEqual(resolved.source_type, SOURCE_OPENFOAM)
            self.assertAlmostEqual(resolved.physical_time, WRITE_INTERVAL)

        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _write_time_dir(td, f"{WRITE_INTERVAL:g}")
            _arrived_completion(td, FINAL_T)
            self.assertFalse(snapshot_exists(td))
            resolved = resolve_remap_source(td)
            self.assertFalse(resolved.ok)
            self.assertFalse(resolved.blocked)
            self.assertIn("stale", resolved.message.lower())
            avail = availability_for_case(td)
            self.assertEqual(avail.status, "stale")
            self.assertFalse(avail.snapshot_available)

    def test_2d_hob_centred_remap_from_snapshot_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            n = 21
            r = np.linspace(0.0, 1.0, n)
            p = 1.01325e5 + (1.0e7 - 1.01325e5) * np.exp(-(r / 0.05) ** 2)
            write_snapshot(
                td,
                {
                    "r": r,
                    "p": p,
                    "T": np.full(n, 300.0),
                    "U_mag": np.zeros(n),
                },
                physical_time=FINAL_T,
            )
            profile, error = load_profile_for_remap(td)
            self.assertIsNone(error)
            hob = 1.25
            r_2d = np.array([0.0, 0.0, 0.4, 0.0])
            z_2d = np.array([0.0, hob, hob, hob + 0.4])
            mapped = map_fields_to_2d_cells(r_2d, z_2d, hob, profile["r"], profile)
            self.assertGreater(mapped["p"][1], mapped["p"][0])
            self.assertGreater(mapped["p"][1], mapped["p"][2])
            self.assertGreater(mapped["p"][1], mapped["p"][3])
            self.assertAlmostEqual(mapped["p"][2], mapped["p"][3], places=6)

    def test_3d_source_selection_prefers_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _write_time_dir(td, f"{FINAL_T:g}")
            record = _arrived_completion(td)
            write_snapshot_after_run(td, record)
            shutil.rmtree(
                os.path.join(
                    td,
                    next(
                        name
                        for name in os.listdir(td)
                        if os.path.isdir(os.path.join(td, name))
                        and name not in {"system", "constant"}
                    ),
                )
            )
            resolved = resolve_remap_source(td)
            self.assertTrue(resolved.ok)
            self.assertEqual(resolved.source_type, SOURCE_SNAPSHOT)
            case3d = os.path.join(td, "case3d")
            os.makedirs(case3d)
            Generator3D(td)._write_remap_radial_script(
                case3d, "/mnt/c/source", resolved.time_label, (0.0, 1.2, 0.0)
            )
            with open(os.path.join(case3d, "remap_radial.py"), encoding="utf-8") as handle:
                script = handle.read()
            self.assertIn("load_profile_for_remap", script)
            self.assertIn("using dedicated 1D remap snapshot", script)
            self.assertNotIn('SOURCE_TIME = "snapshot"', script)
            self.assertTrue(os.path.isfile(os.path.join(case3d, "remap_snapshot_1d.py")))
            self.assertTrue(os.path.isfile(os.path.join(case3d, "remap_fields_2d.py")))
            self.assertIn("carry_mixture_mass_in_air", script)
            self.assertNotIn("a4_3d = np.zeros(n_cells)", script)
            self.assertIn(resolved.time_label, script)

    def test_2d_metadata_records_source_type_and_physical_time(self):
        with tempfile.TemporaryDirectory() as td:
            source = os.path.join(td, "src1d")
            os.makedirs(source)
            _write_control(source)
            _write_time_dir(source, f"{FINAL_T:g}")
            record = _arrived_completion(source)
            write_snapshot_after_run(source, record)
            shutil.rmtree(
                os.path.join(
                    source,
                    next(
                        name
                        for name in os.listdir(source)
                        if os.path.isdir(os.path.join(source, name))
                        and name not in {"system", "constant"}
                    ),
                )
            )
            from dataclasses import replace

            inputs = replace(
                CaseInputs2D(),
                initialization_source=REMAP_SOURCE,
                mapping=MappingSource2D(
                    case_path=source,
                    time_mode="latest",
                    mapped_radius=0.5,
                ),
                height_of_burst=1.25,
            )
            case = Generator2D(td).generate("mapped_snap", inputs)
            with open(os.path.join(case, "case_2d.json"), encoding="utf-8") as handle:
                meta = json.loads(handle.read())
            self.assertEqual(meta["remap_region"]["remap_source_type"], SOURCE_SNAPSHOT)
            self.assertAlmostEqual(meta["remap_region"]["source_physical_time"], FINAL_T)
            self.assertAlmostEqual(meta["remap_timing"]["source_physical_time"], FINAL_T)
            self.assertEqual(meta["remap_region"]["center"], [0.0, 1.25, 0.0])
            self.assertTrue(os.path.isfile(os.path.join(case, "remap_snapshot_1d.py")))

    def test_no_openfoam_time_directory_required_after_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            _write_time_dir(td, f"{FINAL_T:g}")
            record = _arrived_completion(td)
            write_snapshot_after_run(td, record)
            for name in list(os.listdir(td)):
                path = os.path.join(td, name)
                if os.path.isdir(path) and name not in {"system", "constant"}:
                    try:
                        float(name)
                    except ValueError:
                        continue
                    shutil.rmtree(path)
            self.assertTrue(snapshot_exists(td))
            self.assertTrue(os.path.isfile(os.path.join(td, SNAPSHOT_NPZ)))
            self.assertTrue(os.path.isfile(os.path.join(td, SNAPSHOT_JSON)))
            resolved = resolve_remap_source(td)
            self.assertTrue(resolved.ok)
            self.assertEqual(resolved.source_type, SOURCE_SNAPSHOT)
            self.assertIsNone(
                next(
                    (
                        name
                        for name in os.listdir(td)
                        if os.path.isdir(os.path.join(td, name))
                        and name not in {"system", "constant"}
                    ),
                    None,
                )
            )

    def test_new_run_invalidates_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            write_snapshot(
                td,
                {
                    "r": np.linspace(0.0, 1.0, 5),
                    "p": np.full(5, 1.0e5),
                    "T": np.full(5, 300.0),
                    "U_mag": np.zeros(5),
                },
                physical_time=FINAL_T,
            )
            self.assertTrue(snapshot_exists(td))
            reset_completion_for_new_run(td)
            self.assertFalse(snapshot_exists(td))

    def test_user_stop_writes_snapshot_only_when_dump_matches(self):
        with tempfile.TemporaryDirectory() as td:
            _write_control(td)
            record = CompletionRecord(
                mode=RUN_MODE_TERMINATE,
                stop_reason="user_stopped",
                final_solver_time_s=FINAL_T,
            )
            write_completion_record(td, record)
            status = write_snapshot_after_run(td, record, user_stopped=True)
            self.assertIn("not written", status.lower())
            self.assertFalse(snapshot_exists(td))
            _write_time_dir(td, f"{FINAL_T:g}")
            status = write_snapshot_after_run(td, record, user_stopped=True)
            self.assertIn("Remap snapshot written", status)
            self.assertTrue(snapshot_exists(td))


if __name__ == "__main__":
    unittest.main()
