"""Focused tests for the VIPER vs GGUI comparison utilities."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from viper_compare.cli import viper_argv
from viper_compare.extract import derived_positive_impulse, histories_empty, parse_viper_th
from viper_compare.physics import TestDefinition
from validation.kb_propagation import CLASS_INSIDE, CLASS_ON_BOUNDARY, CLASS_OUTSIDE

TEST2_VIP = r"C:\Users\migun\Desktop\1TEST\TEST2.vip"
TEST2_JSON = r"C:\Users\migun\Desktop\1TEST\TEST2.json"
NO_REMAP_VIP = r"C:\VIPER_COMPARE\templates\12_pair\1\No_Remap.vip"
NO_REMAP_JSON = r"C:\VIPER_COMPARE\templates\12_pair\1\No_Remap.json"
REMAP_VIP = r"C:\VIPER_COMPARE\templates\12_pair\2\Remap.vip"
REMAP_JSON = r"C:\VIPER_COMPARE\templates\12_pair\2\Remap.json"
REMAP_DIR = r"C:\VIPER_COMPARE\templates\12_pair\2"
DIRECT_DIR = r"C:\VIPER_COMPARE\templates\12_pair\1"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@unittest.skipUnless(os.path.isfile(TEST2_VIP), "TEST2.vip not available")
class Test2SchemaTests(unittest.TestCase):
    def test_known_gauge_schema(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py required")
        from viper_compare.schema import extract_gauge_schema

        schema = extract_gauge_schema(TEST2_VIP)
        self.assertEqual(schema.numthloc_1d, 2)
        self.assertEqual(len(schema.thlocx_1d), 2)
        np.testing.assert_allclose(schema.thlocx_1d, [0.3, 0.5], atol=1e-6)
        self.assertEqual(schema.labels_1d, ["Gauge1d 1", "Gauge1d 2"])
        self.assertEqual(schema.numthloc_2d, 2)
        np.testing.assert_allclose(schema.thlocx_2d, [1.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(schema.thlocy_2d, [0.001, 0.001], atol=1e-6)
        self.assertEqual(schema.labels_2d, ["Gauge2d 1", "Gauge2d 2"])
        self.assertEqual(schema.pressure_1d, 1)
        self.assertEqual(schema.impulse_1d, 1)
        self.assertEqual(schema.pressure_2d, 1)
        self.assertEqual(schema.impulse_2d, 1)


@unittest.skipUnless(os.path.isfile(TEST2_VIP), "TEST2.vip not available")
class VipGaugeEditTests(unittest.TestCase):
    def setUp(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py required")
        self.original_hash = _sha256(TEST2_VIP)

    def tearDown(self):
        self.assertEqual(_sha256(TEST2_VIP), self.original_hash)

    def test_copy_edit_reopen_and_original_unchanged(self):
        from viper_compare.vip_gauges import build_model, validate_gauges

        gauges_1d = [(0.20, "G1D_R0.20"), (0.40, "G1D_R0.40"), (0.70, "G1D_R0.70")]
        gauges_2d = [
            (0.20, 1.0, "G2D_R0.20"),
            (0.70, 1.0, "G2D_R0.70"),
            (1.20, 1.0, "G2D_R1.20"),
        ]
        with tempfile.TemporaryDirectory() as td:
            dest = os.path.join(td, "copy.vip")
            build_model(TEST2_VIP, dest, gauges_1d=gauges_1d, gauges_2d=gauges_2d)
            validate_gauges(dest, gauges_1d=gauges_1d, gauges_2d=gauges_2d)
            import h5py

            with h5py.File(dest, "r") as handle:
                g = handle["vipermodel"]
                self.assertEqual(int(g["numthloc_1d"][()][0]), 3)
                self.assertEqual(int(g["numthloc_2d"][()][0]), 3)
                self.assertEqual(int(g["outputQuantitiesFlag_1D_Gauges_Pressure"][()][0]), 1)
                self.assertEqual(int(g["outputQuantitiesFlag_1D_Gauges_Impulse"][()][0]), 1)
                self.assertEqual(int(g["outputQuantitiesFlag_2D_Gauges_Pressure"][()][0]), 1)
                self.assertEqual(int(g["outputQuantitiesFlag_2D_Gauges_Impulse"][()][0]), 1)
                self.assertIn("th1dlabel_2", g)
                self.assertNotIn("th3dlabel_0", g)
        self.assertEqual(_sha256(TEST2_VIP), self.original_hash)


class CliAndExtractTests(unittest.TestCase):
    def test_cli_argv(self):
        argv = viper_argv(r"C:\viper\viperblast.exe", r"C:\m.vip", r"C:\m.json", ["1d", "2d"])
        self.assertEqual(
            argv,
            [
                r"C:\viper\viperblast.exe",
                "nogui",
                "file=C:\\m.vip",
                "json=C:\\m.json",
                "1d",
                "2d",
            ],
        )

    def test_empty_history_detected(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "empty.txt")
            Path(path).write_text("# time g1 g2\n", encoding="utf-8")
            self.assertTrue(histories_empty(path))
            Path(path).write_text("# time g1\n0.0 101325\n1e-4 2.0e5\n", encoding="utf-8")
            self.assertFalse(histories_empty(path))
            labels, times, values = parse_viper_th(path)
            self.assertEqual(values.shape[1], 1)
            self.assertEqual(times.size, 2)

    def test_same_impulse_integrator(self):
        t = np.array([0.0, 0.001, 0.002, 0.003])
        p = np.array([101325.0, 201325.0, 101325.0, 101325.0])
        a = derived_positive_impulse(t, p, 101325.0)
        b = derived_positive_impulse(t, p, 101325.0)
        self.assertEqual(a, b)
        self.assertGreater(a, 0.0)


class GeometryTests(unittest.TestCase):
    def test_physical_r_and_remap_exclusion(self):
        spec = TestDefinition()
        self.assertAlmostEqual(spec.charge_radius_m, 0.05305, places=4)
        self.assertGreater(spec.cells_through_charge_1d, 50.0)
        self.assertGreater(spec.cells_through_charge_2d, 5.0)
        self.assertEqual(spec.classify_remap(0.50), CLASS_INSIDE)
        self.assertEqual(spec.classify_remap(0.60), CLASS_ON_BOUNDARY)
        self.assertEqual(spec.classify_remap(0.605), CLASS_ON_BOUNDARY)
        self.assertEqual(spec.classify_remap(0.70), CLASS_OUTSIDE)
        self.assertTrue(0.70 > spec.r_remap_m + spec.dx_2d)

    def test_direct_and_remap_configs_distinct(self):
        spec = TestDefinition()
        from viper_compare.ggui_run import inputs_2d

        direct = inputs_2d(spec, remapped=False)
        remap = inputs_2d(spec, remapped=True, source_1d=r"C:\tmp\src")
        self.assertNotEqual(direct.initialization_source, remap.initialization_source)
        self.assertEqual(remap.mapping.mapped_radius, spec.r_remap_m)
        self.assertEqual(direct.mapping.mapped_radius, 0.0)

    def test_no_artificial_x_offset(self):
        from viper_compare.analyze import gauge_row

        t = np.array([0.0, 0.001])
        p = np.array([2.0e5, 1.2e5])
        row = gauge_row(
            solver="VIPER",
            configuration="2d_remap",
            dimension="2d",
            remapped=True,
            gauge_label="G2D_R0.70",
            r_m=0.70,
            mass_kg=1.0,
            times=t,
            pressure=p,
            native_impulse=None,
            p_atm=101325.0,
            receive_r_max=0.60,
            dx_2d=0.01,
            source_case="x",
            source_file="y",
        )
        self.assertEqual(row["R_m"], 0.70)
        self.assertNotEqual(row["R_m"], 0.70 + 0.60)
        self.assertTrue(row["independent_2d"])

    def test_viper_json_overrides_template(self):
        if not os.path.isfile(NO_REMAP_JSON) or not os.path.isfile(REMAP_JSON):
            self.skipTest("12.zip JSON pair not available")
        spec = TestDefinition()
        direct = spec.viper_json(NO_REMAP_JSON, domain_1d_m=0.60)
        remap = spec.viper_json(REMAP_JSON, domain_1d_m=0.60)
        self.assertEqual(direct["params_2d"]["shape"], 1)
        self.assertEqual(remap["params_2d"]["shape"], 0)
        self.assertEqual(direct["params_2d"]["boun_bottom"], 1)
        self.assertEqual(direct["params_2d"]["cellsize"], 0.01)
        self.assertEqual(direct["params_1d"]["domain_radius_od"], 0.60)
        self.assertIn("jwl_od_a1", direct["params_1d"])


@unittest.skipUnless(os.path.isfile(NO_REMAP_VIP) and os.path.isfile(REMAP_VIP), "12.zip VIP pair not available")
class RemapIdentityTests(unittest.TestCase):
    def setUp(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py required")
        self.no_hash = _sha256(NO_REMAP_VIP)
        self.yes_hash = _sha256(REMAP_VIP)

    def tearDown(self):
        self.assertEqual(_sha256(NO_REMAP_VIP), self.no_hash)
        self.assertEqual(_sha256(REMAP_VIP), self.yes_hash)

    def test_pair_diff_is_flag_shape_and_clocks(self):
        from viper_compare.vip_diff import diff_vips

        diff = diff_vips(NO_REMAP_VIP, REMAP_VIP)
        names = {row["name"]: row for row in diff["changed"]}
        self.assertEqual(set(names), {"remapflag", "shape", "dt", "step", "tt", "tt_2d"})
        self.assertEqual(names["remapflag"]["class"], "model-definition")
        self.assertEqual(names["shape"]["class"], "model-definition")
        self.assertEqual(int(names["remapflag"]["left"]), 0)
        self.assertEqual(int(names["remapflag"]["right"]), 1)
        self.assertEqual(int(names["shape"]["left"]), 1)
        self.assertEqual(int(names["shape"]["right"]), 0)
        self.assertEqual(diff["remap_enabling"]["twodremapoption"]["left"], 1)
        self.assertEqual(diff["remap_enabling"]["twodremapoption"]["right"], 1)

    def test_build_model_preserves_direct_and_remap_identity(self):
        from viper_compare.vip_diff import assert_remap_identity
        from viper_compare.vip_gauges import build_model

        gauges_1d = [(0.20, "G1D_R0.20"), (0.40, "G1D_R0.40")]
        gauges_2d = [(0.70, 1.0, "G2D_R0.70"), (1.20, 1.0, "G2D_R1.20")]
        with tempfile.TemporaryDirectory() as td:
            direct = os.path.join(td, "direct.vip")
            remap = os.path.join(td, "remap.vip")
            build_model(NO_REMAP_VIP, direct, gauges_1d=gauges_1d, gauges_2d=gauges_2d)
            build_model(REMAP_VIP, remap, gauges_1d=gauges_1d, gauges_2d=gauges_2d)
            assert_remap_identity(direct, remapflag=0, shape=1)
            assert_remap_identity(remap, remapflag=1, shape=0)

    def test_provided_remap_is_initialized_only(self):
        from viper_compare.remap_proof import remap_propagation_report

        provided = remap_propagation_report(REMAP_DIR)
        self.assertTrue(provided["initialized_only"])
        self.assertEqual(provided["n_vtk"], 1)
        self.assertEqual(provided["vtk_frames"], ["viper2d_0.vtk"])
        self.assertFalse(provided["run_summary_has_2d"])
        direct = remap_propagation_report(DIRECT_DIR)
        self.assertGreater(direct["max_2d_step"], 0)
        self.assertGreaterEqual(direct["n_vtk"], 2)
        self.assertTrue(direct["run_summary_has_2d"])


class AllrunWatchdogDefaultTests(unittest.TestCase):
    def test_terminate_standalone_matches_gui_watchdog(self):
        from models import BOUNDARY_1D_TERMINATE, RUN_MODE_TERMINATE
        from viper_compare.ggui_run import allrun_watchdog_default

        class Inputs:
            stop_mode = RUN_MODE_TERMINATE
            right_boundary = BOUNDARY_1D_TERMINATE
            remap_for_2d = False

        self.assertTrue(allrun_watchdog_default(Inputs()))

    def test_reflect_does_not_enable_watchdog(self):
        from models import BOUNDARY_1D_REFLECT, RUN_MODE_REFLECT
        from viper_compare.ggui_run import allrun_watchdog_default

        class Inputs:
            stop_mode = RUN_MODE_REFLECT
            right_boundary = BOUNDARY_1D_REFLECT
            remap_for_2d = False

        self.assertFalse(allrun_watchdog_default(Inputs()))

    def test_allrun_discards_solver_stdout(self):
        from viper_compare.ggui_run import ALLRUN

        self.assertIn(">/dev/null", ALLRUN)

    def test_harness_watchdog_poll_matches_gui(self):
        from completion_1d import WATCHDOG_POLL_S
        from viper_compare.ggui_run import WATCHDOG_POLL_S as HARNESS_POLL

        self.assertEqual(WATCHDOG_POLL_S, 0.10)
        self.assertEqual(HARNESS_POLL, WATCHDOG_POLL_S)


if __name__ == "__main__":
    unittest.main()
