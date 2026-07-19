"""Loader preservation checks against manual building3D/building3D (optional Tab+generate round-trip)."""
from __future__ import annotations

import os
import tempfile
import unittest

from case_loader import load_case
from generator_3d import Generator3D


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _building3d_dir() -> str:
    return os.path.join(_repo_root(), "building3D", "building3D")


@unittest.skipUnless(os.path.isdir(_building3d_dir()), "building3D reference case not in repo")
class Building3dLoaderExtractionTests(unittest.TestCase):
    def test_load_preserves_setfields_cylinder_seed(self) -> None:
        data = load_case(_building3d_dir())
        self.assertEqual(data.get("charge_shape"), "Cylinder")
        self.assertAlmostEqual(data.get("mass_kg", 0), 25.0)
        self.assertAlmostEqual(data.get("rho_charge", 0), 1601.0)
        cc = data.get("charge_center")
        self.assertIsNotNone(cc)
        self.assertAlmostEqual(cc[0], 0.0)
        self.assertAlmostEqual(cc[1], 0.0)
        self.assertAlmostEqual(cc[2], 0.5)
        self.assertEqual(data.get("cylinder_axis"), "Z")
        self.assertAlmostEqual(data.get("charge_lbyd", 0), 2.5)
        self.assertEqual(data.get("charge_refinement_level"), 5)
        self.assertEqual(data.get("buffer_layers"), 5)
        self.assertEqual(data.get("charge_capture_mode"), "manual")
        self.assertAlmostEqual(data.get("charge_capture_radius", 0), 1.0)
        self.assertAlmostEqual(data.get("charge_backup_length_override", 0), 0.5)

    def test_load_preserves_dynamic_mesh_amr(self) -> None:
        data = load_case(_building3d_dir())
        self.assertTrue(data.get("enable_dyn_refine"))
        self.assertEqual(data.get("refine_indicator_field"), "densityGradient")
        self.assertEqual(data.get("refine_interval"), 3)
        self.assertAlmostEqual(data.get("lower_refine_threshold", 0), 0.1)
        self.assertAlmostEqual(data.get("unrefine_threshold", 0), 0.1)
        self.assertEqual(data.get("n_buffer_layers_dynamic"), 2)
        self.assertEqual(data.get("dyn_refine_max"), 1)

    def test_load_preserves_control_and_decompose(self) -> None:
        data = load_case(_building3d_dir())
        self.assertAlmostEqual(data.get("end_time_s", 0), 0.0025)
        self.assertAlmostEqual(data.get("delta_t", 0), 1e-7)
        self.assertAlmostEqual(data.get("write_interval_time", 0), 5e-5)
        self.assertAlmostEqual(data.get("cfl_value", 0), 0.5)
        self.assertEqual(data.get("write_control_type"), "adjustableRunTime")
        self.assertEqual(data.get("cores"), 4)
        self.assertEqual(data.get("decomposition_method"), "scotch")
        self.assertEqual(data.get("decomposition_simple_n"), (2, 2, 1))

    def test_tab_regenerate_preserves_core_dicts(self) -> None:
        from PyQt5.QtWidgets import QApplication

        from probes_model import ProbesModel
        from tab_3d_general import TabGeneral3D

        data = load_case(_building3d_dir())
        summary = data.get("_load_summary") or {}
        app = QApplication.instance() or QApplication([])
        tab = TabGeneral3D(ProbesModel())
        tab.set_case_inputs(data, summary)
        inputs = tab.get_case_inputs()

        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("rt", inputs)
            with open(os.path.join(case_dir, "system", "setFieldsDict"), encoding="utf-8") as f:
                sf = f.read()
            with open(os.path.join(case_dir, "constant", "dynamicMeshDict"), encoding="utf-8") as f:
                dm = f.read()

        self.assertIn("cylindericalMassToCell", sf)
        self.assertIn("mass 25", sf)
        self.assertIn("rho 1601", sf)
        self.assertIn("LbyD 2.5", sf)
        self.assertIn("nBufferLayers 5", sf)
        self.assertIn("level 5", sf)
        self.assertIn("radius 1", sf)
        self.assertRegex(sf, r"L\s+\(\s*0\s+0\s+0\.5\s*\)")
        self.assertIn("errorEstimator  densityGradient;", dm)
        self.assertIn("refineInterval  3;", dm)
        self.assertIn("maxRefinement   1;", dm)
        self.assertIn("dumpLevel      true;", dm)


if __name__ == "__main__":
    unittest.main()
