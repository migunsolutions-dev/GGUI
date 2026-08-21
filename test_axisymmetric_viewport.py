from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from unittest import mock

from PyQt5.QtWidgets import QApplication

from axisymmetric_viewer import (
    AxisymmetricViewerWidget,
    preview_charge_outline_points,
)
from generator_2d import Generator2D
from models_2d import CaseInputs2D, SimulationState2D
from tab_2d import Tab2D


app = QApplication.instance() or QApplication([])


class AxisymmetricViewportTests(unittest.TestCase):
    def test_reflecting_ground_sphere_computational_is_quarter_circle(self):
        points = preview_charge_outline_points(
            shape="Sphere",
            height=0.0,
            radius=0.25,
            mirrored=False,
            reflecting_ground=True,
        )
        self.assertAlmostEqual(float(points[:, 0].min()), 0.0, places=12)
        self.assertAlmostEqual(float(points[:, 0].max()), 0.25, places=12)
        self.assertAlmostEqual(float(points[:, 1].min()), 0.0, places=12)
        self.assertAlmostEqual(float(points[:, 1].max()), 0.25, places=12)
        self.assertTrue((points[:, 1] >= -1e-12).all())

    def test_reflecting_ground_sphere_mirrored_is_upper_semicircle(self):
        points = preview_charge_outline_points(
            shape="Sphere",
            height=0.0,
            radius=0.25,
            mirrored=True,
            reflecting_ground=True,
        )
        self.assertAlmostEqual(float(points[:, 0].min()), -0.25, places=12)
        self.assertAlmostEqual(float(points[:, 0].max()), 0.25, places=12)
        self.assertAlmostEqual(float(points[:, 1].min()), 0.0, places=12)
        self.assertAlmostEqual(float(points[:, 1].max()), 0.25, places=4)
        self.assertTrue((points[:, 1] >= -1e-12).all())

    def test_sphere_above_ground_preserves_existing_sections(self):
        half = preview_charge_outline_points(
            shape="Sphere",
            height=1.0,
            radius=0.25,
            mirrored=False,
            reflecting_ground=True,
        )
        full = preview_charge_outline_points(
            shape="Sphere",
            height=1.0,
            radius=0.25,
            mirrored=True,
            reflecting_ground=True,
        )
        self.assertGreaterEqual(float(half[:, 0].min()), -1e-12)
        self.assertAlmostEqual(float(half[:, 1].min()), 0.75, places=12)
        self.assertAlmostEqual(float(half[:, 1].max()), 1.25, places=12)
        self.assertAlmostEqual(float(full[:, 0].min()), -0.25, places=4)
        self.assertAlmostEqual(float(full[:, 0].max()), 0.25, places=12)
        self.assertAlmostEqual(float(full[:, 1].min()), 0.75, places=4)
        self.assertAlmostEqual(float(full[:, 1].max()), 1.25, places=4)

    def test_preview_meridional_extents_mirrored_and_half(self):
        viewer = AxisymmetricViewerWidget()
        self.assertEqual(
            viewer.meridional_display_bounds(1.5, 2.5, True),
            (-1.5, 1.5, 0.0, 2.5),
        )
        self.assertEqual(
            viewer.meridional_display_bounds(1.5, 2.5, False),
            (0.0, 1.5, 0.0, 2.5),
        )
        preview_args = (
            1.5,
            2.5,
            {"shape": "Sphere", "height": 0.5, "radius": 0.05, "length": 0.0},
            [(0.2, 0.7)],
        )
        if viewer._plotter is not None:
            with mock.patch.object(viewer._plotter, "add_text") as add_text:
                viewer.update_axisymmetric_preview(*preview_args)
            add_text.assert_not_called()
        else:
            viewer.update_axisymmetric_preview(*preview_args)
        viewer.set_mirrored_view(True)
        self.assertEqual(viewer._axisymmetric_domain, (1.5, 2.5))
        viewer.set_mirrored_view(False)
        self.assertFalse(viewer.mirrored_view)
        self.assertEqual(viewer._axisymmetric_domain, (1.5, 2.5))

    def test_meridional_slice_of_initialized_wedge_is_xy_not_xz(self):
        import pyvista as pv

        case = r"C:\Users\migun\AppData\Local\Temp\ggui_2d_init_fix_20260724\diag_fixed_150k"
        if not os.path.isdir(case):
            self.skipTest("diagnostic OpenFOAM case not present")
        reader = pv.POpenFOAMReader(os.path.join(case, "case.foam"))
        reader.set_active_time_value(reader.time_values[-1])
        data = reader.read()
        mesh = data["internalMesh"] if "internalMesh" in data.keys() else data[0]
        self.assertEqual(int(mesh.n_cells), 150000)
        self.assertIn("p", mesh.array_names)
        slc = mesh.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0))
        self.assertGreater(slc.n_points, 0)
        xmin, xmax, ymin, ymax, zmin, zmax = slc.bounds
        self.assertAlmostEqual(zmin, 0.0, places=6)
        self.assertAlmostEqual(zmax, 0.0, places=6)
        self.assertGreaterEqual(xmin, -1e-9)
        self.assertAlmostEqual(xmax, 1.5, delta=0.01)
        self.assertAlmostEqual(ymin, 0.0, places=6)
        self.assertAlmostEqual(ymax, 2.5, places=6)

    def test_failed_initialization_clears_result_and_disables_exact_end(self):
        tab = Tab2D()
        tab.mark_initialized("/tmp/fake", 900)
        self.assertEqual(tab.simulation_state, SimulationState2D.INITIALIZED)
        self.assertTrue(tab.btn_exact_end.isEnabled())
        tab.handle_initialization_failure(
            "/tmp/fake_failed",
            "Initialization failed — partial mesh is not a valid result.",
        )
        self.assertEqual(tab.simulation_state, SimulationState2D.FAILED)
        self.assertFalse(tab.btn_exact_end.isEnabled())
        self.assertTrue(tab.btn_initialize.isEnabled())
        self.assertIsNone(tab._actual_cell_count)
        self.assertFalse(tab.viewer.is_simulating)

    def test_field_selector_drives_viewer_field_without_silent_alpha_fallback(self):
        tab = Tab2D()
        tab.cmb_field.setCurrentText("p")
        tab.viewer.set_field(tab.cmb_field.currentText())
        self.assertEqual(tab.viewer.current_field, "p")
        tab.cmb_field.setCurrentText("alpha.c4")
        tab.viewer.set_field(tab.cmb_field.currentText())
        self.assertEqual(tab.viewer.current_field, "alpha.c4")

    def test_fixed_case_metadata_and_init_command_for_150k_grid(self):
        with tempfile.TemporaryDirectory() as td:
            inputs = CaseInputs2D(
                radius=1.5,
                height=2.5,
                cell_size=0.005,
                mesh_mode="Fixed Mesh",
            )
            gen = Generator2D(td)
            case = gen.generate("fixed_view", inputs)
            command = gen.initialization_command(inputs)
            self.assertIn("setFields", command)
            self.assertNotIn("setRefinedFields", command)
            block = open(os.path.join(case, "system", "blockMeshDict"), encoding="utf-8").read()
            self.assertIn("(300 500 1)", block)
            self.assertIn("hex (0 1 2 3 0 4 5 3)", block)


if __name__ == "__main__":
    unittest.main()
