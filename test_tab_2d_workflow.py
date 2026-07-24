from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from axisymmetric_2d import DYNAMIC_MESH, FIXED_MESH, REMAP_SOURCE
from axisymmetric_viewer import AxisymmetricViewerWidget
from models_2d import SimulationState2D
from tab_2d import Tab2D


class Tab2DWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = Tab2D()
        self.tab.show()
        self.app.processEvents()

    def tearDown(self):
        self.tab.close()

    def test_direct_remap_gating(self):
        self.assertTrue(self.tab.grp_charge.isEnabled())
        self.assertFalse(self.tab.grp_mapping.isEnabled())
        self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
        self.app.processEvents()
        self.assertFalse(self.tab.grp_charge.isEnabled())
        self.assertTrue(self.tab.grp_mapping.isEnabled())
        self.assertFalse(self.tab.grp_seed.isEnabled())

    def test_fixed_dynamic_gating(self):
        self.tab.cmb_mesh_mode.setCurrentText(FIXED_MESH)
        self.assertFalse(self.tab.grp_seed.isEnabled())
        self.assertFalse(self.tab.grp_amr.isEnabled())
        self.tab.cmb_mesh_mode.setCurrentText(DYNAMIC_MESH)
        self.assertTrue(self.tab.grp_seed.isEnabled())
        self.assertTrue(self.tab.grp_amr.isEnabled())

    def test_sphere_cylinder_enablement(self):
        self.tab.cmb_shape.setCurrentText("Sphere")
        self.assertFalse(self.tab.spin_ld.isVisible())
        self.tab.cmb_shape.setCurrentText("Cylinder")
        self.app.processEvents()
        self.assertTrue(self.tab.spin_ld.isVisible())

    def test_initialized_model_changes_stale_but_view_changes_do_not(self):
        self.tab.mark_initialized("/tmp/case", 900)
        self.tab.cmb_view_mode.setCurrentText("Computational Domain View")
        self.assertEqual(self.tab.simulation_state, SimulationState2D.INITIALIZED)
        self.tab.spin_radius.setValue(self.tab.spin_radius.value() + 0.1)
        self.assertEqual(self.tab.simulation_state, SimulationState2D.STALE)
        self.assertFalse(self.tab.btn_exact_end.isEnabled())

    def test_interruption_and_continuation_controls(self):
        self.tab.set_simulation_state(SimulationState2D.RUNNING)
        self.assertTrue(self.tab.btn_stop.isEnabled())
        self.assertFalse(self.tab.btn_exact_end.isEnabled())
        self.tab.set_simulation_state(SimulationState2D.INTERRUPTED)
        self.assertFalse(self.tab.btn_stop.isEnabled())
        self.assertTrue(self.tab.btn_exact_end.isEnabled())

    def test_exactly_one_viewport_and_approved_actions(self):
        self.assertEqual(len(self.tab.findChildren(AxisymmetricViewerWidget)), 1)
        texts = {button.text() for button in self.tab.findChildren(type(self.tab.btn_stop))}
        self.assertIn("Initialise Model", texts)
        self.assertIn("exact END", texts)
        self.assertIn("Interrupt", texts)
        for forbidden in ("Exec 1", "Exec 10", "Exec 100", "Exec 1000", "Run / Resume"):
            self.assertNotIn(forbidden, texts)

    def test_info_first_count_and_no_base_cell_size(self):
        labels = [
            self.tab.lbl_info_total.text(),
            self.tab.lbl_info_grid.text(),
            self.tab.lbl_info_charge.text(),
            self.tab.lbl_info_resolution.text(),
        ]
        self.assertTrue(labels[0].startswith("Estimated cells before initialization"))
        self.assertNotIn("Base Cell Size", "\n".join(labels))


if __name__ == "__main__":
    unittest.main()
