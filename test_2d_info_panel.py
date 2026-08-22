"""State-driven engineering rows in the Cylindrical–2D Info panel."""
from __future__ import annotations

import math
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication, QScrollArea

from axisymmetric_2d import DIRECT_SOURCE, DYNAMIC_MESH, FIXED_MESH, REMAP_SOURCE
from external_case_workflow_2d import count_initialized_charge_cells
from models_2d import SimulationState2D
from tab_2d import Tab2D


class Tab2DInfoPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = Tab2D()
        self.tab.show()
        self.app.processEvents()

    def tearDown(self):
        self.tab.close()

    def _visible(self):
        return {
            label.text().rstrip(":"): value.text()
            for label, value in self.tab._info_row_widgets.values()
            if not label.isHidden()
        }

    def _set_fixed_direct_fixture(self):
        self.tab.cmb_source.setCurrentText(DIRECT_SOURCE)
        self.tab.cmb_mesh_mode.setCurrentText(FIXED_MESH)
        self.tab.spin_radius.setValue(0.1)
        self.tab.spin_height.setValue(0.1)
        self.tab.spin_cell.setValue(0.01)
        self.tab.spin_hob.setValue(0.025)
        self.tab.spin_density.setValue(1630.0)
        radius = 0.025
        mass = 1630.0 * (4.0 / 3.0) * math.pi * radius**3
        self.tab.spin_mass.setValue(mass)
        self.tab._refresh_derived()

    def test_fixed_direct_has_five_exact_rows(self):
        self._set_fixed_direct_fixture()
        rows = self._visible()
        self.assertEqual(
            list(rows),
            [
                "Radius Cells",
                "Height Cells",
                "Charge Radius",
                "Charge Cells",
                "Total Cells",
            ],
        )
        self.assertEqual(rows["Radius Cells"], "10")
        self.assertEqual(rows["Height Cells"], "10")
        self.assertEqual(rows["Charge Radius"], "0.02500 m")
        self.assertEqual(rows["Charge Cells"], "8")
        self.assertEqual(rows["Total Cells"], "100")

    def test_dynamic_direct_before_initialise_has_only_planned_rows(self):
        self.tab.cmb_source.setCurrentText(DIRECT_SOURCE)
        self.tab.cmb_mesh_mode.setCurrentText(DYNAMIC_MESH)
        self.tab._refresh_derived()
        rows = self._visible()
        self.assertEqual(
            list(rows), ["Base Cell", "Finest Cell", "Charge Radius", "Charge Res."]
        )
        self.assertNotIn("Charge Cells", rows)
        self.assertNotIn("Total Cells", rows)
        self.assertIn("cells/D", rows["Charge Res."])

    def test_dynamic_direct_after_initialise_uses_actual_counts(self):
        self.tab.cmb_source.setCurrentText(DIRECT_SOURCE)
        self.tab.cmb_mesh_mode.setCurrentText(DYNAMIC_MESH)
        self.tab.mark_initialized("case", actual_cells=18_650, charge_cells=412)
        rows = self._visible()
        self.assertEqual(
            list(rows),
            [
                "Finest Cell",
                "Charge Radius",
                "Charge Res.",
                "Charge Cells",
                "Total Cells",
            ],
        )
        self.assertEqual(rows["Charge Cells"], "412")
        self.assertEqual(rows["Total Cells"], "18,650")

    def test_runtime_total_updates_without_changing_labels(self):
        self.tab.cmb_source.setCurrentText(DIRECT_SOURCE)
        self.tab.cmb_mesh_mode.setCurrentText(DYNAMIC_MESH)
        self.tab.mark_initialized("case", actual_cells=18_650, charge_cells=412)
        self.tab.set_simulation_state(SimulationState2D.RUNNING)
        labels_before = list(self._visible())
        self.tab._on_cell_count_updated(23_400)
        rows = self._visible()
        self.assertEqual(list(rows), labels_before)
        self.assertEqual(rows["Total Cells"], "23,400")
        self.assertEqual(rows["Charge Cells"], "412")

    def test_fixed_remap_has_no_charge_rows(self):
        self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
        self.tab.cmb_mesh_mode.setCurrentText(FIXED_MESH)
        self.tab._refresh_derived()
        rows = self._visible()
        self.assertEqual(
            list(rows), ["Radius Cells", "Height Cells", "Cell Size", "Total Cells"]
        )
        self.assertNotIn("Charge Radius", rows)
        self.assertNotIn("Charge Res.", rows)
        self.assertNotIn("Charge Cells", rows)

    def test_domain_match_is_silent_and_mismatch_warns(self):
        self.assertTrue(self.tab.lbl_info_warning.isHidden())
        self.assertNotIn(
            "Requested and effective domain match",
            self.tab.lbl_info_warning.text(),
        )
        self.tab.spin_radius.setValue(1.53)
        self.tab._refresh_derived()
        self.assertFalse(self.tab.lbl_info_warning.isHidden())
        self.assertEqual(
            self.tab.lbl_info_warning.text(),
            "Warning: Effective domain differs from requested.",
        )

    def test_info_container_preserves_left_layout_structure(self):
        self.assertIs(self.tab.info_frame.parentWidget(), self.tab.input_tabs.parentWidget())
        self.assertFalse(isinstance(self.tab.info_frame, QScrollArea))
        self.assertTrue(self.tab.info_frame.isAncestorOf(self.tab.info_body))
        self.assertEqual(
            self.tab.info_frame.objectName(), "cylindrical2dInfoPanel"
        )
        self.assertEqual(self.tab.info_body.objectName(), "cylindrical2dInfoBody")

    def test_initialized_charge_count_reads_only_initial_alpha_field(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "0"))
            with open(
                os.path.join(root, "0", "alpha.c4"), "w", encoding="utf-8"
            ) as stream:
                stream.write(
                    "internalField nonuniform List<scalar>\n"
                    "5\n(\n0\n1\n0.25\n1\n0\n)\n;\n"
                )
            self.assertEqual(count_initialized_charge_cells(root), 3)


if __name__ == "__main__":
    unittest.main()
