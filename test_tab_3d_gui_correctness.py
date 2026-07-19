from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget

import tab_3d_general
from generator_3d import Generator3D
from probes_model import ProbesModel


class DummyViewer(QWidget):
    cell_count_updated = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.current_case_dir = None
        self.show_mesh_lines = False
        self.show_boundaries = False
        self.show_obstacles = False
        self.show_obstacles_wireframe_only = False
        self.show_tracers = False

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class Tab3DGuiCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.real_viewer = tab_3d_general.BlastViewerWidget
        tab_3d_general.BlastViewerWidget = DummyViewer

    @classmethod
    def tearDownClass(cls):
        tab_3d_general.BlastViewerWidget = cls.real_viewer

    def make_tab(self):
        return tab_3d_general.TabGeneral3D(ProbesModel())

    def test_fresh_display_matches_generated_amr_max(self):
        tab = self.make_tab()
        self.assertEqual(tab.spin_refine_max.value(), 1)
        inputs = tab.get_case_inputs()
        self.assertEqual(inputs.dyn_refine_max, 1)
        # Force Dyn Mesh only to make dynamicMeshDict active; do not alter max.
        tab.rad_dyn_mesh.setChecked(True)
        inputs = tab.get_case_inputs()
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("gui", inputs)
            with open(
                os.path.join(case_dir, "constant", "dynamicMeshDict"),
                encoding="utf-8",
            ) as stream:
                text = stream.read()
        self.assertRegex(text, r"maxRefinement\s+1\s*;")

    def test_edit_and_load_keep_one_runtime_value(self):
        tab = self.make_tab()
        tab.spin_refine_max.setValue(4)
        self.assertEqual(tab._dyn_refine_max, 4)
        self.assertEqual(tab.get_case_inputs().dyn_refine_max, 4)
        tab.set_case_inputs({"dyn_refine_max": 2, "enable_dyn_refine": True})
        self.assertEqual(tab.spin_refine_max.value(), 2)
        self.assertEqual(tab._dyn_refine_max, 2)
        self.assertEqual(tab.get_case_inputs().dyn_refine_max, 2)

    def test_seed_and_outer_band_defaults_remain_protected(self):
        tab = self.make_tab()
        self.assertEqual(tab.spin_charge_refine.value(), 0)
        self.assertEqual(tab.spin_charge_outer_min.value(), 2)
        self.assertEqual(tab.spin_charge_outer_max.value(), 3)


if __name__ == "__main__":
    unittest.main()
