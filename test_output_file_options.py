"""Output File Options dialog and generated controlDict wiring."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QLabel

from dialogs import OutputFileOptionsDialog
from generator_1d import Generator1D
from models import BOUNDARY_1D_TRANSMIT, CaseInputs1D, RecommendedParams1D
from output_options import OutputFileOptions, extra_function_objects, surfaces_vtk_block
from tab_1d import Tab1D


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _1d_inputs(**kwargs) -> CaseInputs1D:
    base = dict(
        radius=1.0,
        cell_size=0.05,
        p_atm=101325.0,
        t_atm=288.0,
        mass_kg=1.0,
        rho_charge=1601.0,
        energy_j_per_kg=4.52e6,
        material_props={"rho": 1601.0, "A": 1.0, "B": 1.0, "R1": 4.0, "R2": 1.0, "omega": 0.25, "E0": 1.0},
        max_cfl=0.5,
        end_time_s=1.0e-3,
        right_boundary=BOUNDARY_1D_TRANSMIT,
    )
    base.update(kwargs)
    return CaseInputs1D(**base)


def _rec() -> RecommendedParams1D:
    return RecommendedParams1D(
        r_min=1.0e-4,
        ignition_point=(0.0, 0.0, 0.0),
        ignition_radius=0.01,
        dt0=1.0e-8,
        maxCo=0.5,
        maxDeltaT=1.0e-5,
    )


class OutputFileOptionsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_three_tabs_match_blastfoam_not_viper_extras(self):
        dialog = OutputFileOptionsDialog(None, OutputFileOptions())
        self.assertEqual(dialog.windowTitle(), "Output File Options")
        self.assertEqual(dialog.tabs.count(), 3)
        self.assertEqual([dialog.tabs.tabText(i) for i in range(3)], ["1D", "2D", "3D"])
        page = dialog.tabs.widget(1)
        joined = " ".join(
            child.text() for child in page.findChildren(QLabel) if child.text()
        )
        self.assertNotIn("remap2d.vip", joined)
        self.assertNotIn("ASII", joined)
        self.assertNotIn("HVEL", joined)
        self.assertNotIn("RISK", joined)
        self.assertIn("Gauges", joined)
        self.assertIn("Whole Domain VTKs", joined)
        self.assertTrue(dialog._gauges_1d["overpressure"].isChecked())
        self.assertTrue(dialog._gauges_1d["impulse"].isChecked())
        self.assertFalse(dialog._gauges_1d["density"].isChecked())
        self.assertEqual(dialog.chk_surfaces.text(), "Cross-sections and surfaces")
        self.assertTrue(dialog.chk_surfaces.isChecked())
        self.assertEqual(dialog.chk_volumes.text(), "Volumes")
        dialog.close()

    def test_ok_returns_unchecked_density_and_3d_volumes(self):
        dialog = OutputFileOptionsDialog(None, OutputFileOptions())
        dialog._gauges_1d["density"].setChecked(True)
        dialog.chk_volumes.setChecked(False)
        opts = dialog.get_options()
        self.assertTrue(opts.dim1d.gauges.density)
        self.assertIn("rho", opts.dim1d.gauges.foam_probe_fields(always_p=True))
        self.assertFalse(opts.dim3d.write_volumes)
        self.assertTrue(opts.dim3d.write_surfaces)
        dialog.close()


class SurfaceVtkBlockTests(unittest.TestCase):
    def test_cutting_plane_and_obstacle_patch(self):
        block = surfaces_vtk_block(
            by_time=True,
            interval_time=0.001,
            interval_steps=25,
            fields=("p", "rho"),
            planes=(("Ground", 0.0, 0.0, 0.5, 0.0, 0.0, 1.0),),
            patches=("obs0_wall_stl",),
        )
        self.assertIn("type            surfaces;", block)
        self.assertIn("surfaceFormat   vtk;", block)
        self.assertIn("type            cuttingPlane;", block)
        self.assertIn("point       (0 0 0.5);", block)
        self.assertIn("type            patch;", block)
        self.assertIn("patches         (obs0_wall_stl);", block)
        self.assertIn("writeInterval   0.001;", block)


class OutputFileOptionsGenerateTests(unittest.TestCase):
    def test_1d_controlDict_writes_selected_gauges(self):
        root = tempfile.mkdtemp(prefix="ggui_out_1d_")
        gen = Generator1D(root)
        inputs = _1d_inputs(
            probe_fields=("p", "impulse", "rho"),
            enable_impulse=True,
            enable_dynamic_pressure=False,
        )
        case = gen.generate("case_out", inputs, _rec())
        with open(os.path.join(case, "system", "controlDict"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("fields          (p impulse rho);", text)
        self.assertIn("type            impulse;", text)
        self.assertNotIn("dynamicPressure", text)

    def test_impulse_helper_emits_pRef(self):
        block = extra_function_objects(
            p_atm=101325.0,
            impulse=True,
            overpressure=False,
            dynamic_pressure=False,
            peaks=False,
        )
        self.assertIn("pRef            101325", block)

    def test_tab_1d_apply_gauges_reaches_get_case_inputs(self):
        _app()
        tab = Tab1D()
        tab.apply_output_gauges(("p", "T"), impulse=False, dynamic_pressure=True)
        inputs = tab.get_case_inputs()
        self.assertEqual(inputs.probe_fields, ("p", "T"))
        self.assertFalse(inputs.enable_impulse)
        self.assertTrue(inputs.enable_dynamic_pressure)


if __name__ == "__main__":
    unittest.main()
