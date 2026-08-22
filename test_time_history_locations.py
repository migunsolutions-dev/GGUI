"""Time History Locations dialog: 1D/2D/3D gauges."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from models_2d import ProbePoint2D
from probes_model import ProbePoint, ProbesModel
from tab_1d import Tab1D
from tab_2d import Tab2D
from time_history_dialog import TimeHistoryLocationsDialog, parse_location_file
from generator_1d import Generator1D
from models import BOUNDARY_1D_TRANSMIT, CaseInputs1D, RecommendedParams1D


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TimeHistoryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_tabs_and_columns_match_screenshots(self):
        dialog = TimeHistoryLocationsDialog(None)
        self.assertEqual(dialog.windowTitle(), "Edit Time History Output Locations")
        self.assertEqual(dialog.tabs.count(), 3)
        self.assertEqual([dialog.tabs.tabText(i) for i in range(3)], ["1D", "2D", "3D"])
        self.assertEqual(
            [dialog.tbl_1d.horizontalHeaderItem(i).text() for i in range(2)],
            ["Radius", "Label"],
        )
        self.assertEqual(
            [dialog.tbl_2d.horizontalHeaderItem(i).text() for i in range(3)],
            ["Radius", "Height", "Label"],
        )
        self.assertEqual(
            [dialog.tbl_3d.horizontalHeaderItem(i).text() for i in range(6)],
            ["X", "Y", "Z", "R", "T", "Label"],
        )
        dialog.close()

    def test_add_and_apply_1d_2d_3d(self):
        dialog = TimeHistoryLocationsDialog(
            None,
            gauges_1d=((1.5, "Wall"),),
            probes_2d=(ProbePoint2D("P1", 0.2, 0.4),),
            probes_3d=(ProbePoint("A", 1.0, 0.0, 0.0),),
        )
        self.assertEqual(dialog.gauges_1d(), ((1.5, "Wall"),))
        self.assertEqual(dialog.probes_2d()[0].radius, 0.2)
        self.assertAlmostEqual(dialog.probes_3d()[0].x, 1.0)
        self.assertAlmostEqual(float(dialog.tbl_3d.item(0, 3).text()), 1.0)
        self.assertAlmostEqual(float(dialog.tbl_3d.item(0, 4).text()), 0.0)
        dialog.tbl_3d.selectRow(0)
        dialog._set_3d_flag("remap", True)
        self.assertTrue(dialog.probes_3d()[0].remap)
        self.assertEqual(dialog.remap_origin(), (1.0, 0.0, 0.0))
        dialog.close()

    def test_csv_import_2d(self):
        path = os.path.join(tempfile.mkdtemp(), "g.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("radius,height,label\n0.5,1.0,G1\n")
        rows = parse_location_file(path, "2d")
        self.assertEqual(rows[0]["radius"], "0.5")
        self.assertEqual(rows[0]["label"], "G1")

    def test_apply_updates_tabs(self):
        _app()
        tab1 = Tab1D()
        tab2 = Tab2D()
        model = ProbesModel()
        dialog = TimeHistoryLocationsDialog(None)
        dialog._add_1d()
        dialog.tbl_1d.item(0, 0).setText("2.5")
        dialog.tbl_1d.item(0, 1).setText("R2")
        dialog._add_2d()
        dialog.tbl_2d.item(0, 0).setText("0.3")
        dialog.tbl_2d.item(0, 1).setText("0.8")
        dialog.tbl_2d.item(0, 2).setText("H1")
        tab1.set_gauge_locations(dialog.gauges_1d())
        tab2.replace_probes(dialog.probes_2d())
        model.replace_all(list(dialog.probes_3d()))
        self.assertEqual(tab1.get_case_inputs().gauge_locations, ((2.5, "R2"),))
        probes = tab2._probes()
        self.assertEqual(probes[0].name, "H1")
        self.assertAlmostEqual(probes[0].radius, 0.3)
        dialog.close()

    def test_invalid_or_non_finite_coordinate_blocks_accept(self):
        for invalid in ("not-a-number", "nan", "inf"):
            with self.subTest(invalid=invalid):
                dialog = TimeHistoryLocationsDialog(None)
                dialog._add_2d()
                dialog.tbl_2d.item(0, 0).setText(invalid)
                with mock.patch("time_history_dialog.QMessageBox.warning") as warning:
                    dialog.accept()
                self.assertEqual(dialog.result(), dialog.Rejected)
                warning.assert_called_once()
                dialog.close()


class TimeHistoryGenerateTests(unittest.TestCase):
    def test_1d_writes_gauges1d(self):
        rec = RecommendedParams1D(
            r_min=1.0e-4,
            ignition_point=(0.0, 0.0, 0.0),
            ignition_radius=0.01,
            dt0=1.0e-8,
            maxCo=0.5,
            maxDeltaT=1.0e-5,
        )
        inputs = CaseInputs1D(
            radius=4.0,
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
            gauge_locations=((1.0, "G1"), (2.0, "G2")),
        )
        root = tempfile.mkdtemp(prefix="ggui_th_")
        case = Generator1D(root).generate("g1", inputs, rec)
        with open(os.path.join(case, "system", "controlDict"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("gauges1d", text)
        self.assertEqual(text.count("probeLocations"), 3)


if __name__ == "__main__":
    unittest.main()
