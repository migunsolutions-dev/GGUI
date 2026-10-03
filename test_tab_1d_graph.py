"""1D initial-condition graph and hidden-VTK paint guard."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QApplication, QGroupBox, QLabel, QVBoxLayout

from tab_1d import (
    Tab1D,
    ideal_gas_charge_pressure_pa,
    initial_overpressure_step,
    spherical_charge_radius_m,
)
from viewer_gl import VtkResizeGuard


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class DummyViewer(QObject):
    def __init__(self):
        super().__init__()
        self._shutdown = False
        self._viewport_active = False


class DomainFieldFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_radius_keeps_long_entry_and_cell_size_has_six_decimals(self):
        tab = Tab1D()
        tab.show()
        self.app.processEvents()
        try:
            self.assertEqual(tab.spin_radius.width(), 150)
            self.assertEqual(tab.spin_cellsize.width(), 150)
            self.assertEqual(tab.spin_cellsize.decimals(), 6)
            self.assertEqual(tab.spin_radius.decimals(), 15)
            tab.spin_radius.setValue(1.234567890123456)
            self.assertAlmostEqual(tab.spin_radius.value(), 1.234567890123456, places=12)
            self.assertNotIn("000000", tab.spin_radius.text())
            tab.spin_cellsize.setValue(0.005)
            self.assertEqual(tab.spin_cellsize.text(), "0.005000")
            self.assertEqual(tab.spin_mass.width(), 100)
            self.assertEqual(tab.combo_comp.currentText(), "TNT")
            self.assertEqual(tab.combo_comp.width(), 125)
            self.assertEqual(tab.spin_density.value(), 1630.0)
            labels = [label.text() for label in tab.findChildren(QLabel)]
            self.assertIn("Composition", labels)
            self.assertNotIn("Mat", labels)
            self.assertIn("CFL", labels)
            self.assertIn("Method", labels)
            self.assertIn("End Time", labels)
            self.assertNotIn("Source model", labels)
            self.assertNotIn("Max CFL", labels)
            self.assertEqual(tab.spin_cfl.width(), 150)
            self.assertEqual(tab.combo_source.width(), 150)
            self.assertEqual(tab.spin_endtime.width(), 150)
            titles = [box.title() for box in tab.findChildren(QGroupBox)]
            self.assertIn("Solver Options", titles)
        finally:
            tab.close()


class InitialOverpressureProfileTests(unittest.TestCase):
    def test_200kg_tnt_step_uses_density_and_energy(self):
        mass = 200.0
        rho = 1630.0
        energy = 4.29e6
        domain = 20.0
        p_atm = 101325.0
        p_he = ideal_gas_charge_pressure_pa(rho, energy)
        self.assertAlmostEqual(p_he, 0.4 * rho * energy, delta=1.0)
        self.assertLess(p_he, 3.0e9)
        charge_r = spherical_charge_radius_m(mass, rho)
        self.assertAlmostEqual(charge_r, 0.308, places=2)
        radii, over = initial_overpressure_step(domain, charge_r, p_he, p_atm)
        self.assertEqual(radii[0], 0.0)
        self.assertEqual(radii[-1], domain)
        self.assertAlmostEqual(radii[1], charge_r)
        self.assertAlmostEqual(radii[2], charge_r)
        self.assertAlmostEqual(over[0], p_he - p_atm)
        self.assertAlmostEqual(over[1], p_he - p_atm)
        self.assertEqual(over[2], 0.0)
        self.assertEqual(over[3], 0.0)


class Tab1DInitialGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_actions_stack_and_supported_charts_change_the_graph(self):
        tab = Tab1D()
        tab.show()
        self.app.processEvents()
        page = tab.btn_run.parentWidget()
        column = page.parentWidget().layout().itemAt(0).layout()
        self.assertIsInstance(column, QVBoxLayout)
        group = column.itemAt(0).widget()
        self.assertEqual(group.title(), "")
        self.assertIs(group.layout().itemAt(0).widget(), tab.btn_initialize)
        self.assertIs(group.layout().itemAt(1).widget(), tab.btn_run)
        self.assertIs(group.layout().itemAt(2).widget(), tab.btn_stop)
        self.assertLess(tab.btn_initialize.mapTo(tab, tab.btn_initialize.rect().topLeft()).y(), tab.btn_run.mapTo(tab, tab.btn_run.rect().topLeft()).y())
        self.assertLess(tab.btn_run.mapTo(tab, tab.btn_run.rect().topLeft()).y(), tab.btn_stop.mapTo(tab, tab.btn_stop.rect().topLeft()).y())
        self.assertGreater(
            tab.radio_chart_pressure.mapTo(tab, tab.radio_chart_pressure.rect().topLeft()).x(),
            tab.btn_run.mapTo(tab, tab.btn_run.rect().topRight()).x(),
        )
        labels = [button.text() for button in tab._chart_group.buttons()]
        self.assertEqual(labels, ["Pressure", "PMax", "Density", "Energy", "Velocity"])
        self.assertTrue(tab.radio_chart_pressure.isChecked())
        tab.radio_chart_density.setChecked(True)
        self.app.processEvents()
        self.assertIn("Density", tab.canvas.axes.get_title())
        self.assertAlmostEqual(float(tab.canvas.axes.lines[0].get_ydata()[0]), tab.spin_density.value())
        tab.radio_chart_velocity.setChecked(True)
        self.app.processEvents()
        self.assertIn("Velocity", tab.canvas.axes.get_title())
        self.assertEqual(float(tab.canvas.axes.lines[0].get_ydata()[0]), 0.0)
        tab.radio_chart_pressure.setChecked(True)
        self.app.processEvents()
        self.assertIn("JWL pressure profile", tab.canvas.axes.get_title())
        tab.update_graph([101325.0, 2.0e8, 101325.0], 1.0e-4, (0.1, 0.5, 0.9))
        tab.update_graph([101325.0, 1.0e8, 3.0e8], 2.0e-4, (0.1, 0.5, 0.9))
        tab.radio_chart_pmax.setChecked(True)
        self.app.processEvents()
        ys = [float(value) for value in tab.canvas.axes.lines[0].get_ydata()]
        self.assertAlmostEqual(ys[1], 2.0e8 - 101325.0)
        self.assertAlmostEqual(ys[2], 3.0e8 - 101325.0)
        tab.close()

    def test_jwl_does_not_show_ideal_gas_pressure_before_run(self):
        tab=Tab1D()
        tab.combo_comp.setCurrentText("TNT")
        tab.spin_mass.setValue(200)
        self.app.processEvents()
        self.assertEqual(len(tab.canvas.axes.lines),0)
        self.assertIn("JWL pressure profile",tab.canvas.axes.get_title())
        self.assertEqual(tab.canvas.axes.get_xlabel(),"Radius (m)")
        tab.spin_density.setValue(1800)
        tab.edit_energy.setText("5e6")
        self.app.processEvents()
        self.assertEqual(len(tab.canvas.axes.lines),0)

    def test_ig_graph_tracks_its_existing_ambient_plus_charge_energy(self):
        tab=Tab1D()
        tab.combo_source.setCurrentIndex(tab.combo_source.findData("IG_ISOTHERMAL_BURST"))
        tab.spin_density.setValue(1600)
        tab.edit_energy.setText("4.52e6")
        self.app.processEvents()
        peak=float(tab.canvas.axes.lines[0].get_ydata()[0])
        self.assertAlmostEqual(peak,ideal_gas_charge_pressure_pa(1600,718*tab.spin_temp.value()+4.52e6)-tab.spin_press.value(),delta=1)

    def test_live_update_does_not_call_synchronous_draw(self):
        tab = Tab1D()
        draws = []
        tab.canvas.draw = lambda: draws.append("draw")
        tab.canvas.draw_idle = lambda: draws.append("idle")
        tab.update_graph([101325.0, 2.0e8, 101325.0], 1.0e-4, (.1,.5,.9))
        tab._redraw_canvas()
        self.assertIn("idle", draws)
        self.assertNotIn("draw", draws)
        self.assertTrue(tab._live_graph)
        tab.end_live_graph()
        self.assertFalse(tab._live_graph)

    def test_live_update_does_not_mutate_axes_until_redraw(self):
        tab = Tab1D()
        initial = len(tab.canvas.axes.lines)
        tab.update_graph([101325.0, 2.0e8, 101325.0], 1.0e-4, (.1,.5,.9))
        self.assertEqual(len(tab.canvas.axes.lines), initial)
        tab._redraw_canvas()
        ys = [float(v) for v in tab.canvas.axes.lines[0].get_ydata()]
        self.assertIn(2.0e8 - 101325.0, ys)

    def test_remap_toggle_does_not_change_initial_graph(self):
        tab = Tab1D()
        tab.combo_comp.setCurrentText("TNT")
        tab.spin_mass.setValue(200.0)
        tab.spin_radius.setValue(20.0)
        tab.radio_no.setChecked(True)
        self.app.processEvents()
        xs_no = list(tab.canvas.axes.lines)
        ys_no = tab.canvas.axes.get_title()
        tab.radio_yes.setChecked(True)
        self.app.processEvents()
        xs_yes = list(tab.canvas.axes.lines)
        ys_yes = tab.canvas.axes.get_title()
        self.assertEqual(xs_no, xs_yes)
        self.assertEqual(ys_no, ys_yes)
        self.assertEqual(tab.lbl_adj_density.text(), f"{tab.spin_density.value():.1f}")
        self.assertAlmostEqual(tab.get_case_inputs().rho_charge, tab.spin_density.value())
        self.assertAlmostEqual(tab.get_case_inputs().material_props["rho"], tab.spin_density.value())
        tab.spin_density.setValue(1700.0)
        self.app.processEvents()
        self.assertEqual(tab.lbl_adj_density.text(), "1700.0")
        self.assertAlmostEqual(tab.get_case_inputs().rho_charge, 1700.0)

    def test_remap_status_stays_hidden_until_yes_and_a_run_case(self):
        tab = Tab1D()
        self.assertTrue(tab.lbl_remap_status.isHidden())
        tab.radio_yes.setChecked(True)
        self.app.processEvents()
        self.assertTrue(tab.lbl_remap_status.isHidden())
        tab.refresh_remap_status(tempfile.gettempdir())
        self.app.processEvents()
        self.assertFalse(tab.lbl_remap_status.isHidden())
        self.assertTrue(tab.lbl_remap_status.text())


class ProbeStreamParseTests(unittest.TestCase):
    def test_incomplete_trailing_line_is_left_unread(self):
        from solver_runner import complete_probe_chunk, parse_last_probe_pressures

        raw = b"# Probe 0 (0 0 0)\n0.1 101325 200000\n0.2 101325 2"
        complete, leftover = complete_probe_chunk(raw)
        self.assertTrue(complete.endswith(b"\n"))
        self.assertEqual(leftover, len(b"0.2 101325 2"))
        parsed = parse_last_probe_pressures(complete.decode("utf-8"))
        self.assertIsNotNone(parsed)
        t, pressures, count = parsed
        self.assertAlmostEqual(t, 0.1)
        self.assertEqual(pressures, [101325.0, 200000.0])
        self.assertEqual(count, 1)


class Tab1DBoundariesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_run_mode_radios_default_terminate(self):
        from models import BOUNDARY_1D_REFLECT, BOUNDARY_1D_TERMINATE, RUN_MODE_REFLECT, RUN_MODE_TERMINATE
        from tab_1d import TIP_RUN_REFLECT, TIP_RUN_TERMINATE

        tab = Tab1D()
        self.assertFalse(tab.cmb_left.isEnabled())
        self.assertEqual(tab.cmb_left.currentText(), "Reflecting - spherical")
        self.assertEqual(tab.cmb_right.currentText(), "Terminate")
        self.assertEqual(tab.cmb_right.itemText(1), "Reflect")
        self.assertEqual(tab.cmb_right.minimumWidth(), tab.cmb_left.minimumWidth())
        self.assertEqual(tab.cmb_right.maximumWidth(), tab.cmb_left.maximumWidth())
        self.assertEqual(tab.cmb_right.toolTip(), TIP_RUN_TERMINATE)
        inputs = tab.get_case_inputs()
        self.assertEqual(inputs.right_boundary, BOUNDARY_1D_TERMINATE)
        self.assertEqual(inputs.stop_mode, RUN_MODE_TERMINATE)
        tab.cmb_right.setCurrentIndex(tab.cmb_right.findData(RUN_MODE_REFLECT))
        restored = tab.get_case_inputs()
        self.assertEqual(restored.right_boundary, BOUNDARY_1D_REFLECT)
        self.assertEqual(restored.stop_mode, RUN_MODE_REFLECT)
        self.assertEqual(tab.cmb_right.toolTip(), TIP_RUN_REFLECT)
        self.assertTrue(tab.spin_endtime.isEnabled())
        self.assertFalse(tab.spin_endtime.isHidden())
        tab.cmb_right.setCurrentIndex(tab.cmb_right.findData(RUN_MODE_TERMINATE))
        self.assertTrue(tab.spin_endtime.isEnabled())
        self.assertFalse(tab.spin_endtime.isHidden())


class Tab1DOutputOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_output_options_sit_under_solver_default_25_steps(self):
        from PyQt5.QtWidgets import QGroupBox

        tab = Tab1D()
        self.assertEqual(tab.group_output.title(), "Output Options")
        self.assertEqual(tab.spin_gui_refresh.value(), 25)
        self.assertEqual(tab.spin_gui_refresh.suffix(), "")
        self.assertEqual(tab.get_case_inputs().probe_write_interval_steps, 25)
        tab.spin_gui_refresh.setValue(50)
        self.assertEqual(tab.get_case_inputs().probe_write_interval_steps, 50)

        titles = []
        layout = tab.left_container.layout()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget() if item is not None else None
            if isinstance(widget, QGroupBox):
                titles.append(widget.title())
        self.assertIn("Solver Options", titles)
        self.assertIn("Output Options", titles)
        self.assertEqual(titles.index("Output Options"), titles.index("Solver Options") + 1)


class Tab1DStopModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_default_is_terminate_and_round_trips(self):
        from models import (
            BOUNDARY_1D_REFLECT,
            BOUNDARY_1D_TERMINATE,
            RUN_MODE_REFLECT,
            RUN_MODE_TERMINATE,
        )

        tab = Tab1D()
        self.assertEqual(tab.cmb_right.currentData(), RUN_MODE_TERMINATE)
        inputs = tab.get_case_inputs()
        self.assertEqual(inputs.stop_mode, RUN_MODE_TERMINATE)
        self.assertEqual(inputs.right_boundary, BOUNDARY_1D_TERMINATE)
        self.assertAlmostEqual(inputs.stop_radius_m, tab.spin_radius.value())
        self.assertFalse(tab.spin_endtime.isHidden())
        self.assertTrue(tab.spin_endtime.isEnabled())
        self.assertAlmostEqual(inputs.end_time_s, tab.spin_endtime.value())
        tab.cmb_right.setCurrentIndex(tab.cmb_right.findData(RUN_MODE_REFLECT))
        restored = tab.get_case_inputs()
        self.assertEqual(restored.stop_mode, RUN_MODE_REFLECT)
        self.assertEqual(restored.right_boundary, BOUNDARY_1D_REFLECT)
        self.assertFalse(tab.spin_endtime.isHidden())
        self.assertTrue(tab.spin_endtime.isEnabled())
        tab.set_case_inputs(
            {
                "stop_mode": RUN_MODE_TERMINATE,
                "right_boundary": BOUNDARY_1D_TERMINATE,
                "radius": 1.0,
                "end_time_s": 0.04,
            }
        )
        self.assertEqual(tab.cmb_right.currentData(), RUN_MODE_TERMINATE)
        self.assertTrue(tab.spin_endtime.isEnabled())
        self.assertAlmostEqual(tab.spin_endtime.value(), 0.04)
        self.assertAlmostEqual(tab.get_case_inputs().stop_radius_m, 1.0)
        self.assertAlmostEqual(tab.get_case_inputs().end_time_s, 0.04)

    def test_remap_for_2d_follows_remap_radio(self):
        from tab_1d import Tab1D

        tab = Tab1D()
        self.assertFalse(tab.get_case_inputs().remap_for_2d)
        tab.radio_yes.setChecked(True)
        self.assertTrue(tab.get_case_inputs().remap_for_2d)
        tab.set_case_inputs({"remap_for_2d": False, "radius": 1.0})
        self.assertFalse(tab.radio_yes.isChecked())
        self.assertFalse(tab.get_case_inputs().remap_for_2d)
        tab.set_case_inputs({"remap_for_2d": True, "radius": 1.0})
        self.assertTrue(tab.radio_yes.isChecked())
        self.assertTrue(tab.get_case_inputs().remap_for_2d)

    def test_right_boundary_wins_when_legacy_stop_mode_conflicts(self):
        from models import BOUNDARY_1D_TERMINATE, RUN_MODE_REFLECT, RUN_MODE_TERMINATE
        from tab_1d import Tab1D

        tab = Tab1D()
        tab.set_case_inputs(
            {
                "right_boundary": BOUNDARY_1D_TERMINATE,
                "stop_mode": "end_time",
            }
        )
        self.assertEqual(tab.cmb_right.currentData(), RUN_MODE_TERMINATE)
        self.assertEqual(tab.get_case_inputs().stop_mode, RUN_MODE_TERMINATE)
        tab.set_case_inputs({"right_boundary": "Transmit", "stop_mode": "end_time"})
        self.assertEqual(tab.cmb_right.currentData(), RUN_MODE_TERMINATE)
        tab.set_case_inputs({"stop_mode": "end_time"})
        self.assertEqual(tab.cmb_right.currentData(), RUN_MODE_REFLECT)

    def test_begin_run_graph_clears_stale_profile_times(self):
        from tab_1d import Tab1D

        tab = Tab1D()
        tab._has_run_profile = True
        tab._live_graph = True
        tab._pending_pressures = [101325.0]
        tab._pending_time_s = 0.0123
        tab.begin_run_graph()
        self.assertFalse(tab._has_run_profile)
        self.assertFalse(tab._live_graph)
        self.assertIsNone(tab._pending_pressures)
        self.assertEqual(tab._pending_time_s, 0.0)


class HiddenVtkPaintGuardTests(unittest.TestCase):
    def test_inactive_viewport_eats_paint(self):
        viewer = DummyViewer()
        guard = VtkResizeGuard(viewer)
        paint = QEvent(QEvent.Paint)
        self.assertTrue(guard.eventFilter(viewer, paint))
        viewer._viewport_active = True
        self.assertFalse(guard.eventFilter(viewer, paint))


class Tab1DInfoStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_info_captions_are_bold_and_values_are_not(self):
        from PyQt5.QtWidgets import QFormLayout
        from ui_metrics import INFO_ROW_STYLE, INFO_TITLE_STYLE

        tab = Tab1D()
        layout = tab.lbl_domain_cells.parentWidget().layout()
        self.assertIsInstance(layout, QFormLayout)
        for value in (
            tab.lbl_domain_cells,
            tab.lbl_charge_radius,
            tab.lbl_charge_cells,
            tab.lbl_adj_density,
        ):
            caption = layout.labelForField(value)
            self.assertEqual(caption.styleSheet(), INFO_TITLE_STYLE)
            self.assertIn("font-weight: bold", caption.styleSheet())
            self.assertEqual(value.styleSheet(), INFO_ROW_STYLE)
            self.assertNotIn("font-weight", value.styleSheet())


if __name__ == "__main__":
    unittest.main()
