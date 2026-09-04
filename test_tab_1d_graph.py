"""1D initial-condition graph and hidden-VTK paint guard."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QApplication

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

    def test_graph_shows_tnt_step_from_inputs_before_run(self):
        tab = Tab1D()
        tab.combo_comp.setCurrentText("TNT")
        tab.spin_mass.setValue(200.0)
        tab.spin_radius.setValue(20.0)
        self.app.processEvents()
        self.assertEqual(tab.canvas.axes.get_title(), "Overpressure vs Range")
        self.assertEqual(tab.canvas.axes.get_xlabel(), "Radius (m)")
        self.assertEqual(tab.canvas.axes.get_ylabel(), "Overpressure (Pa)")
        line = tab.canvas.axes.lines[0]
        self.assertEqual(line.get_label(), "Pressure")
        xs, ys = line.get_xdata(), line.get_ydata()
        charge_r = spherical_charge_radius_m(200.0, 1630.0)
        peak = ideal_gas_charge_pressure_pa(1630.0, 4.29e6) - 101325.0
        self.assertAlmostEqual(float(xs[0]), 0.0)
        self.assertAlmostEqual(float(xs[-1]), 20.0)
        self.assertAlmostEqual(float(xs[1]), charge_r, places=5)
        self.assertAlmostEqual(float(ys[0]), peak, delta=1.0)
        self.assertLess(float(ys[0]), 3.0e9)
        self.assertAlmostEqual(float(ys[-1]), 0.0)
        self.assertFalse(tab._live_graph)

    def test_graph_peak_tracks_density_and_energy(self):
        tab = Tab1D()
        tab.combo_comp.setCurrentText("TNT")
        tab.spin_density.setValue(1600.0)
        tab.edit_energy.setText("4.52e6")
        self.app.processEvents()
        first = float(tab.canvas.axes.lines[0].get_ydata()[0])
        self.assertAlmostEqual(
            first,
            ideal_gas_charge_pressure_pa(1600.0, 4.52e6) - 101325.0,
            delta=1.0,
        )
        tab.spin_density.setValue(1800.0)
        self.app.processEvents()
        denser = float(tab.canvas.axes.lines[0].get_ydata()[0])
        self.assertGreater(denser, first)
        tab.edit_energy.setText("5.0e6")
        self.app.processEvents()
        hotter = float(tab.canvas.axes.lines[0].get_ydata()[0])
        self.assertGreater(hotter, denser)

    def test_live_update_does_not_call_synchronous_draw(self):
        tab = Tab1D()
        draws = []
        tab.canvas.draw = lambda: draws.append("draw")
        tab.canvas.draw_idle = lambda: draws.append("idle")
        tab.update_graph([101325.0, 2.0e8, 101325.0], 1.0e-4)
        tab._redraw_canvas()
        self.assertIn("idle", draws)
        self.assertNotIn("draw", draws)
        self.assertTrue(tab._live_graph)
        tab.end_live_graph()
        self.assertFalse(tab._live_graph)

    def test_live_update_does_not_mutate_axes_until_redraw(self):
        tab = Tab1D()
        initial = float(tab.canvas.axes.lines[0].get_ydata()[0])
        tab.update_graph([101325.0, 2.0e8, 101325.0], 1.0e-4)
        self.assertAlmostEqual(float(tab.canvas.axes.lines[0].get_ydata()[0]), initial)
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
        xs_no = [float(v) for v in tab.canvas.axes.lines[0].get_xdata()]
        ys_no = [float(v) for v in tab.canvas.axes.lines[0].get_ydata()]
        tab.radio_yes.setChecked(True)
        self.app.processEvents()
        xs_yes = [float(v) for v in tab.canvas.axes.lines[0].get_xdata()]
        ys_yes = [float(v) for v in tab.canvas.axes.lines[0].get_ydata()]
        self.assertEqual(xs_no, xs_yes)
        self.assertEqual(ys_no, ys_yes)
        self.assertEqual(tab.lbl_adj_density.text(), f"{tab.spin_density.value():.1f}")
        self.assertAlmostEqual(tab.get_case_inputs().rho_charge, tab.spin_density.value())
        self.assertAlmostEqual(tab.get_case_inputs().material_props["rho"], tab.spin_density.value())
        tab.spin_density.setValue(1700.0)
        self.app.processEvents()
        self.assertEqual(tab.lbl_adj_density.text(), "1700.0")
        self.assertAlmostEqual(tab.get_case_inputs().rho_charge, 1700.0)


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

    def test_right_boundary_has_three_options_default_transmit(self):
        from models import BOUNDARY_1D_RIGHT_OPTIONS, BOUNDARY_1D_TRANSMIT

        tab = Tab1D()
        self.assertFalse(tab.cmb_left.isEnabled())
        self.assertEqual(tab.cmb_left.currentText(), "Reflecting - spherical")
        self.assertTrue(tab.cmb_right.isEnabled())
        labels = [tab.cmb_right.itemText(i) for i in range(tab.cmb_right.count())]
        self.assertEqual(labels, list(BOUNDARY_1D_RIGHT_OPTIONS))
        self.assertEqual(tab.cmb_right.currentText(), BOUNDARY_1D_TRANSMIT)
        self.assertEqual(tab.get_case_inputs().right_boundary, BOUNDARY_1D_TRANSMIT)
        tab.cmb_right.setCurrentText("Reflect")
        self.assertEqual(tab.get_case_inputs().right_boundary, "Reflect")


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
        self.assertIn("Solver", titles)
        self.assertIn("Output Options", titles)
        self.assertEqual(titles.index("Output Options"), titles.index("Solver") + 1)


class HiddenVtkPaintGuardTests(unittest.TestCase):
    def test_inactive_viewport_eats_paint(self):
        viewer = DummyViewer()
        guard = VtkResizeGuard(viewer)
        paint = QEvent(QEvent.Paint)
        self.assertTrue(guard.eventFilter(viewer, paint))
        viewer._viewport_active = True
        self.assertFalse(guard.eventFilter(viewer, paint))


if __name__ == "__main__":
    unittest.main()
