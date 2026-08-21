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
    def test_200kg_tnt_step_matches_charge_radius_and_jwl_b(self):
        mass = 200.0
        rho = 1630.0
        domain = 20.0
        p_atm = 101325.0
        p_he = 3.23e9
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
        self.assertAlmostEqual(float(xs[0]), 0.0)
        self.assertAlmostEqual(float(xs[-1]), 20.0)
        self.assertAlmostEqual(float(xs[1]), charge_r, places=5)
        self.assertAlmostEqual(float(ys[0]), 3.23e9 - 101325.0, delta=1.0)
        self.assertAlmostEqual(float(ys[-1]), 0.0)
        self.assertFalse(tab._live_graph)

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
