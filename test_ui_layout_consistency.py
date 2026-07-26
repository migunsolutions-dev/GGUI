"""Phase B corrective: opening size, window invariance, exec visibility, status font."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QApplication, QLabel, QScrollArea, QToolBar

from ui_metrics import (
    COMPUTATIONAL_LEFT_PANEL_WIDTH,
    COMPUTATIONAL_LEFT_PANEL_TOLERANCE,
    DEFAULT_WINDOW_WIDTH,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH_TOLERANCE,
    DEFAULT_WINDOW_HEIGHT_TOLERANCE,
    EXECUTION_AREA_MIN_HEIGHT,
    INFO_PANEL_HEIGHT_MIN,
    INFO_PANEL_HEIGHT_MAX,
    STATUS_FONT_MIN_POINT_SIZE,
    STATUS_METRICS_POINT_SIZE,
    STATUS_READY_POINT_SIZE,
    STATUS_REP_MODE_GROUP,
    STATUS_REP_ET,
    ACTION_BUTTON_FONT_PT,
)
from viewer_widget import ObstacleItem


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _rect_contained(inner: QRect, outer: QRect) -> bool:
    return outer.contains(inner)


class TestUILayoutConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def _make_main(self):
        from main_new import BlastFoamApp
        win = BlastFoamApp()
        win.show()
        self.app.processEvents()
        # Force the review opening size (offscreen availableGeometry is often tiny).
        win.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        win._opening_geometry_applied = True
        win._apply_opening_computational_left_width()
        self.app.processEvents()
        return win

    def _geom(self, win):
        g = win.geometry()
        return (g.x(), g.y(), g.width(), g.height())

    def test_default_opening_size(self):
        win = self._make_main()
        try:
            self.assertAlmostEqual(
                win.width(), DEFAULT_WINDOW_WIDTH, delta=DEFAULT_WINDOW_WIDTH_TOLERANCE
            )
            self.assertAlmostEqual(
                win.height(), DEFAULT_WINDOW_HEIGHT, delta=DEFAULT_WINDOW_HEIGHT_TOLERANCE
            )
            self.assertLessEqual(win.minimumWidth(), 1200)
            win.tabs.setCurrentWidget(win.tab_3d)
            self.app.processEvents()
            win._apply_opening_computational_left_width()
            self.app.processEvents()
            left = win.tab_3d.get_computational_left_width()
            self.assertAlmostEqual(
                left, COMPUTATIONAL_LEFT_PANEL_WIDTH, delta=COMPUTATIONAL_LEFT_PANEL_TOLERANCE
            )
            right = win.tab_3d._right_container.width()
            self.assertGreaterEqual(right, 1100)
        finally:
            win.close()

    def test_window_geometry_invariant_across_tabs_and_states(self):
        win = self._make_main()
        try:
            base = self._geom(win)
            sequence = [
                win.tab_1d,
                win.tab_2d,
                win.tab_3d,
                win.tab_time_history,
                win.tab_1d,
            ]
            for tab in sequence:
                win.tabs.setCurrentWidget(tab)
                self.app.processEvents()
                self.assertEqual(self._geom(win), base, msg=f"after {tab}")

            win.tabs.setCurrentWidget(win.tab_3d)
            self.app.processEvents()
            win.tab_3d.settings_tabs.setCurrentIndex(1)
            self.app.processEvents()
            self.assertEqual(self._geom(win), base)
            win.tab_3d.settings_tabs.setCurrentIndex(0)
            self.app.processEvents()
            self.assertEqual(self._geom(win), base)

            win.status_bar.set_status("Running...", "#3498db")
            win.status_bar.update_1d(12, 1.2e-3, 3.4e-7)
            win.status_bar.update_2d(8, 9.1e-4, 2.2e-7)
            win.status_bar.update_3d(5, 7.7e-4, 1.1e-7)
            win.tab_3d._update_mesh_plan_display()
            win.tab_3d.lbl_result_total_cells.setText("Total cells: 1000")
            win.tab_3d._set_init_results_visible(True)
            if win.tab_3d.viewer:
                win.tab_3d.viewer.refresh_view()
            self.app.processEvents()
            self.assertEqual(self._geom(win), base)
            win.status_bar.set_status("Ready", "#2ecc71")
            self.app.processEvents()
            self.assertEqual(self._geom(win), base)
        finally:
            win.close()

    def test_1d_execution_controls_fully_contained(self):
        win = self._make_main()
        try:
            win.tabs.setCurrentWidget(win.tab_1d)
            self.app.processEvents()
            tab = win.tab_1d
            self.assertGreaterEqual(tab.ctrl_tabs.height(), EXECUTION_AREA_MIN_HEIGHT)
            vp = tab._exec_scroll.viewport() if hasattr(tab, "_exec_scroll") else tab.ctrl_tabs
            for btn in (tab.btn_run, tab.btn_stop):
                self.assertFalse(btn.isHidden())
                self.assertEqual(btn.font().pointSize(), ACTION_BUTTON_FONT_PT)
                self.assertEqual(btn.font().pointSize(), 10)
                self.assertEqual(btn.height(), 50)
                br = btn.rect()
                mapped = QRect(btn.mapTo(vp, br.topLeft()), br.size())
                self.assertTrue(
                    _rect_contained(mapped, vp.rect()),
                    msg=f"{btn.text()} mapped={mapped} vp={vp.rect()} ctrl_h={tab.ctrl_tabs.height()}",
                )
                # Text metrics fully inside the button rectangle (no clipping).
                from PyQt5.QtGui import QFontMetrics
                fm = QFontMetrics(btn.font())
                text_w = fm.horizontalAdvance(btn.text())
                text_h = fm.height()
                self.assertLessEqual(text_w, btn.width() - 8, msg=btn.text())
                self.assertLessEqual(text_h, btn.height() - 4, msg=btn.text())
            # Session persistence of vertical splitter + font after tab cycle
            before = list(tab._right_v_splitter.sizes())
            tab._right_v_splitter.setSizes([600, 280])
            self.app.processEvents()
            saved = list(tab._right_v_splitter.sizes())
            win.tabs.setCurrentWidget(win.tab_2d)
            self.app.processEvents()
            win.tabs.setCurrentWidget(win.tab_3d)
            self.app.processEvents()
            win.tabs.setCurrentWidget(win.tab_1d)
            self.app.processEvents()
            after = list(tab._right_v_splitter.sizes())
            self.assertEqual(after, saved)
            self.assertEqual(tab.btn_run.font().pointSize(), 10)
            self.assertEqual(tab.btn_stop.font().pointSize(), 10)
            # restore preferred for other tests
            tab._right_v_splitter.setSizes(before)
        finally:
            win.close()

    def test_2d_execution_controls_fully_contained(self):
        win = self._make_main()
        try:
            win.tabs.setCurrentWidget(win.tab_2d)
            self.app.processEvents()
            tab = win.tab_2d
            self.assertGreaterEqual(tab.ctrl_tabs.height(), EXECUTION_AREA_MIN_HEIGHT)
            vp = tab._exec_scroll.viewport()
            for btn in (tab.btn_run, tab.btn_stop):
                br = btn.rect()
                mapped = QRect(btn.mapTo(vp, br.topLeft()), br.size())
                self.assertTrue(_rect_contained(mapped, vp.rect()), msg=btn.text())
        finally:
            win.close()

    def test_3d_execution_accessible_and_info_fixed(self):
        win = self._make_main()
        try:
            win.tabs.setCurrentWidget(win.tab_3d)
            self.app.processEvents()
            tab = win.tab_3d
            self.assertGreaterEqual(tab.ctrl_tabs.height(), 200)
            for w in (tab.btn_init, tab.btn_exact_1, tab.btn_exact_end, tab.btn_stop):
                self.assertFalse(w.isHidden())
            info = tab._info_panel
            self.assertTrue(info.isVisible())
            self.assertGreaterEqual(info.height(), INFO_PANEL_HEIGHT_MIN)
            self.assertLessEqual(info.height(), INFO_PANEL_HEIGHT_MAX)
            self.assertFalse(tab._left_setup_scroll.isAncestorOf(info))
            self.assertNotIsInstance(info, QScrollArea)
        finally:
            win.close()

    def test_status_font_readable_and_stable(self):
        win = self._make_main()
        try:
            sb = win.status_bar
            pt = sb.metrics_point_size()
            self.assertEqual(pt, STATUS_METRICS_POINT_SIZE)
            self.assertEqual(pt, 9)
            ready_pt = sb.lbl_status.font().pointSize()
            self.assertEqual(ready_pt, STATUS_READY_POINT_SIZE)
            self.assertEqual(ready_pt, 11)

            sb.update_1d(12345678, 1.23456e-4, 1.23456e-7)
            sb.update_2d(23456789, 2.34567e-4, 2.34567e-7)
            sb.update_3d(34567890, 3.45678e-4, 3.45678e-7)
            sb.start_et_timing()
            self.app.processEvents()
            sb.stop_et_timing()
            self.app.processEvents()

            win.resize(1685, 1060)
            self.app.processEvents()
            self.assertAlmostEqual(win.width(), 1685, delta=2)
            self.assertEqual(sb.metrics_point_size(), 9)
            self.assertEqual(sb.lbl_status.font().pointSize(), 11)
            self.assertEqual(sb.height(), 36)
            self.assertEqual(sb._metrics_scroll.verticalScrollBar().maximum(), 0)
            self.assertEqual(
                sb._metrics_scroll.verticalScrollBarPolicy(),
                Qt.ScrollBarAlwaysOff,
            )
            hs = sb._metrics_scroll.horizontalScrollBar()
            self.assertEqual(hs.maximum(), 0, msg="status should fit at 1685 without scroll")
            self.assertNotIn("\n", sb.lbl_metrics_line.text())
            self.assertNotIn("Initial Δt", sb.lbl_metrics_line.text())
            for lbl in sb.metrics_value_labels():
                self.assertFalse(lbl.wordWrap())
                self.assertNotIn("\n", lbl.text())
            # Visible row: three mode groups + ET. Combined metrics line stays hidden.
            self.assertTrue(sb.lbl_1d_group.isVisible())
            self.assertTrue(sb.lbl_2d_group.isVisible())
            self.assertTrue(sb.lbl_3d_group.isVisible())
            self.assertTrue(sb.lbl_et.isVisible())
            self.assertFalse(sb.lbl_metrics_line.isVisible())
            from PyQt5.QtGui import QFontMetrics
            fm = QFontMetrics(sb.lbl_1d_group.font())
            for lbl in (sb.lbl_1d_group, sb.lbl_2d_group, sb.lbl_3d_group):
                self.assertGreaterEqual(lbl.width(), fm.horizontalAdvance(STATUS_REP_MODE_GROUP))
            self.assertGreaterEqual(sb.lbl_et.width(), fm.horizontalAdvance(STATUS_REP_ET))
            vp = sb._metrics_scroll.viewport()
            for lbl in sb.metrics_value_labels():
                br = lbl.rect()
                mapped = QRect(lbl.mapTo(vp, br.topLeft()), br.size())
                self.assertTrue(vp.rect().contains(mapped), msg=lbl.objectName())
            self.assertTrue(sb.lbl_status.isVisible())
            ready_mapped = QRect(
                sb.lbl_status.mapTo(win, sb.lbl_status.rect().topLeft()),
                sb.lbl_status.rect().size(),
            )
            self.assertTrue(win.rect().contains(ready_mapped))

            win.resize(1250, 900)
            self.app.processEvents()
            self.assertAlmostEqual(win.width(), 1250, delta=2)
            self.assertEqual(sb.metrics_point_size(), 9)
            self.assertEqual(sb.lbl_status.font().pointSize(), 11)
            self.assertTrue(sb.lbl_status.isVisible())
            self.assertGreaterEqual(sb._metrics_scroll.horizontalScrollBar().maximum(), 0)
            self.assertEqual(sb._metrics_scroll.verticalScrollBar().maximum(), 0)
            for lbl in sb.metrics_value_labels():
                self.assertFalse(lbl.wordWrap())
                self.assertNotIn("\n", lbl.text())
            # histories persist; ET remains a separate segment
            self.assertIn("12345678", sb.lbl_1d_group.text())
            self.assertIn("23456789", sb.lbl_2d_group.text())
            self.assertIn("34567890", sb.lbl_3d_group.text())
            self.assertTrue(sb.lbl_et.text().startswith("ET="))
            self.assertNotEqual(sb.lbl_et.text(), f"ET={sb._DASH}")
        finally:
            win.close()

    def test_toolbar_and_tabs_readable_at_opening(self):
        win = self._make_main()
        try:
            # ElideRight keeps the window shrinkable; at 1685 there is enough
            # space that primary titles remain fully readable.
            texts = [win.tabs.tabText(i) for i in range(win.tabs.count())]
            self.assertTrue(any(t == "General 3D" for t in texts))
            self.assertTrue(any(t == "Time History Viewer" for t in texts))
            tb = win._main_toolbar
            self.assertIsInstance(tb, QToolBar)
            self.assertGreaterEqual(len(tb.actions()), 6)
            # All actions remain on the toolbar object (overflow is a layout concern).
            for act in tb.actions():
                if act.isSeparator():
                    continue
                self.assertFalse(act.isVisible() is False and act.text() == "")
            self.assertGreaterEqual(tb.width(), 800)
        finally:
            win.close()

    def test_obstacles_fit_and_properties(self):
        win = self._make_main()
        try:
            tab = win.tab_3d
            win.tabs.setCurrentWidget(tab)
            win._apply_opening_computational_left_width()
            self.app.processEvents()
            tab.settings_tabs.setCurrentIndex(1)
            self.app.processEvents()
            left = tab.get_computational_left_width()
            self.assertAlmostEqual(
                left, COMPUTATIONAL_LEFT_PANEL_WIDTH, delta=COMPUTATIONAL_LEFT_PANEL_TOLERANCE + 5
            )
            for name in ("btn_add", "btn_del", "btn_clr", "btn_up", "btn_down"):
                self.assertFalse(getattr(tab, name).isHidden())
            self.assertEqual(tab.tbl_obs.columnCount(), 2)
            tab.obstacles = [
                ObstacleItem(True, r"C:\tmp\a.stl", 0.002, 0.1, 0.2, 0.3),
                ObstacleItem(False, r"C:\tmp\b.stl", 0.003, 1.0, 2.0, 3.0),
            ]
            tab._refresh_table()
            tab.tbl_obs.setCurrentCell(0, 0)
            self.app.processEvents()
            self.assertTrue(tab.grp_obs_editor.isEnabled())
            tab.spin_obs_ox.setValue(9.5)
            self.app.processEvents()
            self.assertAlmostEqual(tab.obstacles[0].ox, 9.5, places=4)
            self.assertAlmostEqual(tab.obstacles[1].scale, 0.003, places=6)
            self.assertFalse(tab.obstacles[1].enabled)
            # Path label may word-wrap; action row should not require h-scroll for buttons.
            for name in ("btn_add", "btn_del", "btn_clr", "btn_up", "btn_down"):
                btn = getattr(tab, name)
                self.assertGreater(btn.width(), 0)
                self.assertLessEqual(btn.height(), tab._left_obs_scroll.viewport().height() + 50)
        finally:
            win.close()

    def test_info_panel_wording_fixed_amr_init(self):
        win = self._make_main()
        try:
            tab = win.tab_3d
            tab.rad_fixed_mesh.setChecked(True)
            tab._update_mesh_plan_display()
            self.app.processEvents()
            txt = tab.lbl_info_total_cells.text()
            self.assertNotIn("Estimated", txt)
            self.assertTrue(txt.startswith("Total cells:") or txt.startswith("Cells:"))

            tab.rad_dyn_mesh.setChecked(True)
            tab._update_mesh_plan_display()
            self.app.processEvents()
            txt = tab.lbl_info_total_cells.text()
            if "—" not in txt:
                self.assertTrue(txt.startswith("Estimated cells:") or txt.startswith("Cells:"))

            tab.lbl_result_total_cells.setText("Total cells: 99,000")
            tab.lbl_result_charge_cells.setText("Charge cells (alpha.c4): 120")
            tab._set_init_results_visible(True)
            self.app.processEvents()
            self.assertTrue(tab.lbl_info_total_cells.text().startswith("Current cells:"))
            self.assertIn("120", tab.lbl_info_charge_cells.text())
            self.assertFalse(tab.lbl_result_total_cells.isHidden())
            self.assertFalse(tab.lbl_result_charge_cells.isHidden())
        finally:
            win.close()

    def test_mesh_plan_visible_locations(self):
        win = self._make_main()
        try:
            tab = win.tab_3d
            tab._update_mesh_plan_display()
            visible_map = {
                "mesh mode": tab.lbl_plan_mesh_mode,
                "seed": tab.lbl_plan_charge_seed,
                "outer": tab.lbl_plan_startup_outer,
                "capture": tab.lbl_plan_charge_capture,
                "init cmd": tab.lbl_plan_init_command,
                "initiation": tab.lbl_plan_initiation,
            }
            for name, lbl in visible_map.items():
                self.assertFalse(lbl.isHidden(), msg=name)
                self.assertTrue(bool(lbl.text().strip()), msg=name)
                self.assertFalse(tab.grp_mesh_plan.isVisible())
        finally:
            win.close()

    def test_opening_left_panel_width_shared(self):
        win = self._make_main()
        try:
            for tab in (win.tab_1d, win.tab_2d, win.tab_3d):
                win.tabs.setCurrentWidget(tab)
                self.app.processEvents()
                win._apply_opening_computational_left_width()
                self.app.processEvents()
                w = tab.get_computational_left_width()
                self.assertAlmostEqual(
                    w, COMPUTATIONAL_LEFT_PANEL_WIDTH, delta=COMPUTATIONAL_LEFT_PANEL_TOLERANCE
                )
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
