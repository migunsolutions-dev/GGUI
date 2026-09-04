"""Offscreen tests for opt-in UI Review Mode."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui_preview import PreviewStateError, apply_preview_state, disable_cfd_actions
from ui_review_mode import (
    UIReviewController,
    attach_ui_review,
    repo_root,
)


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class MiniWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini")
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        self.tabs = QTabWidget()
        self.page_mesh = QWidget()
        self.page_exec = QWidget()
        mesh_layout = QVBoxLayout(self.page_mesh)
        self.grp_domain = QGroupBox("Domain")
        self.grp_domain.setObjectName("domainGroup")
        self.grp_charge = QGroupBox("Charge")
        self.grp_charge.setObjectName("chargeGroup")
        mesh_layout.addWidget(self.grp_domain)
        mesh_layout.addWidget(self.grp_charge)
        exec_layout = QVBoxLayout(self.page_exec)
        self.btn = QPushButton("Initialise Model")
        self.btn.setObjectName("btnInitialize")
        self.btn2 = QPushButton("exact END")
        self.btn2.setObjectName("btnExactEnd")
        exec_layout.addWidget(self.btn)
        exec_layout.addWidget(self.btn2)
        self.tabs.addTab(self.page_mesh, "Mesh")
        self.tabs.addTab(self.page_exec, "Execution")
        root.addWidget(self.tabs)
        self.clicks = []
        self.btn.clicked.connect(lambda: self.clicks.append("btn"))
        self.btn2.clicked.connect(lambda: self.clicks.append("btn2"))
        self.resize(640, 480)


class UIReviewModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.win = MiniWindow()
        self.win.show()
        self.app.processEvents()
        self.controller = UIReviewController(self.win, output_dir=self.tmp.name)

    def tearDown(self):
        if self.controller.enabled:
            self.controller.disable()
        self.win.close()
        self.tmp.cleanup()

    def test_review_mode_disabled_by_default(self):
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.overlay_widgets(), [])
        self.assertFalse(os.path.isfile(os.path.join(self.tmp.name, "selection.json")))
        self.assertFalse(os.path.isfile(os.path.join(self.tmp.name, "current.png")))

    def test_production_clicks_reach_controls_when_disabled(self):
        self.win.tabs.setCurrentWidget(self.win.page_exec)
        self.app.processEvents()
        self.win.btn.click()
        self.assertEqual(self.win.clicks, ["btn"])
        self.assertEqual(self.controller.sources, [])

    def test_region_numbering_is_deterministic(self):
        self.controller.enable()
        first = {
            self.controller.region_ids[id(self.win.grp_domain)]: "domain",
            self.controller.region_ids[id(self.win.grp_charge)]: "charge",
        }
        self.controller.disable()
        other_dir = tempfile.TemporaryDirectory()
        try:
            other = MiniWindow()
            other.show()
            self.app.processEvents()
            ctrl = UIReviewController(other, output_dir=other_dir.name)
            ctrl.enable()
            second = {
                ctrl.region_ids[id(other.grp_domain)]: "domain",
                ctrl.region_ids[id(other.grp_charge)]: "charge",
            }
            self.assertEqual(first, second)
            ctrl.disable()
            other.close()
        finally:
            other_dir.cleanup()

    def test_persisted_region_keeps_number_after_reorder(self):
        self.controller.enable()
        domain_id = self.controller.region_ids[id(self.win.grp_domain)]
        charge_id = self.controller.region_ids[id(self.win.grp_charge)]
        self.assertNotEqual(domain_id, charge_id)
        layout = self.win.page_mesh.layout()
        layout.removeWidget(self.win.grp_domain)
        layout.removeWidget(self.win.grp_charge)
        layout.addWidget(self.win.grp_charge)
        layout.addWidget(self.win.grp_domain)
        self.app.processEvents()
        self.controller.refresh_regions()
        self.assertEqual(self.controller.region_ids[id(self.win.grp_domain)], domain_id)
        self.assertEqual(self.controller.region_ids[id(self.win.grp_charge)], charge_id)

    def test_normal_click_creates_s1(self):
        self.win.tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.controller.enable()
        before = list(self.win.clicks)
        self.controller.select_widget(self.win.btn)
        self.assertEqual(len(self.controller.sources), 1)
        self.assertEqual(self.controller.sources[0]["alias"], "S1")
        self.assertEqual(self.controller.sources[0]["objectName"], "btnInitialize")
        self.assertEqual(self.win.clicks, before)

    def test_ctrl_click_creates_s2(self):
        self.win.tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.controller.enable()
        self.controller.select_widget(self.win.btn)
        self.controller.select_widget(self.win.btn2, Qt.ControlModifier)
        aliases = [item["alias"] for item in self.controller.sources]
        self.assertEqual(aliases, ["S1", "S2"])
        self.assertEqual(self.controller.sources[1]["objectName"], "btnExactEnd")

    def test_shift_click_creates_a1(self):
        self.win.tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.controller.enable()
        self.controller.select_widget(self.win.btn)
        self.controller.select_widget(self.win.btn2, Qt.ShiftModifier)
        self.assertEqual(self.controller.sources[0]["alias"], "S1")
        self.assertIsNotNone(self.controller.anchor)
        self.assertEqual(self.controller.anchor["alias"], "A1")
        self.assertEqual(self.controller.anchor["objectName"], "btnExactEnd")

    def test_escape_clears_selections(self):
        self.win.tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.controller.enable()
        self.controller.select_widget(self.win.btn)
        self.controller.select_widget(self.win.btn2, Qt.ShiftModifier)
        self.controller.clear_selections()
        self.assertEqual(self.controller.sources, [])
        self.assertIsNone(self.controller.anchor)

    def test_tabbar_selection_records_page_and_text(self):
        self.controller.enable()
        bar = self.win.tabs.tabBar()
        rect = bar.tabRect(1)
        global_pos = bar.mapToGlobal(rect.center())
        record = self.controller.select_widget(bar, Qt.NoModifier, global_pos)
        self.assertEqual(record["kind"], "tab")
        self.assertEqual(record["tab_text"], "Execution")
        self.assertEqual(record["tab_index"], 1)
        self.assertEqual(record["sibling_tabs"], ["Mesh", "Execution"])
        self.assertEqual(record["page_objectName"], "")
        self.assertIn("hierarchy", record)
        self.assertTrue(record.get("region"))

    def test_selection_json_and_screenshot(self):
        self.win.tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.controller.enable()
        self.controller.select_widget(self.win.btn)
        selection_path = os.path.join(self.tmp.name, "selection.json")
        screenshot_path = os.path.join(self.tmp.name, "current.png")
        self.assertTrue(os.path.isfile(selection_path))
        self.assertTrue(os.path.isfile(screenshot_path))
        self.assertGreater(os.path.getsize(screenshot_path), 0)
        with open(selection_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("timestamp", payload)
        self.assertIn("visible_regions", payload)
        self.assertEqual(payload["sources"][0]["alias"], "S1")
        self.assertEqual(payload["sources"][0]["objectName"], "btnInitialize")
        self.assertIn("locator", payload["sources"][0])
        self.assertIn("hierarchy", payload["sources"][0])
        self.assertNotIn("environ", payload)
        self.assertNotIn("PATH", json.dumps(payload))

    def test_overlays_do_not_change_target_geometry(self):
        self.win.tabs.setCurrentIndex(1)
        self.app.processEvents()
        geo_btn = self.win.btn.geometry()
        geo_group = self.win.grp_domain.geometry()
        self.controller.enable()
        self.controller.select_widget(self.win.btn)
        self.app.processEvents()
        self.assertEqual(self.win.btn.geometry(), geo_btn)
        self.assertEqual(self.win.grp_domain.geometry(), geo_group)
        self.assertGreater(len(self.controller.overlay_widgets()), 0)

    def test_unsupported_preview_state_is_reported(self):
        with self.assertRaises(PreviewStateError) as ctx:
            apply_preview_state(None, "1d", "initialized")
        self.assertIn("not supported", str(ctx.exception))
        with self.assertRaises(PreviewStateError):
            apply_preview_state(None, "3d", "completed")

    def test_gitignore_covers_ui_review_dir(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "_ui_review/selection.json"],
            cwd=repo_root(),
        )
        self.assertEqual(result.returncode, 0)
        listed = subprocess.run(
            ["git", "check-ignore", "-v", "_ui_review/current.png"],
            cwd=repo_root(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(listed.returncode, 0)
        self.assertIn("_ui_review/", listed.stdout)


class UIReviewProductionHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_production_window_has_no_overlays_or_review(self):
        from main_new import BlastFoamApp

        win = BlastFoamApp()
        win.show()
        self.app.processEvents()
        try:
            self.assertIsNone(win._ui_review)
            overlays = [
                widget
                for widget in win.findChildren(QWidget)
                if widget.property("ui_review_overlay")
            ]
            self.assertEqual(overlays, [])
        finally:
            win.close()

    def test_preview_review_on_2d_does_not_call_generation(self):
        from main_new import BlastFoamApp
        from ui_preview import select_tab

        tmp = tempfile.TemporaryDirectory()
        win = BlastFoamApp()
        win.show()
        self.app.processEvents()
        try:
            win.service.generate_case = mock.Mock()
            select_tab(win, "2d")
            apply_preview_state(win, "2d", "ready")
            disable_cfd_actions(win)
            controller = attach_ui_review(win, enabled=True, output_dir=tmp.name)
            self.app.processEvents()
            self.assertTrue(controller.enabled)
            self.assertGreater(len(controller.visible_region_map()), 0)
            controller.select_widget(win.tab_2d.btn_initialize)
            bar = win.tab_2d.ctrl_tabs.tabBar()
            pos = bar.mapToGlobal(bar.tabRect(0).center())
            record = controller.select_widget(bar, Qt.ControlModifier, pos)
            self.assertEqual(record["kind"], "tab")
            self.assertIn("Execution", record["tab_text"])
            status_record = controller.select_widget(win.status_bar, Qt.ControlModifier)
            self.assertEqual(status_record["class"], "SegmentedStatusBar")
            win.service.generate_case.assert_not_called()
            self.assertTrue(os.path.isfile(os.path.join(tmp.name, "current.png")))
            controller.disable()
            overlays = [
                widget
                for widget in win.findChildren(QWidget)
                if widget.property("ui_review_overlay")
            ]
            self.assertEqual(overlays, [])
        finally:
            win.close()
            tmp.cleanup()


class VtkResizeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_inactive_and_tiny_resizes_are_swallowed(self):
        from PyQt5.QtCore import QSize
        from PyQt5.QtGui import QResizeEvent

        from viewer_gl import guard_embedded_interactor

        class Probe(QWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.resize_events = 0

            def resizeEvent(self, event):
                self.resize_events += 1
                super().resizeEvent(event)

        class DummyViewer(QWidget):
            def __init__(self):
                super().__init__()
                self._shutdown = False
                self._viewport_active = False
                self.child = Probe(self)
                guard_embedded_interactor(self.child, self)

        viewer = DummyViewer()
        before = viewer.child.resize_events
        self.app.sendEvent(viewer.child, QResizeEvent(QSize(120, 70), QSize(80, 60)))
        self.assertEqual(viewer.child.resize_events, before)

        viewer._viewport_active = True
        self.app.sendEvent(viewer.child, QResizeEvent(QSize(140, 80), QSize(120, 70)))
        self.assertGreater(viewer.child.resize_events, before)

        active_count = viewer.child.resize_events
        self.app.sendEvent(viewer.child, QResizeEvent(QSize(1, 1), QSize(140, 80)))
        self.assertEqual(viewer.child.resize_events, active_count)
        viewer.close()


if __name__ == "__main__":
    unittest.main()
