"""
Regression: top-level window horizontal resize for General 3D + status bar.

Verifies OS-border shrink behavior, Execution Controls scroll isolation,
compact scroll-isolated SegmentedStatusBar (all three stage histories,
Δt notation, fit at ~1685 px), and no window expansion on calculation state.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication, QLabel

from main_new import BlastFoamApp


def _qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class Test3DWindowResizing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qapp = _qapp()

    def setUp(self):
        self.app = BlastFoamApp()
        self.app.tabs.setCurrentWidget(self.app.tab_3d)
        self.app.showNormal()
        self.app.show()
        self.qapp.processEvents()
        self.tab = self.app.tab_3d
        self.splitter = self.tab._main_splitter
        self.left = self.tab._left_setup_scroll
        self.right = self.tab._right_container
        self.exec_scroll = self.tab._exec_scroll
        self.status = self.app.status_bar

    def tearDown(self):
        self.app.close()
        self.app.deleteLater()
        self.qapp.processEvents()

    def _set_left_width(self, left_w: int = 420) -> None:
        total = sum(self.splitter.sizes())
        self.splitter.setSizes([left_w, max(50, total - left_w)])
        self.qapp.processEvents()

    def _metric_labels(self) -> list[QLabel]:
        return [
            self.status.lbl_1d_group,
            self.status._sep_1d_2d,
            self.status.lbl_2d_group,
            self.status._sep_2d_3d,
            self.status.lbl_3d_group,
            self.status._sep_3d_meta,
            self.status.lbl_3d_initial_dt,
            self.status.lbl_3d_et,
        ]

    def _assert_status_labels_readable(self):
        for lbl in self._metric_labels():
            self.assertTrue(lbl.isVisible(), f"{lbl.text()!r} not visible")
            self.assertGreater(lbl.width(), 0, f"{lbl.text()!r} width={lbl.width()}")
            self.assertTrue(bool(lbl.text().strip()), "empty status label text")
            self.assertGreater(lbl.size().height(), 0)
        st = self.status.lbl_status
        self.assertTrue(st.isVisible())
        self.assertGreater(st.width(), 0)
        self.assertTrue(bool(st.text().strip()))

    def _assert_delta_t_notation(self):
        for lbl in (
            self.status.lbl_1d_group,
            self.status.lbl_2d_group,
            self.status.lbl_3d_group,
        ):
            self.assertIn("Δt", lbl.text())
            self.assertNotRegex(lbl.text(), r"\bDT\b")
        self.assertIn("Initial Δt", self.status.lbl_3d_initial_dt.text())
        self.assertNotIn("Initial dt", self.status.lbl_3d_initial_dt.text())

    def _assert_no_overlap_with_status(self):
        """Ready/Running must stay to the right of the metrics viewport."""
        scroll = self.status._metrics_scroll
        st = self.status.lbl_status
        self.assertLessEqual(
            scroll.geometry().right(),
            st.geometry().left() + 1,
            "status label overlaps metrics scroll",
        )

    def _fill_representative_values(self):
        self.status.update_1d(step=123456, tt=0.01234, dt=1.23e-6)
        self.status.update_2d(step=234567, tt=0.02345, dt=2.34e-6)
        self.status.update_3d(step=345678, tt=0.03456, dt=3.45e-6, et=12.34)
        self.status.set_3d_initial_dt(5.56e-7)
        self.qapp.processEvents()

    def test_toplevel_resize_shrinks_right_keeps_left(self):
        self.app.resize(2048, 900)
        self.qapp.processEvents()
        self._set_left_width(420)

        left0 = self.left.width()
        right0 = self.right.width()
        win0 = self.app.width()
        self.assertGreaterEqual(win0, 1900)
        self.assertTrue(self.right.isVisible())
        self.assertTrue(self.tab.ctrl_tabs.isVisible())
        self.assertTrue(self.tab.viewer.isVisible())

        target = 1500
        self.app.resize(target, 900)
        self.qapp.processEvents()

        win1 = self.app.width()
        left1 = self.left.width()
        right1 = self.right.width()

        self.assertLessEqual(abs(win1 - target), 40, f"actual width {win1} vs requested {target}")
        self.assertLessEqual(abs(left1 - left0), 20, f"left changed {left0} -> {left1}")
        win_delta = win0 - win1
        right_delta = right0 - right1
        self.assertGreater(right_delta, win_delta * 0.85, f"right_delta={right_delta} win_delta={win_delta}")
        self.assertTrue(self.right.isVisible() and self.right.width() > 50)

        self.app.resize(1250, 900)
        self.qapp.processEvents()
        win2 = self.app.width()
        left2 = self.left.width()
        self.assertLessEqual(abs(win2 - 1250), 40)
        self.assertLessEqual(abs(left2 - left0), 20)
        self.assertLess(win2, win1)
        self.assertLess(self.right.width(), right1)

    def test_execution_controls_horizontal_scroll_when_narrow(self):
        self.app.resize(2048, 900)
        self.qapp.processEvents()
        self._set_left_width(420)

        inner = self.exec_scroll.widget()
        self.assertIsNotNone(inner)
        inner_need = inner.minimumSizeHint().width()
        self.assertGreater(inner_need, 800)

        self.app.resize(1250, 900)
        self.qapp.processEvents()
        viewport_w = self.exec_scroll.viewport().width()
        self.assertLess(viewport_w, inner_need)
        self.assertLess(self.exec_scroll.width(), inner_need)
        hs = self.exec_scroll.horizontalScrollBar()
        self.assertGreater(hs.maximum(), 0)
        self.assertLess(self.app.minimumSizeHint().width(), inner_need)

    def test_calculation_like_transition_does_not_expand(self):
        self.app.resize(1500, 900)
        self.qapp.processEvents()
        self._set_left_width(420)

        win0 = self.app.width()
        left0 = self.left.width()
        right0 = self.right.width()
        sizes0 = list(self.splitter.sizes())

        self.app.status_bar.update_3d(step=999999, tt=0.123456, dt=1.234e-6, et=99.99)
        self.app.status_bar.set_3d_initial_dt(1.234567e-6)
        self.app.status_bar.set_status("Running...", "#f39c12")
        self.qapp.processEvents()

        viewer = self.tab.viewer
        viewer.is_simulating = True
        tmp = Path(tempfile.mkdtemp(prefix="ggui_resize_sim_"))
        try:
            try:
                viewer.load_case(str(tmp))
            except Exception:
                viewer.is_simulating = True
            viewer.refresh_view()
            self.qapp.processEvents()

            self.assertLessEqual(abs(self.app.width() - win0), 20)
            self.assertLessEqual(abs(self.left.width() - left0), 20)
            self.assertLessEqual(self.right.width(), right0 + 20)
            sizes1 = list(self.splitter.sizes())
            self.assertLessEqual(abs(sizes1[0] - sizes0[0]), 20)
            self.assertTrue(self.right.isVisible())
            self.assertTrue(self.tab.viewer.isVisible())
            self.assertTrue(self.tab.ctrl_tabs.isVisible())
            self.assertTrue(self.tab.btn_exact_1.isVisible())
            self.assertTrue(self.tab.btn_exact_end.isVisible())
            self.assertTrue(self.tab.spin_cycle_write.isVisible())
            self._assert_status_labels_readable()
            self.assertIn("Running", self.status.lbl_status.text())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            viewer.is_simulating = False

    def test_status_bar_fits_at_2048_and_1685_without_scroll(self):
        for width in (2048, 1685):
            with self.subTest(width=width):
                self.app.resize(width, 900)
                self.qapp.processEvents()
                self._set_left_width(420)
                self._fill_representative_values()
                self.status.set_status("Running...", "#f39c12")
                self.qapp.processEvents()

                self.assertLessEqual(abs(self.app.width() - width), 40)
                self._assert_status_labels_readable()
                self._assert_delta_t_notation()
                self._assert_no_overlap_with_status()
                # All three modes + Initial Δt + ET must appear in the metrics strip
                joined = " | ".join(lbl.text() for lbl in self._metric_labels())
                for token in ("1D:", "2D:", "3D:", "Step", "Tt", "Δt", "Initial Δt", "ET:"):
                    self.assertIn(token, joined)
                hs = self.status._metrics_scroll.horizontalScrollBar()
                self.assertEqual(
                    hs.maximum(),
                    0,
                    f"status hscroll at {width}px: content={self.status._metrics_widget.width()} "
                    f"viewport={self.status._metrics_scroll.viewport().width()}",
                )
                self.assertTrue(self.status.lbl_status.isVisible())
                self.assertGreater(self.status.lbl_status.width(), 0)
                self.assertIn("Running", self.status.lbl_status.text())

    def test_status_bar_scrolls_when_narrow_without_clamping_window(self):
        self._fill_representative_values()
        for width in (1500, 1250, 1100):
            with self.subTest(width=width):
                self.app.resize(width, 900)
                self.qapp.processEvents()
                self._set_left_width(420)
                self.qapp.processEvents()

                self.assertLessEqual(abs(self.app.width() - width), 40)
                self._assert_status_labels_readable()
                self.assertTrue(self.status.lbl_status.isVisible())
                self.assertGreater(self.status.lbl_status.width(), 0)

                scroll = self.status._metrics_scroll
                metrics_need = self.status._metrics_widget.width()
                viewport_w = scroll.viewport().width()
                hs = scroll.horizontalScrollBar()
                self.assertLess(self.app.minimumSizeHint().width(), metrics_need)
                self.assertEqual(self.status.minimumSizeHint().width(), 0)
                if viewport_w < metrics_need - 2:
                    self.assertGreater(hs.maximum(), 0)
                    hs.setValue(hs.maximum())
                    self.qapp.processEvents()
                    self.assertGreater(self.status.lbl_3d_et.width(), 0)
                    hs.setValue(0)
                    self.qapp.processEvents()

    def test_long_status_text_does_not_expand_window(self):
        self.app.resize(1250, 900)
        self.qapp.processEvents()
        self._set_left_width(420)
        win0 = self.app.width()
        sizes0 = list(self.splitter.sizes())

        self.status.update_1d(step=123456789, tt=12.34567, dt=1.2345e-4)
        self.status.update_2d(step=222222222, tt=2.22222, dt=2.22e-5)
        self.status.update_3d(step=987654321, tt=0.98765, dt=9.87e-7, et=1234.56)
        self.status.set_3d_initial_dt(1.23456789e-6)
        self.status.set_status("Running...", "#f39c12")
        self.qapp.processEvents()

        self.assertLessEqual(abs(self.app.width() - win0), 20)
        self.assertLessEqual(abs(self.splitter.sizes()[0] - sizes0[0]), 20)
        self._assert_status_labels_readable()
        self._assert_delta_t_notation()
        self.assertIn("Running", self.status.lbl_status.text())

    def test_sequential_1d_2d_3d_histories_retained(self):
        self.app.resize(1685, 900)
        self.qapp.processEvents()

        self.status.update_1d(step=10, tt=0.00100, dt=1.00e-6)
        self.qapp.processEvents()
        self.assertIn("10", self.status.lbl_1d_group.text())
        self.assertIn(self.status._DASH, self.status.lbl_2d_group.text())

        self.status.update_2d(step=20, tt=0.00200, dt=2.00e-6)
        self.qapp.processEvents()
        self.assertIn("10", self.status.lbl_1d_group.text())
        self.assertIn("20", self.status.lbl_2d_group.text())
        self.assertIn(self.status._DASH, self.status.lbl_3d_group.text())

        self.status.update_3d(step=30, tt=0.00300, dt=3.00e-6, et=1.50)
        self.qapp.processEvents()
        self.assertIn("10", self.status.lbl_1d_group.text())
        self.assertIn("20", self.status.lbl_2d_group.text())
        self.assertIn("30", self.status.lbl_3d_group.text())
        self.assertIn("1.50", self.status.lbl_3d_et.text())

        # Tab switching must not reset histories
        self.app.tabs.setCurrentWidget(self.app.tab_1d)
        self.qapp.processEvents()
        self.app.tabs.setCurrentWidget(self.app.tab_2d)
        self.qapp.processEvents()
        self.app.tabs.setCurrentWidget(self.app.tab_3d)
        self.qapp.processEvents()
        self.assertIn("10", self.status.lbl_1d_group.text())
        self.assertIn("20", self.status.lbl_2d_group.text())
        self.assertIn("30", self.status.lbl_3d_group.text())
        self._assert_delta_t_notation()

    def test_update_2d_api_exists_and_is_isolated(self):
        self.status.update_2d(step=7, tt=0.5, dt=1e-5)
        self.qapp.processEvents()
        self.assertIn("7", self.status.lbl_2d_group.text())
        self.assertIn("Δt", self.status.lbl_2d_group.text())
        # 1D/3D remain at initial dashes
        self.assertIn(self.status._DASH, self.status.lbl_1d_group.text())
        self.assertIn(self.status._DASH, self.status.lbl_3d_group.text())


if __name__ == "__main__":
    unittest.main()
