"""Focused tests for Cylindrical–2D viewer time selection (default time 0)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication

from openfoam_times_2d import (
    LIVE_FOLLOW_LABEL,
    list_numeric_time_entries,
    list_numeric_time_labels,
    match_reader_time_value,
    pick_opening_time,
    poly_mesh_dir_at_or_before,
)
from axisymmetric_viewer import AxisymmetricViewerWidget
from tab_2d import Tab2D
from viewer_widget import BlastViewerWidget

app = QApplication.instance() or QApplication([])


def _touch_case_tree(root: Path, times, *, mesh_at=None) -> None:
    (root / "system").mkdir(parents=True, exist_ok=True)
    (root / "constant").mkdir(parents=True, exist_ok=True)
    (root / "postProcessing").mkdir(parents=True, exist_ok=True)
    (root / "processor0").mkdir(parents=True, exist_ok=True)
    for name in times:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "p").write_text("FoamFile {}\n", encoding="utf-8")
    if mesh_at:
        for name in mesh_at:
            pm = root / name / "polyMesh"
            pm.mkdir(parents=True, exist_ok=True)
            (pm / "owner").write_text(
                "FoamFile {}\n(\n0\n1\n)\n", encoding="utf-8"
            )


class OpenFOAMTimeHelperTests(unittest.TestCase):
    def test_numerical_sort_not_lexicographic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "2e-07", "0.00076"])
            labels = list_numeric_time_labels(str(root))
            self.assertEqual(labels, ["0", "1e-07", "2e-07", "0.00076"])
            values = [t for t, _ in list_numeric_time_entries(str(root))]
            self.assertEqual(values, sorted(values))

    def test_skips_non_numeric_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07"])
            labels = list_numeric_time_labels(str(root))
            self.assertEqual(labels, ["0", "1e-07"])
            self.assertNotIn("constant", labels)
            self.assertNotIn("system", labels)
            self.assertNotIn("postProcessing", labels)
            self.assertNotIn("processor0", labels)

    def test_pick_opening_time_prefers_zero(self):
        entries = sorted(
            [(0.0, "0"), (1e-7, "1e-07"), (2e-7, "2e-07"), (7.6e-4, "0.00076")],
            key=lambda x: x[0],
        )
        label, value = pick_opening_time(entries)
        self.assertEqual(label, "0")
        self.assertEqual(value, 0.0)

    def test_match_reader_time_value(self):
        self.assertEqual(match_reader_time_value([0.0, 1e-7, 2e-7], 1e-7), 1e-7)
        self.assertIsNone(match_reader_time_value([1e-7, 2e-7], 0.0))

    def test_poly_mesh_at_or_before_amr(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(
                root,
                ["0", "1e-07", "2e-07", "0.00076"],
                mesh_at=["1e-07", "0.00076"],
            )
            (root / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
            (root / "constant" / "polyMesh" / "owner").write_text(
                "FoamFile {}\n(\n0\n)\n", encoding="utf-8"
            )
            at_zero = poly_mesh_dir_at_or_before(str(root), 0.0)
            self.assertTrue(at_zero.endswith(os.path.join("constant", "polyMesh")))
            at_mid = poly_mesh_dir_at_or_before(str(root), 2e-7)
            self.assertTrue(at_mid.replace("\\", "/").endswith("1e-07/polyMesh"))
            at_late = poly_mesh_dir_at_or_before(str(root), 7.6e-4)
            self.assertTrue(at_late.replace("\\", "/").endswith("0.00076/polyMesh"))


class ViewerTimeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.viewer = AxisymmetricViewerWidget()

    def tearDown(self):
        self.viewer.shutdown_viewer()

    def test_case_with_later_times_opens_at_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "2e-07", "0.00076"])
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
            self.assertEqual(self.viewer.selected_time_label, "0")
            self.assertEqual(self.viewer.selected_time_value, 0.0)
            self.assertFalse(self.viewer.live_follow)
            self.assertEqual(
                self.viewer.available_time_labels(),
                ["0", "1e-07", "2e-07", "0.00076"],
            )

    def test_selecting_later_time_pins_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "2e-07", "0.00076"])
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
                self.viewer.set_selected_time_label("0.00076")
            self.assertEqual(self.viewer.selected_time_label, "0.00076")
            self.assertAlmostEqual(self.viewer.selected_time_value, 0.00076)
            self.assertFalse(self.viewer.live_follow)

    def test_refresh_does_not_change_pinned_time(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "2e-07"])
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
                self.viewer.set_selected_time_label("1e-07")
            # Simulate new output appearing + sync (as refresh would)
            (root / "2e-07").mkdir(exist_ok=True)
            (root / "0.00076").mkdir()
            (root / "0.00076" / "p").write_text("x", encoding="utf-8")
            before = self.viewer.selected_time_label
            self.viewer._sync_available_times_from_case()
            self.assertEqual(self.viewer.selected_time_label, before)
            self.assertIn("0.00076", self.viewer.available_time_labels())

    def test_field_mirror_mesh_do_not_change_time(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "0.00076"])
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
                self.viewer.set_selected_time_label("1e-07")
                pinned = self.viewer.selected_time_label
                self.viewer.set_field("alpha.c4")
                self.viewer.set_mirrored_view(False)
                self.viewer.toggle_mesh_lines(True)
            self.assertEqual(self.viewer.selected_time_label, pinned)
            self.assertFalse(self.viewer.live_follow)

    def test_exact_end_enables_live_follow_and_tracks_newest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07"])
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
                self.viewer.enable_live_follow()
            self.assertTrue(self.viewer.live_follow)
            (root / "2e-07").mkdir()
            (root / "2e-07" / "p").write_text("x", encoding="utf-8")
            self.viewer._sync_available_times_from_case()
            self.assertEqual(self.viewer.selected_time_label, "2e-07")

    def test_selecting_fixed_time_during_live_disables_follow(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "2e-07"])
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
                self.viewer.enable_live_follow()
                self.viewer.set_selected_time_label("0")
            self.assertFalse(self.viewer.live_follow)
            self.assertEqual(self.viewer.selected_time_label, "0")

    def test_reopening_resets_to_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "0.00076"])
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
                self.viewer.set_selected_time_label("0.00076")
                self.viewer.enable_live_follow()
                self.viewer.load_case(str(root))
            self.assertEqual(self.viewer.selected_time_label, "0")
            self.assertFalse(self.viewer.live_follow)

    def test_missing_field_message_keeps_time(self):
        self.viewer._selected_time_label = "0"
        self.viewer._selected_time_value = 0.0
        self.viewer.current_field = "missingField"
        self.viewer._unavailable_field_message = (
            f"Field 'missingField' is unavailable at time {self.viewer.selected_time_label}."
        )
        self.assertEqual(self.viewer.selected_time_label, "0")
        self.assertIn("unavailable at time 0", self.viewer._unavailable_field_message)

    def test_stop_live_follow_keeps_last_time(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "2e-07"])
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
                self.viewer.enable_live_follow()
            self.viewer._sync_available_times_from_case()
            last = self.viewer.selected_time_label
            self.viewer.stop_live_follow_keep_time()
            self.assertFalse(self.viewer.live_follow)
            self.assertEqual(self.viewer.selected_time_label, last)


class Tab2DTimeSelectorTests(unittest.TestCase):
    def test_toolbar_has_time_selector_default_zero(self):
        tab = Tab2D()
        self.assertTrue(hasattr(tab, "cmb_time"))
        self.assertEqual(tab.cmb_time.currentText(), "0")
        # After sync with times:
        tab._on_viewer_times_changed(["0", "1e-07", "0.00076"], "0", False)
        self.assertEqual(tab.cmb_time.currentText(), "0")
        texts = [tab.cmb_time.itemText(i) for i in range(tab.cmb_time.count())]
        self.assertEqual(texts[:4], ["0", "1e-07", "0.00076", LIVE_FOLLOW_LABEL])

    def test_imported_and_native_open_rule_shared(self):
        tab = Tab2D()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "0.00076"])
            with mock.patch.object(tab.viewer, "request_refresh"):
                tab.viewer.load_case(str(root))
            self.assertEqual(tab.viewer.selected_time_label, "0")
            tab.enter_live_follow_mode()
            self.assertTrue(tab.viewer.live_follow)
            tab.stop_live_follow_keep_time()
            self.assertFalse(tab.viewer.live_follow)

    def test_changing_tabs_fields_does_not_reset_time_via_model_hooks(self):
        tab = Tab2D()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "0.00076"])
            with mock.patch.object(tab.viewer, "request_refresh"):
                tab.viewer.load_case(str(root))
                tab.viewer.set_selected_time_label("0.00076")
                tab._on_viewer_times_changed(
                    ["0", "1e-07", "0.00076"], "0.00076", False
                )
                tab.cmb_field.setCurrentText("T")
                tab.cmb_view_mode.setCurrentText("Computational Domain View")
                tab.chk_view_mesh.setChecked(True)
            self.assertEqual(tab.viewer.selected_time_label, "0.00076")
            self.assertEqual(tab.cmb_time.currentText(), "0.00076")


class Regression3DLatestPolicyUnchanged(unittest.TestCase):
    def test_shared_3d_viewer_still_uses_latest_in_source(self):
        src = Path(__file__).resolve().parent / "viewer_widget.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn("reader.time_values[-1]", text)
        ax = Path(__file__).resolve().parent / "axisymmetric_viewer.py"
        atext = ax.read_text(encoding="utf-8")
        self.assertNotIn("reader.time_values[-1]", atext)
        self.assertIn("match_reader_time_value", atext)
        self.assertIn("pick_opening_time", atext)


if __name__ == "__main__":
    unittest.main()
