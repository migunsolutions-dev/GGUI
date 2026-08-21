"""Focused tests for Cylindrical–2D viewer time selection (default time 0)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
)

from axisymmetric_2d import BOUNDARY_SLIP, DYNAMIC_MESH, FIXED_MESH
from openfoam_times_2d import (
    LIVE_FOLLOW_LABEL,
    TIME_ZERO_LABEL,
    list_numeric_time_entries,
    list_numeric_time_labels,
    make_single_time_case_view,
    match_reader_time_value,
    opening_time_entry,
    pick_opening_time,
    poly_mesh_dir_at_or_before,
    poly_mesh_dir_for_time_zero,
    remove_single_time_case_view,
)
from axisymmetric_viewer import AxisymmetricViewerWidget
from tab_2d import Tab2D, compact_spin_text
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

    def test_poly_mesh_time_zero_does_not_list_case_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(
                root,
                ["0", "1e-07", "2e-07", "0.00076"],
                mesh_at=["0"],
            )
            real_listdir = os.listdir

            def guarded(path):
                if os.path.normpath(path) == os.path.normpath(str(root)):
                    raise AssertionError("time-0 mesh lookup listed the case root")
                return real_listdir(path)

            with mock.patch("os.listdir", side_effect=guarded):
                found = poly_mesh_dir_for_time_zero(str(root))
            self.assertTrue(found.replace("\\", "/").endswith("0/polyMesh"))

    def test_single_time_view_exposes_only_requested_time(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "2e-07", "0.00076"])
            before = {
                rel.relative_to(root).as_posix(): rel.stat().st_mtime_ns
                for rel in root.rglob("*")
                if rel.is_file()
            }
            view = make_single_time_case_view(str(root), TIME_ZERO_LABEL)
            try:
                names = set(os.listdir(view))
                self.assertIn("0", names)
                self.assertIn("constant", names)
                self.assertIn("system", names)
                self.assertIn("case.foam", names)
                self.assertNotIn("1e-07", names)
                self.assertNotIn("0.00076", names)
            finally:
                remove_single_time_case_view(view)
            after = {
                rel.relative_to(root).as_posix(): rel.stat().st_mtime_ns
                for rel in root.rglob("*")
                if rel.is_file()
            }
            self.assertEqual(before, after)
            self.assertTrue((root / "0" / "p").is_file())
            self.assertTrue((root / "0.00076").is_dir())


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
            self.assertEqual(self.viewer.available_time_labels(), ["0"])
            self.viewer.ensure_time_catalog()
            self.assertEqual(
                self.viewer.available_time_labels(),
                ["0", "1e-07", "2e-07", "0.00076"],
            )

    def test_initial_load_does_not_scan_or_load_later_times(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            later = ["0"] + [f"{i}e-07" for i in range(1, 40)]
            _touch_case_tree(root, later)
            with mock.patch(
                "axisymmetric_viewer.list_numeric_time_entries",
                wraps=list_numeric_time_entries,
            ) as listed:
                with mock.patch.object(self.viewer, "request_refresh"):
                    self.viewer.load_case(str(root))
                listed.assert_not_called()
            self.assertEqual(self.viewer.selected_time_label, TIME_ZERO_LABEL)
            self.assertEqual(opening_time_entry(), (0.0, TIME_ZERO_LABEL))
            self.assertEqual(self.viewer.available_time_labels(), ["0"])
            self.assertNotIn("All", self.viewer.available_time_labels())
            self.assertNotIn("All timesteps", self.viewer.available_time_labels())

    def test_initial_load_does_not_modify_result_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "0.00076"])
            before = {
                rel.relative_to(root).as_posix(): (rel.stat().st_mtime_ns, rel.stat().st_size)
                for rel in root.rglob("*")
                if rel.is_file()
            }
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
            after = {
                rel.relative_to(root).as_posix(): (rel.stat().st_mtime_ns, rel.stat().st_size)
                for rel in root.rglob("*")
                if rel.is_file()
            }
            self.assertEqual(before, after)
            self.assertFalse((root / "case.foam").exists())

    def test_refresh_at_time_zero_does_not_scan_all_times(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "0.00076"])
            with mock.patch.object(self.viewer, "request_refresh"):
                self.viewer.load_case(str(root))
            with mock.patch(
                "axisymmetric_viewer.list_numeric_time_entries"
            ) as listed:
                with mock.patch(
                    "axisymmetric_viewer.pv.POpenFOAMReader",
                    side_effect=RuntimeError("skip vtk"),
                ):
                    self.viewer._refresh_axisymmetric_result()
                listed.assert_not_called()

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
        self.assertFalse(tab.cmb_time.isVisibleTo(tab))
        self.assertFalse(tab.lbl_time.isVisibleTo(tab))
        viewport = tab.viewer.parentWidget()
        shown = []
        for i in range(viewport.layout().count()):
            item = viewport.layout().itemAt(i)
            if item.layout() is None:
                continue
            for j in range(item.layout().count()):
                widget = item.layout().itemAt(j).widget()
                if widget is not None:
                    shown.append(widget)
                    if isinstance(widget, type(tab.lbl_time)):
                        self.assertNotEqual(widget.text(), "Time:")
        self.assertNotIn(tab.cmb_time, shown)
        self.assertNotIn(tab.lbl_time, shown)
        self.assertEqual(tab.cmb_time.currentText(), "0")
        # After sync with times:
        tab._on_viewer_times_changed(["0", "1e-07", "0.00076"], "0", False)
        self.assertEqual(tab.cmb_time.currentText(), "0")
        texts = [tab.cmb_time.itemText(i) for i in range(tab.cmb_time.count())]
        self.assertEqual(texts[:4], ["0", "1e-07", "0.00076", LIVE_FOLLOW_LABEL])
        self.assertNotIn("All", texts)
        self.assertNotIn("All timesteps", texts)

    def test_load_keeps_time_combo_at_zero_until_popup(self):
        tab = Tab2D()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch_case_tree(root, ["0", "1e-07", "0.00076"])
            with mock.patch(
                "axisymmetric_viewer.list_numeric_time_entries",
                wraps=list_numeric_time_entries,
            ) as listed:
                with mock.patch.object(tab.viewer, "request_refresh"):
                    tab.viewer.load_case(str(root))
                listed.assert_not_called()
            self.assertEqual(tab.cmb_time.currentText(), "0")
            shown = [tab.cmb_time.itemText(i) for i in range(tab.cmb_time.count())]
            self.assertEqual(shown[0], "0")
            self.assertNotIn("0.00076", shown)
            tab._ensure_time_catalog()
            shown = [tab.cmb_time.itemText(i) for i in range(tab.cmb_time.count())]
            self.assertIn("0.00076", shown)
            self.assertEqual(tab.cmb_time.currentText(), "0")

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


class Tab2DFieldSelectorTests(unittest.TestCase):
    def test_field_radios_are_in_simulation_control_not_viewport(self):
        tab = Tab2D()
        group = tab.btn_initialize.parentWidget()
        while group is not None and not isinstance(group, QGroupBox):
            group = group.parentWidget()
        self.assertIsNotNone(group)
        self.assertEqual(group.title(), "Simulation Control")
        field_box = tab._field_radios["p"].parentWidget()
        self.assertTrue(group.isAncestorOf(field_box))
        items = [
            group.layout().itemAt(i).widget()
            for i in range(group.layout().count())
        ]
        self.assertIs(items[-1], field_box)
        self.assertNotIn(tab.lbl_state, items)
        viewport = tab.viewer.parentWidget()
        self.assertFalse(viewport.isAncestorOf(tab.cmb_field))
        self.assertFalse(tab.cmb_field.isVisibleTo(tab))
        self.assertEqual(
            [radio.text() for radio in tab._field_radios.values()],
            [
                "Pressure",
                "Density",
                "Temperature",
                "Velocity",
                "Explosive fraction",
                "Refinement level",
            ],
        )
        self.assertTrue(tab._field_radios["p"].isChecked())
        self.assertEqual(tab.cmb_field.currentText(), "p")
        tab._field_radios["rho"].setChecked(True)
        self.assertEqual(tab.cmb_field.currentText(), "rho")
        tab.cmb_field.setCurrentText("T")
        self.assertTrue(tab._field_radios["T"].isChecked())
        self.assertEqual(tab.viewer.current_field, "T")
        actions = tab.btn_initialize.parentWidget()
        self.assertIsInstance(actions.layout(), QVBoxLayout)
        self.assertIs(actions.layout().itemAt(0).widget(), tab.btn_initialize)
        self.assertIs(actions.layout().itemAt(1).widget(), tab.btn_exact_end)
        self.assertIs(actions.layout().itemAt(2).widget(), tab.btn_stop)
        self.assertIs(actions.layout().itemAt(3).widget(), tab.lbl_state)
        self.assertIs(actions.layout().itemAt(4).widget(), tab.chk_log_scale)
        self.assertEqual(tab.lbl_state.maximumWidth(), 198)
        self.assertTrue(tab.btn_log.isHidden())
        self.assertIn("#3498db", tab.btn_initialize.styleSheet())
        self.assertIn("#1abc9c", tab.btn_exact_end.styleSheet())
        self.assertIn("#e67e22", tab.btn_stop.styleSheet())
        self.assertEqual(tab.btn_initialize.minimumWidth(), 198)
        self.assertEqual(tab.btn_exact_end.minimumWidth(), 198)
        self.assertEqual(tab.btn_stop.minimumWidth(), 198)
        self.assertIsNotNone(actions.layout().itemAt(5).spacerItem())
        exec_page = tab._exec_scroll.widget()
        view_box = tab.cmb_view_mode.parentWidget()
        solver = tab.spin_max_co.parentWidget()
        while solver is not None and not isinstance(solver, QGroupBox):
            solver = solver.parentWidget()
        self.assertIsNotNone(solver)
        self.assertEqual(solver.title(), "Solver Controls")
        self.assertIsInstance(solver.layout(), QGridLayout)
        grid = solver.layout()
        self.assertTrue(grid.itemAtPosition(0, 0).widget().isAncestorOf(tab.spin_end_time))
        time_row = grid.itemAtPosition(0, 0).widget()
        self.assertTrue(time_row.isAncestorOf(tab.spin_delta_t))
        self.assertIs(time_row, grid.itemAtPosition(0, 1).widget())
        initial_label = next(
            label
            for label in time_row.findChildren(QLabel)
            if label.text() == "Initial time step:"
        )
        self.assertGreaterEqual(initial_label.minimumSizeHint().width(), 1)
        self.assertEqual(initial_label.sizePolicy().horizontalPolicy(), QSizePolicy.Minimum)
        self.assertTrue(grid.itemAtPosition(1, 0).widget().isAncestorOf(tab.spin_max_co))
        self.assertTrue(grid.itemAtPosition(1, 0).widget().isAncestorOf(tab.chk_adjust))
        self.assertIs(grid.itemAtPosition(1, 0).widget(), grid.itemAtPosition(1, 1).widget())
        self.assertEqual(tab.spin_max_co.minimumWidth(), tab.spin_max_co.maximumWidth())
        self.assertGreater(tab.spin_max_co.minimumWidth(), 72)
        write_row = grid.itemAtPosition(2, 0).widget()
        write_layout = write_row.layout()
        self.assertTrue(write_row.isAncestorOf(tab.cmb_write_control))
        self.assertEqual(write_layout.stretch(0), 0)
        self.assertEqual(write_layout.stretch(1), 0)
        self.assertIsNotNone(write_layout.itemAt(write_layout.count() - 1).spacerItem())
        self.assertIsInstance(view_box, QGroupBox)
        self.assertEqual(view_box.title(), "View")
        self.assertIs(exec_page.layout().itemAt(0).widget(), group)
        self.assertIs(exec_page.layout().itemAt(1).widget(), solver)
        self.assertIs(exec_page.layout().itemAt(2).widget(), view_box)
        self.assertEqual(tab.cmb_write_control.currentText(), "RunTime")
        self.assertEqual(tab.cmb_write_control.currentData(), "adjustableRunTime")
        self.assertLess(tab.cmb_write_control.maximumWidth(), 222)
        self.assertTrue(grid.itemAtPosition(3, 0).widget().isAncestorOf(tab.spin_write_time))
        self.assertTrue(grid.itemAtPosition(3, 0).widget().isAncestorOf(tab.spin_write_steps))
        self.assertEqual(tab.lbl_write_interval.text(), "Write interval (time):")
        self.assertFalse(tab.spin_write_time.isHidden())
        self.assertTrue(tab.spin_write_steps.isHidden())
        self.assertEqual(tab.lbl_write_interval_unit.text(), "s")
        self.assertFalse(tab.lbl_write_interval_unit.isHidden())
        tab._set_combo_stored_value(tab.cmb_write_control, "timeStep")
        tab._sync_write_interval_display()
        self.assertEqual(tab.lbl_write_interval.text(), "Write interval (steps):")
        self.assertTrue(tab.spin_write_time.isHidden())
        self.assertFalse(tab.spin_write_steps.isHidden())
        self.assertTrue(tab.lbl_write_interval_unit.isHidden())
        tab._set_combo_stored_value(tab.cmb_write_control, "adjustableRunTime")
        tab._sync_write_interval_display()
        stacked = [
            view_box.layout().itemAt(i).widget()
            for i in range(3)
        ]
        self.assertEqual(
            stacked,
            [
                tab.chk_view_mirror,
                tab.chk_view_mesh,
                tab.chk_view_probes,
            ],
        )
        self.assertFalse(view_box.isAncestorOf(tab.chk_log_scale))
        self.assertTrue(tab.cmb_view_mode.isHidden())
        self.assertTrue(tab.lbl_mirror_indicator.isHidden())
        self.assertEqual(tab.chk_view_mirror.text(), "Mirrored View")
        self.assertTrue(tab.chk_view_mirror.isChecked())
        self.assertEqual(tab.cmb_view_mode.currentText(), "Mirrored View")
        tab.chk_view_mirror.setChecked(False)
        self.assertEqual(tab.cmb_view_mode.currentText(), "Computational Domain View")
        tab.chk_view_mirror.setChecked(True)
        self.assertEqual(tab.cmb_view_mode.currentText(), "Mirrored View")
        self.assertFalse(viewport.isAncestorOf(tab.cmb_view_mode))
        self.assertTrue(viewport.isAncestorOf(tab.btn_fit))
        self.assertTrue(viewport.isAncestorOf(tab.cmb_time))
        controls = viewport.layout().itemAt(0).layout()
        self.assertIs(controls.itemAt(0).widget(), tab._status_caption_host)
        self.assertIs(controls.itemAt(1).widget(), tab.btn_fit)


class Tab2DMeshModeSelectorTests(unittest.TestCase):
    def test_mesh_mode_radios_sit_below_base_cell_size(self):
        tab = Tab2D()
        group = tab.spin_cell.parentWidget()
        while group is not None and not isinstance(group, QGroupBox):
            group = group.parentWidget()
        self.assertIsNotNone(group)
        self.assertEqual(group.title(), "Domain Definition")
        self.assertTrue(group.isAncestorOf(tab.rad_fixed_mesh))
        self.assertTrue(group.isAncestorOf(tab.rad_dyn_mesh))
        self.assertEqual(tab.rad_fixed_mesh.text(), "Fixed Mesh")
        self.assertEqual(tab.rad_dyn_mesh.text(), "Dyn Mesh (AMR)")
        self.assertTrue(tab.rad_dyn_mesh.isChecked())
        self.assertEqual(tab.cmb_mesh_mode.currentText(), DYNAMIC_MESH)
        self.assertTrue(group.isAncestorOf(tab.btn_mesh_amr))
        self.assertEqual(tab.btn_mesh_amr.text(), "Mesh & AMR")
        self.assertTrue(tab.btn_mesh_amr.isEnabled())
        self.assertTrue(tab.lbl_radial_cells.isHidden())
        self.assertTrue(tab.lbl_vertical_cells.isHidden())
        self.assertFalse(tab.lbl_info_grid.isHidden())
        self.assertFalse(tab.lbl_info_charge.isHidden())
        self.assertFalse(tab.lbl_info_resolution.isHidden())
        self.assertTrue(
            tab.lbl_info_total.text().startswith("Estimated cells before initialization")
        )
        self.assertEqual(
            tab.btn_mesh_amr.maximumWidth(),
            max(438 // 2, tab.btn_mesh_amr.sizeHint().width()),
        )
        self.assertEqual(
            tab.cmb_source.maximumWidth(),
            max(355 // 2, tab.cmb_source.sizeHint().width()),
        )
        self.assertEqual(tab.input_tabs.count(), 2)
        self.assertEqual(tab.input_tabs.tabText(0), "Setup")
        self.assertEqual(tab.input_tabs.tabText(1), "Output & Probes")
        self.assertEqual(tab._mesh_dialog.windowTitle(), "Mesh & AMR")
        self.assertIsNotNone(tab.grp_seed)
        self.assertIsNotNone(tab.grp_amr)
        with mock.patch.object(tab, "_refresh_derived"):
            tab.cmb_mesh_mode.blockSignals(True)
            tab.cmb_mesh_mode.setCurrentText(FIXED_MESH)
            tab.cmb_mesh_mode.blockSignals(False)
            tab._apply_enablement()
            self.assertFalse(tab.btn_mesh_amr.isEnabled())
            self.assertFalse(tab.lbl_radial_cells.isHidden())
            self.assertFalse(tab.lbl_vertical_cells.isHidden())
            self.assertTrue(tab.lbl_info_grid.isHidden())
            self.assertTrue(tab.lbl_info_charge.isHidden())
            self.assertTrue(tab.lbl_info_resolution.isHidden())
            tab.cmb_mesh_mode.blockSignals(True)
            tab.cmb_mesh_mode.setCurrentText(DYNAMIC_MESH)
            tab.cmb_mesh_mode.blockSignals(False)
            tab._apply_enablement()
            self.assertTrue(tab.btn_mesh_amr.isEnabled())
            self.assertTrue(tab.lbl_radial_cells.isHidden())
            self.assertFalse(tab.lbl_info_grid.isHidden())
            tab._open_mesh_amr_dialog()
            self.assertTrue(tab._mesh_dialog.isVisible())
            tab._mesh_dialog.hide()
        tab.rad_fixed_mesh.setChecked(True)
        QApplication.processEvents()
        self.assertEqual(tab.cmb_mesh_mode.currentText(), FIXED_MESH)
        self.assertFalse(tab.lbl_radial_cells.isHidden())
        self.assertTrue(tab.lbl_info_grid.isHidden())
        tab.rad_dyn_mesh.setChecked(True)
        QApplication.processEvents()
        self.assertEqual(tab.cmb_mesh_mode.currentText(), DYNAMIC_MESH)
        self.assertTrue(tab.lbl_radial_cells.isHidden())


class Tab2DSetupUnitLabelTests(unittest.TestCase):
    def _unit_text(self, spin):
        row = spin.parentWidget()
        labels = [child.text() for child in row.findChildren(QLabel)]
        self.assertEqual(len(labels), 1)
        return labels[0]

    def test_setup_units_sit_right_of_spinboxes(self):
        tab = Tab2D()
        marked = (
            (tab.spin_radius, "m"),
            (tab.spin_height, "m"),
            (tab.spin_cell, "m"),
            (tab.spin_mass, "kg"),
            (tab.spin_density, "kg/m³"),
        )
        for spin, unit in marked:
            self.assertEqual(spin.suffix(), "")
            self.assertEqual(self._unit_text(spin), unit)
            self.assertEqual(spin.buttonSymbols(), QAbstractSpinBox.NoButtons)
            self.assertEqual(spin.maximumWidth(), 124)
            self.assertEqual(spin.alignment() & Qt.AlignRight, Qt.AlignRight)
        self.assertEqual(tab.spin_radius.textFromValue(1.5), "1.50")
        self.assertEqual(tab.spin_cell.textFromValue(0.05), "0.050")
        self.assertEqual(tab.spin_mass.textFromValue(1.0), "1")
        self.assertEqual(tab.spin_density.textFromValue(1630.0), "1630")
        self.assertEqual(tab.cmb_material.maximumWidth(), tab.spin_mass.maximumWidth())
        self.assertEqual(tab.cmb_shape.maximumWidth(), tab.spin_mass.maximumWidth())
        self.assertEqual(tab.cmb_bottom.currentText(), "Reflection")
        self.assertEqual(tab.cmb_bottom.currentData(), BOUNDARY_SLIP)
        self.assertEqual(tab.cmb_outer.currentText(), "Open")
        self.assertEqual(tab.cmb_outer.maximumWidth(), 174)
        self.assertEqual(tab.cmb_top.maximumWidth(), 174)
        self.assertEqual(tab.cmb_bottom.maximumWidth(), 174)
        self.assertTrue(tab.lbl_axis_lock.isHidden())
        setup_widgets = []
        layout = tab.grp_mapping.parentWidget().layout()
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                setup_widgets.append(widget)
        self.assertIs(setup_widgets[-1], tab.grp_mapping)
        self.assertEqual(tab.grp_mapping.title(), "Remap")
        self.assertNotIn(tab.grp_solver, setup_widgets)


    def test_compact_spin_text_rules(self):
        self.assertEqual(compact_spin_text(1.0, 6), "1")
        self.assertEqual(compact_spin_text(1630.0, 6), "1630")
        self.assertEqual(compact_spin_text(1.5, 6), "1.50")
        self.assertEqual(compact_spin_text(0.5, 6), "0.50")
        self.assertEqual(compact_spin_text(0.05, 6), "0.050")
        self.assertEqual(compact_spin_text(288.15, 2), "288.150")


class Regression3DLatestPolicyUnchanged(unittest.TestCase):
    def test_shared_3d_viewer_still_uses_latest_in_source(self):
        src = Path(__file__).resolve().parent / "viewer_widget.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn("reader.time_values[-1]", text)
        ax = Path(__file__).resolve().parent / "axisymmetric_viewer.py"
        atext = ax.read_text(encoding="utf-8")
        self.assertNotIn("reader.time_values[-1]", atext)
        self.assertIn("opening_time_entry", atext)
        self.assertIn("make_single_time_case_view", atext)
        self.assertNotIn("set_active_time_value", atext)


if __name__ == "__main__":
    unittest.main()
