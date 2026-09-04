"""Offscreen Qt tests for the Validation & Verification tab."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication

from models_2d import ProbePoint2D
from probes_model import ProbePoint
from tab_validation import (
    MODE_CONWEP,
    MODE_HOB,
    MODE_KB,
    MODE_NUMERICAL,
    MODE_REMAP,
    TabValidation,
)
from validation.current_run import SOURCE_CURRENT, SOURCE_MANUAL, RunSnapshot
from validation import ufc_hob


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TabValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def _tab(self, snapshot=None, gauges=None, probes2=None, probes3=None):
        tab = TabValidation()
        snap = snapshot if snapshot is not None else RunSnapshot()

        def provider():
            return snap

        tab.set_source_provider(
            context=provider,
            gauges_1d=lambda: gauges if gauges is not None else ((1.5, "G05"),),
            probes_2d=lambda: probes2 if probes2 is not None else (ProbePoint2D("P2", 2.0, 0.5),),
            probes_3d=lambda: probes3 if probes3 is not None else (ProbePoint("Q3", 1.0, 2.0, 3.0),),
        )
        tab.show()
        tab._redraw()
        return tab

    def test_opens_on_kingery_and_current_run_banner(self):
        tab = self._tab()
        self.assertEqual(tab.combo_mode.currentText(), MODE_KB)
        self.assertEqual(tab.lbl_source.text(), "Current Run")
        self.assertTrue(tab.radio_auto_points.isChecked())
        self.assertFalse(tab.grp_gauges.isVisible())
        self.assertTrue(tab.grp_sampling.isVisible())
        self.assertEqual(tab.tbl_gauges.horizontalHeaderItem(0).text(), "ID")
        self.assertEqual(tab.tbl_gauges.columnCount(), 5)

    def test_mode_switch_hides_gauges_except_kb_conwep(self):
        tab = self._tab()
        tab.combo_mode.setCurrentText(MODE_CONWEP)
        tab._on_mode_changed(MODE_CONWEP)
        self.assertTrue(tab.grp_sampling.isVisible())
        self.assertFalse(tab.grp_gauges.isVisible())
        tab.radio_user_gauges.setChecked(True)
        self.assertTrue(tab.grp_gauges.isVisible())
        tab.combo_mode.setCurrentText(MODE_HOB)
        tab._on_mode_changed(MODE_HOB)
        self.assertFalse(tab.grp_gauges.isVisible())
        tab.combo_mode.setCurrentText(MODE_REMAP)
        tab._on_mode_changed(MODE_REMAP)
        self.assertFalse(tab.grp_gauges.isVisible())
        tab.combo_mode.setCurrentText(MODE_NUMERICAL)
        tab._on_mode_changed(MODE_NUMERICAL)
        self.assertFalse(tab.grp_gauges.isVisible())
        tab.combo_mode.setCurrentText(MODE_KB)
        tab._on_mode_changed(MODE_KB)
        tab.radio_user_gauges.setChecked(True)
        self.assertTrue(tab.grp_gauges.isVisible())

    def test_missing_current_run_message(self):
        tab = self._tab(RunSnapshot())
        tab.combo_mode.setCurrentText(MODE_REMAP)
        tab._on_mode_changed(MODE_REMAP)
        tab._redraw()
        self.assertIn("required validation data", tab.lbl_status.text())

    def test_spherical_burst_is_na(self):
        tab = self._tab(RunSnapshot(mass_kg=1.0))
        tab.radio_kb_sph.setChecked(True)
        tab._redraw()
        self.assertIn("ARBRL-TR-02555", tab.lbl_status.text())

    def test_ufc_spherical_reference_is_labeled(self):
        tab = self._tab(RunSnapshot(mass_kg=1.0))
        tab.combo_kb_source.setCurrentText("UFC 3-340-02")
        tab.radio_kb_sph.setChecked(True)
        tab._redraw()
        self.assertNotIn("ARBRL-TR-02555", tab.lbl_status.text())
        labels = [t.get_text() for t in tab.plot_canvas.axes.get_legend().get_texts()] if tab.plot_canvas.axes.get_legend() else []
        self.assertTrue(any("UFC 3-340-02 Figure 2-7" in t for t in labels))

    def test_ufc_hemi_reference_is_labeled(self):
        tab = self._tab(RunSnapshot(mass_kg=1.0))
        tab.combo_kb_source.setCurrentText("UFC 3-340-02")
        tab.radio_kb_hemi.setChecked(True)
        tab._redraw()
        labels = [t.get_text() for t in tab.plot_canvas.axes.get_legend().get_texts()] if tab.plot_canvas.axes.get_legend() else []
        self.assertTrue(any("UFC 3-340-02 Figure 2-15" in t for t in labels))

    def test_add_and_clear(self):
        tab = self._tab()
        self.assertGreater(tab.tbl_gauges.rowCount(), 0)
        tab.tbl_gauges.selectRow(0)
        tab._on_add()
        self.assertEqual(len(tab._added), 1)
        tab._on_clear()
        self.assertEqual(tab._added, [])

    def test_conwep_waveform_status(self):
        tab = self._tab(RunSnapshot(mass_kg=1.0))
        tab.combo_mode.setCurrentText(MODE_CONWEP)
        tab.tbl_gauges.selectRow(0)
        tab._on_add()
        tab._redraw()
        self.assertIn("N/A", tab.lbl_status.text())
        labels = []
        if tab.plot_canvas.axes.get_legend():
            labels = [t.get_text() for t in tab.plot_canvas.axes.get_legend().get_texts()]
        self.assertTrue(any("not CONWEP" in t for t in labels) or "UFC Calc" in tab.lbl_status.text())

    def test_hob_ufc_na(self):
        tab = self._tab()
        tab.combo_mode.setCurrentText(MODE_HOB)
        tab._on_mode_changed(MODE_HOB)
        tab._redraw()
        self.assertIn("UFC 3-340-02", ufc_hob.lookup_mach_stem_height(0.0).unavailable_reason)

    def test_hob_kinds_include_ufc_figures(self):
        tab = self._tab(RunSnapshot(mass_kg=1.0, hob_m=1.0))
        tab.combo_mode.setCurrentText(MODE_HOB)
        tab._on_mode_changed(MODE_HOB)
        items = [tab.combo_hob_kind.itemText(i) for i in range(tab.combo_hob_kind.count())]
        self.assertTrue(any("Figure 2-13" in t or "Triple-Point" in t for t in items))
        self.assertTrue(any("Fig 2-7" in t for t in items))
        self.assertTrue(any("Fig 2-9" in t for t in items))
        self.assertTrue(any("Fig 2-10" in t for t in items))
        self.assertTrue(any("pressure-time" in t for t in items))
        for text in items:
            tab.combo_hob_kind.setCurrentText(text)
            tab._redraw()

    def test_manual_banner_and_reset(self):
        tab = self._tab(RunSnapshot(source=SOURCE_CURRENT, case_2d="x"))
        tab._snapshot = RunSnapshot(source=SOURCE_MANUAL, case_2d="manual")
        tab._refresh_banner()
        self.assertEqual(tab.lbl_source.text(), "Manual Result")
        tab.refresh_current_run(reset_manual=True)
        self.assertEqual(tab.lbl_source.text(), "Current Run")

    def test_mass_prefill_from_snapshot(self):
        tab = self._tab(RunSnapshot(mass_kg=12.5, material_name="C4"))
        self.assertAlmostEqual(tab.spin_kb_mass.value(), 12.5)
        self.assertAlmostEqual(tab.spin_cw_mass.value(), 12.5)
        self.assertEqual(tab.edit_cw_type.text(), "C4")
        self.assertFalse(tab.spin_kb_mass.isEnabled())

    def test_auto_elevated_2d_defaults_to_ufc_spherical(self):
        snap = RunSnapshot(
            live_mode="2d",
            mass_kg=1.0,
            hob_m=0.5,
            domain_radius_2d=1.5,
            domain_height_2d=1.5,
        )
        tab = self._tab(snap)
        tab._redraw()
        self.assertTrue(tab.radio_kb_sph.isChecked())
        self.assertTrue(str(tab.combo_kb_source.currentText()).startswith("UFC"))

    def test_auto_surface_2d_defaults_to_hemispherical(self):
        snap = RunSnapshot(
            live_mode="2d",
            mass_kg=1.0,
            hob_m=0.0,
            domain_radius_2d=1.5,
            domain_height_2d=1.5,
        )
        tab = self._tab(snap)
        tab._redraw()
        self.assertTrue(tab.radio_kb_hemi.isChecked())

    def test_automatic_points_without_manual_gauges(self):
        snap = RunSnapshot(
            live_mode="2d",
            mass_kg=1.0,
            material_name="TNT",
            hob_m=0.5,
            domain_radius_2d=1.5,
            domain_height_2d=1.5,
            domain_cell_2d=0.05,
        )
        tab = self._tab(snap)
        tab._redraw()
        self.assertTrue(tab.radio_auto_points.isChecked())
        self.assertGreater(tab.table.rowCount(), 0)
        self.assertTrue(tab.radio_kb_log.isChecked())
        self.assertEqual(tab.plot_canvas.axes.get_xscale(), "log")
        self.assertIn("Automatic points", tab.lbl_kb_info.text())

    def test_range_vs_z_and_log_linear_are_display_only(self):
        snap = RunSnapshot(
            live_mode="2d",
            mass_kg=1.0,
            hob_m=0.5,
            domain_radius_2d=1.5,
            domain_height_2d=1.5,
        )
        tab = self._tab(snap)
        tab._redraw()
        n0 = tab.table.rowCount()
        ids0 = [tab.table.item(r, 0).text() for r in range(n0)]
        tab.radio_kb_z.setChecked(True)
        tab._redraw()
        self.assertEqual([tab.table.item(r, 0).text() for r in range(tab.table.rowCount())], ids0)
        tab.radio_kb_lin.setChecked(True)
        tab._redraw()
        self.assertEqual(tab.plot_canvas.axes.get_xscale(), "linear")
        self.assertEqual([tab.table.item(r, 0).text() for r in range(tab.table.rowCount())], ids0)

    def test_user_gauges_mode_keeps_manual_add(self):
        tab = self._tab()
        tab.radio_user_gauges.setChecked(True)
        self.assertTrue(tab.grp_gauges.isVisible())
        self.assertGreater(tab.tbl_gauges.rowCount(), 0)
        tab.tbl_gauges.selectRow(0)
        tab._on_add()
        self.assertEqual(len(tab._added), 1)
        tab._on_clear()
        self.assertEqual(tab._added, [])

    def test_reference_switch_reuses_automatic_points(self):
        snap = RunSnapshot(
            live_mode="2d",
            mass_kg=1.0,
            hob_m=0.5,
            domain_radius_2d=1.5,
            domain_height_2d=1.5,
        )
        tab = self._tab(snap)
        tab._redraw()
        ids0 = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
        tab.combo_kb_source.setCurrentText("UFC 3-340-02")
        tab._redraw()
        ids1 = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
        self.assertEqual(ids0, ids1)
        self.assertTrue(ids0)
        self.assertTrue(all(t.startswith("VAL_2D_") for t in ids0))

    def test_auto_1d_checkbox_uses_val_1d_ids(self):
        snap = RunSnapshot(live_mode="1d", mass_kg=1.0, domain_radius_1d=2.0, domain_cell_1d=0.05)
        tab = self._tab(snap)
        tab.chk_auto_1d.setChecked(True)
        tab.chk_auto_2d.setChecked(False)
        tab._redraw()
        ids0 = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
        self.assertTrue(ids0)
        self.assertTrue(all(t.startswith("VAL_1D_") for t in ids0))

    def test_legacy_run_without_histories_is_na(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            snap = RunSnapshot(
                live_mode="2d",
                mass_kg=1.0,
                hob_m=0.5,
                domain_radius_2d=1.5,
                domain_height_2d=1.5,
                case_2d=td,
            )
            tab = self._tab(snap)
            tab._redraw()
            self.assertIn("VTK", tab.lbl_status.text())
            bf_col = tab.table.item(0, 4)
            self.assertIsNotNone(bf_col)
            self.assertEqual(bf_col.text(), "N/A")

    def test_current_run_change_invalidates_auto_plans(self):
        snap_a = RunSnapshot(live_mode="2d", mass_kg=1.0, hob_m=0.5, domain_radius_2d=1.5, domain_height_2d=1.5, case_2d="a")
        holder = {"snap": snap_a}

        def provider():
            return holder["snap"]

        tab = TabValidation()
        tab.set_source_provider(context=provider, gauges_1d=lambda: ((1.5, "G05"),))
        tab._redraw()
        first = [pt.point_id for p in tab._collect_auto_plans() for pt in p.points]
        holder["snap"] = RunSnapshot(live_mode="2d", mass_kg=1.0, hob_m=0.5, domain_radius_2d=1.5, domain_height_2d=1.5, case_2d="b")
        tab.refresh_current_run()
        self.assertEqual(tab._auto_plans, [])
        tab._redraw()
        second = [pt.point_id for p in tab._collect_auto_plans() for pt in p.points]
        self.assertTrue(first)
        self.assertTrue(second)

    def test_numerical_empty_run_is_na_not_3d(self):
        tab = self._tab(RunSnapshot())
        tab.combo_mode.setCurrentText(MODE_NUMERICAL)
        tab._on_mode_changed(MODE_NUMERICAL)
        tab._redraw()
        dim = tab.table.item(0, 1)
        self.assertIsNotNone(dim)
        self.assertEqual(dim.text(), "N/A")

    def test_current_run_1d_does_not_show_idle_2d_as_bf(self):
        snap = RunSnapshot(
            live_mode="1d",
            mass_kg=1.0,
            domain_radius_1d=2.0,
            domain_radius_2d=1.5,
            domain_height_2d=1.5,
            hob_m=0.5,
        )
        tab = self._tab(snap)
        tab._redraw()
        self.assertTrue(tab.chk_show_1d.isChecked())
        self.assertFalse(tab.chk_show_2d.isChecked())
        self.assertFalse(tab.chk_show_3d.isChecked())
        ids0 = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
        self.assertTrue(ids0)
        self.assertTrue(all(t.startswith("VAL_1D_") for t in ids0))
        sources = [tab.table.item(r, 2).text() for r in range(tab.table.rowCount())]
        self.assertTrue(all(s == "Planned" for s in sources))
        self.assertIn("Planned validation points", tab.lbl_status.text())

    def test_multidim_selection_overlays_1d_and_2d(self):
        snap = RunSnapshot(
            live_mode="1d",
            mass_kg=1.0,
            domain_radius_1d=2.0,
            domain_radius_2d=1.5,
            domain_height_2d=1.5,
            hob_m=0.5,
        )
        tab = self._tab(snap)
        tab.chk_show_2d.setChecked(True)
        tab._redraw()
        ids0 = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
        self.assertTrue(any(t.startswith("VAL_1D_") for t in ids0))
        self.assertTrue(any(t.startswith("VAL_2D_") for t in ids0))
        dims = {tab.table.item(r, 1).text() for r in range(tab.table.rowCount())}
        self.assertEqual(dims, {"1D", "2D"})

    def test_3d_user_gauge_standoff_and_hemi_na(self):
        from probes_model import ProbePoint

        snap = RunSnapshot(
            live_mode="3d",
            mass_kg=1.0,
            charge_center_3d=(1.0, 2.0, 3.0),
        )
        tab = self._tab(snap, probes3=(ProbePoint("G01", 1.0, 2.0, 6.0),))
        tab.chk_show_3d.setChecked(True)
        tab.radio_kb_sph.setChecked(True)
        tab.combo_kb_source.setCurrentText("UFC 3-340-02")
        tab._redraw()
        self.assertGreater(tab.table.rowCount(), 0)
        self.assertEqual(tab.table.item(0, 1).text(), "3D")
        self.assertTrue(tab.table.item(0, 3).text().replace(" ", "").startswith("3"))
        tab.radio_kb_hemi.setChecked(True)
        tab._redraw()
        self.assertEqual(tab.table.item(0, 5).text(), "N/A")
        self.assertIn("surface/orientation", tab.lbl_status.text())


class MainWindowRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_tab_is_after_time_history(self):
        from main_new import BlastFoamApp

        win = BlastFoamApp()
        try:
            tabs = win.tabs
            names = [tabs.tabText(i) for i in range(tabs.count())]
            th = names.index("Time History Viewer")
            vv = names.index("Validation & Verification")
            self.assertEqual(vv, th + 1)
            self.assertIsInstance(win.tab_validation, TabValidation)
            snap = win._validation_context()
            self.assertEqual(snap.source, SOURCE_CURRENT)
            win.tabs.setCurrentWidget(win.tab_validation)
            self.assertEqual(win.tabs.currentWidget(), win.tab_validation)
        finally:
            win.close()

    def test_ui_preview_selects_validation(self):
        from ui_preview import launch_preview

        window, _app = launch_preview("validation", "ready", False, show=False, exec_loop=False)
        try:
            self.assertEqual(window.tabs.tabText(window.tabs.currentIndex()), "Validation & Verification")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
