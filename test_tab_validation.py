"""Offscreen Qt tests for the Validation & Verification tab."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

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
from ui_metrics import ERROR_STATUS_STYLE, INFO_STATUS_STYLE, WARNING_STYLE
from validation.auto_points import ValidationPoint
from validation.current_run import SOURCE_CURRENT, SOURCE_MANUAL, RunSnapshot
from validation import remap as remap_engine
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

    def test_mixed_1d_spherical_and_2d_hemi_use_separate_reference_curves(self):
        from validation import ufc_airblast as ufc_ab
        from validation.units import fmt, pa_to_kpa

        snap = RunSnapshot(
            live_mode="1d",
            mass_kg=1.0,
            domain_radius_1d=2.0,
            domain_radius_2d=2.0,
            domain_height_2d=1.0,
            hob_m=0.0,
        )
        tab = self._tab(snap)
        tab.chk_show_2d.setChecked(True)
        tab.combo_kb_source.setCurrentText("UFC 3-340-02")
        tab._redraw()
        labels = []
        if tab.plot_canvas.axes.get_legend():
            labels = [t.get_text() for t in tab.plot_canvas.axes.get_legend().get_texts()]
        self.assertTrue(any("Figure 2-7" in t for t in labels))
        self.assertTrue(any("Figure 2-15" in t for t in labels))
        plans = {p.dim: p for p in tab._collect_auto_plans()}
        pt_1d = plans["1d"].points[0]
        pt_2d = plans["2d"].points[0]
        ev_1d = ufc_ab.evaluate(
            ufc_ab.QUANTITY_PEAK_PRESSURE,
            range_m=pt_1d.range_m,
            mass_kg=1.0,
            burst_type=ufc_ab.BURST_SPHERICAL,
        )
        ev_2d = ufc_ab.evaluate(
            ufc_ab.QUANTITY_PEAK_PRESSURE,
            range_m=pt_2d.range_m,
            mass_kg=1.0,
            burst_type=ufc_ab.BURST_HEMISPHERICAL,
        )
        wrong_1d = ufc_ab.evaluate(
            ufc_ab.QUANTITY_PEAK_PRESSURE,
            range_m=pt_1d.range_m,
            mass_kg=1.0,
            burst_type=ufc_ab.BURST_HEMISPHERICAL,
        )
        row_1d = None
        row_2d = None
        for r in range(tab.table.rowCount()):
            gid = tab.table.item(r, 0).text()
            if gid == pt_1d.point_id:
                row_1d = r
            if gid == pt_2d.point_id:
                row_2d = r
        self.assertIsNotNone(row_1d)
        self.assertIsNotNone(row_2d)
        self.assertTrue(ev_1d.ok)
        self.assertTrue(ev_2d.ok)
        self.assertEqual(tab.table.item(row_1d, 5).text(), fmt(pa_to_kpa(ev_1d.value_si), suffix="kPa"))
        self.assertEqual(tab.table.item(row_2d, 5).text(), fmt(pa_to_kpa(ev_2d.value_si), suffix="kPa"))
        self.assertFalse(wrong_1d.ok)
        self.assertEqual(tab.table.item(row_1d, 6).text(), "N/A")
        self.assertEqual(tab.table.item(row_2d, 6).text(), "N/A")

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


class ValidationCacheAndStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def _tab(self, snapshot):
        tab = TabValidation()

        def provider():
            return snapshot

        tab.set_source_provider(context=provider, gauges_1d=lambda: (), probes_2d=lambda: (), probes_3d=lambda: ())
        tab.show()
        return tab

    def _write_scalar(self, path, values):
        body = " ".join(f"{float(v)}" for v in values)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"internalField   nonuniform List<scalar> {len(values)} ({body});\n")

    def _write_vector(self, path, pts):
        chunks = " ".join(f"({p[0]} {p[1]} {p[2]})" for p in pts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"internalField   nonuniform List<vector> {len(pts)} ({chunks});\n")

    def test_completed_case_reuses_stored_plan_when_receive_fingerprint_differs(self):
        from validation.fingerprint import case_id_from_path

        with tempfile.TemporaryDirectory() as td:
            case_id = case_id_from_path(td)
            payload = {
                "dim": "2d",
                "burst_master": "ufc_free_air_spherical",
                "figure": "2-7",
                "mass_kg": 1.0,
                "charge_center": [0.0, 1.0, 0.0],
                "r_min": 0.67,
                "r_max": 1.9,
                "z_min": 0.15,
                "z_max": 30.0,
                "n_points": 1,
                "line_kind": "horizontal_through_charge_centre",
                "line_z": 1.0,
                "points": [
                    {
                        "point_id": "VAL_2D_STORED",
                        "dim": "2d",
                        "index": 0,
                        "range_m": 0.67,
                        "x": 0.67,
                        "y": 1.0,
                        "z": 0.0,
                        "mass_kg": 1.0,
                        "burst": "ufc_free_air_spherical",
                        "figure": "2-7",
                    }
                ],
                "domain_r_max": 2.0,
                "remap_receive_r_max": 0.658,
                "fingerprint": {
                    "dim": "2d",
                    "case_id": case_id,
                    "mass_kg": 1.0,
                    "domain_size": {"radius": 2.0, "height": 2.0},
                    "hob_m": 1.0,
                    "charge_center": [0.0, 1.0, 0.0],
                    "cell_size": 0.01,
                    "burst_mode": "ufc_free_air_spherical",
                    "reference_mode": "UFC 3-340-02 Figure 2-7",
                    "remap_receive_r_max": 0.658,
                },
            }
            with open(os.path.join(td, "ggui_validation_sampling.json"), "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            tab = self._tab(
                RunSnapshot(
                    case_2d=td,
                    mass_kg=1.0,
                    hob_m=1.0,
                    domain_radius_2d=2.0,
                    domain_height_2d=2.0,
                    domain_cell_2d=0.01,
                    mapped_radius=0.6,
                )
            )
            plans = tab._collect_auto_plans()
            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0].points[0].point_id, "VAL_2D_STORED")

    def test_dim_toggle_does_not_clear_auto_plans(self):
        tab = self._tab(RunSnapshot(mass_kg=1.0, domain_radius_1d=1.0, domain_cell_1d=0.01))
        sentinel = [object()]
        tab._auto_plans = sentinel
        tab._on_auto_dim_changed()
        self.assertIs(tab._auto_plans, sentinel)

    def test_probe_cache_reuses_and_invalidates_on_case_change(self):
        point = ValidationPoint(point_id="p1", dim="1d", index=0, range_m=1.0, x=1.0, y=0.0, z=0.0)
        tab = self._tab(RunSnapshot(case_1d="caseA", domain_radius_1d=1.0, mass_kg=1.0))
        calls = []

        def fake_compute(_point):
            calls.append(1)
            return 1.0, 2.0, "", True

        tab._bf_auto_peak_impulse_compute = fake_compute
        tab._auto_key = tab._snapshot_cache_key()
        first = tab._bf_auto_peak_impulse(point)
        second = tab._bf_auto_peak_impulse(point)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        other = RunSnapshot(case_1d="caseB", domain_radius_1d=1.0, mass_kg=1.0)
        tab.set_source_provider(context=lambda: other, gauges_1d=lambda: (), probes_2d=lambda: (), probes_3d=lambda: ())
        self.assertEqual(tab._probe_cache, {})
        self.assertEqual(tab._remap_cache, {})
        self.assertEqual(tab._numerical_cache, {})

    def test_fixed_mesh_cell_count_is_info_not_error(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "system"))
            os.makedirs(os.path.join(td, "constant", "polyMesh"))
            with open(os.path.join(td, "system", "controlDict"), "w", encoding="utf-8") as handle:
                handle.write("maxCo 0.4;\nendTime 0.01;\nstartTime 0;\n")
            with open(os.path.join(td, "log.blastFoam"), "w", encoding="utf-8") as handle:
                handle.write("Time = 0.001\ndeltaT = 1e-6\nCourant Number Mean/Max = 0.1, 0.4\nEnd\n")
            with open(os.path.join(td, "constant", "polyMesh", "owner"), "w", encoding="utf-8") as handle:
                handle.write("nCells: 40000\n")
            tab = self._tab(RunSnapshot(case_2d=td, keep_openfoam_2d=False))
            tab.combo_mode.setCurrentText(MODE_NUMERICAL)
            tab._on_mode_changed(MODE_NUMERICAL)
            tab._redraw()
            self.assertIn("Cell count is constant", tab.lbl_status.text())
            self.assertEqual(tab.lbl_status.styleSheet(), INFO_STATUS_STYLE)
            self.assertNotEqual(tab.lbl_status.styleSheet(), ERROR_STATUS_STYLE)
            self.assertNotEqual(tab.lbl_status.styleSheet(), WARNING_STYLE)
            from validation import numerical as numerical_engine

            builds = []
            original = numerical_engine.build_report

            def wrapped(*args, **kwargs):
                builds.append(1)
                return original(*args, **kwargs)

            numerical_engine.build_report = wrapped
            try:
                tab.combo_num_plot.setCurrentText("deltaT vs Time")
                tab._draw_numerical()
                tab.combo_num_plot.setCurrentText("Total Cells vs Time")
                tab._draw_numerical()
                self.assertEqual(len(builds), 0)
            finally:
                numerical_engine.build_report = original

    def test_fatal_log_uses_error_style(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "system"))
            with open(os.path.join(td, "system", "controlDict"), "w", encoding="utf-8") as handle:
                handle.write("maxCo 0.4;\n")
            with open(os.path.join(td, "log.blastFoam"), "w", encoding="utf-8") as handle:
                handle.write("FOAM FATAL ERROR\nfloating point exception\n")
            tab = self._tab(RunSnapshot(case_2d=td, keep_openfoam_2d=False))
            tab.combo_mode.setCurrentText(MODE_NUMERICAL)
            tab._on_mode_changed(MODE_NUMERICAL)
            tab._redraw()
            self.assertEqual(tab.lbl_status.styleSheet(), ERROR_STATUS_STYLE)

    def test_remap_graph_is_sorted_clipped_and_cached(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "1d")
            tgt = os.path.join(td, "2d")
            src_r = np.array([0.10, 0.30, 0.50, 0.70])
            src_p = np.array([4.0, 3.0, 2.0, 1.0])
            src_rho = np.array([1.4, 1.3, 1.2, 1.1])
            tgt_r = np.array([0.70, 0.50, 0.30, 0.10])
            tgt_p = np.array([1.05, 2.10, 3.05, 4.00])
            tgt_rho = np.array([1.11, 1.21, 1.31, 1.40])
            off_r = np.array([0.10, 0.30])
            self._write_vector(os.path.join(src, "0.000184", "C"), np.column_stack([src_r, np.zeros(4), np.zeros(4)]))
            self._write_scalar(os.path.join(src, "0.000184", "p"), src_p)
            self._write_scalar(os.path.join(src, "0.000184", "rho.air"), src_rho)
            ray = np.column_stack([tgt_r, np.ones(4), np.zeros(4)])
            off = np.column_stack([off_r, np.zeros(2), np.zeros(2)])
            self._write_vector(os.path.join(tgt, "0", "C"), np.vstack([ray, off]))
            self._write_scalar(os.path.join(tgt, "0", "p"), np.concatenate([tgt_p, np.array([99.0, 88.0])]))
            self._write_scalar(os.path.join(tgt, "0", "rho.air"), np.concatenate([tgt_rho, np.array([9.0, 8.0])]))
            with open(os.path.join(tgt, "case_2d.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "mapping": {"case_path": src, "specific_time": "0.000184"},
                        "remap_timing": {
                            "source_time_label": "0.000184",
                            "source_physical_time": 0.000184,
                            "physical_time_offset": 0.000184,
                            "target_time_label": "0",
                        },
                        "remap_handoff": {"remap_radius_m": 0.6},
                    },
                    handle,
                )
            tab = self._tab(
                RunSnapshot(
                    case_1d=src,
                    case_2d=tgt,
                    mapping_source_2d=src,
                    mapping_time_2d="0.000184",
                    mapped_radius=0.6,
                    hob_m=1.0,
                    domain_cell_2d=0.01,
                )
            )
            tab.combo_mode.setCurrentText(MODE_REMAP)
            tab._on_mode_changed(MODE_REMAP)
            loads = []
            original = remap_engine.load_physical_radial_profile

            def wrapped(*args, **kwargs):
                loads.append(1)
                return original(*args, **kwargs)

            remap_engine.load_physical_radial_profile = wrapped
            try:
                tab._draw_remap()
                first_loads = len(loads)
                self.assertGreaterEqual(first_loads, 2)
                xs = tab.plot_canvas.axes.lines[0].get_xdata()
                self.assertTrue(all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1)))
                self.assertLessEqual(max(xs), 0.6 + 1e-12)
                self.assertNotIn(0.70, [round(float(v), 6) for v in xs])
                self.assertIn("0.000184", tab.lbl_remap_times.text())
                self.assertIn("Physical remap radius: 0.6 m", tab.lbl_remap_times.text())
                self.assertIn("|Target − Source|", [line.get_label() for line in tab.plot_canvas.axes.lines])
                tab.combo_remap_diff.setCurrentIndex(1)
                tab._draw_remap()
                self.assertEqual(len(loads), first_loads)
                tab.combo_remap_field.setCurrentText("Density")
                tab._draw_remap()
                after_density = len(loads)
                self.assertGreater(after_density, first_loads)
                tab.combo_remap_field.setCurrentText("Pressure")
                tab._draw_remap()
                self.assertEqual(len(loads), after_density)
            finally:
                remap_engine.load_physical_radial_profile = original


if __name__ == "__main__":
    unittest.main()
