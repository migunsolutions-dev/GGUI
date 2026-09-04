from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel

from axisymmetric_2d import DIRECT_SOURCE, DYNAMIC_MESH, FIXED_MESH, REMAP_SOURCE
from axisymmetric_viewer import AxisymmetricViewerWidget, preview_charge_markers
from dialogs import RemapFromDialog
from models_2d import SimulationState2D
from tab_2d import Tab2D, case_dir_from_picked_path, latest_case_1d_dir


class Tab2DWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = Tab2D()
        self.tab.show()
        self.app.processEvents()

    def tearDown(self):
        self.tab.close()

    def test_direct_remap_gating(self):
        self.assertTrue(self.tab.grp_charge.isEnabled())
        self.assertTrue(self.tab.spin_hob.isEnabled())
        self.assertFalse(self.tab.grp_mapping.isEnabled())
        self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
        self.app.processEvents()
        self.assertTrue(self.tab.grp_charge.isEnabled())
        self.assertTrue(self.tab.spin_hob.isEnabled())
        self.assertTrue(self.tab.grp_mapping.isEnabled())
        self.assertFalse(self.tab.grp_seed.isEnabled())
        self.assertFalse(self.tab.spin_det_height.isEnabled())
        self.assertFalse(self.tab.spin_mass.isEnabled())

    def test_remap_hob_stays_editable_and_clears_stale_enablement(self):
        self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
        self.app.processEvents()
        self.assertTrue(self.tab.spin_hob.isEnabled())
        self.tab.spin_hob.setValue(1.25)
        self.assertAlmostEqual(self.tab.get_case_inputs().height_of_burst, 1.25)

        self.tab.cmb_source.setCurrentText(DIRECT_SOURCE)
        self.app.processEvents()
        self.assertTrue(self.tab.spin_hob.isEnabled())
        self.assertTrue(self.tab.spin_det_height.isEnabled())
        self.assertTrue(self.tab.spin_mass.isEnabled())
        self.assertFalse(self.tab.grp_mapping.isEnabled())
        self.assertAlmostEqual(self.tab.get_case_inputs().height_of_burst, 1.25)

        self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
        self.app.processEvents()
        self.assertTrue(self.tab.spin_hob.isEnabled())
        self.assertFalse(self.tab.spin_det_height.isEnabled())
        self.assertTrue(self.tab.grp_mapping.isEnabled())
        self.assertAlmostEqual(self.tab.get_case_inputs().height_of_burst, 1.25)

    def test_fixed_dynamic_gating(self):
        self.tab.cmb_mesh_mode.setCurrentText(FIXED_MESH)
        self.assertFalse(self.tab.grp_seed.isEnabled())
        self.assertFalse(self.tab.grp_amr.isEnabled())
        self.tab.cmb_mesh_mode.setCurrentText(DYNAMIC_MESH)
        self.assertTrue(self.tab.grp_seed.isEnabled())
        self.assertTrue(self.tab.grp_amr.isEnabled())

    def test_sphere_cylinder_enablement(self):
        self.tab.cmb_shape.setCurrentText("Sphere")
        self.assertFalse(self.tab.spin_ld.isVisible())
        self.tab.cmb_shape.setCurrentText("Cylinder")
        self.app.processEvents()
        self.assertTrue(self.tab.spin_ld.isVisible())

    def test_initialized_model_changes_stale_but_view_changes_do_not(self):
        self.tab.mark_initialized("/tmp/case", 900)
        self.tab.cmb_view_mode.setCurrentText("Computational Domain View")
        self.assertEqual(self.tab.simulation_state, SimulationState2D.INITIALIZED)
        self.tab.spin_radius.setValue(self.tab.spin_radius.value() + 0.1)
        self.assertEqual(self.tab.simulation_state, SimulationState2D.STALE)
        self.assertFalse(self.tab.btn_exact_end.isEnabled())

    def test_interruption_and_continuation_controls(self):
        self.tab.set_simulation_state(SimulationState2D.RUNNING)
        self.assertTrue(self.tab.btn_stop.isEnabled())
        self.assertFalse(self.tab.btn_exact_end.isEnabled())
        self.tab.set_simulation_state(SimulationState2D.INTERRUPTED)
        self.assertFalse(self.tab.btn_stop.isEnabled())
        self.assertTrue(self.tab.btn_exact_end.isEnabled())

    def test_exactly_one_viewport_and_approved_actions(self):
        self.assertEqual(len(self.tab.findChildren(AxisymmetricViewerWidget)), 1)
        texts = {button.text() for button in self.tab.findChildren(type(self.tab.btn_stop))}
        self.assertIn("Initialise Model", texts)
        self.assertIn("exact END", texts)
        self.assertIn("Interrupt", texts)
        for forbidden in ("Exec 1", "Exec 10", "Exec 100", "Exec 1000", "Run / Resume"):
            self.assertNotIn(forbidden, texts)

    def test_info_preinit_uses_compact_planned_rows(self):
        rows = {
            label.text().rstrip(":"): value.text()
            for label, value in self.tab._info_row_widgets.values()
            if not label.isHidden()
        }
        self.assertEqual(
            list(rows), ["Base Cell", "Finest Cell", "Charge Radius", "Charge Res."]
        )
        self.assertNotIn("Total Cells", rows)

    def test_remap_edit_replaces_path_and_browse(self):
        self.assertFalse(hasattr(self.tab, "btn_browse_source_case"))
        self.assertEqual(self.tab.btn_edit_remap.text(), "Edit...")
        self.assertFalse(self.tab.btn_edit_remap.isEnabled())
        self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
        self.app.processEvents()
        self.assertTrue(self.tab.btn_edit_remap.isEnabled())

    def test_advanced_remap_rows_are_hidden_and_time_is_always_latest(self):
        hidden_labels = {
            "Source time:",
            "Specific time:",
            "Mapped radius:",
            "Source resolution:",
            (
                "Radial mapping about the target charge centre [0, HOB, 0] is not conservative; "
                "cells sample the 1D spherical profile by distance from that centre."
            ),
        }
        labels = {
            label.text(): label
            for label in self.tab.grp_mapping.findChildren(QLabel)
            if label.text() in hidden_labels
        }
        self.assertEqual(set(labels), hidden_labels)
        self.assertTrue(all(label.isHidden() for label in labels.values()))
        self.assertTrue(self.tab.cmb_source_time_mode.isHidden())
        self.assertTrue(self.tab.txt_source_time.isHidden())
        form = self.tab.grp_mapping.layout()
        self.assertTrue(form.formAlignment() & Qt.AlignVCenter)
        self.assertEqual(self.tab.grp_mapping.minimumHeight(), 0)

        self.tab.set_case_inputs(
            {
                "initialization_source": REMAP_SOURCE,
                "mapping": {
                    "case_path": "source-case",
                    "time_mode": "specific",
                    "specific_time": "0.25",
                },
            }
        )
        mapping = self.tab.get_case_inputs().mapping
        self.assertEqual(mapping.time_mode, "latest")
        self.assertEqual(mapping.specific_time, "")

    def test_remap_from_dialog_matches_requested_choices(self):
        dialog = RemapFromDialog(None, current_kind=RemapFromDialog.CURRENT_1D, has_current_1d=True)
        self.assertEqual(dialog.rad_current_1d.text(), "Current 1D model")
        self.assertEqual(dialog.rad_file_1d.text(), "1D results file")
        self.assertEqual(dialog.rad_file_2d.text(), "2D results file")
        self.assertTrue(dialog.rad_file_1d.isEnabled())
        self.assertTrue(dialog.rad_file_2d.isEnabled())
        self.assertTrue(dialog.rad_current_1d.isChecked())
        self.assertEqual(dialog.selected_kind(), RemapFromDialog.CURRENT_1D)
        dialog.rad_file_2d.setChecked(True)
        self.assertEqual(dialog.selected_kind(), RemapFromDialog.FILE_2D)
        dialog.close()

    def test_edit_remap_current_1d_keeps_last_run(self):
        with tempfile.TemporaryDirectory() as td:
            latest = os.path.join(td, "Case_1D_20260821_185000")
            os.mkdir(latest)
            self.tab.set_source_cases_root(td)
            self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
            self.tab._apply_remap_from_choice(RemapFromDialog.CURRENT_1D)
            self.assertEqual(
                os.path.normpath(self.tab.get_case_inputs().mapping.case_path),
                os.path.normpath(latest),
            )
            self.assertEqual(self.tab.txt_source_case.text(), "Current 1D model")

    def test_edit_remap_imports_other_1d_results_file(self):
        with tempfile.TemporaryDirectory() as td:
            latest = os.path.join(td, "Case_1D_20260821_185000")
            picked = os.path.join(td, "Case_1D_other")
            os.mkdir(latest)
            os.makedirs(os.path.join(picked, "system"))
            foam = os.path.join(picked, "case.foam")
            with open(foam, "w", encoding="utf-8") as handle:
                handle.write("")
            self.tab.set_source_cases_root(td)
            self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
            with mock.patch(
                "tab_2d.QFileDialog.getOpenFileName",
                return_value=(foam, "OpenFOAM (case.foam)"),
            ) as picker:
                self.tab._apply_remap_from_choice(RemapFromDialog.FILE_1D)
            picker.assert_called_once()
            self.assertEqual(
                os.path.normpath(self.tab.get_case_inputs().mapping.case_path),
                os.path.normpath(picked),
            )
            self.tab.set_last_1d_case(latest)
            self.assertEqual(
                os.path.normpath(self.tab.get_case_inputs().mapping.case_path),
                os.path.normpath(picked),
            )

    def test_selecting_from_1d_uses_latest_case_from_current_session(self):
        with tempfile.TemporaryDirectory() as td:
            imported = os.path.join(td, "Case_1D_imported")
            current = os.path.join(td, "Case_1D_current_session")
            os.mkdir(imported)
            os.mkdir(current)
            self.tab._set_remap_case_path(imported, from_last_1d=False)
            self.tab.set_last_1d_case(current)
            self.assertEqual(
                os.path.normpath(self.tab.get_case_inputs().mapping.case_path),
                os.path.normpath(imported),
            )

            self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
            self.app.processEvents()

            self.assertEqual(
                os.path.normpath(self.tab.get_case_inputs().mapping.case_path),
                os.path.normpath(current),
            )
            self.assertEqual(self.tab.cmb_source_time_mode.currentText(), "latest")
            self.assertEqual(self.tab.txt_source_case.text(), "Current 1D model")

    def test_edit_remap_imports_2d_results_file(self):
        with tempfile.TemporaryDirectory() as td:
            latest = os.path.join(td, "Case_1D_20260821_185000")
            picked = os.path.join(td, "Case_2D_other")
            os.mkdir(latest)
            os.makedirs(os.path.join(picked, "system"))
            foam = os.path.join(picked, "case.foam")
            with open(foam, "w", encoding="utf-8") as handle:
                handle.write("")
            self.tab.set_source_cases_root(td)
            self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
            with mock.patch(
                "tab_2d.QFileDialog.getOpenFileName",
                return_value=(foam, "OpenFOAM (case.foam)"),
            ):
                self.tab._apply_remap_from_choice(RemapFromDialog.FILE_2D)
            self.assertEqual(
                os.path.normpath(self.tab.get_case_inputs().mapping.case_path),
                os.path.normpath(picked),
            )
            self.assertEqual(self.tab._remap_kind, RemapFromDialog.FILE_2D)

    def test_case_dir_from_picked_foam_file(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "system"))
            foam = os.path.join(td, "case.foam")
            with open(foam, "w", encoding="utf-8") as handle:
                handle.write("")
            self.assertEqual(
                os.path.normpath(case_dir_from_picked_path(foam)),
                os.path.normpath(td),
            )

    def test_remap_defaults_to_latest_1d_case(self):
        with tempfile.TemporaryDirectory() as td:
            older = os.path.join(td, "Case_1D_20260101_000000")
            newer = os.path.join(td, "Case_1D_20260821_185000")
            os.mkdir(older)
            os.mkdir(newer)
            os.mkdir(os.path.join(td, "Case_2D_ignore"))
            self.assertEqual(
                os.path.normpath(latest_case_1d_dir(td)),
                os.path.normpath(newer),
            )
            self.tab.set_source_cases_root(td)
            self.assertEqual(
                os.path.normpath(self.tab.get_case_inputs().mapping.case_path),
                os.path.normpath(newer),
            )
            self.assertEqual(self.tab.txt_source_case.text(), "Current 1D model")

    def test_remap_preview_shows_one_target_hob_marker(self):
        self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
        self.tab.spin_det_height.setValue(0.1)
        self.tab.spin_hob.setValue(1.25)
        self.app.processEvents()
        preview = self.tab.viewer._last_preview_data
        self.assertIsNotNone(preview)
        charge = preview[2]
        self.assertTrue(charge.get("remap"))
        markers = preview_charge_markers(charge)
        self.assertEqual(markers, [(0.0, 1.25)])
        self.assertNotAlmostEqual(float(charge.get("detonation_height")), 1.25)

    def test_reinitialize_clears_stale_preview_markers(self):
        self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
        self.tab.spin_hob.setValue(0.5)
        self.tab.spin_det_height.setValue(0.1)
        self.app.processEvents()
        self.assertIsNotNone(self.tab.viewer._last_preview_data)
        self.tab.spin_hob.setValue(1.25)
        self.app.processEvents()
        stale = self.tab.viewer._last_preview_data
        self.assertAlmostEqual(stale[2]["height"], 1.25)
        self.assertEqual(preview_charge_markers(stale[2]), [(0.0, 1.25)])
        self.tab.viewer.load_case(
            tempfile.gettempdir(),
            charge_center=(0.0, 1.25, 0.0),
        )
        self.assertIsNone(self.tab.viewer._last_preview_data)
        self.assertEqual(self.tab.viewer._charge_center, (0.0, 1.25, 0.0))
        self.assertTrue(self.tab.viewer.is_simulating)

    def test_remap_status_visible_for_from_1d_not_hidden_advanced_row(self):
        self.assertTrue(self.tab.lbl_remap_status.isHidden())
        self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
        self.app.processEvents()
        self.assertFalse(self.tab.lbl_remap_status.isHidden())
        text = self.tab.lbl_remap_status.text().lower()
        self.assertTrue("missing" in text or "unavailable" in text or "no 1d" in text)

    def test_remap_status_reports_snapshot_from_last_solver_state(self):
        import numpy as np

        from completion_1d import (
            RUN_MODE_TERMINATE,
            STOP_REASON_WAVE_RADIUS_REACHED,
            CompletionRecord,
            write_completion_record,
        )
        from remap_snapshot_1d import write_snapshot

        with tempfile.TemporaryDirectory() as td:
            n = 8
            write_completion_record(
                td,
                CompletionRecord(
                    mode=RUN_MODE_TERMINATE,
                    stop_reason=STOP_REASON_WAVE_RADIUS_REACHED,
                    wave_radius_reached=True,
                    final_solver_time_s=0.000639296,
                    requested_stop_radius_m=0.8,
                ),
            )
            write_snapshot(
                td,
                {
                    "r": np.linspace(0.0, 1.0, n),
                    "p": np.full(n, 2.0e5),
                    "T": np.full(n, 300.0),
                    "U_mag": np.linspace(0.0, 100.0, n),
                },
                physical_time=0.000639296,
            )
            self.tab.cmb_source.setCurrentText(REMAP_SOURCE)
            self.tab._set_remap_case_path(td, from_last_1d=False)
            self.app.processEvents()
            text = self.tab.lbl_remap_status.text().lower()
            self.assertIn("snapshot", text)
            self.assertIn("last solver state", text)
            self.assertIn("0.000639296", self.tab.lbl_remap_status.text())


if __name__ == "__main__":
    unittest.main()
