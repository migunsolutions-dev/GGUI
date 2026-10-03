"""R-013/R-078/R-099: Run never silently regenerates or uses changed inputs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

class LifecycleGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.qt=QApplication.instance() or QApplication([])

    def setUp(self):
        from main_new import BlastFoamApp
        self.app=BlastFoamApp()
        self.messages=patch('main_new.QMessageBox.critical').start()
        patch('main_new.QMessageBox.warning').start()
        patch('main_new.QMessageBox.information').start()
        self.start=patch.object(self.app,'_start_solver').start()

    def tearDown(self):
        self.app.close()
        patch.stopall()

    def test_all_native_run_paths_require_explicit_initialization(self):
        for run,init in [('run_1d_process','on_initialize_model_1d'),('run_3d_process','on_initialize_model_3d'),('run_3d_process_exact_1','on_initialize_model_3d'),('run_2d_process_exact_end','on_initialize_model_2d')]:
            with self.subTest(run=run),patch.object(self.app,init) as initialize:
                getattr(self.app,run)()
                initialize.assert_not_called()
                self.start.assert_not_called()

    def test_core_or_physics_change_blocks_before_runtime_patch(self):
        with tempfile.TemporaryDirectory() as root:
            self.app.active_case_dir_3d=root
            self.app.active_case_initialized_3d=True
            inputs=self.app.tab_3d.get_case_inputs()
            self.app._record_case_initialized('3D',root,inputs)
            self.app.tab_3d.spin_cores.setValue(inputs.cores+1)
            self.qt.processEvents()
            with patch.object(self.app,'_update_control_dict_end_time') as update:
                self.app.run_3d_process()
                update.assert_not_called()
            self.start.assert_not_called()
            self.assertFalse(self.app.active_case_initialized_3d)

    def test_view_changes_do_not_invalidate(self):
        with tempfile.TemporaryDirectory() as root:
            self.app.active_case_dir_3d=root
            self.app.active_case_initialized_3d=True
            self.app._record_case_initialized('3D',root,self.app.tab_3d.get_case_inputs())
            self.app.tab_3d.cmb_field.setCurrentIndex(1)
            self.qt.processEvents()
            self.assertTrue(self.app._case_run_guard('3D',self.app.tab_3d.get_case_inputs()))

    def test_1d_run_uses_initialized_directory_without_regeneration(self):
        from execution_plan import ExecutionIntent
        with tempfile.TemporaryDirectory() as root:
            self.app.active_case_dir_1d=root
            self.app.active_case_initialized_1d=True
            self.app._record_case_initialized('1D',root,self.app.tab_1d.get_case_inputs())
            with patch.object(self.app.service,'generate_case') as generate:
                self.app.run_1d_process()
                generate.assert_not_called()
            self.start.assert_called_once_with(root,cores=1,mode='1D',intent=ExecutionIntent.INITIALIZED_SOLVER_RUN)

    def test_first_run_of_initialized_2d_and_3d_uses_same_case(self):
        for dimension,run in [('2D','run_2d_process_exact_end'),('3D','run_3d_process')]:
            with self.subTest(dimension=dimension), tempfile.TemporaryDirectory() as root:
                self.start.reset_mock();self.messages.reset_mock()
                tab=getattr(self.app,'tab_'+dimension.lower())
                inputs=tab.get_case_inputs()
                setattr(self.app,'active_case_dir_'+dimension.lower(),root)
                setattr(self.app,'active_case_initialized_'+dimension.lower(),True)
                self.app._record_case_initialized(dimension,root,inputs)
                with patch.object(self.app,'_update_control_dict_end_time'):
                    getattr(self.app,run)()
                self.messages.assert_not_called()
                self.start.assert_called_once()
                self.assertEqual(self.start.call_args.args[0],root)

    def test_preparation_blocks_native_run_before_runtime_mutation(self):
        with patch.object(self.app,'_prep_is_active',return_value=True), \
             patch.object(self.app,'_update_control_dict_end_time') as update:
            self.app.run_3d_process()
            update.assert_not_called()
            self.start.assert_not_called()

    def test_imported_run_blocks_stale_and_busy_before_writing(self):
        from types import SimpleNamespace
        from external_case_workflow_2d import ImportMode2D
        with tempfile.TemporaryDirectory() as root:
            self.app.active_case_dir_2d=root
            self.app.active_case_initialized_2d=True
            inputs=self.app.tab_2d.get_case_inputs()
            self.app._record_case_initialized('2D',root,inputs)
            self.app.tab_2d._imported_case=SimpleNamespace(
                mode=ImportMode2D.IMPORTED_2D_READY, working_copy_dir=root,
                case_dir=root, source_dir=root+'_source')
            with patch.object(self.app,'_prep_is_active',return_value=True), \
                 patch('main_new.write_control_dict_entries') as write:
                self.app.run_imported_2d_exact_end()
                write.assert_not_called()
                self.start.assert_not_called()
            from dataclasses import replace
            inputs=replace(inputs,cores=inputs.cores+1)
            with patch.object(self.app.tab_2d,'get_case_inputs',return_value=inputs), \
                 patch('main_new.write_control_dict_entries') as write:
                self.app.run_imported_2d_exact_end()
                write.assert_not_called()
                self.start.assert_not_called()
            self.assertTrue(self.app._case_sessions['2D'].stale)

    def test_1d_end_time_edit_does_not_require_initialize(self):
        from execution_plan import ExecutionIntent
        with tempfile.TemporaryDirectory() as root:
            system = Path(root) / "system"
            system.mkdir()
            (system / "controlDict").write_text(
                "startFrom       startTime;\nendTime         0.001;\n",
                encoding="utf-8",
            )
            self.app.active_case_dir_1d = root
            self.app.active_case_initialized_1d = True
            inputs = self.app.tab_1d.get_case_inputs()
            self.app._record_case_initialized("1D", root, inputs)
            self.app.tab_1d.spin_endtime.setValue(inputs.end_time_s + 0.001)
            self.qt.processEvents()
            self.assertTrue(self.app.active_case_initialized_1d)
            self.assertFalse(self.app._case_sessions["1D"].stale)
            self.app.run_1d_process()
            self.start.assert_called_once()
            self.assertEqual(self.start.call_args.kwargs["intent"], ExecutionIntent.INITIALIZED_SOLVER_RUN)
            text = (system / "controlDict").read_text(encoding="utf-8")
            self.assertIn(f"{inputs.end_time_s + 0.001:.12g}", text)

    def test_1d_radius_edit_still_requires_initialize(self):
        with tempfile.TemporaryDirectory() as root:
            self.app.active_case_dir_1d = root
            self.app.active_case_initialized_1d = True
            inputs = self.app.tab_1d.get_case_inputs()
            self.app._record_case_initialized("1D", root, inputs)
            self.app.tab_1d.spin_radius.setValue(inputs.radius + 0.1)
            self.qt.processEvents()
            self.assertFalse(self.app.active_case_initialized_1d)
            self.assertFalse(self.app.tab_1d.btn_run.isEnabled())
            self.app.run_1d_process()
            self.start.assert_not_called()

    def test_completed_1d_run_is_not_relaunched(self):
        from completion_1d import (
            initial_completion_record,
            finalize_completion_record,
            write_completion_record,
        )
        with tempfile.TemporaryDirectory() as root:
            system = Path(root) / "system"
            system.mkdir()
            (system / "controlDict").write_text("endTime         0.002;\n", encoding="utf-8")
            write_completion_record(root, initial_completion_record(mode="reflect", requested_stop_radius_m=0.5))
            finalize_completion_record(
                root,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.002,
                reached_end_time=True,
                end_time_s=0.002,
            )
            self.app.active_case_dir_1d = root
            self.app.active_case_initialized_1d = True
            inputs = self.app.tab_1d.get_case_inputs()
            self.app.tab_1d.spin_endtime.setValue(0.002)
            self.app._record_case_initialized("1D", root, self.app.tab_1d.get_case_inputs())
            self.app.run_1d_process()
            self.start.assert_not_called()

    def test_increased_end_time_resumes_completed_1d_run(self):
        from completion_1d import finalize_completion_record, initial_completion_record, write_completion_record
        from execution_plan import ExecutionIntent
        with tempfile.TemporaryDirectory() as root:
            system = Path(root) / "system"
            system.mkdir()
            (system / "controlDict").write_text(
                "startFrom       startTime;\nendTime         0.002;\n",
                encoding="utf-8",
            )
            (Path(root) / "0.002").mkdir()
            write_completion_record(root, initial_completion_record(mode="reflect", requested_stop_radius_m=0.5))
            finalize_completion_record(
                root,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.002,
                reached_end_time=True,
                end_time_s=0.002,
            )
            self.app.active_case_dir_1d = root
            self.app.active_case_initialized_1d = True
            self.app.tab_1d.spin_endtime.setValue(0.002)
            self.app._record_case_initialized("1D", root, self.app.tab_1d.get_case_inputs())
            self.app.tab_1d.spin_endtime.setValue(0.004)
            self.qt.processEvents()
            self.app.run_1d_process()
            self.start.assert_called_once()
            self.assertEqual(self.start.call_args.kwargs["intent"], ExecutionIntent.RESUME)

    def test_unprogressed_interrupt_returns_to_initialized(self):
        from completion_1d import read_completion_record, initial_completion_record, write_completion_record
        with tempfile.TemporaryDirectory() as root:
            write_completion_record(root, initial_completion_record(mode="reflect", requested_stop_radius_m=0.5))
            record = read_completion_record(root)
            record.stop_reason = "user_stopped"
            write_completion_record(root, record)
            self.app.active_case_dir_1d = root
            self.app.active_case_initialized_1d = True
            self.app._active_run_mode = "1D"
            self.app._active_run_case_dir = root
            self.app._run_user_interrupted = True
            self.app.runner = None
            self.app.status_bar.update_1d(step=13000, tt=0.003411, dt=2.5e-7)
            self.app.status_bar.set_progress(40)
            with patch.object(self.app, "_refresh_validation"):
                self.app.on_simulation_finished(False)
            self.assertEqual(self.app.status_bar.lbl_status.text(), "Initialized, not yet progressed")
            self.assertEqual(self.app.status_bar._1d["step"], 0)
            self.assertEqual(self.app.status_bar._1d["tt"], 0.0)
            self.assertEqual(self.app.status_bar._1d["dt"], 0.0)
            self.assertIn("Step=      0", self.app.status_bar.lbl_1d_group.text())
            self.assertEqual(read_completion_record(root).stop_reason, "")
            self.assertTrue(self.app.active_case_initialized_1d)

    def test_completed_1d_metrics_follow_the_solver_log(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / "log.blastFoam"
            log.write_text(
                "Time = 0.00199999\ndeltaT = 2.5e-07\nTime = 0.00300002\ndeltaT = 2.2e-07\nEnd\n",
                encoding="utf-8",
            )
            self.app.status_bar.update_1d(step=7225, tt=0.001999, dt=2.5e-7)
            self.app.runner = SimpleNamespace(snapshot_error="", final_display_metrics=None)
            self.app._active_run_mode = "1D"
            self.app._active_run_case_dir = root
            with patch.object(self.app, "_refresh_validation"):
                self.app.on_simulation_finished(True)
            self.assertEqual(self.app.status_bar.lbl_status.text(), "Done")
            self.assertEqual(self.app.status_bar._1d["step"], 2)
            self.assertAlmostEqual(self.app.status_bar._1d["tt"], 0.00300002)
            self.assertAlmostEqual(self.app.status_bar._1d["dt"], 2.2e-7)

    def test_same_window_project_open_drops_1d_runtime_case(self):
        with tempfile.TemporaryDirectory() as root:
            self.app.active_case_dir_1d = root
            self.app.active_case_initialized_1d = True
            self.app.active_case_dir_2d = root
            self.app.active_case_initialized_2d = True
            self.app._record_case_initialized("1D", root, self.app.tab_1d.get_case_inputs())
            self.app.tab_1d.set_gauge_locations(((0.2, "G1"),))
            self.app.tab_2d._last_1d_case_dir = root
            self.app.tab_2d._remap_from_last_1d = True
            self.app.tab_2d._remap_case_path = root
            self.app.tab_time_history._run_cases["1d"] = root
            self.app.tab_time_history._added.append(("1d", 0))
            self.app.status_bar.update_1d(step=50, tt=0.01, dt=1e-6)
            self.app.tab_1d.btn_run.setEnabled(True)
            self.app._release_open_project_runtime()
            self.assertIsNone(self.app.active_case_dir_1d)
            self.assertFalse(self.app.active_case_initialized_1d)
            self.assertIsNone(self.app.active_case_dir_2d)
            self.assertNotIn("1D", self.app._case_sessions)
            self.assertEqual(self.app.tab_2d._last_1d_case_dir, "")
            self.assertFalse(self.app.tab_2d._remap_from_last_1d)
            self.assertEqual(self.app.tab_2d._remap_case_path, "")
            self.assertEqual(self.app.tab_time_history._run_cases["1d"], "")
            self.assertEqual(self.app.tab_time_history._added, [])
            self.assertEqual(self.app.status_bar._1d["step"], 0)
            self.assertEqual(self.app.tab_1d._gauge_locations, ((0.2, "G1"),))
            self.assertFalse(self.app.tab_1d.btn_run.isEnabled())
            saved = str(Path(root) / "saved_remap")
            self.app.tab_2d._remap_from_last_1d = False
            self.app.tab_2d._remap_case_path = saved
            self.app._release_open_project_runtime()
            self.assertEqual(self.app.tab_2d._remap_case_path, saved)

    def test_solver_success_with_snapshot_failure_shows_separate_warning(self):
        from types import SimpleNamespace
        self.app.runner=SimpleNamespace(snapshot_error='OSError: disk full')
        self.app._active_run_mode=None
        with patch.object(self.app.status_bar,'set_status') as status, \
             patch.object(self.app,'_refresh_validation'):
            self.app.on_simulation_finished(True)
        status.assert_called_with('Done — remap snapshot unavailable','#e67e22')

if __name__=='__main__':unittest.main()
