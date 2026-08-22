"""Focused regressions for GUI lifecycle and signal-wiring health fixes."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication

from bf_option_discovery import _deduplicate_scan_roots, _discover_phase_tokens
from main_new import BlastFoamApp
from output_options import (
    Dim2DOutput,
    Dim3DOutput,
    GaugeFlags,
    OutputFileOptions,
    output_file_options_from_dict,
)
from probes_model import ProbesModel
from project_io import build_project, read_project, write_project_atomic
from solver_runner import SolverRunner
from tab_1d import Tab1D
from tab_3d_general import TabGeneral3D
from test_generator_3d import _minimal
from viewer_widget import BlastViewerWidget


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class GuiLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_view_refresh_timer_is_owned_by_main_window(self):
        window = BlastFoamApp()
        try:
            self.assertIs(window.view_timer.parent(), window)
        finally:
            window.close()

    def test_main_tab_change_handler_runs_once_per_change(self):
        window = BlastFoamApp()
        try:
            with mock.patch.object(
                window.tab_2d.viewer, "set_viewport_active"
            ) as set_2d_active, mock.patch.object(
                window.tab_3d.viewer, "set_viewport_active"
            ) as set_3d_active:
                window.tabs.setCurrentWidget(window.tab_2d)
                self.app.processEvents()

            set_2d_active.assert_called_once_with(True)
            set_3d_active.assert_called_once_with(False)
        finally:
            window.close()

    def test_close_is_deferred_while_preparation_thread_is_still_running(self):
        window = BlastFoamApp.__new__(BlastFoamApp)
        window.view_timer = mock.Mock()
        window._prep_phase = "active"
        window._prep_worker = mock.Mock()
        window._prep_worker.isRunning.return_value = True
        window._prep_worker.wait.return_value = False
        window.runner = None
        window.status_bar = mock.Mock()
        window.tab_jotter = mock.Mock()
        window.tab_2d = mock.Mock()
        window.tab_3d = mock.Mock()
        event = mock.Mock()

        window.closeEvent(event)

        window._prep_worker.request_cancel.assert_called_once_with()
        window._prep_worker.wait.assert_called_once_with(5000)
        event.ignore.assert_called_once_with()
        event.accept.assert_not_called()
        window.tab_jotter.stop_monitoring.assert_not_called()

    def test_prep_log_is_routed_to_jotter(self):
        window = BlastFoamApp.__new__(BlastFoamApp)
        window.tab_jotter = mock.Mock()
        window._on_prep_log("mesh step")
        window.tab_jotter.append_line.assert_called_once_with("mesh step")

    def test_user_interrupt_cannot_be_reclassified_as_2d_failure(self):
        window = BlastFoamApp.__new__(BlastFoamApp)
        window._active_run_mode = "2D"
        window._run_user_interrupted = True
        window.view_timer = mock.Mock()
        window.tab_jotter = mock.Mock()
        window.runner = mock.Mock()
        window.status_bar = mock.Mock()
        window.tab_1d = mock.Mock()
        window.tab_2d = mock.Mock()
        window.tab_2d.is_imported_mode = True

        window.on_simulation_finished(False)

        from external_case_workflow_2d import ImportMode2D
        from models_2d import SimulationState2D

        window.tab_2d.set_import_mode.assert_called_once_with(
            ImportMode2D.IMPORTED_2D_READY
        )
        window.tab_2d.set_simulation_state.assert_called_once_with(
            SimulationState2D.INTERRUPTED
        )
        window.status_bar.set_status.assert_called_with("Interrupted", "#e67e22")

    def test_solver_stop_terminates_solver_and_reconstruct_process_trees(self):
        runner = SolverRunner(".", cores=2)
        runner._proc = mock.Mock()
        runner._proc.poll.return_value = None
        runner._reconstruct_proc = mock.Mock()
        runner._reconstruct_proc.poll.return_value = None
        reconstruct = runner._reconstruct_proc

        with mock.patch("solver_runner.terminate_process_tree") as terminate:
            runner.stop()

        self.assertEqual(
            terminate.call_args_list,
            [mock.call(runner._proc), mock.call(reconstruct)],
        )
        self.assertFalse(runner.keep_running)

    def test_stop_before_reconstruct_launch_prevents_new_child(self):
        with tempfile.TemporaryDirectory() as root:
            log_path = os.path.join(root, "log.blastFoam")
            with open(log_path, "w", encoding="utf-8") as stream:
                stream.write("x\nTime = 0.1\n")
            time_dir = os.path.join(root, "processor0", "0.1")
            os.makedirs(time_dir)
            with open(os.path.join(time_dir, "p"), "w", encoding="utf-8"):
                pass

            runner = SolverRunner(root, cores=2)
            runner._log_blastfoam_pos = 1
            runner.stop()
            with mock.patch("solver_runner.subprocess.Popen") as popen:
                runner._maybe_reconstruct_new_times()
            popen.assert_not_called()

    def test_3d_field_change_forces_redraw_at_unchanged_time(self):
        viewer = BlastViewerWidget()
        viewer._last_refresh_time = 1.25
        with mock.patch.object(viewer, "refresh_view") as refresh:
            viewer.set_field("rho")
        self.assertIsNone(viewer._last_refresh_time)
        refresh.assert_called_once_with()


class DiscoveryResourceTests(unittest.TestCase):
    def test_nested_discovery_roots_are_scanned_once(self):
        with tempfile.TemporaryDirectory() as root:
            child = os.path.join(root, "nested")
            os.mkdir(child)
            self.assertEqual(
                _deduplicate_scan_roots([child, root, child]),
                [os.path.normcase(root)],
            )

    def test_phase_discovery_closes_each_scanned_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "phaseProperties")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("equationOfState JWL;\n")

            tracked = mock.mock_open(read_data="equationOfState JWL;\n")
            with mock.patch("builtins.open", tracked):
                eos, _activation, _thermo = _discover_phase_tokens([root])

            self.assertIn("JWL", eos)
            tracked.return_value.__enter__.assert_called_once_with()
            tracked.return_value.__exit__.assert_called_once()


class ProjectRoundTripHealthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_1d_inputs_round_trip_through_project_file_and_controls(self):
        source_tab = Tab1D()
        source_tab.combo_comp.setCurrentText("PETN")
        source_tab.spin_radius.setValue(12.5)
        source_tab.spin_cellsize.setValue(0.025)
        source_tab.spin_mass.setValue(3.0)
        source_tab.spin_gui_refresh.setValue(37)
        source_tab.set_gauge_locations(((1.5, "G1"),))
        expected = source_tab.get_case_inputs()
        payload = build_project(
            _minimal(),
            probes={"probes": []},
            gui_state={"selected_primary_tab": "Spherical – 1D"},
            inputs_1d=expected,
        )

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "roundtrip.ggui.json")
            write_project_atomic(path, payload)
            restored = read_project(path)["inputs_1d"]

        self.assertEqual(restored, expected)
        target_tab = Tab1D()
        target_tab.set_case_inputs(asdict(restored))
        self.assertEqual(target_tab.get_case_inputs(), expected)

    def test_output_dialog_state_round_trip_keeps_2d_gauge_vtk_distinct(self):
        options = OutputFileOptions(
            dim2d=Dim2DOutput(
                gauges=GaugeFlags(pressure=False, density=True),
                vtk=GaugeFlags(pressure=True, density=False),
                output_remap_data=True,
            )
        )
        restored = output_file_options_from_dict(asdict(options))
        self.assertFalse(restored.dim2d.gauges.pressure)
        self.assertTrue(restored.dim2d.gauges.density)
        self.assertTrue(restored.dim2d.vtk.pressure)
        self.assertFalse(restored.dim2d.vtk.density)
        self.assertTrue(restored.dim2d.output_remap_data)

    def test_3d_output_generation_fields_restore_from_case_inputs(self):
        probes = ProbesModel()
        source = TabGeneral3D(probes)
        quantities = {
            "pressure": {
                "gauges": True,
                "sections": False,
                "obstacles": True,
                "volumes": True,
            },
            "arrival_initial": {"obstacles": True},
            "obstacle_id": {"obstacles": True},
        }
        source.apply_output_file_options(
            Dim3DOutput(
                write_surfaces=False,
                write_volumes=True,
                surface_by_time=False,
                surface_steps=17,
                quantities=quantities,
            )
        )
        expected = source.get_case_inputs()

        target = TabGeneral3D(probes)
        target.set_case_inputs(asdict(expected))
        restored = target.get_case_inputs()

        for name in (
            "probe_fields",
            "write_surfaces",
            "write_volumes",
            "surface_write_by_time",
            "surface_write_interval_steps",
            "section_fields",
            "obstacle_fields",
            "volume_fields",
            "write_arrival",
            "write_obstacle_id",
        ):
            self.assertEqual(getattr(restored, name), getattr(expected, name), name)

    def test_malformed_probe_load_is_rejected_without_partial_state(self):
        model = ProbesModel()
        model.add_probe("existing", 1.0, 2.0, 3.0)
        with self.assertRaisesRegex(ValueError, "index 1"):
            model.load_dict(
                {
                    "probes": [
                        {"name": "valid", "x": 0, "y": 0, "z": 0},
                        {"name": "bad", "x": "nan", "y": 0, "z": 0},
                    ]
                }
            )
        self.assertEqual([point.name for point in model.probes()], ["existing"])


if __name__ == "__main__":
    unittest.main()
