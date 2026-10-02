"""2D outer-radius stop and meridional camera behaviour."""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from axisymmetric_viewer import AxisymmetricViewerWidget, _install_meridional_mouse
from completion_1d import (
    RUN_MODE_TERMINATE,
    STOP_REASON_WAVE_RADIUS_REACHED,
    CompletionRecord,
    resolve_arrival_probe,
    reset_2d_wave_stop_for_new_run,
    write_2d_wave_stop,
)


class ArrivalProbeTests(unittest.TestCase):
    def test_watchdog2d_probe_resolves_by_spherical_radius(self):
        with tempfile.TemporaryDirectory() as case:
            folder = os.path.join(case, "postProcessing", "watchdog2d", "0")
            os.makedirs(folder)
            # Probe at r=1.9, z=1.0. Spherical radius is the match key.
            with open(os.path.join(folder, "p"), "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1.9 1 0)\n")
                handle.write("0 101325\n")
                handle.write("0.001 180000\n")
            found = resolve_arrival_probe(case, (1.9**2 + 1.0**2) ** 0.5)
            self.assertIsNotNone(found)
            fo, index, _loc, radius = found
            self.assertEqual(fo, "watchdog2d")
            self.assertEqual(index, 0)
            self.assertAlmostEqual(radius, (1.9**2 + 1.0**2) ** 0.5)

    def test_new_run_clears_arrival_without_a_1d_file(self):
        with tempfile.TemporaryDirectory() as case:
            write_2d_wave_stop(
                case,
                CompletionRecord(
                    mode=RUN_MODE_TERMINATE,
                    requested_stop_radius_m=2.0,
                    wave_radius_reached=True,
                    detected_arrival_time_s=0.01,
                    stop_reason=STOP_REASON_WAVE_RADIUS_REACHED,
                ),
            )
            reset_2d_wave_stop_for_new_run(case)
            from completion_1d import read_2d_wave_stop

            record = read_2d_wave_stop(case)
            self.assertFalse(record.wave_radius_reached)
            self.assertIsNone(record.detected_arrival_time_s)
            self.assertEqual(record.stop_reason, "")
            self.assertAlmostEqual(record.requested_stop_radius_m, 2.0)
            self.assertFalse(os.path.isfile(os.path.join(case, "ggui_1d_run_completion.json")))


class MeridionalCameraTests(unittest.TestCase):
    def test_live_refresh_keeps_mouse_pan_and_zoom(self):
        viewer = AxisymmetricViewerWidget.__new__(AxisymmetricViewerWidget)
        camera = SimpleNamespace(parallel_projection=True, parallel_scale=4.5)
        plotter = SimpleNamespace(
            camera_position=[(1.0, 2.0, 9.0), (1.0, 2.0, 0.0), (0.0, 1.0, 0.0)],
            camera=camera,
            enable_parallel_projection=lambda: None,
        )
        viewer._plotter = plotter
        viewer._shutdown = False
        viewer._axisymmetric_domain = (5.0, 8.0)
        viewer.mirrored_view = False
        AxisymmetricViewerWidget._apply_meridional_camera(viewer, force=False)
        self.assertEqual(plotter.camera_position[0], (1.0, 2.0, 9.0))
        self.assertEqual(camera.parallel_scale, 4.5)

    def test_left_button_pans_instead_of_window_level(self):
        from unittest import mock

        captured = []

        class _Plotter:
            class _Interactor:
                def SetInteractorStyle(self, style):
                    captured.append(style)

            interactor = _Interactor()

        _install_meridional_mouse(_Plotter())
        style = captured[0]
        with mock.patch("axisymmetric_viewer._meridional_left_down") as down, mock.patch(
            "axisymmetric_viewer._meridional_left_up"
        ) as up:
            style.OnLeftButtonDown()
            style.OnLeftButtonUp()
        down.assert_called_once_with(style)
        up.assert_called_once_with(style)
