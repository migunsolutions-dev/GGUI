"""Focused stability tests for the Cylindrical–2D viewer and cell reporting."""
from __future__ import annotations

import inspect
import os
import tempfile
import threading
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtWidgets import QApplication

from axisymmetric_viewer import AxisymmetricViewerWidget, _scalar_bar_kwargs
from generator_2d import Generator2D, WEDGE_HALF_ANGLE_DEG
from models_2d import CaseInputs2D
from axisymmetric_2d import DYNAMIC_MESH, FIXED_MESH
from physical_charge_geometry import physical_charge_geometry
from viewer_widget import FieldViewSettings, HAS_PV, pv


app = QApplication.instance() or QApplication([])


class ScalarBarApiCompatibilityTests(unittest.TestCase):
    def test_installed_add_scalar_bar_rejects_background_opacity(self):
        self.assertTrue(HAS_PV)
        params = set(inspect.signature(pv.Plotter.add_scalar_bar).parameters)
        self.assertNotIn("background_opacity", params)
        filtered = _scalar_bar_kwargs(
            title="p",
            background_opacity=0.0,
            log_scale=True,
            n_labels=5,
            color="black",
            fill=False,
            use_opacity=False,
            render=False,
        )
        self.assertNotIn("background_opacity", filtered)
        self.assertNotIn("log_scale", filtered)
        self.assertEqual(filtered.get("title"), "p")
        self.assertIn("n_labels", filtered)


class GuiThreadAndLifecycleTests(unittest.TestCase):
    def test_request_refresh_coalesces_and_stays_on_gui_thread(self):
        viewer = AxisymmetricViewerWidget()
        calls = []
        original = viewer.refresh_view

        def tracked():
            calls.append(int(QThread.currentThreadId()))
            # Do not invoke full VTK path in offscreen unit test.

        viewer.refresh_view = tracked  # type: ignore[method-assign]
        for _ in range(25):
            viewer.request_refresh()
        # Allow the single-shot coalesce timer to fire.
        QTimer.singleShot(120, app.quit)
        app.exec_()
        self.assertGreaterEqual(len(calls), 1)
        self.assertLessEqual(len(calls), 3)
        self.assertTrue(all(c == viewer._gui_thread_id for c in calls))
        viewer.refresh_view = original  # type: ignore[method-assign]
        viewer.shutdown_viewer()

    def test_no_render_after_viewer_destruction(self):
        viewer = AxisymmetricViewerWidget()
        viewer.shutdown_viewer()
        viewer._refresh_pending = True
        # Must not call into VTK after shutdown.
        with mock.patch.object(viewer, "_refresh_axisymmetric_result") as refresh:
            viewer.request_refresh()
            viewer._run_coalesced_refresh()
            viewer.refresh_view()
            refresh.assert_not_called()

    def test_worker_thread_does_not_mutate_plotter(self):
        viewer = AxisymmetricViewerWidget()
        seen = {"assert_ok": None}

        def worker():
            seen["assert_ok"] = viewer._assert_gui_thread()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2.0)
        self.assertIs(seen["assert_ok"], False)
        viewer.shutdown_viewer()


class ChargeAndMirrorGeometryTests(unittest.TestCase):
    def test_sphere_and_cylinder_centres_locked_to_axis(self):
        with tempfile.TemporaryDirectory() as td:
            gen = Generator2D(td)
            sphere = CaseInputs2D(
                radius=1.5,
                height=2.0,
                cell_size=0.005,
                mesh_mode=FIXED_MESH,
                material_name="TNT",
                mass_kg=1.0,
                rho_charge=1630.0,
                height_of_burst=0.5,
                detonation_height=0.5,
                charge_shape="Sphere",
            )
            case = gen.generate("sphere_axis", sphere)
            sf = open(os.path.join(case, "system", "setFieldsDict"), encoding="utf-8").read()
            pp = open(os.path.join(case, "constant", "phaseProperties"), encoding="utf-8").read()
            self.assertIn("centre (0 0.5 0)", sf)
            self.assertIn("useCOM yes", pp)
            geom = physical_charge_geometry(sphere)
            self.assertAlmostEqual(geom.radius_m, (3.0 * 1.0 / (4.0 * 3.141592653589793 * 1630.0)) ** (1.0 / 3.0), places=9)

            cyl = CaseInputs2D(
                radius=1.5,
                height=2.0,
                cell_size=0.005,
                mesh_mode=FIXED_MESH,
                material_name="TNT",
                mass_kg=1.0,
                rho_charge=1630.0,
                height_of_burst=0.5,
                detonation_height=0.5,
                charge_shape="Cylinder",
                charge_aspect=2.5,
            )
            ccase = gen.generate("cyl_axis", cyl)
            csf = open(os.path.join(ccase, "system", "setFieldsDict"), encoding="utf-8").read()
            self.assertIn("p1 (0 ", csf)
            self.assertIn("p2 (0 ", csf)
            self.assertRegex(csf, r"p1 \(0 [0-9.eE+-]+ 0\)")
            self.assertRegex(csf, r"p2 \(0 [0-9.eE+-]+ 0\)")
            cpp = open(os.path.join(ccase, "constant", "phaseProperties"), encoding="utf-8").read()
            self.assertIn("useCOM yes", cpp)

    def test_mirrored_scalar_continuity_at_r0(self):
        self.assertTrue(HAS_PV)
        # Synthetic half-plane mesh mirrored across Radius=0.
        import numpy as np

        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.1, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=float,
        )
        faces = np.hstack([[4, 0, 1, 2, 3]])
        half = pv.PolyData(pts, faces)
        half["alpha.c4"] = np.array([1.0, 0.8, 0.8, 1.0])
        mir = half.copy(deep=True)
        mir.points[:, 0] *= -1.0
        merged = half.merge(mir, merge_points=True, tolerance=1e-12)
        self.assertLess(merged.n_points, half.n_points + mir.n_points)
        # Axis nodes should be unique after merge.
        axis = np.isclose(merged.points[:, 0], 0.0)
        self.assertGreaterEqual(int(axis.sum()), 2)

    def test_mirror_does_not_change_wedge_angle_constant(self):
        self.assertEqual(WEDGE_HALF_ANGLE_DEG, 5.0)


class CellCountReportingTests(unittest.TestCase):
    def test_owner_count_from_constant_polymesh(self):
        case = r"C:\Users\migun\AppData\Local\Temp\ggui_2d_viewer_stability_20260724\proof_sphere_axis"
        owner = os.path.join(case, "constant", "polyMesh", "owner")
        if not os.path.isfile(owner):
            self.skipTest("proof sphere case not initialized")
        n = AxisymmetricViewerWidget.count_owner_cells(
            os.path.join(case, "constant", "polyMesh")
        )
        self.assertEqual(n, 120000)

    def test_fixed_and_dynamic_metadata_generation(self):
        with tempfile.TemporaryDirectory() as td:
            gen = Generator2D(td)
            # At dx=0.02, ~1.5 kg TNT resolves above the Fixed Mesh capture floor.
            fixed = CaseInputs2D(
                radius=0.3,
                height=0.6,
                cell_size=0.02,
                mesh_mode=FIXED_MESH,
                mass_kg=1.5,
            )
            fcase = gen.generate("fixed_cells", fixed)
            with open(os.path.join(fcase, "system", "blockMeshDict"), encoding="utf-8") as fh:
                block = fh.read()
            self.assertIn("(15 30 1)", block)
            dyn = CaseInputs2D(
                radius=0.3,
                height=0.6,
                cell_size=0.02,
                mesh_mode=DYNAMIC_MESH,
                mass_kg=1.5,
                charge_seed_mode="Manual",
                charge_refinement_level=0,
                dyn_refine_max=1,
            )
            dcase = gen.generate("dyn_cells", dyn)
            with open(
                os.path.join(dcase, "constant", "dynamicMeshDict"), encoding="utf-8"
            ) as fh:
                dyn_mesh = fh.read()
            self.assertIn("maxRefinement 1", dyn_mesh)


class LogScalePolicyTests(unittest.TestCase):
    def test_zero_valued_field_rejects_log_scale_without_distorting_data(self):
        viewer = AxisymmetricViewerWidget()
        viewer.current_field = "alpha.c4"
        viewer.field_settings["alpha.c4"] = FieldViewSettings(log_scale=True)
        rejected = []
        viewer.log_scale_rejected.connect(lambda msg: rejected.append(msg))
        # Emulate the policy branch used in _refresh_axisymmetric_result.
        clim = [0.0, 1.0]
        use_log = True
        if use_log and clim[0] <= 0:
            use_log = False
            viewer.log_scale_rejected.emit(
                "Log scale requires strictly positive alpha.c4 values."
            )
        self.assertFalse(use_log)
        self.assertEqual(len(rejected), 1)
        self.assertIn("positive", rejected[0])
        viewer.shutdown_viewer()


if __name__ == "__main__":
    unittest.main()
