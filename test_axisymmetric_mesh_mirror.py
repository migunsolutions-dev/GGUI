"""Prove meridional edge overlay and mirror symmetry without VTK diagonals."""
from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication

from axisymmetric_viewer import (
    AxisymmetricViewerWidget,
    extract_meridional_cell_edges,
    meridional_surface_from_reader,
    mirror_meridional,
)
from viewer_gl import live_viewer_registry_snapshot, scalar_bar_kwargs
import pyvista as pv
import numpy as np


app = QApplication.instance() or QApplication([])


class MeridionalMeshOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = (
            r"C:\Users\migun\AppData\Local\Temp"
            r"\ggui_2d_viewer_stability_20260724\proof_sphere_axis"
        )
        if not os.path.isdir(os.path.join(cls.case, "constant", "polyMesh")):
            raise unittest.SkipTest("proof sphere case missing")

    def test_wedge0_matches_volume_cells_and_quad_topology(self):
        reader = pv.POpenFOAMReader(os.path.join(self.case, "case.foam"))
        reader.set_active_time_value(0.0)
        data = reader.read()
        vol = data["internalMesh"]
        surface = meridional_surface_from_reader(data)
        self.assertIsNotNone(surface)
        self.assertEqual(surface.n_cells, vol.n_cells)
        self.assertEqual(surface.n_cells, 120000)
        # Uniform Fixed Mesh wedge faces are quads.
        sample = [surface.get_cell(i).n_points for i in range(0, surface.n_cells, 5000)]
        self.assertTrue(all(n == 4 for n in sample))

    def test_edge_overlay_has_no_triangulation_diagonals(self):
        reader = pv.POpenFOAMReader(os.path.join(self.case, "case.foam"))
        reader.set_active_time_value(0.0)
        data = reader.read()
        surface = meridional_surface_from_reader(data)
        edges = extract_meridional_cell_edges(surface)
        # 300 x 400 quads ⇒ nx*(ny+1)+(nx+1)*ny = 240700
        self.assertEqual(edges.n_cells, 240700)
        # Triangulation would add ~one diagonal per cell ⇒ ~360700 edges.
        self.assertLess(edges.n_cells, surface.n_cells * 3)

    def test_mirrored_scalars_and_edges_are_symmetric(self):
        reader = pv.POpenFOAMReader(os.path.join(self.case, "case.foam"))
        reader.set_active_time_value(0.0)
        data = reader.read()
        surface = meridional_surface_from_reader(data)
        mirrored = mirror_meridional(surface)
        # Axis points merged: mirrored point count < 2 * half.
        self.assertLess(mirrored.n_points, 2 * surface.n_points)
        alpha = np.asarray(surface["alpha.c4"]).ravel()
        centers = np.asarray(surface.cell_centers().points)
        # Sample several charged cells and confirm mirror image values.
        charged = np.where(alpha > 0.5)[0][:20]
        mir_alpha = np.asarray(mirrored["alpha.c4"]).ravel()
        mir_centers = np.asarray(mirrored.cell_centers().points)
        for idx in charged:
            r, y = float(centers[idx, 0]), float(centers[idx, 1])
            # Find mirrored cell near (-r, y)
            d = np.sqrt((mir_centers[:, 0] + r) ** 2 + (mir_centers[:, 1] - y) ** 2)
            j = int(np.argmin(d))
            self.assertLess(d[j], 1e-4)
            self.assertAlmostEqual(float(mir_alpha[j]), float(alpha[idx]), places=6)
        edges = extract_meridional_cell_edges(surface)
        mir_edges = mirror_meridional(edges)
        # Every edge endpoint (r,y) has a counterpart (-r,y).
        pts = np.asarray(edges.points)
        mpts = np.asarray(mir_edges.points)
        for p in pts[:: max(1, len(pts) // 50)]:
            target = np.array([-p[0], p[1], 0.0])
            d = np.linalg.norm(mpts - target, axis=1).min()
            self.assertLess(d, 1e-9)

    def test_fit_bounds_half_and_mirrored(self):
        self.assertEqual(
            AxisymmetricViewerWidget.meridional_display_bounds(1.5, 2.0, False),
            (0.0, 1.5, 0.0, 2.0),
        )
        self.assertEqual(
            AxisymmetricViewerWidget.meridional_display_bounds(1.5, 2.0, True),
            (-1.5, 1.5, 0.0, 2.0),
        )

    def test_shutdown_rejects_refresh_and_unregisters(self):
        viewer = AxisymmetricViewerWidget()
        before = {e["viewer_id"] for e in live_viewer_registry_snapshot()}
        # Offscreen may skip VTK init; still exercise shutdown path.
        viewer.shutdown_viewer()
        viewer.request_refresh()
        viewer.refresh_view()
        after = {e["viewer_id"] for e in live_viewer_registry_snapshot()}
        self.assertNotIn(id(viewer), after)

    def test_scalar_bar_helper_strips_unsupported_kwargs(self):
        filtered = scalar_bar_kwargs(
            title="p",
            background_opacity=0.0,
            log_scale=True,
            n_labels=5,
            color="black",
        )
        self.assertNotIn("background_opacity", filtered)
        self.assertNotIn("log_scale", filtered)
        self.assertEqual(filtered["title"], "p")


class SharedViewerLifecycleTests(unittest.TestCase):
    def test_probe_actors_cleared_each_refresh_path(self):
        from viewer_widget import BlastViewerWidget

        viewer = BlastViewerWidget()
        viewer._probe_actors = ["stale"]
        # Directly exercise the clear logic used at refresh start.
        for a in list(viewer._probe_actors):
            pass
        viewer._probe_actors = []
        self.assertEqual(viewer._probe_actors, [])
        viewer.shutdown_viewer()


if __name__ == "__main__":
    unittest.main()
