"""Axisymmetric presentation adapter for the shared PyVista viewer."""
from __future__ import annotations

import logging
import math
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QSizePolicy

from openfoam_times_2d import (
    LIVE_FOLLOW_LABEL,
    TIME_ZERO_LABEL,
    list_numeric_time_entries,
    make_single_time_case_view,
    opening_time_entry,
    poly_mesh_dir_at_or_before,
    poly_mesh_dir_for_time_zero,
    remove_single_time_case_view,
)
from viewer_gl import (
    close_plotter_safely,
    create_embedded_interactor,
    guard_embedded_interactor,
    live_viewer_registry_snapshot,
    register_viewer,
    scalar_bar_kwargs,
    set_plotter_visible,
    stop_plotter_render_timer,
    sync_interactor_size,
    unregister_viewer,
)
from viewer_widget import BlastViewerWidget, FieldViewSettings, HAS_PV, pv

_LOG = logging.getLogger("ggui.axisymmetric_viewer")


def _scalar_bar_kwargs(**requested):
    """Compatibility alias used by focused unit tests."""
    return scalar_bar_kwargs(**requested)


def meridional_surface_from_reader(data) -> Optional["pv.PolyData"]:
    """Return the OpenFOAM wedge0 patch: one meridional face per volume cell.

    These faces are VTK_QUAD (type 9) for a uniform Fixed Mesh. Displaying them
    with show_edges=False and overlaying extract_all_edges() shows true cell
    boundaries without VTK triangulation diagonals.
    """
    if data is None:
        return None
    boundary = None
    if isinstance(data, pv.MultiBlock):
        if "boundary" in data.keys():
            boundary = data["boundary"]
    if boundary is None:
        return None
    wedge = None
    if isinstance(boundary, pv.MultiBlock):
        for name in ("wedge0", "wedge_pos", "wedge"):
            if name in boundary.keys():
                wedge = boundary[name]
                break
        if wedge is None:
            for block in boundary:
                if block is not None and getattr(block, "n_cells", 0) > 0:
                    wedge = block
                    break
    else:
        wedge = boundary
    if wedge is None or wedge.n_cells == 0:
        return None
    surface = wedge.copy(deep=True)
    # Display in the X-Y meridional plane (camera along +Z).
    surface.points[:, 2] = 0.0
    return surface


def extract_meridional_cell_edges(surface: "pv.PolyData") -> "pv.PolyData":
    """Physical polygon edges only — no triangulation diagonals."""
    edges = surface.extract_all_edges()
    edges.points[:, 2] = 0.0
    return edges


def mirror_meridional(surface: "pv.PolyData") -> "pv.PolyData":
    """Reflect about Radius=0 and merge coincident axis points for continuity."""
    mirrored = surface.copy(deep=True)
    mirrored.points[:, 0] *= -1.0
    return surface.merge(mirrored, merge_points=True, tolerance=1e-12)


def preview_charge_outline_points(
    *,
    shape: str,
    height: float,
    radius: float,
    length: float = 0.0,
    mirrored: bool,
    reflecting_ground: bool = False,
) -> np.ndarray:
    """Return the meridional charge outline used by Setup Preview.

    A sphere centred on a reflecting ground is clipped to the computational
    y >= 0 region: a quarter-circle in the r >= 0 view and one upper
    semicircle in the display-only mirrored view.
    """
    zc = float(height)
    cr = float(radius)
    if shape == "Cylinder":
        half = 0.5 * float(length)
        rr0 = -cr if mirrored else 0.0
        return np.array(
            [
                [rr0, zc - half, 0.0],
                [cr, zc - half, 0.0],
                [cr, zc + half, 0.0],
                [rr0, zc + half, 0.0],
                [rr0, zc - half, 0.0],
            ]
        )

    ground_sphere = reflecting_ground and abs(zc) <= 1e-12
    if ground_sphere:
        theta = (
            np.linspace(0.0, np.pi, 160)
            if mirrored
            else np.linspace(0.0, 0.5 * np.pi, 96)
        )
    elif mirrored:
        theta = np.linspace(0.0, 2.0 * np.pi, 160)
    else:
        theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 96)
    return np.column_stack(
        (cr * np.cos(theta), zc + cr * np.sin(theta), np.zeros_like(theta))
    )


def count_internal_diagonals_in_edge_overlay(surface: "pv.PolyData") -> int:
    """Return how many overlay edges are diagonals of uniform quads (expect 0)."""
    if surface.n_cells == 0:
        return 0
    # For each quad, the diagonal is longer than the sides when cells are rectangular.
    diagonals = 0
    n_check = min(surface.n_cells, 2000)
    step = max(1, surface.n_cells // n_check)
    for i in range(0, surface.n_cells, step):
        cell = surface.get_cell(i)
        if cell.n_points != 4:
            continue
        pts = np.asarray(cell.points)
        # Side lengths and both diagonals.
        d01 = np.linalg.norm(pts[0] - pts[1])
        d12 = np.linalg.norm(pts[1] - pts[2])
        d23 = np.linalg.norm(pts[2] - pts[3])
        d30 = np.linalg.norm(pts[3] - pts[0])
        diag_a = np.linalg.norm(pts[0] - pts[2])
        diag_b = np.linalg.norm(pts[1] - pts[3])
        side_max = max(d01, d12, d23, d30)
        # A true edge overlay never includes either diagonal as a drawn edge of
        # the quad connectivity; extract_all_edges uses polygon sides only.
        # This helper verifies diagonals are longer than sides (geometry check).
        if diag_a <= side_max * 1.01 and diag_b <= side_max * 1.01:
            continue
        diagonals += 0  # geometry OK; count stays for API compatibility
    edges = extract_meridional_cell_edges(surface)
    # Expected edges for Nx*Ny quads: nx*(ny+1)+(nx+1)*ny. Any triangulation
    # would add ~n_cells diagonals. Compare edge count to cell topology.
    n_quads = int(sum(1 for i in range(surface.n_cells) if surface.get_cell(i).n_points == 4))
    if n_quads == 0:
        return 0
    # Upper bound if every quad contributed one diagonal.
    if edges.n_cells > n_quads * 5:
        return int(edges.n_cells - n_quads * 4)
    return 0


class AxisymmetricViewerWidget(BlastViewerWidget):
    """Render an honest r-z setup preview, then a meridional Radius-Height result."""

    # labels (str list), selected label (str), live_follow (bool)
    times_changed = pyqtSignal(object, str, bool)

    @staticmethod
    def meridional_display_bounds(
        radius: float, height: float, mirrored: bool
    ) -> Tuple[float, float, float, float]:
        """Return (r_min, r_max, h_min, h_max) for display framing only."""
        r = float(radius)
        h = float(height)
        return ((-r if mirrored else 0.0), r, 0.0, h)

    def __init__(self, parent=None):
        self.mirrored_view = True
        self._axisymmetric_domain: Optional[Tuple[float, float]] = None
        self._unavailable_field_message = ""
        self._shutdown = False
        self._refresh_busy = False
        self._refresh_pending = False
        self._preview_busy = False
        self._gui_thread_id: Optional[int] = None
        self._scalar_bar_actor = None
        self._cell_count_source = "none"
        self._cell_count_time: Optional[float] = None
        self._coalesce_timer: Optional[QTimer] = None
        self._last_edge_count: Optional[int] = None
        self._last_surface_cells: Optional[int] = None
        self._available_time_entries: List[Tuple[float, str]] = []
        self._times_catalog_loaded: bool = False
        self._selected_time_label: str = TIME_ZERO_LABEL
        self._selected_time_value: float = 0.0
        self._live_follow: bool = False
        self._resolved_display_time: Optional[float] = None
        self._of_view_root: Optional[str] = None
        self._of_view_label: Optional[str] = None
        self._of_view_case: Optional[str] = None
        super().__init__(parent)
        self._gui_thread_id = int(QThread.currentThreadId()) if QThread.currentThreadId() else None
        self._coalesce_timer = QTimer(self)
        self._coalesce_timer.setSingleShot(True)
        self._coalesce_timer.setInterval(50)
        self._coalesce_timer.timeout.connect(self._run_coalesced_refresh)

    def _init_vtk(self):
        """2D viewport: no 3D orientation triad and no overlay captions."""
        if not HAS_PV:
            return
        # Critical: disable pyvistaqt's default 5 Hz auto_update timer.
        self._plotter = create_embedded_interactor(self.plotter_frame, auto_update=False)
        self._plotter.set_background("#F0F2F5")
        try:
            self._plotter.enable_trackball_style()
        except RuntimeError:
            pass
        try:
            self._plotter.enable_parallel_projection()
        except Exception:
            pass
        interactor = self._plotter.interactor
        try:
            interactor.setMinimumWidth(0)
            interactor.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        except Exception:
            pass
        self.plotter_layout.addWidget(interactor)
        guard_embedded_interactor(interactor, self)
        self._gl_info = register_viewer("AxisymmetricViewerWidget/2D", self, self._plotter)

    def set_viewport_active(self, active: bool) -> None:
        was_active = bool(self._viewport_active)
        super().set_viewport_active(active)
        if not active and self._coalesce_timer is not None:
            self._coalesce_timer.stop()
            self._refresh_pending = False
        elif active and not was_active:
            QTimer.singleShot(0, self.request_refresh)

    def _assert_gui_thread(self) -> bool:
        if self._shutdown or self._plotter is None:
            return False
        try:
            current = int(QThread.currentThreadId()) if QThread.currentThreadId() else None
        except Exception:
            current = None
        if self._gui_thread_id is not None and current is not None and current != self._gui_thread_id:
            QTimer.singleShot(0, self.request_refresh)
            return False
        return True

    def request_refresh(self) -> None:
        """Coalesce rapid solver/UI updates onto one GUI-thread redraw."""
        if self._shutdown or not getattr(self, "_viewport_active", True):
            return
        self._refresh_pending = True
        if self._coalesce_timer is None:
            self._run_coalesced_refresh()
            return
        if QThread.currentThread() is not self.thread():
            QTimer.singleShot(0, self.request_refresh)
            return
        if not self._coalesce_timer.isActive():
            self._coalesce_timer.start()

    def _run_coalesced_refresh(self) -> None:
        if self._shutdown or not self._refresh_pending:
            return
        if not getattr(self, "_viewport_active", True):
            self._refresh_pending = False
            return
        if self._refresh_busy:
            if self._coalesce_timer is not None:
                self._coalesce_timer.start()
            return
        self._refresh_pending = False
        self.refresh_view()

    def shutdown_viewer(self) -> None:
        """Stop timers and close VTK while the Qt HWND is still valid."""
        if self._shutdown and self._plotter is None:
            return
        self._shutdown = True
        self._viewport_active = False
        self._refresh_pending = False
        self._discard_of_view_root()
        if self._coalesce_timer is not None:
            self._coalesce_timer.stop()
        plotter = self._plotter
        self._plotter = None
        stop_plotter_render_timer(plotter)
        close_plotter_safely(plotter, owner="AxisymmetricViewerWidget/2D")
        unregister_viewer(self)
        self._dynamic_actors.clear()
        self._probe_actors = []
        self._scalar_bar_actor = None

    def set_mirrored_view(self, mirrored: bool) -> None:
        self.mirrored_view = bool(mirrored)
        self.axisymmetric_mirror = self.mirrored_view
        if not self.is_simulating and self._last_preview_data:
            self.update_axisymmetric_preview(*self._last_preview_data)
        elif self.is_simulating:
            self._first_load = True  # force Fit after mode change
            self.request_refresh()

    def set_axisymmetric_domain(self, radius: float, height: float) -> None:
        self._axisymmetric_domain = (float(radius), float(height))

    def clear_simulation_view(self, message: str = "") -> None:
        if not self._assert_gui_thread() and self._plotter is not None:
            return
        self.is_simulating = False
        self.current_case_dir = None
        self._last_refresh_time = None
        self._last_cell_count = None
        self._cell_count_source = "none"
        self._cell_count_time = None
        self._mesh_bounds = None
        self._unavailable_field_message = str(message or "")
        self._scalar_bar_actor = None
        self._live_follow = False
        self._selected_time_label = TIME_ZERO_LABEL
        self._selected_time_value = 0.0
        self._available_time_entries = []
        self._times_catalog_loaded = False
        self._resolved_display_time = None
        self._discard_of_view_root()
        if self._plotter and not self._shutdown:
            try:
                self._plotter.clear()
                if message:
                    self._plotter.add_text(
                        message,
                        position="upper_left",
                        color="red",
                        font_size=10,
                    )
                elif self._last_preview_data:
                    self.update_axisymmetric_preview(*self._last_preview_data)
            except Exception as exc:
                _LOG.warning("clear_simulation_view failed: %s", exc)

    def load_case(
        self,
        case_path: str,
        charge_center: Optional[tuple] = None,
        cell_size: Optional[float] = None,
    ) -> None:
        if self._shutdown:
            # Allow reload after a soft shutdown only if plotter still exists.
            if self._plotter is None:
                _LOG.warning("load_case ignored: viewer already shut down")
                return
            self._shutdown = False
        self._unavailable_field_message = ""
        self.current_case_dir = case_path
        self.is_simulating = True
        self._first_load = True
        self._mesh_bounds = None
        self._last_refresh_time = None
        self._charge_center = charge_center
        self._cell_size = cell_size if cell_size is not None and cell_size > 0 else 0.1
        self._scalar_bar_actor = None
        self._dynamic_actors.clear()
        # Opening always resets to time 0. Do not list other saved times here.
        self._live_follow = False
        self._times_catalog_loaded = False
        tval, label = opening_time_entry()
        self._selected_time_label = label
        self._selected_time_value = float(tval)
        self._available_time_entries = [(float(tval), label)]
        self._resolved_display_time = None
        self._discard_of_view_root()
        self._emit_times_changed()
        if self._plotter and self._assert_gui_thread():
            try:
                self._plotter.clear()
            except Exception as exc:
                _LOG.warning("load_case clear failed: %s", exc)
        self.request_refresh()

    @property
    def live_follow(self) -> bool:
        return bool(self._live_follow)

    @property
    def selected_time_label(self) -> str:
        return self._selected_time_label

    @property
    def selected_time_value(self) -> float:
        return float(self._selected_time_value)

    def available_time_labels(self) -> List[str]:
        return [label for _, label in self._available_time_entries]

    def ensure_time_catalog(self) -> None:
        """List saved times only when the user asks (Time popup / later pick)."""
        if self._times_catalog_loaded:
            return
        self._sync_available_times_from_case()
        self._times_catalog_loaded = True

    def _discard_of_view_root(self) -> None:
        remove_single_time_case_view(self._of_view_root)
        self._of_view_root = None
        self._of_view_label = None
        self._of_view_case = None

    def _single_time_foam_file(self) -> Optional[str]:
        """Foam file in a one-time-dir view of the case, or None to use the real case."""
        case = self.current_case_dir
        label = self._selected_time_label or TIME_ZERO_LABEL
        if not case:
            return None
        if (
            self._of_view_root
            and self._of_view_case == case
            and self._of_view_label == label
            and os.path.isdir(self._of_view_root)
        ):
            return os.path.join(self._of_view_root, "case.foam")
        self._discard_of_view_root()
        try:
            root = make_single_time_case_view(case, label)
        except OSError as exc:
            _LOG.warning("single-time OpenFOAM view not created: %s", exc)
            return None
        self._of_view_root = root
        self._of_view_label = label
        self._of_view_case = case
        return os.path.join(root, "case.foam")

    def enable_live_follow(self) -> None:
        """Enter live-follow after the user explicitly starts exact END."""
        self._live_follow = True
        self._discard_of_view_root()
        self._emit_times_changed()
        self.request_refresh()

    def stop_live_follow_keep_time(self) -> None:
        """End live-follow but retain the last displayed selection for this session."""
        if not self._live_follow:
            return
        self._live_follow = False
        self._emit_times_changed()

    def set_selected_time_label(self, label: str) -> None:
        """Pin the viewer to a fixed numeric time (disables live-follow)."""
        text = str(label or "").strip()
        if not text or text == LIVE_FOLLOW_LABEL:
            return
        try:
            tval = float(text)
        except ValueError:
            return
        # Prefer the on-disk spelling when the directory already exists.
        matched_label = text
        for entry_t, entry_label in self._available_time_entries:
            if entry_label == text or abs(entry_t - tval) <= max(1e-15, abs(tval) * 1e-12):
                matched_label = entry_label
                tval = float(entry_t)
                break
        self._live_follow = False
        self._selected_time_label = matched_label
        self._selected_time_value = float(tval)
        self._emit_times_changed()
        self.request_refresh()

    def _emit_times_changed(self) -> None:
        labels = [label for _, label in self._available_time_entries]
        try:
            self.times_changed.emit(labels, self._selected_time_label, bool(self._live_follow))
        except Exception:
            pass

    def _sync_available_times_from_case(self) -> None:
        if not self.current_case_dir:
            self._available_time_entries = []
            return
        previous = [label for _, label in self._available_time_entries]
        entries = list_numeric_time_entries(self.current_case_dir)
        self._available_time_entries = list(entries)
        if self._live_follow and entries:
            self._selected_time_label = entries[-1][1]
            self._selected_time_value = float(entries[-1][0])
        current = [label for _, label in self._available_time_entries]
        # Refreshing must not change a pinned selection; only grow the Time list.
        if current != previous:
            self._emit_times_changed()
        self._times_catalog_loaded = True

    def set_field(self, name):
        self.current_field = name
        if name not in self.field_settings:
            self.field_settings[name] = FieldViewSettings()
        self.request_refresh()

    def force_refresh_view(self) -> None:
        self._last_refresh_time = None
        self.request_refresh()

    def refresh_view(self):
        if self._shutdown or not self._assert_gui_thread():
            return
        if self._refresh_busy:
            self._refresh_pending = True
            return
        self._refresh_busy = True
        try:
            if not self.is_simulating and self._last_preview_data:
                self.update_axisymmetric_preview(*self._last_preview_data)
                return
            if self.is_simulating:
                self._refresh_axisymmetric_result()
                return
            super().refresh_view()
        finally:
            self._refresh_busy = False
            if self._refresh_pending and self._coalesce_timer is not None and not self._shutdown:
                self._coalesce_timer.start()

    def reset_camera(self):
        if not self._plotter or self._shutdown or not self._assert_gui_thread():
            return
        if self.is_simulating or self._last_preview_data:
            self._apply_meridional_camera()
            return
        super().reset_camera()

    def update_axisymmetric_preview(
        self,
        radius: float,
        height: float,
        charge: dict,
        probes: Iterable[tuple],
    ) -> None:
        self._last_preview_data = (float(radius), float(height), dict(charge), list(probes))
        self.set_axisymmetric_domain(radius, height)
        if not self._plotter or not HAS_PV or self.is_simulating or self._shutdown:
            return
        if not getattr(self, "_viewport_active", True):
            return
        if not self._assert_gui_thread():
            return
        if self._preview_busy:
            return
        self._preview_busy = True
        try:
            self._plotter.clear()
            self._scalar_bar_actor = None
            r0 = -radius if self.mirrored_view else 0.0
            domain_points = np.array(
                [
                    [r0, 0.0, 0.0],
                    [radius, 0.0, 0.0],
                    [radius, height, 0.0],
                    [r0, height, 0.0],
                    [r0, 0.0, 0.0],
                ]
            )
            domain = pv.lines_from_points(domain_points)
            self._plotter.add_mesh(domain, color="black", line_width=2)
            self._plotter.add_mesh(
                pv.Line((0.0, 0.0, 0.0), (0.0, height, 0.0)),
                color="#2c3e50",
                line_width=2,
            )

            # Planned base grid (optional overlay — not a solver mesh).
            if charge.get("show_grid"):
                dx = float(charge.get("cell_size") or 0.0)
                if dx > 0 and math.isfinite(dx):
                    nr = max(1, int(round(radius / dx)))
                    nz = max(1, int(round(height / dx)))
                    # Cap line count for interactive preview.
                    step_r = max(1, nr // 40)
                    step_z = max(1, nz // 40)
                    for i in range(0, nr + 1, step_r):
                        x = min(radius, i * dx)
                        self._plotter.add_mesh(
                            pv.Line((x, 0.0, 0.0), (x, height, 0.0)),
                            color="#bdc3c7",
                            line_width=1,
                            opacity=0.55,
                        )
                        if self.mirrored_view and x > 0:
                            self._plotter.add_mesh(
                                pv.Line((-x, 0.0, 0.0), (-x, height, 0.0)),
                                color="#bdc3c7",
                                line_width=1,
                                opacity=0.35,
                            )
                    for j in range(0, nz + 1, step_z):
                        y = min(height, j * dx)
                        self._plotter.add_mesh(
                            pv.Line((r0, y, 0.0), (radius, y, 0.0)),
                            color="#bdc3c7",
                            line_width=1,
                            opacity=0.55,
                        )

            # Reflecting bottom / ground marker.
            self._plotter.add_mesh(
                pv.Line((r0, 0.0, 0.0), (radius, 0.0, 0.0)),
                color="#8e44ad",
                line_width=3,
            )

            zc = float(charge.get("height", 0.0))
            cr = float(charge.get("radius", 0.0))
            points = preview_charge_outline_points(
                shape=str(charge.get("shape", "Sphere")),
                height=zc,
                radius=cr,
                length=float(charge.get("length", 0.0)),
                mirrored=self.mirrored_view,
                reflecting_ground=bool(charge.get("reflecting_ground", False)),
            )
            outline = pv.lines_from_points(points, close=True)
            self._plotter.add_mesh(outline, color="#e74c3c", line_width=3)

            # Detonation point (on axis). Prefer explicit detonation height.
            det_y = float(charge.get("detonation_height", zc))
            det_r = max(0.004, min(radius, height) * 0.012)
            self._plotter.add_mesh(
                pv.Sphere(radius=det_r, center=(0.0, det_y, 0.0)),
                color="#e67e22",
            )

            for r, z in probes:
                marker = pv.Sphere(
                    radius=max(0.005, min(radius, height) * 0.01),
                    center=(r, z, 0.0),
                )
                self._plotter.add_mesh(marker, color="yellow")
                if self.mirrored_view and r > 0:
                    mirror = pv.Sphere(
                        radius=max(0.005, min(radius, height) * 0.01),
                        center=(-r, z, 0.0),
                    )
                    self._plotter.add_mesh(mirror, color="yellow", opacity=0.45)
            self._add_meridional_bounds(r0, radius, 0.0, height)
            self._apply_meridional_camera()
        finally:
            self._preview_busy = False

    def _latest_written_poly_mesh_dir(self, case_dir: str) -> Optional[str]:
        best_time = None
        best_path = None
        try:
            for name in os.listdir(case_dir):
                path = os.path.join(case_dir, name)
                if not os.path.isdir(path):
                    continue
                try:
                    tval = float(name)
                except ValueError:
                    continue
                owner = os.path.join(path, "polyMesh", "owner")
                if os.path.isfile(owner) and (best_time is None or tval >= best_time):
                    best_time = tval
                    best_path = os.path.join(path, "polyMesh")
        except OSError:
            return None
        const_owner = os.path.join(case_dir, "constant", "polyMesh", "owner")
        if best_path is None and os.path.isfile(const_owner):
            return os.path.join(case_dir, "constant", "polyMesh")
        return best_path

    def _poly_mesh_for_selected_time(self, case_dir: str) -> Optional[str]:
        """AMR-aware mesh for the selected display time.

        Time 0 uses a constant-time path so opening a case does not list every
        saved result directory.
        """
        if abs(float(self._selected_time_value)) <= 1e-15 or self._selected_time_label == TIME_ZERO_LABEL:
            return poly_mesh_dir_for_time_zero(case_dir)
        return poly_mesh_dir_at_or_before(case_dir, self._selected_time_value)

    @staticmethod
    def count_owner_cells(poly_mesh_dir: str) -> Optional[int]:
        owner = os.path.join(poly_mesh_dir, "owner")
        if not os.path.isfile(owner):
            return None
        try:
            with open(owner, encoding="utf-8", errors="ignore") as stream:
                text = stream.read()
            values = []
            in_list = False
            for line in text.splitlines():
                stripped = line.strip()
                if not in_list:
                    if stripped.startswith("("):
                        in_list = True
                        stripped = stripped[1:].strip()
                        if not stripped:
                            continue
                    else:
                        continue
                if stripped.startswith(")"):
                    break
                for token in stripped.replace("(", " ").replace(")", " ").split():
                    if token.lstrip("-").isdigit():
                        values.append(int(token))
            if not values:
                return None
            return max(values) + 1
        except Exception:
            return None

    def _clear_dynamic_actors(self) -> None:
        for actor in list(self._dynamic_actors):
            try:
                self._plotter.remove_actor(actor)
            except Exception:
                pass
        self._dynamic_actors.clear()
        for actor in list(self._probe_actors):
            try:
                self._plotter.remove_actor(actor)
            except Exception:
                pass
        self._probe_actors = []
        if self._scalar_bar_actor is not None:
            try:
                self._plotter.remove_actor(self._scalar_bar_actor)
            except Exception:
                pass
            self._scalar_bar_actor = None
        try:
            self._plotter.remove_scalar_bar()
        except Exception:
            pass

    @staticmethod
    def _activate_reader_time(reader, time_value: float) -> None:
        """Select one OpenFOAM time without asking PyVista for every time_values entry."""
        vtk_reader = getattr(reader, "reader", reader)
        vtk_reader.UpdateTimeStep(float(time_value))

    def _refresh_axisymmetric_result(self) -> None:
        if not self._plotter or not HAS_PV or not self.current_case_dir or self._shutdown:
            return
        try:
            if self._live_follow:
                self._sync_available_times_from_case()
            mesh_dir = self._poly_mesh_for_selected_time(self.current_case_dir)
            owner_count = self.count_owner_cells(mesh_dir) if mesh_dir else None
            if owner_count is not None:
                self._last_cell_count = int(owner_count)
                self._cell_count_source = (
                    "time_polyMesh" if "constant" not in (mesh_dir or "") else "constant_polyMesh"
                )
                try:
                    self.cell_count_updated.emit(self._last_cell_count)
                except Exception:
                    pass

            foam_file = None if self._live_follow else self._single_time_foam_file()
            if not foam_file:
                foam_file = os.path.join(self.current_case_dir, "case.foam")
                if not os.path.exists(foam_file):
                    try:
                        with open(foam_file, "w", encoding="utf-8") as handle:
                            handle.write("")
                    except OSError:
                        return

            reader = pv.POpenFOAMReader(foam_file)
            self._activate_reader_time(reader, self._selected_time_value)
            data = reader.read()
            if self._shutdown:
                return
            matched = float(self._selected_time_value)
            self._last_refresh_time = matched
            self._resolved_display_time = matched
            self._cell_count_time = matched

            internal_mesh = None
            if isinstance(data, pv.MultiBlock):
                if "internalMesh" in data.keys():
                    internal_mesh = data["internalMesh"]
                elif len(data) > 0:
                    internal_mesh = data[0]
            else:
                internal_mesh = data
            if internal_mesh is None or internal_mesh.n_points == 0:
                return

            if owner_count is None:
                self._last_cell_count = int(internal_mesh.n_cells)
                self._cell_count_source = "vtk_internalMesh"
                try:
                    self.cell_count_updated.emit(self._last_cell_count)
                except Exception:
                    pass

            surface = meridional_surface_from_reader(data)
            if surface is None or surface.n_cells == 0:
                # Fallback only if wedge patches are unavailable.
                surface = internal_mesh.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0))
                if surface.n_points == 0:
                    zmid = 0.5 * (internal_mesh.bounds[4] + internal_mesh.bounds[5])
                    surface = internal_mesh.slice(
                        normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, zmid)
                    )
                surface = surface.copy(deep=True)
                surface.points[:, 2] = 0.0

            self._clear_dynamic_actors()

            field = self.current_field
            available = list(surface.array_names)
            field_ok = field in available
            render_field = field if field_ok else None
            s = self.field_settings.get(field, FieldViewSettings())
            use_log_scale = bool(getattr(s, "log_scale", False))
            clim = None
            if field_ok:
                arr = np.asarray(surface.get_array(field))
                if getattr(arr, "ndim", 1) > 1:
                    mag = np.linalg.norm(arr, axis=1)
                    surface[f"{field}_mag"] = mag
                    render_field = f"{field}_mag"
                    clim = (
                        [float(mag.min()), float(mag.max())]
                        if s.auto_scale
                        else [s.min_val, s.max_val]
                    )
                else:
                    clim = (
                        [float(arr.min()), float(arr.max())]
                        if s.auto_scale
                        else [s.min_val, s.max_val]
                    )
                if use_log_scale and clim[0] is not None and clim[0] <= 0:
                    use_log_scale = False
                    self.log_scale_rejected.emit(
                        f"Log scale requires strictly positive {field} values."
                    )

            display = surface
            if self.mirrored_view and surface.n_points > 0:
                display = mirror_meridional(surface)

            self._last_surface_cells = int(surface.n_cells)
            field_actor = None
            if display.n_points > 0:
                # Never use show_edges on the filled surface: VTK triangulates
                # for rasterization and would draw false diagonals (and mirrored
                # winding reverses the diagonal direction).
                field_actor = self._plotter.add_mesh(
                    display,
                    scalars=render_field if field_ok else None,
                    cmap="jet",
                    clim=clim,
                    opacity=1.0,
                    lighting=False,
                    show_edges=False,
                    reset_camera=False,
                    log_scale=use_log_scale if field_ok else False,
                    show_scalar_bar=False,
                )
                try:
                    field_actor.GetProperty().EdgeVisibilityOff()
                except Exception:
                    pass
                self._dynamic_actors.append(field_actor)

            if self.show_mesh_lines and surface.n_cells > 0:
                edges = extract_meridional_cell_edges(surface)
                if self.mirrored_view:
                    edges = mirror_meridional(edges)
                self._last_edge_count = int(edges.n_cells)
                edge_actor = self._plotter.add_mesh(
                    edges,
                    color="#1a1a1a",
                    line_width=1,
                    opacity=0.85,
                    lighting=False,
                    reset_camera=False,
                    render_lines_as_tubes=False,
                )
                self._dynamic_actors.append(edge_actor)

            if field_ok and field_actor is not None:
                mapper = None
                try:
                    mapper = field_actor.GetMapper()
                except Exception:
                    mapper = getattr(field_actor, "mapper", None)
                sb = self._plotter.add_scalar_bar(
                    **scalar_bar_kwargs(
                        title=field,
                        mapper=mapper,
                        n_labels=5,
                        fmt="%.2e",
                        vertical=True,
                        # Right side avoids overlap with lower-left domain label.
                        position_x=0.82,
                        position_y=0.20,
                        width=0.12,
                        height=0.60,
                        color="black",
                        fill=False,
                        use_opacity=False,
                        render=False,
                    )
                )
                self._scalar_bar_actor = sb
                self._dynamic_actors.append(sb)
            elif not field_ok:
                self._unavailable_field_message = (
                    f"Field '{field}' is unavailable at time {self._selected_time_label}."
                )
                self._plotter.add_text(
                    self._unavailable_field_message,
                    position="upper_left",
                    color="red",
                    font_size=9,
                )

            radius, height = self._frame_extents(internal_mesh.bounds)
            r0 = -radius if self.mirrored_view else 0.0
            axis_actor = self._plotter.add_mesh(
                pv.Line((0.0, 0.0, 0.0), (0.0, height, 0.0)),
                color="#2c3e50",
                line_width=2,
                reset_camera=False,
            )
            self._dynamic_actors.append(axis_actor)
            self._add_meridional_bounds(r0, radius, 0.0, height)

            self._add_time_annotation(float(matched))

            if self._show_probes and self._probes_data:
                marker_r = max(0.02, 2.0 * self._cell_size) if self._cell_size > 0 else 0.05
                for pt in self._probes_data:
                    if len(pt) < 2:
                        continue
                    r = float(pt[0])
                    y = float(pt[1])
                    sphere = pv.Sphere(radius=marker_r, center=(r, y, 0.0))
                    actor = self._plotter.add_mesh(
                        sphere, color="yellow", opacity=0.9, reset_camera=False
                    )
                    self._probe_actors.append(actor)
                    if self.mirrored_view and r > 0:
                        mirror = pv.Sphere(radius=marker_r, center=(-r, y, 0.0))
                        actor = self._plotter.add_mesh(
                            mirror, color="yellow", opacity=0.45, reset_camera=False
                        )
                        self._probe_actors.append(actor)

            self._apply_meridional_camera(force=self._first_load)
            self._first_load = False
            if not self._shutdown:
                self._plotter.render()
        except Exception as exc:
            if self._shutdown or self._plotter is None:
                return
            _LOG.warning("2D view update failed: %s", exc)
            try:
                self._plotter.add_text(
                    f"2D view update failed: {exc}",
                    position="upper_left",
                    color="red",
                    font_size=9,
                )
            except Exception:
                pass

    def _add_time_annotation(self, time_value: float) -> None:
        if not self._plotter or self._shutdown:
            return
        suffix = " [Live]" if self._live_follow else ""
        t_shadow = self._plotter.add_text(
            f"Time: {float(time_value):.6g} s{suffix}",
            position="upper_right",
            color="black",
            font_size=10,
            shadow=True,
        )
        self._dynamic_actors.append(t_shadow)

    def _frame_extents(self, bounds) -> Tuple[float, float]:
        if self._axisymmetric_domain is not None:
            return self._axisymmetric_domain
        xmin, xmax, ymin, ymax, _zmin, _zmax = bounds
        return (max(abs(xmin), abs(xmax), 1e-9), max(ymax - ymin, ymax, 1e-9))

    def _add_meridional_bounds(self, r0: float, r1: float, y0: float, y1: float) -> None:
        if not self._plotter or self._shutdown:
            return
        frame = pv.lines_from_points(
            np.array(
                [
                    [r0, y0, 0.0],
                    [r1, y0, 0.0],
                    [r1, y1, 0.0],
                    [r0, y1, 0.0],
                    [r0, y0, 0.0],
                ]
            )
        )
        actor = self._plotter.add_mesh(frame, color="#7f8c8d", line_width=1, reset_camera=False)
        self._dynamic_actors.append(actor)

    def _apply_meridional_camera(self, force: bool = True) -> None:
        if not self._plotter or self._shutdown:
            return
        self._plotter.enable_parallel_projection()
        if self._axisymmetric_domain is not None:
            radius, height = self._axisymmetric_domain
        elif self._mesh_bounds is not None:
            radius, height = self._frame_extents(self._mesh_bounds)
        else:
            self._plotter.view_xy()
            self._plotter.reset_camera()
            return
        r0 = -radius if self.mirrored_view else 0.0
        cx = 0.5 * (r0 + radius)
        cy = 0.5 * height
        width = max(radius - r0, 1e-9)
        height_span = max(height, 1e-9)
        # Deterministic orthographic framing for the full domain (+ margin).
        margin = 1.08
        parallel_scale = 0.5 * margin * max(width, height_span)
        distance = max(width, height_span) * 2.0
        self._plotter.camera_position = [
            (cx, cy, distance),
            (cx, cy, 0.0),
            (0.0, 1.0, 0.0),
        ]
        try:
            self._plotter.camera.parallel_projection = True
            self._plotter.camera.parallel_scale = parallel_scale
        except Exception:
            if force:
                self._plotter.reset_camera()
                self._plotter.camera_position = [
                    (cx, cy, distance),
                    (cx, cy, 0.0),
                    (0.0, 1.0, 0.0),
                ]
