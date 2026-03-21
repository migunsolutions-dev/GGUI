import os
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFrame, QLabel
)
from PyQt5.QtCore import pyqtSignal
from dataclasses import dataclass, field
from typing import List, Dict, Optional

HAS_PV = True
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError:
    HAS_PV = False

@dataclass
class ObstacleItem:
    enabled: bool
    path: str
    scale: float
    ox: float
    oy: float
    oz: float

@dataclass
class FieldViewSettings:
    min_val: float = 0.0
    max_val: float = 1.0
    auto_scale: bool = True
    log_scale: bool = False

@dataclass
class SectionItem:
    enabled: bool
    name: str
    origin: List[float]
    normal: List[float]
    opacity: float
    position_m: float = 0.0  # absolute coordinate [m] along plane normal; clamped to mesh bounds

class BlastViewerWidget(QWidget):
    cell_count_updated = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_case_dir = None
        self.is_simulating = False
        self._stl_cache: Dict[str, pv.PolyData] = {}
        self.field_settings: Dict[str, FieldViewSettings] = {}
        self.current_field = "p"
        self._first_load = True
        
        self.show_mesh_lines = False
        self.parallel_projection = False
        self.show_tracers = False
        self.show_boundaries = True
        self.show_obstacles = True
        self.show_obstacles_wireframe_only = False
        self.sections: List[SectionItem] = []
        self._last_preview_data = None
        self._mesh_bounds: Optional[tuple] = None
        self._charge_center: Optional[tuple] = None
        self._cell_size: float = 0.1
        self._probe_actors: List = None
        self._show_probes = False
        self._probes_data: List[tuple] = []

        # Patch names that form the outer domain box (wireframe); all others = obstacles (solid)
        self._domain_patch_names = frozenset({"minX", "maxX", "minY", "maxY", "minZ", "maxZ"})

        self.field_settings["p"] = FieldViewSettings(100000, 200000, True)
        self.field_settings["rho"] = FieldViewSettings(1, 2, True)
        self.field_settings["alpha.c4"] = FieldViewSettings(0, 1, False)
        self.field_settings["U"] = FieldViewSettings(0, 1000, True)

        self._plotter = None
        self._probe_actors = []
        self._last_refresh_time: Optional[float] = None
        self._last_cell_count: Optional[int] = None
        self._dynamic_actors: List = []
        self._obstacle_actors: List = []
        self._init_ui()
        if HAS_PV: self._init_vtk()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.plotter_frame = QFrame()
        self.plotter_layout = QVBoxLayout(self.plotter_frame)
        self.plotter_layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.plotter_frame)

    def _init_vtk(self):
        if not HAS_PV: return
        self._plotter = QtInteractor(self.plotter_frame)
        self._plotter.set_background("#F0F2F5")  # Light bluish-grey (Engineering style)
        self._plotter.add_axes()
        self._plotter.enable_trackball_style()
        self.plotter_layout.addWidget(self._plotter.interactor)

    def set_field(self, name):
        self.current_field = name
        if name not in self.field_settings:
            self.field_settings[name] = FieldViewSettings()
        self.refresh_view()

    def set_field_range(self, mn, mx, auto):
        if self.current_field in self.field_settings:
            s = self.field_settings[self.current_field]
            s.min_val = mn
            s.max_val = mx
            s.auto_scale = auto
            self.refresh_view()
    
    def toggle_mesh_lines(self, state):
        self.show_mesh_lines = state
        self.refresh_view()
    
    def toggle_boundaries(self, state):
        self.show_boundaries = state
        self.refresh_view()

    def force_refresh_view(self) -> None:
        """Force a full redraw (e.g. after viewport option change) even if time step unchanged."""
        self._last_refresh_time = None
        self.refresh_view()

    def toggle_parallel_projection(self, state):
        self.parallel_projection = state
        if self._plotter:
            if state: self._plotter.enable_parallel_projection()
            else: self._plotter.disable_parallel_projection()

    def toggle_tracers(self, state):
        self.show_tracers = state
        self.refresh_view()

    def set_log_scale(self, state: bool) -> None:
        for s in self.field_settings.values():
            s.log_scale = state

    def toggle_probes(self, state: bool, probes_data: Optional[List[tuple]] = None) -> None:
        self._show_probes = bool(state)
        self._probes_data = list(probes_data) if probes_data else []
        if not self._plotter or not HAS_PV:
            return
        for actor in self._probe_actors:
            try:
                self._plotter.remove_actor(actor)
            except Exception:
                pass
        self._probe_actors.clear()
        if self._show_probes and self._probes_data:
            radius = max(0.02, 2.0 * self._cell_size) if self._cell_size > 0 else 0.05
            for pt in self._probes_data:
                if len(pt) < 3:
                    continue
                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                sphere = pv.Sphere(radius=radius, center=(x, y, z))
                actor = self._plotter.add_mesh(sphere, color="yellow", opacity=0.9, reset_camera=False)
                self._probe_actors.append(actor)
        if self.is_simulating:
            self.refresh_view()

    def set_standard_view(self, name):
        if not self._plotter: return
        if name == "Iso": self._plotter.view_isometric()
        elif name == "Top": self._plotter.view_xy()
        elif name == "Front": self._plotter.view_zx()
        elif name == "Right": self._plotter.view_yz()
        elif name == "Bottom":
            self._plotter.view_xy()
            pos, fp, vup = self._plotter.camera_position
            self._plotter.camera_position = (tuple(2 * np.array(fp) - np.array(pos)), fp, vup)
        elif name == "Left":
            self._plotter.view_yz()
            pos, fp, vup = self._plotter.camera_position
            self._plotter.camera_position = (tuple(2 * np.array(fp) - np.array(pos)), fp, vup)
        elif name == "Back":
            self._plotter.view_zx()
            pos, fp, vup = self._plotter.camera_position
            self._plotter.camera_position = (tuple(2 * np.array(fp) - np.array(pos)), fp, vup)
        self._plotter.reset_camera()

    def update_sections(self, sections: List[SectionItem]):
        self.sections = sections
        if self.is_simulating:
            self.refresh_view()
        else:
            if self._last_preview_data:
                self.update_preview(*self._last_preview_data)

    def set_obstacles(self, obstacles: List[ObstacleItem]):
        pass

    def reset_camera(self):
        if self._plotter: self._plotter.reset_camera()

    def update_preview(self, domain_bounds, charge_data, obstacles):
        if not self._plotter or self.is_simulating: return
        
        self._last_preview_data = (domain_bounds, charge_data, obstacles)
        self._plotter.clear()
        
        xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
        
        # Calculate max dimension to ensure plane covers everything
        dx = abs(xmax - xmin)
        dy = abs(ymax - ymin)
        dz = abs(zmax - zmin)
        plane_size = max(dx, dy, dz) * 1.5 
        
        # 1. Domain Box
        box = pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
        self._plotter.add_mesh(box, style='wireframe', color='black', line_width=2)
        
        # 2. Charge
        cx, cy, cz, shape, mass, rho = charge_data
        vol = mass / rho
        if shape == "Sphere":
            import math
            r = (3 * vol / (4 * math.pi)) ** (1 / 3)
            mesh = pv.Sphere(radius=r, center=(cx, cy, cz))
        elif shape == "Cuboid":
            side = vol ** (1 / 3)
            half = side / 2.0
            mesh = pv.Box(bounds=(cx - half, cx + half, cy - half, cy + half, cz - half, cz + half))
        else:
            import math
            r = 0.05
            h = vol / (math.pi * r * r)
            mesh = pv.Cylinder(center=(cx, cy, cz), radius=r, height=h, direction=(1, 0, 0))
        self._plotter.add_mesh(mesh, color="#e74c3c", opacity=0.8)
        
        # 3. Obstacles (respect Show Obstacles + Wireframe only in preview too)
        if self.show_obstacles:
            for obs in obstacles:
                if not obs.enabled: continue
                if obs.path not in self._stl_cache:
                    try:
                        self._stl_cache[obs.path] = pv.read(obs.path)
                    except Exception: continue
                m = self._stl_cache[obs.path].copy()
                if abs(obs.scale - 1.0) > 1e-6: m.scale([obs.scale]*3, inplace=True)
                if abs(obs.ox)+abs(obs.oy)+abs(obs.oz) > 1e-6: m.translate([obs.ox, obs.oy, obs.oz], inplace=True)
                if self.show_obstacles_wireframe_only:
                    self._plotter.add_mesh(m, style='wireframe', color='black', line_width=1)
                else:
                    self._plotter.add_mesh(m, color='#95a5a6', opacity=0.6)

        # 4. Preview Sections at exact position_m, clamped to domain
        bounds_tuple = (xmin, xmax, ymin, ymax, zmin, zmax)
        for sec in self.sections:
            if sec.enabled:
                origin = self._section_origin_clamped(sec, bounds_tuple)
                plane = pv.Plane(center=origin, direction=sec.normal, i_size=plane_size, j_size=plane_size)
                op = float(np.clip(sec.opacity, 0.0, 1.0))
                self._plotter.add_mesh(plane, color="blue", opacity=op, style="surface")
                self._plotter.add_mesh(plane, color="blue", style="wireframe", opacity=0.5)

        self.reset_camera()

    def set_simulation_mode(self, case_path: str) -> None:
        """Legacy alias for load_case(case_path)."""
        self.load_case(case_path, charge_center=None, cell_size=None)

    def load_case(
        self,
        case_path: str,
        charge_center: Optional[tuple] = None,
        cell_size: Optional[float] = None,
    ) -> None:
        """Set simulation case path and optional case params (charge center, cell size) for slice defaults."""
        self.current_case_dir = case_path
        self.is_simulating = True
        self._first_load = True
        self._mesh_bounds = None
        self._last_refresh_time = None
        self._charge_center = charge_center
        self._cell_size = cell_size if cell_size is not None and cell_size > 0 else 0.1
        if self._plotter:
            self._plotter.clear()
            self._plotter.add_text("Initializing...", position="upper_left", font_size=10)

    def get_mesh_bounds(self) -> Optional[tuple]:
        """Return (xmin, xmax, ymin, ymax, zmin, zmax) when simulation mesh is loaded; else None."""
        return self._mesh_bounds

    def _section_origin_clamped(self, sec: SectionItem, bounds: tuple) -> List[float]:
        """Slice origin from section: position_m along normal, clamped to mesh bounds with np.clip."""
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0
        pos_m = float(getattr(sec, "position_m", 0.0))
        nx, ny, nz = sec.normal[0], sec.normal[1], sec.normal[2]
        if abs(nz) >= 0.9:
            oz = float(np.clip(pos_m, zmin, zmax))
            ox, oy = cx, cy
        elif abs(ny) >= 0.9:
            oy = float(np.clip(pos_m, ymin, ymax))
            ox, oz = cx, cz
        else:
            ox = float(np.clip(pos_m, xmin, xmax))
            oy, oz = cy, cz
        ox = float(np.clip(ox, xmin, xmax))
        oy = float(np.clip(oy, ymin, ymax))
        oz = float(np.clip(oz, zmin, zmax))
        return [ox, oy, oz]

    def refresh_view(self):
        if not self._plotter:
            return
        # In preview mode, redraw preview with current viewport options (no Initialize needed)
        if not self.is_simulating and self._last_preview_data:
            self.update_preview(*self._last_preview_data)
            return
        if not self.is_simulating or not self.current_case_dir:
            return

        poly_points = os.path.join(self.current_case_dir, "constant", "polyMesh", "points")
        if not os.path.exists(poly_points):
            return

        foam_file = os.path.join(self.current_case_dir, "case.foam")
        if not os.path.exists(foam_file):
            try:
                with open(foam_file, "w") as f:
                    f.write("")
            except Exception:
                return

        try:
            reader = pv.POpenFOAMReader(foam_file)
            if not reader.time_values:
                return
            latest = reader.time_values[-1]
            if self._last_refresh_time is not None and abs(self._last_refresh_time - latest) < 1e-12:
                return
            self._last_refresh_time = latest
            reader.set_active_time_value(latest)
            data = reader.read()

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

            self._last_cell_count = int(internal_mesh.n_cells)
            try:
                self.cell_count_updated.emit(self._last_cell_count)
            except Exception:
                pass

            bounds = internal_mesh.bounds
            self._mesh_bounds = bounds
            xmin, xmax, ymin, ymax, zmin, zmax = bounds

            if self._first_load:
                self._plotter.clear()
                self._dynamic_actors.clear()
                self._obstacle_actors.clear()
            else:
                for a in self._dynamic_actors:
                    try:
                        self._plotter.remove_actor(a)
                    except Exception:
                        pass
                self._dynamic_actors.clear()
                for a in self._obstacle_actors:
                    try:
                        self._plotter.remove_actor(a)
                    except Exception:
                        pass
                self._obstacle_actors.clear()

            field = self.current_field
            s = self.field_settings.get(field, FieldViewSettings())
            use_log_scale = getattr(s, "log_scale", False)
            clim = None
            if field in internal_mesh.array_names:
                arr = internal_mesh.get_array(field)
                clim = [arr.min(), arr.max()] if s.auto_scale else [s.min_val, s.max_val]
                if use_log_scale and clim[0] is not None and clim[1] is not None and clim[0] <= 0:
                    clim = [max(1e-30, clim[0]), clim[1]]

            if self._first_load:
                # 1. Floor: white wireframe grid only (no solid color)
                dx, dy = xmax - xmin, ymax - ymin
                cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                floor_grid = pv.Plane(
                    center=(cx, cy, zmin),
                    direction=(0, 0, 1),
                    i_size=dx,
                    j_size=dy,
                    i_resolution=10,
                    j_resolution=10,
                )
                self._plotter.add_mesh(floor_grid, style="wireframe", color="white", line_width=1, reset_camera=False)

                # 2. Domain: pure empty wireframe (pv.Box only; no internal mesh or boundary faces)
                box = pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
                self._plotter.add_mesh(
                    box,
                    style="wireframe",
                    color="black",
                    opacity=0.1,
                    line_width=1,
                    lighting=False,
                    reset_camera=False,
                )

                # 3. Rulers on box edges (ticks and labels only; no grid lines)
                self._plotter.show_bounds(bounds=bounds, grid=False, location="outer")

                # 4. Obstacles: added every refresh when show_obstacles; tracked in _obstacle_actors for toggle
                pass  # obstacle block moved below so it runs every refresh

            # 4b. Obstacles (when visible; domain patches skipped); re-add each refresh so visibility/style toggles apply
            if self.show_obstacles and self.show_boundaries and isinstance(data, pv.MultiBlock) and "boundary" in data.keys():
                boundary_block = data["boundary"]
                obstacle_meshes = []
                try:
                    names = boundary_block.keys() if hasattr(boundary_block, "keys") else None
                except Exception:
                    names = None
                for i in range(boundary_block.n_blocks):
                    b_mesh = boundary_block[i]
                    if b_mesh is None or b_mesh.n_points == 0:
                        continue
                    patch_name = names[i] if names is not None and i < len(names) else None
                    if patch_name is None:
                        try:
                            patch_name = boundary_block.get_block_name(i)
                        except Exception:
                            patch_name = ""
                    is_domain = patch_name in self._domain_patch_names
                    if not is_domain:
                        obstacle_meshes.append(b_mesh)
                for b_mesh in obstacle_meshes:
                    if self.show_obstacles_wireframe_only:
                        try:
                            outer_edges = b_mesh.extract_feature_edges(
                                boundary_edges=True,
                                feature_edges=True,
                                feature_angle=45.0,
                                non_manifold_edges=True,
                                manifold_edges=False,
                            )
                            if outer_edges.n_points > 0:
                                actor = self._plotter.add_mesh(
                                    outer_edges,
                                    color="black",
                                    line_width=1,
                                    reset_camera=False,
                                    show_scalar_bar=False,
                                )
                            else:
                                actor = self._plotter.add_mesh(
                                    b_mesh, style="wireframe", color="black",
                                    line_width=1, reset_camera=False, show_scalar_bar=False,
                                )
                            self._obstacle_actors.append(actor)
                        except Exception:
                            actor = self._plotter.add_mesh(
                                b_mesh, style="wireframe", color="black",
                                line_width=1, reset_camera=False, show_scalar_bar=False,
                            )
                            self._obstacle_actors.append(actor)
                    else:
                        actor = self._plotter.add_mesh(
                            b_mesh,
                            color="#34495E",
                            opacity=1.0,
                            smooth_shading=False,
                            show_edges=self.show_mesh_lines,
                            reset_camera=False,
                            show_scalar_bar=False,
                        )
                        self._obstacle_actors.append(actor)

            # 5. Slices (cross-sections): user opacity, jet colormap
            sections_to_use = [sec for sec in self.sections if sec.enabled]
            if not sections_to_use:
                cx_charge = cy_charge = cz_charge = (zmin + zmax) / 2.0
                if self._charge_center:
                    cx_charge, cy_charge, cz_charge = self._charge_center
                sections_to_use = [
                    SectionItem(
                        True, "Default",
                        [(xmin + xmax) / 2, (ymin + ymax) / 2, cz_charge],
                        [0, 0, 1], 1.0, cz_charge,
                    )
                ]

            z_span = max(1e-9, zmax - zmin)
            for sec in sections_to_use:
                try:
                    origin = self._section_origin_clamped(sec, bounds)
                    is_floor_slice = abs(sec.normal[2]) >= 0.9 and abs(origin[2] - zmin) < 1e-9 * z_span
                    if is_floor_slice:
                        origin = [origin[0], origin[1], zmin + 1e-6 * z_span]
                    slc = internal_mesh.slice(normal=sec.normal, origin=origin)
                    if slc.n_points == 0:
                        continue
                    opacity = float(np.clip(sec.opacity, 0.0, 1.0))
                    actor = self._plotter.add_mesh(
                        slc,
                        scalars=field if field in internal_mesh.array_names else None,
                        cmap="jet",
                        clim=clim,
                        opacity=opacity,
                        lighting=False,
                        show_edges=self.show_mesh_lines,
                        reset_camera=False,
                        log_scale=use_log_scale,
                    )
                    self._dynamic_actors.append(actor)
                except Exception:
                    pass

            if field in internal_mesh.array_names:
                sb = self._plotter.add_scalar_bar(
                    title=field,
                    n_labels=6,
                    fmt="%.2e",
                    vertical=True,
                    position_x=0.02,
                    color="black",
                    background_opacity=0.0,
                    log_scale=use_log_scale,
                )
                self._dynamic_actors.append(sb)
            tt = self._plotter.add_text(f"Time: {latest:.5f} s", position="upper_left", color="black", font_size=10)
            self._dynamic_actors.append(tt)

            if self._first_load:
                self._plotter.reset_camera()
                self._first_load = False

            if self._show_probes and self._probes_data:
                radius = max(0.02, 2.0 * self._cell_size) if self._cell_size > 0 else 0.05
                for pt in self._probes_data:
                    if len(pt) < 3:
                        continue
                    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                    sphere = pv.Sphere(radius=radius, center=(x, y, z))
                    actor = self._plotter.add_mesh(sphere, color="yellow", opacity=0.9, reset_camera=False)
                    self._probe_actors.append(actor)

        except Exception:
            pass