"""Stable 1D review IDs and lightweight cross-tab comparison targets.

IDs are assigned by this list, not by on-screen position. Append new controls
instead of inserting them, so existing IDs stay put for the review session.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from PyQt5.QtWidgets import QGroupBox, QWidget


@dataclass(frozen=True)
class ReviewMark:
    review_id: str
    role: str
    label: str
    kind: str  # "area" or "control"


@dataclass(frozen=True)
class ReviewDestination:
    key: str
    label: str
    owner_attr: str
    sub_attr: str = ""
    sub_title: str = ""


@dataclass(frozen=True)
class ComparableTarget:
    destination_key: str
    widget_attr: str
    label: str


def _attr(name: str) -> Callable[[QWidget], Optional[QWidget]]:
    def resolve(tab: QWidget) -> Optional[QWidget]:
        widget = getattr(tab, name, None)
        return widget if isinstance(widget, QWidget) else None

    return resolve


def _group_of(name: str) -> Callable[[QWidget], Optional[QWidget]]:
    def resolve(tab: QWidget) -> Optional[QWidget]:
        widget = getattr(tab, name, None)
        if not isinstance(widget, QWidget):
            return None
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QGroupBox):
                return parent
            parent = parent.parentWidget()
        return None

    return resolve


def _plot_frame(tab: QWidget) -> Optional[QWidget]:
    canvas = getattr(tab, "canvas", None)
    if isinstance(canvas, QWidget):
        return canvas.parentWidget()
    return None


# (mark, resolver). Resolvers receive the 1D tab widget.
_ONE_D_MARKS: Sequence[Tuple[ReviewMark, Callable[[QWidget], Optional[QWidget]]]] = (
    (ReviewMark("1D-A", "setup", "Setup inputs", "area"), _attr("scroll_area")),
    (ReviewMark("1D-A01", "domain", "Domain", "control"), _group_of("spin_radius")),
    (ReviewMark("1D-A02", "radius", "Radius", "control"), _attr("spin_radius")),
    (ReviewMark("1D-A03", "cell_size", "Cell size", "control"), _attr("spin_cellsize")),
    (ReviewMark("1D-A04", "charge", "Charge", "control"), _group_of("combo_comp")),
    (ReviewMark("1D-A05", "material", "Material", "control"), _attr("combo_comp")),
    (ReviewMark("1D-A06", "edit_material", "Edit material", "control"), _attr("btn_edit_comp")),
    (ReviewMark("1D-A07", "mass", "Mass", "control"), _attr("spin_mass")),
    (ReviewMark("1D-A08", "density", "Density", "control"), _attr("spin_density")),
    (ReviewMark("1D-A09", "energy", "Energy", "control"), _attr("edit_energy")),
    (ReviewMark("1D-A10", "source_model", "Method", "control"), _attr("combo_source")),
    (ReviewMark("1D-A12", "remap", "Remap: Yes", "control"), _attr("radio_yes")),
    (ReviewMark("1D-A13", "remap", "Remap: No", "control"), _attr("radio_no")),
    (ReviewMark("1D-A14", "remap_status", "Remap status", "control"), _attr("lbl_remap_status")),
    (ReviewMark("1D-A15", "atmosphere", "Atmosphere", "control"), _group_of("spin_press")),
    (ReviewMark("1D-A16", "pressure", "Pressure", "control"), _attr("spin_press")),
    (ReviewMark("1D-A17", "temperature", "Temperature", "control"), _attr("spin_temp")),
    (ReviewMark("1D-A18", "boundaries", "Boundaries", "control"), _group_of("cmb_left")),
    (ReviewMark("1D-A19", "boundary", "Left boundary", "control"), _attr("cmb_left")),
    (ReviewMark("1D-A20", "run_mode", "Right boundary", "control"), _attr("cmb_right")),
    (ReviewMark("1D-A22", "solver", "Solver Options", "control"), _group_of("spin_cfl")),
    (ReviewMark("1D-A23", "cfl", "CFL", "control"), _attr("spin_cfl")),
    (ReviewMark("1D-A24", "end_time", "End time", "control"), _attr("spin_endtime")),
    (ReviewMark("1D-A25", "output", "Output options", "control"), _attr("group_output")),
    (ReviewMark("1D-A26", "gui_refresh", "GUI refresh", "control"), _attr("spin_gui_refresh")),
    (ReviewMark("1D-B", "info", "Info", "area"), _group_of("lbl_domain_cells")),
    (ReviewMark("1D-B01", "info_cells", "Domain cells", "control"), _attr("lbl_domain_cells")),
    (ReviewMark("1D-B02", "info_charge_radius", "Charge radius", "control"), _attr("lbl_charge_radius")),
    (ReviewMark("1D-B03", "info_charge_cells", "Charge cells", "control"), _attr("lbl_charge_cells")),
    (ReviewMark("1D-B04", "info_density", "Field density", "control"), _attr("lbl_adj_density")),
    (ReviewMark("1D-C", "plot", "Plot", "area"), _plot_frame),
    (ReviewMark("1D-C01", "status_caption", "Status caption", "control"), _attr("_status_caption_host")),
    (ReviewMark("1D-C02", "fit", "Fit", "control"), _attr("btn_fit")),
    (ReviewMark("1D-C03", "plot", "Overpressure plot", "control"), _attr("canvas")),
    (ReviewMark("1D-D", "execution", "Execution", "area"), _attr("ctrl_tabs")),
    (ReviewMark("1D-D01", "initialize", "Initialize", "control"), _attr("btn_initialize")),
    (ReviewMark("1D-D02", "execution_group", "Simulation control", "control"), _group_of("btn_run")),
    (ReviewMark("1D-D03", "run", "Run", "control"), _attr("btn_run")),
    (ReviewMark("1D-D04", "stop", "Interrupt", "control"), _attr("btn_stop")),
    (ReviewMark("1D-D05", "chart", "Chart: Pressure", "control"), _attr("radio_chart_pressure")),
    (ReviewMark("1D-D06", "chart", "Chart: PMax", "control"), _attr("radio_chart_pmax")),
    (ReviewMark("1D-D07", "chart", "Chart: Density", "control"), _attr("radio_chart_density")),
    (ReviewMark("1D-D08", "chart", "Chart: Energy", "control"), _attr("radio_chart_energy")),
    (ReviewMark("1D-D09", "chart", "Chart: Velocity", "control"), _attr("radio_chart_velocity")),
)


DESTINATIONS: Sequence[ReviewDestination] = (
    ReviewDestination("1d", "Spherical – 1D", "tab_1d"),
    ReviewDestination("2d-setup", "2D · Setup", "tab_2d", "input_tabs", "Setup"),
    ReviewDestination("2d-output", "2D · Output & Probes", "tab_2d", "input_tabs", "Output & Probes"),
    ReviewDestination("2d-exec", "2D · Execution", "tab_2d", "ctrl_tabs", "Execution Controls"),
    ReviewDestination("3d-model", "3D · Model Setup", "tab_3d", "settings_tabs", "Model Setup"),
    ReviewDestination("3d-obstacles", "3D · Obstacles", "tab_3d", "settings_tabs", "Obstacles"),
    ReviewDestination("3d-exec", "3D · Execution", "tab_3d", "ctrl_tabs", "Execution Controls"),
    ReviewDestination("3d-view", "3D · Viewport Options", "tab_3d", "ctrl_tabs", "Viewport Options"),
    ReviewDestination("3d-sections", "3D · Cross-Sections", "tab_3d", "ctrl_tabs", "Cross-Sections"),
    ReviewDestination("time-history", "Time History", "tab_time_history"),
    ReviewDestination("validation", "Validation", "tab_validation"),
)


# Role -> places that show the same kind of control. Missing widgets are skipped.
ROLE_LINKS = {
    "radius": (ComparableTarget("2d-setup", "spin_radius", "Radius"),),
    "cell_size": (
        ComparableTarget("2d-setup", "spin_cell", "Cell size"),
        ComparableTarget("3d-model", "scell", "Cell size"),
    ),
    "material": (
        ComparableTarget("2d-setup", "cmb_material", "Material"),
        ComparableTarget("3d-model", "c_mat", "Material"),
    ),
    "edit_material": (ComparableTarget("3d-model", "btn_edit_custom", "Edit material"),),
    "mass": (
        ComparableTarget("2d-setup", "spin_mass", "Mass"),
        ComparableTarget("3d-model", "c_mass", "Mass"),
    ),
    "density": (
        ComparableTarget("2d-setup", "spin_density", "Density"),
        ComparableTarget("3d-model", "c_rho", "Density"),
    ),
    "energy": (ComparableTarget("2d-setup", "spin_energy", "Energy"),),
    "remap": (
        ComparableTarget("2d-setup", "cmb_source", "Initialization source"),
        ComparableTarget("3d-model", "grp_init_method", "Initialize method"),
    ),
    "pressure": (
        ComparableTarget("2d-setup", "spin_pressure", "Pressure"),
        ComparableTarget("3d-model", "p0", "Pressure"),
    ),
    "temperature": (
        ComparableTarget("2d-setup", "spin_temperature", "Temperature"),
        ComparableTarget("3d-model", "t0", "Temperature"),
    ),
    "cfl": (
        ComparableTarget("2d-setup", "spin_max_co", "Max Co"),
        ComparableTarget("3d-model", "spin_cfl", "CFL"),
    ),
    "end_time": (
        ComparableTarget("2d-setup", "spin_end_time", "End time"),
        ComparableTarget("3d-exec", "spin_end", "End time"),
    ),
    "info_cells": (ComparableTarget("2d-setup", "lbl_radial_cells", "Radial cells"),),
    "info_charge_radius": (ComparableTarget("2d-setup", "lbl_charge_r", "Charge radius"),),
    "plot": (
        ComparableTarget("2d-setup", "viewer", "Viewer"),
        ComparableTarget("3d-model", "viewer", "Viewer"),
        ComparableTarget("time-history", "canvas", "History plot"),
    ),
    "initialize": (
        ComparableTarget("2d-exec", "btn_initialize", "Initialize"),
        ComparableTarget("3d-exec", "btn_init", "Initialize"),
    ),
    "run": (
        ComparableTarget("2d-exec", "btn_exact_end", "Run to end"),
        ComparableTarget("3d-exec", "btn_exact_end", "Run to end"),
    ),
    "stop": (
        ComparableTarget("2d-exec", "btn_stop", "Interrupt"),
        ComparableTarget("3d-exec", "btn_stop", "Interrupt"),
    ),
}


def destination_by_key(key: str) -> Optional[ReviewDestination]:
    for item in DESTINATIONS:
        if item.key == key:
            return item
    return None


def resolve_1d_marks(window: QWidget) -> List[Tuple[ReviewMark, Optional[QWidget]]]:
    tab = getattr(window, "tab_1d", None)
    if not isinstance(tab, QWidget):
        return []
    found: List[Tuple[ReviewMark, Optional[QWidget]]] = []
    for mark, resolver in _ONE_D_MARKS:
        try:
            widget = resolver(tab)
        except Exception:
            widget = None
        if widget is not None and not isinstance(widget, QWidget):
            widget = None
        found.append((mark, widget))
    return found


def mark_for_widget(
    marks: Sequence[Tuple[ReviewMark, Optional[QWidget]]],
    widget: Optional[QWidget],
) -> Optional[Tuple[ReviewMark, QWidget]]:
    """Deepest catalog widget at or above ``widget``."""
    if widget is None:
        return None
    by_id = {id(target): (mark, target) for mark, target in marks if target is not None}
    current: Optional[QWidget] = widget
    while current is not None:
        hit = by_id.get(id(current))
        if hit is not None:
            return hit
        current = current.parentWidget()
    return None
