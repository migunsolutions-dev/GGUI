"""Production Cylindrical–2D axisymmetric workflow tab."""
from __future__ import annotations

import math
import os
from dataclasses import asdict, replace
from typing import Dict, List

from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from axisymmetric_2d import (
    BOUNDARY_OPEN,
    BOUNDARY_SLIP,
    DIRECT_SOURCE,
    DYNAMIC_MESH,
    FIXED_MESH,
    REMAP_SOURCE,
    align_axisymmetric_domain,
    validate_case_inputs_2d,
)
from axisymmetric_viewer import AxisymmetricViewerWidget
from charge_seed_plan import SEED_MODE_AUTO, SEED_MODE_MANUAL, SEED_MODE_OFF
from dialogs import RemapFromDialog
from external_case_workflow_2d import ImportMode2D, import_mode_label
from imported_case_mapping_2d import FieldProvenance
from material_catalog import materials_copy
from material_validation import REQUIRED_IMPORTED_PHYSICS_KEYS, UNSUPPORTED_IMPORT_KEYS
from models_2d import (
    DEFAULT_REFINE_INTERVAL,
    CaseInputs2D,
    MappingSource2D,
    ProbePoint2D,
    SimulationState2D,
)

MATERIAL_UNDEFINED_PLACEHOLDER = "— select material —"
UNDEFINED_CONTROL_STYLE = "background-color: #fff3cd;"
VIEW_FIELD_OPTIONS = (
    ("p", "Pressure"),
    ("rho", "Density"),
    ("T", "Temperature"),
    ("U", "Velocity"),
    ("alpha.c4", "Explosive fraction"),
    ("cellLevel", "Refinement level"),
)
from physical_charge_geometry import physical_charge_geometry
from ui_metrics import (
    COMPUTATIONAL_LEFT_PANEL_MIN,
    COMPUTATIONAL_LEFT_PANEL_WIDTH,
    CONTROL_MAX_WIDTH_DEFAULT,
    EXECUTION_AREA_MIN_HEIGHT,
    EXECUTION_AREA_PREFERRED_HEIGHT_2D,
    FORM_ROW_SPACING,
    GROUP_MARGINS,
    INFO_PANEL_HEIGHT,
    INFO_ROW_STYLE,
    INFO_TITLE_STYLE,
    SECONDARY_INFO_STYLE,
    WARNING_STYLE,
)


class TimeComboBox(QComboBox):
    """Time selector. The on-disk catalog is loaded only when the popup opens."""

    popup_requested = pyqtSignal()

    def showPopup(self):
        self.popup_requested.emit()
        super().showPopup()


def compact_spin_text(value: float, max_decimals: int) -> str:
    """Whole values have no decimal; other values keep one trailing zero."""
    if not math.isfinite(value):
        return ""
    places = max(0, int(max_decimals))
    rounded = round(float(value), places)
    if places == 0 or abs(rounded - round(rounded)) < 10 ** (-places):
        return str(int(round(rounded)))
    text = f"{rounded:.{places}f}".rstrip("0")
    if "." not in text or text.endswith("."):
        return str(int(round(rounded)))
    return text + "0"


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Setup numeric field: right-aligned compact display, stored value unchanged."""

    def textFromValue(self, value: float) -> str:
        return compact_spin_text(float(value), self.decimals())


def latest_case_1d_dir(work_root: str) -> str:
    """Newest Case_1D_* folder under the Work/run directory, or empty."""
    if not work_root or not os.path.isdir(work_root):
        return ""
    try:
        names = os.listdir(work_root)
    except OSError:
        return ""
    newest_name = ""
    newest_path = ""
    for name in names:
        if not name.startswith("Case_1D_"):
            continue
        path = os.path.join(work_root, name)
        if os.path.isdir(path) and name > newest_name:
            newest_name = name
            newest_path = path
    return os.path.normpath(newest_path) if newest_path else ""


def case_dir_from_picked_path(path: str) -> str:
    """Map a picked results file or folder to the OpenFOAM case root."""
    path = os.path.normpath(path) if path else ""
    if not path:
        return ""
    if os.path.isfile(path):
        parent = os.path.dirname(path)
        name = os.path.basename(path).lower()
        if name.endswith(".foam") and os.path.isdir(os.path.join(parent, "system")):
            return parent
        if os.path.isdir(os.path.join(parent, "system")):
            return parent
        grand = os.path.dirname(parent)
        if os.path.isdir(os.path.join(grand, "system")):
            return grand
        return parent
    if os.path.isdir(os.path.join(path, "system")):
        return path
    parent = os.path.dirname(path)
    if os.path.isdir(os.path.join(parent, "system")):
        return parent
    return path


class Tab2D(QWidget):
    sig_request_init = pyqtSignal(object)
    sig_request_run_exact_end = pyqtSignal()
    sig_request_stop = pyqtSignal()
    sig_request_log = pyqtSignal()
    sig_request_prepare_transfer = pyqtSignal()
    sig_state_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.materials_db = materials_copy()
        self._state = SimulationState2D.DRAFT
        self._loading = False
        self._actual_cell_count = None
        self._active_case_dir = None
        self._imported_case = None
        self._import_mode = ImportMode2D.NATIVE_GGUI_2D
        self._imported_field_meta = {}
        self._unrecovered_gui_keys = set()
        self._case_defined_gui_keys = set()
        self._undefined_gui_keys = set()
        self._unsupported_features = []
        self._defer_viewer_preview = False
        self._pending_setup_preview = None
        self._source_cases_root = ""
        self._remap_case_path = ""
        self._remap_from_last_1d = True
        self._remap_kind = RemapFromDialog.CURRENT_1D
        self._last_1d_case_dir = ""
        self._enable_impulse = True
        self._enable_dynamic_pressure = False
        self._preview_flush_timer = QTimer(self)
        self._preview_flush_timer.setSingleShot(True)
        self._preview_flush_timer.timeout.connect(self._flush_setup_preview)
        self._build_ui()
        self.viewer.cell_count_updated.connect(self._on_cell_count_updated)
        self.viewer.log_scale_rejected.connect(self._on_log_scale_rejected)
        self._connect_signals()
        self._apply_enablement()
        self._refresh_derived()

    @property
    def simulation_state(self) -> SimulationState2D:
        return self._state

    @property
    def is_imported_mode(self) -> bool:
        return self._import_mode != ImportMode2D.NATIVE_GGUI_2D

    @property
    def import_mode(self) -> ImportMode2D:
        return self._import_mode

    # Compatibility for older tests / call sites.
    @property
    def _external_case(self):
        return self._imported_case

    def set_simulation_state(self, state: SimulationState2D | str) -> None:
        state = SimulationState2D(state)
        self._state = state
        if self.is_imported_mode:
            self.lbl_state.setText(f"State: {import_mode_label(self._import_mode)}")
        else:
            self.lbl_state.setText(f"State: {state.value}")
        running = state == SimulationState2D.RUNNING or (
            self._import_mode == ImportMode2D.IMPORTED_2D_RUNNING
        )
        from state_machine_2d import can_run

        # Buttons derive from the explicit state machine; STALE/FAILED/INITIALIZING
        # never look initialized. Imported-ready mode is an additional enablement path.
        initialized = can_run(state) or (
            self._import_mode == ImportMode2D.IMPORTED_2D_READY
            and state
            not in (
                SimulationState2D.INITIALIZING,
                SimulationState2D.FAILED,
                SimulationState2D.STALE,
            )
        )
        self._apply_action_buttons(running=running, initialized=initialized)
        self.sig_state_changed.emit(state.value)

    def _apply_action_buttons(
        self, *, running: bool = False, initialized: bool = False
    ) -> None:
        self.btn_initialize.setText("Initialise Model")
        self.btn_initialize.setMinimumWidth(198)
        if not self.is_imported_mode:
            preparing = self._state == SimulationState2D.INITIALIZING
            self.btn_initialize.setEnabled(not running and not preparing)
            self.btn_exact_end.setEnabled(initialized and not running and not preparing)
            if preparing:
                self.btn_stop.setText("Cancel Preparation")
                self.btn_stop.setEnabled(True)
                self.btn_stop.setToolTip("Cancel the active 2D preparation operation")
            else:
                self.btn_stop.setText("Interrupt")
                self.btn_stop.setToolTip("")
                self.btn_stop.setEnabled(running)
            return

        mode = self._import_mode
        if mode == ImportMode2D.IMPORTED_2D_UNINITIALIZED:
            self.btn_initialize.setEnabled(not running)
            self.btn_exact_end.setEnabled(False)
            self.btn_stop.setEnabled(False)
        elif mode == ImportMode2D.IMPORTED_2D_INITIALIZING:
            self.btn_initialize.setEnabled(False)
            self.btn_exact_end.setEnabled(False)
            self.btn_stop.setText("Cancel Preparation")
            self.btn_stop.setEnabled(True)
            self.btn_stop.setToolTip("Cancel the active 2D preparation operation")
        elif mode == ImportMode2D.IMPORTED_2D_READY:
            self.btn_initialize.setText("Reinitialise")
            self.btn_initialize.setEnabled(not running)
            self.btn_exact_end.setEnabled(not running)
            self.btn_stop.setEnabled(False)
        elif mode == ImportMode2D.IMPORTED_2D_RUNNING:
            self.btn_initialize.setEnabled(False)
            self.btn_exact_end.setEnabled(False)
            self.btn_stop.setEnabled(True)
        else:  # FAILED
            self.btn_initialize.setEnabled(not running)
            self.btn_exact_end.setEnabled(False)
            self.btn_stop.setEnabled(False)
        self.lbl_state.setText(f"State: {import_mode_label(mode)}")

    def set_import_mode(self, mode: ImportMode2D) -> None:
        self._import_mode = mode
        if self._imported_case is not None:
            self._imported_case.mode = mode
        self._apply_action_buttons(
            running=mode == ImportMode2D.IMPORTED_2D_RUNNING,
            initialized=mode == ImportMode2D.IMPORTED_2D_READY,
        )
        self._update_provenance_banner()

    def load_imported_case(self, state, *, apply_mapping: bool = True) -> None:
        """Attach an imported case and optionally populate 2D controls.

        After generation in the same session, validated in-memory controls stay
        authoritative while inspection contributes only runtime/mesh metadata.
        """
        self._imported_case = state
        self._import_mode = state.mode
        self._active_case_dir = state.active_case_path
        self._actual_cell_count = state.cell_count
        if apply_mapping:
            self._apply_import_mapping(state)
        self._update_provenance_banner()

        radius = float(state.radius_m) if state.radius_m is not None else None
        height = float(state.height_m) if state.height_m is not None else None
        if radius is not None and height is not None:
            self.viewer.set_axisymmetric_domain(radius, height)

        if state.mesh_present and state.display_compatible and state.viewable:
            field = "alpha.c4" if "alpha.c4" in state.fields else (
                "p" if "p" in state.fields else (state.fields[0] if state.fields else "p")
            )
            self._ensure_view_field_option(field)
            self.cmb_field.setCurrentText(field)
            self.viewer.load_case(state.active_case_path)
            self.viewer.set_field(field)
            self.set_simulation_state(SimulationState2D.INITIALIZED)
            self._set_result_controls_available(True)
        else:
            # Setup Preview from parsed/UI values — do not blank the viewport
            # merely because polyMesh has not been generated yet.
            self.viewer.is_simulating = False
            self.set_simulation_state(SimulationState2D.DRAFT)
            self._set_result_controls_available(False)
            self._refresh_derived()
        self._apply_action_buttons()
        self._refresh_info()

    # Compatibility aliases.
    def load_external_case(self, state) -> None:
        self.load_imported_case(state)

    def attach_working_copy(self, state) -> None:
        self.load_imported_case(state)

    def clear_imported_case(self) -> None:
        was_loading = self._loading
        self._loading = True
        try:
            self._imported_case = None
            self._import_mode = ImportMode2D.NATIVE_GGUI_2D
            self._imported_field_meta = {}
            self._unrecovered_gui_keys = set()
            self._case_defined_gui_keys = set()
            self._clear_all_undefined_controls()
            # Undefined spins restore to their minimum (often 0). Reinstate native
            # defaults so a subsequent Setup Preview refresh cannot abort on
            # non-positive physics inputs.
            defaults = CaseInputs2D()
            if float(self.spin_mass.value()) <= 0.0:
                self.spin_mass.setValue(float(defaults.mass_kg or 1.0))
            if float(self.spin_density.value()) <= 0.0:
                self.spin_density.setValue(float(defaults.rho_charge or 1600.0))
            if float(self.spin_energy.value()) <= 0.0:
                self.spin_energy.setValue(float(defaults.energy_j_per_kg or 4.5e6))
            if float(self.spin_cell.value()) <= 0.0:
                self.spin_cell.setValue(float(defaults.cell_size or 0.05))
            if self.cmb_material.currentText() in ("", MATERIAL_UNDEFINED_PLACEHOLDER):
                self.cmb_material.setCurrentText(str(defaults.material_name or "TNT"))
            self._unsupported_features = []
            self._restore_native_control_editability()
            self._active_case_dir = None
            if hasattr(self, "lbl_import_banner"):
                self.lbl_import_banner.setVisible(False)
            self.btn_initialize.setText("Initialise Model")
            self.btn_initialize.setMinimumWidth(198)
            try:
                clearer = getattr(self.viewer, "clear_simulation_view", None)
                if callable(clearer):
                    clearer()
            except Exception:
                pass
            self.set_simulation_state(SimulationState2D.DRAFT)
            self._apply_enablement()
            self._apply_action_buttons(running=False, initialized=False)
        finally:
            self._loading = was_loading

    def clear_external_case(self) -> None:
        self.clear_imported_case()

    def set_prepare_progress(self, utility: str) -> None:
        if not self.is_imported_mode:
            return
        self.set_import_mode(ImportMode2D.IMPORTED_2D_INITIALIZING)
        self.lbl_state.setText(f"State: Initialising imported case ({utility})")

    def _update_provenance_banner(self) -> None:
        if not hasattr(self, "lbl_import_banner"):
            return
        if not self.is_imported_mode or self._imported_case is None:
            self.lbl_import_banner.setVisible(False)
            return
        src = os.path.basename(self._imported_case.source_dir or "")
        wc = os.path.basename(
            self._imported_case.working_copy_dir
            or self._imported_case.case_dir
            or ""
        )
        self.lbl_import_banner.setText(
            f"Editable GGUI model from BF source | Source: {src} | Generated case: {wc}"
        )
        self.lbl_import_banner.setVisible(True)
        self.lbl_import_banner.setToolTip(
            f"Source (read-only, never modified):\n{self._imported_case.source_dir}\n\n"
            f"GGUI working / generated case:\n"
            f"{self._imported_case.working_copy_dir or self._imported_case.case_dir}\n\n"
            "Controls are editable. Initialise Model regenerates a complete case "
            "via the GGUI 2D generator (canonical ±5° wedge topology)."
        )

    def _apply_import_mapping(self, state) -> None:
        """Populate Setup/Mesh/Output from mapping; never invent native defaults."""
        mapping = state.mapping
        self._imported_field_meta = {}
        self._unrecovered_gui_keys = set()
        self._case_defined_gui_keys = set()
        self._clear_all_undefined_controls()
        self._unsupported_features = list(getattr(mapping, "unsupported_features", ()) or ())
        self._loading = True
        try:
            self._restore_native_control_editability()
            # Clear probes — do not keep native defaults.
            self.tbl_probes.setRowCount(0)
            if mapping is None:
                return
            # Apply recovered gui values only.
            values = dict(mapping.gui_values)
            # Mass / cell_size special handling for unrecovered/case-defined.
            for key, mf in mapping.fields.items():
                gui_key = mf.gui_key or key
                self._imported_field_meta[gui_key] = mf
                if mf.provenance == FieldProvenance.NOT_RECOVERED:
                    self._unrecovered_gui_keys.add(gui_key)
                elif mf.provenance == FieldProvenance.CASE_DEFINED:
                    self._case_defined_gui_keys.add(gui_key)

            # Use setters without clearing imported mode.
            was_loading = self._loading
            self._loading = True
            try:
                self._set_control_values(values, clear_imported=False, manage_loading=False)
            finally:
                self._loading = was_loading

            # Required physics / unsupported keys without recovered values stay undefined.
            pending_undefined = set()
            for key in REQUIRED_IMPORTED_PHYSICS_KEYS:
                if key not in values or values.get(key) is None:
                    pending_undefined.add(key)
            for key in UNSUPPORTED_IMPORT_KEYS:
                if key in self._case_defined_gui_keys and key not in values:
                    pending_undefined.add(key)
            for key in sorted(pending_undefined):
                self._mark_control_undefined(key)

            # Explicit charge radius display when mass unrecovered.
            cr = mapping.get("charge_radius")
            if cr and cr.displayed_value is not None:
                self.lbl_charge_r.setText(f"{float(cr.displayed_value):.6g} m")
                self.lbl_charge_d.setText(f"{2 * float(cr.displayed_value):.6g} m")

            # Grid counts from mapping when available.
            rc = mapping.get("radial_cells")
            vc = mapping.get("vertical_cells")
            if rc and rc.displayed_value is not None:
                self.lbl_radial_cells.setText(str(int(rc.displayed_value)))
            if vc and vc.displayed_value is not None:
                self.lbl_vertical_cells.setText(str(int(vc.displayed_value)))

            # Converted BF → editable GGUI model: normal contextual enablement only.
            self._restore_native_control_editability()
            self._apply_enablement()
        finally:
            self._loading = False

    def _prepare_spin_undefined(self, spin: QDoubleSpinBox) -> None:
        if getattr(spin, "_ggui_undef_prepared", False):
            return
        spin._ggui_real_min = float(spin.minimum())
        sentinel = spin._ggui_real_min - 1.0
        spin.setMinimum(sentinel)
        spin.setSpecialValueText("required")
        spin._ggui_undef_prepared = True

    def _restore_spin_undefined(self, spin: QDoubleSpinBox) -> None:
        if not getattr(spin, "_ggui_undef_prepared", False):
            return
        real_min = float(getattr(spin, "_ggui_real_min", 0.0))
        spin.setSpecialValueText("")
        spin.setMinimum(real_min)
        if spin.value() < real_min:
            spin.setValue(real_min)
        spin._ggui_undef_prepared = False
        spin.setStyleSheet("")

    def _mark_control_undefined(self, key: str) -> None:
        self._undefined_gui_keys.add(key)
        widget = self._widget_for_gui_key(key)
        if widget is None:
            return
        was_loading = self._loading
        self._loading = True
        try:
            if key == "material_name" and isinstance(widget, QComboBox):
                if widget.findText(MATERIAL_UNDEFINED_PLACEHOLDER) < 0:
                    widget.insertItem(0, MATERIAL_UNDEFINED_PLACEHOLDER)
                widget.setCurrentText(MATERIAL_UNDEFINED_PLACEHOLDER)
                widget.setStyleSheet(UNDEFINED_CONTROL_STYLE)
                widget.setToolTip("Required: select a material explicitly.")
            elif isinstance(widget, QDoubleSpinBox):
                self._prepare_spin_undefined(widget)
                widget.setValue(widget.minimum())
                widget.setStyleSheet(UNDEFINED_CONTROL_STYLE)
                widget.setToolTip("Required: enter an explicit value.")
            elif isinstance(widget, QComboBox):
                widget.setStyleSheet(UNDEFINED_CONTROL_STYLE)
                widget.setToolTip("Required: select an explicit value.")
        finally:
            self._loading = was_loading

    def _clear_control_undefined(self, key: str) -> None:
        if key not in self._undefined_gui_keys:
            return
        self._undefined_gui_keys.discard(key)
        widget = self._widget_for_gui_key(key)
        if widget is None:
            return
        if key == "material_name" and isinstance(widget, QComboBox):
            idx = widget.findText(MATERIAL_UNDEFINED_PLACEHOLDER)
            current = widget.currentText()
            if idx >= 0:
                widget.removeItem(idx)
            if current == MATERIAL_UNDEFINED_PLACEHOLDER and widget.count():
                # Leave selection to caller; default to first real catalog item only
                # when restoring native mode, not while still imported.
                pass
            widget.setStyleSheet("")
            widget.setToolTip("")
        elif isinstance(widget, QDoubleSpinBox):
            self._restore_spin_undefined(widget)
            widget.setToolTip("")
        elif isinstance(widget, QComboBox):
            widget.setStyleSheet("")
            widget.setToolTip("")

    def _clear_all_undefined_controls(self) -> None:
        # Suppress spin/combo change handlers while restoring defaults — those
        # handlers refresh the VTK preview and can abort offscreen plotters.
        was_loading = self._loading
        self._loading = True
        try:
            for key in list(self._undefined_gui_keys):
                self._clear_control_undefined(key)
            self._undefined_gui_keys.clear()
            # Ensure placeholder is removed when leaving imported undefined state.
            if hasattr(self, "cmb_material"):
                idx = self.cmb_material.findText(MATERIAL_UNDEFINED_PLACEHOLDER)
                if idx >= 0:
                    current = self.cmb_material.currentText()
                    self.cmb_material.removeItem(idx)
                    if current == MATERIAL_UNDEFINED_PLACEHOLDER:
                        self.cmb_material.setCurrentText("TNT")
                self.cmb_material.setStyleSheet("")
                self.cmb_material.setToolTip("")
        finally:
            self._loading = was_loading

    def undefined_gui_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._undefined_gui_keys))

    def unsupported_import_features(self) -> tuple:
        return tuple(self._unsupported_features)

    def _widget_for_gui_key(self, key: str):
        return {
            "radius": self.spin_radius,
            "height": self.spin_height,
            "cell_size": self.spin_cell,
            "height_of_burst": self.spin_hob,
            "detonation_height": self.spin_det_height,
            "charge_aspect": self.spin_ld,
            "mass_kg": self.spin_mass,
            "rho_charge": self.spin_density,
            "energy_j_per_kg": self.spin_energy,
            "p_atm": self.spin_pressure,
            "t_atm": self.spin_temperature,
            "max_co": self.spin_max_co,
            "end_time_s": self.spin_end_time,
            "delta_t": self.spin_delta_t,
            "write_interval_time": self.spin_write_time,
            "write_interval_steps": self.spin_write_steps,
            "material_name": self.cmb_material,
            "charge_shape": self.cmb_shape,
            "initialization_source": self.cmb_source,
            "write_control_type": self.cmb_write_control,
            "mesh_mode": self.cmb_mesh_mode,
            "charge_seed_mode": self.cmb_seed_mode,
            "refine_indicator_field": self.cmb_estimator,
            "charge_refinement_level": self.spin_seed_level,
            "buffer_layers": self.spin_seed_buffer,
            "refine_interval": self.spin_refine_interval,
            "unrefine_interval": self.spin_unrefine_interval,
            "lower_refine_threshold": self.spin_refine_threshold,
            "unrefine_threshold": self.spin_unrefine_threshold,
            "n_buffer_layers_dynamic": self.spin_runtime_buffer,
            "dyn_refine_max": self.spin_runtime_level,
            "dynamic_max_cells": self.spin_max_cells,
            "dump_level": self.chk_dump_level,
            "refine_probes": self.chk_refine_probes,
            "adjust_time_step": self.chk_adjust,
            "outer_boundary": self.cmb_outer,
            "top_boundary": self.cmb_top,
            "bottom_boundary": self.cmb_bottom,
            "enable_balancing": self.chk_balancing,
        }.get(key)

    def _apply_imported_control_editability(self, mapping) -> None:
        """Deprecated read-only policy — converted imports use normal enablement."""
        self._restore_native_control_editability()
        self._apply_enablement()

    def _set_result_controls_available(self, available: bool) -> None:
        """Disable result-only field/log controls before initialization."""
        tip = "" if available else "Unavailable until Initialise Model completes"
        widgets = [self.cmb_field, self.chk_log_scale, *self._field_radios.values()]
        for widget in widgets:
            widget.setEnabled(bool(available))
            widget.setToolTip(tip)
        # Mesh/probe overlays remain available for Setup Preview.
        self.chk_view_mesh.setEnabled(True)
        self.chk_view_probes.setEnabled(True)
        if not available:
            self.chk_view_mesh.setToolTip("Setup Preview planned grid overlay")
            self.chk_view_probes.setToolTip("Setup Preview probe markers")
        else:
            self.chk_view_mesh.setToolTip("")
            self.chk_view_probes.setToolTip("")

    def _restore_native_control_editability(self) -> None:
        for key in (
            "radius", "height", "cell_size", "height_of_burst", "detonation_height",
            "charge_aspect", "mass_kg", "rho_charge", "energy_j_per_kg", "p_atm",
            "t_atm", "max_co", "end_time_s", "delta_t", "write_interval_time",
            "write_interval_steps", "material_name", "charge_shape",
            "initialization_source", "write_control_type", "mesh_mode",
            "charge_seed_mode", "refine_indicator_field", "charge_refinement_level",
            "buffer_layers", "refine_interval", "unrefine_interval",
            "lower_refine_threshold", "unrefine_threshold", "n_buffer_layers_dynamic",
            "dyn_refine_max", "dynamic_max_cells", "dump_level", "refine_probes",
            "adjust_time_step", "outer_boundary", "top_boundary", "bottom_boundary",
            "enable_balancing",
        ):
            widget = self._widget_for_gui_key(key)
            if widget is None:
                continue
            widget.setToolTip("")
            if isinstance(widget, QDoubleSpinBox):
                widget.setSpecialValueText("")
            # Actual enablement is owned by _apply_enablement.

    def mark_initialized(self, case_dir: str, actual_cells: int | None = None) -> None:
        self._active_case_dir = case_dir
        self._actual_cell_count = actual_cells
        self.set_simulation_state(SimulationState2D.INITIALIZED)
        self._on_probe_view_toggled(self.chk_view_probes.isChecked())
        self._refresh_info()

    def handle_initialization_failure(
        self, case_dir: str | None = None, message: str = ""
    ) -> None:
        """Failed init must not look like a valid initialized model."""
        self._active_case_dir = case_dir
        self._actual_cell_count = None
        self.set_simulation_state(SimulationState2D.FAILED)
        if hasattr(self.viewer, "clear_simulation_view"):
            self.viewer.clear_simulation_view(
                message
                or "Initialization failed — partial mesh is not a valid result. See Open Log."
            )
        self._refresh_info()

    def _on_cell_count_updated(self, count: int) -> None:
        self._actual_cell_count = int(count)
        self._refresh_info()

    def _on_log_scale_rejected(self, message: str) -> None:
        self.chk_log_scale.blockSignals(True)
        self.chk_log_scale.setChecked(False)
        self.chk_log_scale.blockSignals(False)
        self.chk_log_scale.setToolTip(message)
        # Keep field settings in sync with the explicit disable.
        if hasattr(self.viewer, "set_log_scale"):
            settings = self.viewer.field_settings.get(self.viewer.current_field)
            if settings is not None:
                settings.log_scale = False
        if hasattr(self, "lbl_info_actual"):
            tip = str(message or "Log scale disabled: field has non-positive values.")
            self.chk_log_scale.setToolTip(tip)
            # Surface the policy in the info strip without changing physics/data.
            current = self.lbl_info_actual.text()
            note = f"Log scale off: {tip}"
            if note not in current:
                self.lbl_info_actual.setText(
                    f"{current}  |  {note}" if current else note
                )

    def set_preparation_step(self, name: str) -> None:
        """Show the current asynchronous preparation step in the state label."""
        if hasattr(self, "lbl_state") and self.lbl_state is not None:
            self.lbl_state.setText(f"Preparing: {name}")

    def mark_stale(self) -> None:
        if self._loading:
            return
        if self.is_imported_mode:
            # Imported mode: only editable controlDict fields may dirty the case.
            return
        from state_machine_2d import state_after_input_edit

        next_state = state_after_input_edit(self._state)
        if next_state != self._state:
            self.set_simulation_state(next_state)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        root.addWidget(self._main_splitter)

        left = QWidget()
        left.setMinimumWidth(COMPUTATIONAL_LEFT_PANEL_MIN)
        left.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(5, 5, 5, 5)
        ll.setSpacing(5)
        self.lbl_import_banner = QLabel("")
        self.lbl_import_banner.setWordWrap(True)
        self.lbl_import_banner.setStyleSheet(SECONDARY_INFO_STYLE)
        self.lbl_import_banner.setVisible(False)
        ll.addWidget(self.lbl_import_banner, 0)
        self.input_tabs = QTabWidget()
        self.input_tabs.setMinimumWidth(0)
        self.input_tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.input_tabs.addTab(self._scroll_tab(self._build_setup_tab()), "Setup")
        self._build_mesh_amr_dialog()
        self.input_tabs.addTab(self._scroll_tab(self._build_output_tab()), "Output & Probes")
        ll.addWidget(self.input_tabs, 1)
        self._left_setup_scroll = self.input_tabs.widget(0)
        ll.addWidget(self._build_info_panel(), 0)
        self._main_splitter.addWidget(left)

        right = QWidget()
        right.setMinimumWidth(0)
        right.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        self._right_v_splitter = QSplitter(Qt.Vertical)
        self._right_v_splitter.setChildrenCollapsible(False)
        self._right_v_splitter.addWidget(self._build_viewport())
        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.setMinimumHeight(EXECUTION_AREA_MIN_HEIGHT)
        self.ctrl_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._exec_scroll = self._scroll_tab(self._build_execution_controls(), horizontal=True)
        self.ctrl_tabs.addTab(self._exec_scroll, "Execution Controls")
        self._right_v_splitter.addWidget(self.ctrl_tabs)
        self._right_v_splitter.setStretchFactor(0, 1)
        self._right_v_splitter.setStretchFactor(1, 0)
        self._right_v_splitter.setSizes([600, EXECUTION_AREA_PREFERRED_HEIGHT_2D])
        self._2d_exec_splitter_sizes = list(self._right_v_splitter.sizes())
        self._right_v_splitter.splitterMoved.connect(self._on_2d_exec_splitter_moved)
        rl.addWidget(self._right_v_splitter)
        self._main_splitter.addWidget(right)
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes(
            [COMPUTATIONAL_LEFT_PANEL_WIDTH, 1200 - COMPUTATIONAL_LEFT_PANEL_WIDTH]
        )

    @staticmethod
    def _scroll_tab(widget: QWidget, horizontal: bool = False) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded if horizontal else Qt.ScrollBarAlwaysOff
        )
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(0)
        scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        return scroll

    @staticmethod
    def _group(title: str) -> tuple[QGroupBox, QFormLayout]:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setContentsMargins(*GROUP_MARGINS)
        form.setVerticalSpacing(FORM_ROW_SPACING)
        return group, form

    @staticmethod
    def _double(value, minimum=0.0, maximum=1e9, decimals=6, suffix="") -> QDoubleSpinBox:
        spin = CompactDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        spin.wheelEvent = lambda event: event.ignore()
        spin.setMaximumWidth(max(72, CONTROL_MAX_WIDTH_DEFAULT - 20))
        return spin

    @staticmethod
    def _with_unit(spin: QDoubleSpinBox, unit: str, stretch: bool = True) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        spin.setSuffix("")
        layout.addWidget(spin, 0)
        layout.addWidget(QLabel(unit), 0)
        if stretch:
            layout.addStretch(1)
        return row

    @staticmethod
    def _solver_field(label: str, widget: QWidget, stretch: bool = True) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        lbl = QLabel(label)
        lbl.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        layout.addWidget(lbl, 0)
        layout.addWidget(widget, 0)
        if stretch:
            layout.addStretch(1)
        return row

    @staticmethod
    def _limit_width_to_half_or_text(widget, current_width: int) -> None:
        half = max(1, int(current_width) // 2)
        widget.setMaximumWidth(max(half, widget.sizeHint().width()))

    @staticmethod
    def _int(value, minimum=0, maximum=1000000000) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.wheelEvent = lambda event: event.ignore()
        spin.setMaximumWidth(max(72, CONTROL_MAX_WIDTH_DEFAULT - 20))
        return spin

    @staticmethod
    def _fill_boundary_combo(combo: QComboBox) -> None:
        combo.clear()
        combo.addItem("Open", BOUNDARY_OPEN)
        combo.addItem("Reflection", BOUNDARY_SLIP)

    @staticmethod
    def _combo_stored_value(combo: QComboBox) -> str:
        data = combo.currentData()
        return str(data) if data is not None else combo.currentText()

    @staticmethod
    def _set_combo_stored_value(combo: QComboBox, value: str) -> None:
        text = str(value)
        idx = combo.findData(text)
        if idx < 0:
            idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _sync_write_interval_display(self) -> None:
        time_write = self._combo_stored_value(self.cmb_write_control) == "adjustableRunTime"
        self.lbl_write_interval.setText(
            "Write interval (time):" if time_write else "Write interval (steps):"
        )
        self.spin_write_time.setVisible(time_write)
        self.spin_write_time.setEnabled(time_write)
        self.spin_write_steps.setVisible(not time_write)
        self.spin_write_steps.setEnabled(not time_write)
        self.lbl_write_interval_unit.setVisible(time_write)

    def _build_setup_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        group, form = self._group("Domain Definition")
        self.spin_radius = self._double(1.5, 1e-6)
        self.spin_height = self._double(1.5, 1e-6)
        self.spin_cell = self._double(0.05, 1e-6)
        form.addRow("Radius:", self._with_unit(self.spin_radius, "m"))
        form.addRow("Height:", self._with_unit(self.spin_height, "m"))
        form.addRow("Base Cell Size:", self._with_unit(self.spin_cell, "m"))
        form.addRow(self._build_mesh_mode_selector())
        layout.addWidget(group)

        group, form = self._group("Initialization Source")
        self.cmb_source = QComboBox()
        self.cmb_source.addItems([DIRECT_SOURCE, REMAP_SOURCE])
        form.addRow("Source:", self.cmb_source)
        self._limit_width_to_half_or_text(self.cmb_source, 355)
        layout.addWidget(group)

        self.grp_charge, form = self._group("Direct Charge")
        self.cmb_material = QComboBox()
        self.cmb_material.addItems(self.materials_db.keys())
        self.cmb_shape = QComboBox()
        self.cmb_shape.addItems(["Sphere", "Cylinder"])
        self.spin_mass = self._double(1.0, 1e-9)
        self.cmb_material.setMaximumWidth(self.spin_mass.maximumWidth())
        self.cmb_shape.setMaximumWidth(self.spin_mass.maximumWidth())
        self.spin_density = self._double(1630.0, 1e-9)
        self.spin_energy = self._double(4.29e6, 1e-9, 1e12, 2)
        self.spin_hob = self._double(0.5, 0.0)
        self.spin_ld = self._double(2.5, 1e-6, 100.0, 3)
        self.spin_det_height = self._double(0.5, 0.0)
        self.lbl_charge_r = QLabel("—")
        self.lbl_charge_d = QLabel("—")
        self.lbl_charge_l = QLabel("—")
        self.lbl_axis_lock = QLabel("Charge centre r = 0; detonation r = 0 (axisymmetric, locked)")
        self.lbl_axis_lock.setWordWrap(True)
        self.lbl_axis_lock.setStyleSheet(SECONDARY_INFO_STYLE)
        self.lbl_axis_lock.setVisible(False)
        form.addRow("Composition:", self.cmb_material)
        form.addRow("Shape:", self.cmb_shape)
        form.addRow("Mass:", self._with_unit(self.spin_mass, "kg"))
        form.addRow("Density:", self._with_unit(self.spin_density, "kg/m³"))
        form.addRow("Energy:", self._with_unit(self.spin_energy, "J/kg"))
        form.addRow("Height of Burst:", self._with_unit(self.spin_hob, "m"))
        self.lbl_ld_title = QLabel("Cylinder L/D:")
        form.addRow(self.lbl_ld_title, self.spin_ld)
        form.addRow("Detonation height:", self._with_unit(self.spin_det_height, "m"))
        form.addRow("Computed radius:", self.lbl_charge_r)
        form.addRow("Computed diameter:", self.lbl_charge_d)
        self.lbl_length_title = QLabel("Computed length:")
        form.addRow(self.lbl_length_title, self.lbl_charge_l)
        form.addRow(self.lbl_axis_lock)
        layout.addWidget(self.grp_charge)

        self.grp_mapping, form = self._group("Remap")
        self.txt_source_case = QLabel()
        self.txt_source_case.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.txt_source_case.setWordWrap(True)
        self.btn_edit_remap = QPushButton("Edit...")
        self.btn_edit_remap.setToolTip(
            "Choose the current 1D model or import 1D/2D results."
        )
        self._limit_width_to_half_or_text(self.btn_edit_remap, 438)
        source_row = QWidget()
        source_lay = QHBoxLayout(source_row)
        source_lay.setContentsMargins(0, 0, 0, 0)
        source_lay.setSpacing(8)
        source_lay.addWidget(self.txt_source_case, 1)
        source_lay.addWidget(self.btn_edit_remap, 0)
        self.cmb_source_time_mode = QComboBox()
        self.cmb_source_time_mode.addItems(["latest", "specific"])
        self.txt_source_time = QComboBox()
        self.txt_source_time.setEditable(True)
        self.spin_mapped_radius = self._double(0.5, 1e-9)
        self.spin_source_resolution = self._double(0.01, 1e-9)
        self.lbl_mapping_note = QLabel(
            "rotateFields mapping is not conservative; normal mapping uses source-volume "
            "weighting and fallback/extension uses nearest cells."
        )
        self.lbl_mapping_note.setWordWrap(True)
        self.lbl_mapping_note.setStyleSheet(WARNING_STYLE)
        form.addRow("Remap from:", source_row)
        form.addRow("Source time:", self.cmb_source_time_mode)
        form.addRow("Specific time:", self.txt_source_time)
        form.addRow("Mapped radius:", self._with_unit(self.spin_mapped_radius, "m"))
        form.addRow("Source resolution:", self._with_unit(self.spin_source_resolution, "m"))
        form.addRow(self.lbl_mapping_note)

        group, form = self._group("Atmosphere")
        self.spin_pressure = self._double(101325.0, 1.0, 1e10, 2)
        self.spin_temperature = self._double(288.15, 1.0, 1e5, 2)
        form.addRow("Pressure:", self._with_unit(self.spin_pressure, "Pa"))
        form.addRow("Temperature:", self._with_unit(self.spin_temperature, "K"))
        layout.addWidget(group)

        group, form = self._group("Boundaries")
        self.lbl_axis = QLabel("Axisymmetric, locked")
        self.cmb_outer = QComboBox()
        self.cmb_top = QComboBox()
        self.cmb_bottom = QComboBox()
        boundary_width = max(1, (261 * 2) // 3)
        for combo in (self.cmb_outer, self.cmb_top, self.cmb_bottom):
            self._fill_boundary_combo(combo)
            combo.setMaximumWidth(boundary_width)
        self._set_combo_stored_value(self.cmb_bottom, BOUNDARY_SLIP)
        form.addRow("Axis:", self.lbl_axis)
        form.addRow("Outer Radius:", self.cmb_outer)
        form.addRow("Top:", self.cmb_top)
        form.addRow("Ground / Bottom:", self.cmb_bottom)
        layout.addWidget(group)
        layout.addWidget(self.grp_mapping)
        layout.addStretch()
        return page

    def _build_mesh_mode_selector(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)
        title = QLabel("Mesh mode")
        self.cmb_mesh_mode = QComboBox(box)
        self.cmb_mesh_mode.addItems([FIXED_MESH, DYNAMIC_MESH])
        self.cmb_mesh_mode.setCurrentText(DYNAMIC_MESH)
        self.cmb_mesh_mode.hide()
        self.rad_fixed_mesh = QRadioButton("Fixed Mesh")
        self.rad_dyn_mesh = QRadioButton("Dyn Mesh (AMR)")
        self.rad_dyn_mesh.setChecked(True)
        self._mesh_mode_group = QButtonGroup(box)
        self._mesh_mode_group.setExclusive(True)
        self._mesh_mode_group.addButton(self.rad_fixed_mesh)
        self._mesh_mode_group.addButton(self.rad_dyn_mesh)
        layout.addWidget(title)
        layout.addWidget(self.rad_fixed_mesh)
        layout.addWidget(self.rad_dyn_mesh)
        self.btn_mesh_amr = QPushButton("Mesh & AMR")
        self.btn_mesh_amr.setToolTip(
            "Startup charge refinement and runtime wave AMR. "
            "Available when Mesh mode is Dyn Mesh (AMR)."
        )
        layout.addWidget(self.btn_mesh_amr)
        self._limit_width_to_half_or_text(self.btn_mesh_amr, 438)
        return box

    def _open_mesh_amr_dialog(self) -> None:
        if self.cmb_mesh_mode.currentText() != DYNAMIC_MESH:
            return
        self._mesh_dialog.show()
        self._mesh_dialog.raise_()
        self._mesh_dialog.activateWindow()

    def _build_mesh_amr_dialog(self) -> None:
        self._mesh_dialog = QDialog(self)
        self._mesh_dialog.setWindowTitle("Mesh & AMR")
        self._mesh_dialog.setModal(False)
        self._mesh_dialog.resize(480, 640)
        layout = QVBoxLayout(self._mesh_dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._scroll_tab(self._build_mesh_tab()))

    def _on_mesh_mode_radio_toggled(self, checked: bool) -> None:
        if not checked:
            return
        if self.sender() is self.rad_fixed_mesh:
            target = FIXED_MESH
        elif self.sender() is self.rad_dyn_mesh:
            target = DYNAMIC_MESH
        else:
            return
        if self.cmb_mesh_mode.currentText() != target:
            self._defer_viewer_preview = True
            try:
                self.cmb_mesh_mode.setCurrentText(target)
            finally:
                self._defer_viewer_preview = False
                self._preview_flush_timer.start(0)

    def _sync_mesh_mode_radios(self, text: str) -> None:
        dynamic = str(text) == DYNAMIC_MESH
        radio = self.rad_dyn_mesh if dynamic else self.rad_fixed_mesh
        if radio.isChecked():
            return
        radio.blockSignals(True)
        radio.setChecked(True)
        radio.blockSignals(False)

    def _build_mesh_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.lbl_mesh_definition = QLabel()
        self.lbl_mesh_definition.setWordWrap(True)
        self.lbl_mesh_definition.setStyleSheet(SECONDARY_INFO_STYLE)
        layout.addWidget(self.lbl_mesh_definition)

        self.grp_seed, form = self._group("Startup Charge Refinement")
        self.cmb_seed_mode = QComboBox()
        self.cmb_seed_mode.addItems([SEED_MODE_AUTO, SEED_MODE_MANUAL, SEED_MODE_OFF])
        self.spin_seed_target = self._int(8, 1, 100)
        self.spin_seed_min = self._int(6, 1, 100)
        self.spin_seed_level = self._int(0, 0, 8)
        self.spin_seed_max = self._int(5, 0, 8)
        self.spin_seed_buffer = self._int(5, 0, 20)
        self.lbl_seed_plan = QLabel("—")
        self.lbl_seed_plan.setWordWrap(True)
        self.lbl_seed_plan.setStyleSheet(SECONDARY_INFO_STYLE)
        form.addRow("Seed mode:", self.cmb_seed_mode)
        form.addRow("Target cells across:", self.spin_seed_target)
        form.addRow("Minimum acceptable:", self.spin_seed_min)
        form.addRow("Manual level:", self.spin_seed_level)
        form.addRow("Maximum Auto level:", self.spin_seed_max)
        form.addRow("Initial buffer layers:", self.spin_seed_buffer)
        form.addRow(self.lbl_seed_plan)
        layout.addWidget(self.grp_seed)

        self.grp_amr, form = self._group("Runtime Wave AMR")
        self.cmb_estimator = QComboBox()
        self.cmb_estimator.addItems(["densityGradient"])
        self.spin_runtime_level = self._int(1, 1, 8)
        self.spin_refine_threshold = self._double(0.1, 0.0, 1e9, 6)
        self.spin_unrefine_threshold = self._double(0.1, 0.0, 1e9, 6)
        self.spin_refine_interval = self._int(DEFAULT_REFINE_INTERVAL, 1, 100000)
        self.spin_unrefine_interval = self._int(DEFAULT_REFINE_INTERVAL, 1, 100000)
        self.spin_unrefine_interval.setToolTip(
            "Solver steps between unrefinement attempts. "
            "New native Dynamic cases default to the same value as Refine interval."
        )
        self.spin_begin_unrefine = self._double(0.0, 0.0, 1e9, 9, " s")
        self.chk_begin_unrefine = QCheckBox("Enable")
        self.spin_runtime_buffer = self._int(2, 0, 20)
        self.spin_max_cells = self._int(200000000, 1, 2147483647)
        self.chk_dump_level = QCheckBox()
        self.chk_dump_level.setChecked(True)
        self.chk_refine_probes = QCheckBox()
        self.chk_refine_probes.setChecked(True)
        self.chk_refine_probes.setToolTip(
            "dynamicMeshDict Switch: when enabled, force refinement at cells "
            "containing probe locations from the probes2d function object. "
            "This is not a separate controlDict function type."
        )
        self.chk_balancing = QCheckBox()
        self.spin_balance_interval = self._int(10, 1, 100000)
        form.addRow("Estimator:", self.cmb_estimator)
        form.addRow("Max runtime level:", self.spin_runtime_level)
        form.addRow("Refine threshold:", self.spin_refine_threshold)
        form.addRow("Unrefine threshold:", self.spin_unrefine_threshold)
        form.addRow("Refine interval:", self.spin_refine_interval)
        form.addRow("Unrefine interval:", self.spin_unrefine_interval)
        form.addRow("Begin unrefine:", self.chk_begin_unrefine)
        form.addRow("Begin time:", self.spin_begin_unrefine)
        form.addRow("Buffer layers:", self.spin_runtime_buffer)
        form.addRow("maxCells:", self.spin_max_cells)
        form.addRow("dumpLevel:", self.chk_dump_level)
        form.addRow("refineProbes:", self.chk_refine_probes)
        form.addRow("Load balancing:", self.chk_balancing)
        form.addRow("Balance interval:", self.spin_balance_interval)
        layout.addWidget(self.grp_amr)
        layout.addStretch()
        return page

    def _build_output_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group, form = self._group("Fields and VTK/ParaView Output")
        self.output_checks: Dict[str, QCheckBox] = {}
        labels = (
            ("p", "Pressure / overpressure"),
            ("rho", "Density"),
            ("T", "Temperature"),
            ("U", "Velocity"),
            ("alpha.c4", "Explosive fraction"),
            ("lambda.c4", "Reaction progress"),
            ("cellLevel", "Refinement level"),
        )
        for field, label in labels:
            check = QCheckBox(label)
            check.setChecked(field in ("p", "rho", "T", "U", "alpha.c4"))
            self.output_checks[field] = check
            form.addRow(field + ":", check)
        layout.addWidget(group)

        group = QGroupBox("2D Probes — Radius / Height")
        gl = QVBoxLayout(group)
        self.tbl_probes = QTableWidget(0, 3)
        self.tbl_probes.setHorizontalHeaderLabels(["Name", "Radius", "Height"])
        self.tbl_probes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_probes.setSelectionBehavior(QAbstractItemView.SelectRows)
        gl.addWidget(self.tbl_probes)
        buttons = QHBoxLayout()
        self.btn_add_probe = QPushButton("Add Probe")
        self.btn_remove_probe = QPushButton("Remove Probe")
        buttons.addWidget(self.btn_add_probe)
        buttons.addWidget(self.btn_remove_probe)
        buttons.addStretch()
        gl.addLayout(buttons)
        layout.addWidget(group)

        self.btn_prepare_transfer = QPushButton("Prepare 3D Transfer")
        self.btn_prepare_transfer.setToolTip(
            "Write validated 2D source metadata for a later reviewed 2D → 3D workflow."
        )
        layout.addWidget(self.btn_prepare_transfer)
        layout.addStretch()
        return page

    def _build_info_panel(self) -> QFrame:
        frame = QFrame()
        frame.setFixedHeight(INFO_PANEL_HEIGHT + 84)
        frame.setStyleSheet(
            "background:#eef2f6; border:1px solid #c7d0da; border-radius:4px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)
        title = QLabel("Info")
        title.setStyleSheet(INFO_TITLE_STYLE)
        layout.addWidget(title)
        layout.addSpacing(3)
        cells = QFormLayout()
        cells.setContentsMargins(0, 0, 0, 0)
        cells.setHorizontalSpacing(8)
        cells.setVerticalSpacing(2)
        self.lbl_radial_cells_title = QLabel("Radial cells:")
        self.lbl_vertical_cells_title = QLabel("Vertical cells:")
        self.lbl_radial_cells = QLabel("—")
        self.lbl_vertical_cells = QLabel("—")
        for label in (
            self.lbl_radial_cells_title,
            self.lbl_vertical_cells_title,
            self.lbl_radial_cells,
            self.lbl_vertical_cells,
        ):
            label.setStyleSheet(INFO_ROW_STYLE)
        cells.addRow(self.lbl_radial_cells_title, self.lbl_radial_cells)
        cells.addRow(self.lbl_vertical_cells_title, self.lbl_vertical_cells)
        layout.addLayout(cells)
        self.lbl_effective_domain = QLabel("—")
        self.lbl_effective_domain.setWordWrap(True)
        self.lbl_effective_domain.setStyleSheet(INFO_ROW_STYLE)
        layout.addWidget(self.lbl_effective_domain)
        self.lbl_info_total = QLabel("Total computational cells: —")
        self.lbl_info_grid = QLabel("Base grid: —")
        self.lbl_info_charge = QLabel("Charge size/radius: —")
        self.lbl_info_resolution = QLabel("Planned charge resolution: —")
        self.lbl_info_actual = QLabel("")
        for label in (
            self.lbl_info_total,
            self.lbl_info_grid,
            self.lbl_info_charge,
            self.lbl_info_resolution,
            self.lbl_info_actual,
        ):
            label.setStyleSheet(INFO_ROW_STYLE)
            label.setWordWrap(False)
            layout.addWidget(label)
        layout.addStretch()
        self.info_frame = frame
        return frame

    def _build_viewport(self) -> QWidget:
        frame = QWidget()
        frame.setMinimumWidth(0)
        frame.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        controls = QHBoxLayout()
        self.lbl_time = QLabel("Time:", frame)
        self.cmb_time = TimeComboBox(frame)
        self.cmb_time.setMinimumWidth(110)
        self.cmb_time.setToolTip(
            "Simulation time. Cases open at 0. Open this list to pick a later saved time."
        )
        self.cmb_time.addItems(["0"])
        self.lbl_time.hide()
        self.cmb_time.hide()
        self._status_caption_host = QWidget(frame)
        self._status_caption_host.setObjectName("viewportStatusHost")
        host_layout = QHBoxLayout(self._status_caption_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(8)
        host_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._status_caption_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.btn_fit = QPushButton("Fit")
        controls.addWidget(self._status_caption_host, 1)
        controls.addWidget(self.btn_fit, 0)
        layout.addLayout(controls)
        self.viewer = AxisymmetricViewerWidget()
        self.viewer.setMinimumHeight(120)
        layout.addWidget(self.viewer, 1)
        return frame

    def embed_status_caption(self, *widgets) -> None:
        """Place the Ready/status caption on the Fit row, left-aligned."""
        layout = self._status_caption_host.layout()
        for widget in widgets:
            if widget is None:
                continue
            layout.addWidget(widget, 0)
            if isinstance(widget, QLabel):
                widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def _build_solver_controls(self) -> QGroupBox:
        group = QGroupBox("Solver Controls")
        group.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        grid = QGridLayout(group)
        grid.setContentsMargins(*GROUP_MARGINS)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(FORM_ROW_SPACING)
        self.spin_max_co = self._double(0.5, 1e-6, 10.0, 3)
        self.spin_max_co.setFixedWidth(self.spin_max_co.minimumSizeHint().width() * 2)
        self.spin_end_time = self._double(1e-3, 1e-12, 1e6, 9)
        self.spin_delta_t = self._double(1e-8, 1e-15, 1.0, 12)
        self.chk_adjust = QCheckBox()
        self.chk_adjust.setChecked(True)
        self.cmb_write_control = QComboBox()
        self.cmb_write_control.addItem("RunTime", "adjustableRunTime")
        self.cmb_write_control.addItem("timeStep", "timeStep")
        self.cmb_write_control.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.cmb_write_control.setMaximumWidth(self.cmb_write_control.sizeHint().width())
        self.spin_write_time = self._double(1e-5, 1e-12, 1e6, 9)
        self.spin_write_steps = self._int(100, 1)
        self.spin_cycle_write = self._int(0, 0)
        self.spin_cores = self._int(1, 1, 1024)
        time_row = QWidget()
        time_layout = QHBoxLayout(time_row)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(12)
        time_layout.addWidget(
            self._solver_field(
                "End Time:",
                self._with_unit(self.spin_end_time, "s", stretch=False),
                stretch=False,
            ),
            0,
        )
        time_layout.addWidget(
            self._solver_field(
                "Initial time step:",
                self._with_unit(self.spin_delta_t, "s", stretch=False),
                stretch=False,
            ),
            0,
        )
        time_layout.addStretch(1)
        grid.addWidget(time_row, 0, 0, 1, 2)
        cfl_row = QWidget()
        cfl_layout = QHBoxLayout(cfl_row)
        cfl_layout.setContentsMargins(0, 0, 0, 0)
        cfl_layout.setSpacing(6)
        lbl_cfl = QLabel("CFL / maxCo:")
        lbl_cfl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        lbl_adjust = QLabel("Adjust time step:")
        lbl_adjust.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        cfl_layout.addWidget(lbl_cfl, 0)
        cfl_layout.addWidget(self.spin_max_co, 0)
        cfl_layout.addWidget(lbl_adjust, 0)
        cfl_layout.addWidget(self.chk_adjust, 0)
        cfl_layout.addStretch(1)
        grid.addWidget(cfl_row, 1, 0, 1, 2)
        grid.addWidget(self._solver_field("Write control:", self.cmb_write_control), 2, 0)
        self.lbl_write_interval = QLabel("Write interval (time):")
        self.lbl_write_interval.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.lbl_write_interval_unit = QLabel("s")
        interval_row = QWidget()
        interval_layout = QHBoxLayout(interval_row)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        interval_layout.setSpacing(6)
        interval_layout.addWidget(self.lbl_write_interval, 0)
        interval_layout.addWidget(self.spin_write_time, 0)
        interval_layout.addWidget(self.spin_write_steps, 0)
        interval_layout.addWidget(self.lbl_write_interval_unit, 0)
        interval_layout.addStretch(1)
        grid.addWidget(interval_row, 3, 0)
        grid.addWidget(self._solver_field("cycleWrite:", self.spin_cycle_write), 3, 1)
        grid.addWidget(self._solver_field("Processor cores:", self.spin_cores), 4, 0)
        self._sync_write_interval_display()
        self.grp_solver = group
        return group

    def _build_execution_controls(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        group = QGroupBox("Simulation Control")
        gl = QHBoxLayout(group)
        self.btn_initialize = QPushButton("Initialise Model")
        self.btn_exact_end = QPushButton("exact END")
        self.btn_stop = QPushButton("Interrupt")
        self.btn_log = QPushButton("Open Log", group)
        self.btn_log.setVisible(False)
        self.btn_run = self.btn_exact_end  # compatibility alias; no separate Run/Resume action
        self.btn_initialize.setStyleSheet("background-color: #3498db; color: white; padding: 5px;")
        self.btn_exact_end.setStyleSheet("background-color: #1abc9c; color: white; padding: 4px;")
        self.btn_stop.setStyleSheet("background-color: #e67e22; color: white; padding: 5px;")
        for button in (self.btn_initialize, self.btn_exact_end, self.btn_stop):
            button.setMinimumWidth(198)
        actions = QWidget()
        actions.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        v = QVBoxLayout(actions)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.btn_initialize)
        v.addWidget(self.btn_exact_end)
        v.addWidget(self.btn_stop)
        self.lbl_state = QLabel("State: Draft")
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setMaximumWidth(198)
        self.lbl_state.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        v.addWidget(self.lbl_state)
        self.chk_log_scale = QCheckBox("Log scale")
        v.addWidget(self.chk_log_scale)
        v.addStretch()
        gl.addWidget(actions)
        gl.addWidget(self._build_field_selector())
        layout.addWidget(group)
        layout.addWidget(self._build_solver_controls())
        layout.addWidget(self._build_view_display_controls())
        layout.addStretch()
        return page

    def _build_view_display_controls(self) -> QWidget:
        box = QGroupBox("View")
        box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        self.cmb_view_mode = QComboBox()
        self.cmb_view_mode.addItems(["Mirrored View", "Computational Domain View"])
        self.cmb_view_mode.hide()
        self.lbl_mirror_indicator = QLabel(
            "Mirrored display — computational domain is r ≥ 0"
        )
        self.lbl_mirror_indicator.setStyleSheet(SECONDARY_INFO_STYLE)
        self.lbl_mirror_indicator.setWordWrap(True)
        self.lbl_mirror_indicator.hide()
        self.chk_view_mirror = QCheckBox("Mirrored View")
        self.chk_view_mirror.setChecked(True)
        self.chk_view_mesh = QCheckBox("Mesh overlay")
        self.chk_view_probes = QCheckBox("Probe markers")
        self.chk_view_probes.setChecked(True)
        layout.addWidget(self.chk_view_mirror)
        layout.addWidget(self.chk_view_mesh)
        layout.addWidget(self.chk_view_probes)
        layout.addWidget(self.cmb_view_mode)
        layout.addWidget(self.lbl_mirror_indicator)
        layout.addStretch()
        return box

    def _build_field_selector(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(2)
        self.cmb_field = QComboBox(box)
        self.cmb_field.addItems([field for field, _label in VIEW_FIELD_OPTIONS])
        self.cmb_field.hide()
        self._field_group = QButtonGroup(box)
        self._field_group.setExclusive(True)
        self._field_radios: Dict[str, QRadioButton] = {}
        for field, label in VIEW_FIELD_OPTIONS:
            radio = QRadioButton(label)
            self._field_group.addButton(radio)
            self._field_radios[field] = radio
            layout.addWidget(radio)
        self._field_radios["p"].setChecked(True)
        return box

    def _ensure_view_field_option(self, field: str) -> None:
        if self.cmb_field.findText(field) < 0:
            self.cmb_field.addItem(field)
        if field in self._field_radios:
            return
        radio = QRadioButton(field)
        self._field_group.addButton(radio)
        self._field_radios[field] = radio
        self._field_radios[field].toggled.connect(self._on_field_radio_toggled)
        self.cmb_field.parentWidget().layout().addWidget(radio)

    def _on_field_radio_toggled(self, checked: bool) -> None:
        if not checked:
            return
        radio = self.sender()
        for field, widget in self._field_radios.items():
            if widget is radio:
                if self.cmb_field.currentText() != field:
                    self.cmb_field.setCurrentText(field)
                return

    def _sync_field_radios_from_combo(self, field: str) -> None:
        radio = self._field_radios.get(str(field or "").strip())
        if radio is None or radio.isChecked():
            return
        radio.blockSignals(True)
        radio.setChecked(True)
        radio.blockSignals(False)

    def _connect_signals(self) -> None:
        self.cmb_source.currentTextChanged.connect(self._apply_enablement)
        self.btn_edit_remap.clicked.connect(self._open_remap_from_dialog)
        self.cmb_shape.currentTextChanged.connect(self._apply_enablement)
        self.cmb_mesh_mode.currentTextChanged.connect(self._apply_enablement)
        self.cmb_mesh_mode.currentTextChanged.connect(self._sync_mesh_mode_radios)
        self.rad_fixed_mesh.toggled.connect(self._on_mesh_mode_radio_toggled)
        self.rad_dyn_mesh.toggled.connect(self._on_mesh_mode_radio_toggled)
        self.btn_mesh_amr.clicked.connect(self._open_mesh_amr_dialog)
        self.cmb_seed_mode.currentTextChanged.connect(self._apply_enablement)
        self.cmb_write_control.currentTextChanged.connect(self._apply_enablement)
        self.cmb_material.currentTextChanged.connect(self._on_material_changed)
        self.chk_begin_unrefine.toggled.connect(self._apply_enablement)
        self.chk_balancing.toggled.connect(self._apply_enablement)
        self.cmb_view_mode.currentTextChanged.connect(self._on_view_changed)
        self.chk_view_mirror.toggled.connect(self._on_mirror_checkbox_toggled)
        self.cmb_time.currentTextChanged.connect(self._on_time_selector_changed)
        self.cmb_time.popup_requested.connect(self._ensure_time_catalog)
        self.viewer.times_changed.connect(self._on_viewer_times_changed)
        self.cmb_field.currentTextChanged.connect(self.viewer.set_field)
        self.cmb_field.currentTextChanged.connect(self._sync_field_radios_from_combo)
        for radio in self._field_radios.values():
            radio.toggled.connect(self._on_field_radio_toggled)
        self.chk_view_mesh.toggled.connect(self._on_mesh_overlay_toggled)
        self.chk_view_probes.toggled.connect(self._on_probe_view_toggled)
        self.chk_log_scale.toggled.connect(self._on_log_scale_toggled)
        self.btn_fit.clicked.connect(self.viewer.reset_camera)
        self.btn_add_probe.clicked.connect(self._add_probe)
        self.btn_remove_probe.clicked.connect(self._remove_probe)
        self.btn_initialize.clicked.connect(self._request_initialize)
        self.btn_exact_end.clicked.connect(self.sig_request_run_exact_end)
        self.btn_stop.clicked.connect(self.sig_request_stop)
        self.btn_log.clicked.connect(self.sig_request_log)
        self.btn_prepare_transfer.clicked.connect(self._prepare_transfer_requested)

        view_only = {
            self.cmb_view_mode,
            self.chk_view_mirror,
            self.cmb_time,
            self.cmb_field,
            self.chk_view_mesh,
            self.chk_view_probes,
            self.chk_log_scale,
        }
        model_widgets = (
            self.findChildren(QDoubleSpinBox)
            + self.findChildren(QSpinBox)
            + self.findChildren(QComboBox)
            + self.findChildren(QCheckBox)
        )
        for widget in model_widgets:
            if widget in view_only:
                continue
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.valueChanged.connect(self._on_model_changed)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._on_model_changed)
            else:
                widget.toggled.connect(self._on_model_changed)
        self.tbl_probes.itemChanged.connect(self._on_model_changed)

    def _on_material_changed(self, name: str) -> None:
        if self._loading:
            return
        if name == MATERIAL_UNDEFINED_PLACEHOLDER or not name:
            return
        if name not in self.materials_db:
            # Never map an unknown label onto Custom / another catalog entry.
            return
        self._clear_control_undefined("material_name")
        props = self.materials_db[name]
        self._clear_control_undefined("rho_charge")
        self._clear_control_undefined("energy_j_per_kg")
        self.spin_density.setValue(float(props["rho"]))
        self.spin_energy.setValue(float(props["energy"]))

    def _apply_enablement(self, *_args) -> None:
        direct = self.cmb_source.currentText() == DIRECT_SOURCE
        dynamic = self.cmb_mesh_mode.currentText() == DYNAMIC_MESH
        cylinder = self.cmb_shape.currentText() == "Cylinder"
        self.grp_charge.setEnabled(direct)
        self.grp_mapping.setEnabled(not direct)
        self.spin_ld.setVisible(cylinder)
        self.lbl_ld_title.setVisible(cylinder)
        self.lbl_charge_l.setVisible(cylinder)
        self.lbl_length_title.setVisible(cylinder)
        self.grp_seed.setEnabled(direct and dynamic)
        self.grp_amr.setEnabled(dynamic)
        self.btn_mesh_amr.setEnabled(dynamic)
        if not dynamic:
            self._mesh_dialog.hide()
        self.spin_seed_level.setEnabled(self.cmb_seed_mode.currentText() == SEED_MODE_MANUAL)
        self.spin_seed_target.setEnabled(self.cmb_seed_mode.currentText() == SEED_MODE_AUTO)
        self.spin_seed_min.setEnabled(self.cmb_seed_mode.currentText() == SEED_MODE_AUTO)
        self.spin_seed_max.setEnabled(self.cmb_seed_mode.currentText() == SEED_MODE_AUTO)
        self.txt_source_time.setEnabled(self.cmb_source_time_mode.currentText() == "specific")
        self._sync_write_interval_display()
        self.spin_begin_unrefine.setEnabled(self.chk_begin_unrefine.isChecked())
        self.spin_balance_interval.setEnabled(self.chk_balancing.isChecked())
        self.lbl_mesh_definition.setText(
            "Uniform computational wedge; no startup or runtime refinement."
            if not dynamic
            else "Startup charge refinement and moving-wave runtime AMR are independent."
        )
        self._refresh_derived()
        if not self._defer_viewer_preview:
            self._apply_info_mode_visibility()

    def set_source_cases_root(self, path: str) -> None:
        """Work folder that the From-1D browse dialog opens in."""
        self._source_cases_root = os.path.normpath(path) if path else ""
        self.apply_last_1d_remap_default()

    def apply_last_1d_remap_default(self) -> None:
        """Fill Remap from the newest 1D run unless the user picked another case."""
        if not self._remap_from_last_1d:
            return
        latest = self._last_1d_case_dir or latest_case_1d_dir(self._source_cases_root)
        if latest:
            self._set_remap_case_path(latest, from_last_1d=True)

    def set_last_1d_case(self, path: str) -> None:
        """Remember the 1D case that just started and refresh the Remap default."""
        if not path:
            return
        self._last_1d_case_dir = os.path.normpath(path)
        self.apply_last_1d_remap_default()

    def _set_remap_case_path(self, path: str, *, from_last_1d: bool) -> None:
        path = os.path.normpath(path) if path else ""
        self._remap_case_path = path
        self._remap_from_last_1d = bool(from_last_1d)
        if from_last_1d:
            self._remap_kind = RemapFromDialog.CURRENT_1D
        if path:
            if from_last_1d:
                self.txt_source_case.setText("Current 1D model")
            else:
                self.txt_source_case.setText(os.path.basename(path))
            self.txt_source_case.setToolTip(path)
        else:
            self.txt_source_case.setText("")
            self.txt_source_case.setToolTip("")

    def _current_1d_remap_path(self) -> str:
        return self._last_1d_case_dir or latest_case_1d_dir(self._source_cases_root)

    def _open_remap_from_dialog(self) -> None:
        if self.cmb_source.currentText() != REMAP_SOURCE:
            return
        current_1d = self._current_1d_remap_path()
        dialog = RemapFromDialog(
            self,
            current_kind=self._remap_kind,
            has_current_1d=bool(current_1d and os.path.isdir(current_1d)),
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self._apply_remap_from_choice(dialog.selected_kind())

    def _apply_remap_from_choice(self, kind: str) -> None:
        if kind == RemapFromDialog.CURRENT_1D:
            self._remap_from_last_1d = True
            self.apply_last_1d_remap_default()
            if not self._loading:
                self._on_model_changed()
            return
        chosen = self._pick_results_path(kind)
        if not chosen:
            return
        self._remap_kind = kind
        self._set_remap_case_path(chosen, from_last_1d=False)
        if not self._loading:
            self._on_model_changed()

    def _pick_results_path(self, kind: str) -> str:
        start = getattr(self, "_source_cases_root", "") or ""
        if kind == RemapFromDialog.FILE_2D:
            caption = "Select 2D results file"
        else:
            caption = "Select 1D results file"
        path, _filter = QFileDialog.getOpenFileName(
            self,
            caption,
            start,
            "OpenFOAM (case.foam);;All files (*)",
        )
        if not path:
            return ""
        return case_dir_from_picked_path(path)

    def _on_model_changed(self, *_args) -> None:
        if self._loading:
            return
        sender = self.sender()
        if sender is not None and self._undefined_gui_keys:
            for key in list(self._undefined_gui_keys):
                widget = self._widget_for_gui_key(key)
                if widget is not sender:
                    continue
                if key == "material_name":
                    if self.cmb_material.currentText() != MATERIAL_UNDEFINED_PLACEHOLDER:
                        self._clear_control_undefined(key)
                elif isinstance(widget, QDoubleSpinBox):
                    if getattr(widget, "_ggui_undef_prepared", False):
                        if widget.value() > widget.minimum() + 1e-15:
                            self._clear_control_undefined(key)
                    else:
                        self._clear_control_undefined(key)
                else:
                    self._clear_control_undefined(key)
        self.mark_stale()
        if self.sender() in (
            self.cmb_source,
            self.cmb_shape,
            self.cmb_mesh_mode,
            self.cmb_seed_mode,
            self.cmb_write_control,
            self.chk_begin_unrefine,
            self.chk_balancing,
        ):
            return
        self._refresh_derived()

    def _on_mirror_checkbox_toggled(self, checked: bool) -> None:
        self.cmb_view_mode.setCurrentText(
            "Mirrored View" if checked else "Computational Domain View"
        )

    def _on_view_changed(self, text: str) -> None:
        mirrored = text == "Mirrored View"
        if self.chk_view_mirror.isChecked() != mirrored:
            self.chk_view_mirror.blockSignals(True)
            self.chk_view_mirror.setChecked(mirrored)
            self.chk_view_mirror.blockSignals(False)
        self.viewer.set_mirrored_view(mirrored)

    def _ensure_time_catalog(self) -> None:
        self.viewer.ensure_time_catalog()

    def _on_viewer_times_changed(self, labels, selected: str, live_follow: bool) -> None:
        """Keep the Time combo synchronized without mutating the selection."""
        from openfoam_times_2d import LIVE_FOLLOW_LABEL

        items = [str(x) for x in (labels or [])]
        if "0" not in items:
            items = ["0"] + items
        if LIVE_FOLLOW_LABEL not in items:
            items = items + [LIVE_FOLLOW_LABEL]
        current = LIVE_FOLLOW_LABEL if live_follow else str(selected or "0")
        self.cmb_time.blockSignals(True)
        self.cmb_time.clear()
        self.cmb_time.addItems(items)
        idx = self.cmb_time.findText(current)
        if idx < 0 and not live_follow:
            self.cmb_time.addItem(current)
            idx = self.cmb_time.findText(current)
        if idx >= 0:
            self.cmb_time.setCurrentIndex(idx)
        self.cmb_time.blockSignals(False)

    def _on_time_selector_changed(self, text: str) -> None:
        from openfoam_times_2d import LIVE_FOLLOW_LABEL

        label = str(text or "").strip()
        if not label:
            return
        if label == LIVE_FOLLOW_LABEL:
            self.viewer.enable_live_follow()
            return
        self.viewer.set_selected_time_label(label)

    def enter_live_follow_mode(self) -> None:
        """Called when the user starts exact END."""
        self.viewer.enable_live_follow()

    def stop_live_follow_keep_time(self) -> None:
        self.viewer.stop_live_follow_keep_time()

    def _on_mesh_overlay_toggled(self, checked: bool) -> None:
        if self.viewer.is_simulating:
            self.viewer.toggle_mesh_lines(bool(checked))
        else:
            self._refresh_derived()

    def _on_probe_view_toggled(self, checked: bool) -> None:
        if not self.viewer.is_simulating:
            self._refresh_derived()
            return
        probes = [(p.radius, p.height, 0.0) for p in self._probes()]
        self.viewer.toggle_probes(bool(checked), probes)

    def _on_log_scale_toggled(self, checked: bool) -> None:
        self.viewer.set_log_scale(bool(checked))
        self.viewer.force_refresh_view()

    def _refresh_derived(self) -> None:
        # Converted imports use the same live Setup Preview path as native cases
        # so mass/HOB/mesh edits update geometry immediately.
        try:
            if self._undefined_gui_keys & (
                set(REQUIRED_IMPORTED_PHYSICS_KEYS) | {"cell_size"}
            ):
                self.lbl_charge_r.setText("—")
                self.lbl_charge_d.setText("—")
                self.lbl_charge_l.setText("—")
                self._refresh_info()
                return
            inputs = self.get_case_inputs()
            result = validate_case_inputs_2d(inputs)
            domain = result.domain
            if domain:
                self.lbl_radial_cells.setText(str(domain.radial_cells))
                self.lbl_vertical_cells.setText(str(domain.vertical_cells))
                if domain.adjusted:
                    self.lbl_effective_domain.setText(
                        f"Requested R×H {domain.requested_radius:.6g}×{domain.requested_height:.6g} m; "
                        f"effective {domain.effective_radius:.6g}×{domain.effective_height:.6g} m."
                    )
                else:
                    self.lbl_effective_domain.setText("Requested and effective domain match.")
            charge = physical_charge_geometry(inputs)
            radius = charge.radius_m if charge.shape == "Sphere" else charge.cylinder_radius_m
            self.lbl_charge_r.setText(f"{radius:.6g} m")
            self.lbl_charge_d.setText(f"{2 * radius:.6g} m")
            self.lbl_charge_l.setText(
                f"{charge.length_m:.6g} m" if charge.shape == "Cylinder" else "—"
            )
            if result.seed_plan:
                self.lbl_seed_plan.setText(
                    f"{result.seed_plan.reason}. {result.seed_plan.independence_note}"
                )
            elif inputs.mesh_mode == FIXED_MESH:
                self.lbl_seed_plan.setText("Disabled: Fixed Mesh remains uniform.")
            else:
                self.lbl_seed_plan.setText("Disabled for From 1D mapping.")
            if domain and not self.viewer.is_simulating:
                preview = (
                    domain.effective_radius,
                    domain.effective_height,
                    {
                        "shape": inputs.charge_shape,
                        "height": inputs.height_of_burst,
                        "detonation_height": float(inputs.detonation_height),
                        "reflecting_ground": (
                            inputs.bottom_boundary == BOUNDARY_SLIP
                        ),
                        "radius": radius,
                        "length": charge.length_m,
                        "show_grid": bool(self.chk_view_mesh.isChecked()),
                        "cell_size": float(inputs.cell_size),
                        "seed_level": (
                            int(result.seed_plan.level_effective)
                            if result.seed_plan is not None
                            else int(inputs.charge_refinement_level)
                        ),
                        "buffer_layers": int(inputs.buffer_layers),
                    },
                    [(p.radius, p.height) for p in inputs.probes],
                )
                if self._defer_viewer_preview:
                    self._pending_setup_preview = preview
                else:
                    self.viewer.update_axisymmetric_preview(*preview)
            if not self.is_imported_mode:
                self.set_simulation_state(
                    SimulationState2D.VALIDATED
                    if result.valid and self._state == SimulationState2D.DRAFT
                    else self._state
                )
        except Exception as exc:
            # Timer/preview refresh: log without flooding the user with dialogs.
            from ggui_logging import log_operation

            import logging

            log_operation(
                "tab_2d",
                "setup_preview_refresh",
                case_dir=getattr(self, "active_case_dir", None),
                exc=exc,
                level=logging.ERROR,
            )
        self._refresh_info()

    def _refresh_info(self) -> None:
        try:
            if self.is_imported_mode and self._imported_case is not None:
                ext = self._imported_case
                self.lbl_info_total.setText(
                    f"Imported working case: {os.path.basename(ext.active_case_path)}"
                )
                rc = (ext.mapping.get("radial_cells").displayed_value
                      if ext.mapping and ext.mapping.get("radial_cells") else None)
                vc = (ext.mapping.get("vertical_cells").displayed_value
                      if ext.mapping and ext.mapping.get("vertical_cells") else None)
                if rc is not None and vc is not None:
                    self.lbl_info_grid.setText(
                        f"Base grid: {int(rc)} radial × {int(vc)} vertical"
                    )
                else:
                    self.lbl_info_grid.setText(
                        f"Mode: {ext.lifecycle_label} | evidence: "
                        f"{ext.classification.evidence.source}"
                    )
                cr = None
                if ext.mapping and ext.mapping.get("charge_radius"):
                    cr = ext.mapping.get("charge_radius").displayed_value
                centre = None
                if ext.mapping and ext.mapping.get("charge_centre"):
                    centre = ext.mapping.get("charge_centre").displayed_value
                mat = None
                if ext.mapping and ext.mapping.get("material_name"):
                    mat = ext.mapping.get("material_name").displayed_value
                self.lbl_info_charge.setText(
                    f"Charge: {mat or '—'} | r={cr if cr is not None else '—'} m | "
                    f"centre={centre if centre is not None else '—'}"
                )
                self.lbl_info_resolution.setText(
                    f"Source unchanged | R={ext.radius_m if ext.radius_m is not None else '—'} "
                    f"H={ext.height_m if ext.height_m is not None else '—'}"
                )
                if ext.cell_count is not None:
                    owner = os.path.basename(os.path.dirname(ext.mesh_owner_path)) if ext.mesh_owner_path else ""
                    self.lbl_info_actual.setText(
                        f"Actual current cells: {ext.cell_count:,} "
                        f"({ext.cell_count_source}"
                        + (f" / {owner}" if owner else "")
                        + ")"
                    )
                else:
                    self.lbl_info_actual.setText(
                        "Actual current cells: (initialise to generate mesh)"
                    )
                self._apply_info_mode_visibility()
                return
            inputs = self.get_case_inputs()
            result = validate_case_inputs_2d(inputs)
            if not result.domain:
                self._apply_info_mode_visibility()
                return
            domain = result.domain
            self.lbl_info_total.setText(
                f"{'Estimated cells before initialization' if inputs.mesh_mode == DYNAMIC_MESH else 'Total computational cells'}: "
                f"{domain.total_cells:,}"
            )
            self.lbl_info_grid.setText(
                f"Base grid: {domain.radial_cells} radial × {domain.vertical_cells} vertical"
            )
            if result.charge:
                charge = result.charge
                radius = charge.radius_m if charge.shape == "Sphere" else charge.cylinder_radius_m
                self.lbl_info_charge.setText(
                    f"Charge size/radius: {2 * radius:.5g} m diameter / {radius:.5g} m radius"
                )
                level = result.seed_plan.level_effective if result.seed_plan else 0
                resolution = charge.d_min_m / (inputs.cell_size / (2**level))
                self.lbl_info_resolution.setText(
                    f"Planned {'startup ' if result.seed_plan else ''}charge resolution: {resolution:.2f} cells"
                )
            else:
                self.lbl_info_charge.setText("Charge size/radius: mapped source state")
                self.lbl_info_resolution.setText(
                    f"Planned target resolution: {inputs.cell_size:.6g} m"
                )
            source = getattr(self.viewer, "_cell_count_source", None)
            count_time = getattr(self.viewer, "_cell_count_time", None)
            if (
                self._actual_cell_count is not None
                and self._state
                in (
                    SimulationState2D.INITIALIZED,
                    SimulationState2D.RUNNING,
                    SimulationState2D.INTERRUPTED,
                    SimulationState2D.COMPLETED,
                )
            ):
                suffix = ""
                if source and source != "none":
                    if count_time is not None and source == "time_polyMesh":
                        suffix = f" (measured at t={count_time:.6g} s)"
                    elif source == "constant_polyMesh":
                        suffix = " (initialized mesh)"
                    elif source == "vtk_internalMesh":
                        suffix = " (loaded mesh)"
                self.lbl_info_actual.setText(
                    f"Actual current cells: {self._actual_cell_count:,}{suffix}"
                )
            else:
                self.lbl_info_actual.setText("")
        except Exception:
            self.lbl_info_total.setText("Total computational cells: —")
        if not self._defer_viewer_preview:
            self._apply_info_mode_visibility()

    def _apply_info_mode_visibility(self) -> None:
        if not hasattr(self, "info_frame"):
            return
        imported = bool(self.is_imported_mode and self._imported_case is not None)
        dynamic = self.cmb_mesh_mode.currentText() == DYNAMIC_MESH
        show_fixed_counts = imported or not dynamic
        show_amr = imported or dynamic
        self.lbl_radial_cells_title.setVisible(show_fixed_counts)
        self.lbl_radial_cells.setVisible(show_fixed_counts)
        self.lbl_vertical_cells_title.setVisible(show_fixed_counts)
        self.lbl_vertical_cells.setVisible(show_fixed_counts)
        self.lbl_info_grid.setVisible(show_amr)
        self.lbl_info_charge.setVisible(show_amr)
        self.lbl_info_resolution.setVisible(show_amr)
        self.lbl_info_actual.setVisible(bool(self.lbl_info_actual.text().strip()))

    def _flush_setup_preview(self) -> None:
        preview = self._pending_setup_preview
        self._pending_setup_preview = None
        if preview is not None and not self.viewer.is_simulating:
            self.viewer.update_axisymmetric_preview(*preview)
        self._apply_info_mode_visibility()

    def _add_probe(self) -> None:
        row = self.tbl_probes.rowCount()
        self.tbl_probes.insertRow(row)
        for col, text in enumerate((f"P{row + 1}", "0", "0")):
            self.tbl_probes.setItem(row, col, QTableWidgetItem(text))
        self.mark_stale()

    def _remove_probe(self) -> None:
        row = self.tbl_probes.currentRow()
        if row >= 0:
            self.tbl_probes.removeRow(row)
            self.mark_stale()

    def _probes(self) -> tuple[ProbePoint2D, ...]:
        probes: List[ProbePoint2D] = []
        for row in range(self.tbl_probes.rowCount()):
            try:
                probes.append(
                    ProbePoint2D(
                        self.tbl_probes.item(row, 0).text(),
                        float(self.tbl_probes.item(row, 1).text()),
                        float(self.tbl_probes.item(row, 2).text()),
                    )
                )
            except (AttributeError, ValueError):
                continue
        return tuple(probes)

    def apply_output_file_options(self, dim2d) -> None:
        """Apply Output File Options 2D tab: VTK cadence, gauges, VTK field checks."""
        if dim2d.vtk_by_time:
            idx = self.cmb_write_control.findData("adjustableRunTime")
            self.cmb_write_control.setCurrentIndex(idx if idx >= 0 else 0)
            self.spin_write_time.setValue(float(dim2d.vtk_time_s))
        else:
            idx = self.cmb_write_control.findData("timeStep")
            self.cmb_write_control.setCurrentIndex(idx if idx >= 0 else 1)
            self.spin_write_steps.setValue(int(dim2d.vtk_steps))
        g, v = dim2d.gauges, dim2d.vtk
        pairs = (
            ("p", g.pressure or v.pressure),
            ("rho", g.density or v.density),
            ("T", g.temperature or v.temperature),
            ("U", g.velocity or v.velocity),
            ("alpha.c4", g.mass_fractions or v.mass_fractions),
        )
        for field, on in pairs:
            if field in self.output_checks:
                self.output_checks[field].setChecked(bool(on))
        self._enable_impulse = bool(g.impulse)
        self._enable_dynamic_pressure = bool(g.dynamic_pressure)

    def get_case_inputs(self) -> CaseInputs2D:
        name = self.cmb_material.currentText()
        material_undefined = (
            "material_name" in self._undefined_gui_keys
            or name == MATERIAL_UNDEFINED_PLACEHOLDER
            or not name
        )
        if material_undefined:
            name = ""
            props: dict = {}
        else:
            props = dict(self.materials_db.get(name, {}))
            if name == "Custom":
                if "rho_charge" not in self._undefined_gui_keys:
                    props.update(rho=self.spin_density.value())
                if "energy_j_per_kg" not in self._undefined_gui_keys:
                    props.update(energy=self.spin_energy.value())
            elif name and name not in self.materials_db:
                # Unknown combo text must not inherit another material's props.
                props = {}

        def _num_or_none(key: str, spin: QDoubleSpinBox):
            if key in self._undefined_gui_keys:
                return None
            if getattr(spin, "_ggui_undef_prepared", False) and abs(
                spin.value() - spin.minimum()
            ) < 1e-15:
                return None
            return float(spin.value())

        rho_charge = _num_or_none("rho_charge", self.spin_density)
        energy_j_per_kg = _num_or_none("energy_j_per_kg", self.spin_energy)
        mass_kg = _num_or_none("mass_kg", self.spin_mass)
        if "cell_size" in self._undefined_gui_keys:
            cell_size = None
        else:
            cell_size = float(self.spin_cell.value())

        mapping = MappingSource2D(
            case_path=self._remap_case_path or self.txt_source_case.text().strip(),
            time_mode=self.cmb_source_time_mode.currentText(),
            specific_time=self.txt_source_time.currentText().strip(),
            mapped_radius=self.spin_mapped_radius.value(),
            source_resolution=self.spin_source_resolution.value(),
        )
        return CaseInputs2D(
            radius=self.spin_radius.value(),
            height=self.spin_height.value(),
            cell_size=cell_size,
            initialization_source=self.cmb_source.currentText(),
            charge_shape=self.cmb_shape.currentText(),
            height_of_burst=self.spin_hob.value(),
            detonation_height=self.spin_det_height.value(),
            charge_aspect=self.spin_ld.value(),
            mass_kg=mass_kg,
            material_name=name or "",
            rho_charge=rho_charge,
            energy_j_per_kg=energy_j_per_kg,
            material_props=props,
            p_atm=self.spin_pressure.value(),
            t_atm=self.spin_temperature.value(),
            outer_boundary=self._combo_stored_value(self.cmb_outer),
            top_boundary=self._combo_stored_value(self.cmb_top),
            bottom_boundary=self._combo_stored_value(self.cmb_bottom),
            max_co=self.spin_max_co.value(),
            end_time_s=self.spin_end_time.value(),
            delta_t=self.spin_delta_t.value(),
            adjust_time_step=self.chk_adjust.isChecked(),
            write_control_type=self._combo_stored_value(self.cmb_write_control),
            write_interval_time=self.spin_write_time.value(),
            write_interval_steps=self.spin_write_steps.value(),
            cycle_write=self.spin_cycle_write.value(),
            cores=self.spin_cores.value(),
            mesh_mode=self.cmb_mesh_mode.currentText(),
            charge_seed_mode=self.cmb_seed_mode.currentText(),
            charge_refinement_level=self.spin_seed_level.value(),
            charge_seed_target_cells=self.spin_seed_target.value(),
            charge_seed_min_cells=self.spin_seed_min.value(),
            charge_seed_max_level=self.spin_seed_max.value(),
            buffer_layers=self.spin_seed_buffer.value(),
            refine_indicator_field=self.cmb_estimator.currentText(),
            dyn_refine_max=self.spin_runtime_level.value(),
            refine_interval=self.spin_refine_interval.value(),
            unrefine_interval=self.spin_unrefine_interval.value(),
            lower_refine_threshold=self.spin_refine_threshold.value(),
            unrefine_threshold=self.spin_unrefine_threshold.value(),
            begin_unrefine=(
                self.spin_begin_unrefine.value()
                if self.chk_begin_unrefine.isChecked()
                else None
            ),
            n_buffer_layers_dynamic=self.spin_runtime_buffer.value(),
            dynamic_max_cells=self.spin_max_cells.value(),
            dump_level=self.chk_dump_level.isChecked(),
            refine_probes=self.chk_refine_probes.isChecked(),
            enable_balancing=self.chk_balancing.isChecked(),
            balance_interval=(
                self.spin_balance_interval.value()
                if self.chk_balancing.isChecked()
                else None
            ),
            mapping=mapping,
            probes=self._probes(),
            output_fields=tuple(
                field for field, check in self.output_checks.items() if check.isChecked()
            ),
            enable_impulse=bool(getattr(self, "_enable_impulse", True)),
            enable_dynamic_pressure=bool(getattr(self, "_enable_dynamic_pressure", False)),
            mirrored_view=self.cmb_view_mode.currentText() == "Mirrored View",
            show_mesh=self.chk_view_mesh.isChecked(),
            show_probes=self.chk_view_probes.isChecked(),
            log_scale=self.chk_log_scale.isChecked(),
            undefined_keys=tuple(sorted(self._undefined_gui_keys)),
        )

    def set_case_inputs(self, data: dict) -> None:
        values = dict(data)
        self._set_control_values(values, clear_imported=True, manage_loading=True)

    def _set_control_values(
        self, values: dict, *, clear_imported: bool, manage_loading: bool = True
    ) -> None:
        mapping = values.get("mapping", {})
        if isinstance(mapping, MappingSource2D):
            mapping = asdict(mapping)
        probes = values.get("probes", [])
        if manage_loading:
            self._loading = True
        try:
            setters = (
                (self.spin_radius, "radius"),
                (self.spin_height, "height"),
                (self.spin_cell, "cell_size"),
                (self.spin_hob, "height_of_burst"),
                (self.spin_det_height, "detonation_height"),
                (self.spin_ld, "charge_aspect"),
                (self.spin_mass, "mass_kg"),
                (self.spin_density, "rho_charge"),
                (self.spin_energy, "energy_j_per_kg"),
                (self.spin_pressure, "p_atm"),
                (self.spin_temperature, "t_atm"),
                (self.spin_max_co, "max_co"),
                (self.spin_end_time, "end_time_s"),
                (self.spin_delta_t, "delta_t"),
                (self.spin_write_time, "write_interval_time"),
                (self.spin_write_steps, "write_interval_steps"),
                (self.spin_cycle_write, "cycle_write"),
                (self.spin_cores, "cores"),
                (self.spin_seed_level, "charge_refinement_level"),
                (self.spin_seed_target, "charge_seed_target_cells"),
                (self.spin_seed_min, "charge_seed_min_cells"),
                (self.spin_seed_max, "charge_seed_max_level"),
                (self.spin_seed_buffer, "buffer_layers"),
                (self.spin_runtime_level, "dyn_refine_max"),
                (self.spin_refine_interval, "refine_interval"),
                (self.spin_unrefine_interval, "unrefine_interval"),
                (self.spin_refine_threshold, "lower_refine_threshold"),
                (self.spin_unrefine_threshold, "unrefine_threshold"),
                (self.spin_runtime_buffer, "n_buffer_layers_dynamic"),
                (self.spin_max_cells, "dynamic_max_cells"),
            )
            for widget, key in setters:
                if key in values and values[key] is not None:
                    widget.setValue(values[key])
            combos = (
                (self.cmb_source, "initialization_source"),
                (self.cmb_shape, "charge_shape"),
                (self.cmb_material, "material_name"),
                (self.cmb_outer, "outer_boundary"),
                (self.cmb_top, "top_boundary"),
                (self.cmb_bottom, "bottom_boundary"),
                (self.cmb_write_control, "write_control_type"),
                (self.cmb_mesh_mode, "mesh_mode"),
                (self.cmb_seed_mode, "charge_seed_mode"),
                (self.cmb_estimator, "refine_indicator_field"),
            )
            for widget, key in combos:
                if key not in values:
                    continue
                if key == "material_name" and not values[key]:
                    continue
                stored = str(values[key])
                if widget in (
                    self.cmb_outer,
                    self.cmb_top,
                    self.cmb_bottom,
                    self.cmb_write_control,
                ):
                    self._set_combo_stored_value(widget, stored)
                else:
                    widget.setCurrentText(stored)
            checks = (
                (self.chk_adjust, "adjust_time_step"),
                (self.chk_dump_level, "dump_level"),
                (self.chk_refine_probes, "refine_probes"),
                (self.chk_balancing, "enable_balancing"),
                (self.chk_view_mesh, "show_mesh"),
                (self.chk_view_probes, "show_probes"),
                (self.chk_log_scale, "log_scale"),
            )
            for widget, key in checks:
                if key in values:
                    widget.setChecked(bool(values[key]))
            if "begin_unrefine" in values:
                self.chk_begin_unrefine.setChecked(values.get("begin_unrefine") is not None)
                if values.get("begin_unrefine") is not None:
                    self.spin_begin_unrefine.setValue(values["begin_unrefine"])
            if "balance_interval" in values and values.get("balance_interval") is not None:
                self.chk_balancing.setChecked(True)
                self.spin_balance_interval.setValue(int(values["balance_interval"]))
            if "mirrored_view" in values:
                self.cmb_view_mode.setCurrentText(
                    "Mirrored View" if values.get("mirrored_view", True)
                    else "Computational Domain View"
                )
            if mapping:
                loaded_case = str(mapping.get("case_path", "") or "").strip()
                if loaded_case:
                    self._remap_kind = RemapFromDialog.FILE_1D
                    self._set_remap_case_path(loaded_case, from_last_1d=False)
                self.cmb_source_time_mode.setCurrentText(str(mapping.get("time_mode", "latest")))
                self.txt_source_time.setEditText(str(mapping.get("specific_time", "")))
                if mapping.get("mapped_radius") is not None:
                    self.spin_mapped_radius.setValue(float(mapping.get("mapped_radius", 0.5)))
                if mapping.get("source_resolution") is not None:
                    self.spin_source_resolution.setValue(
                        float(mapping.get("source_resolution") or 0.01)
                    )
            if "probes" in values or probes:
                self.tbl_probes.setRowCount(0)
                for item in probes:
                    if isinstance(item, ProbePoint2D):
                        item = asdict(item)
                    row = self.tbl_probes.rowCount()
                    self.tbl_probes.insertRow(row)
                    for col, key in enumerate(("name", "radius", "height")):
                        self.tbl_probes.setItem(row, col, QTableWidgetItem(str(item[key])))
            if "output_fields" in values:
                selected = set(values.get("output_fields", ()))
                for field, check in self.output_checks.items():
                    check.setChecked(field in selected)
            if "enable_impulse" in values:
                self._enable_impulse = bool(values["enable_impulse"])
            if "enable_dynamic_pressure" in values:
                self._enable_dynamic_pressure = bool(values["enable_dynamic_pressure"])
        finally:
            if manage_loading:
                self._loading = False
        pending_undefined = tuple(values.get("undefined_keys") or ())
        if clear_imported:
            self._active_case_dir = None
            self._actual_cell_count = None
            self.clear_imported_case()
            self.set_simulation_state(SimulationState2D.DRAFT)
            self._apply_enablement()
        if pending_undefined:
            was_loading = self._loading
            self._loading = True
            try:
                for key in pending_undefined:
                    self._mark_control_undefined(str(key))
            finally:
                self._loading = was_loading

    def _request_initialize(self) -> None:
        if self.is_imported_mode:
            self.sig_request_init.emit(self.get_case_inputs())
            return
        inputs = self.get_case_inputs()
        result = validate_case_inputs_2d(inputs)
        if result.valid:
            self.set_simulation_state(SimulationState2D.VALIDATED)
        self.sig_request_init.emit(inputs)

    def _prepare_transfer_requested(self) -> None:
        self.sig_request_prepare_transfer.emit()

    def _on_2d_exec_splitter_moved(self, _pos: int = 0, _index: int = 0) -> None:
        self._2d_exec_splitter_sizes = list(self._right_v_splitter.sizes())

    def get_computational_left_width(self) -> int:
        sizes = self._main_splitter.sizes()
        return int(sizes[0]) if sizes else COMPUTATIONAL_LEFT_PANEL_WIDTH

    def set_computational_left_width(self, width: int) -> None:
        width = max(COMPUTATIONAL_LEFT_PANEL_MIN, int(width))
        total = sum(self._main_splitter.sizes()) or width + 800
        self._main_splitter.setSizes([width, max(50, total - width)])
