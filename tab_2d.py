"""Production Cylindrical–2D axisymmetric workflow tab."""
from __future__ import annotations

import os
from dataclasses import asdict, replace
from typing import Dict, List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
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
    DIRECT_SOURCE,
    DYNAMIC_MESH,
    FIXED_MESH,
    REMAP_SOURCE,
    align_axisymmetric_domain,
    validate_case_inputs_2d,
)
from axisymmetric_viewer import AxisymmetricViewerWidget
from charge_seed_plan import SEED_MODE_AUTO, SEED_MODE_MANUAL, SEED_MODE_OFF
from external_case_workflow_2d import ImportMode2D, import_mode_label
from imported_case_mapping_2d import FieldProvenance
from material_catalog import materials_copy
from models_2d import (
    CaseInputs2D,
    MappingSource2D,
    ProbePoint2D,
    SimulationState2D,
)
from physical_charge_geometry import physical_charge_geometry
from ui_metrics import (
    ACTION_BUTTON_FONT_PT,
    COMPUTATIONAL_LEFT_PANEL_MIN,
    COMPUTATIONAL_LEFT_PANEL_WIDTH,
    CONTROL_MAX_WIDTH_DEFAULT,
    EXECUTION_AREA_MIN_HEIGHT,
    EXECUTION_AREA_PREFERRED_HEIGHT,
    FORM_ROW_SPACING,
    GROUP_MARGINS,
    INFO_PANEL_HEIGHT,
    INFO_ROW_STYLE,
    INFO_TITLE_STYLE,
    SECONDARY_INFO_STYLE,
    WARNING_STYLE,
)


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
        initialized = state in (
            SimulationState2D.INITIALIZED,
            SimulationState2D.INTERRUPTED,
            SimulationState2D.COMPLETED,
        ) or self._import_mode == ImportMode2D.IMPORTED_2D_READY
        self._apply_action_buttons(running=running, initialized=initialized)
        self.sig_state_changed.emit(state.value)

    def _apply_action_buttons(
        self, *, running: bool = False, initialized: bool = False
    ) -> None:
        self.btn_initialize.setText("Initialise Model")
        self.btn_initialize.setMinimumWidth(140)
        if not self.is_imported_mode:
            self.btn_initialize.setEnabled(not running)
            self.btn_exact_end.setEnabled(initialized and not running)
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
            self.btn_stop.setEnabled(False)  # prep has no safe interrupt yet
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

    def load_imported_case(self, state) -> None:
        """Attach an imported working case and populate normal 2D controls."""
        self._imported_case = state
        self._import_mode = state.mode
        self._active_case_dir = state.active_case_path
        self._actual_cell_count = state.cell_count
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
            if self.cmb_field.findText(field) < 0:
                self.cmb_field.addItem(field)
            self.cmb_field.setCurrentText(field)
            self.viewer.load_case(state.active_case_path)
            self.viewer.set_field(field)
            self.set_simulation_state(SimulationState2D.INITIALIZED)
        else:
            msg = "Imported case — ready to initialise (mesh not generated yet)"
            self.viewer.clear_simulation_view(msg)
            self.viewer.is_simulating = False
            self.set_simulation_state(SimulationState2D.DRAFT)
            self.chk_view_mesh.setToolTip(msg)
        self._apply_action_buttons()
        self._refresh_info()

    # Compatibility aliases.
    def load_external_case(self, state) -> None:
        self.load_imported_case(state)

    def attach_working_copy(self, state) -> None:
        self.load_imported_case(state)

    def clear_imported_case(self) -> None:
        self._imported_case = None
        self._import_mode = ImportMode2D.NATIVE_GGUI_2D
        self._imported_field_meta = {}
        self._unrecovered_gui_keys = set()
        self._case_defined_gui_keys = set()
        self._restore_native_control_editability()
        if hasattr(self, "lbl_import_banner"):
            self.lbl_import_banner.setVisible(False)
        self.btn_initialize.setText("Initialise Model")
        self.btn_initialize.setMinimumWidth(140)
        self.btn_initialize.setEnabled(True)
        self._apply_enablement()

    def clear_external_case(self) -> None:
        self.clear_imported_case()

    def set_prepare_progress(self, utility: str) -> None:
        if not self.is_imported_mode:
            return
        self.set_import_mode(ImportMode2D.IMPORTED_2D_INITIALIZING)
        self.lbl_state.setText(f"State: Initialising imported case ({utility})")
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            app.processEvents()

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
            f"Imported BF case | Source: {src} | Working case: {wc}"
        )
        self.lbl_import_banner.setVisible(True)
        self.lbl_import_banner.setToolTip(
            f"Source (unchanged):\n{self._imported_case.source_dir}\n\n"
            f"Working case:\n{self._imported_case.working_copy_dir or self._imported_case.case_dir}"
        )

    def _apply_import_mapping(self, state) -> None:
        """Populate Setup/Mesh/Output from mapping; never invent native defaults."""
        mapping = state.mapping
        self._imported_field_meta = {}
        self._unrecovered_gui_keys = set()
        self._case_defined_gui_keys = set()
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

            # Apply editability / unrecovered presentation.
            self._apply_imported_control_editability(mapping)
        finally:
            self._loading = False

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
            "lower_refine_threshold": self.spin_refine_threshold,
            "unrefine_threshold": self.spin_unrefine_threshold,
            "n_buffer_layers_dynamic": self.spin_runtime_buffer,
            "dyn_refine_max": self.spin_runtime_level,
            "dynamic_max_cells": self.spin_max_cells,
            "dump_level": self.chk_dump_level,
            "adjust_time_step": self.chk_adjust,
            "outer_boundary": self.cmb_outer,
            "top_boundary": self.cmb_top,
            "bottom_boundary": self.cmb_bottom,
        }.get(key)

    def _apply_imported_control_editability(self, mapping) -> None:
        tip_ro = "Imported case setting — preserved in working copy"
        tip_nr = "Not recovered from imported case — native default not applied"
        tip_cd = "Case-defined — preserved in working copy; not representable by this control"

        for key, mf in mapping.fields.items():
            gui_key = mf.gui_key or key
            widget = self._widget_for_gui_key(gui_key)
            if widget is None:
                continue
            if mf.provenance == FieldProvenance.NOT_RECOVERED:
                widget.setEnabled(False)
                widget.setToolTip(tip_nr + (f": {mf.reason}" if mf.reason else ""))
                if isinstance(widget, QDoubleSpinBox):
                    widget.setSpecialValueText("—")
                    widget.blockSignals(True)
                    widget.setValue(widget.minimum())
                    widget.blockSignals(False)
                continue
            if mf.provenance == FieldProvenance.CASE_DEFINED or not mf.editable:
                widget.setEnabled(False)
                widget.setToolTip(tip_cd if mf.provenance == FieldProvenance.CASE_DEFINED else tip_ro)
                continue
            # Editable proven writers
            widget.setEnabled(True)
            widget.setToolTip(f"Editable — writes {mf.write_target} in working copy only")

        # cell_size case-defined: keep visible but disabled
        if "cell_size" in mapping.case_defined_keys or (
            mapping.get("cell_size")
            and mapping.get("cell_size").provenance == FieldProvenance.CASE_DEFINED
        ):
            self.spin_cell.setEnabled(False)
            self.spin_cell.setToolTip(tip_cd)
            self.lbl_effective_domain.setText("Base mesh: case-defined (see blockMeshDict)")

    def _restore_native_control_editability(self) -> None:
        for key in (
            "radius", "height", "cell_size", "height_of_burst", "detonation_height",
            "charge_aspect", "mass_kg", "rho_charge", "energy_j_per_kg", "p_atm",
            "t_atm", "max_co", "end_time_s", "delta_t", "write_interval_time",
            "write_interval_steps", "material_name", "charge_shape",
            "initialization_source", "write_control_type", "mesh_mode",
            "charge_seed_mode", "refine_indicator_field", "charge_refinement_level",
            "buffer_layers", "refine_interval", "lower_refine_threshold",
            "unrefine_threshold", "n_buffer_layers_dynamic", "dyn_refine_max",
            "dynamic_max_cells", "dump_level", "adjust_time_step",
            "outer_boundary", "top_boundary", "bottom_boundary",
        ):
            widget = self._widget_for_gui_key(key)
            if widget is None:
                continue
            widget.setToolTip("")
            if isinstance(widget, QDoubleSpinBox):
                widget.setSpecialValueText("")
            # Actual enablement is owned by _apply_enablement for native mode.

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

    def mark_stale(self) -> None:
        if self._loading:
            return
        if self.is_imported_mode:
            # Imported mode: only editable controlDict fields may dirty the case.
            return
        if self._state in (
            SimulationState2D.INITIALIZED,
            SimulationState2D.INTERRUPTED,
            SimulationState2D.COMPLETED,
            SimulationState2D.FAILED,
        ):
            self.set_simulation_state(SimulationState2D.STALE)
        elif self._state != SimulationState2D.RUNNING:
            self.set_simulation_state(SimulationState2D.DRAFT)

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
        self.input_tabs.addTab(self._scroll_tab(self._build_mesh_tab()), "Mesh & AMR")
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
        self._right_v_splitter.setSizes([800, EXECUTION_AREA_PREFERRED_HEIGHT])
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
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setMaximumWidth(CONTROL_MAX_WIDTH_DEFAULT)
        return spin

    @staticmethod
    def _int(value, minimum=0, maximum=1000000000) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setMaximumWidth(CONTROL_MAX_WIDTH_DEFAULT)
        return spin

    def _build_setup_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        group, form = self._group("Domain Definition")
        self.spin_radius = self._double(1.5, 1e-6, suffix=" m")
        self.spin_height = self._double(1.5, 1e-6, suffix=" m")
        self.spin_cell = self._double(0.05, 1e-6, suffix=" m")
        self.lbl_radial_cells = QLabel("—")
        self.lbl_vertical_cells = QLabel("—")
        self.lbl_effective_domain = QLabel("—")
        self.lbl_effective_domain.setWordWrap(True)
        self.lbl_effective_domain.setStyleSheet(SECONDARY_INFO_STYLE)
        form.addRow("Radius:", self.spin_radius)
        form.addRow("Height:", self.spin_height)
        form.addRow("Base Cell Size:", self.spin_cell)
        form.addRow("Radial cells:", self.lbl_radial_cells)
        form.addRow("Vertical cells:", self.lbl_vertical_cells)
        form.addRow(self.lbl_effective_domain)
        layout.addWidget(group)

        group, form = self._group("Initialization Source")
        self.cmb_source = QComboBox()
        self.cmb_source.addItems([DIRECT_SOURCE, REMAP_SOURCE])
        form.addRow("Source:", self.cmb_source)
        layout.addWidget(group)

        self.grp_charge, form = self._group("Direct Charge")
        self.cmb_material = QComboBox()
        self.cmb_material.addItems(self.materials_db.keys())
        self.cmb_shape = QComboBox()
        self.cmb_shape.addItems(["Sphere", "Cylinder"])
        self.spin_mass = self._double(1.0, 1e-9, suffix=" kg")
        self.spin_density = self._double(1630.0, 1e-9, suffix=" kg/m³")
        self.spin_energy = self._double(4.29e6, 1e-9, 1e12, 2, " J/kg")
        self.spin_hob = self._double(0.5, 0.0, suffix=" m")
        self.spin_ld = self._double(2.5, 1e-6, 100.0, 3)
        self.spin_det_height = self._double(0.5, 0.0, suffix=" m")
        self.lbl_charge_r = QLabel("—")
        self.lbl_charge_d = QLabel("—")
        self.lbl_charge_l = QLabel("—")
        self.lbl_axis_lock = QLabel("Charge centre r = 0; detonation r = 0 (axisymmetric, locked)")
        self.lbl_axis_lock.setWordWrap(True)
        self.lbl_axis_lock.setStyleSheet(SECONDARY_INFO_STYLE)
        form.addRow("Composition:", self.cmb_material)
        form.addRow("Shape:", self.cmb_shape)
        form.addRow("Mass:", self.spin_mass)
        form.addRow("Density:", self.spin_density)
        form.addRow("Energy:", self.spin_energy)
        form.addRow("Height of Burst:", self.spin_hob)
        self.lbl_ld_title = QLabel("Cylinder L/D:")
        form.addRow(self.lbl_ld_title, self.spin_ld)
        form.addRow("Detonation height:", self.spin_det_height)
        form.addRow("Computed radius:", self.lbl_charge_r)
        form.addRow("Computed diameter:", self.lbl_charge_d)
        self.lbl_length_title = QLabel("Computed length:")
        form.addRow(self.lbl_length_title, self.lbl_charge_l)
        form.addRow(self.lbl_axis_lock)
        layout.addWidget(self.grp_charge)

        self.grp_mapping, form = self._group("1D → 2D rotateFields")
        self.txt_source_case = QComboBox()
        self.txt_source_case.setEditable(True)
        self.cmb_source_time_mode = QComboBox()
        self.cmb_source_time_mode.addItems(["latest", "specific"])
        self.txt_source_time = QComboBox()
        self.txt_source_time.setEditable(True)
        self.spin_mapped_radius = self._double(0.5, 1e-9, suffix=" m")
        self.spin_source_resolution = self._double(0.01, 1e-9, suffix=" m")
        self.lbl_mapping_note = QLabel(
            "rotateFields mapping is not conservative; normal mapping uses source-volume "
            "weighting and fallback/extension uses nearest cells."
        )
        self.lbl_mapping_note.setWordWrap(True)
        self.lbl_mapping_note.setStyleSheet(WARNING_STYLE)
        form.addRow("Source case:", self.txt_source_case)
        form.addRow("Source time:", self.cmb_source_time_mode)
        form.addRow("Specific time:", self.txt_source_time)
        form.addRow("Mapped radius:", self.spin_mapped_radius)
        form.addRow("Source resolution:", self.spin_source_resolution)
        form.addRow(self.lbl_mapping_note)
        layout.addWidget(self.grp_mapping)

        group, form = self._group("Atmosphere")
        self.spin_pressure = self._double(101325.0, 1.0, 1e10, 2, " Pa")
        self.spin_temperature = self._double(288.15, 1.0, 1e5, 2, " K")
        form.addRow("Pressure:", self.spin_pressure)
        form.addRow("Temperature:", self.spin_temperature)
        layout.addWidget(group)

        group, form = self._group("Boundaries")
        self.lbl_axis = QLabel("Axisymmetric, locked")
        self.cmb_outer = QComboBox()
        self.cmb_top = QComboBox()
        self.cmb_bottom = QComboBox()
        for combo in (self.cmb_outer, self.cmb_top, self.cmb_bottom):
            combo.addItems(["Open", "Reflecting slip wall"])
        self.cmb_bottom.setCurrentText("Reflecting slip wall")
        form.addRow("Axis:", self.lbl_axis)
        form.addRow("Outer Radius:", self.cmb_outer)
        form.addRow("Top:", self.cmb_top)
        form.addRow("Ground / Bottom:", self.cmb_bottom)
        layout.addWidget(group)

        group, form = self._group("Solver Controls")
        self.spin_max_co = self._double(0.5, 1e-6, 10.0, 3)
        self.spin_end_time = self._double(1e-3, 1e-12, 1e6, 9, " s")
        self.spin_delta_t = self._double(1e-8, 1e-15, 1.0, 12, " s")
        self.chk_adjust = QCheckBox()
        self.chk_adjust.setChecked(True)
        self.cmb_write_control = QComboBox()
        self.cmb_write_control.addItems(["adjustableRunTime", "timeStep"])
        self.spin_write_time = self._double(1e-5, 1e-12, 1e6, 9, " s")
        self.spin_write_steps = self._int(100, 1)
        self.spin_cycle_write = self._int(0, 0)
        self.spin_cores = self._int(1, 1, 1024)
        form.addRow("CFL / maxCo:", self.spin_max_co)
        form.addRow("End Time:", self.spin_end_time)
        form.addRow("Initial time step:", self.spin_delta_t)
        form.addRow("Adjust time step:", self.chk_adjust)
        form.addRow("Write control:", self.cmb_write_control)
        form.addRow("Write interval (time):", self.spin_write_time)
        form.addRow("Write interval (steps):", self.spin_write_steps)
        form.addRow("cycleWrite:", self.spin_cycle_write)
        form.addRow("Processor cores:", self.spin_cores)
        layout.addWidget(group)
        layout.addStretch()
        return page

    def _build_mesh_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group, form = self._group("Mesh Mode")
        self.cmb_mesh_mode = QComboBox()
        self.cmb_mesh_mode.addItems([FIXED_MESH, DYNAMIC_MESH])
        self.cmb_mesh_mode.setCurrentText(DYNAMIC_MESH)
        self.lbl_mesh_definition = QLabel()
        self.lbl_mesh_definition.setWordWrap(True)
        self.lbl_mesh_definition.setStyleSheet(SECONDARY_INFO_STYLE)
        form.addRow("Mode:", self.cmb_mesh_mode)
        form.addRow(self.lbl_mesh_definition)
        layout.addWidget(group)

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
        self.spin_refine_interval = self._int(3, 1, 100000)
        self.spin_unrefine_interval = self._int(1, 1, 100000)
        self.spin_begin_unrefine = self._double(0.0, 0.0, 1e9, 9, " s")
        self.chk_begin_unrefine = QCheckBox("Enable")
        self.spin_runtime_buffer = self._int(2, 0, 20)
        self.spin_max_cells = self._int(200000000, 1, 2147483647)
        self.chk_dump_level = QCheckBox()
        self.chk_dump_level.setChecked(True)
        self.chk_refine_probes = QCheckBox()
        self.chk_refine_probes.setChecked(True)
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
        frame.setFixedHeight(INFO_PANEL_HEIGHT)
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
        self.cmb_view_mode = QComboBox()
        self.cmb_view_mode.addItems(["Mirrored View", "Computational Domain View"])
        self.lbl_mirror_indicator = QLabel(
            "Mirrored display — computational domain is r ≥ 0"
        )
        self.lbl_mirror_indicator.setStyleSheet(SECONDARY_INFO_STYLE)
        self.cmb_field = QComboBox()
        self.cmb_field.addItems(["p", "rho", "T", "U", "alpha.c4", "cellLevel"])
        self.chk_view_mesh = QCheckBox("Mesh overlay")
        self.chk_view_probes = QCheckBox("Probe markers")
        self.chk_view_probes.setChecked(True)
        self.chk_log_scale = QCheckBox("Log scale")
        self.btn_fit = QPushButton("Fit")
        controls.addWidget(self.cmb_view_mode)
        controls.addWidget(self.lbl_mirror_indicator, 1)
        controls.addWidget(QLabel("Field:"))
        controls.addWidget(self.cmb_field)
        controls.addWidget(self.chk_view_mesh)
        controls.addWidget(self.chk_view_probes)
        controls.addWidget(self.chk_log_scale)
        controls.addWidget(self.btn_fit)
        layout.addLayout(controls)
        self.viewer = AxisymmetricViewerWidget()
        self.viewer.setMinimumHeight(120)
        layout.addWidget(self.viewer, 1)
        return frame

    def _build_execution_controls(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        group = QGroupBox("Simulation Control")
        gl = QHBoxLayout(group)
        self.btn_initialize = QPushButton("Initialise Model")
        self.btn_exact_end = QPushButton("exact END")
        self.btn_stop = QPushButton("Interrupt")
        self.btn_log = QPushButton("Open Log")
        self.btn_run = self.btn_exact_end  # compatibility alias; no separate Run/Resume action
        for button in (self.btn_initialize, self.btn_exact_end, self.btn_stop):
            button.setFixedHeight(50)
            button.setMinimumWidth(140)
            font = button.font()
            font.setPointSize(ACTION_BUTTON_FONT_PT)
            font.setBold(True)
            button.setFont(font)
        self.btn_initialize.setStyleSheet("background:#3498db;color:white;border-radius:6px;")
        self.btn_exact_end.setStyleSheet("background:#2ecc71;color:white;border-radius:6px;")
        self.btn_stop.setStyleSheet("background:#e67e22;color:white;border-radius:6px;")
        gl.addWidget(self.btn_initialize)
        gl.addWidget(self.btn_exact_end)
        gl.addWidget(self.btn_stop)
        gl.addWidget(self.btn_log)
        self.lbl_state = QLabel("State: Draft")
        gl.addWidget(self.lbl_state)
        layout.addWidget(group)
        layout.addStretch()
        return page

    def _connect_signals(self) -> None:
        self.cmb_source.currentTextChanged.connect(self._apply_enablement)
        self.cmb_shape.currentTextChanged.connect(self._apply_enablement)
        self.cmb_mesh_mode.currentTextChanged.connect(self._apply_enablement)
        self.cmb_seed_mode.currentTextChanged.connect(self._apply_enablement)
        self.cmb_write_control.currentTextChanged.connect(self._apply_enablement)
        self.cmb_material.currentTextChanged.connect(self._on_material_changed)
        self.chk_begin_unrefine.toggled.connect(self._apply_enablement)
        self.chk_balancing.toggled.connect(self._apply_enablement)
        self.cmb_view_mode.currentTextChanged.connect(self._on_view_changed)
        self.cmb_field.currentTextChanged.connect(self.viewer.set_field)
        self.chk_view_mesh.toggled.connect(self.viewer.toggle_mesh_lines)
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
        props = self.materials_db.get(name, self.materials_db["Custom"])
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
        self.spin_seed_level.setEnabled(self.cmb_seed_mode.currentText() == SEED_MODE_MANUAL)
        self.spin_seed_target.setEnabled(self.cmb_seed_mode.currentText() == SEED_MODE_AUTO)
        self.spin_seed_min.setEnabled(self.cmb_seed_mode.currentText() == SEED_MODE_AUTO)
        self.spin_seed_max.setEnabled(self.cmb_seed_mode.currentText() == SEED_MODE_AUTO)
        self.txt_source_time.setEnabled(self.cmb_source_time_mode.currentText() == "specific")
        time_write = self.cmb_write_control.currentText() == "adjustableRunTime"
        self.spin_write_time.setEnabled(time_write)
        self.spin_write_steps.setEnabled(not time_write)
        self.spin_begin_unrefine.setEnabled(self.chk_begin_unrefine.isChecked())
        self.spin_balance_interval.setEnabled(self.chk_balancing.isChecked())
        self.lbl_mesh_definition.setText(
            "Uniform computational wedge; no startup or runtime refinement."
            if not dynamic
            else "Startup charge refinement and moving-wave runtime AMR are independent."
        )
        self._refresh_derived()

    def _on_model_changed(self, *_args) -> None:
        if self._loading:
            return
        self.mark_stale()
        self._refresh_derived()

    def _on_view_changed(self, text: str) -> None:
        mirrored = text == "Mirrored View"
        self.lbl_mirror_indicator.setVisible(mirrored)
        self.viewer.set_mirrored_view(mirrored)

    def _on_probe_view_toggled(self, checked: bool) -> None:
        probes = [(p.radius, p.height, 0.0) for p in self._probes()]
        self.viewer.toggle_probes(bool(checked), probes)

    def _on_log_scale_toggled(self, checked: bool) -> None:
        self.viewer.set_log_scale(bool(checked))
        self.viewer.force_refresh_view()

    def _refresh_derived(self) -> None:
        if self.is_imported_mode:
            # Imported display uses mapping/info panel; avoid native mass→radius derivation.
            try:
                if self._imported_case and self._imported_case.radius_m and self._imported_case.height_m:
                    cr = None
                    hob = self.spin_hob.value()
                    if self._imported_case.mapping and self._imported_case.mapping.get("charge_radius"):
                        cr = self._imported_case.mapping.get("charge_radius").displayed_value
                    self.viewer.update_axisymmetric_preview(
                        float(self._imported_case.radius_m),
                        float(self._imported_case.height_m),
                        {
                            "shape": self.cmb_shape.currentText(),
                            "height": hob,
                            "radius": float(cr) if cr is not None else 0.0,
                            "length": 0.0,
                        },
                        [(p.radius, p.height) for p in self._probes()],
                    )
            except Exception:
                pass
            self._refresh_info()
            return
        try:
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
            if domain:
                self.viewer.update_axisymmetric_preview(
                    domain.effective_radius,
                    domain.effective_height,
                    {
                        "shape": inputs.charge_shape,
                        "height": inputs.height_of_burst,
                        "radius": radius,
                        "length": charge.length_m,
                    },
                    [(p.radius, p.height) for p in inputs.probes],
                )
            self.set_simulation_state(
                SimulationState2D.VALIDATED if result.valid and self._state == SimulationState2D.DRAFT
                else self._state
            )
        except Exception:
            pass
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
                return
            inputs = self.get_case_inputs()
            result = validate_case_inputs_2d(inputs)
            if not result.domain:
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

    def get_case_inputs(self) -> CaseInputs2D:
        name = self.cmb_material.currentText()
        props = dict(self.materials_db.get(name, {}))
        if name == "Custom":
            props.update(
                rho=self.spin_density.value(),
                energy=self.spin_energy.value(),
            )
        mapping = MappingSource2D(
            case_path=self.txt_source_case.currentText().strip(),
            time_mode=self.cmb_source_time_mode.currentText(),
            specific_time=self.txt_source_time.currentText().strip(),
            mapped_radius=self.spin_mapped_radius.value(),
            source_resolution=self.spin_source_resolution.value(),
        )
        return CaseInputs2D(
            radius=self.spin_radius.value(),
            height=self.spin_height.value(),
            cell_size=self.spin_cell.value(),
            initialization_source=self.cmb_source.currentText(),
            charge_shape=self.cmb_shape.currentText(),
            height_of_burst=self.spin_hob.value(),
            detonation_height=self.spin_det_height.value(),
            charge_aspect=self.spin_ld.value(),
            mass_kg=self.spin_mass.value(),
            material_name=name,
            rho_charge=self.spin_density.value(),
            energy_j_per_kg=self.spin_energy.value(),
            material_props=props,
            p_atm=self.spin_pressure.value(),
            t_atm=self.spin_temperature.value(),
            outer_boundary=self.cmb_outer.currentText(),
            top_boundary=self.cmb_top.currentText(),
            bottom_boundary=self.cmb_bottom.currentText(),
            max_co=self.spin_max_co.value(),
            end_time_s=self.spin_end_time.value(),
            delta_t=self.spin_delta_t.value(),
            adjust_time_step=self.chk_adjust.isChecked(),
            write_control_type=self.cmb_write_control.currentText(),
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
            mirrored_view=self.cmb_view_mode.currentText() == "Mirrored View",
            show_mesh=self.chk_view_mesh.isChecked(),
            show_probes=self.chk_view_probes.isChecked(),
            log_scale=self.chk_log_scale.isChecked(),
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
                if key in values:
                    widget.setCurrentText(str(values[key]))
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
            if "mirrored_view" in values:
                self.cmb_view_mode.setCurrentText(
                    "Mirrored View" if values.get("mirrored_view", True)
                    else "Computational Domain View"
                )
            if mapping:
                self.txt_source_case.setEditText(str(mapping.get("case_path", "")))
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
        finally:
            if manage_loading:
                self._loading = False
        if clear_imported:
            self._active_case_dir = None
            self._actual_cell_count = None
            self.clear_imported_case()
            self.set_simulation_state(SimulationState2D.DRAFT)
            self._apply_enablement()

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
