"""Production Cylindrical–2D axisymmetric workflow tab."""
from __future__ import annotations

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
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
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
        self._build_ui()
        self.viewer.cell_count_updated.connect(self._on_cell_count_updated)
        self.viewer.log_scale_rejected.connect(self._on_log_scale_rejected)
        self._connect_signals()
        self._apply_enablement()
        self._refresh_derived()

    @property
    def simulation_state(self) -> SimulationState2D:
        return self._state

    def set_simulation_state(self, state: SimulationState2D | str) -> None:
        state = SimulationState2D(state)
        self._state = state
        self.lbl_state.setText(f"State: {state.value}")
        running = state == SimulationState2D.RUNNING
        initialized = state in (
            SimulationState2D.INITIALIZED,
            SimulationState2D.INTERRUPTED,
            SimulationState2D.COMPLETED,
        )
        self.btn_initialize.setEnabled(not running)
        self.btn_exact_end.setEnabled(initialized and not running)
        self.btn_stop.setEnabled(running)
        self.sig_state_changed.emit(state.value)

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
        mapping = values.get("mapping", {})
        if isinstance(mapping, MappingSource2D):
            mapping = asdict(mapping)
        probes = values.get("probes", [])
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
            self.chk_begin_unrefine.setChecked(values.get("begin_unrefine") is not None)
            if values.get("begin_unrefine") is not None:
                self.spin_begin_unrefine.setValue(values["begin_unrefine"])
            self.cmb_view_mode.setCurrentText(
                "Mirrored View" if values.get("mirrored_view", True)
                else "Computational Domain View"
            )
            self.txt_source_case.setEditText(str(mapping.get("case_path", "")))
            self.cmb_source_time_mode.setCurrentText(str(mapping.get("time_mode", "latest")))
            self.txt_source_time.setEditText(str(mapping.get("specific_time", "")))
            self.spin_mapped_radius.setValue(float(mapping.get("mapped_radius", 0.5)))
            self.spin_source_resolution.setValue(
                float(mapping.get("source_resolution") or 0.01)
            )
            self.tbl_probes.setRowCount(0)
            for item in probes:
                if isinstance(item, ProbePoint2D):
                    item = asdict(item)
                row = self.tbl_probes.rowCount()
                self.tbl_probes.insertRow(row)
                for col, key in enumerate(("name", "radius", "height")):
                    self.tbl_probes.setItem(row, col, QTableWidgetItem(str(item[key])))
            selected = set(values.get("output_fields", ()))
            for field, check in self.output_checks.items():
                check.setChecked(field in selected)
        finally:
            self._loading = False
        self._active_case_dir = None
        self._actual_cell_count = None
        self.set_simulation_state(SimulationState2D.DRAFT)
        self._apply_enablement()

    def _request_initialize(self) -> None:
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
