"""Validation & Verification workspace. Calculation engines live in validation/."""
from __future__ import annotations

from dataclasses import replace as dc_replace
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tab_1d import MplCanvas
from tab_time_history import GaugeRow, catalog_rows
from ui_metrics import (
    COMPUTATIONAL_LEFT_PANEL_MIN,
    COMPUTATIONAL_LEFT_PANEL_WIDTH,
    ERROR_STATUS_STYLE,
    INFO_STATUS_STYLE,
    SECONDARY_INFO_STYLE,
    WARNING_STYLE,
)
from validation.auto_points import (
    DEFAULT_LOGICAL_DPI_X,
    DEFAULT_PLOT_WIDTH_PX,
    SamplingPlan,
    ValidationPoint,
    cache_key,
    live_fingerprint,
    marker_stride,
    plan_1d,
    plan_2d,
    sampling_burst_1d,
    sampling_burst_2d,
    stamp_plan,
)
from validation import conwep as conwep_engine
from validation import hob as hob_engine
from validation import kb_overlay
from validation import kingery_bulmash as kb
from validation import numerical as numerical_engine
from validation import rankine_hugoniot as rh
from validation import remap as remap_engine
from validation import ufc_airblast as ufc_ab
from validation import ufc_ground
from validation import ufc_hob
from validation import ufc_waveform
from validation.history_quality import assess_history, comparable_peak_impulse
from validation.kb_propagation import (
    copied_1d2d_radius_m,
    copied_2d3d_radius_m,
    kb_propagation_eligible,
    physical_standoff_m,
    scaled_z_from_r,
    series_label,
)
from validation.current_run import (
    MISSING_CURRENT_RUN,
    SOURCE_CURRENT,
    SOURCE_MANUAL,
    ContextProvider,
    RunSnapshot,
    case_dir_for_dim,
    charge_center_for_dim,
    default_display_dims,
    histories_available,
    primary_case_dir,
)
from validation.map_1d import KIND_EXACT, mapped_peak_impulse, map_radius
from validation.metrics import is_finite_number, relative_error_percent
from validation.probes import (
    EXISTING_1D_GRAPH_FO,
    PROBE_FO,
    VALIDATION_FO,
    latest_probe_field_file,
    match_probe_to_point,
    parse_probe_history,
    peak_and_impulse,
    radial_distance,
    radii_from_locations,
    series_for_index,
    standoff_m,
)
from validation.sampling_io import (
    LEGACY_NO_VALIDATION_HISTORIES,
    PLANNED_NOT_RUN,
    THREE_D_HEMI_NA,
    load_matching_plan,
    read_sampling_plan,
)
from validation.spatial import list_saved_times, load_pressure_rz
from validation.units import fmt, pa_s_to_kpa_ms, pa_to_kpa, s_to_ms

MODE_KB = "Kingery-Bulmash"
MODE_CONWEP = "CONWEP"
MODE_HOB = "HOB / Single Reflection"
MODE_REMAP = "Remap Validation"
MODE_NUMERICAL = "Numerical"

ERROR_TOOLTIP = (
    "Error % = (BF - Reference) / Reference * 100. "
    "Near-zero references are reported as N/A."
)


class HobExtractWorker(QThread):
    finished_ok = pyqtSignal(object)
    finished_error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(
        self,
        case_dir: str,
        times: Sequence[Tuple[float, str]],
        field: str,
        z_ground: float,
        *,
        plane: str = "axisymmetric",
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        super().__init__()
        self._case_dir = case_dir
        self._times = list(times)
        self._field = field
        self._z_ground = z_ground
        self._plane = plane
        self._origin = origin
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        samples = []
        total = max(len(self._times), 1)
        for index, (tval, label) in enumerate(self._times):
            if self._cancel:
                self.finished_error.emit("Cancelled.")
                return
            self.progress.emit(int(100 * index / total), f"Time {label}")
            r, z, p, err = load_pressure_rz(
                self._case_dir, label, self._field, plane=self._plane, origin=self._origin
            )
            if err or r is None:
                samples.append(
                    hob_engine.TriplePointSample(time_s=float(tval), x_tp=None, z_tp=None, hm=None, reason=err)
                )
                continue
            fronts = hob_engine.extract_fronts(r, z, p, z_ground=self._z_ground)
            tp = fronts.triple_point
            samples.append(
                hob_engine.TriplePointSample(
                    time_s=float(tval),
                    x_tp=None if tp is None else tp[0],
                    z_tp=None if tp is None else tp[1],
                    hm=fronts.mach_stem_height,
                    reason=fronts.reason,
                )
            )
        self.finished_ok.emit(samples)


class TabValidation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._context_fn: Optional[ContextProvider] = None
        self._gauges_1d: Callable = lambda: ()
        self._probes_2d: Callable = lambda: ()
        self._probes_3d: Callable = lambda: ()
        self._snapshot = RunSnapshot()
        self._added: List[Tuple[str, int]] = []
        self._conwep_key: Optional[Tuple[str, int]] = None
        self._hob_worker: Optional[HobExtractWorker] = None
        self._hob_cache: Dict[str, object] = {}
        self._auto_key: Optional[tuple] = None
        self._auto_plans: List[SamplingPlan] = []
        self._auto_defaults_key: Optional[tuple] = None
        self._display_sync_key: Optional[tuple] = None
        self._sampling_notes: List[str] = []
        self._probe_cache: Dict[tuple, object] = {}
        self._probe_file_cache: Dict[tuple, object] = {}
        self._probe_path_cache: Dict[tuple, str] = {}
        self._case_end_cache: Dict[tuple, object] = {}
        self._remap_cache: Dict[object, object] = {}
        self._numerical_cache: Dict[str, object] = {}
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(50)
        self._redraw_timer.timeout.connect(self._redraw)
        self._build_ui()

    def set_source_provider(
        self,
        *,
        context: ContextProvider,
        gauges_1d=None,
        probes_2d=None,
        probes_3d=None,
    ) -> None:
        self._context_fn = context
        if gauges_1d is not None:
            self._gauges_1d = gauges_1d
        if probes_2d is not None:
            self._probes_2d = probes_2d
        if probes_3d is not None:
            self._probes_3d = probes_3d
        self.refresh_current_run()

    def refresh_current_run(self, *, reset_manual: bool = False) -> None:
        if self._snapshot.source == SOURCE_MANUAL and not reset_manual:
            self._refresh_banner()
            self.refresh_catalog()
            return
        previous_key = self._auto_key
        if self._context_fn is None:
            self._snapshot = RunSnapshot()
        else:
            try:
                self._snapshot = self._context_fn()
            except Exception:
                self._snapshot = RunSnapshot()
        new_key = self._snapshot_cache_key()
        if new_key != previous_key:
            self._auto_key = new_key
            self._auto_plans = []
            self._auto_defaults_key = None
            self._display_sync_key = None
            self._clear_validation_caches()
        self._refresh_banner()
        self._prefill_from_snapshot()
        self.refresh_catalog()
        self._schedule_redraw()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(QLabel("Mode"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([MODE_KB, MODE_CONWEP, MODE_HOB, MODE_REMAP, MODE_NUMERICAL])
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        top.addWidget(self.combo_mode)
        self.lbl_source = QLabel("Current Run")
        self.lbl_source.setStyleSheet("font-weight: bold;")
        top.addWidget(self.lbl_source)
        self.btn_manual = QPushButton("Browse result…")
        self.btn_manual.clicked.connect(self._browse_manual)
        top.addWidget(self.btn_manual)
        self.btn_use_current = QPushButton("Use current run")
        self.btn_use_current.clicked.connect(self._use_current)
        top.addWidget(self.btn_use_current)
        top.addStretch(1)
        root.addLayout(top)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(INFO_STATUS_STYLE)
        root.addWidget(self.lbl_status)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([COMPUTATIONAL_LEFT_PANEL_WIDTH, 1100])
        splitter.splitterMoved.connect(lambda *_: self._schedule_redraw())
        self._splitter = splitter
        root.addWidget(splitter, 1)
        self._on_mode_changed(MODE_KB)

    def _build_left(self) -> QWidget:
        page = QWidget()
        page.setMinimumWidth(COMPUTATIONAL_LEFT_PANEL_MIN)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.grp_sampling = QGroupBox("Sampling")
        samp = QVBoxLayout(self.grp_sampling)
        self.radio_auto_points = QRadioButton("Automatic Validation Points")
        self.radio_user_gauges = QRadioButton("User Gauges")
        self.radio_auto_points.setChecked(True)
        samp_bg = QButtonGroup(self.grp_sampling)
        samp_bg.addButton(self.radio_auto_points)
        samp_bg.addButton(self.radio_user_gauges)
        self.radio_auto_points.toggled.connect(self._on_sampling_changed)
        samp.addWidget(self.radio_auto_points)
        samp.addWidget(self.radio_user_gauges)
        samp.addWidget(QLabel("Results to display:"))
        dim_row = QHBoxLayout()
        self.chk_show_1d = QCheckBox("1D")
        self.chk_show_2d = QCheckBox("2D")
        self.chk_show_3d = QCheckBox("3D")
        self.chk_auto_1d = self.chk_show_1d
        self.chk_auto_2d = self.chk_show_2d
        for chk in (self.chk_show_1d, self.chk_show_2d, self.chk_show_3d):
            chk.toggled.connect(self._on_auto_dim_changed)
            dim_row.addWidget(chk)
        dim_row.addStretch(1)
        samp.addLayout(dim_row)
        layout.addWidget(self.grp_sampling)
        self.grp_gauges = self._build_gauges()
        layout.addWidget(self.grp_gauges)
        self.stack_mode = QStackedWidget()
        self.page_kb = self._build_kb_controls()
        self.page_conwep = self._build_conwep_controls()
        self.page_hob = self._build_hob_controls()
        self.page_remap = self._build_remap_controls()
        self.page_num = self._build_numerical_controls()
        for page_w in (self.page_kb, self.page_conwep, self.page_hob, self.page_remap, self.page_num):
            self.stack_mode.addWidget(page_w)
        layout.addWidget(self.stack_mode, 1)
        return page

    def _build_gauges(self) -> QGroupBox:
        box = QGroupBox("User Gauges")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel("Filter"))
        filt = QHBoxLayout()
        self.chk_3d = QCheckBox("3D")
        self.chk_2d = QCheckBox("2D")
        self.chk_1d = QCheckBox("1D")
        for chk in (self.chk_3d, self.chk_2d, self.chk_1d):
            chk.setChecked(True)
            chk.toggled.connect(self.refresh_catalog)
            filt.addWidget(chk)
        self.btn_add = QPushButton("Add")
        self.btn_clear = QPushButton("Clear")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_clear.clicked.connect(self._on_clear)
        filt.addWidget(self.btn_add)
        filt.addWidget(self.btn_clear)
        layout.addLayout(filt)
        self.tbl_gauges = QTableWidget(0, 5)
        self.tbl_gauges.setHorizontalHeaderLabels(["ID", "X", "Y", "Z", "Label"])
        self.tbl_gauges.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_gauges.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_gauges.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tbl_gauges, 1)
        return box

    def _build_kb_controls(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        grp = QGroupBox("Configuration")
        form = QVBoxLayout(grp)
        self.radio_kb_sph = QRadioButton("Free-air spherical")
        self.radio_kb_hemi = QRadioButton("Hemispherical surface burst")
        self.radio_kb_hemi.setChecked(True)
        bg = QButtonGroup(grp)
        bg.addButton(self.radio_kb_sph)
        bg.addButton(self.radio_kb_hemi)
        self.radio_kb_sph.toggled.connect(self._schedule_redraw)
        self.radio_kb_hemi.toggled.connect(self._schedule_redraw)
        form.addWidget(self.radio_kb_sph)
        form.addWidget(self.radio_kb_hemi)
        self.combo_kb_source = QComboBox()
        self.combo_kb_source.addItems(["Kingery-Bulmash / Swisdak 1994", "UFC 3-340-02"])
        self.combo_kb_source.currentTextChanged.connect(self._schedule_redraw)
        form.addWidget(QLabel("Reference"))
        form.addWidget(self.combo_kb_source)
        self.spin_kb_mass = QDoubleSpinBox()
        self.spin_kb_mass.setRange(1e-6, 1e9)
        self.spin_kb_mass.setDecimals(4)
        self.spin_kb_mass.setValue(1.0)
        self.spin_kb_mass.valueChanged.connect(self._on_kb_mass)
        mass_row = QHBoxLayout()
        mass_row.addWidget(QLabel("Charge mass W"))
        mass_row.addWidget(self.spin_kb_mass)
        form.addLayout(mass_row)
        self.lbl_kb_mass_conv = QLabel(kb.MASS_CONVENTION)
        self.lbl_kb_mass_conv.setWordWrap(True)
        form.addWidget(self.lbl_kb_mass_conv)
        layout.addWidget(grp)
        self.combo_kb_qty = QComboBox()
        self.combo_kb_qty.addItems(["Peak Pressure", "Positive Impulse"])
        self.combo_kb_qty.currentTextChanged.connect(self._schedule_redraw)
        layout.addWidget(QLabel("Quantity"))
        layout.addWidget(self.combo_kb_qty)
        x_box = QGroupBox("X Axis")
        x_l = QVBoxLayout(x_box)
        self.radio_kb_range = QRadioButton("Range [m]")
        self.radio_kb_z = QRadioButton("Scaled Distance Z")
        self.radio_kb_range.setChecked(True)
        x_bg = QButtonGroup(x_box)
        x_bg.addButton(self.radio_kb_range)
        x_bg.addButton(self.radio_kb_z)
        self.radio_kb_range.toggled.connect(self._schedule_redraw)
        self.radio_kb_z.toggled.connect(self._schedule_redraw)
        x_l.addWidget(self.radio_kb_range)
        x_l.addWidget(self.radio_kb_z)
        layout.addWidget(x_box)
        self.chk_kb_z = self.radio_kb_z
        sc_box = QGroupBox("Scale")
        sc_l = QVBoxLayout(sc_box)
        self.radio_kb_log = QRadioButton("Logarithmic")
        self.radio_kb_lin = QRadioButton("Linear")
        self.radio_kb_log.setChecked(True)
        sc_bg = QButtonGroup(sc_box)
        sc_bg.addButton(self.radio_kb_log)
        sc_bg.addButton(self.radio_kb_lin)
        self.radio_kb_log.toggled.connect(self._schedule_redraw)
        self.radio_kb_lin.toggled.connect(self._schedule_redraw)
        sc_l.addWidget(self.radio_kb_log)
        sc_l.addWidget(self.radio_kb_lin)
        layout.addWidget(sc_box)
        self.lbl_kb_info = QLabel("")
        self.lbl_kb_info.setWordWrap(True)
        self.lbl_kb_info.setStyleSheet(SECONDARY_INFO_STYLE)
        layout.addWidget(self.lbl_kb_info)
        layout.addStretch(1)
        return w

    def _build_conwep_controls(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.spin_cw_mass = QDoubleSpinBox()
        self.spin_cw_mass.setRange(1e-6, 1e9)
        self.spin_cw_mass.setDecimals(4)
        self.spin_cw_mass.setValue(1.0)
        self.spin_cw_mass.valueChanged.connect(self._schedule_redraw)
        self.edit_cw_type = QLineEdit("TNT")
        self.spin_cw_standoff = QDoubleSpinBox()
        self.spin_cw_standoff.setRange(1e-6, 1e6)
        self.spin_cw_standoff.setDecimals(4)
        self.spin_cw_standoff.setValue(1.0)
        self.spin_cw_standoff.valueChanged.connect(self._schedule_redraw)
        self.combo_cw_ptype = QComboBox()
        self.combo_cw_ptype.addItems(["Incident", "Reflected"])
        self.combo_cw_ptype.currentTextChanged.connect(self._schedule_redraw)
        form.addRow("Explosive Weight", self.spin_cw_mass)
        form.addRow("Explosive Type", self.edit_cw_type)
        form.addRow("Standoff Distance", self.spin_cw_standoff)
        self.lbl_cw_standoff_src = QLabel("")
        self.lbl_cw_standoff_src.setWordWrap(True)
        self.lbl_cw_standoff_src.setStyleSheet(SECONDARY_INFO_STYLE)
        form.addRow(self.lbl_cw_standoff_src)
        form.addRow("Pressure Type", self.combo_cw_ptype)
        hint = QLabel(
            "CONWEP scalars reuse Swisdak 1994 Kingery-Bulmash parameters. "
            "CONWEP waveform is N/A. An overlay labeled UFC Calc is the workbook "
            "modified Friedlander (not CONWEP)."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        return w

    def _build_hob_controls(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Result Source"))
        self.radio_hob_2d = QRadioButton("2D Axisymmetric")
        self.radio_hob_3d = QRadioButton("3D Section")
        self.radio_hob_2d.setChecked(True)
        g = QButtonGroup(w)
        g.addButton(self.radio_hob_2d)
        g.addButton(self.radio_hob_3d)
        self.radio_hob_2d.toggled.connect(self._on_hob_source)
        self.radio_hob_3d.toggled.connect(self._on_hob_source)
        layout.addWidget(self.radio_hob_2d)
        layout.addWidget(self.radio_hob_3d)
        self.combo_hob_plane = QComboBox()
        self.combo_hob_plane.addItems(["X-Z", "Y-Z"])
        self.combo_hob_plane.currentTextChanged.connect(self._schedule_redraw)
        layout.addWidget(QLabel("Plane"))
        layout.addWidget(self.combo_hob_plane)
        self.combo_hob_plane.setEnabled(False)
        self.combo_hob_field = QComboBox()
        self.combo_hob_field.addItems(["Pressure", "Peak Overpressure", "Density", "Velocity", "Temperature"])
        self.combo_hob_field.currentTextChanged.connect(self._schedule_redraw)
        layout.addWidget(QLabel("Field"))
        layout.addWidget(self.combo_hob_field)
        time_row = QHBoxLayout()
        self.btn_hob_prev = QPushButton("<")
        self.btn_hob_next = QPushButton(">")
        self.combo_hob_time = QComboBox()
        self.slider_hob = QSlider(Qt.Horizontal)
        self.btn_hob_prev.clicked.connect(lambda: self._step_hob_time(-1))
        self.btn_hob_next.clicked.connect(lambda: self._step_hob_time(1))
        self.combo_hob_time.currentIndexChanged.connect(self._on_hob_time)
        self.slider_hob.valueChanged.connect(self._on_hob_slider)
        time_row.addWidget(self.btn_hob_prev)
        time_row.addWidget(self.combo_hob_time, 1)
        time_row.addWidget(self.btn_hob_next)
        layout.addWidget(QLabel("Time"))
        layout.addLayout(time_row)
        layout.addWidget(self.slider_hob)
        self.combo_hob_kind = QComboBox()
        self.combo_hob_kind.addItems(
            [
                "Triple-Point Height vs Ground Range",
                "Shock Front Position vs Time",
                "Incident-wave (UFC Fig 2-7 spherical)",
                "Ground reflected pressure (UFC Fig 2-9)",
                "Ground reflected impulse (UFC Fig 2-10)",
                "UFC pressure-time (workbook Friedlander)",
            ]
        )
        self.combo_hob_kind.currentTextChanged.connect(self._schedule_redraw)
        layout.addWidget(QLabel("Validation type"))
        layout.addWidget(self.combo_hob_kind)
        self.btn_hob_extract = QPushButton("Detect fronts / trajectory")
        self.btn_hob_extract.clicked.connect(self._run_hob_extract)
        layout.addWidget(self.btn_hob_extract)
        self.lbl_hob_progress = QLabel("")
        layout.addWidget(self.lbl_hob_progress)
        rh_box = QGroupBox("Local Rankine–Hugoniot")
        rh_form = QFormLayout(rh_box)
        self.spin_rh_p1 = QDoubleSpinBox()
        self.spin_rh_p1.setRange(1.0, 1e9)
        self.spin_rh_p1.setDecimals(1)
        self.spin_rh_p1.setValue(101325.0)
        self.spin_rh_rho = QDoubleSpinBox()
        self.spin_rh_rho.setRange(1e-6, 1e3)
        self.spin_rh_rho.setDecimals(4)
        self.spin_rh_rho.setValue(1.225)
        self.spin_rh_gamma = QDoubleSpinBox()
        self.spin_rh_gamma.setRange(1.01, 2.0)
        self.spin_rh_gamma.setDecimals(3)
        self.spin_rh_gamma.setValue(1.4)
        self.spin_rh_us = QDoubleSpinBox()
        self.spin_rh_us.setRange(1.0, 1e6)
        self.spin_rh_us.setDecimals(1)
        self.spin_rh_us.setValue(400.0)
        self.spin_rh_theta = QDoubleSpinBox()
        self.spin_rh_theta.setRange(0.0, 45.0)
        self.spin_rh_theta.setDecimals(2)
        self.spin_rh_theta.setValue(10.0)
        for spin in (self.spin_rh_p1, self.spin_rh_rho, self.spin_rh_gamma, self.spin_rh_us, self.spin_rh_theta):
            spin.valueChanged.connect(self._schedule_redraw)
        rh_form.addRow("Pre-shock p", self.spin_rh_p1)
        rh_form.addRow("Pre-shock ρ", self.spin_rh_rho)
        rh_form.addRow("γ", self.spin_rh_gamma)
        rh_form.addRow("Shock speed", self.spin_rh_us)
        rh_form.addRow("Deflection θ (regular)", self.spin_rh_theta)
        hint = QLabel("Oblique relations are disabled when a Mach stem / triple point is detected. Velocity uses the shock-normal component only.")
        hint.setWordWrap(True)
        rh_form.addRow(hint)
        layout.addWidget(rh_box)
        layout.addStretch(1)
        return w

    def _build_remap_controls(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.combo_remap_mode = QComboBox()
        self.combo_remap_mode.addItems(["1D → 2D", "2D → 3D"])
        self.combo_remap_mode.currentTextChanged.connect(self._schedule_redraw)
        layout.addWidget(QLabel("Remap"))
        layout.addWidget(self.combo_remap_mode)
        self.combo_remap_field = QComboBox()
        self.combo_remap_field.addItems(["Pressure", "Density", "Radial Velocity", "Temperature", "alpha.c4"])
        self.combo_remap_field.currentTextChanged.connect(self._schedule_redraw)
        layout.addWidget(QLabel("Field"))
        layout.addWidget(self.combo_remap_field)
        self.combo_remap_diff = QComboBox()
        self.combo_remap_diff.addItems(["Absolute Difference", "Relative Difference"])
        self.combo_remap_diff.currentTextChanged.connect(self._schedule_redraw)
        layout.addWidget(self.combo_remap_diff)
        self.lbl_remap_times = QLabel("Source Time: N/A\nTarget Initialization Time: N/A\nDelta t: N/A")
        self.lbl_remap_times.setWordWrap(True)
        layout.addWidget(self.lbl_remap_times)
        layout.addStretch(1)
        return w

    def _build_numerical_controls(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        num_hint = QLabel(
            "Numerical diagnostics use the current-run solver log, checkMesh log, "
            "and Output File Options. Missing items are N/A, never PASS."
        )
        num_hint.setWordWrap(True)
        layout.addWidget(num_hint)
        self.combo_num_plot = QComboBox()
        self.combo_num_plot.addItems(["Courant Number vs Time", "deltaT vs Time", "Total Cells vs Time"])
        self.combo_num_plot.currentTextChanged.connect(self._schedule_redraw)
        layout.addWidget(self.combo_num_plot)
        layout.addStretch(1)
        return w

    def _build_right(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spatial_canvas = MplCanvas(self, width=6, height=3, dpi=100)
        self.spatial_canvas.tight_layout_rect = (0.0, 0.0, 1.0, 1.0)
        self.plot_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.plot_canvas.tight_layout_rect = (0.08, 0.12, 0.98, 0.92)
        self.spatial_canvas.hide()
        layout.addWidget(self.spatial_canvas, 1)
        layout.addWidget(self.plot_canvas, 1)
        self.table = QTableWidget(0, 1)
        self.table.setToolTip(ERROR_TOOLTIP)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 0)
        return page

    def _mode(self) -> str:
        return self.combo_mode.currentText()

    def _on_mode_changed(self, text: str) -> None:
        mapping = {
            MODE_KB: 0,
            MODE_CONWEP: 1,
            MODE_HOB: 2,
            MODE_REMAP: 3,
            MODE_NUMERICAL: 4,
        }
        self.stack_mode.setCurrentIndex(mapping.get(text, 0))
        show_sampling = text in (MODE_KB, MODE_CONWEP)
        self.grp_sampling.setVisible(show_sampling)
        gauges = show_sampling and self.radio_user_gauges.isChecked()
        self.grp_gauges.setVisible(gauges)
        self.spatial_canvas.setVisible(text == MODE_HOB)
        if text == MODE_HOB:
            self._reload_hob_times()
        self._schedule_redraw()

    def _auto_sampling(self) -> bool:
        return self.radio_auto_points.isChecked()

    def _on_auto_dim_changed(self, *_args) -> None:
        self._schedule_redraw()

    def _on_sampling_changed(self, *_args) -> None:
        show = self._mode() in (MODE_KB, MODE_CONWEP) and self.radio_user_gauges.isChecked()
        self.grp_gauges.setVisible(show)
        self._schedule_redraw()

    def _display_dims(self) -> set:
        self._sync_display_dims()
        dims = set()
        if self.chk_show_1d.isChecked():
            dims.add("1d")
        if self.chk_show_2d.isChecked():
            dims.add("2d")
        if self.chk_show_3d.isChecked():
            dims.add("3d")
        return dims

    def _sync_display_dims(self) -> None:
        key = self._snapshot_cache_key()
        if self._display_sync_key == key:
            return
        wanted = default_display_dims(self._snapshot)
        self._display_sync_key = key
        mapping = (("1d", self.chk_show_1d), ("2d", self.chk_show_2d), ("3d", self.chk_show_3d))
        for dim, chk in mapping:
            chk.blockSignals(True)
            chk.setChecked(dim in wanted)
            chk.blockSignals(False)

    def _catalog(self) -> List[GaugeRow]:
        try:
            g1 = self._gauges_1d() if callable(self._gauges_1d) else ()
            g2 = self._probes_2d() if callable(self._probes_2d) else ()
            g3 = self._probes_3d() if callable(self._probes_3d) else ()
        except Exception:
            g1, g2, g3 = (), (), ()
        return catalog_rows(g1, g2, g3)

    def _enabled_dims(self) -> set:
        dims = set()
        if self.chk_1d.isChecked():
            dims.add("1d")
        if self.chk_2d.isChecked():
            dims.add("2d")
        if self.chk_3d.isChecked():
            dims.add("3d")
        return dims

    def refresh_catalog(self) -> None:
        rows = [row for row in self._catalog() if row.dim in self._enabled_dims()]
        self.tbl_gauges.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = (row.gauge_id, f"{row.x:.6g}", f"{row.y:.6g}", f"{row.z:.6g}", row.label)
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.UserRole, row.key)
                self.tbl_gauges.setItem(r, c, item)

    def _on_add(self) -> None:
        rows = self.tbl_gauges.selectionModel().selectedRows() if self.tbl_gauges.selectionModel() else []
        for model_index in rows:
            item = self.tbl_gauges.item(model_index.row(), 0)
            if item is None:
                continue
            key = item.data(Qt.UserRole)
            if key and key not in self._added:
                self._added.append(tuple(key))
            if key:
                self._conwep_key = tuple(key)
                self._prefill_conwep_standoff()
        self._schedule_redraw()

    def _on_clear(self) -> None:
        self._added.clear()
        self._conwep_key = None
        self._schedule_redraw()

    def _row_by_key(self, key: Tuple[str, int]) -> Optional[GaugeRow]:
        for row in self._catalog():
            if row.key == key:
                return row
        return None

    def _on_kb_mass(self, *_args) -> None:
        if not (is_finite_number(self._snapshot.mass_kg) and float(self._snapshot.mass_kg) > 0.0):
            self._auto_plans = []
        self._schedule_redraw()

    def _kb_use_ufc(self) -> bool:
        return str(self.combo_kb_source.currentText()).startswith("UFC")

    def _kb_burst(self) -> str:
        spherical = self.radio_kb_sph.isChecked()
        if self._kb_use_ufc():
            return ufc_ab.BURST_SPHERICAL if spherical else ufc_ab.BURST_HEMISPHERICAL
        if spherical:
            return kb.BURST_SPHERICAL
        return kb.BURST_HEMISPHERICAL

    def _prefill_from_snapshot(self) -> None:
        mass = self._snapshot.mass_kg
        has_mass = is_finite_number(mass) and float(mass) > 0.0
        self.spin_kb_mass.setEnabled(not has_mass)
        self.spin_cw_mass.setEnabled(not has_mass)
        if has_mass:
            self.spin_kb_mass.blockSignals(True)
            self.spin_cw_mass.blockSignals(True)
            self.spin_kb_mass.setValue(float(mass))
            self.spin_cw_mass.setValue(float(mass))
            self.spin_kb_mass.blockSignals(False)
            self.spin_cw_mass.blockSignals(False)
        if self._snapshot.material_name:
            self.edit_cw_type.setText(str(self._snapshot.material_name))
            self.edit_cw_type.setReadOnly(True)
        else:
            self.edit_cw_type.setReadOnly(False)
        if is_finite_number(self._snapshot.p_atm) and float(self._snapshot.p_atm) > 0.0:
            self.spin_rh_p1.blockSignals(True)
            self.spin_rh_p1.setValue(float(self._snapshot.p_atm))
            self.spin_rh_p1.blockSignals(False)

    def _snapshot_cache_key(self) -> tuple:
        snap = self._snapshot
        return cache_key(
            case_1d=case_dir_for_dim(snap, "1d"),
            case_2d=case_dir_for_dim(snap, "2d"),
            mass_kg=snap.mass_kg,
            domain_1d=snap.domain_radius_1d,
            domain_2d=snap.domain_radius_2d,
            height_2d=snap.domain_height_2d,
            hob_m=snap.hob_m,
        ) + (str(case_dir_for_dim(snap, "3d") or ""), str(snap.live_mode or ""))

    def _logical_dpi(self) -> float:
        try:
            dpi = float(self.logicalDpiX())
        except Exception:
            dpi = DEFAULT_LOGICAL_DPI_X
        return dpi if dpi > 0.0 else DEFAULT_LOGICAL_DPI_X

    def _plot_width_px(self) -> float:
        try:
            w = float(self.plot_canvas.width())
        except Exception:
            w = 0.0
        return w if w >= 80.0 else DEFAULT_PLOT_WIDTH_PX

    def _remap_receive_2d(self) -> Optional[float]:
        snap = self._snapshot
        return copied_1d2d_radius_m(
            target_2d_case=case_dir_for_dim(snap, "2d"),
            source_1d_case=snap.mapping_source_2d or case_dir_for_dim(snap, "1d"),
            widget_mapped_radius=snap.mapped_radius,
        )

    def _remap_receive_3d(self) -> Optional[float]:
        snap = self._snapshot
        return copied_2d3d_radius_m(
            prepare_path=snap.prepare_3d_transfer,
            source_2d_case=case_dir_for_dim(snap, "2d") or snap.remap_3d_source,
        )

    def _live_fingerprint(self, dim: str) -> dict:
        snap = self._snapshot
        mass = snap.mass_kg if is_finite_number(snap.mass_kg) and float(snap.mass_kg) > 0.0 else float(self.spin_kb_mass.value())
        if dim == "1d":
            burst = sampling_burst_1d()
            return live_fingerprint(
                dim="1d",
                case_path=case_dir_for_dim(snap, "1d"),
                mass_kg=mass,
                domain_radius_m=snap.domain_radius_1d,
                hob_m=0.0,
                charge_center=(0.0, 0.0, 0.0),
                cell_size=snap.domain_cell_1d,
                burst_mode=burst,
                figure=ufc_ab.figure_id(burst),
            )
        burst = sampling_burst_2d(snap.hob_m)
        hob = float(snap.hob_m) if is_finite_number(snap.hob_m) else 0.0
        receive = self._remap_receive_2d()
        return live_fingerprint(
            dim="2d",
            case_path=case_dir_for_dim(snap, "2d"),
            mass_kg=mass,
            domain_radius_m=snap.domain_radius_2d,
            domain_height_m=snap.domain_height_2d,
            hob_m=hob,
            charge_center=(0.0, hob, 0.0),
            cell_size=snap.domain_cell_2d,
            burst_mode=burst,
            figure=ufc_ab.figure_id(burst),
            remap_receive_r_max=receive,
        )

    def _stored_plan_for_case(self, case: Optional[str], dim: str, expected: dict) -> Tuple[Optional[SamplingPlan], str]:
        """Reuse the on-disk plan for this completed case even if remap policy drifted.

        Probe locations are facts of the run. A newer receive-radius fingerprint must
        not invent new sample stations that the existing histories do not contain.
        """
        if not case:
            return None, ""
        loaded, mismatch = load_matching_plan(case, expected)
        if loaded is not None and loaded.dim == dim and (loaded.points or loaded.notes):
            return loaded, ""
        stored = read_sampling_plan(case)
        if stored is None or stored.dim != dim or not stored.points:
            return None, mismatch
        stored_id = str((stored.fingerprint or {}).get("case_id") or "")
        live_id = str((expected or {}).get("case_id") or "")
        if stored_id and live_id and stored_id == live_id:
            return stored, ""
        return None, mismatch

    def _collect_auto_plans(self) -> List[SamplingPlan]:
        if self._auto_plans:
            return list(self._auto_plans)
        snap = self._snapshot
        mass = snap.mass_kg if is_finite_number(snap.mass_kg) and float(snap.mass_kg) > 0.0 else float(self.spin_kb_mass.value())
        dpi = self._logical_dpi()
        width = DEFAULT_PLOT_WIDTH_PX
        plans: List[SamplingPlan] = []
        notes: List[str] = []
        if is_finite_number(snap.domain_radius_1d) and float(snap.domain_radius_1d) > 0.0:
            case = case_dir_for_dim(snap, "1d")
            loaded, mismatch = self._stored_plan_for_case(case, "1d", self._live_fingerprint("1d")) if case else (None, "")
            if loaded is not None and loaded.dim == "1d" and loaded.points:
                plans.append(loaded)
            else:
                if mismatch:
                    notes.append(mismatch)
                generated = plan_1d(
                    mass_kg=mass,
                    domain_radius_m=float(snap.domain_radius_1d),
                    cell_size=snap.domain_cell_1d,
                    usable_width_px=width,
                    logical_dpi_x=dpi,
                )
                plans.append(
                    stamp_plan(
                        generated,
                        case_path=case,
                        cell_size=snap.domain_cell_1d,
                        hob_m=0.0,
                    )
                )
        if is_finite_number(snap.domain_radius_2d) and float(snap.domain_radius_2d) > 0.0:
            case = case_dir_for_dim(snap, "2d")
            hob = float(snap.hob_m) if is_finite_number(snap.hob_m) else 0.0
            height = float(snap.domain_height_2d) if is_finite_number(snap.domain_height_2d) else hob
            receive = self._remap_receive_2d()
            loaded, mismatch = self._stored_plan_for_case(case, "2d", self._live_fingerprint("2d")) if case else (None, "")
            if loaded is not None and loaded.dim == "2d" and (loaded.points or loaded.notes):
                plans.append(loaded)
            else:
                if mismatch:
                    notes.append(mismatch)
                generated = plan_2d(
                    mass_kg=mass,
                    domain_radius_m=float(snap.domain_radius_2d),
                    domain_height_m=height,
                    hob_m=hob,
                    cell_size=snap.domain_cell_2d,
                    usable_width_px=width,
                    logical_dpi_x=dpi,
                    remap_receive_r_max=receive,
                )
                plans.append(
                    stamp_plan(
                        generated,
                        case_path=case,
                        cell_size=snap.domain_cell_2d,
                        hob_m=hob,
                        domain_height_m=height,
                        remap_receive_r_max=receive,
                    )
                )
        self._sampling_notes = notes
        self._auto_plans = list(plans)
        return list(self._auto_plans)

    def _maybe_apply_auto_config_defaults(self, plans: Sequence[SamplingPlan]) -> None:
        """Match Configuration/Reference to the sampling-master burst once per current run.

        Elevated 2D and 1D sample the free-air spherical UFC table. Surface-burst
        2D samples Figure 2-15. Kingery-Bulmash spherical remains N/A, so UFC is
        selected when the automatic line is spherical. The user may still switch
        the overlay afterwards; that does not regenerate gauges.
        """
        if not self._auto_sampling() or not plans:
            return
        if self._auto_defaults_key == self._auto_key and self._auto_key is not None:
            return
        self._auto_defaults_key = self._auto_key
        spherical = any(
            p.dim == "1d" or p.burst_master == ufc_ab.BURST_SPHERICAL for p in plans
        )
        hemi = any(p.burst_master == ufc_ab.BURST_HEMISPHERICAL for p in plans)
        if spherical and hemi:
            return
        self.radio_kb_sph.blockSignals(True)
        self.radio_kb_hemi.blockSignals(True)
        self.combo_kb_source.blockSignals(True)
        try:
            if spherical:
                self.radio_kb_sph.setChecked(True)
                self.combo_kb_source.setCurrentText("UFC 3-340-02")
            else:
                self.radio_kb_hemi.setChecked(True)
        finally:
            self.radio_kb_sph.blockSignals(False)
            self.radio_kb_hemi.blockSignals(False)
            self.combo_kb_source.blockSignals(False)

    def _case_end_state(self, case: Optional[str], last_time: Optional[float]) -> Tuple[Optional[float], Optional[bool]]:
        if not case:
            return None, None
        ctrl = os.path.join(case, "system", "controlDict")
        try:
            stamp = os.path.getmtime(ctrl) if os.path.isfile(ctrl) else None
        except OSError:
            stamp = None
        cached = self._case_end_cache.get((case, stamp))
        if cached is not None:
            end_time, reached = cached
        else:
            end_time = None
            reached = None
            try:
                from result_storage import control_dict_root_end_time, run_reached_configured_end

                end_time = control_dict_root_end_time(case)
                reached = run_reached_configured_end(case)
            except Exception:
                end_time = None
                reached = None
            self._case_end_cache[(case, stamp)] = (end_time, reached)
        if last_time is not None and end_time is not None:
            from validation.history_quality import run_reached_end_time

            reached = run_reached_end_time(last_time_s=last_time, end_time_s=end_time, reached_end=reached)
        return end_time, reached

    def _assess_probe_series(
        self,
        times: Sequence[float],
        pvals: Sequence[float],
        impulse_vals: Optional[Sequence[float]],
        case: Optional[str],
    ) -> Tuple[Optional[float], Optional[float], str, bool]:
        last_t = times[-1] if times else None
        end_time, reached = self._case_end_state(case, last_t)
        validity = assess_history(
            times,
            pvals,
            impulse_vals,
            p_atm=self._snapshot.p_atm,
            end_time_s=end_time,
            reached_end=reached,
        )
        peak, impulse, reason = comparable_peak_impulse(validity)
        return peak, impulse, reason or validity.reason, validity.comparable

    def _bf_auto_peak_impulse(self, point: ValidationPoint) -> Tuple[Optional[float], Optional[float], str, bool]:
        case = case_dir_for_dim(self._snapshot, point.dim)
        fo = EXISTING_1D_GRAPH_FO if point.dim == "1d" else VALIDATION_FO.get(point.dim, "")
        p_path = self._latest_probe_path(case, fo, "p") if case and fo else ""
        try:
            stamp = os.path.getmtime(p_path) if p_path and os.path.isfile(p_path) else None
        except OSError:
            stamp = None
        key = (self._auto_key, point.dim, getattr(point, "point_id", ""), float(point.range_m or 0.0), p_path, stamp)
        hit = self._probe_cache.get(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        result = self._bf_auto_peak_impulse_compute(point)
        self._probe_cache[key] = result
        return result

    def _bf_auto_peak_impulse_compute(self, point: ValidationPoint) -> Tuple[Optional[float], Optional[float], str, bool]:
        case = case_dir_for_dim(self._snapshot, point.dim)
        if not case:
            return None, None, PLANNED_NOT_RUN, False
        if point.dim == "1d":
            fo = EXISTING_1D_GRAPH_FO
            p_path = self._latest_probe_path(case, fo, "p")
            i_path = self._latest_probe_path(case, fo, "impulse")
            if not p_path:
                return None, None, LEGACY_NO_VALIDATION_HISTORIES, False
            locs, times, cols = self._parse_probe_cached(p_path)
            radii = radii_from_locations(locs, dim="1d")
            mapping = map_radius(radii, point.range_m)
            if not mapping.ok:
                return None, None, mapping.reason or LEGACY_NO_VALIDATION_HISTORIES, False
            impulse_cols = None
            if i_path:
                _li, _itimes, impulse_cols = self._parse_probe_cached(i_path)
            peak_raw, impl_raw, t_use, p_use = mapped_peak_impulse(
                mapping,
                times,
                cols,
                impulse_cols,
                p_atm=self._snapshot.p_atm,
            )
            impulse_series = None
            if impulse_cols is not None and mapping.index_lo is not None:
                _it0, i_lo = series_for_index(times, impulse_cols, mapping.index_lo)
                if mapping.kind == KIND_EXACT:
                    impulse_series = i_lo
                elif mapping.index_hi is not None and mapping.weight is not None:
                    _it1, i_hi = series_for_index(times, impulse_cols, int(mapping.index_hi))
                    n = min(len(i_lo), len(i_hi), len(t_use))
                    w = float(mapping.weight)
                    impulse_series = [
                        float(i_lo[i]) + w * (float(i_hi[i]) - float(i_lo[i])) for i in range(n)
                    ]
            return self._assess_probe_series(t_use, p_use, impulse_series, case)
        fo = VALIDATION_FO.get(point.dim, "")
        p_path = self._latest_probe_path(case, fo, "p") if fo else ""
        i_path = self._latest_probe_path(case, fo, "impulse") if fo else ""
        if not p_path:
            return None, None, LEGACY_NO_VALIDATION_HISTORIES, False
        locs, times, cols = self._parse_probe_cached(p_path)
        idx, match_reason = match_probe_to_point(locs, (point.x, point.y, point.z))
        if idx is None:
            return None, None, match_reason, False
        _t, pvals = series_for_index(times, cols, idx)
        impulse_vals = None
        if i_path:
            _li, itimes, icols = self._parse_probe_cached(i_path)
            _it, ivals = series_for_index(itimes, icols, idx)
            impulse_vals = ivals or None
        return self._assess_probe_series(_t, pvals, impulse_vals, case)

    def _prefill_conwep_standoff(self) -> None:
        if self._conwep_key is None:
            return
        row = self._row_by_key(self._conwep_key)
        if row is None:
            return
        dist = standoff_m((row.x, row.y, row.z), self._snapshot.charge_center)
        if dist > 0.0:
            self.spin_cw_standoff.blockSignals(True)
            self.spin_cw_standoff.setValue(dist)
            self.spin_cw_standoff.blockSignals(False)

    def _refresh_banner(self) -> None:
        if self._snapshot.source == SOURCE_MANUAL:
            self.lbl_source.setText("Manual Result")
        else:
            self.lbl_source.setText("Current Run")

    def _use_current(self) -> None:
        self._snapshot = RunSnapshot(source=SOURCE_CURRENT)
        if self._context_fn:
            try:
                self._snapshot = self._context_fn()
            except Exception:
                pass
        self._auto_plans = []
        self._auto_key = self._snapshot_cache_key()
        self._auto_defaults_key = None
        self._display_sync_key = None
        self._clear_validation_caches()
        self._refresh_banner()
        self._prefill_from_snapshot()
        self.refresh_catalog()
        self._schedule_redraw()

    def _browse_manual(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select OpenFOAM case")
        if not path:
            return
        self._snapshot = RunSnapshot(
            source=SOURCE_MANUAL,
            case_2d=path,
            case_3d=path,
            case_1d=path,
            last_run_1d=path,
            last_run_2d=path,
            last_run_3d=path,
            mass_kg=self.spin_kb_mass.value(),
            material_name=self.edit_cw_type.text(),
            p_atm=self._snapshot.p_atm,
        )
        self._auto_plans = []
        self._auto_key = self._snapshot_cache_key()
        self._auto_defaults_key = None
        self._display_sync_key = None
        self._clear_validation_caches()
        self._refresh_banner()
        self._reload_hob_times()
        self._schedule_redraw()

    def _clear_validation_caches(self) -> None:
        self._probe_cache = {}
        self._probe_file_cache = {}
        self._probe_path_cache = {}
        self._case_end_cache = {}
        self._remap_cache = {}
        self._numerical_cache = {}

    def _latest_probe_path(self, case: str, fo: str, field: str) -> str:
        key = (case, fo, field)
        hit = self._probe_path_cache.get(key)
        if hit is not None:
            return hit
        path = latest_probe_field_file(case, fo, field) if case and fo else ""
        self._probe_path_cache[key] = path
        return path

    def _parse_probe_cached(self, path: str):
        if not path:
            return [], [], []
        try:
            stamp = os.path.getmtime(path) if os.path.isfile(path) else None
        except OSError:
            stamp = None
        key = (path, stamp)
        hit = self._probe_file_cache.get(key)
        if hit is not None:
            return hit
        parsed = parse_probe_history(path)
        self._probe_file_cache[key] = parsed
        return parsed

    def _schedule_redraw(self) -> None:
        self._redraw_timer.start()

    def _set_status(self, text: str, level: str = "info") -> None:
        self.lbl_status.setText(text)
        if not text:
            self.lbl_status.setStyleSheet(INFO_STATUS_STYLE)
            return
        if level == "error":
            self.lbl_status.setStyleSheet(ERROR_STATUS_STYLE)
        elif level == "warning":
            self.lbl_status.setStyleSheet(WARNING_STYLE)
        else:
            self.lbl_status.setStyleSheet(INFO_STATUS_STYLE)

    def _redraw(self) -> None:
        mode = self._mode()
        self._set_status("")
        if mode == MODE_KB:
            self._draw_kb()
        elif mode == MODE_CONWEP:
            self._draw_conwep()
        elif mode == MODE_HOB:
            self._draw_hob()
        elif mode == MODE_REMAP:
            self._draw_remap()
        else:
            self._draw_numerical()

    def _bf_peak_impulse(self, row: GaugeRow) -> Tuple[Optional[float], Optional[float], str, bool]:
        case = case_dir_for_dim(self._snapshot, row.dim)
        if not case:
            return None, None, "", False
        fo = PROBE_FO[row.dim]
        p_path = self._latest_probe_path(case, fo, "p")
        i_path = self._latest_probe_path(case, fo, "impulse")
        if not p_path:
            return None, None, "", False
        locs, times, cols = self._parse_probe_cached(p_path)
        idx = row.index
        if row.dim in ("2d", "3d"):
            matched, reason = match_probe_to_point(locs, (row.x, row.y, row.z))
            if matched is None:
                return None, None, reason, False
            idx = matched
        _t, pvals = series_for_index(times, cols, idx)
        impulse_vals = None
        if i_path:
            _li, itimes, icols = self._parse_probe_cached(i_path)
            _it, ivals = series_for_index(itimes, icols, idx)
            impulse_vals = ivals or None
        return self._assess_probe_series(_t, pvals, impulse_vals, case)

    def _gui_reference_identity(self) -> Tuple[str, str, str]:
        burst = self._kb_burst()
        source = kb_overlay.SOURCE_UFC if self._kb_use_ufc() else kb_overlay.SOURCE_SWISDAK
        figure = ufc_ab.figure_id(burst) if self._kb_use_ufc() else ""
        return source, burst, figure

    def _draw_kb(self) -> None:
        ax = self.plot_canvas.axes
        ax.clear()
        if self._auto_sampling():
            self._maybe_apply_auto_config_defaults(self._collect_auto_plans())
        gui_source, gui_burst, gui_figure = self._gui_reference_identity()
        pressure_mode = self.combo_kb_qty.currentIndex() == 0
        vs_z = self.chk_kb_z.isChecked()
        overlay_samples: List[kb_overlay.OverlaySample] = []
        hist_note = ""
        display = self._display_dims()
        receive_2d = self._remap_receive_2d()
        receive_3d = self._remap_receive_3d()
        dx_2d = self._snapshot.domain_cell_2d
        mass_ref = float(self.spin_kb_mass.value())
        if self._auto_sampling():
            for plan in self._collect_auto_plans():
                if plan.dim not in display:
                    continue
                if plan.notes and not plan.points and not hist_note:
                    hist_note = plan.notes[-1]
                for point in plan.points:
                    rng = physical_standoff_m(
                        plan.dim,
                        (point.x, point.y, point.z),
                        plan.charge_center,
                    )
                    if plan.dim == "2d" and not kb_propagation_eligible(rng, receive_2d, dx_2d):
                        continue
                    bf_p, bf_i, reason, comparable = self._bf_auto_peak_impulse(point)
                    has_bf = histories_available(self._snapshot, plan.dim)
                    probe_ok = "match" not in (reason or "").lower() and "missing" not in (reason or "").lower()
                    if reason and ("match" in reason.lower() or "missing" in reason.lower()):
                        probe_ok = False
                    if comparable:
                        kind = "bf"
                    elif reason and not probe_ok:
                        kind = "invalid"
                    elif has_bf or case_dir_for_dim(self._snapshot, plan.dim):
                        kind = "invalid" if reason else "missing"
                    else:
                        kind = "planned"
                    if reason and not hist_note:
                        hist_note = reason
                    sample = kb_overlay.sample_from_plan_point(
                        plan,
                        point,
                        bf_peak=bf_p,
                        bf_impulse=bf_i,
                        comparable=comparable,
                        validity_reason=reason,
                        kind=kind,
                        probe_ok=probe_ok,
                    )
                    overlay_samples.append(
                        dc_replace(
                            sample,
                            range_m=rng,
                            scaled_z=scaled_z_from_r(rng, mass_ref),
                            mass_kg=mass_ref,
                        )
                    )
            if "3d" in display:
                kept_3d = 0
                skipped_remap = 0
                for row in self._catalog():
                    if row.dim != "3d":
                        continue
                    rng = physical_standoff_m(
                        "3d",
                        (row.x, row.y, row.z),
                        charge_center_for_dim(self._snapshot, "3d"),
                    )
                    if not kb_propagation_eligible(rng, receive_3d, None):
                        skipped_remap += 1
                        continue
                    kept_3d += 1
                    bf_p, bf_i, reason, comparable = self._bf_peak_impulse(row)
                    has_bf = histories_available(self._snapshot, "3d")
                    kind = "bf" if comparable else ("planned" if not has_bf else "invalid")
                    overlay_samples.append(
                        kb_overlay.OverlaySample(
                            point_id=row.label or row.gauge_id,
                            dim="3d",
                            mass_kg=mass_ref,
                            burst=gui_burst,
                            figure=gui_figure,
                            reference_source=gui_source,
                            range_m=rng,
                            scaled_z=scaled_z_from_r(rng, mass_ref),
                            bf_peak=bf_p,
                            bf_impulse=bf_i,
                            comparable=comparable,
                            validity_reason=reason,
                            kind=kind,
                            probe_ok=not reason or "match" not in reason.lower(),
                        )
                    )
                if kept_3d == 0 and skipped_remap:
                    hist_note = hist_note or (
                        "Validation cannot be generated reliably: not enough physical domain "
                        "outside the remap receiving region."
                    )
        else:
            for key in self._added:
                row = self._row_by_key(key)
                if row is None or row.dim not in display:
                    continue
                rng = physical_standoff_m(
                    row.dim,
                    (row.x, row.y, row.z),
                    charge_center_for_dim(self._snapshot, row.dim),
                )
                if row.dim == "2d" and not kb_propagation_eligible(rng, receive_2d, dx_2d):
                    continue
                if row.dim == "3d" and not kb_propagation_eligible(rng, receive_3d, None):
                    continue
                bf_p, bf_i, reason, comparable = self._bf_peak_impulse(row)
                overlay_samples.append(
                    kb_overlay.OverlaySample(
                        point_id=row.label or row.gauge_id,
                        dim=row.dim,
                        mass_kg=mass_ref,
                        burst=gui_burst,
                        figure=gui_figure,
                        reference_source=gui_source,
                        range_m=rng,
                        scaled_z=scaled_z_from_r(rng, mass_ref),
                        bf_peak=bf_p,
                        bf_impulse=bf_i,
                        comparable=comparable,
                        validity_reason=reason,
                        kind="user",
                        probe_ok=not reason or "match" not in reason.lower(),
                    )
                )
        groups = kb_overlay.group_samples(overlay_samples)
        ref_colors = ("#444444", "#666666", "#888888", "#555555")
        use_ufc_fallback = self._kb_use_ufc()
        engine = ufc_ab if use_ufc_fallback else kb
        qty = engine.QUANTITY_PEAK_PRESSURE if pressure_mode else engine.QUANTITY_INCIDENT_IMPULSE
        xr = []
        if groups:
            for g_i, group in enumerate(groups):
                g_engine = kb_overlay.engine_for(group.source)
                g_qty = g_engine.QUANTITY_PEAK_PRESSURE if pressure_mode else g_engine.QUANTITY_INCIDENT_IMPULSE
                xs, ys = kb_overlay.curve_for_group(group, quantity=g_qty, vs_z=vs_z)
                if xs:
                    xr = xs
                    color = ref_colors[g_i % len(ref_colors)]
                    if pressure_mode:
                        ax.plot(xs, [pa_to_kpa(v) for v in ys], color=color, label=group.label)
                    else:
                        ax.plot(xs, [pa_s_to_kpa_ms(v) for v in ys], color=color, label=group.label)
                elif not kb_overlay.is_ufc_source(group.source) and group.burst == kb.BURST_SPHERICAL:
                    self._set_status(kb.SPHERICAL_UNAVAILABLE)
        else:
            mass = float(self.spin_kb_mass.value())
            ref_label = kb_overlay.reference_label(gui_source, gui_burst, gui_figure)
            xs, ys = (engine.curve_vs_z if vs_z else engine.curve)(qty, mass_kg=mass, burst_type=gui_burst)
            if xs:
                xr = xs
                if pressure_mode:
                    ax.plot(xs, [pa_to_kpa(v) for v in ys], color="#444444", label=ref_label)
                else:
                    ax.plot(xs, [pa_s_to_kpa_ms(v) for v in ys], color="#444444", label=ref_label)
            elif not use_ufc_fallback and gui_burst == kb.BURST_SPHERICAL:
                self._set_status(kb.SPHERICAL_UNAVAILABLE)
        col_p = "UFC Peak Pressure" if (groups and all(kb_overlay.is_ufc_source(g.source) for g in groups)) or (not groups and use_ufc_fallback) else "KB Peak Pressure"
        col_i = "UFC Positive Impulse" if col_p.startswith("UFC") else "KB Positive Impulse"
        self._init_table(
            [
                "Gauge",
                "Dimension",
                "Source",
                "Range",
                "BF Peak Pressure",
                col_p,
                "Error %",
                "BF Positive Impulse",
                col_i,
                "Error %",
            ]
        )
        series = {}
        hemi_3d_note = False
        evals = [kb_overlay.evaluate_sample(sample) for sample in overlay_samples]
        samples_for_info = []
        for ev in evals:
            sample = ev.sample
            if sample.dim == "3d" and not ev.applicable:
                hemi_3d_note = True
            rng = sample.range_m
            mass_s = sample.mass_kg
            xval = sample.scaled_z if vs_z else rng
            if vs_z and sample.scaled_z is None and is_finite_number(mass_s) and mass_s > 0:
                xval = rng / (mass_s ** (1.0 / 3.0))
            show_bf_p = sample.bf_peak if sample.comparable else None
            show_bf_i = sample.bf_impulse if sample.comparable else None
            y_bf = pa_to_kpa(show_bf_p) if pressure_mode and show_bf_p is not None else (
                pa_s_to_kpa_ms(show_bf_i) if (not pressure_mode and show_bf_i is not None) else None
            )
            y_ref = None
            if pressure_mode and ev.ref_peak is not None:
                y_ref = pa_to_kpa(ev.ref_peak)
            elif (not pressure_mode) and ev.ref_impulse is not None:
                y_ref = pa_s_to_kpa_ms(ev.ref_impulse)
            kind = sample.kind
            source_label = {"bf": "BF", "planned": "Planned", "missing": "N/A", "user": "User", "invalid": "N/A"}.get(kind, kind)
            samples_for_info.append((sample.point_id, sample.dim, rng, show_bf_p, show_bf_i, kind))
            if kind == "bf" and is_finite_number(xval) and y_bf is not None:
                key = self._series_key(sample.dim, sample.point_id, samples_for_info)
                series.setdefault(key, ([], []))
                series[key][0].append(xval)
                series[key][1].append(y_bf)
            elif kind == "planned" and is_finite_number(xval) and y_ref is not None:
                series.setdefault(PLANNED_NOT_RUN, ([], []))
                series[PLANNED_NOT_RUN][0].append(xval)
                series[PLANNED_NOT_RUN][1].append(y_ref)
            elif kind == "user" and is_finite_number(xval) and y_bf is not None:
                series.setdefault("User Gauges", ([], []))
                series["User Gauges"][0].append(xval)
                series["User Gauges"][1].append(y_bf)
            self._append_table_row(
                [
                    sample.point_id,
                    sample.dim.upper(),
                    source_label,
                    fmt(rng, suffix="m"),
                    fmt(None if show_bf_p is None else pa_to_kpa(show_bf_p), suffix="kPa"),
                    fmt(None if ev.ref_peak is None else pa_to_kpa(ev.ref_peak), suffix="kPa"),
                    fmt(ev.error_peak_pct),
                    fmt(None if show_bf_i is None else pa_s_to_kpa_ms(show_bf_i), suffix="kPa·ms"),
                    fmt(None if ev.ref_impulse is None else pa_s_to_kpa_ms(ev.ref_impulse), suffix="kPa·ms"),
                    fmt(ev.error_impulse_pct),
                ]
            )
        colors = {
            "BF 1D": "#1f77b4",
            "BF 2D": "#d62728",
            "BF 3D": "#2ca02c",
            PLANNED_NOT_RUN: "#7f7f7f",
            "User Gauges": "#9467bd",
        }
        for label, (xs_mark, ys_mark) in series.items():
            if not xs_mark:
                continue
            stride = marker_stride(len(xs_mark), self._plot_width_px(), self._logical_dpi())
            face = "none" if label == PLANNED_NOT_RUN else colors.get(label, "#1f77b4")
            ax.scatter(
                xs_mark[::stride],
                ys_mark[::stride],
                s=18,
                zorder=3,
                facecolors=face,
                edgecolors=colors.get(label, "#1f77b4"),
                label=label,
            )
        self._apply_kb_axes(ax, vs_z=vs_z, log_scale=self.radio_kb_log.isChecked(), pressure_mode=pressure_mode)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")
        self.plot_canvas.draw_idle()
        ref_label = ", ".join(dict.fromkeys(g.label for g in groups)) if groups else kb_overlay.reference_label(gui_source, gui_burst, gui_figure)
        self._update_kb_info(ref_label, samples_for_info)
        mismatch_note = next((n for n in self._sampling_notes if n), "")
        if hemi_3d_note and not self.lbl_status.text():
            self._set_status(THREE_D_HEMI_NA)
        elif mismatch_note and not self.lbl_status.text():
            self._set_status(mismatch_note)
        elif hist_note and self._auto_sampling() and not self.lbl_status.text():
            self._set_status(hist_note)
        elif kb_overlay.mixed_references(groups) and not self.lbl_status.text():
            self._set_status("Multiple references shown; each series uses its own mass, burst, and figure.")
        elif not overlay_samples and not xr and not self.lbl_status.text():
            if not primary_case_dir(self._snapshot):
                self._set_status(MISSING_CURRENT_RUN)

    def _series_key(self, dim: str, name: str, samples: list) -> str:
        return series_label(dim)

    def _reference_applicable(self, dim: str, burst: str) -> bool:
        if dim != "3d":
            return True
        if burst in (kb.BURST_HEMISPHERICAL, ufc_ab.BURST_HEMISPHERICAL):
            return False
        return True

    def _apply_kb_axes(self, ax, *, vs_z: bool, log_scale: bool, pressure_mode: bool) -> None:
        ax.set_xlabel("Scaled Distance Z" if vs_z else "Range [m]")
        ax.set_ylabel("Peak Pressure [kPa]" if pressure_mode else "Positive Impulse [kPa·ms]")
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.set_xscale("linear")
            ax.set_yscale("linear")

    def _update_kb_info(self, ref_label: str, samples: list) -> None:
        charge = self._snapshot.material_name or "—"
        mass = float(self.spin_kb_mass.value())
        if self._auto_sampling():
            plans = self._collect_auto_plans()
            rmin = min((p.r_min for p in plans), default=None)
            rmax = max((p.r_max for p in plans), default=None)
            n_auto = sum(len(p.points) for p in plans)
            n_3d = sum(1 for item in samples if len(item) > 1 and item[1] == "3d")
            npts = n_auto + n_3d
            range_txt = "N/A" if rmin is None or rmax is None else f"{rmin:.3g} – {rmax:.3g} m"
            n_bf = sum(1 for item in samples if item[-1] == "bf")
            n_plan = sum(1 for item in samples if item[-1] == "planned")
            self.lbl_kb_info.setText(
                f"Charge: {charge}\n"
                f"Mass: {mass:.4g} kg\n"
                f"Validation range: {range_txt}\n"
                f"Reference: {ref_label}\n"
                f"Automatic points: {npts}"
                + (f"\nBF results: {n_bf}" if n_bf else "")
                + (f"\n{PLANNED_NOT_RUN}" if n_plan and not n_bf else "")
                + "".join(
                    f"\n{p.dim.upper()}: W={p.mass_kg:.4g} kg, {p.burst_master}, Fig {p.figure}"
                    for p in plans
                    if p.points or p.notes
                )
            )
        else:
            self.lbl_kb_info.setText(
                f"Charge: {charge}\nMass: {mass:.4g} kg\nUser gauges: {len(samples)}"
            )

    def _kb_legend(self, row: GaugeRow, peak: Optional[float], impulse: Optional[float]) -> str:
        ptxt = "N/A" if peak is None else f"{pa_to_kpa(peak):.3g} kPa"
        itxt = "N/A" if impulse is None else f"{pa_s_to_kpa_ms(impulse):.3g} kPa·ms"
        return f"{row.dim.upper()} - {row.label or row.gauge_id} | Pmax = {ptxt} | I = {itxt}"

    def _draw_conwep(self) -> None:
        ax = self.plot_canvas.axes
        ax.clear()
        key = self._conwep_key or (self._added[-1] if self._added else None)
        auto_point = None
        if hasattr(self, "lbl_cw_standoff_src"):
            self.lbl_cw_standoff_src.setText("")
        if self._auto_sampling() and key is None:
            for plan in self._collect_auto_plans():
                if plan.points:
                    auto_point = plan.points[0]
                    break
            if auto_point is not None:
                self.spin_cw_standoff.blockSignals(True)
                self.spin_cw_standoff.setValue(float(auto_point.range_m))
                self.spin_cw_standoff.blockSignals(False)
                self.lbl_cw_standoff_src.setText(
                    f"Standoff = automatic point {auto_point.point_id} (R_min = {auto_point.range_m:.4g} m)"
                )
        if key is None and auto_point is None:
            self._set_status("Select a gauge and click Add.")
            self._init_table(["Metric", "BF", "CONWEP", "Difference", "Error %"])
            self.plot_canvas.draw_idle()
            return
        row = self._row_by_key(key) if key is not None else None
        case = case_dir_for_dim(self._snapshot, row.dim) if row is not None else case_dir_for_dim(
            self._snapshot, auto_point.dim if auto_point else "2d"
        )
        fo = PROBE_FO[row.dim] if row is not None else VALIDATION_FO.get(auto_point.dim if auto_point else "2d", "")
        if auto_point is not None and auto_point.dim == "1d":
            fo = EXISTING_1D_GRAPH_FO
        p_path = self._latest_probe_path(case, fo, "p") if case else ""
        i_path = self._latest_probe_path(case, fo, "impulse") if case else ""
        times, pvals, ivals = [], [], []
        series_index = row.index if row is not None else (auto_point.index if auto_point else 0)
        if auto_point is not None and auto_point.dim == "1d" and p_path:
            locs, times0, cols = self._parse_probe_cached(p_path)
            radii = radii_from_locations(locs, dim="1d")
            mapping = map_radius(radii, auto_point.range_m)
            if mapping.ok:
                peak, impl, t_use, p_abs = mapped_peak_impulse(
                    mapping,
                    times0,
                    cols,
                    None,
                    p_atm=self._snapshot.p_atm,
                )
                times = t_use
                pvals = [p - self._snapshot.p_atm for p in p_abs]
                if i_path:
                    _l, it, ic = self._parse_probe_cached(i_path)
                    _peak, _impl, _tt, _ = mapped_peak_impulse(
                        mapping, it or times0, ic, ic, p_atm=self._snapshot.p_atm
                    )
                    if mapping.kind == KIND_EXACT and mapping.index_lo is not None:
                        _it, ivals = series_for_index(it, ic, mapping.index_lo)
                    elif mapping.index_lo is not None and mapping.index_hi is not None:
                        _it0, i_lo = series_for_index(it, ic, mapping.index_lo)
                        _it1, i_hi = series_for_index(it, ic, mapping.index_hi)
                        n = min(len(i_lo), len(i_hi))
                        w = float(mapping.weight or 0.0)
                        ivals = [i_lo[i] + w * (i_hi[i] - i_lo[i]) for i in range(n)]
        elif p_path:
            _l, times, cols = self._parse_probe_cached(p_path)
            times, pvals = series_for_index(times, cols, series_index)
            pvals = [p - self._snapshot.p_atm for p in pvals]
            if i_path:
                _l, it, ic = self._parse_probe_cached(i_path)
                _it, ivals = series_for_index(it, ic, series_index)
        if times and pvals:
            ax.plot([s_to_ms(t) for t in times], [pa_to_kpa(p) for p in pvals], "-", label="BF Pressure")
        right = None
        if ivals:
            right = ax.twinx()
            right.plot([s_to_ms(t) for t in times[: len(ivals)]], [pa_s_to_kpa_ms(v) for v in ivals], "-", color="#d62728", label="BF Impulse")
        ptype = conwep_engine.PRESSURE_INCIDENT if self.combo_cw_ptype.currentIndex() == 0 else conwep_engine.PRESSURE_REFLECTED
        result = conwep_engine.evaluate(
            range_m=float(self.spin_cw_standoff.value()),
            mass_kg=float(self.spin_cw_mass.value()),
            pressure_type=ptype,
            explosive_type=self.edit_cw_type.text(),
        )
        if result.pressure_history is None:
            wave = ufc_waveform.evaluate(
                range_m=float(self.spin_cw_standoff.value()),
                mass_kg=float(self.spin_cw_mass.value()),
                burst_type=ufc_ab.BURST_HEMISPHERICAL,
                family=(
                    ufc_waveform.FAMILY_INCIDENT
                    if ptype == conwep_engine.PRESSURE_INCIDENT
                    else ufc_waveform.FAMILY_REFLECTED
                ),
            )
            if wave.ok:
                ax.plot(
                    [s_to_ms(t) for t in wave.times_s],
                    [pa_to_kpa(p) for p in wave.overpressure_pa],
                    "--",
                    color="#444444",
                    label="UFC Calc P(t) (not CONWEP)",
                )
                self._set_status(
                    "CONWEP waveform: N/A. Overlay is UFC Calc modified Friedlander "
                    "from Figures 2-15 / DataHemiSpherical; decay b is derived, not CONWEP."
                )
            else:
                self._set_status(conwep_engine.WAVEFORM_UNAVAILABLE)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Pressure [kPa]")
        if right is not None:
            right.set_ylabel("Impulse [kPa·ms]")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")
        self.plot_canvas.draw_idle()
        bf_p = max(pvals) if pvals else None
        bf_i = ivals[-1] if ivals else None
        self._init_table(["Metric", "BF", "CONWEP", "Difference", "Error %"])
        rows = [
            ("Peak Pressure", bf_p, result.peak_pressure.value_si, "kPa", pa_to_kpa),
            ("Positive Impulse", bf_i, result.positive_impulse.value_si, "kPa·ms", pa_s_to_kpa_ms),
            ("Arrival Time", None, result.arrival_time.value_si, "ms", s_to_ms),
            ("Positive Phase Duration", None, result.positive_duration.value_si, "ms", s_to_ms),
        ]
        for name, bf, ref, unit, conv in rows:
            bf_d = None if bf is None else conv(bf)
            ref_d = None if ref is None else conv(ref)
            diff = None if bf is None or ref is None else bf_d - ref_d
            self._append_table_row(
                [
                    name,
                    fmt(bf_d, suffix=unit),
                    fmt(ref_d, suffix=unit),
                    fmt(diff, suffix=unit),
                    fmt(relative_error_percent(bf, ref)),
                ]
            )

    def _hob_case(self) -> Optional[str]:
        if self.radio_hob_2d.isChecked():
            return case_dir_for_dim(self._snapshot, "2d")
        return case_dir_for_dim(self._snapshot, "3d")

    def _on_hob_source(self, *_args) -> None:
        self.combo_hob_plane.setEnabled(self.radio_hob_3d.isChecked())
        self._reload_hob_times()
        self._schedule_redraw()

    def _hob_load_kwargs(self) -> dict:
        if self.radio_hob_3d.isChecked():
            return {
                "plane": self.combo_hob_plane.currentText(),
                "origin": self._snapshot.charge_center,
            }
        return {"plane": "axisymmetric", "origin": (0.0, 0.0, 0.0)}

    def _hob_field_name(self) -> str:
        return {
            "Pressure": "p",
            "Peak Overpressure": "overpressure",
            "Density": "rho",
            "Velocity": "U",
            "Temperature": "T",
        }.get(self.combo_hob_field.currentText(), "p")

    def _reload_hob_times(self) -> None:
        case = self._hob_case()
        self.combo_hob_time.blockSignals(True)
        self.slider_hob.blockSignals(True)
        self.combo_hob_time.clear()
        times = list_saved_times(case or "") if case else []
        for tval, label in times:
            self.combo_hob_time.addItem(f"{s_to_ms(tval):.4g} ms", (tval, label))
        self.slider_hob.setRange(0, max(len(times) - 1, 0))
        self.combo_hob_time.blockSignals(False)
        self.slider_hob.blockSignals(False)
        if not times and self._mode() == MODE_HOB:
            self._set_status(MISSING_CURRENT_RUN)

    def _step_hob_time(self, delta: int) -> None:
        idx = self.combo_hob_time.currentIndex() + delta
        if 0 <= idx < self.combo_hob_time.count():
            self.combo_hob_time.setCurrentIndex(idx)

    def _on_hob_time(self, index: int) -> None:
        if index >= 0:
            self.slider_hob.blockSignals(True)
            self.slider_hob.setValue(index)
            self.slider_hob.blockSignals(False)
        self._schedule_redraw()

    def _on_hob_slider(self, value: int) -> None:
        if 0 <= value < self.combo_hob_time.count():
            self.combo_hob_time.blockSignals(True)
            self.combo_hob_time.setCurrentIndex(value)
            self.combo_hob_time.blockSignals(False)
            self._schedule_redraw()

    def _run_hob_extract(self) -> None:
        case = self._hob_case()
        if not case:
            self._set_status(MISSING_CURRENT_RUN)
            return
        times = []
        for i in range(self.combo_hob_time.count()):
            times.append(self.combo_hob_time.itemData(i))
        if not times:
            self._set_status(MISSING_CURRENT_RUN)
            return
        cache_key = f"{os.path.normpath(case)}|p|{tuple(times)}"
        cached = self._hob_cache.get(cache_key)
        if cached is not None:
            self._hob_cache["trajectory"] = cached
            self._hob_cache["key"] = cache_key
            self.lbl_hob_progress.setText("Using cached trajectory.")
            self._schedule_redraw()
            return
        if self._hob_worker is not None:
            self._hob_worker.cancel()
        z_g = 0.0
        kwargs = self._hob_load_kwargs()
        worker = HobExtractWorker(
            case, times, "p", z_g, plane=kwargs["plane"], origin=kwargs["origin"]
        )
        worker.progress.connect(lambda pct, msg: self.lbl_hob_progress.setText(f"{pct}% {msg}"))
        worker.finished_ok.connect(lambda samples, key=cache_key: self._hob_done(samples, key))
        worker.finished_error.connect(lambda m: self.lbl_hob_progress.setText(m))
        self._hob_worker = worker
        worker.start()

    def _hob_done(self, samples, key: str = "") -> None:
        if key:
            self._hob_cache[key] = samples
            self._hob_cache["key"] = key
        self._hob_cache["trajectory"] = samples
        self.lbl_hob_progress.setText("Detection finished.")
        self._schedule_redraw()

    def _hob_mass_kg(self) -> Optional[float]:
        mass = float(self.spin_kb_mass.value())
        if mass > 0.0:
            return mass
        snap = self._snapshot.mass_kg
        if is_finite_number(snap) and float(snap) > 0.0:
            return float(snap)
        return None

    def _hob_height_m(self) -> Optional[float]:
        hob = self._snapshot.hob_m
        if is_finite_number(hob) and float(hob) > 0.0:
            return float(hob)
        return None

    def _draw_hob(self) -> None:
        sax = self.spatial_canvas.axes
        sax.clear()
        ax = self.plot_canvas.axes
        ax.clear()
        case = self._hob_case()
        data = self.combo_hob_time.currentData()
        if not case or not data:
            self._set_status(MISSING_CURRENT_RUN)
            self._init_table(["Item", "Value"])
            self.spatial_canvas.draw_idle()
            self.plot_canvas.draw_idle()
            return
        tval, label = data
        field = self._hob_field_name()
        r, z, p, err = load_pressure_rz(case, label, field, **self._hob_load_kwargs())
        if err:
            self._set_status(err)
        elif r is not None:
            fronts = hob_engine.extract_fronts(r, z, p, z_ground=0.0)
            sax.pcolormesh(fronts.r, fronts.z, fronts.pressure, shading="auto")
            if np.isfinite(fronts.r_shock).any():
                sax.plot(fronts.r_shock, fronts.z_shock, "w-", label="Shock front")
            if fronts.triple_point:
                sax.plot([fronts.triple_point[0]], [fronts.triple_point[1]], "rx", label="Triple Point")
            sax.set_xlabel("r [m]")
            sax.set_ylabel("z [m]")
            sax.set_title(f"{field} @ {s_to_ms(tval):.4g} ms")
            sax.legend(loc="best")
        samples = self._hob_cache.get("trajectory") or []
        kind = self.combo_hob_kind.currentText()
        hob_m = self._hob_height_m()
        mass_kg = self._hob_mass_kg()
        ufc_xs, ufc_ys = ufc_hob.reference_curve(hob_m=hob_m, mass_kg=mass_kg)
        if kind.startswith("Triple") and samples:
            _t, xs, hs = hob_engine.trajectory(samples)
            ax.plot(xs, hs, "o-", label="BF extracted trajectory")
            if ufc_xs:
                ax.plot(ufc_xs, ufc_ys, "--", label="UFC 3-340-02 Figure 2-13")
            else:
                ev0 = ufc_hob.lookup_mach_stem_height(0.0, hob_m=hob_m, mass_kg=mass_kg)
                self._set_status(ev0.unavailable_reason or ufc_hob.REQUIRED_CHART)
            ax.set_xlabel("Ground Range [m]")
            ax.set_ylabel("Triple-Point Height [m]")
            ax.legend(loc="best")
        elif kind.startswith("Shock") and samples:
            times = [s.time_s for s in samples if s.x_tp is not None]
            xs = [s.x_tp for s in samples if s.x_tp is not None]
            ax.plot([s_to_ms(t) for t in times], xs, "o-", label="BF front position")
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Shock Front Position [m]")
        elif kind.startswith("Incident"):
            if mass_kg:
                xr, yr = ufc_ab.curve(
                    ufc_ab.QUANTITY_PEAK_PRESSURE,
                    mass_kg=mass_kg,
                    burst_type=ufc_ab.BURST_SPHERICAL,
                )
                if xr:
                    ax.plot(xr, [pa_to_kpa(v) for v in yr], "--", label="UFC 3-340-02 Figure 2-7")
                    ax.set_xlabel("Range [m]")
                    ax.set_ylabel("Peak incident overpressure [kPa]")
                    ax.legend(loc="best")
                else:
                    self._set_status("UFC Figure 2-7: N/A for the current mass/range.")
            else:
                self._set_status("UFC Figure 2-7: N/A — charge mass is required.")
        elif "Fig 2-9" in kind or "Fig 2-10" in kind:
            figure = ufc_ground.FIGURE_PRESSURE if "2-9" in kind else ufc_ground.FIGURE_IMPULSE
            if hob_m and mass_kg:
                gx, gy = ufc_ground.reference_curve_vs_range(figure, hob_m=hob_m, mass_kg=mass_kg)
                if gx:
                    if figure == ufc_ground.FIGURE_PRESSURE:
                        ax.plot(gx, [pa_to_kpa(v) for v in gy], "--", label="UFC 3-340-02 Figure 2-9")
                        ax.set_ylabel("Reflected pressure [kPa]")
                    else:
                        ax.plot(gx, [pa_s_to_kpa_ms(v) for v in gy], "--", label="UFC 3-340-02 Figure 2-10")
                        ax.set_ylabel("Reflected impulse [kPa·ms]")
                    ax.set_xlabel("Ground Range [m]")
                    if r is not None and p is not None:
                        fronts = hob_engine.extract_fronts(r, z, p, z_ground=0.0)
                        j = int(np.argmin(np.abs(fronts.z[0, :] - fronts.z_ground)))
                        ax.plot(fronts.r[:, j], [pa_to_kpa(v - self._snapshot.p_atm) for v in fronts.pressure[:, j]],
                                "o", ms=3, label="BF at ground (current time)")
                    ax.legend(loc="best")
                else:
                    evg = ufc_ground.lookup(
                        figure, ground_range_m=1.0, hob_m=hob_m, mass_kg=mass_kg, observer_z_m=0.0
                    )
                    self._set_status(evg.unavailable_reason)
            else:
                self._set_status("UFC Figures 2-9/2-10: N/A — HOB and charge mass are required.")
        elif kind.startswith("UFC pressure-time"):
            samples_tp = hob_engine.trajectory(samples) if samples else ((), (), ())
            r_tp = samples_tp[1][-1] if samples_tp[1] else None
            if r_tp is None and r is not None and p is not None:
                fronts = hob_engine.extract_fronts(r, z, p, z_ground=0.0)
                if fronts.triple_point:
                    r_tp = fronts.triple_point[0]
            if r_tp is None or hob_m is None or mass_kg is None:
                self._set_status("UFC Calc P(t): N/A — need HOB, mass, and a ground range / triple-point range.")
            else:
                slant = float(math.hypot(float(r_tp), float(hob_m)))
                wave = ufc_waveform.evaluate(
                    range_m=slant,
                    mass_kg=mass_kg,
                    burst_type=ufc_ab.BURST_SPHERICAL,
                    family=ufc_waveform.FAMILY_INCIDENT,
                )
                if wave.ok:
                    ax.plot(
                        [s_to_ms(t) for t in wave.times_s],
                        [pa_to_kpa(p) for p in wave.overpressure_pa],
                        "--",
                        label="UFC Calc P(t) (not CONWEP)",
                    )
                    ax.set_xlabel("Time [ms]")
                    ax.set_ylabel("Incident overpressure [kPa]")
                    ax.legend(loc="best")
                else:
                    self._set_status(wave.unavailable_reason)
        self._init_table(["Item", "BF", "UFC", "Difference", "Error %"])
        tp = None
        fronts = None
        if r is not None and p is not None:
            fronts = hob_engine.extract_fronts(r, z, p, z_ground=0.0)
            tp = fronts.triple_point
            self._append_table_row(["Triple Point", "N/A" if tp is None else f"{tp[0]:.4g}, {tp[1]:.4g} m", "", "", ""])
            self._append_table_row(["Mach stem height", fmt(fronts.mach_stem_height, suffix="m"), "", "", ""])
            self._append_table_row(["Detection", fronts.reason or ("ok" if tp else "N/A"), "", "", ""])
        ufc_ev = ufc_hob.lookup_mach_stem_height(
            float(tp[0]) if tp else 0.0, hob_m=hob_m, mass_kg=mass_kg
        )
        bf_hm = None if tp is None else (tp[1] if fronts is None else fronts.mach_stem_height)
        self._append_table_row(
            [
                "UFC 3-340-02 Figure 2-13 HT",
                fmt(bf_hm, suffix="m"),
                fmt(ufc_ev.hm_m, suffix="m") if ufc_ev.hm_m is not None else (ufc_ev.unavailable_reason or "N/A"),
                fmt(None if bf_hm is None or ufc_ev.hm_m is None else bf_hm - ufc_ev.hm_m, suffix="m"),
                fmt(relative_error_percent(bf_hm, ufc_ev.hm_m)),
            ]
        )
        mach_region = tp is not None
        self.spin_rh_theta.setEnabled(not mach_region)
        shock = rh.normal_shock(
            shock_speed=float(self.spin_rh_us.value()),
            p1=float(self.spin_rh_p1.value()),
            rho1=float(self.spin_rh_rho.value()),
            gamma=float(self.spin_rh_gamma.value()),
        )
        self._append_table_row(["RH Mach", fmt(shock.mach) if shock.mach is not None else (shock.unavailable_reason or "N/A")])
        self._append_table_row(["RH p2", fmt(shock.p2, suffix="Pa")])
        self._append_table_row(["RH ρ2", fmt(shock.rho2, suffix="kg/m³")])
        self._append_table_row(["RH u2n", fmt(shock.u2_normal, suffix="m/s")])
        if mach_region:
            self._append_table_row(["Oblique shock", "N/A — Mach-reflection region (triple point detected)"])
        else:
            obl = rh.regular_oblique_shock(
                mach1=float(shock.mach or 0.0),
                theta_deg=float(self.spin_rh_theta.value()),
                gamma=float(self.spin_rh_gamma.value()),
            )
            self._append_table_row(["Oblique β", fmt(obl.beta_deg, suffix="deg") if obl.beta_deg is not None else (obl.unavailable_reason or "N/A")])
            self._append_table_row(["Oblique p2/p1", fmt(obl.p2_over_p1)])
        self.spatial_canvas.draw_idle()
        self.plot_canvas.draw_idle()

    def _remap_field_mtime(self, case_dir: Optional[str], time_label: Optional[str], field: str) -> Optional[float]:
        if not case_dir or not time_label:
            return None
        stored = remap_engine.resolve_stored_field_name(case_dir, time_label, field)
        path = os.path.join(case_dir, time_label, stored)
        try:
            return os.path.getmtime(path) if os.path.isfile(path) else None
        except OSError:
            return None

    def _remap_field_key(self) -> str:
        return {
            "Pressure": "p",
            "Density": "rho",
            "Radial Velocity": "U",
            "Temperature": "T",
            "alpha.c4": "alpha.c4",
        }.get(self.combo_remap_field.currentText(), "p")

    def _draw_remap(self) -> None:
        ax = self.plot_canvas.axes
        ax.clear()
        if self.combo_remap_mode.currentIndex() == 0:
            target = case_dir_for_dim(self._snapshot, "2d")
            src, st, tt, msg = remap_engine.resolve_1d_to_2d(
                target_case=target or "",
                mapping_source=self._snapshot.mapping_source_2d,
                mapping_time=self._snapshot.mapping_time_2d,
            )
        else:
            target = case_dir_for_dim(self._snapshot, "3d")
            src, st, tt, msg = remap_engine.resolve_2d_to_3d(
                target_case=target or "",
                remap_source_type=self._snapshot.remap_3d_source_type,
                prepare_3d_transfer=self._snapshot.prepare_3d_transfer,
            )
        if msg:
            self._set_status(msg)
            self.lbl_remap_times.setText("Source Time: N/A\nTarget Initialization Time: N/A\nDelta t: N/A")
            self._init_table(["Metric", "Value"])
            self.plot_canvas.draw_idle()
            return
        field = self._remap_field_key()
        src_dim = "1d" if self.combo_remap_mode.currentIndex() == 0 else "2d"
        tgt_dim = "2d" if self.combo_remap_mode.currentIndex() == 0 else "3d"
        receive = self._remap_receive_2d() if self.combo_remap_mode.currentIndex() == 0 else self._remap_receive_3d()
        hob = float(self._snapshot.hob_m) if is_finite_number(self._snapshot.hob_m) else 0.0
        centre = (0.0, hob, 0.0) if src_dim == "1d" else charge_center_for_dim(self._snapshot, tgt_dim)
        dx = self._snapshot.domain_cell_2d if tgt_dim == "2d" else None
        ray = float(dx) if is_finite_number(dx) and float(dx) > 0.0 else None
        src_stamp = self._remap_field_mtime(src, st, field)
        tgt_stamp = self._remap_field_mtime(target, tt, field)
        cache_key = (src, target, st, tt, field, src_dim, tgt_dim, centre, ray, receive, src_stamp, tgt_stamp)
        cached = self._remap_cache.get(cache_key)
        if cached is not None:
            cmp, r_max, status_text, status_level = cached
        else:
            r_s, v_s, src_err = remap_engine.load_physical_radial_profile(
                src, st or "0", field, dim=src_dim, charge_centre=centre, ray_half_width=None
            )
            r_t, v_t, tgt_err = remap_engine.load_physical_radial_profile(
                target, tt or "0", field, dim=tgt_dim, charge_centre=centre, ray_half_width=ray
            )
            r_max = float(receive or self._snapshot.mapped_radius or 0.0 or (float(r_s.max()) if r_s.size else 0.0))
            if r_max <= 0 and r_s.size:
                r_max = float(r_s.max())
            st_f = None
            tt_f = None
            try:
                st_f = float(st) if st is not None else None
            except (TypeError, ValueError):
                st_f = None
            try:
                tt_f = float(tt) if tt is not None else None
            except (TypeError, ValueError):
                tt_f = None
            offset = None
            plan = read_sampling_plan(target) if target else None
            if plan and plan.remap_timing:
                raw = plan.remap_timing.get("physical_time_offset")
                if is_finite_number(raw):
                    offset = float(raw)
            if offset is None and self.combo_remap_mode.currentIndex() == 0:
                recorded = remap_engine.recorded_1d2d_source_time(target or "")
                if recorded:
                    offset = remap_engine.build_remap_timing(
                        source_time_label=recorded, target_time_label=tt
                    ).physical_time_offset
            if offset is None:
                offset = remap_engine.build_remap_timing(
                    source_time_label=st, target_time_label=tt
                ).physical_time_offset
            corr = (
                "1D spherical profile vs 2D charge-height ray; "
                "source interpolated onto target physical R from the charge centre."
                if src_dim == "1d"
                else "Source and target paired at the same physical R from the charge centre."
            )
            cmp = remap_engine.compare_profiles(
                field=field,
                source_r=r_s,
                source_v=v_s,
                target_r=r_t,
                target_v=v_t,
                r_max=r_max,
                source_time=st_f,
                target_time=tt_f,
                physical_time_offset=offset,
                correspondence=corr,
            )
            if src_err:
                status_text, status_level = src_err, "warning"
            elif tgt_err:
                status_text, status_level = tgt_err, "warning"
            elif cmp.message:
                status_text, status_level = cmp.message, "warning"
            else:
                status_text, status_level = "", "info"
            self._remap_cache[cache_key] = (cmp, r_max, status_text, status_level)
        if status_text:
            self._set_status(status_text, status_level)
        dt_txt = fmt(cmp.delta_t, suffix="s")
        warn = "" if cmp.synchronized else "  (NOT SYNCHRONIZED)"
        unit = cmp.field_unit
        unit_txt = f" {unit}" if unit else ""
        self.lbl_remap_times.setText(
            f"Source physical time: {fmt(cmp.source_physical_time, suffix='s')}\n"
            f"Target physical time: {fmt(cmp.target_physical_time, suffix='s')}\n"
            f"Physical time offset: {fmt(cmp.physical_time_offset, suffix='s')}\n"
            f"OpenFOAM source/target: {fmt(cmp.source_time, suffix='s')} / {fmt(cmp.target_time, suffix='s')}\n"
            f"Delta t: {dt_txt}{warn}\n"
            f"Physical remap radius: {fmt(r_max, suffix='m')}"
        )
        ax.plot(cmp.r, cmp.source, label="Source Remap")
        ax.plot(cmp.r, cmp.target, label="Target After Initialize")
        if self.combo_remap_diff.currentIndex() == 0:
            ax.plot(cmp.r, cmp.abs_diff, "--", label="|Target − Source|")
        else:
            rel = [v if v is not None else math.nan for v in cmp.rel_diff]
            ax.plot(cmp.r, rel, "--", label="(Target − Source) / Source")
        if r_max > 0:
            ax.axvline(r_max, color="#888888", linestyle=":", label="R_remap")
        ax.set_xlabel("Physical R from charge centre [m]")
        if self.combo_remap_diff.currentIndex() == 0:
            ax.set_ylabel(f"{field}{unit_txt}")
        else:
            ax.set_ylabel("Relative difference")
        ax.legend(loc="best")
        self.plot_canvas.draw_idle()
        self._init_table(["Metric", "Value"])
        self._append_table_row(["Physical remap radius", fmt(r_max, suffix="m")])
        self._append_table_row(["Comparison interval", f"{cmp.interval[0]:.4g} … {cmp.interval[1]:.4g} m"])
        self._append_table_row(["Paired samples", str(cmp.n_pairs)])
        self._append_table_row(["Correspondence", cmp.correspondence or "N/A"])
        self._append_table_row(["RMS |Target − Source|", fmt(cmp.rms, suffix=unit)])
        self._append_table_row(["Mean |Target − Source|", fmt(cmp.mae, suffix=unit)])
        self._append_table_row(["Maximum |Target − Source|", fmt(cmp.max_abs, suffix=unit)])
        self._append_table_row(["Maximum meaningful relative error", fmt(cmp.max_rel)])
        self._append_table_row(["Source shock-front position", fmt(cmp.shock_source, suffix="m")])
        self._append_table_row(["Target shock-front position", fmt(cmp.shock_target, suffix="m")])
        self._append_table_row(
            ["Shock-front difference", fmt(None if cmp.shock_source is None or cmp.shock_target is None else cmp.shock_target - cmp.shock_source, suffix="m")]
        )
        self._append_table_row(["Source peak on interval", fmt(cmp.peak_source, suffix=unit)])
        self._append_table_row(["Target peak on interval", fmt(cmp.peak_target, suffix=unit)])
        self._append_table_row(["Source peak R", fmt(cmp.peak_r_source, suffix="m")])
        self._append_table_row(["Target peak R", fmt(cmp.peak_r_target, suffix="m")])

    def _draw_numerical(self) -> None:
        ax = self.plot_canvas.axes
        ax.clear()
        case = primary_case_dir(self._snapshot)
        dim = ""
        if case:
            dim = "2d"
            if case_dir_for_dim(self._snapshot, "3d") == case:
                dim = "3d"
            elif case_dir_for_dim(self._snapshot, "1d") == case:
                dim = "1d"
            live = str(self._snapshot.live_mode or "").strip().lower()
            if live in ("1d", "2d", "3d"):
                dim = live
            elif self._snapshot.live_mode:
                dim = {"1D": "1d", "2D": "2d", "3D": "3d"}.get(self._snapshot.live_mode, dim)
        keep = self._snapshot.keep_openfoam_2d if dim == "2d" else self._snapshot.keep_openfoam_3d
        log_path = os.path.join(case, "log.blastFoam") if case else ""
        try:
            log_mtime = os.path.getmtime(log_path) if log_path and os.path.isfile(log_path) else None
        except OSError:
            log_mtime = None
        num_key = (case, dim, keep, log_mtime, id(self._snapshot.output_options))
        if self._numerical_cache.get("key") == num_key:
            report = self._numerical_cache["report"]
        else:
            report = numerical_engine.build_report(
                case or "",
                dim=dim,
                options=self._snapshot.output_options,
                keep_openfoam_time_folders=keep,
            )
            self._numerical_cache = {"key": num_key, "report": report}
        if report.foam_fatal or report.fpe:
            self._set_status(
                "; ".join([n for n in report.notes if n] + [
                    "FOAM FATAL" if report.foam_fatal else "",
                    "Floating-point exception" if report.fpe else "",
                ]).strip("; "),
                "error",
            )
        elif report.notes:
            note = "; ".join(report.notes)
            level = "info" if "constant" in note.lower() and "amr" in note.lower() else "warning"
            if "cell count is constant" in note.lower():
                level = "info"
            self._set_status(note, level)
        choice = self.combo_num_plot.currentText()
        if "Courant" in choice:
            if report.courant.time and report.courant.values:
                ax.plot([s_to_ms(t) for t in report.courant.time], report.courant.values, label="max Co")
            if report.max_co_configured is not None:
                ax.axhline(report.max_co_configured, color="#888888", linestyle="--", label="maxCo")
            ax.set_ylabel("Courant Number")
        elif "deltaT" in choice:
            if report.delta_t.time and report.delta_t.values:
                ax.plot([s_to_ms(t) for t in report.delta_t.time], report.delta_t.values, label="deltaT")
            ax.set_ylabel("deltaT [s]")
        else:
            if report.cells.time and report.cells.values:
                ax.plot([s_to_ms(t) for t in report.cells.time], report.cells.values, label="cells")
            ax.set_ylabel("Total Cells")
        ax.set_xlabel("Time [ms]")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")
        self.plot_canvas.draw_idle()
        self._init_table(["Item", "Value"])
        def yn(v):
            if v is None:
                return "N/A"
            return "yes" if v else "no"
        rows = [
            ("Dimension", report.dimension or "N/A"),
            ("Solver", report.solver),
            ("Run status", report.run_status),
            ("Simulated start", fmt(report.start_time, suffix="s")),
            ("Simulated end", fmt(report.end_time, suffix="s")),
            ("Time steps", fmt(None if report.n_steps is None else float(report.n_steps))),
            ("CPU time", fmt(report.cpu_time_s, suffix="s")),
            ("Wall time", fmt(report.wall_time_s, suffix="s")),
            ("Cores", fmt(None if report.n_cores is None else float(report.n_cores))),
            ("Configured maxCo", fmt(report.max_co_configured)),
            ("Refine events", str(report.refine_events)),
            ("Unrefine events", str(report.unrefine_events)),
            ("checkMesh OK", "N/A" if report.checkmesh_ok is None else ("yes" if report.checkmesh_ok else "no")),
            ("Cell count", fmt(None if report.n_cells is None else float(report.n_cells))),
            ("Max non-orthogonality", fmt(report.max_nonortho)),
            ("Max skewness", fmt(report.max_skewness)),
            ("FOAM FATAL", yn(report.foam_fatal)),
            ("FOAM ERROR", yn(report.foam_error)),
            ("Floating point exception", yn(report.fpe)),
            ("Solver completion", "N/A" if report.completed is None else ("yes" if report.completed else "no")),
            ("Reconstruction", "N/A" if report.reconstruct_ok is None else ("yes" if report.reconstruct_ok else "no")),
        ]
        for a, b in rows:
            self._append_table_row([a, b])
        for name, status in report.completeness:
            self._append_table_row([name, status])

    def _init_table(self, headers: Sequence[str]) -> None:
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(list(headers))
        self.table.setRowCount(0)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def _append_table_row(self, values: Sequence[str]) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, text in enumerate(values):
            self.table.setItem(r, c, QTableWidgetItem(str(text)))
