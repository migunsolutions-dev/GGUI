"""Shared Time History Viewer: catalog of 1D/2D/3D gauges and a live p(t) plot."""
from __future__ import annotations

import os
import re
import textwrap
import csv
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QVBoxLayout,
    QWidget,
)

from models_2d import ProbePoint2D
from probes_model import ProbePoint
from tab_1d import MplCanvas
from ui_metrics import COMPUTATIONAL_LEFT_PANEL_MIN, COMPUTATIONAL_LEFT_PANEL_WIDTH

DEFAULT_P_ATM = 101325.0
PROBE_FO = {"1d": "gauges1d", "2d": "probes2d", "3d": "probes3d"}
PLOT_PAD = 0.12
_DIM_FROM_MODE = {"1D": "1d", "2D": "2d", "3D": "3d", "1d": "1d", "2d": "2d", "3d": "3d"}
GAUGE_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
)
_PROBE_HEADER = re.compile(r"Probe\s+(\d+)\s+\(([^)]+)\)")


@dataclass(frozen=True)
class GaugeRow:
    dim: str
    index: int
    gauge_id: str
    x: float
    y: float
    z: float
    label: str

    @property
    def key(self) -> Tuple[str, int]:
        return (self.dim, self.index)


@dataclass
class ImportedSeries:
    uid: str
    source: str
    dim: str
    field: str
    label: str
    times: List[float]
    values: List[float]
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    axis: str = "left"
    plotted: bool = False
    color: str = ""

    def extrema(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        pairs = [
            (float(t), float(v))
            for t, v in zip(self.times, self.values)
            if math.isfinite(float(t)) and math.isfinite(float(v))
        ]
        if not pairs:
            return None, None, None, None
        min_t, min_v = min(pairs, key=lambda pair: pair[1])
        max_t, max_v = max(pairs, key=lambda pair: pair[1])
        return min_v, min_t, max_v, max_t


@dataclass
class AxisAppearance:
    title: str
    minimum: str = ""
    maximum: str = ""
    axis_color: str = "#303030"
    major_grid_color: str = "#b0b0b0"
    minor_grid_color: str = "#d8d8d8"
    show_major_grid: bool = True
    show_minor_grid: bool = False
    position: str = ""


@dataclass
class PlotAppearance:
    title: str = ""
    show_legend: bool = True
    legend_position: str = "Right"
    x_axis: AxisAppearance = field(
        default_factory=lambda: AxisAppearance("Time (s)", position="Bottom")
    )
    left_axis: AxisAppearance = field(
        default_factory=lambda: AxisAppearance("Overpressure (Pa)", position="Left")
    )
    right_axis: AxisAppearance = field(
        default_factory=lambda: AxisAppearance("Impulse", position="Right")
    )


def catalog_rows(
    gauges_1d: Sequence[Tuple[float, str]] = (),
    probes_2d: Sequence[ProbePoint2D] = (),
    probes_3d: Sequence[ProbePoint] = (),
) -> List[GaugeRow]:
    """Flatten defined 1D/2D/3D locations into table rows. Regions contribute none."""
    rows: List[GaugeRow] = []
    for index, item in enumerate(gauges_1d or ()):
        radius, label = item
        rows.append(
            GaugeRow(
                dim="1d",
                index=index,
                gauge_id=f"1D-{index + 1}",
                x=float(radius),
                y=0.0,
                z=0.0,
                label=str(label or f"G{index + 1}"),
            )
        )
    for index, probe in enumerate(probes_2d or ()):
        rows.append(
            GaugeRow(
                dim="2d",
                index=index,
                gauge_id=f"2D-{index + 1}",
                x=float(probe.radius),
                y=float(probe.height),
                z=0.0,
                label=str(probe.name or f"P{index + 1}"),
            )
        )
    for index, probe in enumerate(probes_3d or ()):
        rows.append(
            GaugeRow(
                dim="3d",
                index=index,
                gauge_id=f"3D-{index + 1}",
                x=float(probe.x),
                y=float(probe.y),
                z=float(probe.z),
                label=str(probe.name or f"G{index + 1}"),
            )
        )
    return rows


def latest_probe_field_file(case_dir: str, fo_name: str, field: str) -> str:
    """Newest postProcessing/<fo>/<time>/<field> path, or empty."""
    root = os.path.join(case_dir or "", "postProcessing", fo_name)
    if not os.path.isdir(root):
        return ""
    best_t = None
    best_path = ""
    try:
        names = os.listdir(root)
    except OSError:
        return ""
    for name in names:
        path = os.path.join(root, name, field)
        if not os.path.isfile(path):
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        if best_t is None or t >= best_t:
            best_t = t
            best_path = path
    return best_path


def parse_probe_history(path: str) -> Tuple[List[str], List[float], List[List[float]]]:
    """Parse an OpenFOAM probes ASCII file into locations, times, and per-probe series."""
    locations: List[str] = []
    times: List[float] = []
    columns: List[List[float]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return locations, times, columns
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines = lines[:-1]
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = _PROBE_HEADER.search(line)
            if match:
                idx = int(match.group(1))
                while len(locations) <= idx:
                    locations.append("")
                locations[idx] = match.group(2).strip()
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            values = [float(part) for part in parts[1:]]
        except ValueError:
            continue
        times.append(t)
        if not columns:
            columns = [[] for _ in values]
        if len(columns) < len(values):
            columns.extend([] for _ in range(len(values) - len(columns)))
        for index, value in enumerate(values):
            columns[index].append(value)
        for extra in columns[len(values) :]:
            extra.append(float("nan"))
    return locations, times, columns


def parse_external_timeseries(path: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """Parse named CSV/TXT columns: first column is time, remaining columns are series."""
    try:
        with open(path, encoding="utf-8-sig", errors="replace", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(handle, dialect)
            rows = list(reader)
    except OSError:
        return [], {}
    if not rows or len(rows[0]) < 2:
        return [], {}
    headers = [str(value).strip() for value in rows[0]]
    if not headers[0] or any(not name for name in headers[1:]):
        return [], {}
    times: List[float] = []
    series: Dict[str, List[float]] = {name: [] for name in headers[1:]}
    for row in rows[1:]:
        if len(row) < len(headers):
            continue
        try:
            time_value = float(row[0])
            values = [float(row[index]) for index in range(1, len(headers))]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(time_value) or any(not math.isfinite(value) for value in values):
            continue
        times.append(time_value)
        for name, value in zip(headers[1:], values):
            series[name].append(value)
    return times, series


def _location_xyz(text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    parts = str(text or "").replace(",", " ").split()
    try:
        values = [float(part) for part in parts[:3]]
    except ValueError:
        return None, None, None
    while len(values) < 3:
        values.append(0.0)
    return values[0], values[1], values[2]


def wrap_legend_name(name: str, max_chars: int) -> str:
    """Fit a gauge name in the legend column; wrap long names onto the next line."""
    text = str(name or "").strip() or "—"
    width = max(8, int(max_chars))
    return textwrap.fill(text, width=width, break_long_words=True, break_on_hyphens=True)


def padded_axis_limits(
    low: float,
    high: float,
    *,
    pad: float = PLOT_PAD,
    pin_zero: bool = True,
) -> Tuple[float, float]:
    """Keep data inside the axes with a little headroom above the peak."""
    lo = float(low)
    hi = float(high)
    if pin_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = hi - lo
    if span <= 0.0:
        span = max(abs(hi), 1.0)
    return lo, hi + span * pad


class TabTimeHistory(QWidget):
    """Main-tab viewer: filter gauges, add series, plot live probe histories."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._gauges_1d_fn: Callable[[], Sequence] = lambda: ()
        self._probes_2d_fn: Callable[[], Sequence] = lambda: ()
        self._probes_3d_fn: Callable[[], Sequence] = lambda: ()
        self._case_dir_fn: Callable[[str], str] = lambda _dim: ""
        self._p_atm_fn: Callable[[str], float] = lambda _dim: DEFAULT_P_ATM
        self._impulse_ok_fn: Callable[[], bool] = lambda: True
        self._rows: List[GaugeRow] = []
        self._added: List[Tuple[str, int]] = []
        self._sim_time = {"1d": 0.0, "2d": 0.0, "3d": 0.0}
        self._run_cases = {"1d": "", "2d": "", "3d": ""}
        self._run_baselines = {}
        self._imported: List[ImportedSeries] = []
        self._loaded_sources: List[str] = []
        self._appearance = PlotAppearance()
        self._right_axes = None
        self._color_buttons: Dict[str, QPushButton] = {}
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.setInterval(50)
        self._plot_timer.timeout.connect(self._redraw_plot)
        self._build_ui()
        self._sync_impulse_checkbox()

    def set_source_provider(
        self,
        *,
        gauges_1d: Callable[[], Sequence] = None,
        probes_2d: Callable[[], Sequence] = None,
        probes_3d: Callable[[], Sequence] = None,
        case_dir: Callable[[str], str] = None,
        p_atm: Callable[[str], float] = None,
        impulse_available: Callable[[], bool] = None,
    ) -> None:
        if gauges_1d is not None:
            self._gauges_1d_fn = gauges_1d
        if probes_2d is not None:
            self._probes_2d_fn = probes_2d
        if probes_3d is not None:
            self._probes_3d_fn = probes_3d
        if case_dir is not None:
            self._case_dir_fn = case_dir
        if p_atm is not None:
            self._p_atm_fn = p_atm
        if impulse_available is not None:
            self._impulse_ok_fn = impulse_available
        self.set_impulse_available()

    def set_impulse_available(self, available: Callable[[], bool] = None) -> None:
        if available is not None:
            self._impulse_ok_fn = available
        self._sync_impulse_checkbox()

    def _sync_impulse_checkbox(self) -> None:
        ok = True
        try:
            ok = bool(self._impulse_ok_fn())
        except Exception:
            ok = True
        self.chk_impulse.setEnabled(ok)
        self.chk_impulse.setVisible(ok)
        if not ok and self.chk_impulse.isChecked():
            self.chk_impulse.blockSignals(True)
            self.chk_impulse.setChecked(False)
            self.chk_impulse.blockSignals(False)

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        self.canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.canvas.tight_layout_rect = (0.0, 0.0, 0.78, 1.0)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([COMPUTATIONAL_LEFT_PANEL_WIDTH, 1100])
        splitter.splitterMoved.connect(self._on_plot_splitter_moved)
        self._plot_splitter = splitter
        root.addWidget(splitter)

    def _build_left(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setObjectName("timeHistoryWorkspaceTabs")
        tabs.setMinimumWidth(COMPUTATIONAL_LEFT_PANEL_MIN)
        tabs.addTab(self._build_gauges_tab(), "Gauges")
        tabs.addTab(self._build_add_data_tab(), "Add Data")
        tabs.addTab(self._build_appearance_tab(), "Appearance")
        self.workspace_tabs = tabs
        return tabs

    def _build_gauges_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Filter"))

        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        self.chk_3d = QCheckBox("3D")
        self.chk_2d = QCheckBox("2D")
        self.chk_1d = QCheckBox("1D")
        self.chk_3d.setChecked(True)
        self.chk_2d.setChecked(True)
        self.chk_1d.setChecked(True)
        for box in (self.chk_3d, self.chk_2d, self.chk_1d):
            box.toggled.connect(self._apply_filters)
            filter_row.addWidget(box)
        filter_row.addStretch(1)
        self.btn_add = QPushButton("Add")
        self.btn_clear = QPushButton("Clear")
        self.btn_add.clicked.connect(self.add_selected)
        self.btn_clear.clicked.connect(self.clear_series)
        filter_row.addWidget(self.btn_add)
        filter_row.addWidget(self.btn_clear)
        layout.addLayout(filter_row)

        region_row = QHBoxLayout()
        region_row.setSpacing(8)
        self.chk_regions = QCheckBox("Regions")
        self.chk_regions.setChecked(False)
        self.chk_regions.setToolTip("Region gauges are not defined yet.")
        self.chk_regions.toggled.connect(self._apply_filters)
        region_row.addWidget(self.chk_regions)
        region_row.addStretch(1)
        layout.addLayout(region_row)

        field_row = QHBoxLayout()
        field_row.setSpacing(8)
        self.chk_pressure = QCheckBox("Pressure")
        self.chk_impulse = QCheckBox("Impulse")
        self.chk_pressure.setChecked(True)
        self.chk_impulse.setChecked(False)
        self.chk_pressure.toggled.connect(self.refresh_plot)
        self.chk_impulse.toggled.connect(self.refresh_plot)
        field_row.addWidget(self.chk_pressure)
        field_row.addWidget(self.chk_impulse)
        field_row.addStretch(1)
        layout.addLayout(field_row)

        self.tbl_gauges = QTableWidget(0, 5)
        self.tbl_gauges.setHorizontalHeaderLabels(["ID", "X", "Y", "Z", "Label"])
        self.tbl_gauges.verticalHeader().setVisible(False)
        self.tbl_gauges.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_gauges.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl_gauges.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.tbl_gauges.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(35)
        header.setDefaultSectionSize(70)
        header.resizeSection(4, 100)
        layout.addWidget(self.tbl_gauges, 1)
        return page

    def _build_add_data_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        btn_case = QPushButton("Open Completed OpenFOAM Case")
        btn_case.clicked.connect(self._choose_completed_case)
        btn_file = QPushButton("Import External Dataset (.csv, .txt)")
        btn_file.clicked.connect(self._choose_external_file)
        layout.addWidget(btn_case)
        layout.addWidget(btn_file)

        layout.addWidget(QLabel("Loaded Sources"))
        self.lst_import_sources = QListWidget()
        self.lst_import_sources.setMaximumHeight(90)
        layout.addWidget(self.lst_import_sources)

        variable_row = QFormLayout()
        self.cmb_import_variable = QComboBox()
        self.cmb_import_variable.addItem("Pressure", "p")
        self.cmb_import_variable.addItem("Impulse", "impulse")
        self.cmb_import_variable.currentIndexChanged.connect(self._refresh_import_table)
        variable_row.addRow("Variable", self.cmb_import_variable)
        layout.addLayout(variable_row)

        dims_row = QHBoxLayout()
        self.import_dim_group = QButtonGroup(self)
        for index, (text, dim) in enumerate(
            (("1D", "1d"), ("2D", "2d"), ("3D", "3d"), ("Regions", "regions"))
        ):
            radio = QRadioButton(text)
            radio.setProperty("dimension", dim)
            radio.setEnabled(dim != "regions")
            self.import_dim_group.addButton(radio)
            dims_row.addWidget(radio)
            if index == 0:
                radio.setChecked(True)
        self.import_dim_group.buttonClicked.connect(self._refresh_import_table)
        layout.addLayout(dims_row)

        self.tbl_imported = QTableWidget(0, 9)
        self.tbl_imported.setHorizontalHeaderLabels(
            ["ID", "Label", "X", "Y", "Z", "Min", "MinT", "Max", "MaxT"]
        )
        self.tbl_imported.verticalHeader().setVisible(False)
        self.tbl_imported.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_imported.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl_imported.setEditTriggers(QAbstractItemView.NoEditTriggers)
        imported_header = self.tbl_imported.horizontalHeader()
        imported_header.setSectionResizeMode(QHeaderView.Interactive)
        imported_header.setDefaultSectionSize(65)
        imported_header.resizeSection(1, 100)
        layout.addWidget(self.tbl_imported, 1)

        axis_buttons = QHBoxLayout()
        self.btn_plot_left = QPushButton("Plot on Left Axis")
        self.btn_plot_right = QPushButton("Plot on Right Axis")
        self.btn_plot_left.clicked.connect(lambda: self._plot_import_selection("left"))
        self.btn_plot_right.clicked.connect(lambda: self._plot_import_selection("right"))
        axis_buttons.addWidget(self.btn_plot_left)
        axis_buttons.addWidget(self.btn_plot_right)
        layout.addLayout(axis_buttons)

        clear_buttons = QHBoxLayout()
        btn_remove = QPushButton("Remove Selected")
        btn_clear = QPushButton("Clear Imported")
        btn_remove.clicked.connect(self._remove_import_selection)
        btn_clear.clicked.connect(self.clear_imported)
        clear_buttons.addWidget(btn_remove)
        clear_buttons.addWidget(btn_clear)
        layout.addLayout(clear_buttons)
        return page

    def _build_appearance_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        general = QGroupBox("Plot")
        general_form = QFormLayout(general)
        self.edit_plot_title = QLineEdit()
        self.edit_plot_title.textChanged.connect(self._appearance_changed)
        self.chk_show_legend = QCheckBox("Show Legend")
        self.chk_show_legend.setChecked(True)
        self.chk_show_legend.toggled.connect(self._appearance_changed)
        self.cmb_legend_position = QComboBox()
        self.cmb_legend_position.addItems(["Left", "Right", "Top", "Bottom"])
        self.cmb_legend_position.setCurrentText("Right")
        self.cmb_legend_position.currentTextChanged.connect(self._appearance_changed)
        general_form.addRow("Plot Title", self.edit_plot_title)
        general_form.addRow("", self.chk_show_legend)
        general_form.addRow("Legend Position", self.cmb_legend_position)
        layout.addWidget(general)

        self._appearance_controls = {}
        layout.addWidget(
            self._build_axis_appearance_group(
                "X Axis Settings", "x", self._appearance.x_axis, ("Bottom", "Top")
            )
        )
        layout.addWidget(
            self._build_axis_appearance_group(
                "Left Y Axis Settings", "left", self._appearance.left_axis, ("Left", "Right")
            )
        )
        layout.addWidget(
            self._build_axis_appearance_group(
                "Right Y Axis Settings", "right", self._appearance.right_axis, ("Right", "Left")
            )
        )
        layout.addStretch(1)
        scroll.setWidget(page)
        return scroll

    def _build_axis_appearance_group(
        self,
        title: str,
        key: str,
        settings: AxisAppearance,
        positions: Sequence[str],
    ) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        title_edit = QLineEdit(settings.title)
        minimum_edit = QLineEdit()
        minimum_edit.setPlaceholderText("Scaled to Data")
        maximum_edit = QLineEdit()
        maximum_edit.setPlaceholderText("Scaled to Data")
        major = QCheckBox("Show")
        major.setChecked(settings.show_major_grid)
        minor = QCheckBox("Show")
        minor.setChecked(settings.show_minor_grid)
        position = QComboBox()
        position.addItems(list(positions))
        position.setCurrentText(settings.position)
        axis_color = self._make_color_button(f"{key}.axis", settings.axis_color)
        major_color = self._make_color_button(f"{key}.major", settings.major_grid_color)
        minor_color = self._make_color_button(f"{key}.minor", settings.minor_grid_color)
        form.addRow("Axis Title", title_edit)
        form.addRow("Axis Colour", axis_color)
        form.addRow("Primary Grid Colour", self._color_check_row(major_color, major))
        form.addRow("Secondary Grid Colour", self._color_check_row(minor_color, minor))
        form.addRow("Minimum", minimum_edit)
        form.addRow("Maximum", maximum_edit)
        form.addRow("Axis Position", position)
        self._appearance_controls[key] = {
            "title": title_edit,
            "minimum": minimum_edit,
            "maximum": maximum_edit,
            "major": major,
            "minor": minor,
            "position": position,
        }
        for widget in (title_edit, minimum_edit, maximum_edit):
            widget.textChanged.connect(self._appearance_changed)
        major.toggled.connect(self._appearance_changed)
        minor.toggled.connect(self._appearance_changed)
        position.currentTextChanged.connect(self._appearance_changed)
        return group

    @staticmethod
    def _color_check_row(button: QPushButton, checkbox: QCheckBox) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(button, 1)
        row.addWidget(checkbox)
        return host

    def _make_color_button(self, key: str, color: str) -> QPushButton:
        button = QPushButton(color)
        button.setProperty("plotColor", color)
        button.setStyleSheet(f"background:{color}; color:white;")
        button.clicked.connect(lambda _checked=False, k=key: self._choose_color(k))
        self._color_buttons[key] = button
        return button

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_catalog()
        QTimer.singleShot(0, self.canvas.draw_idle)

    def _on_plot_splitter_moved(self, *_args) -> None:
        if self.isVisible() and self.canvas.isVisible():
            self.canvas.draw_idle()

    def has_series(self) -> bool:
        return bool(self._added)

    def added_keys(self) -> List[Tuple[str, int]]:
        return list(self._added)

    def visible_rows(self) -> List[GaugeRow]:
        enabled = self._enabled_dims()
        return [row for row in self._rows if row.dim in enabled]

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
        self._rows = catalog_rows(
            self._gauges_1d_fn(),
            self._probes_2d_fn(),
            self._probes_3d_fn(),
        )
        self._apply_filters()

    def _choose_completed_case(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Completed OpenFOAM Case")
        if path:
            self.load_completed_case(path)

    def _choose_external_file(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Import External Time History Dataset",
            "",
            "Time series (*.csv *.txt);;All files (*)",
        )
        if path:
            self.load_external_file(path)

    def _selected_import_dim(self) -> str:
        button = self.import_dim_group.checkedButton()
        return str(button.property("dimension") or "1d") if button else "1d"

    def load_completed_case(self, case_dir: str) -> int:
        """Load pressure/impulse histories from a completed GGUI OpenFOAM case."""
        case_dir = os.path.normpath(case_dir or "")
        if not case_dir or not os.path.isdir(case_dir):
            return 0
        source = os.path.basename(case_dir) or case_dir
        before = len(self._imported)
        row_lookup = {row.key: row for row in self._rows}
        for dim, fo_name in PROBE_FO.items():
            for field_name in ("p", "impulse"):
                path = latest_probe_field_file(case_dir, fo_name, field_name)
                if not path:
                    continue
                locations, times, columns = parse_probe_history(path)
                for index, column in enumerate(columns):
                    count = min(len(times), len(column))
                    if count <= 0:
                        continue
                    values = [float(value) for value in column[:count]]
                    if field_name == "p":
                        values = [value - DEFAULT_P_ATM for value in values]
                    location = locations[index] if index < len(locations) else ""
                    x, y, z = _location_xyz(location)
                    catalog_row = row_lookup.get((dim, index))
                    label = (
                        catalog_row.label
                        if catalog_row is not None
                        else f"{dim.upper()}-{index + 1}"
                    )
                    self._upsert_imported(
                        ImportedSeries(
                            uid=f"case:{case_dir}:{dim}:{field_name}:{index}",
                            source=source,
                            dim=dim,
                            field=field_name,
                            label=label,
                            times=list(times[:count]),
                            values=values,
                            x=x,
                            y=y,
                            z=z,
                            color=GAUGE_COLORS[
                                (len(self._added) + index) % len(GAUGE_COLORS)
                            ],
                        )
                    )
        self._register_source(source)
        self._refresh_import_table()
        return len(self._imported) - before

    def load_external_file(
        self,
        path: str,
        *,
        field_name: Optional[str] = None,
        dim: Optional[str] = None,
    ) -> int:
        """Load a named-column CSV/TXT time history dataset."""
        times, columns = parse_external_timeseries(path)
        if not times or not columns:
            return 0
        source = os.path.basename(path)
        selected_field = field_name or str(self.cmb_import_variable.currentData() or "p")
        selected_dim = dim or self._selected_import_dim()
        before = len(self._imported)
        for index, (label, values) in enumerate(columns.items()):
            self._upsert_imported(
                ImportedSeries(
                    uid=f"file:{os.path.normpath(path)}:{selected_field}:{label}",
                    source=source,
                    dim=selected_dim,
                    field=selected_field,
                    label=label,
                    times=list(times),
                    values=list(values),
                )
            )
        self._register_source(source)
        self._refresh_import_table()
        return len(self._imported) - before

    def _upsert_imported(self, series: ImportedSeries) -> None:
        for index, existing in enumerate(self._imported):
            if existing.uid == series.uid:
                series.axis = existing.axis
                series.plotted = existing.plotted
                series.color = existing.color
                self._imported[index] = series
                return
        if not series.color:
            series.color = GAUGE_COLORS[
                (len(self._added) + len(self._imported)) % len(GAUGE_COLORS)
            ]
        self._imported.append(series)

    def _register_source(self, source: str) -> None:
        if source in self._loaded_sources:
            return
        self._loaded_sources.append(source)
        self.lst_import_sources.addItem(source)

    def _filtered_imported(self) -> List[ImportedSeries]:
        field_name = str(self.cmb_import_variable.currentData() or "p")
        dim = self._selected_import_dim()
        return [
            series
            for series in self._imported
            if series.field == field_name and series.dim == dim
        ]

    def _refresh_import_table(self, *_args) -> None:
        if not hasattr(self, "tbl_imported"):
            return
        rows = self._filtered_imported()
        self.tbl_imported.setRowCount(len(rows))
        for row_index, series in enumerate(rows):
            minimum, min_time, maximum, max_time = series.extrema()
            values = (
                str(row_index + 1),
                series.label,
                self._fmt_optional(series.x),
                self._fmt_optional(series.y),
                self._fmt_optional(series.z),
                self._fmt_optional(minimum),
                self._fmt_optional(min_time),
                self._fmt_optional(maximum),
                self._fmt_optional(max_time),
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                if column == 0:
                    item.setData(Qt.UserRole, series.uid)
                self.tbl_imported.setItem(row_index, column, item)

    def _selected_import_uids(self) -> set:
        uids = set()
        for index in self.tbl_imported.selectionModel().selectedRows():
            item = self.tbl_imported.item(index.row(), 0)
            if item is not None and item.data(Qt.UserRole):
                uids.add(str(item.data(Qt.UserRole)))
        return uids

    def _plot_import_selection(self, axis: str) -> None:
        uids = self._selected_import_uids()
        for series in self._imported:
            if series.uid in uids:
                series.axis = axis
                series.plotted = True
        self.refresh_plot()

    def _remove_import_selection(self) -> None:
        uids = self._selected_import_uids()
        self._imported = [series for series in self._imported if series.uid not in uids]
        self._refresh_import_table()
        self.refresh_plot()

    def clear_imported(self) -> None:
        self._imported.clear()
        self._loaded_sources.clear()
        self.lst_import_sources.clear()
        self._refresh_import_table()
        self.refresh_plot()

    def _apply_filters(self) -> None:
        selected = set(self._selected_keys())
        visible = self.visible_rows()
        self.tbl_gauges.setRowCount(len(visible))
        for row_i, row in enumerate(visible):
            values = (
                row.gauge_id,
                self._fmt(row.x),
                self._fmt(row.y),
                self._fmt(row.z),
                row.label,
            )
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                if col == 0:
                    item.setData(Qt.UserRole, row.key)
                self.tbl_gauges.setItem(row_i, col, item)
            if row.key in selected:
                self.tbl_gauges.setRangeSelected(
                    QTableWidgetSelectionRange(row_i, 0, row_i, 4), True
                )

    def _selected_keys(self) -> List[Tuple[str, int]]:
        keys: List[Tuple[str, int]] = []
        for index in self.tbl_gauges.selectionModel().selectedRows():
            item = self.tbl_gauges.item(index.row(), 0)
            if item is None:
                continue
            key = item.data(Qt.UserRole)
            if key:
                keys.append(tuple(key))
        return keys

    def add_selected(self) -> None:
        for key in self._selected_keys():
            if key not in self._added:
                self._added.append(key)
        self.refresh_plot()

    def clear_series(self) -> None:
        self._added.clear()
        self.refresh_plot()

    def refresh_plot(self) -> None:
        self._plot_timer.start()

    def begin_run(self, mode: str, case_dir: str) -> None:
        """Start a viewer session without exposing samples from an earlier run."""
        dim = _DIM_FROM_MODE.get(str(mode), "")
        if not dim:
            return
        case_dir = os.path.normpath(case_dir) if case_dir else ""
        self._run_cases[dim] = case_dir
        self._sim_time[dim] = 0.0
        for field in ("p", "impulse"):
            path = latest_probe_field_file(case_dir, PROBE_FO[dim], field)
            count = 0
            size = 0
            if path:
                _locs, times, _columns = parse_probe_history(path)
                count = len(times)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = 0
            self._run_baselines[(dim, field)] = (path, count, size)
        self.refresh_plot()

    def note_sim_progress(self, mode: str, time_s: float) -> None:
        """Track live solver time so the plot grows with the current run."""
        dim = _DIM_FROM_MODE.get(str(mode), "")
        if dim:
            try:
                self._sim_time[dim] = max(self._sim_time.get(dim, 0.0), float(time_s))
            except (TypeError, ValueError):
                pass
        if self.has_series():
            self.refresh_plot()

    def _active_fields(self) -> List[str]:
        fields: List[str] = []
        if self.chk_pressure.isChecked():
            fields.append("p")
        if self.chk_impulse.isEnabled() and self.chk_impulse.isChecked():
            fields.append("impulse")
        return fields

    def _legend_wrap_chars(self) -> int:
        width_px = max(1, int(self.canvas.contentsRect().width()))
        right = 0.78
        rect = getattr(self.canvas, "tight_layout_rect", None)
        if rect:
            right = float(rect[2])
        legend_px = max(40.0, width_px * (1.0 - right) - 18.0)
        return max(8, int(legend_px / 5.5))

    def _color_for_key(self, key: Tuple[str, int]) -> str:
        try:
            index = self._added.index(key)
        except ValueError:
            index = 0
        return GAUGE_COLORS[index % len(GAUGE_COLORS)]

    def _redraw_plot(self) -> None:
        try:
            self._redraw_plot_body()
        except Exception:
            return

    def _redraw_plot_body(self) -> None:
        axes = self.canvas.axes
        if self._right_axes is not None:
            try:
                self._right_axes.remove()
            except (KeyError, ValueError):
                pass
            self._right_axes = None
        axes.clear()
        self._sync_appearance_from_controls()
        fields = self._active_fields() or ["p"]
        plotted = False
        wrap_chars = self._legend_wrap_chars()
        lookup = {row.key: row for row in self._rows}
        all_times: List[float] = []
        left_values: List[float] = []
        right_values: List[float] = []
        for key in self._added:
            row = lookup.get(key)
            if row is None:
                continue
            color = self._color_for_key(key)
            case_dir = self._run_cases.get(row.dim, "") or self._case_dir_fn(row.dim) or ""
            p_atm = float(self._p_atm_fn(row.dim) or DEFAULT_P_ATM)
            fo_name = PROBE_FO[row.dim]
            both_fields = "p" in fields and "impulse" in fields
            wrapped = wrap_legend_name(row.label, wrap_chars)
            for field in fields:
                path = latest_probe_field_file(case_dir, fo_name, field)
                times: List[float] = []
                values: List[float] = []
                if path:
                    _locs, times, columns = parse_probe_history(path)
                    if row.index < len(columns) and times:
                        values = list(columns[row.index])
                        if field == "p":
                            values = [value - p_atm for value in values]
                        times, values = self._current_run_samples(
                            row.dim, field, path, times, values
                        )
                if len(times) != len(values):
                    times, values = [], []
                style = "--" if field == "impulse" else "-"
                label = "_nolegend_" if both_fields and field == "impulse" else wrapped
                axes.plot(times, values, color=color, linestyle=style, label=label)
                plotted = True
                all_times.extend(times)
                left_values.extend(v for v in values if math.isfinite(float(v)))

        imported_right = any(
            series.plotted and series.axis == "right" for series in self._imported
        )
        if imported_right:
            self._right_axes = axes.twinx()
        for series in self._imported:
            if not series.plotted:
                continue
            target = self._right_axes if series.axis == "right" else axes
            if target is None:
                continue
            style = "--" if series.field == "impulse" else "-"
            label = wrap_legend_name(f"{series.label} ({series.source})", wrap_chars)
            target.plot(
                series.times,
                series.values,
                color=series.color or GAUGE_COLORS[0],
                linestyle=style,
                label=label,
            )
            plotted = True
            all_times.extend(series.times)
            target_values = right_values if series.axis == "right" else left_values
            target_values.extend(
                value for value in series.values if math.isfinite(float(value))
            )

        self._apply_plot_appearance(
            axes, self._right_axes, all_times, left_values, right_values
        )
        if plotted and self._appearance.show_legend:
            self._draw_combined_legend(axes, self._right_axes)
        else:
            self.canvas.tight_layout_rect = (0.0, 0.0, 1.0, 1.0)
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _draw_combined_legend(self, axes, right_axes) -> None:
        handles, labels = axes.get_legend_handles_labels()
        if right_axes is not None:
            right_handles, right_labels = right_axes.get_legend_handles_labels()
            handles.extend(right_handles)
            labels.extend(right_labels)
        pairs = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label and label != "_nolegend_"
        ]
        if not pairs:
            return
        handles, labels = zip(*pairs)
        position = self._appearance.legend_position
        kwargs = {"frameon": True, "fontsize": 8}
        if position == "Left":
            kwargs.update(loc="center right", bbox_to_anchor=(-0.02, 0.5))
            self.canvas.tight_layout_rect = (0.18, 0.0, 1.0, 1.0)
        elif position == "Top":
            kwargs.update(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=max(1, min(4, len(labels))),
            )
            self.canvas.tight_layout_rect = (0.0, 0.0, 1.0, 0.88)
        elif position == "Bottom":
            kwargs.update(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=max(1, min(4, len(labels))),
            )
            self.canvas.tight_layout_rect = (0.0, 0.14, 1.0, 1.0)
        else:
            kwargs.update(loc="center left", bbox_to_anchor=(1.02, 0.5))
            self.canvas.tight_layout_rect = (0.0, 0.0, 0.78, 1.0)
        axes.legend(handles, labels, **kwargs)

    def _apply_plot_appearance(
        self,
        axes,
        right_axes,
        times: Sequence[float],
        left_values: Sequence[float],
        right_values: Sequence[float],
    ) -> None:
        appearance = self._appearance
        axes.set_title(appearance.title)
        self._style_axis(axes, appearance.x_axis, axis="x")
        self._style_axis(axes, appearance.left_axis, axis="y")
        if right_axes is not None:
            self._style_axis(right_axes, appearance.right_axis, axis="y")

        t_max = max(times) if times else 0.0
        t_max = max(t_max, self._live_sim_time())
        x_auto = padded_axis_limits(0.0, t_max)
        axes.set_xlim(*self._axis_limits(appearance.x_axis, x_auto))
        if left_values:
            left_auto = padded_axis_limits(min(left_values), max(left_values))
            axes.set_ylim(*self._axis_limits(appearance.left_axis, left_auto))
        elif self._manual_limits(appearance.left_axis) is not None:
            axes.set_ylim(*self._manual_limits(appearance.left_axis))
        if right_axes is not None:
            if right_values:
                right_auto = padded_axis_limits(min(right_values), max(right_values))
                right_axes.set_ylim(*self._axis_limits(appearance.right_axis, right_auto))
            elif self._manual_limits(appearance.right_axis) is not None:
                right_axes.set_ylim(*self._manual_limits(appearance.right_axis))
        if not appearance.show_legend:
            self.canvas.tight_layout_rect = (0.0, 0.0, 1.0, 1.0)

    @staticmethod
    def _manual_limits(settings: AxisAppearance) -> Optional[Tuple[float, float]]:
        try:
            minimum = float(settings.minimum)
            maximum = float(settings.maximum)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            return None
        return minimum, maximum

    def _axis_limits(
        self, settings: AxisAppearance, automatic: Tuple[float, float]
    ) -> Tuple[float, float]:
        return self._manual_limits(settings) or automatic

    @staticmethod
    def _style_axis(axes, settings: AxisAppearance, *, axis: str) -> None:
        if axis == "x":
            axes.set_xlabel(settings.title, color=settings.axis_color)
            axes.tick_params(axis="x", colors=settings.axis_color)
            if settings.position == "Top":
                axes.xaxis.set_label_position("top")
                axes.xaxis.tick_top()
            else:
                axes.xaxis.set_label_position("bottom")
                axes.xaxis.tick_bottom()
        else:
            axes.set_ylabel(settings.title, color=settings.axis_color)
            axes.tick_params(axis="y", colors=settings.axis_color)
            if settings.position == "Right":
                axes.yaxis.set_label_position("right")
                axes.yaxis.tick_right()
            else:
                axes.yaxis.set_label_position("left")
                axes.yaxis.tick_left()
        axes.minorticks_on()
        if settings.show_major_grid:
            axes.grid(
                True,
                which="major",
                axis=axis,
                color=settings.major_grid_color,
                alpha=0.45,
            )
        else:
            axes.grid(False, which="major", axis=axis)
        if settings.show_minor_grid:
            axes.grid(
                True,
                which="minor",
                axis=axis,
                color=settings.minor_grid_color,
                alpha=0.3,
            )
        else:
            axes.grid(False, which="minor", axis=axis)

    def _choose_color(self, key: str) -> None:
        button = self._color_buttons.get(key)
        if button is None:
            return
        current = QColor(str(button.property("plotColor") or "#303030"))
        color = QColorDialog.getColor(current, self, "Select Plot Colour")
        if not color.isValid():
            return
        value = color.name()
        button.setProperty("plotColor", value)
        button.setText(value)
        button.setStyleSheet(f"background:{value}; color:white;")
        self._appearance_changed()

    def _appearance_changed(self, *_args) -> None:
        self._sync_appearance_from_controls()
        self.refresh_plot()

    def _sync_appearance_from_controls(self) -> None:
        if not hasattr(self, "edit_plot_title"):
            return
        self._appearance.title = self.edit_plot_title.text()
        self._appearance.show_legend = self.chk_show_legend.isChecked()
        self._appearance.legend_position = self.cmb_legend_position.currentText()
        mapping = {
            "x": self._appearance.x_axis,
            "left": self._appearance.left_axis,
            "right": self._appearance.right_axis,
        }
        for key, settings in mapping.items():
            controls = self._appearance_controls[key]
            settings.title = controls["title"].text()
            settings.minimum = controls["minimum"].text().strip()
            settings.maximum = controls["maximum"].text().strip()
            settings.show_major_grid = controls["major"].isChecked()
            settings.show_minor_grid = controls["minor"].isChecked()
            settings.position = controls["position"].currentText()
            settings.axis_color = str(
                self._color_buttons[f"{key}.axis"].property("plotColor")
            )
            settings.major_grid_color = str(
                self._color_buttons[f"{key}.major"].property("plotColor")
            )
            settings.minor_grid_color = str(
                self._color_buttons[f"{key}.minor"].property("plotColor")
            )

    def _current_run_samples(
        self,
        dim: str,
        field: str,
        path: str,
        times: Sequence[float],
        values: Sequence[float],
    ) -> Tuple[List[float], List[float]]:
        baseline_path, baseline_count, baseline_size = self._run_baselines.get(
            (dim, field), ("", 0, 0)
        )
        try:
            current_size = os.path.getsize(path)
        except OSError:
            current_size = 0
        # A different/truncated file belongs to the new run; otherwise skip
        # the rows that already existed when this run was started.
        start = 0
        if path == baseline_path and current_size >= baseline_size:
            start = min(int(baseline_count), len(times), len(values))
        else:
            self._run_baselines[(dim, field)] = (path, 0, 0)
        return list(times[start:]), list(values[start:])

    def _live_sim_time(self) -> float:
        latest = 0.0
        for dim, _index in self._added:
            latest = max(latest, float(self._sim_time.get(dim, 0.0) or 0.0))
        return latest

    def _apply_live_limits(self, axes, times: Sequence[float], values: Sequence[float]) -> None:
        t_max = max(times) if times else 0.0
        t_max = max(t_max, self._live_sim_time())
        x0, x1 = padded_axis_limits(0.0, t_max)
        axes.set_xlim(x0, x1)
        if values:
            y0, y1 = padded_axis_limits(min(values), max(values))
            axes.set_ylim(y0, y1)

    @staticmethod
    def _fmt(value: float) -> str:
        if abs(value) >= 1.0 or value == 0.0:
            text = f"{value:.6g}"
        else:
            text = f"{value:.6g}"
        return text

    @classmethod
    def _fmt_optional(cls, value: Optional[float]) -> str:
        return "—" if value is None else cls._fmt(float(value))
