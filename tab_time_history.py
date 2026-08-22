"""Shared Time History Viewer: catalog of 1D/2D/3D gauges and a live p(t) plot."""
from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
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
        splitter.splitterMoved.connect(self.canvas.draw_idle)
        self._plot_splitter = splitter
        root.addWidget(splitter)
        self._redraw_plot()

    def _build_left(self) -> QWidget:
        left = QWidget()
        left.setMinimumWidth(COMPUTATIONAL_LEFT_PANEL_MIN)
        layout = QVBoxLayout(left)
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
        self.tbl_gauges.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tbl_gauges, 1)
        return left

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_catalog()
        QTimer.singleShot(0, self.canvas.draw_idle)

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
        axes = self.canvas.axes
        axes.clear()
        fields = self._active_fields() or ["p"]
        plotted = False
        wrap_chars = self._legend_wrap_chars()
        lookup = {row.key: row for row in self._rows}
        all_times: List[float] = []
        all_values: List[float] = []
        for key in self._added:
            row = lookup.get(key)
            if row is None:
                continue
            color = self._color_for_key(key)
            case_dir = self._case_dir_fn(row.dim) or ""
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
                style = "--" if field == "impulse" else "-"
                label = "_nolegend_" if both_fields and field == "impulse" else wrapped
                axes.plot(times, values, color=color, linestyle=style, label=label)
                plotted = True
                all_times.extend(times)
                all_values.extend(v for v in values if v == v)
        only_impulse = fields == ["impulse"]
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Impulse" if only_impulse else "Overpressure (Pa)")
        self._apply_live_limits(axes, all_times, all_values)
        if plotted:
            axes.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                frameon=True,
                fontsize=8,
            )
        axes.grid(True, alpha=0.3)
        self.canvas.draw_idle()

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
