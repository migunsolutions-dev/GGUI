"""Edit Time History Output Locations: 1D / 2D / 3D gauges."""
from __future__ import annotations

import csv
import json
import math
from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from models_2d import ProbePoint2D
from probes_model import ProbePoint


def _item(text: str) -> QTableWidgetItem:
    cell = QTableWidgetItem(str(text))
    cell.setFlags(cell.flags() | Qt.ItemIsEditable)
    return cell


def _num(table: QTableWidget, row: int, col: int, default: float = 0.0) -> float:
    item = table.item(row, col)
    if item is None:
        return default
    try:
        return float(item.text().strip())
    except (TypeError, ValueError):
        return default


def _text(table: QTableWidget, row: int, col: int, default: str = "") -> str:
    item = table.item(row, col)
    if item is None:
        return default
    return item.text().strip() or default


def cylindrical_rt(x: float, y: float) -> Tuple[float, float]:
    radius = math.hypot(x, y)
    theta = math.degrees(math.atan2(y, x)) if radius > 0.0 else 0.0
    return radius, theta


def xy_from_rt(radius: float, theta_deg: float) -> Tuple[float, float]:
    rad = math.radians(theta_deg)
    return radius * math.cos(rad), radius * math.sin(rad)


def parse_location_file(path: str, kind: str) -> List[dict]:
    """Parse JSON or CSV/TSV gauge files into dict rows."""
    with open(path, encoding="utf-8") as handle:
        raw = handle.read().strip()
    if not raw:
        return []
    if path.lower().endswith(".json") or raw[:1] in "[{":
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("probes") or data.get("locations") or []
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of locations")
        return [item if isinstance(item, dict) else {} for item in data]
    rows: List[dict] = []
    reader = csv.reader(raw.splitlines())
    lines = list(reader)
    if not lines:
        return []
    header = [cell.strip().lower() for cell in lines[0]]
    has_header = any(
        token in header for token in ("radius", "height", "label", "name", "x", "y", "z")
    )
    start = 1 if has_header else 0
    for line in lines[start:]:
        if not line or all(not cell.strip() for cell in line):
            continue
        if has_header:
            row = {header[i]: line[i].strip() for i in range(min(len(header), len(line)))}
        elif kind == "2d":
            row = {"radius": line[0], "height": line[1] if len(line) > 1 else "0", "label": line[2] if len(line) > 2 else ""}
        else:
            row = {
                "x": line[0],
                "y": line[1] if len(line) > 1 else "0",
                "z": line[2] if len(line) > 2 else "0",
                "label": line[3] if len(line) > 3 else "",
            }
        rows.append(row)
    return rows


class TimeHistoryLocationsDialog(QDialog):
    """Toolbar Time History Locations: gauges for 1D, 2D, and 3D."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        gauges_1d: Sequence[Tuple[float, str]] = (),
        probes_2d: Sequence[ProbePoint2D] = (),
        probes_3d: Sequence[ProbePoint] = (),
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit Time History Output Locations")
        self.setObjectName("dlgTimeHistoryLocations")
        self.resize(720, 420)
        self._block_3d = False
        self._build()
        self._load_1d(gauges_1d)
        self._load_2d(probes_2d)
        self._load_3d(probes_3d)

    def accept(self) -> None:
        """Reject malformed or non-finite coordinates before mutating the model."""
        numeric_columns = (
            (self.tbl_1d, (0,), "1D"),
            (self.tbl_2d, (0, 1), "2D"),
            (self.tbl_3d, (0, 1, 2, 3, 4), "3D"),
        )
        for table, columns, dimension in numeric_columns:
            for row in range(table.rowCount()):
                for col in columns:
                    item = table.item(row, col)
                    text = item.text().strip() if item is not None else ""
                    try:
                        value = float(text)
                    except (TypeError, ValueError):
                        value = math.nan
                    if not math.isfinite(value):
                        if item is not None:
                            table.setCurrentItem(item)
                        self.tabs.setCurrentWidget(table.parentWidget())
                        QMessageBox.warning(
                            self,
                            "Invalid location",
                            f"{dimension} row {row + 1} contains an invalid numeric value.",
                        )
                        return
        super().accept()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("tabsTimeHistoryLocations")
        self.tabs.addTab(self._build_1d(), "1D")
        self.tabs.addTab(self._build_2d(), "2D")
        self.tabs.addTab(self._build_3d(), "3D")
        layout.addWidget(self.tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons, 0, Qt.AlignRight)

    def _page(self, table: QTableWidget, button_widgets: Sequence[QWidget]) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        row.addWidget(table, 1)
        col = QVBoxLayout()
        for widget in button_widgets:
            if widget is None:
                col.addSpacing(12)
            else:
                col.addWidget(widget)
        col.addStretch()
        row.addLayout(col)
        return page

    def _build_1d(self) -> QWidget:
        self.tbl_1d = QTableWidget(0, 2)
        self.tbl_1d.setObjectName("tblTimeHistory1d")
        self.tbl_1d.setHorizontalHeaderLabels(["Radius", "Label"])
        btn_add = QPushButton("Add")
        btn_add.setObjectName("btnTimeHistory1dAdd")
        btn_del = QPushButton("Delete")
        btn_del.setObjectName("btnTimeHistory1dDelete")
        btn_add.clicked.connect(self._add_1d)
        btn_del.clicked.connect(lambda: self._delete_row(self.tbl_1d))
        return self._page(self.tbl_1d, (btn_add, btn_del))

    def _build_2d(self) -> QWidget:
        self.tbl_2d = QTableWidget(0, 3)
        self.tbl_2d.setObjectName("tblTimeHistory2d")
        self.tbl_2d.setHorizontalHeaderLabels(["Radius", "Height", "Label"])
        btn_add = QPushButton("Add")
        btn_add.setObjectName("btnTimeHistory2dAdd")
        btn_imp = QPushButton("Import...")
        btn_imp.setObjectName("btnTimeHistory2dImport")
        btn_del = QPushButton("Delete")
        btn_del.setObjectName("btnTimeHistory2dDelete")
        btn_add.clicked.connect(self._add_2d)
        btn_imp.clicked.connect(lambda: self._import_into("2d"))
        btn_del.clicked.connect(lambda: self._delete_row(self.tbl_2d))
        return self._page(self.tbl_2d, (btn_add, btn_imp, None, btn_del))

    def _build_3d(self) -> QWidget:
        self.tbl_3d = QTableWidget(0, 6)
        self.tbl_3d.setObjectName("tblTimeHistory3d")
        self.tbl_3d.setHorizontalHeaderLabels(["X", "Y", "Z", "R", "T", "Label"])
        self.tbl_3d.itemChanged.connect(self._on_3d_item_changed)
        btn_add = QPushButton("Add")
        btn_add.setObjectName("btnTimeHistory3dAdd")
        btn_imp = QPushButton("Import...")
        btn_imp.setObjectName("btnTimeHistory3dImport")
        btn_del = QPushButton("Delete")
        btn_del.setObjectName("btnTimeHistory3dDelete")
        btn_set_remap = QPushButton("Set remap")
        btn_unset_remap = QPushButton("Unset remap")
        btn_set_term = QPushButton("Set terminate")
        btn_unset_term = QPushButton("Unset terminate")
        btn_add.clicked.connect(self._add_3d)
        btn_imp.clicked.connect(lambda: self._import_into("3d"))
        btn_del.clicked.connect(lambda: self._delete_row(self.tbl_3d))
        btn_set_remap.clicked.connect(lambda: self._set_3d_flag("remap", True))
        btn_unset_remap.clicked.connect(lambda: self._set_3d_flag("remap", False))
        btn_set_term.clicked.connect(lambda: self._set_3d_flag("terminate", True))
        btn_unset_term.clicked.connect(lambda: self._set_3d_flag("terminate", False))
        return self._page(
            self.tbl_3d,
            (btn_add, btn_imp, None, btn_del, btn_set_remap, btn_unset_remap, btn_set_term, btn_unset_term),
        )

    def _add_1d(self) -> None:
        row = self.tbl_1d.rowCount()
        self.tbl_1d.insertRow(row)
        self.tbl_1d.setItem(row, 0, _item("0"))
        self.tbl_1d.setItem(row, 1, _item(f"G{row + 1}"))
        self.tbl_1d.selectRow(row)

    def _add_2d(self) -> None:
        row = self.tbl_2d.rowCount()
        self.tbl_2d.insertRow(row)
        self.tbl_2d.setItem(row, 0, _item("0"))
        self.tbl_2d.setItem(row, 1, _item("0"))
        self.tbl_2d.setItem(row, 2, _item(f"P{row + 1}"))
        self.tbl_2d.selectRow(row)

    def _add_3d(self) -> None:
        row = self.tbl_3d.rowCount()
        self._block_3d = True
        try:
            self.tbl_3d.insertRow(row)
            for col, value in enumerate(("0", "0", "0", "0", "0", f"P{row + 1}")):
                self.tbl_3d.setItem(row, col, _item(value))
            self._paint_3d_row(row)
        finally:
            self._block_3d = False
        self.tbl_3d.selectRow(row)

    def _delete_row(self, table: QTableWidget) -> None:
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _import_into(self, kind: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import locations",
            "",
            "Location files (*.csv *.txt *.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            rows = parse_location_file(path, kind)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Import", f"Could not read locations:\n{exc}")
            return
        if kind == "2d":
            for item in rows:
                row = self.tbl_2d.rowCount()
                self.tbl_2d.insertRow(row)
                radius = item.get("radius") or item.get("r") or "0"
                height = item.get("height") or item.get("z") or "0"
                label = item.get("label") or item.get("name") or f"P{row + 1}"
                self.tbl_2d.setItem(row, 0, _item(radius))
                self.tbl_2d.setItem(row, 1, _item(height))
                self.tbl_2d.setItem(row, 2, _item(label))
        else:
            for item in rows:
                x = float(item.get("x", 0) or 0)
                y = float(item.get("y", 0) or 0)
                z = float(item.get("z", 0) or 0)
                if "r" in item and item.get("r") not in ("", None) and "x" not in item:
                    radius = float(item.get("r") or 0)
                    theta = float(item.get("t") or item.get("theta") or 0)
                    x, y = xy_from_rt(radius, theta)
                label = str(item.get("label") or item.get("name") or f"P{self.tbl_3d.rowCount() + 1}")
                self._append_3d(x, y, z, label, bool(item.get("remap")), bool(item.get("terminate")))

    def _append_3d(self, x: float, y: float, z: float, label: str, remap: bool = False, terminate: bool = False) -> None:
        row = self.tbl_3d.rowCount()
        radius, theta = cylindrical_rt(x, y)
        self._block_3d = True
        try:
            self.tbl_3d.insertRow(row)
            values = (x, y, z, radius, theta, label)
            for col, value in enumerate(values):
                text = value if col == 5 else f"{float(value):.6g}" if col < 5 else value
                if col == 5:
                    text = str(value)
                self.tbl_3d.setItem(row, col, _item(text))
            self.tbl_3d.item(row, 0).setData(Qt.UserRole, {"remap": remap, "terminate": terminate})
            self._paint_3d_row(row)
        finally:
            self._block_3d = False

    def _flags(self, row: int) -> dict:
        item = self.tbl_3d.item(row, 0)
        data = item.data(Qt.UserRole) if item is not None else None
        if not isinstance(data, dict):
            return {"remap": False, "terminate": False}
        return {"remap": bool(data.get("remap")), "terminate": bool(data.get("terminate"))}

    def _set_3d_flag(self, key: str, value: bool) -> None:
        row = self.tbl_3d.currentRow()
        if row < 0:
            return
        flags = self._flags(row)
        if key == "remap" and value:
            for other in range(self.tbl_3d.rowCount()):
                other_flags = self._flags(other)
                other_flags["remap"] = False
                if self.tbl_3d.item(other, 0) is not None:
                    self.tbl_3d.item(other, 0).setData(Qt.UserRole, other_flags)
                self._paint_3d_row(other)
        flags[key] = value
        if self.tbl_3d.item(row, 0) is None:
            self.tbl_3d.setItem(row, 0, _item("0"))
        self.tbl_3d.item(row, 0).setData(Qt.UserRole, flags)
        self._paint_3d_row(row)

    def _paint_3d_row(self, row: int) -> None:
        flags = self._flags(row)
        if flags["remap"] and flags["terminate"]:
            color = QColor("#f5d0a9")
        elif flags["remap"]:
            color = QColor("#d6eaf8")
        elif flags["terminate"]:
            color = QColor("#f5b7b1")
        else:
            color = QColor(Qt.white)
        brush = QBrush(color)
        for col in range(self.tbl_3d.columnCount()):
            item = self.tbl_3d.item(row, col)
            if item is not None:
                item.setBackground(brush)

    def _on_3d_item_changed(self, item: QTableWidgetItem) -> None:
        if self._block_3d or item is None:
            return
        row, col = item.row(), item.column()
        self._block_3d = True
        try:
            if col in (0, 1, 2):
                x, y = _num(self.tbl_3d, row, 0), _num(self.tbl_3d, row, 1)
                radius, theta = cylindrical_rt(x, y)
                self.tbl_3d.setItem(row, 3, _item(f"{radius:.6g}"))
                self.tbl_3d.setItem(row, 4, _item(f"{theta:.6g}"))
            elif col in (3, 4):
                radius, theta = _num(self.tbl_3d, row, 3), _num(self.tbl_3d, row, 4)
                x, y = xy_from_rt(radius, theta)
                flags = self._flags(row)
                x_item = _item(f"{x:.6g}")
                x_item.setData(Qt.UserRole, flags)
                self.tbl_3d.setItem(row, 0, x_item)
                self.tbl_3d.setItem(row, 1, _item(f"{y:.6g}"))
            self._paint_3d_row(row)
        finally:
            self._block_3d = False

    def _load_1d(self, gauges: Sequence[Tuple[float, str]]) -> None:
        self.tbl_1d.setRowCount(0)
        for radius, label in gauges:
            row = self.tbl_1d.rowCount()
            self.tbl_1d.insertRow(row)
            self.tbl_1d.setItem(row, 0, _item(f"{float(radius):.6g}"))
            self.tbl_1d.setItem(row, 1, _item(label))

    def _load_2d(self, probes: Sequence[ProbePoint2D]) -> None:
        self.tbl_2d.setRowCount(0)
        for probe in probes:
            row = self.tbl_2d.rowCount()
            self.tbl_2d.insertRow(row)
            self.tbl_2d.setItem(row, 0, _item(f"{float(probe.radius):.6g}"))
            self.tbl_2d.setItem(row, 1, _item(f"{float(probe.height):.6g}"))
            self.tbl_2d.setItem(row, 2, _item(probe.name))

    def _load_3d(self, probes: Sequence[ProbePoint]) -> None:
        self.tbl_3d.setRowCount(0)
        for probe in probes:
            self._append_3d(probe.x, probe.y, probe.z, probe.name, probe.remap, probe.terminate)

    def gauges_1d(self) -> Tuple[Tuple[float, str], ...]:
        out = []
        for row in range(self.tbl_1d.rowCount()):
            out.append((_num(self.tbl_1d, row, 0), _text(self.tbl_1d, row, 1, f"G{row + 1}")))
        return tuple(out)

    def probes_2d(self) -> Tuple[ProbePoint2D, ...]:
        out = []
        for row in range(self.tbl_2d.rowCount()):
            out.append(
                ProbePoint2D(
                    name=_text(self.tbl_2d, row, 2, f"P{row + 1}"),
                    radius=_num(self.tbl_2d, row, 0),
                    height=_num(self.tbl_2d, row, 1),
                )
            )
        return tuple(out)

    def probes_3d(self) -> Tuple[ProbePoint, ...]:
        out = []
        for row in range(self.tbl_3d.rowCount()):
            flags = self._flags(row)
            out.append(
                ProbePoint(
                    name=_text(self.tbl_3d, row, 5, f"P{row + 1}"),
                    x=_num(self.tbl_3d, row, 0),
                    y=_num(self.tbl_3d, row, 1),
                    z=_num(self.tbl_3d, row, 2),
                    remap=flags["remap"],
                    terminate=flags["terminate"],
                )
            )
        return tuple(out)

    def remap_origin(self) -> Optional[Tuple[float, float, float]]:
        for probe in self.probes_3d():
            if probe.remap:
                return (probe.x, probe.y, probe.z)
        return None
