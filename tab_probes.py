import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QLabel
)
from PyQt5.QtCore import Qt

from probes_model import ProbesModel


class TabProbes(QWidget):
    """
    Separate tab to manage probe points (to keep General 3D clean).
    """
    def __init__(self, probes_model: ProbesModel):
        super().__init__()
        self.model = probes_model
        self._block_table_signal = False

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("Probe / Gauge points (shared with 3D Viewer):"))
        header.addStretch(1)

        self.btn_add = QPushButton("Add")
        self.btn_remove = QPushButton("Remove")
        self.btn_import = QPushButton("Import JSON")
        self.btn_export = QPushButton("Export JSON")
        self.btn_clear = QPushButton("Clear")

        header.addWidget(self.btn_add)
        header.addWidget(self.btn_remove)
        header.addWidget(self.btn_import)
        header.addWidget(self.btn_export)
        header.addWidget(self.btn_clear)
        layout.addLayout(header)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "X [m]", "Y [m]", "Z [m]"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.table)

        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_import.clicked.connect(self._import_json)
        self.btn_export.clicked.connect(self._export_json)
        self.table.itemChanged.connect(self._on_item_changed)

        self.model.changed.connect(self._refresh_from_model)
        self._refresh_from_model()

    def _refresh_from_model(self):
        self._block_table_signal = True
        try:
            probes = self.model.probes()
            self.table.setRowCount(len(probes))
            for r, p in enumerate(probes):
                self.table.setItem(r, 0, QTableWidgetItem(p.name))
                self.table.setItem(r, 1, QTableWidgetItem(f"{p.x:.6g}"))
                self.table.setItem(r, 2, QTableWidgetItem(f"{p.y:.6g}"))
                self.table.setItem(r, 3, QTableWidgetItem(f"{p.z:.6g}"))
        finally:
            self._block_table_signal = False

    def _add_row(self):
        probes = self.model.probes()
        idx = len(probes) + 1
        self.model.add_probe(f"P{idx}", 0.0, 0.0, 0.0)
        self.table.selectRow(len(self.model.probes()) - 1)

    def _remove_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self.model.remove_probe(row)

    def _clear_all(self):
        if QMessageBox.question(self, "Clear probes", "Remove all probe points?") == QMessageBox.Yes:
            self.model.clear()

    def _on_item_changed(self, item):
        if self._block_table_signal:
            return
        row = item.row()
        col = item.column()
        text = item.text().strip()

        if col == 0:
            self.model.update_probe(row, name=text)
            return

        try:
            val = float(text)
        except ValueError:
            return

        if col == 1:
            self.model.update_probe(row, x=val)
        elif col == 2:
            self.model.update_probe(row, y=val)
        elif col == 3:
            self.model.update_probe(row, z=val)

    def _import_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import probes JSON", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.model.load_dict(data)
        except Exception as e:
            QMessageBox.critical(self, "Import failed", f"Could not import JSON:\n{e}")

    def _export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export probes JSON", "probes.json", "JSON files (*.json)")
        if not path:
            return
        try:
            data = self.model.to_dict()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Could not export JSON:\n{e}")
