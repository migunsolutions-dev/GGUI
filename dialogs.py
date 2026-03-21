"""
Dialogs for the BlastFoam GUI.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt


def _remap_defaults() -> Dict[str, Any]:
    return {
        "remap_source_type": "1D",
        "remap_case_path": "",
        "remap_origin": (0.0, 0.0, 0.0),
        "remap_time_mode": "latest",
        "remap_specific_time": "1e-4",
    }


class RemapConfigDialog(QDialog):
    """
    Dialog for configuring remap (initialize 3D from 1D/2D pre-cursor).
    Title: "Remap Configuration".
    Sections: Source Type, Dataset Source, Time Selection.
    OK / Cancel; values are read via get_remap_config() after exec() == Accepted.
    """

    def __init__(self, parent: QWidget = None, initial: Dict[str, Any] = None):
        super().__init__(parent)
        self.setWindowTitle("Remap Configuration")
        self._initial = initial or _remap_defaults()
        self._build_ui()
        self._load_initial()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Section 1: Source Type ---
        grp_source = QGroupBox("Source Type")
        f1 = QFormLayout(grp_source)
        self.rad_1d = QRadioButton("Spherical 1D")
        self.rad_2d = QRadioButton("Cylindrical 2D")
        self.rad_1d.setChecked(True)
        f1.addRow(self.rad_1d)
        f1.addRow(self.rad_2d)
        layout.addWidget(grp_source)

        # --- Section 2: Dataset Source ---
        grp_dataset = QGroupBox("Dataset Source")
        f2 = QFormLayout(grp_dataset)
        self.le_case_path = QLineEdit()
        self.le_case_path.setPlaceholderText("Path to source case (e.g. C:\\... or \\\\wsl...\\...)")
        self.le_case_path.setToolTip("Select the root folder of the 1D case (containing 0, constant, system).")
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        row_path = QWidget()
        row_path_h = QHBoxLayout(row_path)
        row_path_h.setContentsMargins(0, 0, 0, 0)
        row_path_h.addWidget(self.le_case_path)
        row_path_h.addWidget(self.btn_browse)
        f2.addRow("Source Case Directory (root folder)", row_path)
        layout.addWidget(grp_dataset)

        # --- Remap Origin (radial mapping center) ---
        grp_origin = QGroupBox("Remap Origin")
        f_origin = QFormLayout(grp_origin)
        self.spin_remap_ox = QDoubleSpinBox()
        self.spin_remap_oy = QDoubleSpinBox()
        self.spin_remap_oz = QDoubleSpinBox()
        for s in (self.spin_remap_ox, self.spin_remap_oy, self.spin_remap_oz):
            s.setRange(-1e6, 1e6)
            s.setDecimals(4)
            s.setSingleStep(0.01)
            s.setValue(0.0)
        self.spin_remap_ox.setToolTip("X coordinate of radial mapping origin.")
        self.spin_remap_oy.setToolTip("Y coordinate of radial mapping origin.")
        self.spin_remap_oz.setToolTip("Z coordinate of radial mapping origin.")
        row_origin = QWidget()
        row_origin_h = QHBoxLayout(row_origin)
        row_origin_h.setContentsMargins(0, 0, 0, 0)
        row_origin_h.addWidget(QLabel("X"))
        row_origin_h.addWidget(self.spin_remap_ox)
        row_origin_h.addWidget(QLabel("Y"))
        row_origin_h.addWidget(self.spin_remap_oy)
        row_origin_h.addWidget(QLabel("Z"))
        row_origin_h.addWidget(self.spin_remap_oz)
        f_origin.addRow("Origin (x, y, z)", row_origin)
        layout.addWidget(grp_origin)

        # --- Section 3: Time Selection ---
        grp_time = QGroupBox("Time Selection")
        f3 = QFormLayout(grp_time)
        self.rad_latest = QRadioButton("Use Latest Solved Time")
        self.rad_specific = QRadioButton("Specific Time")
        self.rad_latest.setChecked(True)
        self.rad_specific.toggled.connect(self._on_specific_toggled)
        f3.addRow(self.rad_latest)
        f3.addRow(self.rad_specific)
        self.le_specific_time = QLineEdit()
        self.le_specific_time.setPlaceholderText("e.g. 1e-4 or 0.001")
        self.le_specific_time.setText("1e-4")
        self.le_specific_time.setEnabled(False)
        f3.addRow("Time", self.le_specific_time)
        layout.addWidget(grp_time)

        # --- Buttons ---
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def _load_initial(self) -> None:
        d = self._initial
        st = (d.get("remap_source_type") or "1D").upper()
        self.rad_1d.setChecked(st == "1D")
        self.rad_2d.setChecked(st == "2D")
        self.le_case_path.setText(d.get("remap_case_path") or "")
        origin = d.get("remap_origin") or (0.0, 0.0, 0.0)
        if len(origin) >= 3:
            self.spin_remap_ox.setValue(float(origin[0]))
            self.spin_remap_oy.setValue(float(origin[1]))
            self.spin_remap_oz.setValue(float(origin[2]))
        tm = d.get("remap_time_mode") or "latest"
        self.rad_latest.setChecked(tm == "latest")
        self.rad_specific.setChecked(tm == "specific")
        self.le_specific_time.setText(d.get("remap_specific_time") or "1e-4")
        self.le_specific_time.setEnabled(tm == "specific")

    def _on_specific_toggled(self, checked: bool) -> None:
        self.le_specific_time.setEnabled(checked)

    def _browse(self) -> None:
        start = self.le_case_path.text().strip() or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, "Select Source Case Directory", start)
        if path:
            self.le_case_path.setText(path)

    def get_remap_config(self) -> Dict[str, Any]:
        """Return current remap configuration (call after accept())."""
        return {
            "remap_source_type": "1D" if self.rad_1d.isChecked() else "2D",
            "remap_case_path": self.le_case_path.text().strip(),
            "remap_origin": (self.spin_remap_ox.value(), self.spin_remap_oy.value(), self.spin_remap_oz.value()),
            "remap_time_mode": "latest" if self.rad_latest.isChecked() else "specific",
            "remap_specific_time": self.le_specific_time.text().strip() or "1e-4",
        }
