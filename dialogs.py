"""
Dialogs for the BlastFoam GUI.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt

from output_options import (
    GAUGE_LABELS_1D,
    GAUGE_LABELS_2D,
    VTK_KEYS_2D,
    Dim1DOutput,
    Dim2DOutput,
    Dim3DOutput,
    GaugeFlags,
    OutputFileOptions,
)


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


class RemapFromDialog(QDialog):
    """Choose whether 2D remap uses the current 1D model or a results file."""

    CURRENT_1D = "current_1d"
    FILE_1D = "file_1d"
    FILE_2D = "file_2d"

    def __init__(
        self,
        parent: QWidget = None,
        *,
        current_kind: str = CURRENT_1D,
        has_current_1d: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle("Remap from")
        self.setModal(True)
        layout = QVBoxLayout(self)
        title = QLabel("Remap from:")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self.rad_current_1d = QRadioButton("Current 1D model")
        self.rad_file_1d = QRadioButton("1D results file")
        self.rad_file_2d = QRadioButton("2D results file")
        self.rad_current_1d.setEnabled(bool(has_current_1d))
        for radio in (self.rad_current_1d, self.rad_file_1d, self.rad_file_2d):
            layout.addWidget(radio)

        kind = current_kind if current_kind in (
            self.CURRENT_1D, self.FILE_1D, self.FILE_2D
        ) else self.CURRENT_1D
        if kind == self.CURRENT_1D and not has_current_1d:
            kind = self.FILE_1D
        self.rad_current_1d.setChecked(kind == self.CURRENT_1D)
        self.rad_file_1d.setChecked(kind == self.FILE_1D)
        self.rad_file_2d.setChecked(kind == self.FILE_2D)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_kind(self) -> str:
        if self.rad_file_1d.isChecked():
            return self.FILE_1D
        if self.rad_file_2d.isChecked():
            return self.FILE_2D
        return self.CURRENT_1D


class _TimeStepRate(QWidget):
    """Time / Step radios with matching values, like Viper VTK framerate rows."""

    def __init__(
        self,
        parent: QWidget = None,
        *,
        by_time: bool = True,
        time_s: float = 0.001,
        steps: int = 25,
    ):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.rad_time = QRadioButton("Time")
        self.rad_step = QRadioButton("Step")
        self.group = QButtonGroup(self)
        self.group.addButton(self.rad_time)
        self.group.addButton(self.rad_step)
        self.spin_time = QDoubleSpinBox()
        self.spin_time.setDecimals(6)
        self.spin_time.setRange(1e-12, 1e6)
        self.spin_time.setValue(float(time_s))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 1_000_000)
        self.spin_steps.setValue(int(steps))
        layout.addWidget(self.rad_time, 0, 0)
        layout.addWidget(self.rad_step, 0, 1)
        layout.addWidget(self.spin_time, 1, 0)
        layout.addWidget(self.spin_steps, 1, 1)
        self.rad_time.toggled.connect(self._sync_enabled)
        self.rad_time.setChecked(bool(by_time))
        self.rad_step.setChecked(not bool(by_time))
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        by_time = self.rad_time.isChecked()
        self.spin_time.setEnabled(by_time)
        self.spin_steps.setEnabled(not by_time)

    def set_enabled_rate(self, on: bool) -> None:
        self.rad_time.setEnabled(on)
        self.rad_step.setEnabled(on)
        if on:
            self._sync_enabled()
        else:
            self.spin_time.setEnabled(False)
            self.spin_steps.setEnabled(False)

    def by_time(self) -> bool:
        return self.rad_time.isChecked()

    def time_s(self) -> float:
        return float(self.spin_time.value())

    def steps(self) -> int:
        return int(self.spin_steps.value())


def _flags_from_checks(checks: Dict[str, QCheckBox], extra: Optional[Dict[str, bool]] = None) -> GaugeFlags:
    values = {key: box.isChecked() for key, box in checks.items()}
    if extra:
        values.update(extra)
    return GaugeFlags(
        overpressure=values.get("overpressure", values.get("pressure", False)),
        pressure=values.get("pressure", values.get("overpressure", False)),
        impulse=values.get("impulse", False),
        density=values.get("density", False),
        velocity=values.get("velocity", False),
        mass_fractions=values.get("mass_fractions", False),
        temperature=values.get("temperature", False),
        energy=values.get("energy", False),
        dynamic_pressure=values.get("dynamic_pressure", False),
        peak_overpressure=values.get("peak_overpressure", False),
        peak_impulse=values.get("peak_impulse", False),
    )


class OutputFileOptionsDialog(QDialog):
    """Toolbar Output Options: 1D / 2D / 3D gauges and VTK write cadence."""

    def __init__(self, parent: QWidget = None, initial: Optional[OutputFileOptions] = None):
        super().__init__(parent)
        self.setWindowTitle("Output File Options")
        self.setObjectName("dlgOutputFileOptions")
        self._initial = initial or OutputFileOptions()
        self._gauges_1d: Dict[str, QCheckBox] = {}
        self._gauges_2d: Dict[str, QCheckBox] = {}
        self._vtk_2d: Dict[str, QCheckBox] = {}
        self._gauges_3d: Dict[str, QCheckBox] = {}
        self._build()
        self._load(self._initial)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("tabsOutputFileOptions")
        self.tabs.addTab(self._build_1d(), "1D")
        self.tabs.addTab(self._build_2d(), "2D")
        self.tabs.addTab(self._build_3d(), "3D")
        layout.addWidget(self.tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons, 0, Qt.AlignRight)

    def _build_1d(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Output quantities")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        grid = QGridLayout()
        hdr = QLabel("Gauges")
        hdr.setStyleSheet("font-weight: bold;")
        hdr.setAlignment(Qt.AlignCenter)
        grid.addWidget(hdr, 0, 1)
        for row, (key, label) in enumerate(GAUGE_LABELS_1D, start=1):
            grid.addWidget(QLabel(label), row, 0)
            box = QCheckBox()
            box.setObjectName(f"chk1dGauge_{key}")
            self._gauges_1d[key] = box
            grid.addWidget(box, row, 1, Qt.AlignCenter)
        layout.addLayout(grid)
        layout.addStretch()
        return page

    def _build_2d(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        vtk_box = QGroupBox("Whole domain VTK outputs and framerates")
        vtk_lay = QVBoxLayout(vtk_box)
        self.rate_2d = _TimeStepRate(self)
        vtk_lay.addWidget(self.rate_2d)
        layout.addWidget(vtk_box)

        title = QLabel("Output quantities")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        grid = QGridLayout()
        for col, text in enumerate(("Gauges", "Whole Domain VTKs"), start=1):
            hdr = QLabel(text)
            hdr.setStyleSheet("font-weight: bold;")
            hdr.setAlignment(Qt.AlignCenter)
            grid.addWidget(hdr, 0, col)
        for row, (key, label) in enumerate(GAUGE_LABELS_2D, start=1):
            grid.addWidget(QLabel(label), row, 0)
            g = QCheckBox()
            g.setObjectName(f"chk2dGauge_{key}")
            self._gauges_2d[key] = g
            grid.addWidget(g, row, 1, Qt.AlignCenter)
            if key in VTK_KEYS_2D and key != "dynamic_pressure":
                v = QCheckBox()
                v.setObjectName(f"chk2dVtk_{key}")
                self._vtk_2d[key] = v
                grid.addWidget(v, row, 2, Qt.AlignCenter)
        layout.addLayout(grid)
        layout.addStretch()
        return page

    def _rate_row(self, checkbox: QCheckBox, rate: _TimeStepRate) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(checkbox)
        lay.addWidget(rate, 1)
        return row

    def _build_3d(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        vtk_box = QGroupBox("VTKs and Framerates")
        form = QFormLayout(vtk_box)
        self.chk_surfaces = QCheckBox("Cross-sections and surfaces")
        self.chk_surfaces.setObjectName("chk3dSurfaces")
        self.rate_3d_surfaces = _TimeStepRate(self)
        self.chk_surfaces.toggled.connect(self.rate_3d_surfaces.set_enabled_rate)
        form.addRow(self._rate_row(self.chk_surfaces, self.rate_3d_surfaces))
        self.chk_volumes = QCheckBox("Volumes")
        self.chk_volumes.setObjectName("chk3dVolumes")
        self.rate_3d = _TimeStepRate(self)
        self.chk_volumes.toggled.connect(self.rate_3d.set_enabled_rate)
        form.addRow(self._rate_row(self.chk_volumes, self.rate_3d))
        layout.addWidget(vtk_box)

        title = QLabel("Output quantities")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        grid = QGridLayout()
        hdr = QLabel("Gauges")
        hdr.setStyleSheet("font-weight: bold;")
        hdr.setAlignment(Qt.AlignCenter)
        grid.addWidget(hdr, 0, 1)
        for row, (key, label) in enumerate(GAUGE_LABELS_2D, start=1):
            grid.addWidget(QLabel(label), row, 0)
            box = QCheckBox()
            box.setObjectName(f"chk3dGauge_{key}")
            self._gauges_3d[key] = box
            grid.addWidget(box, row, 1, Qt.AlignCenter)
        peak_row = len(GAUGE_LABELS_2D) + 1
        grid.addWidget(QLabel("Peak overpressure"), peak_row, 0)
        self.chk_peak_over = QCheckBox()
        self.chk_peak_over.setObjectName("chk3dPeakOverpressure")
        grid.addWidget(self.chk_peak_over, peak_row, 1, Qt.AlignCenter)
        grid.addWidget(QLabel("Peak impulse"), peak_row + 1, 0)
        self.chk_peak_imp = QCheckBox()
        self.chk_peak_imp.setObjectName("chk3dPeakImpulse")
        grid.addWidget(self.chk_peak_imp, peak_row + 1, 1, Qt.AlignCenter)
        layout.addLayout(grid)
        layout.addStretch()
        return page

    def _load(self, opts: OutputFileOptions) -> None:
        g1 = opts.dim1d.gauges
        mapping_1d = {
            "overpressure": g1.overpressure,
            "impulse": g1.impulse,
            "density": g1.density,
            "velocity": g1.velocity,
            "mass_fractions": g1.mass_fractions,
            "temperature": g1.temperature,
            "energy": g1.energy,
            "dynamic_pressure": g1.dynamic_pressure,
        }
        for key, box in self._gauges_1d.items():
            box.setChecked(bool(mapping_1d.get(key, False)))

        d2 = opts.dim2d
        self.rate_2d.rad_time.setChecked(d2.vtk_by_time)
        self.rate_2d.rad_step.setChecked(not d2.vtk_by_time)
        self.rate_2d.spin_time.setValue(d2.vtk_time_s)
        self.rate_2d.spin_steps.setValue(d2.vtk_steps)
        self.rate_2d._sync_enabled()
        g2 = d2.gauges
        mapping_2d = {
            "pressure": g2.pressure,
            "impulse": g2.impulse,
            "density": g2.density,
            "velocity": g2.velocity,
            "mass_fractions": g2.mass_fractions,
            "temperature": g2.temperature,
            "energy": g2.energy,
            "dynamic_pressure": g2.dynamic_pressure,
        }
        for key, box in self._gauges_2d.items():
            box.setChecked(bool(mapping_2d.get(key, False)))
        v2 = d2.vtk
        vtk_map = {
            "pressure": v2.pressure,
            "impulse": v2.impulse,
            "density": v2.density,
            "velocity": v2.velocity,
            "mass_fractions": v2.mass_fractions,
            "temperature": v2.temperature,
            "energy": v2.energy,
        }
        for key, box in self._vtk_2d.items():
            box.setChecked(bool(vtk_map.get(key, False)))

        d3 = opts.dim3d
        self.chk_surfaces.setChecked(d3.write_surfaces)
        self.rate_3d_surfaces.rad_time.setChecked(d3.surface_by_time)
        self.rate_3d_surfaces.rad_step.setChecked(not d3.surface_by_time)
        self.rate_3d_surfaces.spin_time.setValue(d3.surface_time_s)
        self.rate_3d_surfaces.spin_steps.setValue(d3.surface_steps)
        self.rate_3d_surfaces.set_enabled_rate(d3.write_surfaces)
        self.chk_volumes.setChecked(d3.write_volumes)
        self.rate_3d.rad_time.setChecked(d3.vtk_by_time)
        self.rate_3d.rad_step.setChecked(not d3.vtk_by_time)
        self.rate_3d.spin_time.setValue(d3.vtk_time_s)
        self.rate_3d.spin_steps.setValue(d3.vtk_steps)
        self.rate_3d.set_enabled_rate(d3.write_volumes)
        g3 = d3.gauges
        mapping_3d = {
            "pressure": g3.pressure,
            "impulse": g3.impulse,
            "density": g3.density,
            "velocity": g3.velocity,
            "mass_fractions": g3.mass_fractions,
            "temperature": g3.temperature,
            "energy": g3.energy,
            "dynamic_pressure": g3.dynamic_pressure,
        }
        for key, box in self._gauges_3d.items():
            box.setChecked(bool(mapping_3d.get(key, False)))
        self.chk_peak_over.setChecked(d3.peak_overpressure)
        self.chk_peak_imp.setChecked(d3.peak_impulse)

    def get_options(self) -> OutputFileOptions:
        g1 = _flags_from_checks(self._gauges_1d)
        g2 = _flags_from_checks(self._gauges_2d)
        v2 = _flags_from_checks(self._vtk_2d)
        g3 = _flags_from_checks(self._gauges_3d)
        return OutputFileOptions(
            dim1d=Dim1DOutput(gauges=g1),
            dim2d=Dim2DOutput(
                vtk_by_time=self.rate_2d.by_time(),
                vtk_time_s=self.rate_2d.time_s(),
                vtk_steps=self.rate_2d.steps(),
                gauges=g2,
                vtk=v2,
            ),
            dim3d=Dim3DOutput(
                write_surfaces=self.chk_surfaces.isChecked(),
                surface_by_time=self.rate_3d_surfaces.by_time(),
                surface_time_s=self.rate_3d_surfaces.time_s(),
                surface_steps=self.rate_3d_surfaces.steps(),
                write_volumes=self.chk_volumes.isChecked(),
                vtk_by_time=self.rate_3d.by_time(),
                vtk_time_s=self.rate_3d.time_s(),
                vtk_steps=self.rate_3d.steps(),
                gauges=g3,
                peak_overpressure=self.chk_peak_over.isChecked(),
                peak_impulse=self.chk_peak_imp.isChecked(),
            ),
        )

