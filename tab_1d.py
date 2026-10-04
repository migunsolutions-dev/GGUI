import math
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton, QFrame,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QRadioButton, QButtonGroup, QSplitter, QScrollArea, QSizePolicy, QTabWidget
)
from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from models import (
    BOUNDARY_1D_REFLECT,
    BOUNDARY_1D_TERMINATE,
    CaseInputs1D,
    RUN_MODE_REFLECT,
    RUN_MODE_TERMINATE,
    SOURCE_MODEL_IG,
    SOURCE_MODEL_JWL,
    SOURCE_MODEL_LABELS,
    is_ideal_gas_source,
    normalize_source_model,
)
import ig_source_state as igs
from completion_1d import normalize_run_mode
from ui_metrics import (
    COMPUTATIONAL_LEFT_PANEL_WIDTH,
    COMPUTATIONAL_LEFT_PANEL_MIN,
    EXECUTION_AREA_MIN_HEIGHT,
    EXECUTION_AREA_PREFERRED_HEIGHT,
    ACTION_BUTTON_FONT_PT,
    INFO_ROW_STYLE,
    INFO_TITLE_STYLE,
    SECONDARY_INFO_STYLE,
    WARNING_STYLE,
)

TIP_RUN_TERMINATE = (
    "Stop when the wave reaches the radius; End Time is the upper bound."
)
TIP_RUN_REFLECT = (
    "Reflect at the radius and run until End Time or manual stop."
)
TIP_END_TIME = (
    "Written to controlDict as endTime. In Terminate mode this is an upper bound; "
    "in Reflect mode the run completes at this time."
)

# Shared 1D numeric boxes are 100px. Domain radius and cell size are 1.5x.
_DOMAIN_SPIN_WIDTH = 150
# A double keeps 15 decimal digits. That is the precision this spin can store.
_RADIUS_DECIMALS = 15


class _RadiusSpinBox(QDoubleSpinBox):
    """Radius entry: keep typed digits instead of padding to a fixed 2 places."""

    def textFromValue(self, value: float) -> str:
        text = f"{float(value):.{_RADIUS_DECIMALS}f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def valueFromText(self, text: str) -> float:
        cleaned = str(text).strip().replace(" ", "")
        if not cleaned:
            return 0.0
        try:
            return float(cleaned)
        except ValueError:
            return float(super().valueFromText(text))


def spherical_charge_radius_m(mass_kg: float, rho_kg_m3: float) -> float:
    """Equivalent sphere radius for a given charge mass and density."""
    rho = max(float(rho_kg_m3), 1e-12)
    mass = max(float(mass_kg), 0.0)
    return ((3.0 * mass) / (4.0 * math.pi * rho)) ** (1.0 / 3.0)


# Ideal-gas sketch of the charge, same γ as the air phase. Peak tracks ρ and energy.
INITIAL_CHARGE_GAMMA = 1.4


def ideal_gas_charge_pressure_pa(
    rho_kg_m3: float,
    energy_j_per_kg: float,
    gamma: float = INITIAL_CHARGE_GAMMA,
) -> float:
    """P = (γ-1) ρ e for the pre-run overpressure step."""
    return max(float(gamma) - 1.0, 0.0) * max(float(rho_kg_m3), 0.0) * max(float(energy_j_per_kg), 0.0)


def initial_radial_step(domain_radius_m: float, charge_radius_m: float, inside: float, outside: float):
    """Step profile: one value inside the charge radius, another outside it."""
    r_max = max(float(domain_radius_m), 1e-9)
    r_c = min(max(float(charge_radius_m), 0.0), r_max)
    return [0.0, r_c, r_c, r_max], [float(inside), float(inside), float(outside), float(outside)]


def initial_overpressure_step(
    domain_radius_m: float,
    charge_radius_m: float,
    charge_pressure_pa: float,
    p_atm: float,
):
    """Step profile: charge pressure inside R_charge, ambient overpressure outside."""
    over = max(float(charge_pressure_pa) - float(p_atm), 0.0)
    return initial_radial_step(domain_radius_m, charge_radius_m, over, 0.0)


def _qimage_from_rgba(rgba: np.ndarray, width: int, height: int) -> QImage:
    """Build a Qt-owned QImage. Never wrap a Python/numpy buffer (that aborts on Windows)."""
    image = QImage(width, height, QImage.Format_RGBA8888)
    if image.isNull():
        return image
    raw = np.ascontiguousarray(rgba, dtype=np.uint8).tobytes()
    expected = width * height * 4
    if len(raw) < expected:
        return QImage()
    bits = image.bits()
    nbytes = image.byteCount() if hasattr(image, "byteCount") else image.sizeInBytes()
    bits.setsize(nbytes)
    bpl = image.bytesPerLine()
    if bpl == width * 4:
        bits[:expected] = raw
    else:
        for y in range(height):
            src = y * width * 4
            dst = y * bpl
            bits[dst : dst + width * 4] = raw[src : src + width * 4]
    return image


class MplCanvas(QLabel):
    """Raster matplotlib figure. Avoids Qt5Agg OpenGL swaps that abort on Windows."""

    def __init__(self, parent=None, width=5, height=4, dpi=100, min_height=120):
        super().__init__(parent)
        self._dpi = float(dpi)
        self._min_height = max(1, int(min_height))
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor="white")
        self.axes = self.figure.add_subplot(111)
        self._agg = FigureCanvasAgg(self.figure)
        self._drawing = False
        self._draw_timer = QTimer(self)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.setInterval(0)
        self._draw_timer.timeout.connect(self._render_to_label)
        self.setMinimumHeight(self._min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.tight_layout_rect = None

    def sizeHint(self):
        return QSize(400, max(300, self._min_height))

    def minimumSizeHint(self):
        return QSize(40, self._min_height)

    def draw(self):
        self._render_to_label()

    def draw_idle(self):
        if not self.isVisible():
            return
        self._render_to_label()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.isVisible() or self._draw_timer.isActive():
            return
        self._draw_timer.start()

    def _render_to_label(self) -> None:
        if self._drawing:
            return
        self._drawing = True
        try:
            area = self.contentsRect()
            w = max(int(area.width()), 1)
            h = max(int(area.height()), 1)
            if w < 40 or h < 40:
                w, h = max(w, 40), max(h, 40)
            self.figure.set_size_inches(w / self._dpi, h / self._dpi, forward=False)
            if w >= 200 and h >= 200:
                try:
                    if self.tight_layout_rect is not None:
                        self.figure.tight_layout(rect=self.tight_layout_rect)
                    else:
                        self.figure.tight_layout()
                except Exception:
                    pass
            self._agg.draw()
            renderer = self._agg.get_renderer()
            buf = np.asarray(renderer.buffer_rgba())
            if buf.ndim != 3 or buf.shape[0] < 1 or buf.shape[1] < 1:
                return
            height, width = int(buf.shape[0]), int(buf.shape[1])
            image = _qimage_from_rgba(np.ascontiguousarray(buf[:, :, :4]), width, height)
            if image is None or image.isNull():
                return
            self.setPixmap(QPixmap.fromImage(image))
        except Exception:
            pass
        finally:
            self._drawing = False


class Tab1D(QWidget):
    # --- הוספה: סיגנלים לתקשורת עם Main ---
    sig_request_init = pyqtSignal(object)
    sig_request_run = pyqtSignal()
    sig_request_stop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.calculated_adj_rho = 0.0
        self.calculated_discrete_radius = 0.0
        
        self.last_r_min = None
        self.last_r_max = None
        self._live_graph = False
        self._has_run_profile = False
        self._pending_pressures = None
        self._pending_time_s = 0.0
        self._pmax_over = None
        self._graph_timer = QTimer(self)
        self._graph_timer.setSingleShot(True)
        self._graph_timer.setInterval(50)
        self._graph_timer.timeout.connect(self._redraw_canvas)
        self._probe_fields = ("p", "impulse")
        self._enable_impulse = True
        self._enable_dynamic_pressure = False
        self._gauge_locations = ()
        self._last_domain_radius = 1.0
        self._last_run_case_dir = ""

        self.setup_ui()
        self._last_domain_radius = float(self.spin_radius.value())
        self.recalc_stats()

    # --- הוספה: פונקציה שה-Main דורש (מותאמת למשתנים שלך) ---
    def get_case_inputs(self) -> CaseInputs1D:
        """אוספת את כל הנתונים מהממשק ומחזירה אובייקט מסודר"""
        
        rho_final = float(self.spin_density.value())
        mat_props = dict(self.get_selected_material_properties())
        mat_props["rho"] = rho_final
        
        return CaseInputs1D(
            radius=self.spin_radius.value(),
            cell_size=self.spin_cellsize.value(),
            p_atm=self.spin_press.value(),     # השם המקורי שלך
            t_atm=self.spin_temp.value(),      # השם המקורי שלך (תיקון השגיאה)
            mass_kg=self.spin_mass.value(),
            rho_charge=rho_final,
            energy_j_per_kg=float(self.edit_energy.text()),
            material_props=mat_props,
            max_cfl=self.spin_cfl.value(),
            end_time_s=self.spin_endtime.value(),
            # ברירות מחדל קבועות (כי אין להן שדות ב-UI המקורי)
            write_interval_s=0.0,
            n_probes=200,
            probe_write_interval_steps=int(self.spin_gui_refresh.value()),
            wedge_angle_deg=15.0,
            cone_half_angle_deg=12.0,
            axis_epsilon=0.10,
            right_boundary=(
                BOUNDARY_1D_REFLECT
                if self._right_boundary_is_reflect()
                else BOUNDARY_1D_TERMINATE
            ),
            probe_fields=tuple(getattr(self, "_probe_fields", ("p", "impulse"))),
            enable_impulse=bool(getattr(self, "_enable_impulse", True)),
            enable_dynamic_pressure=bool(getattr(self, "_enable_dynamic_pressure", False)),
            gauge_locations=tuple(getattr(self, "_gauge_locations", ()) or ()),
            material_name=self.combo_comp.currentText(),
            stop_mode=(
                RUN_MODE_REFLECT
                if self._right_boundary_is_reflect()
                else RUN_MODE_TERMINATE
            ),
            stop_radius_m=float(self.spin_radius.value()),
            remap_for_2d=bool(self.radio_yes.isChecked()),
            source_model=self.selected_source_model(),
        )

    def set_case_inputs(self, data: dict) -> None:
        """Restore persisted 1D inputs without changing control semantics."""
        values = dict(data)
        material_name = str(values.get("material_name") or "")
        if self.combo_comp.findText(material_name) < 0:
            material_name = "Custom"
        self.combo_comp.blockSignals(True)
        try:
            self.combo_comp.setCurrentText(material_name or "Custom")
            for widget, key in (
                (self.spin_radius, "radius"),
                (self.spin_cellsize, "cell_size"),
                (self.spin_press, "p_atm"),
                (self.spin_temp, "t_atm"),
                (self.spin_mass, "mass_kg"),
                (self.spin_density, "rho_charge"),
                (self.spin_cfl, "max_cfl"),
                (self.spin_endtime, "end_time_s"),
                (self.spin_gui_refresh, "probe_write_interval_steps"),
            ):
                if key in values:
                    widget.setValue(values[key])
            if "energy_j_per_kg" in values:
                self.edit_energy.setText(f"{float(values['energy_j_per_kg']):.12g}")
            self._apply_run_mode_radios(values)
            self._sync_end_time_always_editable()
            self._last_domain_radius = float(self.spin_radius.value())
            self._probe_fields = tuple(values.get("probe_fields") or ("p",))
            self._enable_impulse = bool(values.get("enable_impulse", True))
            self._enable_dynamic_pressure = bool(
                values.get("enable_dynamic_pressure", False)
            )
            self.set_gauge_locations(tuple(values.get("gauge_locations") or ()))
            if "remap_for_2d" in values:
                remap = bool(values.get("remap_for_2d"))
                self.radio_yes.setChecked(remap)
                self.radio_no.setChecked(not remap)
            self._apply_source_model_radios(values.get("source_model"))
        finally:
            self.combo_comp.blockSignals(False)
        self.recalc_stats()

    def selected_source_model(self) -> str:
        combo = getattr(self, "combo_source", None)
        if combo is None:
            return SOURCE_MODEL_JWL
        return normalize_source_model(combo.currentData())

    def _apply_source_model_radios(self, value) -> None:
        model = normalize_source_model(value)
        index = self.combo_source.findData(model)
        if index < 0:
            index = 0
        self.combo_source.setCurrentIndex(index)
        self.on_source_model_changed()

    def on_source_model_changed(self, *_args) -> None:
        ig = is_ideal_gas_source(self.selected_source_model())
        if hasattr(self, "btn_edit_comp"):
            self.btn_edit_comp.setEnabled(
                (not ig) and self.combo_comp.currentText() == "Custom"
            )
            self.btn_edit_comp.setToolTip(
                "JWL coefficients are unused for Ideal-Gas."
                if ig
                else ""
            )
        self.recalc_stats()

    def apply_output_gauges(self, fields: tuple, *, impulse: bool, dynamic_pressure: bool) -> None:
        """Store gauge field list from Output File Options (1D probes)."""
        names = tuple(fields) if fields else ("p",)
        if "p" not in names:
            names = ("p",) + names
        self._probe_fields = names
        self._enable_impulse = bool(impulse)
        self._enable_dynamic_pressure = bool(dynamic_pressure)

    def set_gauge_locations(self, locations: tuple) -> None:
        self._gauge_locations = tuple(
            (float(radius), str(label)) for radius, label in (locations or ())
        )

    def get_selected_material_properties(self):
        mat_name = self.combo_comp.currentText()
        materials = {
            "TNT":  {"rho": 1630, "A": 371.2e9, "B": 3.23e9,  "R1": 4.15, "R2": 0.95, "omega": 0.30, "E0": 4.29e6},
            "C4":   {"rho": 1601, "A": 609.77e9,"B": 12.95e9, "R1": 4.50, "R2": 1.40, "omega": 0.25, "E0": 4.52e6},
            "PETN": {"rho": 1770, "A": 617.0e9, "B": 16.9e9,  "R1": 4.40, "R2": 1.20, "omega": 0.25, "E0": 6.11e6},
            "ANFO": {"rho": 840,  "A": 49.46e9, "B": 1.89e9,  "R1": 3.90, "R2": 1.10, "omega": 0.33, "E0": 3.79e6},
            "Custom": {"rho": 1000,"A": 300.0e9, "B": 3.0e9,   "R1": 4.0,  "R2": 1.0,  "omega": 0.30, "E0": 3.00e6}
        }
        return materials.get(mat_name, materials["C4"])

    def on_material_changed(self):
        props = self.get_selected_material_properties()
        if self.combo_comp.currentText() != "Custom":
            self.spin_density.setValue(props["rho"])
            self.edit_energy.setText(f"{props['E0']:.2e}")
        self.on_source_model_changed()

    def create_input_row(self, unit_text, default_val, decimals=2, step=1.0, width=100, spin_cls=QDoubleSpinBox):
        layout = QHBoxLayout()
        spin = spin_cls()
        spin.setRange(0, 1_000_000)
        spin.setDecimals(decimals)
        spin.setValue(default_val)
        spin.setSingleStep(step)
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        spin.wheelEvent = lambda event: event.ignore()
        spin.setFixedWidth(width)
        spin.valueChanged.connect(self.recalc_stats)
        layout.addWidget(spin)
        if unit_text:
            layout.addWidget(QLabel(f"({unit_text})"))
        layout.addStretch()
        return spin, layout

    def setup_ui(self):
        # Two-column resizable layout: Left = Input + Info, Right = Viewport + Execution Control
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # ===== LEFT COLUMN: Input Parameters (top) + Info Panel (bottom) =====
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)

        # Input parameters (scrollable, top of left column)
        self.left_container = QWidget()
        input_layout = QVBoxLayout(self.left_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        group_domain = QGroupBox("Domain")
        domain_layout = QFormLayout()
        self.spin_radius, lay_radius = self.create_input_row(
            "m", 1.0, _RADIUS_DECIMALS, width=_DOMAIN_SPIN_WIDTH, spin_cls=_RadiusSpinBox,
        )
        domain_layout.addRow("Radius", lay_radius)
        self.spin_cellsize, lay_cell = self.create_input_row(
            "m", 0.005, 6, width=_DOMAIN_SPIN_WIDTH,
        )
        domain_layout.addRow("Cellsize", lay_cell)
        group_domain.setLayout(domain_layout)
        input_layout.addWidget(group_domain)

        group_charge = QGroupBox("Charge")
        charge_layout = QFormLayout()
        
        comp_layout = QHBoxLayout()
        self.combo_comp = QComboBox()
        self.combo_comp.addItems(["TNT", "C4", "PETN", "ANFO", "Custom"])
        self.combo_comp.setCurrentText("TNT")
        # 1.25 times the Mass field (S3), which is the 100px spin box.
        self.combo_comp.setFixedWidth(125)
        self.combo_comp.currentIndexChanged.connect(self.on_material_changed)
        comp_layout.addWidget(self.combo_comp)
        
        self.btn_edit_comp = QPushButton("Edit..")
        self.btn_edit_comp.setFixedWidth(100)
        self.btn_edit_comp.setEnabled(False)
        comp_layout.addWidget(self.btn_edit_comp)
        charge_layout.addRow("Composition", comp_layout)

        self.spin_mass, lay_mass = self.create_input_row("kg", 1.0)
        charge_layout.addRow("Mass", lay_mass)
        self.spin_density, lay_dens = self.create_input_row("kg/m3", 1630.0)
        charge_layout.addRow("Density", lay_dens)

        self.edit_energy = QLineEdit("4.29e+06")
        self.edit_energy.setFixedWidth(100)
        self.edit_energy.editingFinished.connect(self.recalc_stats)
        self.edit_energy.textChanged.connect(self.recalc_stats)
        energy_lay = QHBoxLayout()
        energy_lay.addWidget(self.edit_energy)
        energy_lay.addWidget(QLabel("(J/kg)"))
        energy_lay.addStretch()
        charge_layout.addRow("Energy", energy_lay)

        remap_layout = QHBoxLayout()
        self.radio_yes = QRadioButton("Yes")
        self.radio_no = QRadioButton("No")
        self.radio_no.setChecked(True)
        self.radio_yes.toggled.connect(self.recalc_stats) 
        remap_layout.addWidget(self.radio_yes)
        remap_layout.addWidget(self.radio_no)
        remap_layout.addStretch()
        charge_layout.addRow("Remap?", remap_layout)
        self.lbl_remap_status = QLabel("")
        self.lbl_remap_status.setWordWrap(True)
        self.lbl_remap_status.setStyleSheet(SECONDARY_INFO_STYLE)
        self.lbl_remap_status.hide()
        charge_layout.addRow(self.lbl_remap_status)
        self.radio_yes.toggled.connect(lambda *_args: self.refresh_remap_status())
        
        group_charge.setLayout(charge_layout)
        input_layout.addWidget(group_charge)

        group_atmo = QGroupBox("Atmosphere")
        atmo_layout = QFormLayout()
        self.spin_press, lay_press = self.create_input_row("Pa", 101325.0)
        atmo_layout.addRow("Press.", lay_press)
        self.spin_temp, lay_temp = self.create_input_row("K", 288.0)
        atmo_layout.addRow("Temp.", lay_temp)
        group_atmo.setLayout(atmo_layout)
        input_layout.addWidget(group_atmo)

        group_bounds = QGroupBox("Boundaries")
        bounds_layout = QFormLayout()
        self.cmb_left = QComboBox()
        self.cmb_left.addItem("Reflecting - spherical")
        self.cmb_left.setEnabled(False)
        self.cmb_left.setMinimumWidth(160)
        self.cmb_left.setMaximumWidth(220)
        bounds_layout.addRow("Left", self.cmb_left)
        self.cmb_right = QComboBox()
        self.cmb_right.addItem("Terminate", RUN_MODE_TERMINATE)
        self.cmb_right.addItem("Reflect", RUN_MODE_REFLECT)
        self.cmb_right.setItemData(0, TIP_RUN_TERMINATE, Qt.ToolTipRole)
        self.cmb_right.setItemData(1, TIP_RUN_REFLECT, Qt.ToolTipRole)
        self.cmb_right.setMinimumWidth(self.cmb_left.minimumWidth())
        self.cmb_right.setMaximumWidth(self.cmb_left.maximumWidth())
        self.cmb_right.currentIndexChanged.connect(self._sync_end_time_always_editable)
        bounds_layout.addRow("Right", self.cmb_right)
        group_bounds.setLayout(bounds_layout)
        input_layout.addWidget(group_bounds)

        group_solver = QGroupBox("Solver Options")
        solver_layout = QFormLayout()
        self.spin_cfl, lay_cfl = self.create_input_row("", 0.50, 2, 0.1, width=_DOMAIN_SPIN_WIDTH)
        solver_layout.addRow("CFL", lay_cfl)
        self.combo_source = QComboBox()
        self.combo_source.setFixedWidth(_DOMAIN_SPIN_WIDTH)
        for model_id in (SOURCE_MODEL_JWL, SOURCE_MODEL_IG):
            self.combo_source.addItem(SOURCE_MODEL_LABELS[model_id], model_id)
        self.combo_source.currentIndexChanged.connect(self.on_source_model_changed)
        solver_layout.addRow("Method", self.combo_source)
        self.spin_endtime, lay_etime = self.create_input_row("s", 0.025, 4, 0.001, width=_DOMAIN_SPIN_WIDTH)
        self.spin_endtime.setToolTip(TIP_END_TIME)
        solver_layout.addRow("End Time", lay_etime)
        self._sync_end_time_always_editable()
        
        group_solver.setLayout(solver_layout)
        input_layout.addWidget(group_solver)

        self.group_output = QGroupBox("Output Options")
        output_layout = QFormLayout()
        self.spin_gui_refresh = QSpinBox()
        self.spin_gui_refresh.setObjectName("spin1dGuiRefresh")
        self.spin_gui_refresh.setRange(1, 1_000_000)
        self.spin_gui_refresh.setValue(25)
        self.spin_gui_refresh.setButtonSymbols(QSpinBox.NoButtons)
        self.spin_gui_refresh.wheelEvent = lambda event: event.ignore()
        self.spin_gui_refresh.setFixedWidth(100)
        self.spin_gui_refresh.setAlignment(Qt.AlignRight)
        lay_refresh = QHBoxLayout()
        lay_refresh.addWidget(self.spin_gui_refresh)
        lay_refresh.addWidget(QLabel("Steps"))
        lay_refresh.addStretch()
        output_layout.addRow("GUI refresh freq.", lay_refresh)
        self.group_output.setLayout(output_layout)
        input_layout.addWidget(self.group_output)
        input_layout.addStretch()

        # Scroll area for input parameters only
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.left_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setMinimumWidth(COMPUTATIONAL_LEFT_PANEL_MIN)
        left_layout.addWidget(self.scroll_area, stretch=1)

        # Info Panel (bottom of left column)
        group_stats = QGroupBox("Info")
        stats_layout = QFormLayout()
        self.lbl_domain_cells = QLabel("0")
        self.lbl_charge_radius = QLabel("0.00")
        self.lbl_charge_cells = QLabel("0")
        self.lbl_adj_density = QLabel("0.00")
        for caption, value in (
            ("Dom. Cells:", self.lbl_domain_cells),
            ("Charge R:", self.lbl_charge_radius),
            ("Chrg. Cells:", self.lbl_charge_cells),
            ("Field Rho:", self.lbl_adj_density),
        ):
            label = QLabel(caption)
            label.setStyleSheet(INFO_TITLE_STYLE)
            value.setStyleSheet(INFO_ROW_STYLE)
            stats_layout.addRow(label, value)
        group_stats.setLayout(stats_layout)
        left_layout.addWidget(group_stats)

        self.splitter.addWidget(left_column)

        # ===== RIGHT COLUMN: Viewport (top) + Execution Control (bottom) =====
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)

        self._right_v_splitter = QSplitter(Qt.Vertical)
        self._right_v_splitter.setChildrenCollapsible(False)
        self._right_v_splitter.setObjectName("tab1dRightVerticalSplitter")

        viewport = self._build_viewport()

        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.setMinimumHeight(EXECUTION_AREA_MIN_HEIGHT)
        self.ctrl_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tab_exec = QWidget()
        self._build_exec_tab(self.tab_exec)
        # Wrap execution content so very short windows can scroll locally.
        exec_scroll = QScrollArea()
        exec_scroll.setWidgetResizable(True)
        exec_scroll.setFrameShape(QFrame.NoFrame)
        exec_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        exec_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        exec_scroll.setWidget(self.tab_exec)
        self._exec_scroll = exec_scroll
        self.ctrl_tabs.addTab(exec_scroll, "Execution Controls")

        self._right_v_splitter.addWidget(viewport)
        self._right_v_splitter.addWidget(self.ctrl_tabs)
        self._right_v_splitter.setStretchFactor(0, 1)
        self._right_v_splitter.setStretchFactor(1, 0)
        # Graph gets remainder; execution gets content-preferred height (~220–240).
        self._right_v_splitter.setSizes([800, EXECUTION_AREA_PREFERRED_HEIGHT])
        self._1d_exec_splitter_sizes = list(self._right_v_splitter.sizes())
        self._right_v_splitter.splitterMoved.connect(self._on_1d_exec_splitter_moved)

        self.right_layout.addWidget(self._right_v_splitter)
        self.splitter.addWidget(self.right_container)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        left_w = COMPUTATIONAL_LEFT_PANEL_WIDTH
        self.splitter.setSizes([left_w, max(400, 1200 - left_w)])
        self._main_splitter = self.splitter
        root_layout.addWidget(self.splitter)

    def _build_viewport(self) -> QWidget:
        frame = QWidget()
        frame.setMinimumWidth(0)
        frame.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        controls = QHBoxLayout()
        self._status_caption_host = QWidget(frame)
        self._status_caption_host.setObjectName("viewportStatusHost")
        host_layout = QHBoxLayout(self._status_caption_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(8)
        host_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._status_caption_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self._fit_graph)
        controls.addWidget(self._status_caption_host, 1)
        controls.addWidget(self.btn_fit, 0)
        layout.addLayout(controls)
        self.canvas = MplCanvas(self)
        self.canvas.setMinimumHeight(120)
        layout.addWidget(self.canvas, 1)
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

    def _fit_graph(self) -> None:
        self.canvas.axes.relim()
        self.canvas.axes.autoscale_view()
        self._redraw_canvas()

    def _redraw_canvas(self) -> None:
        """Apply pending live data, then paint a software bitmap (no Qt OpenGL)."""
        if (
            self._live_graph
            and self._pending_pressures
            and self.selected_chart() in ("pressure", "pmax")
        ):
            self._apply_live_profile(self._pending_pressures, self._pending_time_s)
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def selected_chart(self) -> str:
        group = getattr(self, "_chart_group", None)
        if group is None:
            return "pressure"
        button = group.checkedButton()
        if button is None:
            return "pressure"
        return str(button.property("chart") or "pressure")

    def _style_chart_axes(self, chart: str) -> None:
        titles = {
            "pressure": ("Overpressure vs Range", "Overpressure (Pa)"),
            "pmax": ("PMax vs Range", "Peak overpressure (Pa)"),
            "density": ("Density vs Range", "Density (kg/m3)"),
            "energy": ("Energy vs Range", "Energy (J/kg)"),
            "velocity": ("Velocity vs Range", "Velocity (m/s)"),
        }
        title, ylabel = titles.get(chart, titles["pressure"])
        axes = self.canvas.axes
        axes.set_title(title)
        axes.set_xlabel("Radius (m)")
        axes.set_ylabel(ylabel)
        axes.grid(True)
        axes.legend(loc="upper right")

    def _style_overpressure_axes(self) -> None:
        self._style_chart_axes("pressure")

    def charge_pressure_pa(self) -> float:
        """Charge pressure for the pre-run sketch from the entered density and energy."""
        rho = float(self.spin_density.value())
        try:
            energy = float(self.edit_energy.text())
        except (TypeError, ValueError):
            energy = float(self.get_selected_material_properties().get("E0") or 0.0)
        if is_ideal_gas_source(self.selected_source_model()):
            e_source = igs.CV_IDEAL_GAS * float(self.spin_temp.value()) + energy
            return ideal_gas_charge_pressure_pa(rho, e_source)
        return ideal_gas_charge_pressure_pa(rho, energy)

    def initial_overpressure_profile(self):
        rho = float(self.spin_density.value())
        charge_r = spherical_charge_radius_m(self.spin_mass.value(), rho)
        return initial_overpressure_step(
            self.spin_radius.value(),
            charge_r,
            self.charge_pressure_pa(),
            self.spin_press.value(),
        )

    def plot_initial_condition(self) -> None:
        """Show the entered charge before a run. A frozen or live profile is left in place."""
        if self._live_graph or getattr(self, "_has_run_profile", False):
            return
        self._plot_setup_chart()

    def _plot_setup_chart(self) -> None:
        if not hasattr(self, "canvas"):
            return
        chart = self.selected_chart()
        if chart in ("pressure", "pmax") and not is_ideal_gas_source(self.selected_source_model()):
            name = "PMax" if chart == "pmax" else "pressure"
            self.canvas.axes.clear()
            self.canvas.axes.set_title(f"JWL {name} profile available after solver initialization")
            self.canvas.axes.set_xlabel("Radius (m)")
            self.canvas.axes.set_ylabel("Overpressure (Pa)" if chart == "pressure" else "Peak overpressure (Pa)")
            self._redraw_canvas()
            return
        radii, values, label = self._setup_chart_series(chart)
        self.canvas.axes.clear()
        self.canvas.axes.plot(radii, values, color="#c0392b", linewidth=1.8, label=label)
        self._style_chart_axes(chart)
        self._redraw_canvas()

    def _setup_chart_series(self, chart: str):
        rho = float(self.spin_density.value())
        charge_r = spherical_charge_radius_m(self.spin_mass.value(), rho)
        radius = self.spin_radius.value()
        if chart == "density":
            ambient = igs.ambient_state(self.spin_press.value(), self.spin_temp.value())
            return (*initial_radial_step(radius, charge_r, rho, ambient.rho), "Density")
        if chart == "energy":
            ambient = igs.ambient_state(self.spin_press.value(), self.spin_temp.value())
            try:
                entered = float(self.edit_energy.text())
            except (TypeError, ValueError):
                entered = float(self.get_selected_material_properties().get("E0") or 0.0)
            inside = ambient.e + entered if is_ideal_gas_source(self.selected_source_model()) else entered
            return (*initial_radial_step(radius, charge_r, inside, ambient.e), "Energy")
        if chart == "velocity":
            return (*initial_radial_step(radius, charge_r, 0.0, 0.0), "Velocity")
        radii, overpressures = self.initial_overpressure_profile()
        label = "PMax" if chart == "pmax" else "Pressure"
        return radii, overpressures, label

    def on_chart_changed(self, *_args) -> None:
        if (self._live_graph or getattr(self, "_has_run_profile", False)) and self._pending_pressures:
            if self.selected_chart() in ("pressure", "pmax"):
                self._apply_live_profile(self._pending_pressures, self._pending_time_s)
                self._redraw_canvas()
                return
        self._plot_setup_chart()

    def begin_run_graph(self) -> None:
        """Clear stale post-run freeze so the next live profile can replace it."""
        self._has_run_profile = False
        self._live_graph = False
        self._pending_pressures = None
        self._pending_time_s = 0.0
        self._pmax_over = None

    def _right_boundary_is_reflect(self) -> bool:
        combo = getattr(self, "cmb_right", None)
        if combo is None:
            return False
        return combo.currentData() == RUN_MODE_REFLECT

    def _apply_run_mode_radios(self, values: dict) -> None:
        mode = normalize_run_mode(
            values.get("stop_mode") or values.get("mode"),
            values.get("right_boundary"),
        )
        index = self.cmb_right.findData(
            RUN_MODE_REFLECT if mode == RUN_MODE_REFLECT else RUN_MODE_TERMINATE
        )
        if index < 0:
            index = 0
        self.cmb_right.blockSignals(True)
        try:
            self.cmb_right.setCurrentIndex(index)
        finally:
            self.cmb_right.blockSignals(False)
        self._sync_end_time_always_editable()

    def _sync_end_time_always_editable(self, _checked: bool = False) -> None:
        """End Time stays visible and editable in both 1D modes."""
        self.spin_endtime.setVisible(True)
        self.spin_endtime.setEnabled(True)
        if hasattr(self, "cmb_right"):
            reflect = self._right_boundary_is_reflect()
            self.cmb_right.setToolTip(TIP_RUN_REFLECT if reflect else TIP_RUN_TERMINATE)

    def end_live_graph(self) -> None:
        """Freeze the last live profile after a run; do not reset to the t=0 sketch."""
        self._live_graph = False
        if self._pending_pressures:
            self._has_run_profile = True
            self._apply_live_profile(self._pending_pressures, self._pending_time_s)
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

    def _on_1d_exec_splitter_moved(self, _pos: int = 0, _index: int = 0) -> None:
        """Remember the user's graph/execution split for this session."""
        self._1d_exec_splitter_sizes = list(self._right_v_splitter.sizes())

    def restore_1d_exec_splitter_sizes(self) -> None:
        """Re-apply session splitter sizes without resetting on tab return."""
        sizes = getattr(self, "_1d_exec_splitter_sizes", None)
        if sizes and hasattr(self, "_right_v_splitter"):
            self._right_v_splitter.setSizes(sizes)

    def get_computational_left_width(self) -> int:
        return int(self.splitter.sizes()[0]) if self.splitter.sizes() else COMPUTATIONAL_LEFT_PANEL_WIDTH

    def set_computational_left_width(self, width: int) -> None:
        width = max(COMPUTATIONAL_LEFT_PANEL_MIN, int(width))
        total = sum(self.splitter.sizes()) or (width + 800)
        self.splitter.setSizes([width, max(50, total - width)])

    # --- פונקציית עזר לבניית הסרגל התחתון ---
    def _build_exec_tab(self, parent):
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignTop)

        column = QVBoxLayout()
        column.setSpacing(8)

        g_actions = QGroupBox("")
        v_actions = QVBoxLayout(g_actions)
        v_actions.setSpacing(8)

        action_font = QFont()
        action_font.setPointSize(ACTION_BUTTON_FONT_PT)
        action_font.setWeight(QFont.Bold)

        self.btn_initialize = QPushButton("Initialize Model")
        self.btn_initialize.setFont(action_font)
        self.btn_initialize.setMinimumHeight(40)
        self.btn_initialize.setToolTip("Generate mesh and initial fields before Run; required again after case input edits.")
        self.btn_initialize.clicked.connect(lambda: self.sig_request_init.emit(self.get_case_inputs()))
        v_actions.addWidget(self.btn_initialize)

        self.btn_run = QPushButton("▶ Run Simulation")
        self.btn_run.setStyleSheet("background-color: #2ecc71; color: white; padding: 5px;")
        self.btn_run.clicked.connect(self.sig_request_run.emit)

        self.btn_stop = QPushButton("⏸ Interrupt")
        self.btn_stop.setStyleSheet("background-color: #e67e22; color: white; padding: 5px;")
        self.btn_stop.clicked.connect(self.sig_request_stop.emit)
        for button in (self.btn_run, self.btn_stop):
            button.setMinimumWidth(198)
            button.setFont(QFont())

        v_actions.addWidget(self.btn_run)
        v_actions.addWidget(self.btn_stop)
        column.addWidget(g_actions)
        column.addStretch()
        layout.addLayout(column)
        layout.addWidget(self._build_chart_options(), 0, Qt.AlignTop)
        layout.addStretch()

    def _build_chart_options(self) -> QWidget:
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(16, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        title = QLabel("Chart:")
        title_font = QFont(title.font())
        title_font.setBold(True)
        title.setFont(title_font)
        grid.addWidget(title, 0, 0, Qt.AlignTop)
        self._chart_group = QButtonGroup(self)
        options = (
            (0, 1, "pressure", "Pressure", "radio_chart_pressure"),
            (0, 2, "pmax", "PMax", "radio_chart_pmax"),
            (1, 1, "density", "Density", "radio_chart_density"),
            (2, 1, "energy", "Energy", "radio_chart_energy"),
            (3, 1, "velocity", "Velocity", "radio_chart_velocity"),
        )
        for row, column, key, text, attr in options:
            radio = QRadioButton(text)
            radio.setProperty("chart", key)
            setattr(self, attr, radio)
            self._chart_group.addButton(radio)
            grid.addWidget(radio, row, column)
        self.radio_chart_pressure.setChecked(True)
        self._chart_group.buttonToggled.connect(self.on_chart_changed)
        return host

    def recalc_stats(self):
        try:
            r_domain = self.spin_radius.value()
            dx = self.spin_cellsize.value()
            mass = self.spin_mass.value()
            rho_input = self.spin_density.value()

            if dx < 1e-9: dx = 1e-9
            if rho_input <= 0: rho_input = 1600

            cells_dom = int(r_domain / dx)
            
            vol = mass / rho_input
            r_charge = ((3.0 * vol) / (4.0 * math.pi))**(1/3.0)
            
            cells_charge = int(r_charge / dx)
            self.calculated_adj_rho = rho_input
            self.calculated_discrete_radius = r_charge

            self.lbl_domain_cells.setText(f"{cells_dom}")
            self.lbl_charge_radius.setText(f"{r_charge:.6f}")
            self.lbl_charge_cells.setText(f"{cells_charge}")
            self.lbl_adj_density.setText(f"{rho_input:.1f}")
            self._live_graph = False
            self._has_run_profile = False
            self._pending_pressures = None
            self._pmax_over = None
            self.plot_initial_condition()
            self.refresh_remap_status()

        except Exception:
            pass

    def refresh_remap_status(self, case_dir: str = "") -> None:
        """Show last-solver-state remap snapshot availability after a 1D run."""
        if case_dir:
            self._last_run_case_dir = os.path.normpath(case_dir)
        label = getattr(self, "lbl_remap_status", None)
        if label is None:
            return
        if not self.radio_yes.isChecked():
            label.hide()
            label.setText("")
            return
        path = self._last_run_case_dir
        if not path:
            label.hide()
            label.setText("")
            return
        label.show()
        from remap_snapshot_1d import availability_for_case, display_text

        avail = availability_for_case(path)
        label.setText(display_text(avail) or avail.message or "Remap is unavailable.")
        if avail.status in ("invalid", "stale", "missing") and not avail.snapshot_available:
            label.setStyleSheet(WARNING_STYLE)
        else:
            label.setStyleSheet(SECONDARY_INFO_STYLE)

    def update_graph(self, pressures, sim_time_s: float, radii=()):
        if not pressures:
            return
        self._live_probe_radii = tuple(radii)
        self._pending_pressures = [float(p) for p in pressures]
        self._pending_time_s = float(sim_time_s)
        self._accumulate_pmax(self._pending_pressures)
        self._live_graph = True
        if not self._graph_timer.isActive():
            self._graph_timer.start()

    def _accumulate_pmax(self, pressures) -> None:
        p_atm = self.spin_press.value()
        over = [float(p) - p_atm for p in pressures]
        previous = getattr(self, "_pmax_over", None)
        if not previous or len(previous) != len(over):
            self._pmax_over = over
            return
        self._pmax_over = [max(old, new) for old, new in zip(previous, over)]

    def _apply_live_profile(self, pressures, sim_time_s: float) -> None:
        chart = self.selected_chart()
        if chart not in ("pressure", "pmax"):
            self._plot_setup_chart()
            return
        distances = getattr(self, "_live_probe_radii", ())
        if len(distances) != len(pressures) or not distances:
            self.canvas.axes.clear()
            self.canvas.axes.set_title("Live profile unavailable: probe coordinates missing")
            return
        if chart == "pmax" and getattr(self, "_pmax_over", None) and len(self._pmax_over) == len(pressures):
            values = list(self._pmax_over)
        else:
            p_atm = self.spin_press.value()
            values = [p - p_atm for p in pressures]

        self.canvas.axes.clear()
        self.canvas.axes.plot(
            distances, values, color="#c0392b", linewidth=1.8,
            label=f"t = {sim_time_s*1000.0:.3f} ms",
        )
        self._style_chart_axes(chart)