import math
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QRadioButton, QSplitter, QScrollArea, QSizePolicy, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from models import (
    BOUNDARY_1D_RIGHT_OPTIONS,
    BOUNDARY_1D_TRANSMIT,
    CaseInputs1D,
)
from ui_metrics import (
    COMPUTATIONAL_LEFT_PANEL_WIDTH,
    COMPUTATIONAL_LEFT_PANEL_MIN,
    EXECUTION_AREA_MIN_HEIGHT,
    EXECUTION_AREA_PREFERRED_HEIGHT,
    ACTION_BUTTON_FONT_PT,
    GROUP_TITLE_FONT_PT,
)


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


def initial_overpressure_step(
    domain_radius_m: float,
    charge_radius_m: float,
    charge_pressure_pa: float,
    p_atm: float,
):
    """Step profile: charge pressure inside R_charge, ambient overpressure outside."""
    r_max = max(float(domain_radius_m), 1e-9)
    r_c = min(max(float(charge_radius_m), 0.0), r_max)
    over = max(float(charge_pressure_pa) - float(p_atm), 0.0)
    return [0.0, r_c, r_c, r_max], [over, over, 0.0, 0.0]


class MplCanvas(QLabel):
    """Raster matplotlib figure. Avoids Qt5Agg OpenGL swaps that abort on Windows."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)
        self._dpi = float(dpi)
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor="white")
        self.axes = self.figure.add_subplot(111)
        self._agg = FigureCanvasAgg(self.figure)
        self._drawing = False
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)

    def draw(self):
        self._render_to_label()

    def draw_idle(self):
        self._render_to_label()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.axes.lines:
            self.draw_idle()

    def _render_to_label(self) -> None:
        if self._drawing:
            return
        self._drawing = True
        try:
            w = max(int(self.width()), 1)
            h = max(int(self.height()), 1)
            if w < 40 or h < 40:
                w, h = 640, 400
            self.figure.set_size_inches(w / self._dpi, h / self._dpi, forward=False)
            if w >= 200 and h >= 200:
                try:
                    self.figure.tight_layout()
                except Exception:
                    pass
            self._agg.draw()
            renderer = self._agg.get_renderer()
            width = int(renderer.width)
            height = int(renderer.height)
            buf = renderer.buffer_rgba()
            image = QImage(bytes(buf), width, height, QImage.Format_RGBA8888)
            self.setPixmap(QPixmap.fromImage(image))
        except Exception:
            pass
        finally:
            self._drawing = False


class Tab1D(QWidget):
    # --- הוספה: סיגנלים לתקשורת עם Main ---
    sig_request_run = pyqtSignal()
    sig_request_stop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.calculated_adj_rho = 0.0
        self.calculated_discrete_radius = 0.0
        
        self.last_r_min = None
        self.last_r_max = None
        self._live_graph = False
        self._pending_pressures = None
        self._pending_time_s = 0.0
        self._graph_timer = QTimer(self)
        self._graph_timer.setSingleShot(True)
        self._graph_timer.setInterval(50)
        self._graph_timer.timeout.connect(self._redraw_canvas)

        self.setup_ui()
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
            end_time_s=self.spin_endtime.value(), # השם המקורי שלך
            # ברירות מחדל קבועות (כי אין להן שדות ב-UI המקורי)
            write_interval_s=0.0,
            n_probes=200,
            probe_write_interval_steps=int(self.spin_gui_refresh.value()),
            wedge_angle_deg=15.0,
            cone_half_angle_deg=12.0,
            axis_epsilon=0.10,
            right_boundary=self.cmb_right.currentText(),
        )
    # ----------------------------------------

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
        self.recalc_stats()

    def create_input_row(self, unit_text, default_val, decimals=2, step=1.0):
        layout = QHBoxLayout()
        spin = QDoubleSpinBox()
        spin.setRange(0, 1_000_000)
        spin.setValue(default_val)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        spin.wheelEvent = lambda event: event.ignore()
        spin.setFixedWidth(100)
        spin.valueChanged.connect(self.recalc_stats)
        unit_label = QLabel(f"({unit_text})")
        layout.addWidget(spin)
        layout.addWidget(unit_label)
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
        self.spin_radius, lay_radius = self.create_input_row("m", 1.0)
        domain_layout.addRow("Radius", lay_radius)
        self.spin_cellsize, lay_cell = self.create_input_row("m", 0.005, 3)
        domain_layout.addRow("Cellsize", lay_cell)
        group_domain.setLayout(domain_layout)
        input_layout.addWidget(group_domain)

        group_charge = QGroupBox("Charge")
        charge_layout = QFormLayout()
        
        comp_layout = QHBoxLayout()
        self.combo_comp = QComboBox()
        self.combo_comp.addItems(["TNT", "C4", "PETN", "ANFO", "Custom"])
        self.combo_comp.setCurrentText("C4")
        self.combo_comp.currentIndexChanged.connect(self.on_material_changed) 
        comp_layout.addWidget(self.combo_comp)
        
        self.btn_edit_comp = QPushButton("Edit..")
        self.btn_edit_comp.setFixedWidth(100)
        self.btn_edit_comp.setEnabled(False)
        comp_layout.addWidget(self.btn_edit_comp)
        charge_layout.addRow("Mat", comp_layout)

        self.spin_mass, lay_mass = self.create_input_row("kg", 1.0)
        charge_layout.addRow("Mass", lay_mass)
        self.spin_density, lay_dens = self.create_input_row("kg/m3", 1601.0)
        charge_layout.addRow("Density", lay_dens)

        self.edit_energy = QLineEdit("4.52e+06")
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
        self.cmb_right = QComboBox()
        self.cmb_right.addItems(list(BOUNDARY_1D_RIGHT_OPTIONS))
        self.cmb_right.setCurrentText(BOUNDARY_1D_TRANSMIT)
        for combo in (self.cmb_left, self.cmb_right):
            combo.setMinimumWidth(160)
            combo.setMaximumWidth(220)
        bounds_layout.addRow("Left", self.cmb_left)
        bounds_layout.addRow("Right", self.cmb_right)
        group_bounds.setLayout(bounds_layout)
        input_layout.addWidget(group_bounds)

        group_solver = QGroupBox("Solver")
        solver_layout = QFormLayout()
        self.spin_cfl, lay_cfl = self.create_input_row("", 0.50, 2, 0.1)
        solver_layout.addRow("Max CFL", lay_cfl)
        
        self.spin_endtime, lay_etime = self.create_input_row("s", 0.025, 4, 0.001)
        solver_layout.addRow("End Time", lay_etime)
        
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
        for lbl in [self.lbl_domain_cells, self.lbl_charge_radius, self.lbl_charge_cells, self.lbl_adj_density]:
            lbl.setStyleSheet("font-weight: bold; color: #333;")
        stats_layout.addRow("Dom. Cells:", self.lbl_domain_cells)
        stats_layout.addRow("Charge R:", self.lbl_charge_radius)
        stats_layout.addRow("Chrg. Cells:", self.lbl_charge_cells)
        stats_layout.addRow("Field Rho:", self.lbl_adj_density)
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
        if self._live_graph and self._pending_pressures:
            self._apply_live_profile(self._pending_pressures, self._pending_time_s)
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _style_overpressure_axes(self) -> None:
        axes = self.canvas.axes
        axes.set_title("Overpressure vs Range")
        axes.set_xlabel("Radius (m)")
        axes.set_ylabel("Overpressure (Pa)")
        axes.grid(True)
        axes.legend(loc="upper right")

    def charge_pressure_pa(self) -> float:
        """Charge pressure for the pre-run sketch from the entered density and energy."""
        rho = float(self.spin_density.value())
        try:
            energy = float(self.edit_energy.text())
        except (TypeError, ValueError):
            energy = float(self.get_selected_material_properties().get("E0") or 0.0)
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
        """Show the entered charge as a step in overpressure vs radius, before a run."""
        if self._live_graph or not hasattr(self, "canvas"):
            return
        radii, overpressures = self.initial_overpressure_profile()
        self.canvas.axes.clear()
        self.canvas.axes.plot(radii, overpressures, color="#c0392b", linewidth=1.8, label="Pressure")
        self._style_overpressure_axes()
        self._redraw_canvas()

    def end_live_graph(self) -> None:
        """Allow input edits to refresh the initial-condition sketch after a run."""
        self._live_graph = False

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
        
        # כפתורי פעולה בלבד
        g_actions = QGroupBox("Simulation Control")
        title_font = QFont(g_actions.font())
        title_font.setPointSize(GROUP_TITLE_FONT_PT)
        title_font.setBold(True)
        g_actions.setFont(title_font)
        v_actions = QHBoxLayout(g_actions)
        
        action_font = QFont()
        action_font.setPointSize(ACTION_BUTTON_FONT_PT)
        action_font.setWeight(QFont.Bold)

        self.btn_run = QPushButton("▶ Run Simulation")
        # Width sized for 10 pt bold label + padding (native Windows metrics).
        self.btn_run.setFixedWidth(250)
        self.btn_run.setFixedHeight(50)
        self.btn_run.setFont(action_font)
        self.btn_run.setStyleSheet(
            "background-color: #2ecc71; color: white; font-weight: bold; border-radius: 6px;"
        )
        self.btn_run.clicked.connect(self.sig_request_run.emit)

        self.btn_stop = QPushButton("⏸ Interrupt")
        self.btn_stop.setFixedWidth(190)
        self.btn_stop.setFixedHeight(50)
        self.btn_stop.setFont(action_font)
        self.btn_stop.setStyleSheet(
            "background-color: #e67e22; color: white; font-weight: bold; border-radius: 6px;"
        )
        self.btn_stop.clicked.connect(self.sig_request_stop.emit)

        v_actions.addWidget(self.btn_run)
        v_actions.addSpacing(20)
        v_actions.addWidget(self.btn_stop)
        layout.addWidget(g_actions)
        
        layout.addStretch()

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
            self.plot_initial_condition()

        except Exception:
            pass

    def update_graph(self, pressures, sim_time_s: float):
        if not pressures:
            return
        self._pending_pressures = [float(p) for p in pressures]
        self._pending_time_s = float(sim_time_s)
        self._live_graph = True
        if not self._graph_timer.isActive():
            self._graph_timer.start()

    def _apply_live_profile(self, pressures, sim_time_s: float) -> None:
        if self.last_r_min is None:
            try:
                radius = float(self.spin_radius.value())
                dx = float(self.spin_cellsize.value())
                rho = float(self.spin_density.value())

                vol = float(self.spin_mass.value()) / max(rho, 1.0)
                r_ch = ((3.0 * vol) / (4.0 * math.pi)) ** (1 / 3.0)

                r_min_geom = max(1e-6, 0.05 * dx)
                r_min = max(1e-6, min(r_min_geom, 0.2 * r_ch))
                self.last_r_min = r_min
                self.last_r_max = radius
            except (TypeError, ValueError, ZeroDivisionError, AttributeError):
                self.last_r_min = 0.0
                self.last_r_max = 1.0

        r_min = self.last_r_min
        r_max = self.last_r_max
        p_atm = self.spin_press.value()
        overpressures = [p - p_atm for p in pressures]
        n = len(overpressures)
        if n > 1:
            distances = [r_min + (i / (n - 1)) * (r_max - r_min) for i in range(n)]
        else:
            distances = [r_min]

        self.canvas.axes.clear()
        self.canvas.axes.plot(
            distances, overpressures, color="#c0392b", linewidth=1.8,
            label=f"t = {sim_time_s*1000.0:.3f} ms",
        )
        self._style_overpressure_axes()