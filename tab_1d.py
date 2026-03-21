import math
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QLineEdit,
    QRadioButton, QSplitter, QScrollArea, QSizePolicy, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- תיקון: ייבוא CaseInputs1D ---
from models import CaseInputs1D

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


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

        self.setup_ui()
        self.recalc_stats()

    # --- הוספה: פונקציה שה-Main דורש (מותאמת למשתנים שלך) ---
    def get_case_inputs(self) -> CaseInputs1D:
        """אוספת את כל הנתונים מהממשק ומחזירה אובייקט מסודר"""
        
        if self.radio_yes.isChecked():
            rho_final = self.calculated_adj_rho
        else:
            rho_final = self.spin_density.value()
            
        mat_props = self.get_selected_material_properties()
        
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
            write_interval_s=1e-5,
            n_probes=1000,
            probe_write_interval_steps=100,
            wedge_angle_deg=5.0,
            cone_half_angle_deg=12.0,
            axis_epsilon=1e-3
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

        group_solver = QGroupBox("Solver")
        solver_layout = QFormLayout()
        self.spin_cfl, lay_cfl = self.create_input_row("", 0.50, 2, 0.1)
        solver_layout.addRow("Max CFL", lay_cfl)
        
        self.spin_endtime, lay_etime = self.create_input_row("s", 0.025, 4, 0.001)
        solver_layout.addRow("End Time", lay_etime)
        
        group_solver.setLayout(solver_layout)
        input_layout.addWidget(group_solver)
        input_layout.addStretch()

        # Scroll area for input parameters only
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.left_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setMinimumWidth(260)
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
        self.canvas = MplCanvas(self)
        self.canvas.axes.set_title("Overpressure vs Distance")
        self.canvas.axes.set_xlabel("Distance (m)")
        self.canvas.axes.set_ylabel("Overpressure (Pa)")
        self.canvas.axes.grid(True)
        self.right_layout.addWidget(self.canvas, stretch=1)
        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.setMinimumHeight(140)
        self.ctrl_tabs.setMaximumHeight(160)
        self.tab_exec = QWidget()
        self._build_exec_tab(self.tab_exec)
        self.ctrl_tabs.addTab(self.tab_exec, "Execution Controls")
        self.right_layout.addWidget(self.ctrl_tabs)
        self.splitter.addWidget(self.right_container)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([400, 800])
        root_layout.addWidget(self.splitter)

    # --- פונקציית עזר לבניית הסרגל התחתון ---
    def _build_exec_tab(self, parent):
        layout = QHBoxLayout(parent)
        
        # כפתורי פעולה בלבד
        g_actions = QGroupBox("Simulation Control")
        v_actions = QHBoxLayout(g_actions)
        
        self.btn_run = QPushButton("▶ Run Simulation")
        self.btn_run.setFixedWidth(200)
        self.btn_run.setFixedHeight(50) 
        self.btn_run.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        self.btn_run.clicked.connect(self.sig_request_run.emit)

        self.btn_stop = QPushButton("⏸ Interrupt")
        self.btn_stop.setFixedWidth(160)
        self.btn_stop.setFixedHeight(50)
        self.btn_stop.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
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
            
            if self.radio_yes.isChecked(): 
                if cells_charge > 0:
                    discrete_radius = cells_charge * dx
                    discrete_vol = (4.0/3.0) * math.pi * (discrete_radius**3)
                    adj_rho = mass / discrete_vol
                    self.calculated_discrete_radius = discrete_radius
                else:
                    adj_rho = rho_input
                    self.calculated_discrete_radius = r_charge
            else: 
                adj_rho = rho_input
                self.calculated_discrete_radius = r_charge

            self.calculated_adj_rho = adj_rho

            self.lbl_domain_cells.setText(f"{cells_dom}")
            self.lbl_charge_radius.setText(f"{r_charge:.6f}")
            self.lbl_charge_cells.setText(f"{cells_charge}")
            self.lbl_adj_density.setText(f"{adj_rho:.1f}")

        except Exception:
            pass

    def update_graph(self, pressures, sim_time_s: float):
        if not pressures: return
        
        if self.last_r_min is None:
            try:
                radius = float(self.spin_radius.value())
                dx = float(self.spin_cellsize.value())
                if self.radio_yes.isChecked():
                    rho = self.calculated_adj_rho
                else:
                    rho = float(self.spin_density.value())
                
                vol = float(self.spin_mass.value()) / max(rho, 1.0)
                r_ch = ((3.0 * vol) / (4.0 * math.pi))**(1/3.0)
                
                r_min_geom = max(1e-6, 0.05 * dx)
                r_min = max(1e-6, min(r_min_geom, 0.2 * r_ch))
                self.last_r_min = r_min
                self.last_r_max = radius
            except:
                self.last_r_min = 0.0
                self.last_r_max = 1.0

        r_min = self.last_r_min
        r_max = self.last_r_max
        
        p_atm = self.spin_press.value()
        overpressures = [p - p_atm for p in pressures]
        
        n = len(overpressures)
        if n > 1:
            distances = [r_min + (i/(n-1))*(r_max - r_min) for i in range(n)]
        else:
            distances = [r_min]

        self.canvas.axes.clear()
        self.canvas.axes.plot(distances, overpressures, label=f"t = {sim_time_s*1000.0:.3f} ms")
        self.canvas.axes.set_title("Overpressure vs Distance")
        self.canvas.axes.set_xlabel("Distance (m)")
        self.canvas.axes.set_ylabel("Overpressure (Pa)")
        
        self.canvas.axes.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        
        self.canvas.axes.grid(True)
        self.canvas.axes.legend(loc="upper right")
        self.canvas.draw()