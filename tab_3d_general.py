import math
import os
from typing import List
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QScrollArea, QFrame,
    QTabWidget, QGroupBox, QFormLayout, QGridLayout, QLabel, QPushButton, QDoubleSpinBox,
    QComboBox, QTableWidget, QTableWidgetItem, QFileDialog, QSpinBox,
    QCheckBox, QHeaderView, QRadioButton, QButtonGroup,
    QDialog, QDialogButtonBox, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from probes_model import ProbesModel
from models import CaseInputs3D, ObstacleData
from viewer_widget import BlastViewerWidget, ObstacleItem, SectionItem
from dialogs import RemapConfigDialog
try:
    from bf_option_discovery import get_eos_options, get_activation_options, get_thermo_options, get_decomposition_method_options
except ImportError:
    def get_eos_options(loaded_tokens=None):
        return list(loaded_tokens or []) + ["JWL", "BirchMurnaghan3", "idealGas"]
    def get_activation_options(loaded_tokens=None):
        return list(loaded_tokens or []) + ["pressureBased", "none"]
    def get_thermo_options(loaded_tokens=None):
        return list(loaded_tokens or []) + ["eConst", "ePolynomial"]
    def get_decomposition_method_options(loaded_tokens=None):
        return list(loaded_tokens or []) + ["scotch", "simple", "hierarchical", "manual"]

class TabGeneral3D(QWidget):
    sig_request_init = pyqtSignal(object)
    sig_request_run = pyqtSignal()
    sig_request_run_exact_1 = pyqtSignal()
    sig_request_run_exact_end = pyqtSignal()
    sig_request_stop = pyqtSignal()
    initial_dt_changed = pyqtSignal(float)

    def __init__(self, probes_model: ProbesModel):
        super().__init__()
        self.probes_model = probes_model
        self.obstacles: List[ObstacleItem] = []
        self.sections: List[SectionItem] = [] 
        self._block_signals = False
        self.viewer = None
        self._dyn_refine_max = 1  # AMR maxRefinement default (building3D: 1 = moving refinement front)

        self.materials_db = {
            "TNT":   {"rho": 1630, "energy": 4.29e6},
            "C4":    {"rho": 1601, "energy": 4.52e6},
            "PETN":  {"rho": 1770, "energy": 6.11e6},
            "ANFO":  {"rho": 840,  "energy": 3.79e6},
            "Custom": {
                "rho": 1600, "energy": 4.50e6,
                "A": 300.0e9, "B": 3.0e9, "R1": 4.0, "R2": 1.0, "omega": 0.30,
            },
        }

        self._build_ui()
        self._connect_signals()
        self._update_edit_button_visibility()
        self._on_material_changed("TNT")
        self._on_shape_changed("Sphere")
        self._update_preview()
        self.probes_model.changed.connect(self._update_preview)
        self.probes_model.changed.connect(self._refresh_probes_display)
        if self.viewer:
            self.viewer.cell_count_updated.connect(self._on_cell_count_updated)

    def _build_ui(self):
        # --- שינוי: ספליטר אופקי ראשי - Input Panel משמאל לכל הגובה ---
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ===== LEFT SIDE: Input Panel (extends full height) =====
        setup_widget = QWidget()
        setup_layout = QVBoxLayout(setup_widget)
        setup_layout.setContentsMargins(4, 4, 4, 4)
        setup_layout.setSpacing(4)
        
        self.settings_tabs = QTabWidget()
        self._tab_model = QWidget(); self._build_model_tab(self._tab_model)
        self._tab_obs = QWidget(); self._build_obstacles_tab(self._tab_obs)
        
        self.settings_tabs.addTab(self._tab_model, "Model Setup")
        self.settings_tabs.addTab(self._tab_obs, "Obstacles")
        setup_layout.addWidget(self.settings_tabs)
        
        # Info Frame (compact, smaller font)
        info_font = "font-size: 9pt; color: #333;"
        info_frm = QFrame()
        info_frm.setStyleSheet("background:#eef2f6; border:1px solid #c7d0da; border-radius: 4px;")
        il = QVBoxLayout(info_frm)
        self.lbl_cells = QLabel("Cells: -")
        self.lbl_cells.setStyleSheet("font-weight: bold; font-size: 9pt; color: #333;")
        il.addWidget(self.lbl_cells)
        self.lbl_init_mode = QLabel("Init: —")
        self.lbl_init_mode.setStyleSheet(info_font)
        self.lbl_init_mode.setToolTip("Init command actually used: setFields or setRefinedFields.")
        il.addWidget(self.lbl_init_mode)
        self.lbl_initiation_radius = QLabel("Initiation radius: —")
        self.lbl_initiation_radius.setStyleSheet(info_font)
        il.addWidget(self.lbl_initiation_radius)
        self.lbl_charge_refine_info = QLabel("Charge refine: —")
        self.lbl_charge_refine_info.setStyleSheet(info_font)
        il.addWidget(self.lbl_charge_refine_info)
        self.lbl_obstacle_refine_info = QLabel("Obstacle refine: —")
        self.lbl_obstacle_refine_info.setStyleSheet(info_font)
        il.addWidget(self.lbl_obstacle_refine_info)
        self.lbl_charge_cells = QLabel("Charge cells (alpha.c4>thr): —")
        self.lbl_charge_cells.setStyleSheet(info_font)
        self.lbl_charge_cells.setToolTip("Number of cells with alpha.c4 > threshold in 0/ after init. Threshold default 0.5.")
        il.addWidget(self.lbl_charge_cells)
        self.lbl_charge_fraction = QLabel("Charge fraction (%): —")
        self.lbl_charge_fraction.setStyleSheet(info_font)
        il.addWidget(self.lbl_charge_fraction)
        self.lbl_cells_inside_charge = QLabel("Cells inside charge (post-refinement): —")
        self.lbl_cells_inside_charge.setStyleSheet(info_font)
        self.lbl_cells_inside_charge.setToolTip("Number of cells inside the charge after setFields/setRefinedFields.")
        il.addWidget(self.lbl_cells_inside_charge)
        self.lbl_charge_clipped = QLabel("Charge clipped by domain: —")
        self.lbl_charge_clipped.setStyleSheet(info_font)
        self.lbl_charge_clipped.setToolTip("Whether the charge geometry extends outside the domain.")
        il.addWidget(self.lbl_charge_clipped)
        self.lbl_est_charge_cells = QLabel("Est. charge cells: —")
        self.lbl_est_charge_cells.setStyleSheet(info_font)
        self.lbl_est_charge_cells.setToolTip("Pre-flight estimate of cells in charge region (geometry + refinement).")
        il.addWidget(self.lbl_est_charge_cells)
        self.lbl_smallest_cell = QLabel("Smallest cell (est.): —")
        self.lbl_smallest_cell.setStyleSheet(info_font)
        self.lbl_smallest_cell.setToolTip("Estimated smallest cell size near charge (for ignition radius lower bound).")
        il.addWidget(self.lbl_smallest_cell)
        self.lbl_cells_in_ignition = QLabel("Cells in ignition region: —")
        self.lbl_cells_in_ignition.setStyleSheet(info_font)
        self.lbl_cells_in_ignition.setToolTip("Cells with alpha.c4>thr in ignition radius (non-remap).")
        il.addWidget(self.lbl_cells_in_ignition)
        self.lbl_expected_emesh = QLabel("Expected .eMesh: —")
        self.lbl_expected_emesh.setStyleSheet(info_font)
        self.lbl_expected_emesh.setToolTip("Canonical paths constant/extendedFeatureEdgeMesh/<base>.eMesh; missing triggers preflight block.")
        il.addWidget(self.lbl_expected_emesh)
        self.lbl_charge_resolution_warning = QLabel("")
        self.lbl_charge_resolution_warning.setStyleSheet("font-size: 9pt; color: #c00; font-weight: bold;")
        self.lbl_charge_resolution_warning.setWordWrap(True)
        il.addWidget(self.lbl_charge_resolution_warning)
        setup_layout.addWidget(info_frm)
        
        scroll = QScrollArea()
        scroll.setWidget(setup_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        splitter.addWidget(scroll)

        # ===== RIGHT SIDE: Viewport + Execution Controls (stacked vertically) =====
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # Viewport (top)
        self.viewer = BlastViewerWidget()
        right_layout.addWidget(self.viewer, stretch=1)
        
        # Execution Controls (bottom)
        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.setMinimumHeight(220)
        self.ctrl_tabs.setMaximumHeight(400)
        
        self._tab_exec = QWidget(); self._build_exec_tab(self._tab_exec)
        self._tab_view = QWidget(); self._build_view_tab(self._tab_view)
        self._tab_sect = QWidget(); self._build_section_tab(self._tab_sect)

        self.ctrl_tabs.addTab(self._tab_exec, "Execution Controls")
        self.ctrl_tabs.addTab(self._tab_view, "Viewport Options")
        self.ctrl_tabs.addTab(self._tab_sect, "Cross-Sections")
        
        right_layout.addWidget(self.ctrl_tabs)
        
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 700])

    def _lbl(self, text: str) -> QLabel:
        """Create a label with fixed width so all field columns align."""
        lbl = QLabel(text)
        lbl.setFixedWidth(self.LABEL_W)
        return lbl

    def _spin(self, minv, maxv, val, step, dec, max_width=144):
        s = QDoubleSpinBox()
        s.setRange(minv, maxv); s.setValue(val); s.setSingleStep(step); s.setDecimals(dec)
        s.setKeyboardTracking(False)
        s.setMaximumWidth(max_width)
        return s

    # רוחב קבוע לעמודת הטקסט — שנה כאן ויישפיע על כל הסקציות
    LABEL_W = 100
    # מרווח אנכי בין שורות — שנה כאן ויישפיע על כל הסקציות
    ROW_SPACING = 5

    def _build_model_tab(self, parent):
        l = QVBoxLayout(parent)
        l.setSpacing(2)
        g = QGroupBox("Domain Geometry")
        f = QGridLayout(g)
        f.setContentsMargins(4, 4, 4, 4)
        f.setColumnMinimumWidth(0, self.LABEL_W)
        f.setHorizontalSpacing(8) 
        f.setVerticalSpacing(self.ROW_SPACING)
        # יצירת השדות (ללא גלילת עכבר — מניעת שינוי מקרי)
        self.sx1 = self._spin(-100, 100, -2, 0.1, 2, max_width=80) 
        self.sx2 = self._spin(-100, 100, 2, 0.1, 2, max_width=80)
        self.sy1 = self._spin(-100, 100, -2, 0.1, 2, max_width=80)
        self.sy2 = self._spin(-100, 100, 2, 0.1, 2, max_width=80)
        self.sz1 = self._spin(-100, 100, -2, 0.1, 2, max_width=80)
        self.sz2 = self._spin(-100, 100, 2, 0.1, 2, max_width=80)
        self.scell = self._spin(1e-4, 1, 0.1, 0.01, 3, max_width=80)
        for _sb in (self.sx1, self.sx2, self.sy1, self.sy2, self.sz1, self.sz2, self.scell):
            _sb.setButtonSymbols(QDoubleSpinBox.NoButtons)
            _sb.wheelEvent = lambda event: event.ignore()

        # הוספה לגריד: (Widget, Row, Column)
        # עמודה 0 = טקסט, עמודה 1 = שדות, עמודה 2 = רווח ריק
        
        # שורה 0
        f.addWidget(QLabel("Min X / Max X"), 0, 0)
        f.addWidget(self._pair(self.sx1, self.sx2), 0, 1)
        
        # שורה 1
        f.addWidget(QLabel("Min Y / Max Y"), 1, 0)
        f.addWidget(self._pair(self.sy1, self.sy2), 1, 1)

        # שורה 2
        f.addWidget(QLabel("Min Z / Max Z"), 2, 0)
        f.addWidget(self._pair(self.sz1, self.sz2), 2, 1)

        # שורה 3 (Cell Size)
        f.addWidget(QLabel("Cell Size"), 3, 0)
        cell_row = QWidget(); cell_h = QHBoxLayout(cell_row); cell_h.setContentsMargins(0,0,0,0)
        cell_h.addWidget(self.scell); cell_h.addStretch() # מצמיד את השדה לשמאל
        f.addWidget(cell_row, 3, 1)
        # Mesh mode: Fixed Mesh (default) / Dyn Mesh; Dyn Mesh levels to the right
        self.rad_fixed_mesh = QRadioButton("Fixed Mesh")
        self.rad_dyn_mesh = QRadioButton("Dyn Mesh (AMR)")
        self.rad_fixed_mesh.setChecked(True)
        self.rad_fixed_mesh.setToolTip("Static mesh (no AMR).")
        self.rad_dyn_mesh.setToolTip("Dynamic mesh (AMR). Levels control min/max refinement.")
        mesh_mode_bg = QButtonGroup(self)
        mesh_mode_bg.addButton(self.rad_fixed_mesh)
        mesh_mode_bg.addButton(self.rad_dyn_mesh)
        self.spin_refine_min = QSpinBox()
        self.spin_refine_min.setRange(0, 10)
        self.spin_refine_min.setValue(2)
        self.spin_refine_min.setMaximumWidth(60)
        self.spin_refine_min.setToolTip("Min refinement level (0 = no refinement).")
        self.spin_refine_max = QSpinBox()
        self.spin_refine_max.setRange(0, 10)
        self.spin_refine_max.setValue(3)
        self.spin_refine_max.setMaximumWidth(60)
        self.spin_refine_max.setToolTip("Max refinement level.")
        mesh_mode_col = QWidget()
        mesh_mode_v = QVBoxLayout(mesh_mode_col)
        mesh_mode_v.setContentsMargins(0, 0, 0, 0)
        mesh_mode_v.setSpacing(2)
        mesh_mode_v.addWidget(QLabel("Mesh mode"))
        mesh_mode_v.addWidget(self.rad_fixed_mesh)
        dyn_row = QWidget()
        dyn_h = QHBoxLayout(dyn_row)
        dyn_h.setContentsMargins(0, 0, 0, 0)
        dyn_h.addWidget(self.rad_dyn_mesh)
        dyn_h.addWidget(QLabel("Levels:"))
        dyn_h.addWidget(self.spin_refine_min)
        dyn_h.addWidget(self.spin_refine_max)
        dyn_h.addStretch()
        mesh_mode_v.addWidget(dyn_row)
        self.rad_dyn_mesh.toggled.connect(self._on_mesh_mode_changed)
        self.rad_dyn_mesh.toggled.connect(lambda: self._set_provenance_user("enable_dyn_refine"))
        self.spin_refine_min.valueChanged.connect(self._validate_refine_levels)
        self.spin_refine_max.valueChanged.connect(self._validate_refine_levels)
        self.spin_refine_max.valueChanged.connect(self._on_dyn_refine_max_changed)
        f.addWidget(mesh_mode_col, 4, 0, 1, 2)

        # טריק הקסם: עמודה 2 מקבלת את כל המתיחה (Stretch)
        # זה דוחף את עמודות 0 ו-1 שמאלה, צמודות אחת לשנייה
        f.setColumnStretch(2, 1) 
        
        l.addWidget(g)

        # --- CFL Stability Parameter (spec 4.3.2) ---
        grp_cfl = QGroupBox("Simulation Parameters")
        f_cfl = QFormLayout(grp_cfl)
        f_cfl.setHorizontalSpacing(8)
        f_cfl.setVerticalSpacing(self.ROW_SPACING)
        self.spin_cfl = QDoubleSpinBox()
        self.spin_cfl.setRange(0.1, 2.0)
        self.spin_cfl.setValue(0.5)
        self.spin_cfl.setSingleStep(0.1)
        self.spin_cfl.setDecimals(2)
        self.spin_cfl.setKeyboardTracking(False)
        self.spin_cfl.setMaximumWidth(120)
        self.spin_cfl.setToolTip("Courant number (CFL) for time-step stability. Typical ≈ 0.4 for 3D.")
        f_cfl.addRow(self._lbl("CFL Number"), self.spin_cfl)

        l.addWidget(grp_cfl)

        b = QGroupBox("Boundaries"); fb = QFormLayout(b)
        fb.setHorizontalSpacing(8)
        fb.setVerticalSpacing(self.ROW_SPACING)
        self.bound_combos = {}
        for k in ['minX', 'maxX', 'minY', 'maxY', 'minZ', 'maxZ']:
            cmb = QComboBox(); cmb.addItems(["Reflecting", "Transmitting"])
            cmb.setMaximumWidth(200)
            self.bound_combos[k] = cmb
            fb.addRow(self._lbl(k), cmb)
        l.addWidget(b)

        self.grp_charge = QGroupBox("Charge Properties")
        c = self.grp_charge
        charge_main = QVBoxLayout(c)
        charge_main.setSpacing(0)
        charge_main.setContentsMargins(0, 8, 0, 0)

        def _field_wrap(w):
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(18, 0, 0, 0)
            h.addWidget(w)
            h.addStretch()
            return box

        LABEL_MINW = 140

        self.c_mat = QComboBox(); self.c_mat.addItems(["TNT", "C4", "PETN", "ANFO", "Custom"])
        self.c_mat.setMaximumWidth(184)
        self.c_mat.currentTextChanged.connect(self._on_material_changed)
        self.btn_edit_custom = QPushButton("Edit…")
        self.btn_edit_custom.setFixedWidth(100)
        self.btn_edit_custom.setToolTip("Edit Custom material JWL parameters (rho, E0, A, B, R1, R2, ω)")
        self.btn_edit_custom.clicked.connect(self._open_custom_material_dialog)
        mat_row = QWidget(); mat_row_h = QHBoxLayout(mat_row); mat_row_h.setContentsMargins(0, 0, 0, 0); mat_row_h.setSpacing(0)
        mat_row_h.addWidget(self.c_mat); mat_row_h.addWidget(self.btn_edit_custom); mat_row_h.addStretch()
        self.c_mat.currentTextChanged.connect(self._update_edit_button_visibility)
        
        self.c_shape = QComboBox(); self.c_shape.addItems(["Sphere", "Cylinder", "Cuboid"])
        self.c_shape.setMaximumWidth(184)
        self.c_shape.currentTextChanged.connect(self._on_shape_changed)

        # --- Geometry fields: Radius (computed for Sphere/Cylinder), Aspect (L/D), Length, Width, Height ---
        self.c_radius = self._spin(0.001, 100, 0.1, 0.01, 4)
        self.c_radius.setToolTip("Computed from mass and density (and Aspect for Cylinder). Read-only for Sphere and Cylinder.")
        self.c_aspect = self._spin(0.1, 20, 2.5, 0.1, 2)
        self.c_aspect.setToolTip("Length-to-Diameter ratio (L/D). Only for Cylinder shape.")
        self.c_length = self._spin(0.001, 100, 0.5, 0.01, 4)
        self.c_length.setToolTip("Cuboid length along X [m].")
        self.c_width = self._spin(0.001, 100, 0.5, 0.01, 4)
        self.c_width.setToolTip("Cuboid width along Y [m].")
        self.c_height = self._spin(0.001, 100, 0.5, 0.01, 4)
        self.c_height.setToolTip("Cuboid height along Z [m]. Computed from volume/(length×width) when shape is Cuboid so dimensions match mass and density.")
        # Labels for geometry rows
        self.lbl_radius = self._lbl("Radius [m]")
        self.lbl_aspect = self._lbl("Aspect (L/D)")
        self.c_cylinder_axis = QComboBox()
        self.c_cylinder_axis.addItems(["X", "Y", "Z"])
        self.c_cylinder_axis.setMaximumWidth(80)
        self.c_cylinder_axis.setToolTip("Cylinder axis direction. Only for Cylinder shape.")
        self.lbl_cylinder_axis = self._lbl("Cylinder axis")
        self.lbl_length = self._lbl("Length [m]")
        self.lbl_width = self._lbl("Width [m]")
        self.lbl_height = self._lbl("Height [m]")

        self.c_mass = self._spin(0.01, 1e4, 5, 0.5, 2)
        self.c_rho = self._spin(1, 1e4, 1630, 10, 1)
        self.cx = self._spin(-100,100,0,0.1,2, max_width=116)
        self.cy = self._spin(-100,100,0,0.1,2, max_width=116)
        self.cz = self._spin(-100,100,0,0.1,2, max_width=116)
        self.init_ix = self._spin(-100, 100, 0, 0.1, 2, max_width=116)
        self.init_iy = self._spin(-100, 100, 0, 0.1, 2, max_width=116)
        self.init_iz = self._spin(-100, 100, 0, 0.1, 2, max_width=116)
        # Charge refinement (internal + outer) — created here so we can place the block above Center
        self.spin_charge_refine = QSpinBox()
        self.spin_charge_refine.setRange(0, 8)
        self.spin_charge_refine.setValue(0)
        self.spin_charge_refine.setMaximumWidth(60)
        self.spin_charge_refine.setToolTip("0 = no refinement (setFields). 1–8 = refinement level (setRefinedFields). Sphere/Cylinder only.")
        self.spin_charge_outer_min = QSpinBox()
        self.spin_charge_outer_min.setRange(0, 10)
        self.spin_charge_outer_min.setValue(2)
        self.spin_charge_outer_min.setMaximumWidth(60)
        self.spin_charge_outer_min.setToolTip("Min level for charge outer refinement (snappy).")
        self.spin_charge_outer_max = QSpinBox()
        self.spin_charge_outer_max.setRange(0, 10)
        self.spin_charge_outer_max.setValue(3)
        self.spin_charge_outer_max.setMaximumWidth(60)
        self.spin_charge_outer_max.setToolTip("Max level for charge outer refinement (snappy).")
        self.spin_charge_outer_min.valueChanged.connect(self._validate_charge_outer_levels)
        self.spin_charge_outer_max.valueChanged.connect(self._validate_charge_outer_levels)
        
        for _l in (self.lbl_radius, self.lbl_aspect, self.lbl_cylinder_axis, self.lbl_length, self.lbl_width, self.lbl_height):
            _l.setMinimumWidth(LABEL_MINW)
        mass_lbl = self._lbl("Mass [kg]")
        mass_lbl.setMinimumWidth(LABEL_MINW)
        density_lbl = self._lbl("Density")
        density_lbl.setMinimumWidth(LABEL_MINW)

        wrap_geom = QWidget()
        f2a = QFormLayout(wrap_geom)
        f2a.setHorizontalSpacing(8)
        f2a.setVerticalSpacing(self.ROW_SPACING)
        f2a.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        f2a.addRow(self._lbl("Material"), mat_row)
        f2a.addRow(self._lbl("Shape"), self.c_shape)
        f2a.addRow(self.lbl_radius, _field_wrap(self.c_radius))
        f2a.addRow(self.lbl_aspect, _field_wrap(self.c_aspect))
        f2a.addRow(self.lbl_cylinder_axis, _field_wrap(self.c_cylinder_axis))
        f2a.addRow(self.lbl_length, _field_wrap(self.c_length))
        f2a.addRow(self.lbl_width, _field_wrap(self.c_width))
        f2a.addRow(self.lbl_height, _field_wrap(self.c_height))
        f2a.addRow(mass_lbl, _field_wrap(self.c_mass))
        f2a.addRow(density_lbl, _field_wrap(self.c_rho))
        self.combo_eos = QComboBox()
        self.combo_eos.setEditable(True)
        self.combo_eos.addItems(get_eos_options())
        self.combo_eos.setCurrentText("JWL")
        self.combo_eos.setToolTip("Equation of state model for products (phaseProperties). Discovered + loaded-case tokens.")
        f2a.addRow(self._lbl("Equation of State"), self.combo_eos)
        self.btn_charge_advanced = QPushButton("Advanced…")
        self.btn_charge_advanced.setToolTip("Activation model, thermodynamics/energy model, and phaseProperties options.")
        self.btn_charge_advanced.clicked.connect(self._open_charge_advanced_dialog)
        f2a.addRow("", self.btn_charge_advanced)
        charge_main.addWidget(wrap_geom)

        wrap_refine = QWidget()
        f2b = QFormLayout(wrap_refine)
        f2b.setHorizontalSpacing(8)
        f2b.setVerticalSpacing(self.ROW_SPACING)
        f2b.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.lbl_charge_refinement = QLabel("Charge pre-refinement")
        self.lbl_charge_refinement.setStyleSheet("font-weight: bold;")
        f2b.addRow(self.lbl_charge_refinement)
        f2b.addRow(self._lbl("Inside"), self.spin_charge_refine)
        outside_row = QWidget()
        outside_h = QHBoxLayout(outside_row)
        outside_h.setContentsMargins(0, 0, 0, 0)
        outside_h.addWidget(QLabel("Min"))
        outside_h.addWidget(self.spin_charge_outer_min)
        outside_h.addWidget(QLabel("Max"))
        outside_h.addWidget(self.spin_charge_outer_max)
        outside_h.addStretch()
        f2b.addRow(self._lbl("Outside"), outside_row)
        self.spin_transition_cells = QSpinBox()
        self.spin_transition_cells.setRange(1, 10)
        self.spin_transition_cells.setValue(2)
        self.spin_transition_cells.setToolTip("Cells between refinement levels for charge outside region (nCellsBetweenLevels-style graded transition in snappyHexMesh).")
        self.spin_transition_cells.valueChanged.connect(lambda: self._set_provenance_user("transition_cells"))
        f2b.addRow(self._lbl("Transition Cells"), self.spin_transition_cells)
        lbl_center = QLabel("Center (X, Y, Z)")
        f2b.addRow(lbl_center)
        center_wrap = QWidget()
        center_h = QHBoxLayout(center_wrap)
        center_h.setContentsMargins(0, 0, 0, 0)
        center_h.addWidget(self._tri(self.cx, self.cy, self.cz))
        center_h.addStretch()
        f2b.addRow(center_wrap)
        # Ignition mode: Center of Charge (use center) or Manual (user XYZ)
        self.combo_ignition_mode = QComboBox()
        self.combo_ignition_mode.addItems(["Center of Charge", "Manual"])
        self.combo_ignition_mode.setToolTip("Center of Charge = use charge center; Manual = use initiation point below.")
        self.combo_ignition_mode.currentTextChanged.connect(self._on_ignition_mode_changed)
        f2b.addRow(self._lbl("Ignition mode"), self.combo_ignition_mode)
        self.lbl_init_pt = QLabel("Initiation point (X, Y, Z)")
        self.lbl_init_pt.setToolTip("Detonation initiation point (used when Ignition mode = Manual).")
        f2b.addRow(self.lbl_init_pt)
        self.init_wrap = QWidget()
        init_h = QHBoxLayout(self.init_wrap)
        init_h.setContentsMargins(0, 0, 0, 0)
        init_h.addWidget(self._tri(self.init_ix, self.init_iy, self.init_iz))
        init_h.addStretch()
        f2b.addRow(self.init_wrap)
        self._on_ignition_mode_changed(self.combo_ignition_mode.currentText())
        self.spin_backup_factor = self._spin(0.1, 20.0, 1.0, 0.1, 2)
        self.spin_backup_factor.setToolTip("Internal: backup radius factor (not user-facing).")
        self.lbl_backup_factor = self._lbl("Backup radius factor")
        f2b.addRow(self.lbl_backup_factor, self.spin_backup_factor)
        self.lbl_backup_factor.setVisible(False)
        self.spin_backup_factor.setVisible(False)
        self._refine_interval = 3
        self._lower_refine_threshold = 0.1
        self._unrefine_threshold = 0.1
        self._n_buffer_layers_dynamic = 2
        self._enable_balancing = False
        self._refine_indicator_field = "densityGradient"
        self._obstacle_feature_angle = 120
        self._obstacle_cells_between_levels = 2
        self._obstacle_snap_iter = 100
        self._obstacle_feature_snap_iter = 15
        self._provenance = {}  # key -> "LOADED" | "USER" | "UNSET" for optional 3D fields
        # Geometry & mesh quality (snappy/surfaceFeatures); None = UNSET
        self._mesh_included_angle = None
        self._mesh_n_smooth_patch = None
        self._mesh_snap_tolerance = None
        self._mesh_n_solve_iter = None
        self._mesh_n_relax_iter = None
        self._mesh_n_feature_snap_iter = None
        self._mesh_explicit_feature_snap = None
        self._mesh_implicit_feature_snap = None
        self._mesh_multi_region_feature_snap = None
        self._mesh_n_cells_between_levels = None
        self._mesh_resolve_feature_angle = None
        self._mesh_max_non_ortho = None
        self._mesh_max_boundary_skewness = None
        self._mesh_max_internal_skewness = None
        self._mesh_max_concave = None
        self._mesh_min_vol = None
        self._mesh_min_tet_quality = None
        self._mesh_min_twist = None
        self._mesh_min_determinant = None
        self._mesh_min_face_weight = None
        self._mesh_min_vol_ratio = None
        self._mesh_n_smooth_scale = None
        self._mesh_error_reduction = None
        self._mesh_relaxed_max_non_ortho = None
        # EOS / activation / thermo (phaseProperties)
        self._eos_model = "JWL"
        self._activation_model_ui = "pressureBased"
        self._thermo_model = "ePolynomial"
        self._thermo_model_air = "eConst"
        # Decomposition
        self._decomposition_method = "scotch"
        self._decomposition_simple_n = (2, 2, 1)
        self._decomposition_simple_delta = 0.001
        charge_main.addWidget(wrap_refine)
        l.addWidget(c)
        
        a = QGroupBox("Atmosphere"); f3 = QFormLayout(a)
        f3.setHorizontalSpacing(8)
        f3.setVerticalSpacing(self.ROW_SPACING)
        self.p0 = self._spin(1, 1e6, 101325, 100, 0, 120)
        self.t0 = self._spin(1, 5000, 288, 1, 0, 120)
        f3.addRow(self._lbl("Pressure"), self.p0)
        f3.addRow(self._lbl("Temp"), self.t0)
        l.addWidget(a)

        # --- Initialize Method (Standard vs Remap from pre-cursor) ---
        self.grp_init_method = QGroupBox("Initialize Method")
        fm = QFormLayout(self.grp_init_method)
        fm.setSpacing(0)
        fm.setVerticalSpacing(self.ROW_SPACING)
        self.rad_init_standard = QRadioButton("Standard (0.orig)")
        self.rad_init_remap = QRadioButton("Remap from Pre-cursor")
        self.rad_init_standard.setChecked(True)
        self.rad_init_standard.setToolTip("Use 0.orig template and setFields.")
        self.rad_init_remap.setToolTip("Map ICs from 1D case by radial distance (Autodyn-style). Select 1D case root folder.")
        self._init_method_group = QButtonGroup(self)
        self._init_method_group.addButton(self.rad_init_standard)
        self._init_method_group.addButton(self.rad_init_remap)
        self._init_method_group.buttonToggled.connect(self._on_init_method_toggled)
        self.rad_init_standard.toggled.connect(self._update_ui_state)
        self.rad_init_remap.toggled.connect(self._update_ui_state)
        self.btn_remap_edit = QPushButton("Edit…")
        self.btn_remap_edit.setFixedWidth(100)
        self.btn_remap_edit.setToolTip("Configure remap source and time.")
        self.btn_remap_edit.setEnabled(False)
        self.btn_remap_edit.clicked.connect(self._open_remap_config_dialog)
        fm.addRow(QLabel("Method:"))
        row_std = QWidget()
        row_std_h = QHBoxLayout(row_std)
        row_std_h.setContentsMargins(0, 0, 0, 0); row_std_h.setSpacing(0)
        row_std_h.addWidget(self.rad_init_standard)
        row_std_h.addStretch()
        fm.addRow(row_std)
        row_remap = QWidget()
        row_remap_h = QHBoxLayout(row_remap)
        row_remap_h.setContentsMargins(0, 0, 0, 0); row_remap_h.setSpacing(0)
        row_remap_h.addWidget(self.rad_init_remap)
        row_remap_h.addWidget(self.btn_remap_edit)
        row_remap_h.addStretch()
        fm.addRow(row_remap)
        self.spin_remap_ox = QDoubleSpinBox()
        self.spin_remap_oy = QDoubleSpinBox()
        self.spin_remap_oz = QDoubleSpinBox()
        for s in (self.spin_remap_ox, self.spin_remap_oy, self.spin_remap_oz):
            s.setRange(-100, 100)
            s.setDecimals(4)
            s.setSingleStep(0.1)
            s.setValue(0.0)
            s.setMaximumWidth(116)
        self.spin_remap_ox.setToolTip("X coordinate of explosion center for radial remap.")
        self.spin_remap_oy.setToolTip("Y coordinate of explosion center for radial remap.")
        self.spin_remap_oz.setToolTip("Z coordinate of explosion center for radial remap.")
        row_origin = QWidget()
        row_origin_h = QHBoxLayout(row_origin)
        row_origin_h.setContentsMargins(0, 0, 0, 0); row_origin_h.setSpacing(0)
        row_origin_h.addWidget(QLabel("X"))
        row_origin_h.addWidget(self.spin_remap_ox)
        row_origin_h.addWidget(QLabel("Y"))
        row_origin_h.addWidget(self.spin_remap_oy)
        row_origin_h.addWidget(QLabel("Z"))
        row_origin_h.addWidget(self.spin_remap_oz)
        row_origin_h.addStretch()
        fm.addRow(QLabel("Remap Origin (X, Y, Z)"))
        fm.addRow(row_origin)
        # Mode A checkbox removed: remap always uses activationModel none (no mass check needed)
        l.addWidget(self.grp_init_method)

        # Stored remap config (used by get_case_inputs and by RemapConfigDialog load/save)
        self._remap_source_type = "1D"
        self._remap_case_path = ""
        self._remap_origin = (0.0, 0.0, 0.0)
        self._remap_time_mode = "latest"
        self._remap_specific_time = "1e-4"

        self._update_ui_state()

    def _update_ui_state(self) -> None:
        """When Remap is selected, disable Charge group so setFields won't overwrite mapped ICs."""
        remap = self.rad_init_remap.isChecked()
        self.grp_charge.setEnabled(not remap)
        self.btn_remap_edit.setEnabled(remap)

    def _on_init_method_toggled(self, _btn, checked: bool) -> None:
        self._update_ui_state()

    def _open_remap_config_dialog(self) -> None:
        initial = {
            "remap_source_type": self._remap_source_type,
            "remap_case_path": self._remap_case_path,
            "remap_origin": (self.spin_remap_ox.value(), self.spin_remap_oy.value(), self.spin_remap_oz.value()),
            "remap_time_mode": self._remap_time_mode,
            "remap_specific_time": self._remap_specific_time,
        }
        dlg = RemapConfigDialog(self, initial=initial)
        if dlg.exec_() == QDialog.Accepted:
            cfg = dlg.get_remap_config()
            self._remap_source_type = cfg["remap_source_type"]
            self._remap_case_path = cfg["remap_case_path"]
            self._remap_origin = cfg.get("remap_origin", (0.0, 0.0, 0.0))
            if len(self._remap_origin) >= 3:
                self.spin_remap_ox.setValue(float(self._remap_origin[0]))
                self.spin_remap_oy.setValue(float(self._remap_origin[1]))
                self.spin_remap_oz.setValue(float(self._remap_origin[2]))
            self._remap_time_mode = cfg["remap_time_mode"]
            self._remap_specific_time = cfg["remap_specific_time"]

    def _on_ignition_mode_changed(self, mode: str) -> None:
        """Show manual initiation point (X,Y,Z) only when Ignition mode is Manual."""
        is_manual = (mode == "Manual")
        self.lbl_init_pt.setVisible(is_manual)
        self.init_wrap.setVisible(is_manual)

    def _estimate_charge_cells(self) -> float:
        """Pre-flight estimate of number of cells in charge region (geometry + refinement). Returns 0 if invalid."""
        try:
            cell_size = self.scell.value()
            level = self.spin_charge_refine.value()
            mass = self.c_mass.value()
            rho = self.c_rho.value()
            if cell_size <= 0 or rho <= 0:
                return 0.0
            effective_cell = cell_size / (2.0 ** level) if level >= 0 else cell_size
            if effective_cell <= 0:
                return 0.0
            shape = self.c_shape.currentText()
            if shape == "Sphere":
                # V = (4/3)*pi*r^3, r from mass/rho = V => r = (3*m/(4*pi*rho))^(1/3)
                import math
                vol = mass / rho
                if vol <= 0:
                    return 0.0
                r = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
                vol = (4.0 / 3.0) * math.pi * (r ** 3)
            elif shape == "Cylinder":
                r = self.c_radius.value()
                if r <= 0:
                    return 0.0
                length = self.c_length.value() if hasattr(self, "c_length") and self.c_length.value() > 1e-9 else (2.0 * r * (self.c_aspect.value() if self.c_aspect.value() > 1e-9 else 2.5))
                vol = math.pi * (r ** 2) * length
            else:
                # Cuboid
                L = self.c_length.value() if self.c_length.value() > 1e-9 else 0.1
                W = self.c_width.value() if self.c_width.value() > 1e-9 else 0.1
                H = self.c_height.value() if self.c_height.value() > 1e-9 else 0.1
                vol = L * W * H
            return vol / (effective_cell ** 3)
        except Exception:
            return 0.0

    def _update_estimated_charge_cells_display(self) -> None:
        """Update Info panel with estimated charge cells and red warning if < 8."""
        est = self._estimate_charge_cells()
        self.lbl_est_charge_cells.setText(f"Est. charge cells: {int(est):,}" if est > 0 else "Est. charge cells: —")
        if est > 0 and est < 8:
            self.lbl_charge_resolution_warning.setText("Warning: Charge resolution is too low. Blast may fail.")
        else:
            self.lbl_charge_resolution_warning.setText("")

    def _pair(self, a, b):
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0); h.setSpacing(10)
        h.addWidget(a); h.addWidget(b)
        return w
    def _tri(self, a, b, c):
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0); h.setSpacing(10)
        h.addWidget(a); h.addWidget(b); h.addWidget(c)
        return w

    def _build_obstacles_tab(self, parent):
        l = QVBoxLayout(parent)
        btns = QHBoxLayout()
        self.btn_add = QPushButton("Import STL")
        self.btn_clr = QPushButton("Clear")
        self.btn_del = QPushButton("Delete Selected")
        self.btn_up = QPushButton("↑ Move Up")
        self.btn_down = QPushButton("↓ Move Down")
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_clr)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_up)
        btns.addWidget(self.btn_down)
        l.addLayout(btns)
        self.btn_del.clicked.connect(self._del_stl)
        self.btn_up.clicked.connect(self._move_up_stl)
        self.btn_down.clicked.connect(self._move_down_stl)

        self.tbl_obs = QTableWidget(0, 6)
        self.tbl_obs.setHorizontalHeaderLabels(["On", "File", "Scale", "Off X", "Off Y", "Off Z"])
        self.tbl_obs.horizontalHeader().setStretchLastSection(True)
        l.addWidget(self.tbl_obs)
        self.btn_add.clicked.connect(self._add_stl); self.btn_clr.clicked.connect(self._clear_stl)
        self.tbl_obs.cellChanged.connect(self._on_table_change)

    def _build_exec_tab(self, parent):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner = QWidget()
        l = QHBoxLayout(inner)
        g1 = QGroupBox("Simulation Control"); f1 = QFormLayout(g1)
        f1.setVerticalSpacing(3)     # מרווח בין שורות
        f1.setHorizontalSpacing(10)   # מרווח בין label לשדה
        self.spin_end = self._spin(0, 100, 0.030, 0.001, 4)
        self.spin_end.setMaximumWidth(120)
        # spin_cfl is created in _build_model_tab (Simulation Parameters group)
        self.spin_cores = QSpinBox()
        self.spin_cores.setRange(1, 128)
        self.spin_cores.setValue(4)
        self.spin_cores.setMaximumWidth(120)
        self.spin_cores.setToolTip("Number of CPU cores for parallel solving. Parallel scaling typically needs ~50k–100k cells per core for efficiency. Dynamic mesh refinement is fully supported in parallel mode.")
        self.btn_decomposition_edit = QPushButton("Edit…")
        self.btn_decomposition_edit.setToolTip("Decomposition method and coefficients (decomposeParDict).")
        self.btn_decomposition_edit.clicked.connect(self._open_decomposition_dialog)
        self.spin_write = QSpinBox()
        self.spin_write.setRange(1, 1000000)
        self.spin_write.setValue(100)
        self.spin_write.setMaximumWidth(120)
        self.spin_write.setToolTip("Write results every N time steps (when Write control = timeStep).")
        self.combo_write_control = QComboBox()
        self.combo_write_control.addItems(["timeStep", "adjustableRunTime"])
        self.combo_write_control.setMaximumWidth(160)
        self.combo_write_control.setToolTip("timeStep = write every N steps. adjustableRunTime = write every T seconds (simulation time).")
        self.combo_write_control.currentTextChanged.connect(self._on_write_control_changed)
        self.spin_write_time = QDoubleSpinBox()
        self.spin_write_time.setRange(1e-7, 1.0)
        self.spin_write_time.setValue(5e-5)
        self.spin_write_time.setDecimals(7)
        self.spin_write_time.setSingleStep(1e-5)
        self.spin_write_time.setMaximumWidth(120)
        self.spin_write_time.setToolTip("Write results every T seconds of simulation time (when Write control = adjustableRunTime).")
        self.spin_cycle_write = QSpinBox()
        self.spin_cycle_write.setRange(0, 1000)
        self.spin_cycle_write.setValue(0)
        self.spin_cycle_write.setMaximumWidth(80)
        self.spin_cycle_write.setToolTip("cycleWrite in controlDict (0 = off).")
        self.lbl_write_interval = self._lbl("Write interval (steps)")
        self.lbl_write_time = self._lbl("Write interval [s]")

        self.spin_cores.valueChanged.connect(self._on_cores_changed)
        self.scell.valueChanged.connect(self._update_calculated_dt_label)
        self.spin_refine_max.valueChanged.connect(self._update_calculated_dt_label)
        self.spin_cores.valueChanged.connect(self._update_calculated_dt_label)

        f1.addRow("End Time [s]", self.spin_end)
        cores_row = QWidget()
        cores_h = QHBoxLayout(cores_row)
        cores_h.setContentsMargins(0, 0, 0, 0)
        cores_h.addWidget(self.spin_cores)
        cores_h.addWidget(self.btn_decomposition_edit)
        cores_h.addStretch()
        f1.addRow("Cores", cores_row)
        self.chk_obstacle_refine = QCheckBox()
        self.chk_obstacle_refine.setChecked(True)
        self.chk_obstacle_refine.setToolTip("Static refinement around obstacle STL surfaces during meshing (snappy).")
        self.chk_obstacle_refine.toggled.connect(lambda: self._set_provenance_user("enable_obstacle_refine"))
        self.spin_obstacle_refine_min = QSpinBox()
        self.spin_obstacle_refine_min.setRange(0, 10)
        self.spin_obstacle_refine_min.setValue(1)
        self.spin_obstacle_refine_min.setMaximumWidth(60)
        self.spin_obstacle_refine_max = QSpinBox()
        self.spin_obstacle_refine_max.setRange(0, 10)
        self.spin_obstacle_refine_max.setValue(2)
        self.spin_obstacle_refine_max.setMaximumWidth(60)
        obstacle_refine_row = QWidget()
        obst_h = QHBoxLayout(obstacle_refine_row)
        obst_h.setContentsMargins(0, 0, 0, 0)
        obst_h.addWidget(self.chk_obstacle_refine)
        obst_h.addWidget(QLabel("Levels:"))
        obst_h.addWidget(self.spin_obstacle_refine_min)
        obst_h.addWidget(self.spin_obstacle_refine_max)
        obst_h.addStretch()
        f1.addRow("Obstacle refine", obstacle_refine_row)
        self.btn_mesh_properties = QPushButton("Mesh Properties…")
        self.btn_mesh_properties.setToolTip("Advanced mesh parameters (AMR and obstacle refine).")
        self.btn_mesh_properties.clicked.connect(self._open_mesh_properties_dialog)
        f1.addRow("", self.btn_mesh_properties)
        f1.addRow(self._lbl("Write control"), self.combo_write_control)
        wrap_steps = QWidget()
        h_steps = QHBoxLayout(wrap_steps)
        h_steps.setContentsMargins(0, 0, 0, 0)
        h_steps.addWidget(self.lbl_write_interval)
        h_steps.addWidget(self.spin_write)
        h_steps.addStretch()
        wrap_time = QWidget()
        h_time = QHBoxLayout(wrap_time)
        h_time.setContentsMargins(0, 0, 0, 0)
        h_time.addWidget(self.lbl_write_time)
        h_time.addWidget(self.spin_write_time)
        h_time.addStretch()
        f1.addRow(wrap_steps)
        f1.addRow(wrap_time)
        f1.addRow(self._lbl("cycleWrite"), self.spin_cycle_write)
        self.wrap_write_steps = wrap_steps
        self.wrap_write_time = wrap_time
        self._on_write_control_changed(self.combo_write_control.currentText())
        l.addWidget(g1)

        self._on_cores_changed(self.spin_cores.value())
        self._on_mesh_mode_changed()
        self._update_calculated_dt_label()

        g2 = QGroupBox("Actions"); v2 = QVBoxLayout(g2)
        self.btn_init = QPushButton("Initialize Model (Step 0)"); self.btn_init.clicked.connect(self._on_init_clicked)
        self.btn_exact_1 = QPushButton("exact 1")
        self.btn_exact_1.setToolTip("Run exactly one time step and stop.")
        self.btn_exact_1.clicked.connect(self.sig_request_run_exact_1.emit)
        self.btn_exact_end = QPushButton("exact END")
        self.btn_exact_end.setToolTip("Continue run until stop or end time.")
        self.btn_exact_end.clicked.connect(self.sig_request_run_exact_end.emit)
        self.btn_save_remap = QPushButton("Save 3D remap…")
        self.btn_save_remap.setToolTip("Save current 3D state for remap. Still in progress.")
        self.btn_save_remap.clicked.connect(lambda: QMessageBox.information(self, "Save 3D remap", "Save 3D remap: Still in progress."))
        self.btn_run = QPushButton("▶ Run / Resume"); self.btn_run.clicked.connect(self.sig_request_run.emit)
        self.btn_stop = QPushButton("⏸ Interrupt"); self.btn_stop.clicked.connect(self.sig_request_stop.emit)
        self.btn_init.setStyleSheet("background-color: #3498db; color: white; padding: 5px;")
        self.btn_exact_1.setStyleSheet("background-color: #9b59b6; color: white; padding: 4px;")
        self.btn_exact_end.setStyleSheet("background-color: #1abc9c; color: white; padding: 4px;")
        self.btn_run.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 5px;")
        self.btn_stop.setStyleSheet("background-color: #e67e22; color: white; padding: 5px;")
        v2.addWidget(self.btn_init)
        v2.addWidget(self.btn_exact_1)
        v2.addWidget(self.btn_exact_end)
        v2.addWidget(self.btn_save_remap)
        v2.addWidget(self.btn_run)
        v2.addWidget(self.btn_stop)
        v2.addStretch()
        l.addWidget(g2)
        
        g3 = QGroupBox("Field Display"); f3 = QFormLayout(g3)
        # מיפוי שם תצוגה → שם שדה פנימי
        self._field_display_map = {
            "Pressure": "p",
            "Density": "rho",
            "Energy": "alpha.c4",
            "Peak Overpressure": "p",
            "Peak Impulse": "p",
        }
        self.cmb_field = QComboBox()
        self.cmb_field.addItems(list(self._field_display_map.keys()))
        self.cmb_field.setCurrentIndex(0)
        self.cmb_field.currentTextChanged.connect(self._on_field_combo_changed)
        f3.addRow("Field", self.cmb_field)
        self.chk_auto_range = QCheckBox("Auto Range"); self.chk_auto_range.setChecked(True)
        self.chk_log_scale = QCheckBox("Log Scale")
        self.chk_log_scale.setToolTip("Use logarithmic scaling for contour mapping.")
        self.chk_log_scale.toggled.connect(self._on_log_scale_changed)
        self.spin_min_val = self._spin(-1e12, 1e12, 1e5, 1000, 0)
        self.spin_max_val = self._spin(-1e12, 1e12, 2e5, 1000, 0)
        self.spin_min_val.setEnabled(False); self.spin_max_val.setEnabled(False)
        self.chk_auto_range.toggled.connect(lambda c: (self.spin_min_val.setEnabled(not c), self.spin_max_val.setEnabled(not c), self._on_range_change()))
        self.spin_min_val.editingFinished.connect(self._on_range_change)
        self.spin_max_val.editingFinished.connect(self._on_range_change)
        f3.addRow(self.chk_auto_range)
        f3.addRow(self.chk_log_scale)
        f3.addRow("Min", self.spin_min_val)
        f3.addRow("Max", self.spin_max_val)
        l.addWidget(g3)
        scroll.setWidget(inner)
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _on_field_combo_changed(self, display_name):
        field_name = self._field_display_map.get(display_name, "p")
        self._on_field_change_req(field_name)

    def _on_log_scale_changed(self, checked):
        if self.viewer:
            self.viewer.set_log_scale(checked)
            self.viewer.refresh_view()

    def _on_cell_count_updated(self, count: int):
        """Update Cells label when viewer loads a new time step (e.g. AMR)."""
        self.lbl_cells.setText(f"Cells: {count:,}")

    def _on_view_option_changed(self, _checked=None):
        """Sync all Viewport Options from UI to viewer, then refresh (preview or loaded case)."""
        if not self.viewer:
            return
        self.viewer.show_mesh_lines = self.chk_mesh.isChecked()
        self.viewer.show_boundaries = self.chk_bound.isChecked()
        self.viewer.show_obstacles = self.chk_show_obstacles.isChecked()
        self.viewer.show_obstacles_wireframe_only = self.chk_obstacles_wireframe.isChecked()
        self.viewer.show_tracers = self.chk_tracers.isChecked()
        self.viewer.toggle_probes(
            self.chk_probes.isChecked(),
            [(p.x, p.y, p.z) for p in self.probes_model.probes()],
        )
        self._apply_view_options_refresh()

    def _apply_view_options_refresh(self):
        """Refresh viewer: force redraw if case loaded (Step 0 / running), else redraw preview."""
        if not self.viewer:
            return
        if self.viewer.current_case_dir:
            self.viewer.force_refresh_view()
        else:
            self.viewer.refresh_view()

    def _on_show_probes_toggled(self, checked):
        self._refresh_probes_display()

    def _refresh_probes_display(self):
        if not self.viewer:
            return
        checked = self.chk_probes.isChecked()
        probes_data = [(p.x, p.y, p.z) for p in self.probes_model.probes()]
        self.viewer.toggle_probes(checked, probes_data)

    def _del_stl(self):
        r = self.tbl_obs.currentRow()
        if r < 0 or r >= len(self.obstacles):
            return
        self.obstacles.pop(r)
        self._refresh_table()
        self._update_preview()

    def _move_up_stl(self):
        r = self.tbl_obs.currentRow()
        if r <= 0 or r >= len(self.obstacles):
            return
        self.obstacles[r], self.obstacles[r - 1] = self.obstacles[r - 1], self.obstacles[r]
        self._refresh_table()
        self.tbl_obs.setCurrentCell(r - 1, 0)
        self._update_preview()

    def _move_down_stl(self):
        r = self.tbl_obs.currentRow()
        if r < 0 or r >= len(self.obstacles) - 1:
            return
        self.obstacles[r], self.obstacles[r + 1] = self.obstacles[r + 1], self.obstacles[r]
        self._refresh_table()
        self.tbl_obs.setCurrentCell(r + 1, 0)
        self._update_preview()

    def _on_cores_changed(self, val):
        # Mesh refinement is supported in parallel mode (dynamicMeshDict handles it)
        # No restriction needed - user can enable refinement with any number of cores
        self._on_mesh_mode_changed()

    def _validate_refine_levels(self):
        """Ensure refine_min <= refine_max for snappyHexMesh."""
        min_val = self.spin_refine_min.value()
        max_val = self.spin_refine_max.value()
        if min_val > max_val:
            # Auto-correct: set min = max
            self.spin_refine_min.blockSignals(True)
            self.spin_refine_min.setValue(max_val)
            self.spin_refine_min.blockSignals(False)
        self._update_calculated_dt_label()

    def _on_dyn_refine_max_changed(self):
        """Sync AMR maxRefinement with user-visible Levels max (so explicit change is written)."""
        self._dyn_refine_max = self.spin_refine_max.value()

    def _validate_charge_outer_levels(self):
        """Ensure charge outer min <= max."""
        a, b = self.spin_charge_outer_min.value(), self.spin_charge_outer_max.value()
        if a > b:
            self.spin_charge_outer_min.blockSignals(True)
            self.spin_charge_outer_min.setValue(b)
            self.spin_charge_outer_min.blockSignals(False)

    def _compute_safe_dt(self) -> float:
        """Compute safe initial delta_t from smallest cell and detonation velocity.

        The smallest cell in the mesh is determined by the *larger* of:
          - AMR obstacle refinement (spin_refine_max)
          - Charge internal refinement (effective level from setRefinedFields)
        Both divide the base cell_size by 2^level, so the effective smallest
        cell is cell_size / 2^max(amr_level, charge_level).

        The charge refinement level is auto-raised when the charge sphere is
        smaller than the base cell (same logic as effective_charge_refine in
        generator_3d.py).
        """
        V_max = 9000.0  # m/s, conservative for C4/TNT/Custom
        CFL_init = 0.1  # Conservative initial CFL (matches building3D ~1e-7)
        cell_size = max(1e-9, self.scell.value())

        # AMR refinement level (obstacle mesh refinement)
        amr_level = 0
        refine = self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()
        if refine:
            amr_level = max(0, self.spin_refine_max.value())

        # Charge internal refinement level (setRefinedFields)
        charge_level = max(0, self.spin_charge_refine.value())

        # Auto-raise charge level when charge is smaller than base cell
        # (mirrors effective_charge_refine in generator_3d.py)
        shape = self.c_shape.currentText()
        if shape in ("Sphere", "Cylinder"):
            mass_val = self.c_mass.value()
            rho_val = self.c_rho.value() or 1600.0
            if rho_val > 0 and mass_val > 0:
                vol = mass_val / rho_val
                if shape == "Sphere":
                    cr = ((3.0 * vol) / (4.0 * math.pi)) ** (1.0 / 3.0)
                else:
                    cr = getattr(self, "c_radius_display", None)
                    cr = cr.value() if cr and cr.value() > 0 else 0.05
                if cr > 0 and cell_size > 2 * cr:
                    auto_lvl = math.ceil(math.log2(cell_size / (2 * cr)))
                    charge_level = max(charge_level, auto_lvl)

        # The effective finest cell uses the highest refinement level
        effective_max = max(amr_level, charge_level)
        dx_min = cell_size / (2.0 ** effective_max) if effective_max > 0 else cell_size

        return CFL_init * dx_min / V_max

    def _update_calculated_dt_label(self):
        safe_dt = self._compute_safe_dt()
        self.initial_dt_changed.emit(safe_dt)

    def _on_write_control_changed(self, write_control: str):
        is_time_step = write_control == "timeStep"
        self.wrap_write_steps.setVisible(is_time_step)
        self.wrap_write_time.setVisible(not is_time_step)

    def _on_mesh_mode_changed(self):
        """Dyn Mesh: enable Dyn Mesh levels and full Charge pre-refinement group. Fixed Mesh: disable all of them."""
        dyn = self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()
        self.spin_refine_min.setEnabled(dyn)
        self.spin_refine_max.setEnabled(dyn)
        # Charge pre-refinement group (single source of truth for startup charge-region refinement)
        self.lbl_charge_refinement.setEnabled(dyn)
        self.spin_charge_refine.setEnabled(dyn)
        self.spin_charge_outer_min.setEnabled(dyn)
        self.spin_charge_outer_max.setEnabled(dyn)
        self.spin_transition_cells.setEnabled(dyn)
        self._update_calculated_dt_label()

    def _build_view_tab(self, parent):
        l = QHBoxLayout(parent)
        g1 = QGroupBox("Camera"); v1 = QVBoxLayout(g1)
        self.rad_persp = QRadioButton("Perspective"); self.rad_persp.setChecked(True)
        self.rad_ortho = QRadioButton("Parallel (Ortho)")
        bg = QButtonGroup(self); bg.addButton(self.rad_persp); bg.addButton(self.rad_ortho)
        self.rad_persp.toggled.connect(lambda: self.viewer.toggle_parallel_projection(False))
        self.rad_ortho.toggled.connect(lambda: self.viewer.toggle_parallel_projection(True))
        v1.addWidget(self.rad_persp); v1.addWidget(self.rad_ortho)
        
        hb = QHBoxLayout()
        for name in ["Iso", "Top", "Bottom", "Left", "Right", "Front", "Back"]:
            b = QPushButton(name); b.clicked.connect(lambda _, n=name: self.viewer.set_standard_view(n))
            hb.addWidget(b)
        v1.addLayout(hb); v1.addStretch()
        l.addWidget(g1)
        
        g2 = QGroupBox("Visibility"); v2 = QVBoxLayout(g2)
        self.chk_mesh = QCheckBox("Show Mesh Lines")
        self.chk_mesh.toggled.connect(self._on_view_option_changed)
        self.chk_bound = QCheckBox("Show Boundaries")
        self.chk_bound.setChecked(True)
        self.chk_bound.toggled.connect(self._on_view_option_changed)
        self.chk_show_obstacles = QCheckBox("Show Obstacles")
        self.chk_show_obstacles.setChecked(True)
        self.chk_show_obstacles.setToolTip("Toggle obstacle geometry (solid or wireframe). Hide to see flow inside/behind obstacles.")
        self.chk_show_obstacles.toggled.connect(self._on_view_option_changed)
        self.chk_obstacles_wireframe = QCheckBox("Obstacles: Wireframe only")
        self.chk_obstacles_wireframe.setChecked(False)
        self.chk_obstacles_wireframe.setToolTip("Draw obstacles as wireframe/edges instead of solid. Helps see flow inside or behind.")
        self.chk_obstacles_wireframe.toggled.connect(self._on_view_option_changed)
        self.chk_tracers = QCheckBox("Show Tracers")
        self.chk_tracers.toggled.connect(self._on_view_option_changed)
        self.chk_probes = QCheckBox("Show Probes / Gauges")
        self.chk_probes.toggled.connect(self._on_view_option_changed)
        v2.addWidget(self.chk_mesh)
        v2.addWidget(self.chk_bound)
        v2.addWidget(self.chk_show_obstacles)
        v2.addWidget(self.chk_obstacles_wireframe)
        v2.addWidget(self.chk_tracers)
        v2.addWidget(self.chk_probes)
        v2.addStretch()
        l.addWidget(g2); l.addStretch()

    def _build_section_tab(self, parent):
        l = QHBoxLayout(parent)
        self.tbl_sec = QTableWidget(0, 5)
        self.tbl_sec.setHorizontalHeaderLabels(["On", "Name", "Plane", "Position [m]", "Opacity"])
        self.tbl_sec.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_sec.cellChanged.connect(self._update_sections_from_table)
        l.addWidget(self.tbl_sec, stretch=2)
        v = QVBoxLayout()
        self.btn_add_sec = QPushButton("Add Section"); self.btn_del_sec = QPushButton("Remove Selected")
        self.btn_add_sec.clicked.connect(self._add_default_section); self.btn_del_sec.clicked.connect(self._del_section)
        v.addWidget(self.btn_add_sec); v.addWidget(self.btn_del_sec); v.addStretch()
        l.addLayout(v)

    def _connect_signals(self):
        widgets = [self.sx1, self.sx2, self.sy1, self.sy2, self.sz1, self.sz2, self.scell,
                   self.c_mass, self.c_rho, self.cx, self.cy, self.cz]
        for w in widgets: w.valueChanged.connect(self._update_preview)
        self.c_shape.currentIndexChanged.connect(self._update_preview)
        self.c_mass.valueChanged.connect(self._update_charge_radius)
        self.c_rho.valueChanged.connect(self._update_charge_radius)
        self.c_aspect.valueChanged.connect(self._update_charge_radius)
        self.c_length.valueChanged.connect(self._update_cuboid_height)
        self.c_width.valueChanged.connect(self._update_cuboid_height)
        
        # TIKUN: Connect Domain Bounds changes to Section updates!
        self.sx1.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sx2.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sy1.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sy2.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sz1.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sz2.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        # Pre-flight estimated charge cells (and warning when < 8)
        for w in [self.scell, self.spin_charge_refine, self.c_mass, self.c_rho, self.c_radius, self.c_length, self.c_width, self.c_height, self.c_aspect]:
            w.valueChanged.connect(self._update_estimated_charge_cells_display)
        self.c_shape.currentIndexChanged.connect(self._update_estimated_charge_cells_display)

    def _update_edit_button_visibility(self):
        is_custom = self.c_mat.currentText() == "Custom"
        self.btn_edit_custom.setEnabled(is_custom)

    def _open_custom_material_dialog(self):
        d = QDialog(self)
        d.setWindowTitle("Custom Material — JWL Parameters")
        layout = QVBoxLayout(d)
        form = QFormLayout()
        custom = self.materials_db["Custom"]
        spin_rho = QDoubleSpinBox(); spin_rho.setRange(100, 1e4); spin_rho.setValue(custom["rho"]); spin_rho.setDecimals(0)
        spin_E0 = QDoubleSpinBox(); spin_E0.setRange(1e5, 1e8); spin_E0.setValue(custom["energy"]); spin_E0.setDecimals(2); spin_E0.setSingleStep(1e5)
        spin_A = QDoubleSpinBox(); spin_A.setRange(1e8, 1e12); spin_A.setValue(custom["A"]); spin_A.setDecimals(2); spin_A.setSingleStep(1e9)
        spin_B = QDoubleSpinBox(); spin_B.setRange(1e6, 1e12); spin_B.setValue(custom["B"]); spin_B.setDecimals(2); spin_B.setSingleStep(1e8)
        spin_R1 = QDoubleSpinBox(); spin_R1.setRange(1, 10); spin_R1.setValue(custom["R1"]); spin_R1.setDecimals(2)
        spin_R2 = QDoubleSpinBox(); spin_R2.setRange(0.1, 5); spin_R2.setValue(custom["R2"]); spin_R2.setDecimals(2)
        spin_omega = QDoubleSpinBox(); spin_omega.setRange(0.1, 1); spin_omega.setValue(custom["omega"]); spin_omega.setDecimals(2)
        form.addRow("Density ρ [kg/m³]", spin_rho)
        form.addRow("Energy E0 [J/kg]", spin_E0)
        form.addRow("JWL A [Pa]", spin_A)
        form.addRow("JWL B [Pa]", spin_B)
        form.addRow("JWL R1", spin_R1)
        form.addRow("JWL R2", spin_R2)
        form.addRow("JWL ω", spin_omega)
        layout.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept); bb.rejected.connect(d.reject)
        layout.addWidget(bb)
        if d.exec_() == QDialog.Accepted:
            self.materials_db["Custom"] = {
                "rho": spin_rho.value(), "energy": spin_E0.value(),
                "A": spin_A.value(), "B": spin_B.value(),
                "R1": spin_R1.value(), "R2": spin_R2.value(), "omega": spin_omega.value(),
            }
            self._on_material_changed("Custom")
            self._update_preview()

    def _open_mesh_properties_dialog(self):
        """Mesh Properties…: AMR + Obstacle tab and Geometry & Mesh Quality tab."""
        d = QDialog(self)
        d.setWindowTitle("Mesh Properties…")
        layout = QVBoxLayout(d)
        tabs = QTabWidget()

        # Tab 0: AMR & Obstacle refine
        tab0 = QWidget()
        v0 = QVBoxLayout(tab0)
        scroll0 = QScrollArea()
        scroll0.setWidgetResizable(True)
        inner0 = QWidget()
        v = QVBoxLayout(inner0)

        # Buffer layers (setFieldsDict) – advanced
        grp_buf = QGroupBox("Advanced (setFieldsDict)")
        f_buf = QFormLayout(grp_buf)
        spin_buf_setfields = QSpinBox()
        spin_buf_setfields.setRange(0, 20)
        spin_buf_setfields.setValue(getattr(self, "_buffer_layers", 2))
        spin_buf_setfields.setToolTip("nBufferLayers in setFieldsDict. Default 2 (building3D-style).")
        f_buf.addRow("Buffer layers (setFields)", spin_buf_setfields)
        f_buf.addRow("", QLabel("nBufferLayers for charge/refinement regions (building3D: 2)."))
        v.addWidget(grp_buf)

        # Group: Dyn Refine (AMR) – Advanced
        grp_amr = QGroupBox("Dyn Refine (AMR) – Advanced")
        f_amr = QFormLayout(grp_amr)
        spin_ref_int = QSpinBox()
        spin_ref_int.setRange(1, 100)
        spin_ref_int.setValue(self._refine_interval)
        f_amr.addRow("Refine interval", spin_ref_int)
        f_amr.addRow("", QLabel("How often to refine (building3D: 3)."))
        spin_lower = QDoubleSpinBox()
        spin_lower.setRange(0.01, 1.0)
        spin_lower.setValue(self._lower_refine_threshold)
        spin_lower.setDecimals(3)
        f_amr.addRow("Refine threshold", spin_lower)
        f_amr.addRow("", QLabel("lowerRefineLevel: refine field in range (building3D: 0.1)."))
        spin_unref = QDoubleSpinBox()
        spin_unref.setRange(0.01, 20.0)
        spin_unref.setValue(self._unrefine_threshold)
        spin_unref.setDecimals(3)
        f_amr.addRow("Unrefine threshold", spin_unref)
        f_amr.addRow("", QLabel("unrefineLevel: if value &lt; this, unrefine (building3D: 0.1)."))
        spin_buf = QSpinBox()
        spin_buf.setRange(0, 10)
        spin_buf.setValue(self._n_buffer_layers_dynamic)
        f_amr.addRow("Buffer layers", spin_buf)
        f_amr.addRow("", QLabel("nBufferLayers: slower than 2:1 refinement (building3D: 2)."))
        le_indicator = QComboBox()
        le_indicator.setEditable(True)
        le_indicator.addItems(["densityGradient"])
        le_indicator.setCurrentText(getattr(self, "_refine_indicator_field", "densityGradient"))
        f_amr.addRow("Refine indicator field", le_indicator)
        f_amr.addRow("", QLabel("errorEstimator field (building3D: densityGradient)."))
        chk_bal = QCheckBox()
        chk_bal.setChecked(self._enable_balancing)
        f_amr.addRow("Load balancing", chk_bal)
        f_amr.addRow("", QLabel("enableBalancing (building3D: not set)."))
        v.addWidget(grp_amr)

        # Group: Obstacle refine – Advanced
        grp_obs = QGroupBox("Obstacle refine – Advanced")
        f_obs = QFormLayout(grp_obs)
        spin_fa = QSpinBox()
        spin_fa.setRange(30, 180)
        spin_fa.setValue(getattr(self, "_obstacle_feature_angle", 120))
        f_obs.addRow("Feature angle", spin_fa)
        f_obs.addRow("", QLabel("surfaceFeaturesDict includedAngle (building3D: 120)."))
        spin_cbl = QSpinBox()
        spin_cbl.setRange(1, 10)
        spin_cbl.setValue(getattr(self, "_obstacle_cells_between_levels", 2))
        f_obs.addRow("Cells between levels", spin_cbl)
        f_obs.addRow("", QLabel("snappy nCellsBetweenLevels (building3D: 2)."))
        spin_snap = QSpinBox()
        spin_snap.setRange(1, 500)
        spin_snap.setValue(getattr(self, "_obstacle_snap_iter", 100))
        f_obs.addRow("Snap iterations", spin_snap)
        f_obs.addRow("", QLabel("snappy nSolveIter (building3D: 100)."))
        spin_fsnap = QSpinBox()
        spin_fsnap.setRange(1, 100)
        spin_fsnap.setValue(getattr(self, "_obstacle_feature_snap_iter", 15))
        f_obs.addRow("Feature snap iterations", spin_fsnap)
        f_obs.addRow("", QLabel("snappy nFeatureSnapIter (building3D: 15)."))
        v.addWidget(grp_obs)

        # Metrics (Estimated; no runtime updates)
        grp_metrics = QGroupBox("Metrics")
        f_met = QFormLayout(grp_metrics)
        nx = max(1, int((self.sx2.value() - self.sx1.value()) / max(1e-9, self.scell.value())))
        ny = max(1, int((self.sy2.value() - self.sy1.value()) / max(1e-9, self.scell.value())))
        nz = max(1, int((self.sz2.value() - self.sz1.value()) / max(1e-9, self.scell.value())))
        domain_cells = nx * ny * nz
        f_met.addRow("Domain (no refinement): total cells", QLabel(f"{domain_cells:,} (Estimated)"))
        v.addWidget(grp_metrics)

        scroll0.setWidget(inner0)
        v0.addWidget(scroll0)
        tabs.addTab(tab0, "AMR & Obstacle")

        # Tab 1: Geometry & Mesh Quality (snap + castellated + meshQualityControls)
        tab1 = QWidget()
        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(True)
        inner1 = QWidget()
        v1 = QVBoxLayout(inner1)

        def _mesh_row(lbl_bold: str, desc: str, ctrl):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            left = QVBoxLayout()
            lbl = QLabel(lbl_bold)
            lbl.setStyleSheet("font-weight: bold;")
            left.addWidget(lbl)
            left.addWidget(QLabel(desc))
            h.addLayout(left)
            h.addWidget(ctrl, 0, Qt.AlignRight)
            return w

        f_geom = QFormLayout()
        spin_inc = QSpinBox()
        spin_inc.setRange(10, 180)
        spin_inc.setValue(getattr(self, "_mesh_included_angle", None) or getattr(self, "_obstacle_feature_angle", 120))
        f_geom.addRow(_mesh_row("Included angle (deg)", "Edges with angle below this count as features (surfaceFeaturesDict).", spin_inc))
        spin_nsmooth = QSpinBox()
        spin_nsmooth.setRange(0, 20)
        spin_nsmooth.setValue(3 if getattr(self, "_mesh_n_smooth_patch", None) is None else self._mesh_n_smooth_patch)
        f_geom.addRow(_mesh_row("Patch smoothing iterations", "nSmoothPatch in snapControls.", spin_nsmooth))
        spin_tol = QDoubleSpinBox()
        spin_tol.setRange(0.01, 10.0)
        spin_tol.setValue(1.0 if getattr(self, "_mesh_snap_tolerance", None) is None else self._mesh_snap_tolerance)
        f_geom.addRow(_mesh_row("Snap tolerance", "tolerance in snapControls.", spin_tol))
        spin_nsolve = QSpinBox()
        spin_nsolve.setRange(1, 500)
        spin_nsolve.setValue(getattr(self, "_mesh_n_solve_iter", None) or getattr(self, "_obstacle_snap_iter", 100))
        f_geom.addRow(_mesh_row("Solve iterations", "nSolveIter in snapControls.", spin_nsolve))
        spin_nrelax = QSpinBox()
        spin_nrelax.setRange(1, 100)
        spin_nrelax.setValue(10 if getattr(self, "_mesh_n_relax_iter", None) is None else self._mesh_n_relax_iter)
        f_geom.addRow(_mesh_row("Relax iterations", "nRelaxIter in snapControls.", spin_nrelax))
        spin_nfeat = QSpinBox()
        spin_nfeat.setRange(1, 100)
        spin_nfeat.setValue(getattr(self, "_mesh_n_feature_snap_iter", None) or getattr(self, "_obstacle_feature_snap_iter", 15))
        f_geom.addRow(_mesh_row("Feature snap iterations", "nFeatureSnapIter in snapControls.", spin_nfeat))
        chk_explicit = QCheckBox()
        chk_explicit.setChecked(False if getattr(self, "_mesh_explicit_feature_snap", None) is None else self._mesh_explicit_feature_snap)
        f_geom.addRow(_mesh_row("Explicit feature snapping", "explicitFeatureSnap in snapControls.", chk_explicit))
        chk_implicit = QCheckBox()
        chk_implicit.setChecked(True if getattr(self, "_mesh_implicit_feature_snap", None) is None else self._mesh_implicit_feature_snap)
        f_geom.addRow(_mesh_row("Implicit feature snapping", "implicitFeatureSnap in snapControls.", chk_implicit))
        chk_multi = QCheckBox()
        chk_multi.setChecked(False if getattr(self, "_mesh_multi_region_feature_snap", None) is None else self._mesh_multi_region_feature_snap)
        f_geom.addRow(_mesh_row("Multi-region feature snapping", "multiRegionFeatureSnap in snapControls.", chk_multi))
        spin_cbl_m = QSpinBox()
        spin_cbl_m.setRange(1, 10)
        spin_cbl_m.setValue(getattr(self, "_mesh_n_cells_between_levels", None) or getattr(self, "_obstacle_cells_between_levels", 2))
        f_geom.addRow(_mesh_row("Cells between refinement levels", "nCellsBetweenLevels in castellatedMeshControls.", spin_cbl_m))
        spin_resolve = QSpinBox()
        spin_resolve.setRange(5, 180)
        spin_resolve.setValue(getattr(self, "_mesh_resolve_feature_angle", None) or getattr(self, "_obstacle_feature_angle", 30))
        f_geom.addRow(_mesh_row("Feature resolve angle (deg)", "resolveFeatureAngle in castellatedMeshControls.", spin_resolve))
        spin_nonortho = QDoubleSpinBox()
        spin_nonortho.setRange(0, 180)
        spin_nonortho.setValue(65 if getattr(self, "_mesh_max_non_ortho", None) is None else self._mesh_max_non_ortho)
        f_geom.addRow(_mesh_row("Max non-orthogonality", "meshQualityControls maxNonOrtho.", spin_nonortho))
        spin_bskew = QDoubleSpinBox()
        spin_bskew.setValue(20 if getattr(self, "_mesh_max_boundary_skewness", None) is None else self._mesh_max_boundary_skewness)
        f_geom.addRow(_mesh_row("Max boundary skewness", "meshQualityControls maxBoundarySkewness.", spin_bskew))
        spin_iskew = QDoubleSpinBox()
        spin_iskew.setValue(4 if getattr(self, "_mesh_max_internal_skewness", None) is None else self._mesh_max_internal_skewness)
        f_geom.addRow(_mesh_row("Max internal skewness", "meshQualityControls maxInternalSkewness.", spin_iskew))
        spin_concave = QDoubleSpinBox()
        spin_concave.setValue(80 if getattr(self, "_mesh_max_concave", None) is None else self._mesh_max_concave)
        f_geom.addRow(_mesh_row("Max concave angle", "meshQualityControls maxConcave.", spin_concave))
        spin_minvol = QDoubleSpinBox()
        spin_minvol.setDecimals(4)
        spin_minvol.setRange(1e-20, 1.0)
        spin_minvol.setValue(1e-13 if getattr(self, "_mesh_min_vol", None) is None else self._mesh_min_vol)
        f_geom.addRow(_mesh_row("Minimum cell volume", "meshQualityControls minVol.", spin_minvol))
        spin_mintet = QDoubleSpinBox()
        spin_mintet.setDecimals(4)
        spin_mintet.setValue(1e-15 if getattr(self, "_mesh_min_tet_quality", None) is None else self._mesh_min_tet_quality)
        f_geom.addRow(_mesh_row("Minimum tet quality", "meshQualityControls minTetQuality.", spin_mintet))
        spin_twist = QDoubleSpinBox()
        spin_twist.setValue(0.02 if getattr(self, "_mesh_min_twist", None) is None else self._mesh_min_twist)
        f_geom.addRow(_mesh_row("Minimum twist", "meshQualityControls minTwist.", spin_twist))
        spin_det = QDoubleSpinBox()
        spin_det.setDecimals(4)
        spin_det.setValue(0.001 if getattr(self, "_mesh_min_determinant", None) is None else self._mesh_min_determinant)
        f_geom.addRow(_mesh_row("Minimum determinant", "meshQualityControls minDeterminant.", spin_det))
        spin_fw = QDoubleSpinBox()
        spin_fw.setValue(0.05 if getattr(self, "_mesh_min_face_weight", None) is None else self._mesh_min_face_weight)
        f_geom.addRow(_mesh_row("Minimum face weight", "meshQualityControls minFaceWeight.", spin_fw))
        spin_vr = QDoubleSpinBox()
        spin_vr.setValue(0.01 if getattr(self, "_mesh_min_vol_ratio", None) is None else self._mesh_min_vol_ratio)
        f_geom.addRow(_mesh_row("Minimum volume ratio", "meshQualityControls minVolRatio.", spin_vr))
        spin_nscale = QSpinBox()
        spin_nscale.setRange(0, 20)
        spin_nscale.setValue(4 if getattr(self, "_mesh_n_smooth_scale", None) is None else self._mesh_n_smooth_scale)
        f_geom.addRow(_mesh_row("Smoothing scale", "meshQualityControls nSmoothScale.", spin_nscale))
        spin_err = QDoubleSpinBox()
        spin_err.setValue(0.75 if getattr(self, "_mesh_error_reduction", None) is None else self._mesh_error_reduction)
        f_geom.addRow(_mesh_row("Error reduction", "meshQualityControls errorReduction.", spin_err))
        spin_relaxed = QDoubleSpinBox()
        spin_relaxed.setRange(0, 180)
        spin_relaxed.setValue(75 if getattr(self, "_mesh_relaxed_max_non_ortho", None) is None else self._mesh_relaxed_max_non_ortho)
        f_geom.addRow(_mesh_row("Relaxed max non-orthogonality", "relaxed { maxNonOrtho } in meshQualityControls.", spin_relaxed))
        v1.addLayout(f_geom)
        scroll1.setWidget(inner1)
        v1_outer = QVBoxLayout(tab1)
        v1_outer.addWidget(scroll1)
        tabs.addTab(tab1, "Geometry & Mesh Quality")

        layout.addWidget(tabs)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        layout.addWidget(bb)
        if d.exec_() == QDialog.Accepted:
            self._buffer_layers = spin_buf_setfields.value()
            self._refine_interval = spin_ref_int.value()
            self._lower_refine_threshold = spin_lower.value()
            self._unrefine_threshold = spin_unref.value()
            self._n_buffer_layers_dynamic = spin_buf.value()
            self._refine_indicator_field = le_indicator.currentText().strip() or "densityGradient"
            self._enable_balancing = chk_bal.isChecked()
            self._obstacle_feature_angle = spin_fa.value()
            self._obstacle_cells_between_levels = spin_cbl.value()
            self._obstacle_snap_iter = spin_snap.value()
            self._obstacle_feature_snap_iter = spin_fsnap.value()
            self._mesh_included_angle = spin_inc.value()
            self._mesh_n_smooth_patch = spin_nsmooth.value()
            self._mesh_snap_tolerance = spin_tol.value()
            self._mesh_n_solve_iter = spin_nsolve.value()
            self._mesh_n_relax_iter = spin_nrelax.value()
            self._mesh_n_feature_snap_iter = spin_nfeat.value()
            self._mesh_explicit_feature_snap = chk_explicit.isChecked()
            self._mesh_implicit_feature_snap = chk_implicit.isChecked()
            self._mesh_multi_region_feature_snap = chk_multi.isChecked()
            self._mesh_n_cells_between_levels = spin_cbl_m.value()
            self._mesh_resolve_feature_angle = spin_resolve.value()
            self._mesh_max_non_ortho = spin_nonortho.value()
            self._mesh_max_boundary_skewness = spin_bskew.value()
            self._mesh_max_internal_skewness = spin_iskew.value()
            self._mesh_max_concave = spin_concave.value()
            self._mesh_min_vol = spin_minvol.value()
            self._mesh_min_tet_quality = spin_mintet.value()
            self._mesh_min_twist = spin_twist.value()
            self._mesh_min_determinant = spin_det.value()
            self._mesh_min_face_weight = spin_fw.value()
            self._mesh_min_vol_ratio = spin_vr.value()
            self._mesh_n_smooth_scale = spin_nscale.value()
            self._mesh_error_reduction = spin_err.value()
            self._mesh_relaxed_max_non_ortho = spin_relaxed.value()
            for k in ("mesh_included_angle", "mesh_n_smooth_patch", "mesh_snap_tolerance", "mesh_n_solve_iter", "mesh_n_relax_iter", "mesh_n_feature_snap_iter", "mesh_explicit_feature_snap", "mesh_implicit_feature_snap", "mesh_multi_region_feature_snap", "mesh_n_cells_between_levels", "mesh_resolve_feature_angle", "mesh_max_non_ortho", "mesh_max_boundary_skewness", "mesh_max_internal_skewness", "mesh_max_concave", "mesh_min_vol", "mesh_min_tet_quality", "mesh_min_twist", "mesh_min_determinant", "mesh_min_face_weight", "mesh_min_vol_ratio", "mesh_n_smooth_scale", "mesh_error_reduction", "mesh_relaxed_max_non_ortho"):
                self._set_provenance_user(k)

    def _open_charge_advanced_dialog(self):
        """Advanced…: activation model, thermo/energy model (products + air), single dialog."""
        d = QDialog(self)
        d.setWindowTitle("Advanced (phaseProperties)")
        layout = QVBoxLayout(d)
        f = QFormLayout()
        act_opts = get_activation_options([self._activation_model_ui] if self._activation_model_ui else None)
        combo_act = QComboBox()
        combo_act.setEditable(True)
        combo_act.addItems(act_opts)
        combo_act.setCurrentText(self._activation_model_ui or "pressureBased")
        f.addRow(QLabel("Activation model"), combo_act)
        thermo_opts = get_thermo_options([self._thermo_model] if self._thermo_model else None)
        combo_thermo = QComboBox()
        combo_thermo.setEditable(True)
        combo_thermo.addItems(thermo_opts)
        combo_thermo.setCurrentText(self._thermo_model or "ePolynomial")
        f.addRow(QLabel("Thermodynamics / energy model (products)"), combo_thermo)
        thermo_air_opts = get_thermo_options([self._thermo_model_air] if self._thermo_model_air else None)
        combo_thermo_air = QComboBox()
        combo_thermo_air.setEditable(True)
        combo_thermo_air.addItems(thermo_air_opts)
        combo_thermo_air.setCurrentText(self._thermo_model_air or "eConst")
        f.addRow(QLabel("Thermodynamics (air)"), combo_thermo_air)
        layout.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        layout.addWidget(bb)
        if d.exec_() == QDialog.Accepted:
            self._activation_model_ui = combo_act.currentText().strip()
            self._thermo_model = combo_thermo.currentText().strip()
            self._thermo_model_air = combo_thermo_air.currentText().strip()
            for k in ("activation_model_ui", "thermo_model", "thermo_model_air"):
                self._set_provenance_user(k)

    def _open_decomposition_dialog(self):
        """Edit…: decomposition method and coefficients (decomposeParDict)."""
        d = QDialog(self)
        d.setWindowTitle("Advanced Decomposition")
        layout = QVBoxLayout(d)
        f = QFormLayout()
        method_opts = get_decomposition_method_options([self._decomposition_method] if self._decomposition_method else None)
        combo_method = QComboBox()
        combo_method.setEditable(True)
        combo_method.addItems(method_opts)
        combo_method.setCurrentText(self._decomposition_method or "scotch")
        f.addRow(QLabel("Decomposition method"), combo_method)
        n = getattr(self, "_decomposition_simple_n", (2, 2, 1)) or (2, 2, 1)
        spin_n1 = QSpinBox()
        spin_n1.setRange(1, 64)
        spin_n1.setValue(n[0])
        spin_n2 = QSpinBox()
        spin_n2.setRange(1, 64)
        spin_n2.setValue(n[1])
        spin_n3 = QSpinBox()
        spin_n3.setRange(1, 64)
        spin_n3.setValue(n[2])
        n_row = QWidget()
        n_h = QHBoxLayout(n_row)
        n_h.setContentsMargins(0, 0, 0, 0)
        n_h.addWidget(QLabel("n (n1 n2 n3)"))
        n_h.addWidget(spin_n1)
        n_h.addWidget(spin_n2)
        n_h.addWidget(spin_n3)
        f.addRow(n_row)
        spin_delta = QDoubleSpinBox()
        spin_delta.setDecimals(4)
        spin_delta.setRange(0.0001, 1.0)
        spin_delta.setValue(getattr(self, "_decomposition_simple_delta", None) or 0.001)
        f.addRow(QLabel("simpleCoeffs delta"), spin_delta)
        layout.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        layout.addWidget(bb)
        if d.exec_() == QDialog.Accepted:
            self._decomposition_method = combo_method.currentText().strip()
            self._decomposition_simple_n = (spin_n1.value(), spin_n2.value(), spin_n3.value())
            self._decomposition_simple_delta = spin_delta.value()
            for k in ("decomposition_method", "decomposition_simple_n", "decomposition_simple_delta"):
                self._set_provenance_user(k)

    def _update_charge_radius(self):
        """Compute and display radius/side from mass, density and (for Cylinder) Aspect. 
        Used for Sphere, Cylinder, and Cuboid (shows computed side length)."""
        shape = self.c_shape.currentText()
        if shape not in ("Sphere", "Cylinder", "Cuboid"):
            return
        mass = max(1e-9, self.c_mass.value())
        rho = max(1e-9, self.c_rho.value())
        vol = mass / rho
        if shape == "Sphere":
            r = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
        elif shape == "Cuboid":
            # Cube: side = vol^(1/3) (mass-driven geometry)
            r = vol ** (1.0 / 3.0)
            self._update_cuboid_height()
        else:  # Cylinder
            aspect = max(0.1, self.c_aspect.value())
            r = (vol / (2.0 * math.pi * aspect)) ** (1.0 / 3.0)
        self.c_radius.blockSignals(True)
        self.c_radius.setValue(r)
        self.c_radius.blockSignals(False)
        self._update_preview()

    def _update_cuboid_height(self):
        """For Cuboid: set height = V/(L×W) so dimensions match mass and density. Height is read-only."""
        if self.c_shape.currentText() != "Cuboid":
            return
        mass = max(1e-9, self.c_mass.value())
        rho = max(1e-9, self.c_rho.value())
        vol = mass / rho
        L = max(1e-9, self.c_length.value())
        W = max(1e-9, self.c_width.value())
        H = vol / (L * W)
        self.c_height.blockSignals(True)
        self.c_height.setValue(round(H, 6) if H < 1000 else H)
        self.c_height.blockSignals(False)
        self._update_preview()

    def _on_shape_changed(self, shape_name):
        """Enable/disable geometry fields based on selected charge shape. Radius is read-only (computed) for Sphere/Cylinder/Cuboid."""
        is_sphere = (shape_name == "Sphere")
        is_cylinder = (shape_name == "Cylinder")
        is_cuboid = (shape_name == "Cuboid")

        # Radius/Side: Sphere, Cylinder, Cuboid — read-only, gray, value computed from mass/density/(aspect)
        if is_sphere or is_cylinder or is_cuboid:
            self.c_radius.setReadOnly(True)
            for w in (self.c_radius, self.lbl_radius):
                w.setEnabled(False)  # gray, display-only
            # Update label text
            if is_cuboid:
                self.lbl_radius.setText("Side Length [m]")
                side = (max(1e-9, self.c_mass.value()) / max(1e-9, self.c_rho.value())) ** (1.0 / 3.0)
                self.c_length.blockSignals(True)
                self.c_width.blockSignals(True)
                self.c_length.setValue(side)
                self.c_width.setValue(side)
                self.c_length.blockSignals(False)
                self.c_width.blockSignals(False)
            else:
                self.lbl_radius.setText("Radius [m]")
            self._update_charge_radius()
        else:
            for w in (self.c_radius, self.lbl_radius):
                w.setEnabled(False)
            self.c_radius.setReadOnly(True)
        # Aspect (L/D): Cylinder only
        for w in (self.c_aspect, self.lbl_aspect):
            w.setEnabled(is_cylinder)
        # Cylinder axis: Cylinder only
        for w in (self.c_cylinder_axis, self.lbl_cylinder_axis):
            w.setEnabled(is_cylinder)
        # Length, Width, Height: enabled for Cuboid (rectangular prism; height = V/(L×W) to match mass/density)
        if is_cuboid:
            for w in (self.c_length, self.lbl_length, self.c_width, self.lbl_width, self.c_height, self.lbl_height):
                w.setEnabled(True)
            self.c_height.setReadOnly(True)
            self._update_cuboid_height()
        else:
            for w in (self.c_length, self.lbl_length, self.c_width, self.lbl_width, self.c_height, self.lbl_height):
                w.setEnabled(False)
            self.c_height.setReadOnly(True)

        self._update_preview()

    def _on_material_changed(self, mat_name):
        if mat_name in self.materials_db:
            props = self.materials_db[mat_name]
            self.c_rho.setValue(props["rho"])
        self._update_edit_button_visibility()
        self._update_preview()

    def _update_preview(self):
        try:
            dx = max(1e-6, self.scell.value())
            nx = int((self.sx2.value() - self.sx1.value()) / dx)
            ny = int((self.sy2.value() - self.sy1.value()) / dx)
            nz = int((self.sz2.value() - self.sz1.value()) / dx)
            total = nx * ny * nz
            self.lbl_cells.setText(f"Grid: {nx}x{ny}x{nz}  ({total:,} cells)")
        except:
            self.lbl_cells.setText("Grid: Error")

        self._clear_charge_cells_display()
        self._update_estimated_charge_cells_display()

        if not self.viewer: return
        bounds = (self.sx1.value(), self.sx2.value(), self.sy1.value(), self.sy2.value(), self.sz1.value(), self.sz2.value())
        charge = (self.cx.value(), self.cy.value(), self.cz.value(), self.c_shape.currentText(), self.c_mass.value(), self.c_rho.value())
        self.viewer.update_preview(bounds, charge, self.obstacles)

    def _clear_charge_cells_display(self):
        """Reset charge cells info when case is not loaded or before init."""
        self.lbl_charge_cells.setText("Charge cells (alpha.c4>thr): —")
        self.lbl_charge_fraction.setText("Charge fraction (%): —")
        self.lbl_cells_inside_charge.setText("Cells inside charge (post-refinement): —")
        self.lbl_charge_clipped.setText("Charge clipped by domain: —")
        self.lbl_init_mode.setText("Init: —")
        self.lbl_initiation_radius.setText("Initiation radius: —")
        self.lbl_charge_refine_info.setText("Charge refine: —")
        self.lbl_obstacle_refine_info.setText("Obstacle refine: —")
        self.lbl_est_charge_cells.setText("Est. charge cells: —")
        self.lbl_smallest_cell.setText("Smallest cell (est.): —")
        self.lbl_cells_in_ignition.setText("Cells in ignition region: —")
        self.lbl_expected_emesh.setText("Expected .eMesh: —")
        self.lbl_charge_resolution_warning.setText("")

    def _update_info_from_case_init_mode(self, case_dir: str) -> None:
        """Read case_init_mode.json and update Init/effective-value labels. 3D non-remap only."""
        import json
        path = os.path.join(case_dir, "case_init_mode.json")
        if not os.path.isfile(path):
            self.lbl_init_mode.setText("Init: —")
            self.lbl_initiation_radius.setText("Initiation radius: —")
            self.lbl_charge_refine_info.setText("Charge refine: —")
            self.lbl_obstacle_refine_info.setText("Obstacle refine: —")
            self.lbl_smallest_cell.setText("Smallest cell (est.): —")
            self.lbl_cells_in_ignition.setText("Cells in ignition region: —")
            self.lbl_expected_emesh.setText("Expected .eMesh: —")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                mode = json.load(f)
        except (OSError, ValueError):
            return
        set_cmd = mode.get("set_cmd") or "—"
        self.lbl_init_mode.setText(f"Init: {set_cmd}")
        fallback = mode.get("fallback_reason")
        if fallback:
            self.lbl_init_mode.setToolTip(f"Fallback: {fallback}")
        req = mode.get("initiation_radius_requested")
        eff = mode.get("initiation_radius_effective")
        if req is not None and eff is not None:
            self.lbl_initiation_radius.setText(f"Initiation radius: {eff:.4g} (req {req:.4g})")
        else:
            self.lbl_initiation_radius.setText("Initiation radius: —")
        cr_req = mode.get("charge_refinement_requested")
        cr_eff = mode.get("charge_refinement_effective")
        startup_lv = mode.get("startup_refinement_levels")
        remaining_lv = mode.get("remaining_inside_levels")
        auto_adj = mode.get("startup_auto_adjusted", False)
        msg = mode.get("startup_refinement_message")
        if cr_req is not None and cr_eff is not None:
            if startup_lv is not None and remaining_lv is not None:
                self.lbl_charge_refine_info.setText(f"Charge refine: req {cr_req}, eff {cr_eff} (startup {startup_lv} + remaining {remaining_lv})")
            else:
                self.lbl_charge_refine_info.setText(f"Charge refine: req {cr_req}, eff {cr_eff}")
            tip = ""
            if auto_adj and msg:
                tip = msg
            if tip:
                self.lbl_charge_refine_info.setToolTip(tip)
        else:
            self.lbl_charge_refine_info.setText("Charge refine: —")
        obs_r = mode.get("obstacle_refinement")
        snap = mode.get("snappy_refinement")
        if obs_r is not None:
            self.lbl_obstacle_refine_info.setText(f"Obstacle refine: {obs_r}, snappy: {'yes' if snap else 'no'}")
        else:
            self.lbl_obstacle_refine_info.setText("Obstacle refine: —")
        cells_inside = mode.get("cells_inside_charge")
        if cells_inside is not None:
            self.lbl_cells_inside_charge.setText(f"Cells inside charge (post-refinement): {cells_inside:,}")
        else:
            self.lbl_cells_inside_charge.setText("Cells inside charge (post-refinement): —")
        clipped = mode.get("charge_clipped_by_domain")
        warnings = mode.get("charge_warnings") or []
        tip = "Whether the charge geometry extends outside the domain."
        if warnings:
            tip += "\n\n" + "\n".join(warnings)
        self.lbl_charge_clipped.setToolTip(tip)
        cap_impossible = mode.get("charge_capture_impossible_message")
        if cap_impossible:
            self.lbl_charge_resolution_warning.setText("Init will fail: charge capture impossible. Reduce Cell Size or enlarge charge.")
            self.lbl_charge_resolution_warning.setToolTip(cap_impossible)
        else:
            self.lbl_charge_resolution_warning.setText("")
            self.lbl_charge_resolution_warning.setToolTip("")
        if clipped is not None:
            if isinstance(clipped, bool):
                self.lbl_charge_clipped.setText(f"Charge clipped by domain: {'yes' if clipped else 'no'}")
            else:
                self.lbl_charge_clipped.setText(f"Charge clipped by domain: {clipped}")
        else:
            self.lbl_charge_clipped.setText("Charge clipped by domain: —")
        sc = mode.get("smallest_cell_near_charge")
        if sc is not None:
            self.lbl_smallest_cell.setText(f"Smallest cell (est.): {sc:.2e} m")
        else:
            self.lbl_smallest_cell.setText("Smallest cell (est.): —")
        ign_cells = mode.get("cells_in_ignition_region")
        if ign_cells is not None:
            self.lbl_cells_in_ignition.setText(f"Cells in ignition region: {ign_cells:,}")
        else:
            self.lbl_cells_in_ignition.setText("Cells in ignition region: —")
        emesh_list = mode.get("expected_eMesh") or []
        if emesh_list:
            present = sum(1 for p in emesh_list if os.path.isfile(os.path.join(case_dir, p)))
            status = "OK" if present == len(emesh_list) else f"Missing {len(emesh_list) - present}/{len(emesh_list)}"
            self.lbl_expected_emesh.setText(f"Expected .eMesh: {len(emesh_list)} file(s) — {status}")
            self.lbl_expected_emesh.setToolTip("Canonical: " + "\n".join(emesh_list[:10]) + ("\n..." if len(emesh_list) > 10 else ""))
        else:
            self.lbl_expected_emesh.setText("Expected .eMesh: —")

    def update_charge_cells_display(self, case_dir: str, threshold: float = 0.5) -> None:
        """Update Info panel with charge cell count from 0/alpha.c4 and case_init_mode.json (after init)."""
        self._update_info_from_case_init_mode(case_dir)
        try:
            from verification.verify_output import get_charge_cell_count
        except ImportError:
            self.lbl_charge_cells.setText("Charge cells (alpha.c4>thr): —")
            self.lbl_charge_fraction.setText("Charge fraction (%): —")
            return
        charge, total = get_charge_cell_count(case_dir, time_dir="0", threshold=threshold)
        if charge is not None and total is not None and total > 0:
            self.lbl_charge_cells.setText(f"Charge cells (alpha.c4>{threshold}): {charge:,}")
            self.lbl_cells_inside_charge.setText(f"Cells inside charge (post-refinement): {charge:,}")
            pct = 100.0 * charge / total
            self.lbl_charge_fraction.setText(f"Charge fraction (%): {charge:,} / {total:,} ({pct:.2f}%)")
        elif charge is not None and total is not None:
            self.lbl_charge_cells.setText(f"Charge cells (alpha.c4>{threshold}): {charge:,}")
            self.lbl_cells_inside_charge.setText(f"Cells inside charge (post-refinement): {charge:,}")
            self.lbl_charge_fraction.setText("Charge fraction (%): —")
        else:
            self.lbl_charge_cells.setText("Charge cells (alpha.c4>thr): —")
            self.lbl_charge_fraction.setText("Charge fraction (%): —")

    def _on_init_clicked(self):
        self.sig_request_init.emit(self.get_case_inputs())

    def _on_field_change_req(self, txt):
        if self.viewer: self.viewer.set_field(txt)

    def _on_range_change(self):
        if self.viewer:
            self.viewer.set_field_range(self.spin_min_val.value(), self.spin_max_val.value(), self.chk_auto_range.isChecked())

    def check_mesh_update(self):
        self.viewer.refresh_view()

    def _position_bounds_for_plane(self, plane_text: str) -> tuple:
        """Return (min, max, default) in meters for Position [m] for the given plane (axis along normal)."""
        if "XZ" in plane_text:
            return (self.sy1.value(), self.sy2.value(), self.cy.value())
        if "YZ" in plane_text:
            return (self.sx1.value(), self.sx2.value(), self.cx.value())
        return (self.sz1.value(), self.sz2.value(), self.cz.value())

    def _add_default_section(self):
        r = self.tbl_sec.rowCount(); self.tbl_sec.insertRow(r)
        self.tbl_sec.blockSignals(True)
        chk = QTableWidgetItem(); chk.setCheckState(Qt.Checked)
        self.tbl_sec.setItem(r, 0, chk)
        self.tbl_sec.setItem(r, 1, QTableWidgetItem("Section"))
        cmb = QComboBox(); cmb.addItems(["XY (Ground)", "XZ (Front)", "YZ (Side)"])
        cmb.currentIndexChanged.connect(lambda: self._update_section_row_bounds(r))
        cmb.currentIndexChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.tbl_sec.setCellWidget(r, 2, cmb)
        plane = "XY (Ground)"
        pmin, pmax, pdefault = self._position_bounds_for_plane(plane)
        # Stagger position so multiple slices in same direction don't overlap
        same_plane_count = sum(
            1 for i in range(r)
            if (self.tbl_sec.cellWidget(i, 2) and
                getattr(self.tbl_sec.cellWidget(i, 2), "currentText", lambda: "")() == plane)
        )
        span = max(1e-6, pmax - pmin)
        pos_default = max(pmin, min(pmax, pdefault + same_plane_count * 0.25 * span))
        step = max(1e-6, self.scell.value())
        pos_m = QDoubleSpinBox()
        pos_m.setRange(pmin, pmax)
        pos_m.setValue(pos_default)
        pos_m.setSingleStep(step)
        pos_m.setDecimals(4)
        pos_m.setToolTip("Slice position in meters along plane normal; clamped to domain bounds. Step = cell size.")
        pos_m.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.tbl_sec.setCellWidget(r, 3, pos_m)
        op = QDoubleSpinBox(); op.setRange(0, 1); op.setValue(0.5); op.setSingleStep(0.1)
        op.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.tbl_sec.setCellWidget(r, 4, op)
        self.tbl_sec.blockSignals(False)
        self._update_sections_from_table(-1, -1)

    def _del_section(self):
        r = self.tbl_sec.currentRow()
        if r >= 0: self.tbl_sec.removeRow(r); self._update_sections_from_table(-1, -1)

    def _update_section_row_bounds(self, row: int) -> None:
        """Update Position [m] spinbox min/max/step for a row from current domain and plane."""
        if row < 0 or row >= self.tbl_sec.rowCount(): return
        cmb = self.tbl_sec.cellWidget(row, 2)
        pos_w = self.tbl_sec.cellWidget(row, 3)
        if not cmb or not pos_w or not hasattr(pos_w, "setRange"): return
        pmin, pmax, pdefault = self._position_bounds_for_plane(cmb.currentText())
        step = max(1e-6, self.scell.value())
        pos_w.blockSignals(True)
        pos_w.setRange(pmin, pmax)
        pos_w.setSingleStep(step)
        val = pos_w.value()
        if val < pmin or val > pmax:
            pos_w.setValue(max(pmin, min(pmax, val)))
        pos_w.blockSignals(False)

    def _update_sections_from_table(self, r, c):
        new_secs = []
        cx = (self.sx1.value() + self.sx2.value()) / 2.0
        cy = (self.sy1.value() + self.sy2.value()) / 2.0
        cz = (self.sz1.value() + self.sz2.value()) / 2.0

        for i in range(self.tbl_sec.rowCount()):
            self._update_section_row_bounds(i)
            item_chk = self.tbl_sec.item(i, 0)
            item_name = self.tbl_sec.item(i, 1)
            cmb = self.tbl_sec.cellWidget(i, 2)
            pos_w = self.tbl_sec.cellWidget(i, 3) if self.tbl_sec.columnCount() > 3 else None
            op_w = self.tbl_sec.cellWidget(i, 4) if self.tbl_sec.columnCount() > 4 else self.tbl_sec.cellWidget(i, 3)
            if not (item_chk and item_name and cmb and op_w): continue
            enabled = (item_chk.checkState() == Qt.Checked)
            name = item_name.text()
            plane = cmb.currentText()
            normal = [0, 0, 1]
            if "XZ" in plane: normal = [0, 1, 0]
            elif "YZ" in plane: normal = [1, 0, 0]
            _, _, default_m = self._position_bounds_for_plane(plane)
            position_m = float(pos_w.value()) if pos_w is not None and hasattr(pos_w, "value") else default_m
            opacity = float(op_w.value()) if hasattr(op_w, "value") else 0.5
            new_secs.append(SectionItem(enabled, name, [cx, cy, cz], normal, opacity, position_m))

        if self.viewer:
            self.viewer.update_sections(new_secs)

    def _add_stl(self):
        f, _ = QFileDialog.getOpenFileName(self, "STL", "", "STL (*.stl)")
        if f: self.obstacles.append(ObstacleItem(True, f, 0.001, 0,0,0)); self._refresh_table(); self._update_preview()
    
    def _clear_stl(self):
        self.obstacles = []; self._refresh_table(); self._update_preview()

    def _refresh_table(self):
        self._block_signals = True; self.tbl_obs.setRowCount(0)
        for obs in self.obstacles:
            r = self.tbl_obs.rowCount(); self.tbl_obs.insertRow(r)
            it = QTableWidgetItem(); it.setCheckState(Qt.Checked if obs.enabled else Qt.Unchecked)
            self.tbl_obs.setItem(r,0,it); self.tbl_obs.setItem(r,1,QTableWidgetItem(os.path.basename(obs.path)))
            self.tbl_obs.setItem(r,2,QTableWidgetItem(str(obs.scale))); self.tbl_obs.setItem(r,3,QTableWidgetItem(str(obs.ox)))
            self.tbl_obs.setItem(r,4,QTableWidgetItem(str(obs.oy)))
            self.tbl_obs.setItem(r,5,QTableWidgetItem(str(obs.oz)))
        self._block_signals = False

    def _on_table_change(self, r, c):
        if self._block_signals: return
        try:
            o = self.obstacles[r]
            if c==0: o.enabled = (self.tbl_obs.item(r,0).checkState()==Qt.Checked)
            elif c==2: o.scale=float(self.tbl_obs.item(r,2).text())
            elif c==3: o.ox=float(self.tbl_obs.item(r,3).text())
            elif c==4: o.oy=float(self.tbl_obs.item(r,4).text())
            elif c==5: o.oz=float(self.tbl_obs.item(r,5).text())
            self._update_preview()
        except: pass

    # ------------------------------------------------------------------
    #  Load case: populate all UI fields from a parsed dict
    # ------------------------------------------------------------------
    def _apply_default_for_key(self, key: str) -> None:
        """Set the widget(s) for *key* to GUI default. Used when Open does not fill the field."""
        if key == "min_point":
            self.sx1.setValue(0); self.sy1.setValue(0); self.sz1.setValue(0)
        elif key == "max_point":
            self.sx2.setValue(10); self.sy2.setValue(10); self.sz2.setValue(10)
        elif key == "cell_size":
            self.scell.setValue(0.1)
        elif key == "boundaries":
            for face_key, combo in self.bound_combos.items():
                idx = combo.findText("Transmitting")
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            idx_minz = self.bound_combos["minZ"].findText("Reflecting")
            if idx_minz >= 0:
                self.bound_combos["minZ"].setCurrentIndex(idx_minz)
        elif key == "cfl_value":
            self.spin_cfl.setValue(0.5)
        elif key == "end_time_s":
            self.spin_end.setValue(0.0025)
        elif key == "delta_t":
            pass  # computed from CFL / cell size
        elif key == "write_control_type":
            idx = self.combo_write_control.findText("adjustableRunTime")
            if idx >= 0:
                self.combo_write_control.setCurrentIndex(idx)
        elif key == "write_interval_time":
            self.spin_write_time.setValue(5e-5)
        elif key == "write_interval_steps":
            self.spin_write.setValue(100)
        elif key == "cycle_write":
            self.spin_cycle_write.setValue(0)
        elif key == "material_name":
            idx = self.c_mat.findText("C4")
            if idx >= 0:
                self.c_mat.setCurrentIndex(idx)
        elif key == "custom_material_props":
            pass  # used only when material_name is Custom
        elif key == "charge_shape":
            idx = self.c_shape.findText("Sphere")
            if idx >= 0:
                self.c_shape.setCurrentIndex(idx)
        elif key == "mass_kg":
            self.c_mass.setValue(25.0)
        elif key == "rho_charge":
            self.c_rho.setValue(1601.0)
        elif key == "charge_radius":
            self.c_radius.setValue(0.05)
        elif key == "charge_lbyd":
            self.c_aspect.setValue(2.5)
        elif key == "charge_length":
            self.c_length.setValue(0.1)
        elif key == "charge_width":
            self.c_width.setValue(0.1)
        elif key == "charge_height":
            self.c_height.setValue(0.1)
        elif key == "charge_center":
            self.cx.setValue(0); self.cy.setValue(0); self.cz.setValue(0)
        elif key == "initiation_point":
            self.init_ix.setValue(0); self.init_iy.setValue(0); self.init_iz.setValue(0)
        elif key == "ignition_mode":
            idx = self.combo_ignition_mode.findText("Center of Charge")
            if idx >= 0:
                self.combo_ignition_mode.setCurrentIndex(idx)
        elif key == "p_atm":
            self.p0.setValue(101325.0)
        elif key == "t_atm":
            self.t0.setValue(288.0)
        elif key == "refine_max":
            self.spin_refine_max.setValue(3)
        elif key == "refine_min":
            self.spin_refine_min.setValue(2)
        elif key == "enable_local_refinement":
            self.rad_dyn_mesh.setChecked(True)
            self.rad_fixed_mesh.setChecked(False)
        elif key == "enable_dyn_refine":
            self.rad_dyn_mesh.setChecked(True)
            self.rad_fixed_mesh.setChecked(False)
        elif key == "enable_obstacle_refine":
            self.chk_obstacle_refine.setChecked(True)
        elif key == "obstacle_refine_min":
            self.spin_obstacle_refine_min.setValue(1)
        elif key == "obstacle_refine_max":
            self.spin_obstacle_refine_max.setValue(2)
        elif key == "outside_extent":
            self.spin_transition_cells.setValue(2)
        elif key == "transition_cells":
            self.spin_transition_cells.setValue(2)
        elif key == "refine_interval":
            self._refine_interval = 3
        elif key == "lower_refine_threshold":
            self._lower_refine_threshold = 0.1
        elif key == "unrefine_threshold":
            self._unrefine_threshold = 0.1
        elif key == "n_buffer_layers_dynamic":
            self._n_buffer_layers_dynamic = 2
        elif key == "refine_indicator_field":
            self._refine_indicator_field = "densityGradient"
        elif key == "enable_balancing":
            self._enable_balancing = False
        elif key == "obstacle_feature_angle":
            self._obstacle_feature_angle = 120
        elif key == "obstacle_cells_between_levels":
            self._obstacle_cells_between_levels = 2
        elif key == "obstacle_snap_iter":
            self._obstacle_snap_iter = 100
        elif key == "obstacle_feature_snap_iter":
            self._obstacle_feature_snap_iter = 15
        elif key == "cores":
            self.spin_cores.setValue(1)
        elif key == "charge_refinement_level":
            self.spin_charge_refine.setValue(2)
        elif key == "charge_outer_refine_min":
            self.spin_charge_outer_min.setValue(self.spin_refine_min.value())
        elif key == "charge_outer_refine_max":
            self.spin_charge_outer_max.setValue(self.spin_refine_max.value())
        elif key == "charge_outer_refine_enable":
            self.spin_charge_outer_min.setValue(0)
            self.spin_charge_outer_max.setValue(0)
        elif key == "cylinder_axis":
            idx = self.c_cylinder_axis.findText("Z")
            if idx >= 0:
                self.c_cylinder_axis.setCurrentIndex(idx)
        elif key == "charge_backup_radius_factor":
            self.spin_backup_factor.setValue(1.0)
        elif key == "buffer_layers":
            self._buffer_layers = 2
        elif key == "activation_model":
            self.rad_init_standard.setChecked(True)
        elif key == "stl_obstacles":
            self.obstacles.clear()
            self._refresh_table()
        # Keys with no widget or derived: ignition_mode already above

    def _set_provenance_user(self, key: str) -> None:
        """Mark optional field as USER-edited and re-enable if it was UNSET-disabled."""
        if getattr(self, "_block_signals", False):
            return
        self._provenance[key] = "USER"
        if key == "outside_extent" or key == "transition_cells":
            self.spin_transition_cells.setEnabled(True)
        elif key == "enable_dyn_refine":
            self.rad_dyn_mesh.setEnabled(True)
            self.rad_fixed_mesh.setEnabled(True)
        elif key == "enable_obstacle_refine":
            self.chk_obstacle_refine.setEnabled(True)

    def _apply_unset_for_key(self, key: str) -> None:
        """Show optional field as UNSET: disable and/or set sentinel so generation does not override loaded case."""
        if key == "outside_extent" or key == "transition_cells":
            self.spin_transition_cells.setValue(2)
            self.spin_transition_cells.setEnabled(False)
        elif key == "enable_dyn_refine":
            self.rad_dyn_mesh.setEnabled(False)
            self.rad_fixed_mesh.setEnabled(False)
        elif key == "enable_obstacle_refine":
            self.chk_obstacle_refine.setEnabled(False)
        else:
            # eos_model, activation_model_ui, thermo_model, decomposition_*, mesh_*: set internal to None
            attr = "_" + key
            if hasattr(self, attr):
                setattr(self, attr, None)

    def set_case_inputs(self, data: dict, load_summary: dict = None) -> None:
        """Populate GUI from *data*. If *load_summary*: LOADED keys set from case; not_filled left UNSET (no default)."""
        if load_summary:
            self._provenance.update(data.get("_provenance", {}))
            not_filled = load_summary.get("not_filled", [])
            for key, _reason in not_filled:
                self._provenance[key] = "UNSET"
                self._apply_unset_for_key(key)
        self._block_signals = True
        try:
            # --- Domain Geometry ---
            mp = data.get("min_point")
            if mp and len(mp) >= 3:
                self.sx1.setValue(mp[0])
                self.sy1.setValue(mp[1])
                self.sz1.setValue(mp[2])
            xp = data.get("max_point")
            if xp and len(xp) >= 3:
                self.sx2.setValue(xp[0])
                self.sy2.setValue(xp[1])
                self.sz2.setValue(xp[2])
            if "cell_size" in data:
                self.scell.setValue(data["cell_size"])

            # --- CFL ---
            if "cfl_value" in data:
                self.spin_cfl.setValue(data["cfl_value"])

            # --- Boundaries ---
            bounds = data.get("boundaries", {})
            for face_key, combo in self.bound_combos.items():
                if face_key in bounds:
                    idx = combo.findText(bounds[face_key])
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
            # If only minZ was explicitly detected as Reflecting, set the rest to Transmitting
            if bounds.get("minZ") == "Reflecting":
                for face_key in ("minX", "maxX", "minY", "maxY", "maxZ"):
                    if face_key not in bounds:
                        idx = self.bound_combos[face_key].findText("Transmitting")
                        if idx >= 0:
                            self.bound_combos[face_key].setCurrentIndex(idx)

            # --- Material ---
            mat = data.get("material_name")
            if mat:
                idx = self.c_mat.findText(mat)
                if idx >= 0:
                    self.c_mat.setCurrentIndex(idx)
                # If Custom, update the materials_db with parsed JWL params
                if mat == "Custom" and "custom_material_props" in data:
                    cprops = data["custom_material_props"]
                    for k in ("rho", "energy", "A", "B", "R1", "R2", "omega"):
                        if k in cprops:
                            self.materials_db["Custom"][k] = cprops[k]

            # --- Charge Properties ---
            shape = data.get("charge_shape")
            if shape:
                idx = self.c_shape.findText(shape)
                if idx >= 0:
                    self.c_shape.setCurrentIndex(idx)
            if "mass_kg" in data:
                self.c_mass.setValue(data["mass_kg"])
            if "rho_charge" in data:
                self.c_rho.setValue(data["rho_charge"])
            if "charge_radius" in data:
                self.c_radius.setValue(data["charge_radius"])
            if "charge_lbyd" in data:
                self.c_aspect.setValue(data["charge_lbyd"])
            if "charge_length" in data:
                self.c_length.setValue(data["charge_length"])
            if "charge_width" in data:
                self.c_width.setValue(data["charge_width"])
            if "charge_height" in data:
                self.c_height.setValue(data["charge_height"])
            cc = data.get("charge_center")
            if cc and len(cc) >= 3:
                self.cx.setValue(cc[0])
                self.cy.setValue(cc[1])
                self.cz.setValue(cc[2])
            ip = data.get("initiation_point")
            if ip and len(ip) >= 3:
                self.init_ix.setValue(ip[0])
                self.init_iy.setValue(ip[1])
                self.init_iz.setValue(ip[2])
            elif cc and len(cc) >= 3:
                self.init_ix.setValue(cc[0])
                self.init_iy.setValue(cc[1])
                self.init_iz.setValue(cc[2])

            # --- Atmosphere ---
            if "p_atm" in data:
                self.p0.setValue(data["p_atm"])
            if "t_atm" in data:
                self.t0.setValue(data["t_atm"])

            # --- Execution Controls ---
            if "end_time_s" in data:
                self.spin_end.setValue(data["end_time_s"])
            if "cores" in data:
                self.spin_cores.setValue(data["cores"])
            if "refine_min" in data:
                self.spin_refine_min.setValue(int(data["refine_min"]))
            if "dyn_refine_min" in data:
                self.spin_refine_min.setValue(int(data["dyn_refine_min"]))
            if "refine_max" in data:
                self.spin_refine_max.setValue(data["refine_max"])
            if "dyn_refine_max" in data:
                self._dyn_refine_max = int(data["dyn_refine_max"])
                self.spin_refine_max.setValue(self._dyn_refine_max)
            elif "refine_max" in data:
                self._dyn_refine_max = int(data["refine_max"])
            if "enable_local_refinement" in data:
                en = bool(data["enable_local_refinement"])
                self.rad_dyn_mesh.setChecked(en)
                self.rad_fixed_mesh.setChecked(not en)
            if "enable_dyn_refine" in data:
                en = bool(data["enable_dyn_refine"])
                self.rad_dyn_mesh.setChecked(en)
                self.rad_fixed_mesh.setChecked(not en)
                self.rad_dyn_mesh.setEnabled(True)
                self.rad_fixed_mesh.setEnabled(True)
            if "enable_obstacle_refine" in data:
                self.chk_obstacle_refine.setChecked(bool(data["enable_obstacle_refine"]))
                self.chk_obstacle_refine.setEnabled(True)
            if "obstacle_refine_min" in data:
                self.spin_obstacle_refine_min.setValue(int(data["obstacle_refine_min"]))
            if "obstacle_refine_max" in data:
                self.spin_obstacle_refine_max.setValue(int(data["obstacle_refine_max"]))
            if "transition_cells" in data and data["transition_cells"] is not None:
                self.spin_transition_cells.setValue(max(1, min(10, int(data["transition_cells"]))))
                self.spin_transition_cells.setEnabled(True)
            if "match_outer_to_seed" in data:
                self.chk_match_outer_to_seed.setChecked(bool(data["match_outer_to_seed"]))
            elif "outside_extent" in data:
                # Migration: old cases with outside_extent -> default Transition Cells 2
                self.spin_transition_cells.setValue(2)
                self.spin_transition_cells.setEnabled(True)
            if "refine_indicator_field" in data:
                self._refine_indicator_field = str(data["refine_indicator_field"]) or "densityGradient"
            if "obstacle_feature_angle" in data:
                self._obstacle_feature_angle = int(data["obstacle_feature_angle"])
            if "obstacle_cells_between_levels" in data:
                self._obstacle_cells_between_levels = int(data["obstacle_cells_between_levels"])
            if "obstacle_snap_iter" in data:
                self._obstacle_snap_iter = int(data["obstacle_snap_iter"])
            if "obstacle_feature_snap_iter" in data:
                self._obstacle_feature_snap_iter = int(data["obstacle_feature_snap_iter"])
            if "eos_model" in data and data["eos_model"] is not None:
                self._eos_model = str(data["eos_model"])
                opts = get_eos_options([self._eos_model])
                self.combo_eos.clear()
                self.combo_eos.addItems(opts)
                self.combo_eos.setCurrentText(self._eos_model)
            if "activation_model_ui" in data and data["activation_model_ui"] is not None:
                self._activation_model_ui = str(data["activation_model_ui"])
            if "activation_model" in data and data["activation_model"] is not None:
                self._activation_model_ui = str(data["activation_model"])
            if "thermo_model" in data and data["thermo_model"] is not None:
                self._thermo_model = str(data["thermo_model"])
            if "thermo_model_air" in data and data["thermo_model_air"] is not None:
                self._thermo_model_air = str(data["thermo_model_air"])
            if "decomposition_method" in data and data["decomposition_method"] is not None:
                self._decomposition_method = str(data["decomposition_method"])
            if "decomposition_simple_n" in data and data["decomposition_simple_n"] is not None:
                n = data["decomposition_simple_n"]
                if isinstance(n, (list, tuple)) and len(n) >= 3:
                    self._decomposition_simple_n = (int(n[0]), int(n[1]), int(n[2]))
            if "decomposition_simple_delta" in data and data["decomposition_simple_delta"] is not None:
                self._decomposition_simple_delta = float(data["decomposition_simple_delta"])
            for mk, attr in (
                ("mesh_included_angle", "_mesh_included_angle"),
                ("mesh_n_smooth_patch", "_mesh_n_smooth_patch"),
                ("mesh_snap_tolerance", "_mesh_snap_tolerance"),
                ("mesh_n_solve_iter", "_mesh_n_solve_iter"),
                ("mesh_n_relax_iter", "_mesh_n_relax_iter"),
                ("mesh_n_feature_snap_iter", "_mesh_n_feature_snap_iter"),
                ("mesh_explicit_feature_snap", "_mesh_explicit_feature_snap"),
                ("mesh_implicit_feature_snap", "_mesh_implicit_feature_snap"),
                ("mesh_multi_region_feature_snap", "_mesh_multi_region_feature_snap"),
                ("mesh_n_cells_between_levels", "_mesh_n_cells_between_levels"),
                ("mesh_resolve_feature_angle", "_mesh_resolve_feature_angle"),
                ("mesh_max_non_ortho", "_mesh_max_non_ortho"),
                ("mesh_max_boundary_skewness", "_mesh_max_boundary_skewness"),
                ("mesh_max_internal_skewness", "_mesh_max_internal_skewness"),
                ("mesh_max_concave", "_mesh_max_concave"),
                ("mesh_min_vol", "_mesh_min_vol"),
                ("mesh_min_tet_quality", "_mesh_min_tet_quality"),
                ("mesh_min_twist", "_mesh_min_twist"),
                ("mesh_min_determinant", "_mesh_min_determinant"),
                ("mesh_min_face_weight", "_mesh_min_face_weight"),
                ("mesh_min_vol_ratio", "_mesh_min_vol_ratio"),
                ("mesh_n_smooth_scale", "_mesh_n_smooth_scale"),
                ("mesh_error_reduction", "_mesh_error_reduction"),
                ("mesh_relaxed_max_non_ortho", "_mesh_relaxed_max_non_ortho"),
            ):
                if mk in data and data[mk] is not None:
                    setattr(self, attr, data[mk])
            if "charge_refinement_level" in data:
                self.spin_charge_refine.setValue(data["charge_refinement_level"])
            if "charge_outer_refine_min" in data:
                self.spin_charge_outer_min.setValue(data["charge_outer_refine_min"])
            else:
                self.spin_charge_outer_min.setValue(self.spin_refine_min.value())
            if "charge_outer_refine_max" in data:
                self.spin_charge_outer_max.setValue(data["charge_outer_refine_max"])
            else:
                self.spin_charge_outer_max.setValue(self.spin_refine_max.value())
            if data.get("charge_outer_refine_enable") is False:
                self.spin_charge_outer_min.setValue(0)
                self.spin_charge_outer_max.setValue(0)
            if "cylinder_axis" in data:
                idx = self.c_cylinder_axis.findText(data["cylinder_axis"])
                if idx >= 0:
                    self.c_cylinder_axis.setCurrentIndex(idx)
            if "charge_backup_radius_factor" in data:
                self.spin_backup_factor.setValue(data["charge_backup_radius_factor"])
            if "buffer_layers" in data:
                self._buffer_layers = int(data["buffer_layers"])
            if "ignition_mode" in data:
                idx = self.combo_ignition_mode.findText(data["ignition_mode"])
                if idx >= 0:
                    self.combo_ignition_mode.setCurrentIndex(idx)
            self._on_ignition_mode_changed(self.combo_ignition_mode.currentText())
            if "write_control_type" in data:
                idx = self.combo_write_control.findText(data["write_control_type"])
                if idx >= 0:
                    self.combo_write_control.setCurrentIndex(idx)
            if "write_interval_steps" in data:
                self.spin_write.setValue(data["write_interval_steps"])
            if "write_interval_time" in data:
                self.spin_write_time.setValue(data["write_interval_time"])
            if "cycle_write" in data:
                self.spin_cycle_write.setValue(data["cycle_write"])
            self._on_write_control_changed(self.combo_write_control.currentText())

            # --- Obstacles (STL) ---
            stl_list = data.get("stl_obstacles", [])
            if stl_list:
                self.obstacles.clear()
                for stl_info in stl_list:
                    path = stl_info.get("path", "")
                    scale = stl_info.get("scale", 1.0)
                    if stl_info.get("exists", False) and path:
                        self.obstacles.append(
                            ObstacleItem(True, path, scale, 0, 0, 0)
                        )
                self._refresh_table()

            # --- Initialize Method (remap detection) ---
            am = data.get("activation_model")
            if am and am.lower() == "none":
                # activationModel none → likely a remap case
                self.rad_init_remap.setChecked(True)
            else:
                self.rad_init_standard.setChecked(True)

        finally:
            self._block_signals = False

        # Refresh UI after all values are set
        self._on_mesh_mode_changed()
        self._update_ui_state()
        self._on_shape_changed(self.c_shape.currentText())
        # Recompute radius from mass/density only when we have both (setRefinedFields format);
        # when loading setFields format we have radius but not mass/rho — keep loaded radius
        if "mass_kg" in data and "rho_charge" in data and data.get("charge_shape") in ("Sphere", "Cylinder"):
            self._update_charge_radius()
        self._update_edit_button_visibility()
        self._update_preview()
        self._update_calculated_dt_label()

    def get_case_inputs(self) -> CaseInputs3D:
        bounds_dict = {k: cb.currentText() for k, cb in self.bound_combos.items()}
        mat_name = self.c_mat.currentText()
        mat_props = self.materials_db.get(mat_name, {})
        # For Custom: pass full JWL so generator_3d can use user-defined A, B, R1, R2, omega, E0
        material_props_out = {}
        if mat_name == "Custom" and isinstance(mat_props, dict):
            material_props_out = {**mat_props, "E0": mat_props.get("energy", 4.5e6)}
        
        obs_data = []
        for obs in self.obstacles:
            if obs.enabled:
                obs_data.append(ObstacleData(
                    stl_path=obs.path, name=os.path.basename(obs.path),
                    scale=obs.scale, offset_x=obs.ox, offset_y=obs.oy, offset_z=obs.oz
                ))

        prov = getattr(self, "_provenance", {})
        refine = self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()
        obs_refine = self.chk_obstacle_refine.isChecked()
        remap_enabled = self.rad_init_remap.isChecked()
        transition_cells = self.spin_transition_cells.value()
        enable_dyn = None if prov.get("enable_dyn_refine") == "UNSET" else refine
        enable_obs = None if prov.get("enable_obstacle_refine") == "UNSET" else obs_refine
        ignition_mode = self.combo_ignition_mode.currentText()
        initiation_point = None if ignition_mode == "Center of Charge" else (self.init_ix.value(), self.init_iy.value(), self.init_iz.value())
        return CaseInputs3D(
            min_point=(self.sx1.value(), self.sy1.value(), self.sz1.value()),
            max_point=(self.sx2.value(), self.sy2.value(), self.sz2.value()),
            cell_size=self.scell.value(),
            charge_center=(self.cx.value(), self.cy.value(), self.cz.value()),
            initiation_point=initiation_point,
            ignition_mode=ignition_mode,
            buffer_layers=getattr(self, "_buffer_layers", 2),
            charge_shape=self.c_shape.currentText(),
            cylinder_radius=self.c_radius.value() if self.c_shape.currentText() in ("Sphere", "Cylinder") else 0.05,
            cylinder_axis=self.c_cylinder_axis.currentText(),
            mass_kg=self.c_mass.value(),
            rho_charge=self.c_rho.value(),
            material_name=mat_name,
            energy_j_per_kg=mat_props.get("energy", 4.5e6),
            p_atm=self.p0.value(), t_atm=self.t0.value(),
            material_props=material_props_out,
            end_time_s=self.spin_end.value(), delta_t=self._compute_safe_dt(),
            write_interval_steps=self.spin_write.value(),
            write_interval_time=self.spin_write_time.value(),
            write_control_type=self.combo_write_control.currentText(),
            cycle_write=self.spin_cycle_write.value(),
            cores=self.spin_cores.value(),
            cfl_value=self.spin_cfl.value(),
            obstacles=obs_data,
            boundaries=bounds_dict,
            enable_local_refinement=refine or obs_refine,
            refine_min=self.spin_refine_min.value(),
            refine_max=self.spin_refine_max.value(),
            enable_dyn_refine=enable_dyn,
            dyn_refine_min=self.spin_refine_min.value(),
            dyn_refine_max=getattr(self, "_dyn_refine_max", 1),
            enable_obstacle_refine=enable_obs,
            obstacle_refine_min=self.spin_obstacle_refine_min.value(),
            obstacle_refine_max=self.spin_obstacle_refine_max.value(),
            # Fixed Mesh: no charge refinement (inside, outside, or startup seed) — mesh stays uniform
            charge_refinement_level=(self.spin_charge_refine.value() if refine else 0),
            charge_outer_refine_enable=(refine and (self.spin_charge_outer_min.value() != 0 or self.spin_charge_outer_max.value() != 0)),
            charge_outer_refine_min=self.spin_charge_outer_min.value(),
            charge_outer_refine_max=self.spin_charge_outer_max.value(),
            transition_cells=transition_cells,
            use_seed_bubble=(refine and self.spin_charge_refine.value() > 0),
            charge_backup_radius_factor=self.spin_backup_factor.value(),
            charge_aspect=self.c_aspect.value(),
            charge_length=self.c_length.value(),
            charge_width=self.c_width.value(),
            charge_height=self.c_height.value(),
            remap_enabled=remap_enabled,
            remap_post_detonation=False,
            remap_source_type=self._remap_source_type,
            remap_case_path=self._remap_case_path,
            remap_origin=(self.spin_remap_ox.value(), self.spin_remap_oy.value(), self.spin_remap_oz.value()),
            remap_time_mode=self._remap_time_mode,
            remap_specific_time=self._remap_specific_time,
            refine_interval=getattr(self, "_refine_interval", 3),
            lower_refine_threshold=getattr(self, "_lower_refine_threshold", 0.1),
            unrefine_threshold=getattr(self, "_unrefine_threshold", 0.1),
            n_buffer_layers_dynamic=getattr(self, "_n_buffer_layers_dynamic", 2),
            enable_balancing=getattr(self, "_enable_balancing", False),
            dynamic_max_cells=getattr(self, "_dynamic_max_cells", 200000000),
            refine_indicator_field=getattr(self, "_refine_indicator_field", "densityGradient"),
            obstacle_feature_angle=getattr(self, "_obstacle_feature_angle", 120),
            obstacle_cells_between_levels=getattr(self, "_obstacle_cells_between_levels", 2),
            obstacle_snap_iter=getattr(self, "_obstacle_snap_iter", 100),
            obstacle_feature_snap_iter=getattr(self, "_obstacle_feature_snap_iter", 15),
            ignition_radius=getattr(self, "_ignition_radius", None),
            eos_model=getattr(self, "_eos_model", None),
            activation_model_ui=getattr(self, "_activation_model_ui", None),
            thermo_model=getattr(self, "_thermo_model", None),
            thermo_model_air=getattr(self, "_thermo_model_air", None),
            decomposition_method=getattr(self, "_decomposition_method", None),
            decomposition_simple_n=getattr(self, "_decomposition_simple_n", None),
            decomposition_simple_delta=getattr(self, "_decomposition_simple_delta", None),
            mesh_included_angle=getattr(self, "_mesh_included_angle", None),
            mesh_n_smooth_patch=getattr(self, "_mesh_n_smooth_patch", None),
            mesh_snap_tolerance=getattr(self, "_mesh_snap_tolerance", None),
            mesh_n_solve_iter=getattr(self, "_mesh_n_solve_iter", None),
            mesh_n_relax_iter=getattr(self, "_mesh_n_relax_iter", None),
            mesh_n_feature_snap_iter=getattr(self, "_mesh_n_feature_snap_iter", None),
            mesh_explicit_feature_snap=getattr(self, "_mesh_explicit_feature_snap", None),
            mesh_implicit_feature_snap=getattr(self, "_mesh_implicit_feature_snap", None),
            mesh_multi_region_feature_snap=getattr(self, "_mesh_multi_region_feature_snap", None),
            mesh_n_cells_between_levels=getattr(self, "_mesh_n_cells_between_levels", None),
            mesh_resolve_feature_angle=getattr(self, "_mesh_resolve_feature_angle", None),
            mesh_max_non_ortho=getattr(self, "_mesh_max_non_ortho", None),
            mesh_max_boundary_skewness=getattr(self, "_mesh_max_boundary_skewness", None),
            mesh_max_internal_skewness=getattr(self, "_mesh_max_internal_skewness", None),
            mesh_max_concave=getattr(self, "_mesh_max_concave", None),
            mesh_min_vol=getattr(self, "_mesh_min_vol", None),
            mesh_min_tet_quality=getattr(self, "_mesh_min_tet_quality", None),
            mesh_min_twist=getattr(self, "_mesh_min_twist", None),
            mesh_min_determinant=getattr(self, "_mesh_min_determinant", None),
            mesh_min_face_weight=getattr(self, "_mesh_min_face_weight", None),
            mesh_min_vol_ratio=getattr(self, "_mesh_min_vol_ratio", None),
            mesh_n_smooth_scale=getattr(self, "_mesh_n_smooth_scale", None),
            mesh_error_reduction=getattr(self, "_mesh_error_reduction", None),
            mesh_relaxed_max_non_ortho=getattr(self, "_mesh_relaxed_max_non_ortho", None),
            provenance=dict(getattr(self, "_provenance", {})),
        )