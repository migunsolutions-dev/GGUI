import math
import os
from typing import List
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QScrollArea, QFrame,
    QTabWidget, QGroupBox, QFormLayout, QGridLayout, QLabel, QPushButton, QDoubleSpinBox,
    QComboBox, QTableWidget, QTableWidgetItem, QFileDialog, QSpinBox,
    QCheckBox, QHeaderView, QRadioButton, QButtonGroup,
    QDialog, QDialogButtonBox, QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal

from probes_model import ProbesModel
from models import CaseInputs3D, ObstacleData
from initialization_plan import (
    build_initialization_plan,
    outer_band_level_string,
    outer_band_will_be_applied,
)
from charge_seed_plan import (
    SEED_MODE_AUTO,
    SEED_MODE_MANUAL,
    SEED_MODE_OFF,
    build_charge_seed_plan,
    seed_status_label,
)
from charge_capture import resolve_charge_capture_radius_m
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
        self._on_mesh_mode_changed()
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
        
        # Info panel: Mesh Plan (live, pre-init) + Initialization Results (post-init only)
        info_font = "font-size: 9pt; color: #333;"
        info_frm = QFrame()
        info_frm.setStyleSheet("background:#eef2f6; border:1px solid #c7d0da; border-radius: 4px;")
        info_frm.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        il = QVBoxLayout(info_frm)
        il.setContentsMargins(6, 6, 6, 6)
        il.setSpacing(4)

        # Title is a real QLabel (not QGroupBox::title) so spacing after it is layout-controlled.
        self.grp_mesh_plan = QFrame()
        self.grp_mesh_plan.setObjectName("meshPlanGroup")
        self.grp_mesh_plan.setStyleSheet(
            "QFrame#meshPlanGroup { border: none; background: transparent; }"
        )
        plan_l = QVBoxLayout(self.grp_mesh_plan)
        plan_l.setContentsMargins(2, 0, 2, 2)
        plan_l.setSpacing(0)

        def _plan_lbl(tip: str = "", parent: QWidget = None) -> QLabel:
            lbl = QLabel("", parent)
            lbl.setStyleSheet(info_font + " background: transparent; border: none;")
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            if tip:
                lbl.setToolTip(tip)
            return lbl

        self.lbl_mesh_plan_title = QLabel("Mesh Plan — Before Initialize", self.grp_mesh_plan)
        self.lbl_mesh_plan_title.setStyleSheet(
            "font-weight: bold; font-size: 9pt; color: #333; background: transparent; border: none;"
        )
        self.lbl_mesh_plan_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        plan_l.addWidget(self.lbl_mesh_plan_title)
        # Clear ~10–12 px gap under the title (layout spacing, not newlines).
        plan_l.addSpacing(11)

        plan_body = QVBoxLayout()
        plan_body.setContentsMargins(0, 0, 0, 0)
        plan_body.setSpacing(4)

        # Compact grouped summaries (not a tall stack of bordered rows)
        self.lbl_plan_block_mesh = _plan_lbl(parent=self.grp_mesh_plan)
        self.lbl_plan_block_seed = _plan_lbl(parent=self.grp_mesh_plan)
        self.lbl_plan_block_status = _plan_lbl(parent=self.grp_mesh_plan)
        plan_body.addWidget(self.lbl_plan_block_mesh)
        plan_body.addWidget(self.lbl_plan_block_seed)
        # Text holders for tests / tooltips — owned by the plan group, never shown alone
        self.lbl_plan_base_grid = _plan_lbl(
            "Base blockMesh grid from domain extents and Cell Size.", parent=self.grp_mesh_plan
        )
        self.lbl_plan_mesh_mode = _plan_lbl(
            "Fixed mesh or runtime AMR (Wave AMR level / finest wave cell).", parent=self.grp_mesh_plan
        )
        self.lbl_plan_init_command = _plan_lbl(
            "Initialization command selected by the current policy.", parent=self.grp_mesh_plan
        )
        self.lbl_plan_charge_seed = _plan_lbl(
            "Requested/effective charge seed level, estimated smallest charge cell, and estimated charge-cell count.",
            parent=self.grp_mesh_plan,
        )
        self.lbl_plan_charge_capture = _plan_lbl(
            "Charge capture status for setFieldsDict backup region (not the physical charge size).",
            parent=self.grp_mesh_plan,
        )
        self.lbl_plan_startup_outer = _plan_lbl(
            "Whether the current configuration emits the startup outer refinement region (chargeRefineOuter).",
            parent=self.grp_mesh_plan,
        )
        self.lbl_plan_initiation = _plan_lbl(
            "Ignition mode, point, and initiation radius.", parent=self.grp_mesh_plan
        )
        for w in (
            self.lbl_plan_base_grid,
            self.lbl_plan_mesh_mode,
            self.lbl_plan_init_command,
            self.lbl_plan_charge_seed,
            self.lbl_plan_charge_capture,
            self.lbl_plan_startup_outer,
            self.lbl_plan_initiation,
        ):
            w.hide()

        self.lbl_charge_resolution_warning = QLabel("")
        self.lbl_charge_resolution_warning.setStyleSheet(
            "font-size: 9pt; color: #c00; font-weight: bold; padding: 2px 0; border: none;"
        )
        self.lbl_charge_resolution_warning.setWordWrap(True)
        self.lbl_charge_resolution_warning.setMinimumHeight(36)
        self.lbl_charge_resolution_warning.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        # Warning sits above initiation so Phase 1F order is: plan rows → warnings → initiation.
        plan_body.addWidget(self.lbl_charge_resolution_warning)
        plan_body.addWidget(self.lbl_plan_block_status)
        plan_l.addLayout(plan_body)
        il.addWidget(self.grp_mesh_plan)

        self.grp_init_results = QGroupBox("Initialization Results")
        self.grp_init_results.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 9pt; margin-top: 6px; border: none; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 2px; padding: 0 2px; }"
        )
        res_l = QVBoxLayout(self.grp_init_results)
        res_l.setContentsMargins(2, 10, 2, 2)
        res_l.setSpacing(2)
        self.lbl_result_block = _plan_lbl(parent=self.grp_init_results)
        res_l.addWidget(self.lbl_result_block)
        self.lbl_result_total_cells = _plan_lbl(
            "Actual total cell count after initialization.", parent=self.grp_init_results
        )
        self.lbl_result_init_command = _plan_lbl(
            "Initialization command actually executed.", parent=self.grp_init_results
        )
        self.lbl_result_charge_cells = _plan_lbl(
            "Number of cells with alpha.c4 above threshold in 0/ after init.",
            parent=self.grp_init_results,
        )
        self.lbl_result_ignition_cells = _plan_lbl(
            "Cells with alpha.c4>thr in the ignition region (when available).",
            parent=self.grp_init_results,
        )
        for w in (
            self.lbl_result_total_cells,
            self.lbl_result_init_command,
            self.lbl_result_charge_cells,
            self.lbl_result_ignition_cells,
        ):
            w.hide()
        self.grp_init_results.hide()
        il.addWidget(self.grp_init_results)

        # Compatibility aliases / hidden holders (not shown as permanent summary rows)
        self.lbl_cells = self.lbl_plan_base_grid
        self.lbl_init_mode = self.lbl_result_init_command
        self.lbl_charge_cells = self.lbl_result_charge_cells
        self.lbl_cells_in_ignition = self.lbl_result_ignition_cells
        self.lbl_charge_capture_info = self.lbl_plan_charge_capture
        self.lbl_est_charge_cells = self.lbl_plan_charge_seed
        self.lbl_smallest_cell = QLabel("", self)
        self.lbl_smallest_cell.hide()
        self.lbl_initiation_radius = QLabel("", self)
        self.lbl_initiation_radius.hide()
        self.lbl_charge_refine_info = QLabel("", self)
        self.lbl_charge_refine_info.hide()
        self.lbl_obstacle_refine_info = QLabel("", self)
        self.lbl_obstacle_refine_info.hide()
        self.lbl_charge_fraction = QLabel("", self)
        self.lbl_charge_fraction.hide()
        self.lbl_cells_inside_charge = QLabel("", self)
        self.lbl_cells_inside_charge.hide()
        self.lbl_charge_clipped = QLabel("", self)
        self.lbl_charge_clipped.hide()
        self.lbl_expected_emesh = QLabel("", self)
        self.lbl_expected_emesh.hide()
        self._init_results_available = False

        setup_layout.addWidget(info_frm)

        scroll = QScrollArea()
        scroll.setWidget(setup_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        # Keep normal horizontal/vertical scrolling (do not force AlwaysOff).
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._left_setup_scroll = scroll
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

        # Cell Size
        f.addWidget(QLabel("Cell Size"), 3, 0)
        cell_row = QWidget(); cell_h = QHBoxLayout(cell_row); cell_h.setContentsMargins(0, 0, 0, 0)
        cell_h.addWidget(self.scell)
        cell_h.addStretch()
        f.addWidget(cell_row, 3, 1)
        # Mesh Properties near grid/mesh controls (own row so the label is not clipped)
        self.btn_mesh_properties = QPushButton("Mesh Properties…")
        self.btn_mesh_properties.setToolTip("Advanced mesh parameters (AMR, charge seed, and obstacle refine).")
        self.btn_mesh_properties.clicked.connect(self._open_mesh_properties_dialog)
        mesh_props_row = QWidget()
        mesh_props_h = QHBoxLayout(mesh_props_row)
        mesh_props_h.setContentsMargins(0, 0, 0, 0)
        mesh_props_h.addWidget(self.btn_mesh_properties)
        mesh_props_h.addStretch()
        f.addWidget(mesh_props_row, 4, 0, 1, 2)
        # Mesh mode: Dyn Mesh default (building3D-style AMR); Fixed Mesh available
        self.rad_fixed_mesh = QRadioButton("Fixed Mesh")
        self.rad_dyn_mesh = QRadioButton("Dyn Mesh (AMR)")
        self.rad_dyn_mesh.setChecked(True)
        self.rad_fixed_mesh.setToolTip("Static mesh (no AMR).")
        self.rad_dyn_mesh.setToolTip("Dynamic mesh (AMR). Wave AMR level controls maxRefinement.")
        mesh_mode_bg = QButtonGroup(self)
        mesh_mode_bg.addButton(self.rad_fixed_mesh)
        mesh_mode_bg.addButton(self.rad_dyn_mesh)
        self.spin_refine_min = QSpinBox()
        self.spin_refine_min.setRange(0, 10)
        self.spin_refine_min.setValue(2)
        self.spin_refine_min.setMaximumWidth(60)
        self.spin_refine_min.setToolTip("Legacy loaded-state value; runtime AMR has no integer minimum level.")
        self.spin_refine_max = QSpinBox()
        self.spin_refine_max.setRange(0, 10)
        self.spin_refine_max.setValue(1)
        self.spin_refine_max.setMaximumWidth(60)
        self.spin_refine_max.setToolTip("Runtime AMR: maxRefinement in constant/dynamicMeshDict (Wave AMR level).")
        self.lbl_wave_amr_cell = QLabel("")
        self.lbl_wave_amr_cell.setStyleSheet("font-size: 9pt; color: #555;")
        self.lbl_wave_amr_cell.setToolTip("Finest runtime wave cell size = Cell Size / 2^(Wave AMR level).")
        mesh_mode_col = QWidget()
        mesh_mode_v = QVBoxLayout(mesh_mode_col)
        mesh_mode_v.setContentsMargins(0, 0, 0, 0)
        mesh_mode_v.setSpacing(2)
        mesh_mode_v.addWidget(QLabel("Mesh mode"))
        mesh_mode_v.addWidget(self.rad_fixed_mesh)
        mesh_mode_v.addWidget(self.rad_dyn_mesh)
        wave_row = QWidget()
        wave_h = QHBoxLayout(wave_row)
        wave_h.setContentsMargins(18, 0, 0, 0)
        wave_h.addWidget(QLabel("Wave AMR level"))
        wave_h.addWidget(self.spin_refine_max)
        wave_h.addWidget(self.lbl_wave_amr_cell)
        wave_h.addStretch()
        mesh_mode_v.addWidget(wave_row)
        self.rad_dyn_mesh.toggled.connect(self._on_mesh_mode_changed)
        self.rad_dyn_mesh.toggled.connect(lambda: self._set_provenance_user("enable_dyn_refine"))
        self.spin_refine_min.valueChanged.connect(self._validate_refine_levels)
        self.spin_refine_max.valueChanged.connect(self._validate_refine_levels)
        self.spin_refine_max.valueChanged.connect(self._on_dyn_refine_max_changed)
        self.spin_refine_max.valueChanged.connect(self._update_mesh_plan_display)
        self.scell.valueChanged.connect(self._update_wave_amr_cell_label)
        f.addWidget(mesh_mode_col, 5, 0, 1, 2)

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

        # --- Geometry fields: Radius, Aspect (L/D), Length, Width, Height ---
        self.c_radius = self._spin(0.001, 100, 0.1, 0.01, 4)
        self.c_radius.setToolTip(
            "Charge radius [m]. For Cylinder: derived from mass, density and L/D "
            "for cylindericalMassToCell (read-only)."
        )
        self.c_aspect = self._spin(0.1, 20, 2.5, 0.1, 2)
        self.c_aspect.setToolTip("Length-to-Diameter ratio (L/D). Only for Cylinder shape.")
        self.c_length = self._spin(0.001, 100, 0.5, 0.01, 4)
        self.c_length.setToolTip(
            "Cuboid length along X [m]. For Cylinder: derived length "
            "(read-only; from mass, density and L/D)."
        )
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
        self.c_rho.setToolTip("Density is locked to selected explosive material.")
        self.cx = self._spin(-100,100,0,0.1,2, max_width=116)
        self.cy = self._spin(-100,100,0,0.1,2, max_width=116)
        self.cz = self._spin(-100,100,0,0.1,2, max_width=116)
        self.init_ix = self._spin(-100, 100, 0, 0.1, 2, max_width=116)
        self.init_iy = self._spin(-100, 100, 0, 0.1, 2, max_width=116)
        self.init_iz = self._spin(-100, 100, 0, 0.1, 2, max_width=116)
        # Charge seed / outer — created here for Mesh Properties Advanced + load/save hosts
        self.combo_charge_seed_mode = QComboBox()
        self.combo_charge_seed_mode.addItems([SEED_MODE_AUTO, SEED_MODE_MANUAL, SEED_MODE_OFF])
        self.combo_charge_seed_mode.setCurrentText(SEED_MODE_AUTO)
        self.combo_charge_seed_mode.setToolTip(
            "Startup charge seed mode: Auto (target cells), Manual (explicit level), or Off."
        )
        self.spin_charge_seed_target = QSpinBox()
        self.spin_charge_seed_target.setRange(1, 20)
        self.spin_charge_seed_target.setValue(8)
        self.spin_charge_seed_target.setMaximumWidth(60)
        self.spin_charge_seed_target.setToolTip(
            "Auto seed: target cells across the smallest charge dimension."
        )
        self.spin_charge_refine = QSpinBox()
        self.spin_charge_refine.setRange(0, 8)
        self.spin_charge_refine.setValue(0)
        self.spin_charge_refine.setMaximumWidth(60)
        self.spin_charge_refine.setToolTip(
            "Manual startup charge seed level (refineInternal). Used when Seed mode = Manual."
        )
        self.chk_charge_outer_enable = QCheckBox("Startup outer region")
        self.chk_charge_outer_enable.setChecked(False)
        self.chk_charge_outer_enable.setToolTip(
            "Emit snappyHexMesh chargeRefineOuter region at Outer level (Dyn Mesh only)."
        )
        self.spin_charge_outer_level = QSpinBox()
        self.spin_charge_outer_level.setRange(0, 10)
        self.spin_charge_outer_level.setValue(3)
        self.spin_charge_outer_level.setMaximumWidth(60)
        self.spin_charge_outer_level.setToolTip(
            "Single mode-inside level for chargeRefineOuter (source of truth)."
        )
        # Legacy min/max mirrors for load compatibility (hidden; synced from level).
        self.spin_charge_outer_min = QSpinBox()
        self.spin_charge_outer_min.setRange(0, 10)
        self.spin_charge_outer_min.setValue(3)
        self.spin_charge_outer_min.setMaximumWidth(60)
        self.spin_charge_outer_min.setToolTip("Legacy mirror of Outer level (min).")
        self.spin_charge_outer_max = QSpinBox()
        self.spin_charge_outer_max.setRange(0, 10)
        self.spin_charge_outer_max.setValue(3)
        self.spin_charge_outer_max.setMaximumWidth(60)
        self.spin_charge_outer_max.setToolTip("Legacy mirror of Outer level (max).")
        self.spin_charge_outer_level.valueChanged.connect(self._sync_charge_outer_level_mirrors)
        
        for _l in (self.lbl_radius, self.lbl_aspect, self.lbl_cylinder_axis, self.lbl_length, self.lbl_width, self.lbl_height):
            _l.setMinimumWidth(LABEL_MINW)
        mass_lbl = self._lbl("Mass [kg]")
        mass_lbl.setMinimumWidth(LABEL_MINW)
        density_lbl = self._lbl("Density")
        density_lbl.setMinimumWidth(LABEL_MINW)
        self.lbl_density = density_lbl

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

        # Charge seed / outer-band controls live in Mesh Properties (Advanced).
        # Keep permanent value widgets for load/save and get_case_inputs, but host them
        # in a hidden owned container so they never float in the main tab layout.
        self.lbl_charge_refinement = QLabel("Charge seed / outer band")
        self.lbl_charge_refinement.setStyleSheet("font-weight: bold;")
        self.spin_transition_cells = QSpinBox()
        self.spin_transition_cells.setRange(1, 10)
        self.spin_transition_cells.setValue(2)
        self.spin_transition_cells.setMaximumWidth(60)
        self.spin_transition_cells.setToolTip(
            "Global nCellsBetweenLevels used by snappyHexMesh. Controls grading "
            "between refinement levels; it does not change outer-region physical extent."
        )
        self.spin_transition_cells.valueChanged.connect(lambda: self._set_provenance_user("transition_cells"))
        self._charge_seed_host = QWidget(self)
        self._charge_seed_host.setObjectName("chargeSeedAdvancedHost")
        seed_host_l = QFormLayout(self._charge_seed_host)
        seed_host_l.setContentsMargins(0, 0, 0, 0)
        seed_host_l.addRow(self.lbl_charge_refinement)
        seed_host_l.addRow("Seed mode", self.combo_charge_seed_mode)
        seed_host_l.addRow("Target cells", self.spin_charge_seed_target)
        seed_host_l.addRow("Manual seed level", self.spin_charge_refine)
        seed_host_l.addRow(self.chk_charge_outer_enable)
        seed_host_l.addRow("Outer level", self.spin_charge_outer_level)
        # Legacy min/max kept as hidden mirrors for load/save compatibility.
        outside_host_row = QWidget()
        outside_host_h = QHBoxLayout(outside_host_row)
        outside_host_h.setContentsMargins(0, 0, 0, 0)
        outside_host_h.addWidget(self.spin_charge_outer_min)
        outside_host_h.addWidget(self.spin_charge_outer_max)
        seed_host_l.addRow("Outside (legacy mirrors)", outside_host_row)
        seed_host_l.addRow("Snappy cells between levels", self.spin_transition_cells)
        # Keep value widgets owned/laid-out, but never painted in the main tab.
        self._charge_seed_host.setAttribute(Qt.WA_DontShowOnScreen, True)
        self._charge_seed_host.setFixedSize(0, 0)
        self._charge_seed_host.hide()

        wrap_geom_center = QWidget()
        f2b = QFormLayout(wrap_geom_center)
        f2b.setHorizontalSpacing(8)
        f2b.setVerticalSpacing(self.ROW_SPACING)
        f2b.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        lbl_charge_geom = QLabel("Charge Geometry")
        lbl_charge_geom.setStyleSheet("font-weight: bold;")
        f2b.addRow(lbl_charge_geom)
        f2b.addRow(QLabel("Center (X, Y, Z)"))
        center_wrap = QWidget()
        center_h = QHBoxLayout(center_wrap)
        center_h.setContentsMargins(0, 0, 0, 0)
        center_h.addWidget(self._tri(self.cx, self.cy, self.cz))
        center_h.addStretch()
        f2b.addRow(center_wrap)

        wrap_initiation = QWidget()
        f2c = QFormLayout(wrap_initiation)
        f2c.setHorizontalSpacing(8)
        f2c.setVerticalSpacing(self.ROW_SPACING)
        f2c.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        lbl_initiation = QLabel("Initiation")
        lbl_initiation.setStyleSheet("font-weight: bold;")
        f2c.addRow(lbl_initiation)
        self.combo_ignition_mode = QComboBox()
        self.combo_ignition_mode.addItems(["Center of Charge", "Manual"])
        self.combo_ignition_mode.setToolTip("Center of Charge = use charge center; Manual = use initiation point below.")
        self.combo_ignition_mode.currentTextChanged.connect(self._on_ignition_mode_changed)
        self.combo_ignition_mode.currentTextChanged.connect(self._update_mesh_plan_display)
        f2c.addRow(self._lbl("Ignition mode"), self.combo_ignition_mode)
        self.lbl_init_pt = QLabel("Initiation point (X, Y, Z)")
        self.lbl_init_pt.setToolTip("Detonation initiation point (used when Ignition mode = Manual).")
        f2c.addRow(self.lbl_init_pt)
        self.init_wrap = QWidget()
        init_h = QHBoxLayout(self.init_wrap)
        init_h.setContentsMargins(0, 0, 0, 0)
        init_h.addWidget(self._tri(self.init_ix, self.init_iy, self.init_iz))
        init_h.addStretch()
        f2c.addRow(self.init_wrap)
        self._on_ignition_mode_changed(self.combo_ignition_mode.currentText())
        for _w in (self.init_ix, self.init_iy, self.init_iz, self.cx, self.cy, self.cz):
            _w.valueChanged.connect(self._update_mesh_plan_display)
        # Density follows selected material; keep field read-only in UI.
        self.c_rho.setEnabled(False)
        self.lbl_density.setEnabled(False)
        self._refine_interval = 3
        self._lower_refine_threshold = 0.1
        self._unrefine_threshold = 0.1
        self._n_buffer_layers_dynamic = 2
        self._buffer_layers = 5
        self._enable_balancing = False
        self._refine_indicator_field = "densityGradient"
        # Snappy outer transition + optional topoSet seed: R_seed = R_charge * factor (not setRefinedFields capture).
        self._bubble_radius_factor = 1.5
        self._outside_extent = None  # None/0 in Mesh Properties = auto transition shell thickness
        # Imported / preserved chargeRefineOuter state (lossless round-trip).
        # Cleared only when the user deliberately edits Outer settings.
        self._charge_outer_mode = None  # "inside" | "distance" | None
        self._charge_outer_distance_levels = None  # list[(distance, level)]
        self._charge_outer_geometry = None  # dict searchable* params
        self._charge_outer_raw_refinement = None
        # Hidden seed-policy fields (defaults match CaseInputs3D; not all have UI controls).
        self._charge_seed_min_cells = 6
        self._charge_seed_max_level = 5
        self._charge_outer_legacy_migration_warning = None
        self._dynamic_max_cells = 200000000
        self._begin_unrefine = None
        self._upper_refine_level = None
        self._upper_unrefine_level = None
        self._balance_interval = None
        # Charge capture (setFieldsDict ``backup``): separate from transition seed above.
        self._charge_capture_mode = "auto"
        self._charge_capture_factor = 1.0
        self._charge_capture_radius_manual = 0.2
        self._charge_backup_radius_override = None
        self._charge_backup_length_override = None
        self._ignition_radius = None
        self._ignition_radius_manual = False
        self._delta_t_loaded = None
        # Run-mode tradeoffs (default: FAST). Both default OFF so a fresh run is as
        # fast as building3D's hand-tuned Allrun. Loaders flip them ON when an
        # opened case explicitly contains the corresponding constructs.
        self._enable_post_processing = False  # functions { impulse; overpressure; fieldMinMax; }
        self._fast_run_mode = True            # skip stage_check / log.stageVerification / checkMesh / check_internal_patch
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
        charge_main.addWidget(wrap_geom_center)
        charge_main.addWidget(wrap_initiation)
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
        """Compatibility wrapper: Mesh Plan owns estimated charge-cell summary."""
        self._update_mesh_plan_display()

    def _update_wave_amr_cell_label(self) -> None:
        """Show finest runtime wave cell size next to Wave AMR level."""
        if not hasattr(self, "lbl_wave_amr_cell"):
            return
        dyn = self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()
        if not dyn:
            self.lbl_wave_amr_cell.setText("")
            self.lbl_wave_amr_cell.setVisible(False)
            return
        h0 = max(1e-12, float(self.scell.value()))
        level = max(0, int(self.spin_refine_max.value()))
        h_wave = h0 / (2.0 ** level)
        self.lbl_wave_amr_cell.setVisible(True)
        self.lbl_wave_amr_cell.setText(f"{h_wave:g} m")

    def _planned_init_command_label(self, inputs: CaseInputs3D) -> str:
        plan = build_initialization_plan(inputs)
        cmd = plan.command
        if cmd.startswith("remap"):
            return "remap"
        return cmd

    def _planned_initiation_radius_m(self, inputs: CaseInputs3D) -> float:
        """Mirror generator initiation-radius request (display only; no policy change)."""
        user_ign = getattr(inputs, "ignition_radius", None)
        if user_ign is not None:
            try:
                return float(user_ign)
            except (TypeError, ValueError):
                pass
        r_charge = float(getattr(inputs, "cylinder_radius", 0.05) or 0.05)
        return min(0.05, max(0.01, 0.2 * r_charge))

    def _mesh_plan_row(self, label: str, value: str) -> str:
        """One Mesh Plan datum per row (label / value)."""
        return f"{label}:    {value}"

    def _update_mesh_plan_display(self) -> None:
        """Live Mesh Plan summaries from current GUI inputs (pre-initialize)."""
        if not hasattr(self, "lbl_plan_base_grid"):
            return

        total = None
        nx = ny = nz = None
        try:
            dx = max(1e-6, self.scell.value())
            nx = max(1, int((self.sx2.value() - self.sx1.value()) / dx))
            ny = max(1, int((self.sy2.value() - self.sy1.value()) / dx))
            nz = max(1, int((self.sz2.value() - self.sz1.value()) / dx))
            total = nx * ny * nz
            self.lbl_plan_base_grid.setText(self._mesh_plan_row("Base grid", f"{nx} × {ny} × {nz}"))
        except (TypeError, ValueError, OverflowError):
            self.lbl_plan_base_grid.setText(self._mesh_plan_row("Base grid", "(invalid)"))

        self._update_wave_amr_cell_label()
        dyn = self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()
        wave_lvl = max(0, int(self.spin_refine_max.value()))
        seed_mode = self.combo_charge_seed_mode.currentText()
        seed_target = int(self.spin_charge_seed_target.value())
        self.lbl_plan_mesh_mode.setText(
            self._mesh_plan_row("Mesh mode", "AMR" if dyn else "Fixed")
        )

        try:
            inputs = self.get_case_inputs()
        except Exception:
            # Still show grid rows even if case-input assembly fails.
            grid_lines = []
            if total is not None:
                grid_lines.append(self._mesh_plan_row("Total cells before refinement", f"{total:,}"))
            if nx is not None:
                grid_lines.append(self._mesh_plan_row("Base grid", f"{nx} × {ny} × {nz}"))
            grid_lines.append(self._mesh_plan_row("Mesh mode", "AMR" if dyn else "Fixed"))
            grid_lines.append(self._mesh_plan_row("Charge seed mode", seed_mode))
            grid_lines.append(self._mesh_plan_row("Charge seed target cells", str(seed_target)))
            grid_lines.append(self._mesh_plan_row("Wave AMR level", str(wave_lvl)))
            self.lbl_plan_block_mesh.setText("\n".join(grid_lines))
            self.lbl_plan_block_seed.hide()
            self.lbl_plan_block_status.hide()
            return

        init_cmd = self._planned_init_command_label(inputs)
        self.lbl_plan_init_command.setText(self._mesh_plan_row("Planned initialization", init_cmd))

        seed_plan = build_charge_seed_plan(inputs)
        init_plan = build_initialization_plan(inputs)
        # Effective level from init plan (Fixed Mesh / Remap force 0; Auto/Manual use seed plan).
        seed_level = int(init_plan.seed_effective)
        if bool(getattr(inputs, "remap_enabled", False)):
            status = "Not applied — Remap"
            seed_level_display = "0 (Not applied — Remap)"
            target_display = "n/a (Remap)"
        elif not (self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()):
            status = "Not applied — Fixed Mesh"
            seed_level_display = "0 (Not applied — Fixed Mesh)"
            target_display = (
                str(seed_plan.target_cells)
                if seed_plan.mode == "Auto"
                else "n/a"
            )
        else:
            status = seed_status_label(seed_plan)
            seed_level_display = str(seed_level)
            target_display = (
                str(seed_plan.target_cells)
                if seed_plan.mode == "Auto"
                else "n/a (Manual/Off)"
            )
        self.lbl_plan_charge_seed.setText(
            self._mesh_plan_row("Charge seed mode", seed_plan.mode)
            + "\n"
            + self._mesh_plan_row("Charge seed target cells", target_display)
            + "\n"
            + self._mesh_plan_row("Charge seed level", seed_level_display)
            + "\n"
            + self._mesh_plan_row("Charge seed status", status)
        )

        # Charge capture status kept for tooltips / compatibility (not a Phase 1F visible row)
        try:
            r_phys = float(getattr(inputs, "cylinder_radius", 0.05) or 0.05)
            _r_cap, report = resolve_charge_capture_radius_m(inputs, r_phys)
            mode = str(report.mode or "auto").lower()
            warnings = list(report.warnings or [])
            if mode == "manual":
                cap_val = "Manual"
            elif warnings:
                cap_val = "Warning"
            else:
                cap_val = "Safe (Auto)"
            self.lbl_plan_charge_capture.setText(self._mesh_plan_row("Charge capture", cap_val))
            tip_parts = [
                report.formula_description or "",
                f"R_cap={report.charge_capture_radius_used_m:.4g} m",
                f"R_phys={report.physical_charge_radius_m:.4g} m",
                f"ratio={report.ratio_capture_to_physical:.3g}",
            ]
            if report.charge_capture_factor is not None:
                tip_parts.append(f"factor={float(report.charge_capture_factor):.3g}")
            tip_parts.extend(warnings)
            self.lbl_plan_charge_capture.setToolTip("\n".join(p for p in tip_parts if p))
        except Exception:
            self.lbl_plan_charge_capture.setText(self._mesh_plan_row("Charge capture", "Safe (Auto)"))

        outer_level = None
        if outer_band_will_be_applied(inputs):
            level_str = outer_band_level_string(inputs) or ""
            parts = level_str.split()
            try:
                outer_level = max(int(p) for p in parts) if parts else None
            except ValueError:
                outer_level = None
            self.lbl_plan_startup_outer.setText(self._mesh_plan_row("Startup outer region", "On"))
        else:
            self.lbl_plan_startup_outer.setText(self._mesh_plan_row("Startup outer region", "Off"))

        ign_mode = self.combo_ignition_mode.currentText()
        r_ign = self._planned_initiation_radius_m(inputs)
        if ign_mode == "Manual":
            loc = (
                f"Manual point ({self.init_ix.value():g}, "
                f"{self.init_iy.value():g}, {self.init_iz.value():g})"
            )
        else:
            loc = "Charge center"
        self.lbl_plan_initiation.setText(
            self._mesh_plan_row("Initiation location", loc)
            + "\n"
            + self._mesh_plan_row("Initiation radius", f"{r_ign:g} m")
        )

        est = self._estimate_charge_cells()
        warn_parts = []
        if est > 0 and est < 8:
            warn_parts.append("Warning: Charge resolution is too low. Blast may fail.")
        for w in seed_plan.warnings or ():
            text = str(w or "").strip()
            if not text:
                continue
            # Independence note is informational; surface safety/cap warnings only.
            if "intentionally independent" in text.lower():
                continue
            warn_parts.append(text)
        self.lbl_charge_resolution_warning.setText("\n".join(warn_parts))
        if not warn_parts:
            self.lbl_charge_resolution_warning.setToolTip("")

        # Visible Mesh Plan Phase 1F order (one datum per row; no h0 / no arrows).
        plan_lines = []
        if total is not None:
            plan_lines.append(self._mesh_plan_row("Total cells before refinement", f"{total:,}"))
        if nx is not None:
            plan_lines.append(self._mesh_plan_row("Base grid", f"{nx} × {ny} × {nz}"))
        plan_lines.append(self._mesh_plan_row("Mesh mode", "AMR" if dyn else "Fixed"))
        plan_lines.append(self._mesh_plan_row("Charge seed mode", seed_plan.mode))
        plan_lines.append(self._mesh_plan_row("Charge seed target cells", str(seed_plan.target_cells)))
        plan_lines.append(self._mesh_plan_row("Charge seed level", str(seed_level)))
        plan_lines.append(self._mesh_plan_row("Charge seed status", status))
        plan_lines.append(self._mesh_plan_row("Wave AMR level", str(wave_lvl)))
        plan_lines.append(self._mesh_plan_row("Planned initialization", init_cmd))
        plan_lines.append(self.lbl_plan_startup_outer.text())
        if outer_level is not None:
            plan_lines.append(self._mesh_plan_row("Startup outer level", str(outer_level)))

        self.lbl_plan_block_mesh.setText("\n".join(plan_lines))
        self.lbl_plan_block_mesh.setToolTip(self.lbl_plan_charge_capture.toolTip())
        self.lbl_plan_block_seed.hide()
        # Initiation rows after warnings (layout order: mesh → warning → status).
        self.lbl_plan_block_status.setText(
            self._mesh_plan_row("Initiation location", loc)
            + "\n"
            + self._mesh_plan_row("Initiation radius", f"{r_ign:g} m")
        )
        self.lbl_plan_block_status.show()

    def _refresh_init_results_block(self) -> None:
        lines = []
        for lbl in (
            self.lbl_result_total_cells,
            self.lbl_result_init_command,
            self.lbl_result_charge_cells,
            self.lbl_result_ignition_cells,
        ):
            t = (lbl.text() or "").strip()
            if t:
                lines.append(t)
        self.lbl_result_block.setText("\n".join(lines))
        self.lbl_result_block.setVisible(bool(lines))

    def _set_init_results_visible(self, visible: bool) -> None:
        self._init_results_available = bool(visible)
        self.grp_init_results.setVisible(bool(visible))
        if visible:
            self._refresh_init_results_block()
        else:
            self.lbl_result_block.setText("")
            self.lbl_result_block.setVisible(False)

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
        self.spin_end = self._spin(0, 100, 0.030, 0.001, 10)
        self.spin_end.setMaximumWidth(120)
        self.spin_end.setToolTip(
            "Simulation end time [s]. Resolution down to 1e-10 s — values like 2e-6 are preserved.\n"
            "Typical blast: 1e-3..1e-2 s (close-in), 1e-2..1 s (far-field)."
        )
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
        self.spin_write_time.setRange(1e-10, 1.0)
        self.spin_write_time.setValue(5e-5)
        self.spin_write_time.setDecimals(10)
        self.spin_write_time.setSingleStep(1e-5)
        self.spin_write_time.setMaximumWidth(120)
        self.spin_write_time.setToolTip(
            "Write results every T seconds of simulation time (adjustableRunTime).\n"
            "Resolution down to 1e-10 s; values like 5e-7 are preserved."
        )
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
        self.btn_exact_end.setToolTip("Continue run until stop or end time (full run / resume-to-end).")
        self.btn_exact_end.clicked.connect(self.sig_request_run_exact_end.emit)
        self.btn_stop = QPushButton("⏸ Interrupt"); self.btn_stop.clicked.connect(self.sig_request_stop.emit)
        self.btn_exec_advanced = QPushButton("Execution / Diagnostics Advanced…")
        self.btn_exec_advanced.setToolTip("Run-mode tradeoffs: post-processing functions and optional verification skips.")
        self.btn_exec_advanced.clicked.connect(self._open_execution_advanced_dialog)
        self.btn_init.setStyleSheet("background-color: #3498db; color: white; padding: 5px;")
        self.btn_exact_1.setStyleSheet("background-color: #9b59b6; color: white; padding: 4px;")
        self.btn_exact_end.setStyleSheet("background-color: #1abc9c; color: white; padding: 4px;")
        self.btn_stop.setStyleSheet("background-color: #e67e22; color: white; padding: 5px;")
        v2.addWidget(self.btn_init)
        v2.addWidget(self.btn_exact_1)
        v2.addWidget(self.btn_exact_end)
        v2.addWidget(self.btn_stop)
        v2.addWidget(self.btn_exec_advanced)
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
        """Update actual cell count in Initialization Results when a mesh is loaded."""
        if count is None or int(count) <= 0:
            return
        self.lbl_result_total_cells.setText(f"Total cells: {int(count):,}")
        self._set_init_results_visible(True)

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

    def _sync_charge_outer_level_mirrors(self, *_args):
        """Keep legacy min/max spins mirrored to spin_charge_outer_level (source of truth)."""
        lvl = int(self.spin_charge_outer_level.value())
        self.spin_charge_outer_min.blockSignals(True)
        self.spin_charge_outer_max.blockSignals(True)
        self.spin_charge_outer_min.setValue(lvl)
        self.spin_charge_outer_max.setValue(lvl)
        self.spin_charge_outer_min.blockSignals(False)
        self.spin_charge_outer_max.blockSignals(False)

    def _validate_charge_outer_levels(self):
        """Compatibility: prefer outer level spin; keep legacy min <= max."""
        if hasattr(self, "spin_charge_outer_level"):
            self._sync_charge_outer_level_mirrors()
            return
        a, b = self.spin_charge_outer_min.value(), self.spin_charge_outer_max.value()
        if a > b:
            self.spin_charge_outer_min.blockSignals(True)
            self.spin_charge_outer_min.setValue(b)
            self.spin_charge_outer_min.blockSignals(False)

    def _update_charge_seed_controls_enabled(self):
        """Enable Advanced seed hosts for Dyn Mesh; target only in Auto; Manual level only in Manual."""
        dyn = self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()
        mode = self.combo_charge_seed_mode.currentText()
        self.lbl_charge_refinement.setEnabled(dyn)
        self.combo_charge_seed_mode.setEnabled(dyn)
        self.spin_charge_seed_target.setEnabled(dyn and mode == SEED_MODE_AUTO)
        self.spin_charge_refine.setEnabled(dyn and mode == SEED_MODE_MANUAL)
        self.chk_charge_outer_enable.setEnabled(dyn)
        outer_on = dyn and self.chk_charge_outer_enable.isChecked()
        self.spin_charge_outer_level.setEnabled(outer_on)
        self.spin_charge_outer_min.setEnabled(outer_on)
        self.spin_charge_outer_max.setEnabled(outer_on)
        self.spin_transition_cells.setEnabled(dyn)

    def _compute_safe_dt(self) -> float:
        if getattr(self, "_delta_t_loaded", None) is not None:
            return float(self._delta_t_loaded)
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
        """Dyn Mesh: enable Wave AMR level and charge-seed Advanced controls. Fixed Mesh: disable them."""
        dyn = self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()
        self.spin_refine_min.setEnabled(dyn)
        self.spin_refine_max.setEnabled(dyn)
        self._update_charge_seed_controls_enabled()
        self._update_wave_amr_cell_label()
        self._update_calculated_dt_label()
        self._update_mesh_plan_display()

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
        self.c_aspect.valueChanged.connect(self._update_cylinder_derived_geometry)
        self.c_mass.valueChanged.connect(self._update_cylinder_derived_geometry)
        self.c_rho.valueChanged.connect(self._update_cylinder_derived_geometry)
        self.c_length.valueChanged.connect(self._update_cuboid_height)
        self.c_width.valueChanged.connect(self._update_cuboid_height)
        
        # TIKUN: Connect Domain Bounds changes to Section updates!
        self.sx1.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sx2.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sy1.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sy2.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sz1.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        self.sz2.valueChanged.connect(lambda: self._update_sections_from_table(-1, -1))
        # Live Mesh Plan (base grid, seed, capture, outer band, initiation)
        for w in [
            self.scell,
            self.spin_charge_refine,
            self.spin_charge_seed_target,
            self.spin_charge_outer_level,
            self.spin_charge_outer_min,
            self.spin_charge_outer_max,
            self.spin_transition_cells,
            self.c_mass,
            self.c_rho,
            self.c_radius,
            self.c_length,
            self.c_width,
            self.c_height,
            self.c_aspect,
        ]:
            w.valueChanged.connect(self._update_mesh_plan_display)
        self.combo_charge_seed_mode.currentTextChanged.connect(self._on_charge_seed_mode_changed)
        self.chk_charge_outer_enable.toggled.connect(self._on_charge_outer_enable_changed)
        self.spin_charge_outer_level.valueChanged.connect(self._on_charge_outer_level_changed)
        self.c_shape.currentIndexChanged.connect(self._update_mesh_plan_display)
        self.rad_init_standard.toggled.connect(self._update_mesh_plan_display)
        self.rad_init_remap.toggled.connect(self._update_mesh_plan_display)

    def _invalidate_imported_outer_state(self) -> None:
        """Drop retained imported outer geometry/mode after deliberate Outer edits.

        Transition: subsequent generation uses canonical GGUI mode inside + UI level
        and rebuilt searchable geometry from outside_extent / bubble_radius_factor.
        """
        if getattr(self, "_block_signals", False):
            return
        self._charge_outer_geometry = None
        self._charge_outer_distance_levels = None
        self._charge_outer_raw_refinement = None
        self._charge_outer_mode = "inside"

    def _on_charge_seed_mode_changed(self, *_args):
        self._update_charge_seed_controls_enabled()
        self._update_mesh_plan_display()

    def _on_charge_outer_enable_changed(self, *_args):
        self._invalidate_imported_outer_state()
        self._update_charge_seed_controls_enabled()
        self._update_mesh_plan_display()

    def _on_charge_outer_level_changed(self, *_args):
        self._invalidate_imported_outer_state()
        self._update_mesh_plan_display()

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
        spin_buf_setfields.setValue(getattr(self, "_buffer_layers", 5))
        spin_buf_setfields.setToolTip("nBufferLayers in setFieldsDict. Default 5.")
        f_buf.addRow("Buffer layers (setFields)", spin_buf_setfields)
        f_buf.addRow("", QLabel("nBufferLayers for charge/refinement regions (default: 5)."))
        cap_hint = QLabel(
            "Charge capture radius is used only to seed the explosive on a coarse base mesh "
            "(setRefinedFields backup region). It is not the physical charge radius and must not "
            "be treated as a large initial refinement bubble unless you also enlarge outer transition "
            "or AMR settings explicitly."
        )
        cap_hint.setWordWrap(True)
        f_buf.addRow(cap_hint)
        rad_cap_auto = QRadioButton("Auto charge capture radius")
        rad_cap_manual = QRadioButton("Manual charge capture radius [m]")
        cap_mode = str(getattr(self, "_charge_capture_mode", "auto") or "auto").lower()
        if cap_mode == "manual":
            rad_cap_manual.setChecked(True)
        else:
            rad_cap_auto.setChecked(True)
        bg_cap = QButtonGroup(d)
        bg_cap.addButton(rad_cap_auto)
        bg_cap.addButton(rad_cap_manual)
        cap_mode_row = QWidget()
        cap_mode_h = QHBoxLayout(cap_mode_row)
        cap_mode_h.setContentsMargins(0, 0, 0, 0)
        cap_mode_h.addWidget(rad_cap_auto)
        cap_mode_h.addWidget(rad_cap_manual)
        cap_mode_h.addStretch()
        f_buf.addRow("Capture mode", cap_mode_row)
        spin_cap_factor = QDoubleSpinBox()
        spin_cap_factor.setRange(0.01, 20.0)
        spin_cap_factor.setDecimals(4)
        spin_cap_factor.setSingleStep(0.05)
        _ccf_dlg = getattr(self, "_charge_capture_factor", None)
        spin_cap_factor.setValue(float(1.0 if _ccf_dlg is None else _ccf_dlg))
        spin_cap_factor.setToolTip(
            "Auto mode: R_capture = max(1.05·R_charge, 0.5·√(dx²+dy²+dz²)·factor). "
            "Uniform base mesh uses dx = dy = dz = Cell Size."
        )
        f_buf.addRow("Charge capture factor (auto)", spin_cap_factor)
        spin_cap_radius = QDoubleSpinBox()
        spin_cap_radius.setRange(1e-6, 1000.0)
        spin_cap_radius.setDecimals(8)
        spin_cap_radius.setSingleStep(0.01)
        _mrad_dlg = getattr(self, "_charge_capture_radius_manual", None)
        spin_cap_radius.setValue(float(0.2 if _mrad_dlg is None else _mrad_dlg))
        spin_cap_radius.setToolTip("Manual mode: exact radius written to setFieldsDict backup { radius ... } — no hidden minimum.")
        f_buf.addRow("Capture radius (manual)", spin_cap_radius)

        def _refresh_cap_ctrls() -> None:
            manual_on = rad_cap_manual.isChecked()
            spin_cap_radius.setEnabled(manual_on)
            spin_cap_factor.setEnabled(not manual_on)

        _refresh_cap_ctrls()
        rad_cap_auto.toggled.connect(lambda _v: _refresh_cap_ctrls())
        rad_cap_manual.toggled.connect(lambda _v: _refresh_cap_ctrls())
        v.addWidget(grp_buf)

        grp_transition = QGroupBox("Initial dense / transition (charge outer shell)")
        f_tr = QFormLayout(grp_transition)
        spin_out_extent = QDoubleSpinBox()
        spin_out_extent.setRange(0.0, 1e6)
        spin_out_extent.setDecimals(6)
        spin_out_extent.setSingleStep(0.01)
        _oe = getattr(self, "_outside_extent", None)
        spin_out_extent.setValue(0.0 if _oe is None else float(_oe))
        spin_out_extent.setToolTip("Physical thickness [m] added to the charge surface for chargeRefineOuter / outsideShell. "
                                   "0 = Auto (legacy bubble_radius_factor + transition band; see case_init_mode.json).")
        f_tr.addRow("Outside extent (0 = Auto) [m]", spin_out_extent)
        f_tr.addRow("", QLabel("Does not set charge capture radius. Transition shape follows charge geometry (sphere / cylinder / box)."))
        v.addWidget(grp_transition)

        # Charge seed / outer band (relocated from main Charge Properties)
        grp_charge_seed = QGroupBox("Charge seed / outer band (Advanced)")
        f_seed_adv = QFormLayout(grp_charge_seed)
        combo_seed_mode = QComboBox()
        combo_seed_mode.addItems([SEED_MODE_AUTO, SEED_MODE_MANUAL, SEED_MODE_OFF])
        combo_seed_mode.setCurrentText(self.combo_charge_seed_mode.currentText())
        combo_seed_mode.setToolTip(self.combo_charge_seed_mode.toolTip())
        combo_seed_mode.setEnabled(self.combo_charge_seed_mode.isEnabled())
        f_seed_adv.addRow("Seed mode", combo_seed_mode)
        spin_seed_target = QSpinBox()
        spin_seed_target.setRange(self.spin_charge_seed_target.minimum(), self.spin_charge_seed_target.maximum())
        spin_seed_target.setValue(self.spin_charge_seed_target.value())
        spin_seed_target.setToolTip(self.spin_charge_seed_target.toolTip())
        spin_seed_target.setEnabled(self.spin_charge_seed_target.isEnabled())
        f_seed_adv.addRow("Target cells", spin_seed_target)
        spin_seed_inside = QSpinBox()
        spin_seed_inside.setRange(self.spin_charge_refine.minimum(), self.spin_charge_refine.maximum())
        spin_seed_inside.setValue(self.spin_charge_refine.value())
        spin_seed_inside.setToolTip(self.spin_charge_refine.toolTip())
        f_seed_adv.addRow("Manual seed level", spin_seed_inside)
        chk_outer_enable = QCheckBox("Startup outer region")
        chk_outer_enable.setChecked(self.chk_charge_outer_enable.isChecked())
        chk_outer_enable.setToolTip(self.chk_charge_outer_enable.toolTip())
        chk_outer_enable.setEnabled(self.chk_charge_outer_enable.isEnabled())
        f_seed_adv.addRow(chk_outer_enable)
        spin_out_level = QSpinBox()
        spin_out_level.setRange(self.spin_charge_outer_level.minimum(), self.spin_charge_outer_level.maximum())
        spin_out_level.setValue(self.spin_charge_outer_level.value())
        spin_out_level.setToolTip(self.spin_charge_outer_level.toolTip())
        f_seed_adv.addRow("Outer level", spin_out_level)
        spin_trans = QSpinBox()
        spin_trans.setRange(self.spin_transition_cells.minimum(), self.spin_transition_cells.maximum())
        spin_trans.setValue(self.spin_transition_cells.value())
        spin_trans.setToolTip(
            "Global nCellsBetweenLevels used by snappyHexMesh. Controls grading "
            "between refinement levels; it does not change outer-region physical extent."
        )
        spin_trans.setEnabled(self.spin_transition_cells.isEnabled())
        f_seed_adv.addRow("Snappy cells between levels", spin_trans)
        tip_trans = QLabel(
            "Snappy cells between levels sets global nCellsBetweenLevels for grading "
            "between refinement levels; it does not change outer-region size in metres."
        )
        tip_trans.setWordWrap(True)
        f_seed_adv.addRow(tip_trans)
        hint_seed = QLabel(
            "Fixed Mesh disables seed application at initialize (mode is left unchanged). "
            "Manual seed level is used only when Seed mode = Manual."
        )
        hint_seed.setWordWrap(True)
        f_seed_adv.addRow(hint_seed)

        def _refresh_seed_adv_enabled() -> None:
            dyn = self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()
            mode = combo_seed_mode.currentText()
            combo_seed_mode.setEnabled(dyn)
            spin_seed_target.setEnabled(dyn and mode == SEED_MODE_AUTO)
            spin_seed_inside.setEnabled(dyn and mode == SEED_MODE_MANUAL)
            chk_outer_enable.setEnabled(dyn)
            spin_out_level.setEnabled(dyn and chk_outer_enable.isChecked())
            spin_trans.setEnabled(dyn)

        _refresh_seed_adv_enabled()
        combo_seed_mode.currentTextChanged.connect(lambda _t: _refresh_seed_adv_enabled())
        chk_outer_enable.toggled.connect(lambda _v: _refresh_seed_adv_enabled())
        v.addWidget(grp_charge_seed)

        # Group: Dyn Refine (AMR) – only core controls
        grp_amr = QGroupBox("Dyn Refine (AMR)")
        f_amr = QFormLayout(grp_amr)
        spin_ref_int = QSpinBox()
        spin_ref_int.setRange(1, 100)
        spin_ref_int.setValue(self._refine_interval)
        f_amr.addRow("Refine Interval", spin_ref_int)
        f_amr.addRow("", QLabel("How often to refine (building3D: 3)."))
        spin_lower = QDoubleSpinBox()
        spin_lower.setRange(0.01, 1.0)
        spin_lower.setValue(self._lower_refine_threshold)
        spin_lower.setDecimals(3)
        f_amr.addRow("Refine Threshold", spin_lower)
        f_amr.addRow("", QLabel("lowerRefineLevel: refine field in range (building3D: 0.1)."))
        spin_unref = QDoubleSpinBox()
        spin_unref.setRange(0.01, 20.0)
        spin_unref.setValue(self._unrefine_threshold)
        spin_unref.setDecimals(3)
        f_amr.addRow("Unrefine Threshold", spin_unref)
        f_amr.addRow("", QLabel("unrefineLevel: if value &lt; this, unrefine (building3D: 0.1)."))
        spin_amr_max = QSpinBox()
        spin_amr_max.setRange(0, 10)
        spin_amr_max.setValue(int(getattr(self, "_dyn_refine_max", self.spin_refine_max.value())))
        f_amr.addRow("Max Refinement (AMR)", spin_amr_max)
        f_amr.addRow("", QLabel("maxRefinement in dynamicMeshDict (building3D: 1)."))
        spin_dyn_max_cells = QSpinBox()
        spin_dyn_max_cells.setRange(1_000, 2_000_000_000)
        spin_dyn_max_cells.setSingleStep(100_000)
        spin_dyn_max_cells.setValue(int(getattr(self, "_dynamic_max_cells", 200000000)))
        f_amr.addRow("Max cells (maxCells)", spin_dyn_max_cells)
        spin_begin_unref = QDoubleSpinBox()
        spin_begin_unref.setRange(-1.0, 1.0e12)
        spin_begin_unref.setDecimals(8)
        spin_begin_unref.setSpecialValueText("(omit)")
        _bu = getattr(self, "_begin_unrefine", None)
        spin_begin_unref.setValue(-1.0 if _bu is None else float(_bu))
        f_amr.addRow("beginUnrefine (time)", spin_begin_unref)
        spin_upper_ref = QDoubleSpinBox()
        spin_upper_ref.setRange(-1.0, 1.0e6)
        spin_upper_ref.setDecimals(6)
        spin_upper_ref.setSpecialValueText("(omit)")
        _ur = getattr(self, "_upper_refine_level", None)
        spin_upper_ref.setValue(-1.0 if _ur is None else float(_ur))
        f_amr.addRow("upperRefineLevel", spin_upper_ref)
        spin_upper_ur = QDoubleSpinBox()
        spin_upper_ur.setRange(-1.0, 1.0e6)
        spin_upper_ur.setDecimals(6)
        spin_upper_ur.setSpecialValueText("(omit)")
        _uur = getattr(self, "_upper_unrefine_level", None)
        spin_upper_ur.setValue(-1.0 if _uur is None else float(_uur))
        f_amr.addRow("upperUnrefineLevel", spin_upper_ur)
        chk_bal = QCheckBox()
        chk_bal.setChecked(bool(getattr(self, "_enable_balancing", False)))
        f_amr.addRow("Enable load balancing", chk_bal)
        spin_bal_int = QSpinBox()
        spin_bal_int.setRange(0, 1_000_000)
        _bi = getattr(self, "_balance_interval", None)
        spin_bal_int.setValue(0 if _bi is None else int(_bi))
        spin_bal_int.setToolTip("Written inside loadBalance { balanceInterval ... } when load balancing is on. 0 = omit.")
        f_amr.addRow("balanceInterval (0 = omit)", spin_bal_int)
        combo_refine_indicator = QComboBox()
        combo_refine_indicator.addItem("densityGradient", "densityGradient")
        combo_refine_indicator.addItem("scaledDelta (pressure p)", "scaledDelta_p")
        cur_ri = str(getattr(self, "_refine_indicator_field", "densityGradient") or "densityGradient")
        if cur_ri.strip().lower() in ("pressuregradient", "pressure", "scaleddelta_p", "scaleddelta"):
            combo_refine_indicator.setCurrentIndex(1)
        else:
            combo_refine_indicator.setCurrentIndex(0)
        f_amr.addRow("AMR error estimator", combo_refine_indicator)
        f_amr.addRow("", QLabel("densityGradient = default blast AMR. Pressure mode uses OpenFOAM scaledDelta on field p (not bare pressureGradient)."))
        v.addWidget(grp_amr)

        # Seeding and Initiation
        grp_seed = QGroupBox("Seeding and Initiation")
        f_seed = QFormLayout(grp_seed)
        ign_row = QWidget()
        ign_h = QHBoxLayout(ign_row)
        ign_h.setContentsMargins(0, 0, 0, 0)
        chk_ign_manual = QCheckBox()
        chk_ign_manual.setChecked(bool(getattr(self, "_ignition_radius_manual", False)))
        lbl_ign_radius = QLabel("Ignition radius")
        spin_ign_radius = QDoubleSpinBox()
        spin_ign_radius.setRange(1e-4, 1.0)
        spin_ign_radius.setDecimals(6)
        spin_ign_radius.setSingleStep(0.001)
        auto_ign = 0.05
        if getattr(self, "_ignition_radius", None) is not None:
            auto_ign = float(self._ignition_radius)
        spin_ign_radius.setValue(auto_ign)
        def _update_ign_row_style() -> None:
            manual = chk_ign_manual.isChecked()
            lbl_ign_radius.setStyleSheet("color: black;" if manual else "color: gray;")
            spin_ign_radius.setEnabled(manual)
        _update_ign_row_style()
        chk_ign_manual.toggled.connect(lambda _v: _update_ign_row_style())
        ign_h.addWidget(chk_ign_manual)
        ign_h.addWidget(lbl_ign_radius)
        ign_h.addWidget(spin_ign_radius)
        ign_h.addStretch()
        f_seed.addRow(ign_row)
        v.addWidget(grp_seed)

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
        spin_minvol.setDecimals(20)
        spin_minvol.setRange(1e-20, 1.0)
        spin_minvol.setSingleStep(1e-13)
        spin_minvol.setValue(1e-13 if getattr(self, "_mesh_min_vol", None) is None else self._mesh_min_vol)
        spin_minvol.setToolTip("meshQualityControls minVol. Resolution down to 1e-20 — small values like 1e-13 are preserved.")
        f_geom.addRow(_mesh_row("Minimum cell volume", "meshQualityControls minVol.", spin_minvol))
        spin_mintet = QDoubleSpinBox()
        spin_mintet.setDecimals(20)
        spin_mintet.setRange(1e-20, 1.0)
        spin_mintet.setSingleStep(1e-15)
        spin_mintet.setValue(1e-15 if getattr(self, "_mesh_min_tet_quality", None) is None else self._mesh_min_tet_quality)
        spin_mintet.setToolTip("meshQualityControls minTetQuality. Resolution down to 1e-20 — small values like 1e-15 are preserved.")
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
            self._charge_capture_mode = "manual" if rad_cap_manual.isChecked() else "auto"
            self._charge_capture_factor = float(spin_cap_factor.value())
            self._charge_capture_radius_manual = float(spin_cap_radius.value())
            if self._charge_capture_mode == "manual":
                self._charge_backup_radius_override = self._charge_capture_radius_manual
            else:
                self._charge_backup_radius_override = None
            self.combo_charge_seed_mode.setCurrentText(combo_seed_mode.currentText())
            self.spin_charge_seed_target.setValue(int(spin_seed_target.value()))
            self.spin_charge_refine.setValue(int(spin_seed_inside.value()))
            self.chk_charge_outer_enable.setChecked(chk_outer_enable.isChecked())
            self.spin_charge_outer_level.setValue(int(spin_out_level.value()))
            self._sync_charge_outer_level_mirrors()
            self.spin_transition_cells.setValue(int(spin_trans.value()))
            self._set_provenance_user("transition_cells")
            self._update_charge_seed_controls_enabled()
            self._refine_interval = spin_ref_int.value()
            self._lower_refine_threshold = spin_lower.value()
            self._unrefine_threshold = spin_unref.value()
            self._dyn_refine_max = spin_amr_max.value()
            self.spin_refine_max.setValue(self._dyn_refine_max)
            self._dynamic_max_cells = int(spin_dyn_max_cells.value())
            self._begin_unrefine = None if spin_begin_unref.value() < -0.5 else float(spin_begin_unref.value())
            self._upper_refine_level = None if spin_upper_ref.value() < -0.5 else float(spin_upper_ref.value())
            self._upper_unrefine_level = None if spin_upper_ur.value() < -0.5 else float(spin_upper_ur.value())
            self._enable_balancing = chk_bal.isChecked()
            self._balance_interval = None if spin_bal_int.value() <= 0 else int(spin_bal_int.value())
            v_oe = float(spin_out_extent.value())
            prev_oe = getattr(self, "_outside_extent", None)
            self._outside_extent = None if v_oe <= 0.0 else v_oe
            # Deliberate Outside extent edit invalidates imported outer geometry.
            if prev_oe != self._outside_extent:
                self._invalidate_imported_outer_state()
            self._refine_indicator_field = str(combo_refine_indicator.currentData() or "densityGradient")
            for k in (
                "outside_extent",
                "dynamic_max_cells",
                "begin_unrefine",
                "upper_refine_level",
                "upper_unrefine_level",
                "balance_interval",
                "enable_balancing",
            ):
                self._set_provenance_user(k)
            self._ignition_radius_manual = chk_ign_manual.isChecked()
            if self._ignition_radius_manual:
                self._ignition_radius = spin_ign_radius.value()
            else:
                self._ignition_radius = None
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
            for k in ("refine_interval", "lower_refine_threshold", "unrefine_threshold", "refine_indicator_field"):
                self._set_provenance_user(k)
            for k in ("mesh_included_angle", "mesh_n_smooth_patch", "mesh_snap_tolerance", "mesh_n_solve_iter", "mesh_n_relax_iter", "mesh_n_feature_snap_iter", "mesh_explicit_feature_snap", "mesh_implicit_feature_snap", "mesh_multi_region_feature_snap", "mesh_n_cells_between_levels", "mesh_resolve_feature_angle", "mesh_max_non_ortho", "mesh_max_boundary_skewness", "mesh_max_internal_skewness", "mesh_max_concave", "mesh_min_vol", "mesh_min_tet_quality", "mesh_min_twist", "mesh_min_determinant", "mesh_min_face_weight", "mesh_min_vol_ratio", "mesh_n_smooth_scale", "mesh_error_reduction", "mesh_relaxed_max_non_ortho"):
                self._set_provenance_user(k)
            self._update_mesh_plan_display()

    def _open_execution_advanced_dialog(self):
        """Execution / Diagnostics Advanced: run-mode vs verification tradeoffs."""
        d = QDialog(self)
        d.setWindowTitle("Execution / Diagnostics Advanced")
        layout = QVBoxLayout(d)
        grp_run = QGroupBox("Run mode (speed vs verification)")
        f_run = QFormLayout(grp_run)
        chk_post_proc = QCheckBox()
        chk_post_proc.setChecked(bool(getattr(self, "_enable_post_processing", False)))
        chk_post_proc.setToolTip(
            "Add controlDict 'functions { impulse; overpressure; fieldMinMax; }'.\n"
            "Useful for downstream analysis (max overpressure, impulse fields), but\n"
            "adds work at every writeTime. Default OFF for faster runs."
        )
        f_run.addRow("Write impulse / overpressure / fieldMinMax", chk_post_proc)
        f_run.addRow(
            "",
            QLabel(
                "OFF = building3D-style fast solver (no postProcess at writeTime).\n"
                "ON = adds 'functions {...}' block; useful for analysis but slower."
            ),
        )
        chk_fast = QCheckBox()
        chk_fast.setChecked(bool(getattr(self, "_fast_run_mode", True)))
        chk_fast.setToolTip(
            "Fast Allrun: skip optional verification (stage_check / log.stageVerification\n"
            "/ checkMesh / check_internal_patch). check_alpha_c4.sh still gates the\n"
            "solver against 'no mass in domain'. Default ON.\n"
            "Turn OFF only when debugging mesh / charge initialization."
        )
        f_run.addRow("Fast Allrun (skip optional verification)", chk_fast)
        f_run.addRow(
            "",
            QLabel(
                "ON = ~5-8 s saved per run (no stage_check tee, no checkMesh).\n"
                "OFF = full stage logs in log.stageVerification (debugging mode)."
            ),
        )
        layout.addWidget(grp_run)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        layout.addWidget(bb)
        if d.exec_() == QDialog.Accepted:
            self._enable_post_processing = chk_post_proc.isChecked()
            self._fast_run_mode = chk_fast.isChecked()

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
        """Compute and display geometry for shapes driven by mass/density."""
        shape = self.c_shape.currentText()
        if shape not in ("Sphere", "Cuboid"):
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
        self.c_radius.blockSignals(True)
        self.c_radius.setValue(r)
        self.c_radius.blockSignals(False)
        self._update_preview()

    def _update_cylinder_derived_geometry(self):
        """Cylinder Radius/Length are derived from mass, density and L/D (read-only)."""
        if self.c_shape.currentText() != "Cylinder":
            return
        try:
            from physical_charge_geometry import physical_charge_geometry
            from types import SimpleNamespace

            geom = physical_charge_geometry(
                SimpleNamespace(
                    charge_shape="Cylinder",
                    mass_kg=float(self.c_mass.value()),
                    rho_charge=float(self.c_rho.value()),
                    charge_aspect=float(self.c_aspect.value()),
                )
            )
        except ValueError:
            return
        self.c_radius.blockSignals(True)
        self.c_length.blockSignals(True)
        self.c_radius.setValue(float(geom.cylinder_radius_m))
        self.c_length.setValue(float(geom.length_m))
        self.c_radius.blockSignals(False)
        self.c_length.blockSignals(False)
        self._update_preview()

    def _update_cylinder_height(self):
        """Compatibility alias — Cylinder length is mass/ρ/L/D-derived."""
        self._update_cylinder_derived_geometry()

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
        """Enable/disable geometry fields based on selected charge shape."""
        is_sphere = (shape_name == "Sphere")
        is_cylinder = (shape_name == "Cylinder")
        is_cuboid = (shape_name == "Cuboid")

        # Sphere/Cuboid: computed radius/side shown as read-only.
        if is_sphere or is_cuboid:
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
        elif is_cylinder:
            self.lbl_radius.setText("Radius [m]")
            # Derived from mass/ρ/L/D for cylindericalMassToCell — not editable.
            self.c_radius.setReadOnly(True)
            self.c_radius.setToolTip(
                "Derived from mass, density and L/D for cylindericalMassToCell."
            )
            self.c_length.setToolTip(
                "Derived from mass, density and L/D for cylindericalMassToCell."
            )
            for w in (self.c_radius, self.lbl_radius):
                w.setEnabled(False)  # gray display-only, same as Sphere radius
            self._update_cylinder_derived_geometry()
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
        # Length/Width/Height visibility and behavior by shape.
        if is_cuboid:
            self.lbl_length.setText("Length [m]")
            for w in (self.c_length, self.lbl_length, self.c_width, self.lbl_width, self.c_height, self.lbl_height):
                w.setEnabled(True)
                w.setVisible(True)
            self.c_height.setReadOnly(True)
            self._update_cuboid_height()
        elif is_cylinder:
            self.lbl_length.setText("Length [m]")
            for w in (self.c_length, self.lbl_length):
                w.setEnabled(False)
                w.setVisible(True)
            self.c_length.setReadOnly(True)
            for w in (self.c_width, self.lbl_width, self.c_height, self.lbl_height):
                w.setVisible(False)
            self._update_cylinder_derived_geometry()
        else:
            for w in (self.c_length, self.lbl_length, self.c_width, self.lbl_width, self.c_height, self.lbl_height):
                w.setEnabled(False)
                w.setVisible(False)
            self.c_height.setReadOnly(True)

        self._update_preview()

    def _on_material_changed(self, mat_name):
        if mat_name in self.materials_db:
            props = self.materials_db[mat_name]
            self.c_rho.setValue(props["rho"])
        self.c_rho.setEnabled(False)
        self.lbl_density.setEnabled(False)
        self._update_edit_button_visibility()
        self._update_preview()

    def _update_preview(self):
        self._clear_charge_cells_display()
        self._update_mesh_plan_display()

        if not self.viewer: return
        bounds = (self.sx1.value(), self.sx2.value(), self.sy1.value(), self.sy2.value(), self.sz1.value(), self.sz2.value())
        charge = (self.cx.value(), self.cy.value(), self.cz.value(), self.c_shape.currentText(), self.c_mass.value(), self.c_rho.value())
        self.viewer.update_preview(bounds, charge, self.obstacles)

    def _clear_charge_cells_display(self):
        """Hide Initialization Results until real post-init metadata exists."""
        self.lbl_result_total_cells.setText("")
        self.lbl_result_init_command.setText("")
        self.lbl_result_charge_cells.setText("")
        self.lbl_result_ignition_cells.setText("")
        self.lbl_result_block.setText("")
        self.lbl_charge_fraction.setText("")
        self.lbl_cells_inside_charge.setText("")
        self.lbl_charge_clipped.setText("")
        self.lbl_initiation_radius.setText("")
        self.lbl_charge_refine_info.setText("")
        self.lbl_obstacle_refine_info.setText("")
        self.lbl_smallest_cell.setText("")
        self.lbl_expected_emesh.setText("")
        self._set_init_results_visible(False)

    def _update_info_from_case_init_mode(self, case_dir: str) -> None:
        """Read case_init_mode.json and populate Initialization Results (+ warnings)."""
        import json
        path = os.path.join(case_dir, "case_init_mode.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                mode = json.load(f)
        except (OSError, ValueError):
            return

        any_result = False
        set_cmd = mode.get("set_cmd")
        if set_cmd:
            cmd_show = "remap" if str(set_cmd).startswith("remap") else str(set_cmd)
            self.lbl_result_init_command.setText(f"Init command: {cmd_show}")
            any_result = True
            init_tip_lines = []
            fallback = mode.get("fallback_reason")
            if fallback:
                init_tip_lines.append(f"Fallback: {fallback}")
            da = mode.get("domain_alignment") or {}
            if da.get("requested_lengths_m") and da.get("actual_lengths_m"):
                init_tip_lines.append(
                    f"Domain lengths requested {da.get('requested_lengths_m')} m → actual {da.get('actual_lengths_m')} m "
                    f"(cell {da.get('requested_cell_size_m')} m, n_cells {da.get('n_cells_xyz')})"
                )
            tr = mode.get("transition_region") or {}
            if tr.get("outside_extent_m") is not None:
                init_tip_lines.append(
                    f"Transition shell: {tr.get('snappy_type') or '—'}, outside_extent≈{tr.get('outside_extent_m'):.4g} m "
                    f"({'auto' if tr.get('outside_extent_auto') else 'user'})"
                )
            amr_w = mode.get("amr_written")
            if isinstance(amr_w, dict) and amr_w.get("errorEstimator_line"):
                init_tip_lines.append(
                    f"AMR: {amr_w.get('errorEstimator_line')} maxRefinement={amr_w.get('maxRefinement')} "
                    f"refineInterval={amr_w.get('refineInterval')} maxCells={amr_w.get('maxCells')}"
                )
            bc = mode.get("base_cell_count")
            if bc is not None:
                init_tip_lines.append(f"Base mesh cell count (blockMesh): {int(bc):,}")
                self.lbl_result_total_cells.setText(f"Total cells: {int(bc):,}")
            tip = "\n".join(init_tip_lines) if init_tip_lines else ""
            self.lbl_result_init_command.setToolTip(tip)
            self.lbl_result_block.setToolTip(tip)

        # Keep detailed refine/capture notes in tooltips / Mesh Plan, not permanent result rows
        cr_req = mode.get("charge_refinement_requested")
        cr_eff = mode.get("charge_refinement_effective")
        if cr_req is not None and cr_eff is not None:
            self.lbl_charge_refine_info.setToolTip(f"Charge refine: req {cr_req}, eff {cr_eff}")

        cap = mode.get("charge_capture") or {}
        if cap:
            tip_parts = []
            desc = (cap.get("formula_description") or "").strip()
            if desc:
                tip_parts.append(desc)
            r_used = cap.get("charge_capture_radius_used_m")
            if r_used is not None:
                tip_parts.append(f"R_cap={float(r_used):.4g} m")
            tip_parts.extend(cap.get("warnings") or [])
            if tip_parts:
                self.lbl_plan_charge_capture.setToolTip("\n".join(tip_parts))
                self.lbl_plan_block_seed.setToolTip("\n".join(tip_parts))

        cells_inside = mode.get("cells_inside_charge")
        if cells_inside is not None:
            self.lbl_result_charge_cells.setText(f"Charge cells (alpha.c4): {int(cells_inside):,}")
            any_result = True

        clipped = mode.get("charge_clipped_by_domain")
        warnings = list(mode.get("charge_warnings") or [])
        clipped_true = clipped is True or (isinstance(clipped, str) and clipped.strip().lower() in ("yes", "true", "1"))
        warn_parts = []
        if clipped_true:
            warn_parts.append("Warning: Charge clipped by domain.")
        cap_impossible = mode.get("charge_capture_impossible_message")
        if cap_impossible:
            warn_parts.append("Init will fail: charge capture impossible. Reduce Cell Size or enlarge charge.")
            self.lbl_charge_resolution_warning.setToolTip(cap_impossible)
        warn_parts.extend(warnings)
        if warn_parts:
            self.lbl_charge_resolution_warning.setText("\n".join(warn_parts))
            self.lbl_charge_resolution_warning.setWordWrap(True)

        ign_cells = mode.get("cells_in_ignition_region")
        if ign_cells is not None:
            self.lbl_result_ignition_cells.setText(f"Cells in ignition region: {int(ign_cells):,}")
            any_result = True

        # Debug/preflight detail kept off the permanent panel (tooltip only if present)
        emesh_list = mode.get("expected_eMesh") or []
        if emesh_list:
            self.lbl_expected_emesh.setToolTip(
                "Expected .eMesh (preflight):\n" + "\n".join(emesh_list[:10])
                + ("\n..." if len(emesh_list) > 10 else "")
            )

        if any_result:
            self._set_init_results_visible(True)

    def update_charge_cells_display(self, case_dir: str, threshold: float = 0.5) -> None:
        """Update Initialization Results from 0/alpha.c4 and case_init_mode.json (after init)."""
        self._update_info_from_case_init_mode(case_dir)
        try:
            from verification.verify_output import get_charge_cell_count
        except ImportError:
            return
        charge, total = get_charge_cell_count(case_dir, time_dir="0", threshold=threshold)
        if charge is not None:
            self.lbl_result_charge_cells.setText(f"Charge cells (alpha.c4>{threshold}): {charge:,}")
            self._set_init_results_visible(True)
        if total is not None and total > 0:
            self.lbl_result_total_cells.setText(f"Total cells: {total:,}")
            self._set_init_results_visible(True)

    def _on_init_clicked(self):
        self.sig_request_init.emit(self.get_case_inputs())

    def _build_initialize_write_plan(self) -> list[str]:
        """Build explicit initialize write/command preview for user confirmation."""
        plan = []
        remap = self.rad_init_remap.isChecked()
        has_obstacles = any(obs.enabled for obs in self.obstacles)
        shape = self.c_shape.currentText()
        dyn_refine = self.rad_dyn_mesh.isChecked() and self.rad_dyn_mesh.isEnabled()
        seed_mode = self.combo_charge_seed_mode.currentText()
        charge_refine = self.spin_charge_refine.value()
        outer_level = self.spin_charge_outer_level.value()
        outer_on = dyn_refine and self.chk_charge_outer_enable.isChecked()

        plan.append("Create/refresh case dictionaries under system/, constant/, and 0.orig/.")
        plan.append(f"Mesh mode: {'Dynamic/Hybrid' if dyn_refine else 'Fixed/Static'}")
        plan.append(f"Charge geometry: {shape}")
        plan.append(f"Charge seed mode: {seed_mode}")
        plan.append(f"Charge seed target cells: {self.spin_charge_seed_target.value()}")
        plan.append(f"Charge refine level (manual storage): {charge_refine}")
        plan.append(
            f"Startup outer region: {'On (level ' + str(outer_level) + ')' if outer_on else 'Off'}"
        )
        plan.append(f"Obstacle meshing: {'enabled' if has_obstacles else 'disabled'}")
        if remap:
            plan.append("Initialization source: remap from pre-cursor case.")
            plan.append(f"Remap case path: {(self._remap_case_path or '').strip() or '(empty)'}")
            if self._remap_time_mode == "latest":
                plan.append("Remap time: latest solved time.")
            else:
                plan.append(f"Remap time: specific ({(self._remap_specific_time or '').strip() or '(empty)'})")
            plan.append("Initialize commands: blockMesh -> optional surfaceFeatures/snappy -> remap_radial.py.")
        else:
            init_plan = build_initialization_plan(self.get_case_inputs())
            plan.append(f"Initialization command: {init_plan.command}")
            plan.append(f"Initialization reason: {init_plan.reason}")
            if has_obstacles:
                plan.append("Initialize commands: blockMesh -> surfaceFeatures -> snappyHexMesh -> setFields workflow.")
            else:
                plan.append("Initialize commands: blockMesh -> setFields workflow.")
        plan.append("Post-checks: charge-cell verification (alpha.c4) and init summary.")
        return plan

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

        self.sections = new_secs
        if self.viewer:
            self.viewer.update_sections(new_secs)

    def load_project_gui_state(self, state: dict) -> None:
        """Restore GUI-only section definitions from a project JSON."""
        sections = state.get("sections", []) if isinstance(state, dict) else []
        if not isinstance(sections, list):
            raise ValueError("gui_state.sections must be a list")
        self.tbl_sec.setRowCount(0)
        for item in sections:
            if not isinstance(item, dict):
                raise ValueError("Each saved section must be an object")
            self._add_default_section()
            row = self.tbl_sec.rowCount() - 1
            self.tbl_sec.item(row, 0).setCheckState(
                Qt.Checked if bool(item.get("enabled", True)) else Qt.Unchecked
            )
            self.tbl_sec.item(row, 1).setText(str(item.get("name", "Section")))
            normal = item.get("normal", [0, 0, 1])
            plane = "YZ (Side)" if normal == [1, 0, 0] else "XZ (Front)" if normal == [0, 1, 0] else "XY (Ground)"
            self.tbl_sec.cellWidget(row, 2).setCurrentText(plane)
            self.tbl_sec.cellWidget(row, 3).setValue(float(item.get("position_m", 0.0)))
            self.tbl_sec.cellWidget(row, 4).setValue(float(item.get("opacity", 0.5)))
        self._update_sections_from_table(-1, -1)

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
        except (IndexError, AttributeError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid obstacle value", str(exc))

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
            self.spin_refine_max.setValue(1)
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
            self._outside_extent = None
        elif key == "transition_cells":
            self.spin_transition_cells.setValue(2)
        elif key == "refine_interval":
            self._refine_interval = 3
        elif key == "bubble_radius_factor":
            self._bubble_radius_factor = 1.5
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
        elif key == "dynamic_max_cells":
            self._dynamic_max_cells = 200000000
        elif key == "begin_unrefine":
            self._begin_unrefine = None
        elif key == "upper_refine_level":
            self._upper_refine_level = None
        elif key == "upper_unrefine_level":
            self._upper_unrefine_level = None
        elif key == "balance_interval":
            self._balance_interval = None
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
            self.spin_charge_refine.setValue(0)
        elif key == "charge_seed_mode":
            self.combo_charge_seed_mode.setCurrentText(SEED_MODE_AUTO)
        elif key == "charge_seed_target_cells":
            self.spin_charge_seed_target.setValue(8)
        elif key == "charge_outer_refine_level":
            self.spin_charge_outer_level.setValue(3)
            self._sync_charge_outer_level_mirrors()
        elif key == "charge_outer_refine_min":
            self.spin_charge_outer_level.setValue(max(self.spin_charge_outer_level.value(), 0))
            self._sync_charge_outer_level_mirrors()
        elif key == "charge_outer_refine_max":
            self.spin_charge_outer_level.setValue(3)
            self._sync_charge_outer_level_mirrors()
        elif key == "charge_outer_refine_enable":
            self.chk_charge_outer_enable.setChecked(False)
            self._sync_charge_outer_level_mirrors()
        elif key == "cylinder_axis":
            idx = self.c_cylinder_axis.findText("Z")
            if idx >= 0:
                self.c_cylinder_axis.setCurrentIndex(idx)
        elif key == "charge_backup_radius_factor":
            self._bubble_radius_factor = 1.5
            self._charge_capture_mode = "auto"
            self._charge_capture_factor = 1.0
            self._charge_backup_radius_override = None
        elif key == "charge_capture_mode":
            self._charge_capture_mode = "auto"
        elif key == "charge_capture_factor":
            self._charge_capture_factor = 1.0
        elif key == "charge_capture_radius":
            self._charge_capture_radius_manual = 0.2
            self._charge_backup_radius_override = None
        elif key == "buffer_layers":
            self._buffer_layers = 5
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
        if key == "transition_cells":
            self.spin_transition_cells.setEnabled(True)
        elif key == "enable_dyn_refine":
            self.rad_dyn_mesh.setEnabled(True)
            self.rad_fixed_mesh.setEnabled(True)
        elif key == "enable_obstacle_refine":
            self.chk_obstacle_refine.setEnabled(True)

    def _apply_unset_for_key(self, key: str) -> None:
        """Show optional field as UNSET: disable and/or set sentinel so generation does not override loaded case."""
        if key == "outside_extent":
            self._outside_extent = None
        elif key == "transition_cells":
            self.spin_transition_cells.setValue(2)
            self.spin_transition_cells.setEnabled(False)
        elif key == "enable_dyn_refine":
            self.rad_dyn_mesh.setEnabled(False)
            self.rad_fixed_mesh.setEnabled(False)
        elif key == "enable_obstacle_refine":
            self.chk_obstacle_refine.setEnabled(False)
        elif key == "dynamic_max_cells":
            self._dynamic_max_cells = 200000000
        elif key == "begin_unrefine":
            self._begin_unrefine = None
        elif key == "upper_refine_level":
            self._upper_refine_level = None
        elif key == "upper_unrefine_level":
            self._upper_unrefine_level = None
        elif key == "balance_interval":
            self._balance_interval = None
        else:
            # eos_model, activation_model_ui, thermo_model, decomposition_*, mesh_*: set internal to None
            attr = "_" + key
            if hasattr(self, attr):
                setattr(self, attr, None)

    def set_case_inputs(self, data: dict, load_summary: dict = None) -> None:
        """Populate GUI from *data*. If *load_summary*: LOADED keys set from case; not_filled left UNSET (no default).

        Without *load_summary* (GGUI project apply), provenance is reset so stale
        OpenFOAM case-loader UNSET state cannot override authoritative project values.
        """
        if load_summary:
            self._provenance.update(data.get("_provenance", {}))
            not_filled = load_summary.get("not_filled", [])
            for key, _reason in not_filled:
                self._provenance[key] = "UNSET"
                self._apply_unset_for_key(key)
        else:
            # Project / direct apply: drop stale case-loader provenance, then restore
            # intentional LOADED/USER keys from the project CaseInputs3D.provenance.
            self._provenance = {}
            proj_prov = data.get("provenance") or data.get("_provenance") or {}
            if isinstance(proj_prov, dict):
                self._provenance.update(
                    {
                        k: v
                        for k, v in proj_prov.items()
                        if v in ("LOADED", "USER")
                    }
                )
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
                # Custom: canonical CaseInputs3D.material_props; legacy custom_material_props
                if mat == "Custom":
                    cprops = data.get("material_props")
                    if not isinstance(cprops, dict) or not cprops:
                        cprops = data.get("custom_material_props")
                    if isinstance(cprops, dict):
                        for k in ("rho", "energy", "A", "B", "R1", "R2", "omega"):
                            if k in cprops and cprops[k] is not None:
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
            if "refine_min" in data and data["refine_min"] is not None:
                self.spin_refine_min.setValue(int(data["refine_min"]))
            if "dyn_refine_min" in data and data["dyn_refine_min"] is not None:
                self.spin_refine_min.setValue(int(data["dyn_refine_min"]))
            if "refine_max" in data and data["refine_max"] is not None:
                self.spin_refine_max.setValue(data["refine_max"])
            if "dyn_refine_max" in data and data["dyn_refine_max"] is not None:
                self._dyn_refine_max = int(data["dyn_refine_max"])
                self.spin_refine_max.setValue(self._dyn_refine_max)
            elif "refine_max" in data and data["refine_max"] is not None:
                self._dyn_refine_max = int(data["refine_max"])
            if "enable_local_refinement" in data and data["enable_local_refinement"] is not None:
                en = bool(data["enable_local_refinement"])
                self.rad_dyn_mesh.setChecked(en)
                self.rad_fixed_mesh.setChecked(not en)
            if "enable_dyn_refine" in data and data["enable_dyn_refine"] is not None:
                en = bool(data["enable_dyn_refine"])
                self.rad_dyn_mesh.setChecked(en)
                self.rad_fixed_mesh.setChecked(not en)
                self.rad_dyn_mesh.setEnabled(True)
                self.rad_fixed_mesh.setEnabled(True)
            if "enable_obstacle_refine" in data and data["enable_obstacle_refine"] is not None:
                self.chk_obstacle_refine.setChecked(bool(data["enable_obstacle_refine"]))
                self.chk_obstacle_refine.setEnabled(True)
            if "obstacle_refine_min" in data and data["obstacle_refine_min"] is not None:
                self.spin_obstacle_refine_min.setValue(int(data["obstacle_refine_min"]))
            if "obstacle_refine_max" in data and data["obstacle_refine_max"] is not None:
                self.spin_obstacle_refine_max.setValue(int(data["obstacle_refine_max"]))
            if "transition_cells" in data and data["transition_cells"] is not None:
                self.spin_transition_cells.setValue(max(1, min(10, int(data["transition_cells"]))))
                self.spin_transition_cells.setEnabled(True)
            if "match_outer_to_seed" in data:
                self.chk_match_outer_to_seed.setChecked(bool(data["match_outer_to_seed"]))
            if "outside_extent" in data and data["outside_extent"] is not None:
                try:
                    oe = float(data["outside_extent"])
                    self._outside_extent = oe if oe > 0 else None
                except (TypeError, ValueError):
                    self._outside_extent = None
            if "dynamic_max_cells" in data and data["dynamic_max_cells"] is not None:
                try:
                    self._dynamic_max_cells = max(1, int(data["dynamic_max_cells"]))
                except (TypeError, ValueError):
                    pass
            if "begin_unrefine" in data and data["begin_unrefine"] is not None:
                try:
                    self._begin_unrefine = float(data["begin_unrefine"])
                except (TypeError, ValueError):
                    self._begin_unrefine = None
            if "upper_refine_level" in data and data["upper_refine_level"] is not None:
                try:
                    self._upper_refine_level = float(data["upper_refine_level"])
                except (TypeError, ValueError):
                    self._upper_refine_level = None
            if "upper_unrefine_level" in data and data["upper_unrefine_level"] is not None:
                try:
                    self._upper_unrefine_level = float(data["upper_unrefine_level"])
                except (TypeError, ValueError):
                    self._upper_unrefine_level = None
            if "balance_interval" in data and data["balance_interval"] is not None:
                try:
                    self._balance_interval = max(1, int(data["balance_interval"]))
                except (TypeError, ValueError):
                    self._balance_interval = None
            if "refine_indicator_field" in data:
                ri = str(data["refine_indicator_field"]) or "densityGradient"
                if ri.strip().lower() in ("pressuregradient", "pressure"):
                    ri = "scaledDelta_p"
                self._refine_indicator_field = ri
            if "refine_interval" in data and data["refine_interval"] is not None:
                try:
                    self._refine_interval = max(1, int(data["refine_interval"]))
                except (TypeError, ValueError):
                    self._refine_interval = 3
            if "lower_refine_threshold" in data and data["lower_refine_threshold"] is not None:
                try:
                    self._lower_refine_threshold = float(data["lower_refine_threshold"])
                except (TypeError, ValueError):
                    self._lower_refine_threshold = 0.1
            if "unrefine_threshold" in data and data["unrefine_threshold"] is not None:
                try:
                    self._unrefine_threshold = float(data["unrefine_threshold"])
                except (TypeError, ValueError):
                    self._unrefine_threshold = 0.1
            if "n_buffer_layers_dynamic" in data and data["n_buffer_layers_dynamic"] is not None:
                try:
                    self._n_buffer_layers_dynamic = max(0, int(data["n_buffer_layers_dynamic"]))
                except (TypeError, ValueError):
                    self._n_buffer_layers_dynamic = 2
            if "enable_balancing" in data and data["enable_balancing"] is not None:
                self._enable_balancing = bool(data["enable_balancing"])
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
            mesh_int_keys = {
                "mesh_included_angle",
                "mesh_n_smooth_patch",
                "mesh_n_solve_iter",
                "mesh_n_relax_iter",
                "mesh_n_feature_snap_iter",
                "mesh_n_cells_between_levels",
                "mesh_resolve_feature_angle",
                "mesh_n_smooth_scale",
            }
            mesh_bool_keys = {
                "mesh_explicit_feature_snap",
                "mesh_implicit_feature_snap",
                "mesh_multi_region_feature_snap",
            }
            mesh_attr_map = {
                "mesh_included_angle": "_mesh_included_angle",
                "mesh_n_smooth_patch": "_mesh_n_smooth_patch",
                "mesh_snap_tolerance": "_mesh_snap_tolerance",
                "mesh_n_solve_iter": "_mesh_n_solve_iter",
                "mesh_n_relax_iter": "_mesh_n_relax_iter",
                "mesh_n_feature_snap_iter": "_mesh_n_feature_snap_iter",
                "mesh_explicit_feature_snap": "_mesh_explicit_feature_snap",
                "mesh_implicit_feature_snap": "_mesh_implicit_feature_snap",
                "mesh_multi_region_feature_snap": "_mesh_multi_region_feature_snap",
                "mesh_n_cells_between_levels": "_mesh_n_cells_between_levels",
                "mesh_resolve_feature_angle": "_mesh_resolve_feature_angle",
                "mesh_max_non_ortho": "_mesh_max_non_ortho",
                "mesh_max_boundary_skewness": "_mesh_max_boundary_skewness",
                "mesh_max_internal_skewness": "_mesh_max_internal_skewness",
                "mesh_max_concave": "_mesh_max_concave",
                "mesh_min_vol": "_mesh_min_vol",
                "mesh_min_tet_quality": "_mesh_min_tet_quality",
                "mesh_min_twist": "_mesh_min_twist",
                "mesh_min_determinant": "_mesh_min_determinant",
                "mesh_min_face_weight": "_mesh_min_face_weight",
                "mesh_min_vol_ratio": "_mesh_min_vol_ratio",
                "mesh_n_smooth_scale": "_mesh_n_smooth_scale",
                "mesh_error_reduction": "_mesh_error_reduction",
                "mesh_relaxed_max_non_ortho": "_mesh_relaxed_max_non_ortho",
            }
            for mk, attr in mesh_attr_map.items():
                if mk not in data or data[mk] is None:
                    continue
                raw = data[mk]
                try:
                    if mk in mesh_bool_keys:
                        if isinstance(raw, str):
                            val = raw.strip().lower() in ("1", "true", "yes", "on")
                        else:
                            val = bool(raw)
                    elif mk in mesh_int_keys:
                        val = int(float(raw))
                    else:
                        val = float(raw)
                except (TypeError, ValueError):
                    # Ignore malformed values from imported cases and keep current defaults.
                    continue
                setattr(self, attr, val)
            if "charge_seed_mode" in data and data["charge_seed_mode"] not in (None, ""):
                mode_txt = str(data["charge_seed_mode"]).strip()
                idx = self.combo_charge_seed_mode.findText(mode_txt)
                if idx < 0:
                    low = mode_txt.lower()
                    if low == "auto":
                        idx = self.combo_charge_seed_mode.findText(SEED_MODE_AUTO)
                    elif low in ("manual", "man"):
                        idx = self.combo_charge_seed_mode.findText(SEED_MODE_MANUAL)
                    elif low in ("off", "none", "disabled", "0"):
                        idx = self.combo_charge_seed_mode.findText(SEED_MODE_OFF)
                if idx >= 0:
                    self.combo_charge_seed_mode.setCurrentIndex(idx)
            elif "charge_refinement_level" in data and data["charge_refinement_level"] is not None:
                # Legacy projects without explicit mode: non-zero level → Manual, else Off.
                lvl_legacy = int(data["charge_refinement_level"])
                self.combo_charge_seed_mode.setCurrentText(
                    SEED_MODE_MANUAL if lvl_legacy > 0 else SEED_MODE_OFF
                )
            if "charge_seed_target_cells" in data and data["charge_seed_target_cells"] is not None:
                try:
                    self.spin_charge_seed_target.setValue(
                        max(1, min(20, int(data["charge_seed_target_cells"])))
                    )
                except (TypeError, ValueError):
                    pass
            if "charge_seed_min_cells" in data and data["charge_seed_min_cells"] is not None:
                try:
                    self._charge_seed_min_cells = max(1, int(data["charge_seed_min_cells"]))
                except (TypeError, ValueError):
                    pass
            if "charge_seed_max_level" in data and data["charge_seed_max_level"] is not None:
                try:
                    self._charge_seed_max_level = max(0, int(data["charge_seed_max_level"]))
                except (TypeError, ValueError):
                    pass
            if "charge_outer_legacy_migration_warning" in data:
                warn = data.get("charge_outer_legacy_migration_warning")
                self._charge_outer_legacy_migration_warning = (
                    str(warn) if warn else None
                )
            if "charge_refinement_level" in data and data["charge_refinement_level"] is not None:
                self.spin_charge_refine.setValue(int(data["charge_refinement_level"]))
            # Outer: prefer charge_outer_refine_level; migrate legacy min/max → level = max(min, max).
            outer_level_val = None
            if data.get("charge_outer_refine_level") is not None:
                try:
                    outer_level_val = int(data["charge_outer_refine_level"])
                except (TypeError, ValueError):
                    outer_level_val = None
            elif data.get("charge_outer_refine_min") is not None or data.get("charge_outer_refine_max") is not None:
                try:
                    a = int(data["charge_outer_refine_min"]) if data.get("charge_outer_refine_min") is not None else 0
                    b = int(data["charge_outer_refine_max"]) if data.get("charge_outer_refine_max") is not None else a
                    outer_level_val = max(a, b)
                except (TypeError, ValueError):
                    outer_level_val = None
            if outer_level_val is not None:
                self.spin_charge_outer_level.setValue(max(0, outer_level_val))
                self._sync_charge_outer_level_mirrors()
            if "charge_outer_refine_enable" in data:
                self.chk_charge_outer_enable.setChecked(bool(data["charge_outer_refine_enable"]))
            elif any(
                k in data and data[k] is not None
                for k in ("charge_outer_refine_min", "charge_outer_refine_max", "charge_outer_refine_level")
            ):
                # Legacy: presence of outer levels implied enabled (unless explicitly False above).
                self.chk_charge_outer_enable.setChecked(True)
            # Lossless imported outer mode / geometry / distance pairs (hidden state).
            if "charge_outer_mode" in data:
                mode = data.get("charge_outer_mode")
                self._charge_outer_mode = str(mode).strip().lower() if mode else None
            if "charge_outer_distance_levels" in data:
                pairs = data.get("charge_outer_distance_levels")
                if isinstance(pairs, list):
                    norm = []
                    for item in pairs:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            norm.append((float(item[0]), int(item[1])))
                    self._charge_outer_distance_levels = norm or None
                else:
                    self._charge_outer_distance_levels = None
            if "charge_outer_geometry" in data:
                geom = data.get("charge_outer_geometry")
                self._charge_outer_geometry = dict(geom) if isinstance(geom, dict) else None
            if "charge_outer_raw_refinement" in data:
                raw = data.get("charge_outer_raw_refinement")
                self._charge_outer_raw_refinement = str(raw) if raw else None
            if "cylinder_axis" in data:
                idx = self.c_cylinder_axis.findText(data["cylinder_axis"])
                if idx >= 0:
                    self.c_cylinder_axis.setCurrentIndex(idx)
            if "charge_capture_factor" in data and data["charge_capture_factor"] is not None:
                try:
                    self._charge_capture_factor = float(data["charge_capture_factor"])
                except (TypeError, ValueError):
                    pass
            if "charge_capture_radius" in data and data["charge_capture_radius"] is not None:
                try:
                    self._charge_capture_radius_manual = float(data["charge_capture_radius"])
                except (TypeError, ValueError):
                    pass
            if "charge_capture_mode" in data:
                cm = str(data["charge_capture_mode"] or "auto").lower()
                self._charge_capture_mode = cm if cm in ("auto", "manual") else "auto"
            if "charge_backup_radius_factor" in data:
                # Legacy: was tied to capture; now only updates snappy/topoSet transition seed.
                self._bubble_radius_factor = float(data["charge_backup_radius_factor"])
            if "charge_backup_radius_override" in data and data["charge_backup_radius_override"] is not None:
                try:
                    o = float(data["charge_backup_radius_override"])
                    self._charge_backup_radius_override = o
                    if "charge_capture_mode" not in data:
                        self._charge_capture_mode = "manual"
                    if "charge_capture_radius" not in data:
                        self._charge_capture_radius_manual = o
                except (TypeError, ValueError):
                    self._charge_backup_radius_override = None
            if "bubble_radius_factor" in data:
                self._bubble_radius_factor = float(data["bubble_radius_factor"])
            if "ignition_radius" in data and data["ignition_radius"] is not None:
                try:
                    self._ignition_radius = float(data["ignition_radius"])
                    self._ignition_radius_manual = True
                except (TypeError, ValueError):
                    self._ignition_radius = None
                    self._ignition_radius_manual = False
            if "delta_t" in data and data["delta_t"] is not None:
                try:
                    self._delta_t_loaded = float(data["delta_t"])
                except (TypeError, ValueError):
                    self._delta_t_loaded = None
            if "charge_backup_length_override" in data and data["charge_backup_length_override"] is not None:
                try:
                    self._charge_backup_length_override = float(data["charge_backup_length_override"])
                except (TypeError, ValueError):
                    self._charge_backup_length_override = None
            if "buffer_layers" in data:
                self._buffer_layers = int(data["buffer_layers"])
            # Run-mode tradeoffs: load if loader detected them, otherwise leave fast defaults.
            if "enable_post_processing" in data:
                self._enable_post_processing = bool(data["enable_post_processing"])
            if "fast_run_mode" in data:
                self._fast_run_mode = bool(data["fast_run_mode"])
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
        # Recompute geometry from mass/density only for Sphere.
        # Cylinder radius stays user-/file-driven; height is derived from radius and L/D.
        if "mass_kg" in data and "rho_charge" in data and data.get("charge_shape") == "Sphere":
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
        est_cells = self._estimate_charge_cells()
        enable_dyn = None if prov.get("enable_dyn_refine") == "UNSET" else refine
        enable_obs = None if prov.get("enable_obstacle_refine") == "UNSET" else obs_refine
        ignition_mode = self.combo_ignition_mode.currentText()
        initiation_point = None if ignition_mode == "Center of Charge" else (self.init_ix.value(), self.init_iy.value(), self.init_iz.value())
        _ccf_raw = getattr(self, "_charge_capture_factor", None)
        charge_capture_factor_out = float(1.0 if _ccf_raw is None else _ccf_raw)
        _cc_mode_l = str(getattr(self, "_charge_capture_mode", "auto") or "auto").lower()
        _mrad_raw = getattr(self, "_charge_capture_radius_manual", None)
        charge_capture_radius_out = (
            float(_mrad_raw) if _cc_mode_l == "manual" and _mrad_raw is not None else None
        )
        seed_mode = self.combo_charge_seed_mode.currentText()
        outer_level = int(self.spin_charge_outer_level.value())
        # Fixed Mesh: outer enable forced False; seed mode left as-is (init plan forces seed off).
        outer_enable = bool(self.chk_charge_outer_enable.isChecked()) if refine else False
        inputs = CaseInputs3D(
            min_point=(self.sx1.value(), self.sy1.value(), self.sz1.value()),
            max_point=(self.sx2.value(), self.sy2.value(), self.sz2.value()),
            cell_size=self.scell.value(),
            charge_center=(self.cx.value(), self.cy.value(), self.cz.value()),
            initiation_point=initiation_point,
            ignition_mode=ignition_mode,
            buffer_layers=getattr(self, "_buffer_layers", 5),
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
            # Mode is authoritative; still pass manual spin for Manual (and for Auto/Off storage).
            charge_refinement_level=int(self.spin_charge_refine.value()),
            charge_seed_mode=seed_mode,
            charge_seed_target_cells=int(self.spin_charge_seed_target.value()),
            charge_seed_min_cells=int(getattr(self, "_charge_seed_min_cells", 6)),
            charge_seed_max_level=int(getattr(self, "_charge_seed_max_level", 5)),
            charge_outer_refine_enable=outer_enable,
            charge_outer_refine_level=outer_level,
            charge_outer_refine_min=outer_level,
            charge_outer_refine_max=outer_level,
            charge_outer_mode=getattr(self, "_charge_outer_mode", None),
            charge_outer_distance_levels=(
                list(self._charge_outer_distance_levels)
                if getattr(self, "_charge_outer_distance_levels", None)
                else None
            ),
            charge_outer_geometry=(
                dict(self._charge_outer_geometry)
                if isinstance(getattr(self, "_charge_outer_geometry", None), dict)
                else None
            ),
            charge_outer_raw_refinement=getattr(self, "_charge_outer_raw_refinement", None),
            charge_outer_legacy_migration_warning=getattr(
                self, "_charge_outer_legacy_migration_warning", None
            ),
            transition_cells=transition_cells,
            use_seed_bubble=(refine and seed_mode != SEED_MODE_OFF),
            charge_capture_mode=str(getattr(self, "_charge_capture_mode", "auto") or "auto"),
            charge_capture_factor=charge_capture_factor_out,
            charge_capture_radius=charge_capture_radius_out,
            charge_backup_radius_factor=1.0,
            charge_backup_radius_override=getattr(self, "_charge_backup_radius_override", None),
            charge_backup_length_override=getattr(self, "_charge_backup_length_override", None),
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
            bubble_radius_factor=getattr(self, "_bubble_radius_factor", 1.5),
            enable_post_processing=getattr(self, "_enable_post_processing", False),
            fast_run_mode=getattr(self, "_fast_run_mode", True),
            enable_balancing=bool(getattr(self, "_enable_balancing", False)),
            dynamic_max_cells=getattr(self, "_dynamic_max_cells", 200000000),
            outside_extent=getattr(self, "_outside_extent", None),
            begin_unrefine=getattr(self, "_begin_unrefine", None),
            upper_refine_level=getattr(self, "_upper_refine_level", None),
            upper_unrefine_level=getattr(self, "_upper_unrefine_level", None),
            balance_interval=getattr(self, "_balance_interval", None),
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
            estimated_charge_cells=est_cells,
        )
        # Cylinder: mass/ρ/L/D are authoritative for cylindericalMassToCell — sync derived r, L.
        if inputs.charge_shape == "Cylinder":
            from physical_charge_geometry import (
                physical_charge_geometry,
                sync_derived_cylinder_fields,
            )

            try:
                inputs = sync_derived_cylinder_fields(inputs)
                geom = physical_charge_geometry(inputs)
                self.c_radius.blockSignals(True)
                self.c_length.blockSignals(True)
                self.c_radius.setValue(float(geom.cylinder_radius_m))
                self.c_length.setValue(float(geom.length_m))
            except ValueError:
                pass
            finally:
                self.c_radius.blockSignals(False)
                self.c_length.blockSignals(False)
        return inputs