"""
BlastFoam GUI Manager - Refactored UI
Main application window with multi-panel layout following specification.
"""
import sys
import os
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QProgressBar, QMessageBox, QToolBar, QAction,
    QSplitter, QScrollArea, QGroupBox, QFormLayout, QStatusBar, QFileDialog,
    QDialog, QTextEdit, QDialogButtonBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QFont, QFontMetrics

from tab_1d import Tab1D
from tab_2d import Tab2D
from tab_log import LogTab
from tab_3d_general import TabGeneral3D
from tab_probes import TabProbes
from probes_model import ProbesModel
from solver_runner import (
    ExecutionIntent,
    ExecutionPreparationError,
    SolverRunner,
    build_execution_plan,
)
from simulation_service import SimulationService
from models import CaseInputs1D, CaseInputs3D
from path_utils import get_latest_time_dir, win_to_wsl_path
from case_loader import load_case
from initialization_plan import build_initialization_plan
from startup_capture_guard import UNSAFE_CAPTURE_MESSAGE, require_safe_capture
from case_init_mode import record_set_cmd_actual
from project_io import (
    PROJECT_SUFFIX,
    ProjectFormatError,
    apply_project_payload,
    capture_project_payload,
    read_project,
    write_project_atomic,
)
try:
    from verification.verify_output import get_charge_cell_count
except ImportError:
    get_charge_cell_count = None  # optional for case_init_mode.json update
# from dialogs import TimeHistoryLocationsDialog, OutputFileOptionsDialog  # Not yet implemented


class InfoPanel(QFrame):
    """Lower-left info panel showing read-only derived information"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#eef2f6; border:1px solid #c7d0da; border-radius: 4px;")
        self.setMinimumHeight(120)
        self.setMaximumHeight(180)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Model Info")
        title.setStyleSheet("font-weight: bold; font-size: 12pt; color: #2c3e50;")
        layout.addWidget(title)
        
        # Info fields
        form = QFormLayout()
        form.setSpacing(5)
        
        self.lbl_total_cells = QLabel("—")
        self.lbl_cells_per_dir = QLabel("—")
        self.lbl_remap_status = QLabel("—")
        self.lbl_license_status = QLabel("—")
        
        for lbl in [self.lbl_total_cells, self.lbl_cells_per_dir, self.lbl_remap_status, self.lbl_license_status]:
            lbl.setStyleSheet("font-weight: bold; color: #34495e;")
        
        form.addRow("Total Cells:", self.lbl_total_cells)
        form.addRow("Cells/Dir:", self.lbl_cells_per_dir)
        form.addRow("Remap:", self.lbl_remap_status)
        form.addRow("License:", self.lbl_license_status)
        
        layout.addLayout(form)
        layout.addStretch()
    
    def update_info(self, total_cells=None, cells_per_dir=None, remap_status=None, license_status=None):
        """Update info fields"""
        if total_cells is not None:
            self.lbl_total_cells.setText(f"{total_cells:,}")
        if cells_per_dir is not None:
            self.lbl_cells_per_dir.setText(cells_per_dir)
        if remap_status is not None:
            self.lbl_remap_status.setText(remap_status)
        if license_status is not None:
            self.lbl_license_status.setText(license_status)


class SegmentedStatusBar(QFrame):
    """Bottom full-width status bar: one row of 1D | 2D | 3D | ET + Ready.

    Each mode group is a single non-wrapping QLabel with width reserved for the
    representative populated string. ET is wall-clock elapsed run time.
    """

    _DASH = "—"
    _LABEL_STYLE = "color: white; background: transparent;"
    _STATUS_STYLE_TMPL = "color: {color}; font-weight: bold; background: transparent;"
    _REP_MODE = "3D: Step=12345678  Tt=1.234567e-04  Δt=1.234e-07"
    _REP_ET = "ET=12345.6 s"

    def _metrics_font(self, point_size: int = None) -> QFont:
        from ui_metrics import STATUS_METRICS_POINT_SIZE, STATUS_FONT_MIN_POINT_SIZE
        pt = STATUS_METRICS_POINT_SIZE if point_size is None else int(point_size)
        pt = max(STATUS_FONT_MIN_POINT_SIZE, pt)
        font = QFont("Consolas")
        font.setStyleHint(QFont.TypeWriter)
        font.setPointSize(pt)
        font.setWeight(QFont.Normal)
        # Tightened tracking so three reserved mode groups + ET fit at ~1685.
        font.setLetterSpacing(QFont.PercentageSpacing, 70.0)
        return font

    def _ready_font(self) -> QFont:
        from ui_metrics import STATUS_READY_POINT_SIZE, STATUS_FONT_MIN_POINT_SIZE
        font = QFont("Segoe UI")
        font.setStyleHint(QFont.SansSerif)
        font.setPointSize(max(STATUS_FONT_MIN_POINT_SIZE, STATUS_READY_POINT_SIZE))
        font.setWeight(QFont.DemiBold)
        return font

    def _make_group_label(self, text: str, font: QFont, rep: str, object_name: str) -> QLabel:
        fm = QFontMetrics(font)
        width = fm.horizontalAdvance(rep)
        lbl = QLabel(text)
        lbl.setObjectName(object_name)
        lbl.setFont(font)
        lbl.setStyleSheet(self._LABEL_STYLE)
        lbl.setWordWrap(False)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lbl.setMinimumWidth(width)
        lbl.setFixedWidth(width)
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        return lbl

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            "SegmentedStatusBar { background-color: #34495e; border-top: 2px solid #2c3e50; }"
        )

        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(4)

        metrics_font = self._metrics_font()
        self._metrics_point_size = metrics_font.pointSize()

        from ui_metrics import STATUS_REP_MODE_GROUP, STATUS_REP_ET
        self._REP_MODE = STATUS_REP_MODE_GROUP
        self._REP_ET = STATUS_REP_ET

        self._1d = {"step": None, "tt": None, "dt": None}
        self._2d = {"step": None, "tt": None, "dt": None}
        self._3d = {"step": None, "tt": None, "dt": None}
        self._et_seconds = None
        self._et_monotonic_start = None

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setFont(self._ready_font())
        self.lbl_status.setStyleSheet(self._STATUS_STYLE_TMPL.format(color="#2ecc71"))
        self.lbl_status.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_status.setWordWrap(False)

        self.lbl_1d_group = self._make_group_label(
            self._format_mode_group("1D", self._1d), metrics_font, self._REP_MODE, "status1dGroup"
        )
        self.lbl_2d_group = self._make_group_label(
            self._format_mode_group("2D", self._2d), metrics_font, self._REP_MODE, "status2dGroup"
        )
        self.lbl_3d_group = self._make_group_label(
            self._format_mode_group("3D", self._3d), metrics_font, self._REP_MODE, "status3dGroup"
        )
        self.lbl_et = self._make_group_label(
            self._format_et(), metrics_font, self._REP_ET, "statusEt"
        )

        # Compatibility aliases (same group labels — no separate Step/Tt/Δt widgets).
        self.lbl_1d_mode = self.lbl_1d_group
        self.lbl_2d_mode = self.lbl_2d_group
        self.lbl_3d_mode = self.lbl_3d_group
        self.lbl_1d_step = self.lbl_1d_group
        self.lbl_1d_tt = self.lbl_1d_group
        self.lbl_1d_dt = self.lbl_1d_group
        self.lbl_2d_step = self.lbl_2d_group
        self.lbl_2d_tt = self.lbl_2d_group
        self.lbl_2d_dt = self.lbl_2d_group
        self.lbl_3d_step = self.lbl_3d_group
        self.lbl_3d_tt = self.lbl_3d_group
        self.lbl_3d_dt = self.lbl_3d_group
        self.lbl_3d_et = self.lbl_et
        self.lbl_meta = self.lbl_et
        self.lbl_3d_initial_dt = self.lbl_et  # Initial Δt moved to 3D Simulation Control
        self.lbl_metrics_line = QLabel(self._format_metrics_line())
        self.lbl_metrics_line.setObjectName("statusMetricsLine")
        self.lbl_metrics_line.hide()
        self.lbl_metrics_modes = self.lbl_metrics_line
        self.lbl_metrics_meta = self.lbl_et

        sep_font = metrics_font
        self._sep_1d_2d = QLabel(" | ")
        self._sep_2d_3d = QLabel(" | ")
        self._sep_3d_et = QLabel(" | ")
        for sep in (self._sep_1d_2d, self._sep_2d_3d, self._sep_3d_et):
            sep.setFont(sep_font)
            sep.setStyleSheet(self._LABEL_STYLE)
            sep.setWordWrap(False)

        self._metrics_widget = QWidget()
        self._metrics_widget.setObjectName("statusMetricsInner")
        self._metrics_widget.setStyleSheet(
            "QWidget#statusMetricsInner { background-color: #34495e; }"
        )
        metrics_layout = QHBoxLayout(self._metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(0)
        for w in (
            self.lbl_1d_group, self._sep_1d_2d,
            self.lbl_2d_group, self._sep_2d_3d,
            self.lbl_3d_group, self._sep_3d_et,
            self.lbl_et,
        ):
            metrics_layout.addWidget(w, stretch=0)

        self._metrics_scroll = QScrollArea()
        self._metrics_scroll.setObjectName("statusMetricsScroll")
        self._metrics_scroll.setFrameShape(QFrame.NoFrame)
        self._metrics_scroll.setWidgetResizable(False)
        self._metrics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._metrics_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._metrics_scroll.setMinimumWidth(0)
        self._metrics_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._metrics_scroll.setStyleSheet(
            "QScrollArea#statusMetricsScroll { background-color: #34495e; border: none; }"
            "QScrollArea#statusMetricsScroll > QWidget > QWidget { background-color: #34495e; }"
            "QScrollBar:horizontal {"
            "  height: 8px; background: #2c3e50; margin: 0;"
            "}"
            "QScrollBar::handle:horizontal { background: #7f8c8d; min-width: 24px; border-radius: 3px; }"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }"
        )
        self._metrics_scroll.setWidget(self._metrics_widget)
        self._refresh_metrics_width()

        outer.addWidget(self._metrics_scroll, stretch=1)
        outer.addWidget(self.lbl_status, stretch=0)

        self._et_timer = QTimer(self)
        self._et_timer.setInterval(100)
        self._et_timer.timeout.connect(self._on_et_tick)

    def metrics_point_size(self) -> int:
        return int(self.lbl_1d_group.font().pointSize())

    def metrics_value_labels(self):
        return [self.lbl_1d_group, self.lbl_2d_group, self.lbl_3d_group, self.lbl_et]

    def _fmt_step(self, step):
        return self._DASH if step is None else str(int(step))

    def _fmt_tt(self, tt):
        return self._DASH if tt is None else f"{tt:.6e}"

    def _fmt_dt(self, dt):
        return self._DASH if dt is None else f"{dt:.3e}"

    def _format_mode_group(self, mode: str, values: dict) -> str:
        return (
            f"{mode}: Step={self._fmt_step(values.get('step'))}  "
            f"Tt={self._fmt_tt(values.get('tt'))}  "
            f"Δt={self._fmt_dt(values.get('dt'))}"
        )

    def _format_et(self) -> str:
        if self._et_seconds is None:
            return f"ET={self._DASH}"
        return f"ET={self._et_seconds:.1f} s"

    def _format_metrics_line(self) -> str:
        return (
            f"{self._format_mode_group('1D', self._1d)} | "
            f"{self._format_mode_group('2D', self._2d)} | "
            f"{self._format_mode_group('3D', self._3d)} | "
            f"{self._format_et()}"
        )

    def _sync_visible_metrics_line(self) -> None:
        self.lbl_1d_group.setText(self._format_mode_group("1D", self._1d))
        self.lbl_2d_group.setText(self._format_mode_group("2D", self._2d))
        self.lbl_3d_group.setText(self._format_mode_group("3D", self._3d))
        self.lbl_et.setText(self._format_et())
        self.lbl_metrics_line.setText(self._format_metrics_line())
        self._refresh_metrics_width()

    def minimumSizeHint(self):
        sh = super().minimumSizeHint()
        return QSize(0, sh.height())

    def _refresh_metrics_width(self):
        self._metrics_widget.adjustSize()
        hint = self._metrics_widget.sizeHint()
        self._metrics_widget.resize(
            hint.width(), max(hint.height(), self.lbl_1d_group.sizeHint().height())
        )
        self._metrics_scroll.updateGeometry()

    def update_1d(self, step=None, tt=None, dt=None):
        if step is not None:
            self._1d["step"] = step
        if tt is not None:
            self._1d["tt"] = tt
        if dt is not None:
            self._1d["dt"] = dt
        self._sync_visible_metrics_line()

    def update_2d(self, step=None, tt=None, dt=None):
        if step is not None:
            self._2d["step"] = step
        if tt is not None:
            self._2d["tt"] = tt
        if dt is not None:
            self._2d["dt"] = dt
        self._sync_visible_metrics_line()

    def update_3d(self, step=None, tt=None, dt=None, et=None):
        """Update 3D Step/Tt/Δt. Wall-clock ET is managed by start/stop_et_timing."""
        if step is not None:
            self._3d["step"] = step
        if tt is not None:
            self._3d["tt"] = tt
        if dt is not None:
            self._3d["dt"] = dt
        # Ignore legacy et= solver-time argument; ET is real elapsed run time.
        self._sync_visible_metrics_line()

    def set_3d_initial_dt(self, dt_val):
        """Compatibility no-op for status bar — Initial Δt lives in 3D Simulation Control."""
        pass

    def start_et_timing(self) -> None:
        """Begin wall-clock ET at the start of a solver run."""
        import time
        self._et_monotonic_start = time.monotonic()
        self._et_seconds = 0.0
        self._sync_visible_metrics_line()
        self._et_timer.start()

    def stop_et_timing(self) -> None:
        """Freeze ET at the final elapsed value when the run ends or is interrupted."""
        import time
        if self._et_monotonic_start is not None:
            self._et_seconds = time.monotonic() - self._et_monotonic_start
        self._et_monotonic_start = None
        self._et_timer.stop()
        self._sync_visible_metrics_line()

    def _on_et_tick(self) -> None:
        import time
        if self._et_monotonic_start is None:
            return
        self._et_seconds = time.monotonic() - self._et_monotonic_start
        self.lbl_et.setText(self._format_et())

    def set_status(self, text, color="white"):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(self._STATUS_STYLE_TMPL.format(color=color))

    def set_progress(self, value):
        pass  # Progress bar removed


class BlastFoamApp(QMainWindow):
    """Main application window with refactored UI layout"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BlastFoam GUI Manager - v4.0 (Refactored UI)")
        # Default opening size is applied after the central widget is built so
        # size hints cannot defeat the review geometry. This is NOT a minimum.
        self._opening_geometry_applied = False
        
        # Configuration
        self.openfoam_bashrc = "/opt/openfoam9/etc/bashrc"
        self.base_projects_path = r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work"
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
        if os.path.exists(r"\\wsl.localhost\Ubuntu-20.04"):
            os.makedirs(self.base_projects_path, exist_ok=True)
        
        # Services
        self.service = SimulationService(
            base_projects_path=self.base_projects_path,
            openfoam_bashrc=self.openfoam_bashrc
        )
        
        # State
        self.runner = None
        self.active_case_dir_3d = None
        self.active_case_initialized_3d = False
        self.current_project_path = None
        self.view_timer = QTimer()
        self.view_timer.timeout.connect(self.check_3d_updates)
        
        # Build UI
        self._init_toolbar()
        self._init_central_widget()
        
        # Initialize info panel
        self.info_panel.update_info(
            total_cells=0,
            cells_per_dir="—",
            remap_status="None",
            license_status="Active"
        )
        self._apply_default_opening_geometry()
        # Explicitly clear any widget-driven floor so the window stays shrinkable.
        self.setMinimumWidth(0)

    def _apply_default_opening_geometry(self) -> None:
        """First-show default size ≈1685×1060, fitted inside availableGeometry.

        This is intentionally NOT a hard minimum — the user may still shrink
        the window. Never call this from tab/status/run update paths.
        """
        from ui_metrics import DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
        from PyQt5.QtGui import QGuiApplication
        from PyQt5.QtCore import QRect

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
        else:
            avail = QRect(0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        w = min(int(DEFAULT_WINDOW_WIDTH), int(avail.width()))
        h = min(int(DEFAULT_WINDOW_HEIGHT), int(avail.height()))
        x = int(avail.x() + max(0, (avail.width() - w) // 2))
        y = int(avail.y() + max(0, (avail.height() - h) // 2))
        self.setGeometry(x, y, w, h)
        self._opening_geometry_applied = True

    def minimumSizeHint(self):
        """Do not let toolbar/tab size hints impose a large top-level floor."""
        sh = super().minimumSizeHint()
        return QSize(0, sh.height())

    def showEvent(self, event):
        # Ensure the first real show uses the review opening size even if a
        # platform/size-hint path briefly applied a different geometry.
        if not self._opening_geometry_applied:
            self._apply_default_opening_geometry()
        super().showEvent(event)

    def closeEvent(self, event):
        """Stop the solver when closing the window so the simulation does not keep running."""
        if self.runner and self.runner.isRunning():
            self.runner.stop()
            self.runner.wait(3000)
        event.accept()

    def _init_toolbar(self):
        """Create top toolbar with global actions"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize() * 1.2)
        # Keep all actions; rely on QToolBar overflow rather than window min-width.
        toolbar.setMinimumWidth(0)
        toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        self._main_toolbar = toolbar
        
        # Open model
        act_open = QAction("📂 Open Model", self)
        act_open.setToolTip("Open existing model file")
        act_open.triggered.connect(self._on_open_model)
        toolbar.addAction(act_open)

        act_open_case = QAction("📂 Open OpenFOAM Case", self)
        act_open_case.setToolTip("Open an existing generated OpenFOAM case folder")
        act_open_case.triggered.connect(self._on_open_case)
        toolbar.addAction(act_open_case)
        
        # Save model
        act_save = QAction("💾 Save Model", self)
        act_save.setToolTip("Save current model")
        act_save.triggered.connect(self._on_save_model)
        toolbar.addAction(act_save)
        
        # Save As
        act_save_as = QAction("💾 Save As...", self)
        act_save_as.setToolTip("Save model to new file")
        act_save_as.triggered.connect(self._on_save_model_as)
        toolbar.addAction(act_save_as)
        
        toolbar.addSeparator()
        
        # Output options
        act_output = QAction("⚙️ Output Options", self)
        act_output.setToolTip("Configure output file options")
        act_output.triggered.connect(self._on_output_options)
        toolbar.addAction(act_output)
        
        # Time history locations
        act_time_history = QAction("📍 Time History Locations", self)
        act_time_history.setToolTip("Edit gauge/probe locations for time history output")
        act_time_history.triggered.connect(self._on_time_history_locations)
        toolbar.addAction(act_time_history)
        
        toolbar.addSeparator()
        
        # Help/About
        act_help = QAction("❓ Help", self)
        act_help.setToolTip("Show help documentation")
        act_help.triggered.connect(self._on_help)
        toolbar.addAction(act_help)
        
        act_about = QAction("ℹ️ About", self)
        act_about.setToolTip("About BlastFoam GUI Manager")
        act_about.triggered.connect(self._on_about)
        toolbar.addAction(act_about)
    
    def _init_central_widget(self):
        """Create main window layout with panels"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Primary tabs (full width at top for tab bar visibility)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        # Prefer full titles when space allows; elide + scroll when the window is narrowed.
        # ElideNone must NOT be used — it reintroduces a large top-level minimumWidth.
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.ElideRight)
        self.tabs.setMinimumWidth(0)
        
        # Initialize shared data models
        self.probes_model = ProbesModel()
        
        # Create tabs
        self.tab_1d = Tab1D()
        self.tab_2d = Tab2D()
        self.tab_3d = TabGeneral3D(self.probes_model)
        self.tab_time_history = self._create_placeholder_tab("Time History Viewer", "Time history viewing (not yet implemented)")
        self.tab_pi_curves = self._create_placeholder_tab("PI-Curves", "PI curve analysis (not yet implemented)")
        self.tab_monte_carlo = self._create_placeholder_tab("Monte-Carlo", "Monte Carlo batch analysis (not yet implemented)")
        self.tab_jotter = LogTab()  # Keep exact name "Jotter"
        self.tab_plotter = self._create_placeholder_tab("Plotter", "Data plotting utility (not yet implemented)")
        self.tab_probes = TabProbes(self.probes_model)
        
        # Add tabs in specified order
        self.tabs.addTab(self.tab_1d, "Spherical – 1D")
        self.tabs.addTab(self.tab_2d, "Cylindrical – 2D")
        self.tabs.addTab(self.tab_3d, "General 3D")
        self.tabs.addTab(self.tab_time_history, "Time History Viewer")
        self.tabs.addTab(self.tab_pi_curves, "PI-Curves")
        self.tabs.addTab(self.tab_monte_carlo, "Monte-Carlo")
        self.tabs.addTab(self.tab_jotter, "Jotter")
        self.tabs.addTab(self.tab_plotter, "Plotter")
        
        # Connect tab signals (preserve existing connections)
        self.tab_1d.sig_request_run.connect(lambda: (self.tabs.setCurrentWidget(self.tab_1d), self.run_active_tab()))
        self.tab_1d.sig_request_stop.connect(self.on_stop_request)
        
        self.tab_3d.sig_request_init.connect(self.on_initialize_model_3d)
        self.tab_3d.sig_request_run.connect(lambda: (self.tabs.setCurrentWidget(self.tab_3d), self.run_active_tab()))
        self.tab_3d.sig_request_run_exact_1.connect(lambda: (self.tabs.setCurrentWidget(self.tab_3d), self.run_3d_process_exact_1()))
        self.tab_3d.sig_request_run_exact_end.connect(lambda: (self.tabs.setCurrentWidget(self.tab_3d), self.run_3d_process_exact_end()))
        self.tab_3d.sig_request_stop.connect(self.on_stop_request)

        # Shared computational left-panel width across 1D / 2D / 3D (session-preserved).
        from ui_metrics import COMPUTATIONAL_LEFT_PANEL_WIDTH
        self._computational_left_width = COMPUTATIONAL_LEFT_PANEL_WIDTH
        self.tabs.currentChanged.connect(self._on_main_tab_changed)
        for _tab in self._computational_tabs():
            if hasattr(_tab, "set_computational_left_width"):
                _tab.set_computational_left_width(self._computational_left_width)
            splitter = getattr(_tab, "_main_splitter", None) or getattr(_tab, "splitter", None)
            if splitter is not None and not getattr(splitter, "_ggui_width_hooked", False):
                splitter.splitterMoved.connect(self._on_computational_splitter_moved)
                splitter._ggui_width_hooked = True
        # Re-apply after first show so realized splitter geometry matches the target.
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._apply_opening_computational_left_width)
        
        # Main content area with splitter
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        
        # Left side: Input Panel + Info Panel (stacked vertically)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)
        
        # Input Panel (upper left) - currently embedded in tabs, will stay there
        # Info Panel (lower left) - new panel showing derived info
        self.info_panel = InfoPanel()
        left_layout.addWidget(self.info_panel)
        
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(300)
        left_scroll.setMaximumWidth(400)
        left_scroll.setVisible(False)  # Hide Model Info panel (left column with License, etc.)
        
        content_splitter.addWidget(left_scroll)
        
        # Center-right: Tabs (contains viewport and controls for each tab)
        content_splitter.addWidget(self.tabs)
        content_splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(content_splitter, stretch=1)
        
        # Bottom: Segmented Status Bar
        self.status_bar = SegmentedStatusBar()
        main_layout.addWidget(self.status_bar)
        self.tab_3d.initial_dt_changed.connect(self._on_3d_initial_dt_changed)
        self.tab_3d._update_calculated_dt_label()

    def _computational_tabs(self):
        return (self.tab_1d, self.tab_2d, self.tab_3d)

    def _apply_opening_computational_left_width(self) -> None:
        from ui_metrics import COMPUTATIONAL_LEFT_PANEL_WIDTH
        if self._computational_left_width is None:
            self._computational_left_width = COMPUTATIONAL_LEFT_PANEL_WIDTH
        for tab in self._computational_tabs():
            if hasattr(tab, "set_computational_left_width"):
                tab.set_computational_left_width(self._computational_left_width)

    def _on_main_tab_changed(self, _index: int) -> None:
        """Preserve shared left-panel width across 1D/2D/3D tab switches."""
        self._sync_computational_left_width(
            self.tabs.currentWidget(), record_from_current=False
        )

    def _sync_computational_left_width(self, widget, record_from_current: bool = False) -> None:
        if widget not in self._computational_tabs():
            return
        if record_from_current and hasattr(widget, "get_computational_left_width"):
            self._computational_left_width = widget.get_computational_left_width()
        if self._computational_left_width is None:
            from ui_metrics import COMPUTATIONAL_LEFT_PANEL_WIDTH
            self._computational_left_width = COMPUTATIONAL_LEFT_PANEL_WIDTH
        if hasattr(widget, "set_computational_left_width"):
            widget.set_computational_left_width(self._computational_left_width)
        splitter = getattr(widget, "_main_splitter", None) or getattr(widget, "splitter", None)
        if splitter is not None and not getattr(splitter, "_ggui_width_hooked", False):
            splitter.splitterMoved.connect(self._on_computational_splitter_moved)
            splitter._ggui_width_hooked = True

    def _on_computational_splitter_moved(self, _pos: int = 0, _index: int = 0) -> None:
        current = self.tabs.currentWidget()
        if current in self._computational_tabs() and hasattr(current, "get_computational_left_width"):
            self._computational_left_width = current.get_computational_left_width()
    
    def _create_placeholder_tab(self, title, message):
        """Create a placeholder tab for unimplemented features"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        
        label = QLabel(f"<h2>{title}</h2><p>{message}</p><p><i>Still in progress...</i></p>")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(label)
        
        return widget
    
    # ====== Toolbar Action Handlers ======
    
    def _on_open_model(self):
        """Open a versioned GGUI project file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open GGUI Project",
            os.path.dirname(self.current_project_path) if self.current_project_path else "",
            "GGUI Project (*.ggui.json);;JSON (*.json)",
        )
        if not path:
            return
        try:
            project = read_project(path)
            apply_project_payload(self.tab_3d, self.probes_model, project)
            self.current_project_path = os.path.abspath(path)
            self.active_case_dir_3d = None
            self.active_case_initialized_3d = False
            self.tabs.setCurrentWidget(self.tab_3d)
            self.status_bar.set_status("Project loaded", "#2ecc71")
        except (ProjectFormatError, OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Open Project Error", str(exc))

    def _on_open_case(self):
        """Open an existing BlastFoam case folder and populate the 3D tab."""
        case_dir = QFileDialog.getExistingDirectory(
            self,
            "Open BlastFoam Case Folder",
            self.base_projects_path if os.path.isdir(self.base_projects_path) else "",
        )
        if not case_dir:
            return

        # Detect nested case: if selected dir contains a sub-folder with system/
        # (e.g. building3D/building3D/system/), use the inner folder.
        sys_dir = os.path.join(case_dir, "system")
        if not os.path.isdir(sys_dir):
            # Check one level deeper
            for entry in os.listdir(case_dir):
                inner = os.path.join(case_dir, entry)
                if os.path.isdir(inner) and os.path.isdir(os.path.join(inner, "system")):
                    case_dir = inner
                    break

        if not os.path.isdir(os.path.join(case_dir, "system")):
            QMessageBox.warning(
                self, "Invalid Case",
                "The selected folder does not contain a 'system/' sub-directory.\n"
                "Please select a valid OpenFOAM case folder."
            )
            return

        try:
            data = load_case(case_dir)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to parse case:\n{e}")
            return

        load_summary = data.pop("_load_summary", None)
        # Populate 3D tab: full fill (defaults for not_filled, then data)
        self.tab_3d.set_case_inputs(data, load_summary=load_summary)

        # Switch to 3D tab
        self.tabs.setCurrentWidget(self.tab_3d)
        self.active_case_dir_3d = case_dir
        self.active_case_initialized_3d = os.path.isdir(os.path.join(case_dir, "0"))
        self.current_project_path = None

        # Load Summary (non-blocking): fields filled / not filled / unsupported + Copy
        self._show_load_summary_dialog(case_dir, load_summary)

    def _show_load_summary_dialog(self, case_dir: str, load_summary: dict) -> None:
        """Show Load Summary (fields filled, not filled, unsupported by file) with Copy button."""
        if not load_summary:
            return
        filled = load_summary.get("filled", [])
        not_filled = load_summary.get("not_filled", [])
        unsupported = load_summary.get("unsupported", {})
        notes = load_summary.get("notes", [])
        n_filled = len(filled)
        n_not = len(not_filled)
        lines = [
            f"Loaded: {case_dir}",
            "",
            f"Fields filled from case (LOADED): {n_filled}",
            f"Fields not filled (UNSET): {n_not}",
            "",
        ]
        if filled:
            lines.append("Filled: " + ", ".join(sorted(filled)))
            lines.append("")
        if not_filled:
            lines.append("Not filled (key — reason):")
            for k, reason in sorted(not_filled, key=lambda x: x[0]):
                lines.append(f"  {k} — {reason}")
            lines.append("")
        if unsupported:
            lines.append("Keys in case files not mapped to GUI (not supported yet):")
            for fpath in sorted(unsupported.keys()):
                keys = unsupported[fpath]
                if keys:
                    lines.append(f"  {fpath}: " + ", ".join(keys))
        if notes:
            lines.append("")
            lines.append("Load notes (seed / charge capture interpretation):")
            for note in notes:
                lines.append(f"  - {note}")
        lines.append("")
        lines.append("Pre-run parity: To verify generated files match this case (no edits), run:")
        lines.append(f"  python verification/parity_building3d.py -r \"{case_dir}\"")
        text = "\n".join(lines)

        dlg = QDialog(self)
        dlg.setWindowTitle("Load Summary")
        layout = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        te.setMinimumSize(480, 320)
        layout.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        copy_btn = QPushButton("Copy summary")
        copy_btn.clicked.connect(lambda: QApplication.instance().clipboard().setText(text))
        bb.addButton(copy_btn, QDialogButtonBox.ActionRole)
        bb.accepted.connect(dlg.accept)
        layout.addWidget(bb)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_save_model(self):
        """Save current model to its existing project path."""
        if not self.current_project_path:
            self._on_save_model_as()
            return
        self._save_project_to(self.current_project_path)
    
    def _on_save_model_as(self):
        """Save current model to a new project path."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GGUI Project",
            self.current_project_path or f"project{PROJECT_SUFFIX}",
            "GGUI Project (*.ggui.json)",
        )
        if not path:
            return
        if not path.lower().endswith(PROJECT_SUFFIX):
            path += PROJECT_SUFFIX
        if self._save_project_to(path):
            self.current_project_path = os.path.abspath(path)

    def _save_project_to(self, path: str) -> bool:
        try:
            payload = capture_project_payload(self.tab_3d, self.probes_model)
            write_project_atomic(path, payload)
            self.status_bar.set_status("Project saved", "#2ecc71")
            return True
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Save Project Error", f"Could not save project:\n{exc}")
            return False
    
    def _on_output_options(self):
        """Open Output File Options dialog"""
        QMessageBox.information(self, "Output Options", "Output file options: Still in progress...")
    
    def _on_time_history_locations(self):
        """Open Time History Locations dialog"""
        QMessageBox.information(self, "Time History Locations", "Edit time history output locations: Still in progress...")
    
    def _on_help(self):
        """Show help"""
        help_text = """
        <h2>BlastFoam GUI Manager - Help</h2>
        <h3>Getting Started</h3>
        <ul>
        <li><b>1D Simulations:</b> Use "Spherical – 1D" tab for spherical blast analysis</li>
        <li><b>3D Simulations:</b> Use "General 3D" tab for complex 3D scenarios with obstacles</li>
        <li><b>Gauges/Probes:</b> Use "Jotter" tab for time-history data</li>
        </ul>
        <h3>Workflow</h3>
        <ol>
        <li>Configure domain and charge parameters</li>
        <li>Click "Initialize Model" (3D only)</li>
        <li>Click "Run Simulation"</li>
        <li>Monitor progress in status bar and live log</li>
        </ol>
        """
        QMessageBox.information(self, "Help", help_text)
    
    def _on_about(self):
        """Show about dialog"""
        about_text = """
        <h2>BlastFoam GUI Manager</h2>
        <p><b>Version:</b> 4.0 (Refactored UI)</p>
        <p><b>Description:</b> GUI for OpenFOAM/blastFoam blast wave simulations</p>
        <p><b>Features:</b></p>
        <ul>
        <li>1D spherical blast simulations</li>
        <li>3D complex geometry with obstacles</li>
        <li>Radial remapping from 1D to 3D</li>
        <li>Real-time visualization</li>
        </ul>
        """
        QMessageBox.about(self, "About BlastFoam GUI Manager", about_text)
    
    # ====== Simulation Control (Preserved from original) ======
    
    def _run_wsl_commands(self, case_dir, cmds):
        """Execute WSL commands in case directory and log output to file.
        Supports both UNC (\\\\wsl.localhost\\Distro\\...) and Windows paths (C:\\...)."""
        import sys
        from pathlib import Path
        
        distro, linux_path = SolverRunner._win_unc_to_wsl_path_and_distro(case_dir)
        full_cmd = f"source {self.openfoam_bashrc}; cd {linux_path}; {cmds}"
        if distro:
            wsl_args = ["wsl", "-d", distro, "bash", "-lc", full_cmd]
        else:
            wsl_args = ["wsl", "bash", "-lc", full_cmd]
        
        # Create log file for initialization output
        log_file = Path(case_dir) / "log.initialize"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as log:
                log.write("="*60 + "\n")
                log.write("Initialize Command Log\n")
                log.write("="*60 + "\n")
                log.write(f"Directory: {linux_path}\n")
                log.write(f"Command: {cmds}\n")
                log.write("="*60 + "\n\n")
                log.flush()
                
                result = subprocess.run(wsl_args, check=False, capture_output=True, text=True)
                
                if result.stdout:
                    log.write("STDOUT:\n")
                    log.write(result.stdout)
                    log.write("\n\n")
                
                if result.stderr:
                    log.write("STDERR:\n")
                    log.write(result.stderr)
                    log.write("\n\n")
                
                log.write("="*60 + "\n")
                log.write(f"Result: {'SUCCESS' if result.returncode == 0 else 'FAILED with exit code ' + str(result.returncode)}\n")
                log.write("="*60 + "\n")
                
            # Also try to print to console if available
            print(f"\n[Initialize] Command {'succeeded' if result.returncode == 0 else 'FAILED with code ' + str(result.returncode)}")
            print(f"[Initialize] Log saved to: {log_file}")
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"ERROR running WSL commands: {e}", file=sys.stderr)
            return False
    
    def run_active_tab(self):
        """Run simulation for active tab"""
        current_widget = self.tabs.currentWidget()
        if self.runner and self.runner.isRunning():
            QMessageBox.information(self, "Info", "Simulation is already running.")
            return
        
        if current_widget == self.tab_1d:
            self.run_1d_process()
        elif current_widget == self.tab_3d:
            self.run_3d_process()
        else:
            QMessageBox.information(self, "Info", "Please select 1D or 3D tab to run simulation.")
    
    def run_1d_process(self):
        """Execute 1D simulation"""
        try:
            inputs = self.tab_1d.get_case_inputs()
            if not isinstance(inputs, CaseInputs1D):
                raise ValueError("Invalid 1D Inputs")
            
            self.status_bar.set_status("Generating 1D Case...", "#f39c12")
            QApplication.processEvents()
            
            prefix = "Case_1D"
            case_name = self.service.make_case_name(prefix)
            case_dir = self.service.generate_case(case_name, inputs)
            self._start_solver(case_dir, cores=1, mode="1D")
            
        except Exception as e:
            self.status_bar.set_status("Error", "#e74c3c")
            QMessageBox.critical(self, "1D Error", str(e))
    
    def _split_remap_path(self, path):
        """Split remap path into case directory and time directory"""
        norm = os.path.normpath((path or "").strip())
        if not norm:
            return "", None
        last = os.path.basename(norm)
        try:
            if float(last) >= 0:
                parent = os.path.dirname(norm)
                if parent:
                    return parent, last
        except ValueError:
            pass
        return norm, None

    def _show_initialize_result_summary(
        self,
        case_dir: str,
        inputs: CaseInputs3D,
        use_remap: bool,
        mode: dict | None,
        retries_used: int,
        set_cmd_actual: str | None,
    ) -> None:
        """Show concise post-initialize summary with explicit effective outcomes."""
        lines = [
            "3D Initialize Result Summary",
            "",
            f"Case directory: {case_dir}",
            f"Initialize mode: {'Remap' if use_remap else 'Standard'}",
            f"Dynamic refine enabled: {getattr(inputs, 'enable_dyn_refine', None)}",
            f"Obstacle refine enabled: {getattr(inputs, 'enable_obstacle_refine', None)}",
            f"Estimated charge cells (preflight): {getattr(inputs, 'estimated_charge_cells', 0.0):.2f}",
        ]
        if use_remap:
            lines.extend(
                [
                    f"Remap source: {(getattr(inputs, 'remap_case_path', '') or '').strip() or '—'}",
                    f"Remap time mode: {getattr(inputs, 'remap_time_mode', 'latest')}",
                    f"Remap specific time: {getattr(inputs, 'remap_specific_time', '—')}",
                ]
            )
        else:
            mode = mode or {}
            lines.extend(
                [
                    f"set_cmd (effective): {mode.get('set_cmd', '—')}",
                    f"set_cmd_actual (executed): {set_cmd_actual or '—'}",
                    f"capture_levels: {mode.get('capture_levels', 0)}",
                    f"charge_levels: {mode.get('charge_levels', 0)}",
                    f"outside_levels: {mode.get('outside_levels', 0)}",
                    f"charge_refinement_effective: {mode.get('charge_refinement_effective', 0)}",
                    f"charge_clipped_by_domain: {mode.get('charge_clipped_by_domain', '—')}",
                    f"cells_inside_charge: {mode.get('cells_inside_charge', '—')}",
                    f"charge_capture_radius_used_m: {(mode.get('charge_capture') or {}).get('charge_capture_radius_used_m', '—')}",
                    f"charge_capture_mode: {(mode.get('charge_capture') or {}).get('mode', '—')}",
                ]
            )
        lines.append(f"Retries used: {retries_used}")
        self._write_initialize_summary_file(case_dir, lines)

    def _write_initialize_summary_file(self, case_dir: str, lines: list[str]) -> None:
        """Persist initialize summary to case folder for full traceability."""
        try:
            out_path = os.path.join(case_dir, "initialize_summary.txt")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload = [
                "=" * 72,
                f"Initialize Summary @ {ts}",
                "=" * 72,
                *lines,
                "",
            ]
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n".join(payload))
        except OSError:
            # Non-fatal: summary dialog is still shown in UI.
            pass
    
    def on_initialize_model_3d(self, inputs):
        """Initialize 3D model (mesh generation and field initialization)"""
        if isinstance(inputs, CaseInputs3D):
            try:
                try:
                    require_safe_capture(inputs)
                except ValueError as guard_exc:
                    QMessageBox.critical(
                        self,
                        "Unsafe charge capture",
                        str(guard_exc) or UNSAFE_CAPTURE_MESSAGE,
                    )
                    self.status_bar.set_status("Init blocked", "#e74c3c")
                    return
                self.status_bar.set_status("Generating 3D Case...", "#f39c12")
                QApplication.processEvents()

                prefix = "Case_3D"
                case_name = self.service.make_case_name(prefix)
                try:
                    case_dir = self.service.generate_case(case_name, inputs)
                except ValueError as gen_exc:
                    msg = str(gen_exc)
                    if "Initialization is blocked" in msg or msg == UNSAFE_CAPTURE_MESSAGE:
                        QMessageBox.critical(self, "Unsafe charge capture", msg)
                        self.status_bar.set_status("Init blocked", "#e74c3c")
                        return
                    raise
                self.active_case_dir_3d = case_dir
                self.active_case_initialized_3d = False
                if not getattr(inputs, "remap_enabled", False):
                    _cap_path = os.path.join(case_dir, "case_init_mode.json")
                    if os.path.isfile(_cap_path):
                        try:
                            import json
                            with open(_cap_path, "r", encoding="utf-8") as _cf:
                                _cap_mode = json.load(_cf)
                            _cap_ws = (_cap_mode.get("charge_capture") or {}).get("warnings") or []
                            if _cap_ws:
                                QMessageBox.warning(
                                    self,
                                    "Charge capture",
                                    "\n\n".join(_cap_ws),
                                )
                        except (OSError, ValueError, KeyError):
                            pass

                self.status_bar.set_status("Initializing Mesh...", "#f39c12")
                QApplication.processEvents()
                
                has_obstacles = len(inputs.obstacles) > 0
                remap_enabled = getattr(inputs, "remap_enabled", False)
                remap_case_path = (getattr(inputs, "remap_case_path", "") or "").strip()
                use_remap = remap_enabled and bool(remap_case_path)
                
                if use_remap:
                    source_case_dir_win, source_time_from_path = self._split_remap_path(remap_case_path)
                    mapped_source_dir_linux = win_to_wsl_path(source_case_dir_win) if source_case_dir_win else ""
                    remap_time_mode = getattr(inputs, "remap_time_mode", "latest") or "latest"
                    if remap_time_mode == "latest":
                        mapped_source_time = get_latest_time_dir(source_case_dir_win) or source_time_from_path
                        if not mapped_source_time:
                            QMessageBox.critical(
                                self,
                                "Remap time not defined",
                                "Could not resolve a remap time in 'latest' mode.\n"
                                "Please select a source case with solved time folders or switch to 'specific time'."
                            )
                            self.status_bar.set_status("Init blocked", "#e74c3c")
                            return
                    else:
                        mapped_source_time = getattr(inputs, "remap_specific_time", None) or source_time_from_path
                        if not mapped_source_time:
                            QMessageBox.critical(
                                self,
                                "Remap time not defined",
                                "Specific remap time is empty.\n"
                                "Please define a specific time before Initialize."
                            )
                            self.status_bar.set_status("Init blocked", "#e74c3c")
                            return
                    use_remap = bool(mapped_source_dir_linux and mapped_source_time)
                
                # Start clean: run Allclean if present (safe; preserves 0.orig, system/, constant dicts, triSurface)
                clean_first = "([ -x ./Allclean ] && ./Allclean || true) && "
                set_cmd_actual = None
                if use_remap:
                    if has_obstacles:
                        preflight_remap = " && ( [ ! -f system/expectedFeatureEdges.txt ] || ( while read -r f; do [ -z \"$f\" ] && continue; [ -f \"$f\" ] || { echo \"FATAL: .eMesh required at $f missing\"; exit 1; }; done < system/expectedFeatureEdges.txt ) ) && "
                        init_cmd = (
                            clean_first
                            + "blockMesh && surfaceFeatures "
                            + preflight_remap
                            + "snappyHexMesh -overwrite && "
                            "addEmptyPatch internalPatch internal -overwrite && "
                            "rm -rf 0 && cp -r 0.orig 0 && changeDictionary && "
                            "postProcess -func writeCellCentres && python3 remap_radial.py"
                        )
                    else:
                        init_cmd = (
                            clean_first
                            + "blockMesh && rm -rf 0 && cp -r 0.orig 0 && "
                            "postProcess -func writeCellCentres && python3 remap_radial.py"
                        )
                else:
                    # Use generator's decision (case_init_mode.json) so set_cmd and startup_refinement_levels match setFieldsDict/Allrun
                    set_cmd = None
                    mode = {}
                    mode_path = os.path.join(case_dir, "case_init_mode.json")
                    if os.path.isfile(mode_path):
                        try:
                            import json
                            with open(mode_path, "r", encoding="utf-8") as f:
                                mode = json.load(f)
                            set_cmd = mode.get("set_cmd")
                        except (OSError, ValueError, KeyError):
                            pass
                    if set_cmd not in ("setFields", "setRefinedFields"):
                        set_cmd = build_initialization_plan(inputs).command
                    alpha_check = "bash ./check_alpha_c4.sh || exit 1"
                    # Native flow (matches BlastFoam ``building3D`` reference):
                    #   blockMesh → surfaceFeatures (if obstacles) → snappyHexMesh
                    #   → addEmptyPatch → restore 0 → changeDictionary
                    #   → setRefinedFields (refines mesh inside charge AND fills α.c4 in one
                    #   pass via ``refineInternal yes; level N`` in setFieldsDict regions).
                    # No manual topoSet+refineMesh capture/charge stages: setRefinedFields
                    # uses the setFieldsDict charge capture region (backup {{ ... }}) as the
                    # mesh search fallback — separate from the snappy outer transition sphere.
                    preflight = " && ( [ ! -f system/expectedFeatureEdges.txt ] || ( while read -r f; do [ -z \"$f\" ] && continue; [ -f \"$f\" ] || { echo \"FATAL: .eMesh required at $f missing\"; exit 1; }; done < system/expectedFeatureEdges.txt ) ) && "
                    init_part1 = (
                        clean_first
                        + "blockMesh && surfaceFeatures "
                        + preflight
                        + "snappyHexMesh -overwrite"
                        + " && addEmptyPatch internalPatch internal -overwrite && "
                        "rm -rf 0 && cp -r 0.orig 0 && changeDictionary"
                    )
                    init_part2 = f"{set_cmd} && {alpha_check}"
                    set_cmd_actual = set_cmd
                
                if not use_remap and not os.path.isfile(os.path.join(case_dir, "check_alpha_c4.sh")):
                    self.status_bar.set_status("Init Failed", "#e74c3c")
                    QMessageBox.critical(
                        self, "Init Error",
                        "FATAL: missing check_alpha_c4.sh (case not generated with this version)."
                    )
                    return
                
                if use_remap:
                    success = self._run_wsl_commands(case_dir, init_cmd)
                else:
                    success = self._run_wsl_commands(case_dir, init_part1)
                    if not success:
                        self.status_bar.set_status("Init Failed", "#e74c3c")
                        QMessageBox.critical(
                            self, "Init Error",
                            "Mesh initialization (blockMesh / snappyHexMesh / addEmptyPatch / changeDictionary) failed. Check console output for details."
                        )
                        return
                    success = self._run_wsl_commands(case_dir, init_part2)
                retries_used = 0
                if not success and not use_remap:
                    self.status_bar.set_status("Init Failed", "#e74c3c")
                    msg = (
                        "The charge was not captured with the current base mesh and charge-seeding settings.\n\n"
                        "Suggested actions:\n"
                        "• Increase charge seed refinement level (Charge pre-refinement / Inside).\n"
                        "• Increase the charge capture radius (Mesh Properties → Advanced).\n"
                        "• Reduce base cell size.\n"
                        "• Check charge location relative to the mesh.\n"
                    )
                    QMessageBox.critical(self, "Init Error", msg)
                    return
                if not success and use_remap:
                    self.status_bar.set_status("Init Failed", "#e74c3c")
                    QMessageBox.critical(self, "Init Error", "Mesh initialization failed. Check console output for details.")
                    return
                final_mode = None
                if not use_remap:
                    try:
                        charge_cells = None
                        if get_charge_cell_count and os.path.isfile(os.path.join(case_dir, "0", "alpha.c4")):
                            charge_cells, _ = get_charge_cell_count(case_dir, "0", 0.5)
                        final_mode = record_set_cmd_actual(
                            case_dir,
                            set_cmd_actual,
                            retries_used=retries_used,
                            cells_inside_charge=charge_cells,
                        )
                    except (OSError, ValueError, KeyError, TypeError):
                        final_mode = None
                
                # Update viewer with charge center (no auto-adjust; user position is used)
                self.tab_3d.viewer.load_case(
                    case_dir,
                    charge_center=inputs.charge_center,
                    cell_size=inputs.cell_size,
                )
                # Show alpha.c4 (charge) in red at step 0 so the charge is visible after init
                self.tab_3d.viewer.set_field("alpha.c4")
                # Sync Viewport "Field" combo to "Energy" (alpha.c4) without triggering extra refresh
                if hasattr(self.tab_3d, "cmb_field") and self.tab_3d.cmb_field.findText("Energy") >= 0:
                    self.tab_3d.cmb_field.blockSignals(True)
                    self.tab_3d.cmb_field.setCurrentText("Energy")
                    self.tab_3d.cmb_field.blockSignals(False)
                
                # Update info panel (cells + charge cells from 0/alpha.c4)
                self._update_3d_info_panel(inputs)
                if self.active_case_dir_3d:
                    self.tab_3d.update_charge_cells_display(self.active_case_dir_3d, threshold=0.5)
                
                self._show_initialize_result_summary(
                    case_dir=case_dir,
                    inputs=inputs,
                    use_remap=use_remap,
                    mode=final_mode,
                    retries_used=retries_used,
                    set_cmd_actual=(set_cmd_actual if not use_remap else None),
                )
                self.status_bar.set_status("3D Initialized", "#2ecc71")
                self.active_case_initialized_3d = True
                
            except Exception as e:
                self.status_bar.set_status("Error", "#e74c3c")
                QMessageBox.critical(self, "3D Init Error", str(e))
    
    def _update_3d_info_panel(self, inputs):
        """Update info panel with 3D model information"""
        if isinstance(inputs, CaseInputs3D):
            dx = inputs.cell_size
            nx = int((inputs.max_point[0] - inputs.min_point[0]) / dx)
            ny = int((inputs.max_point[1] - inputs.min_point[1]) / dx)
            nz = int((inputs.max_point[2] - inputs.min_point[2]) / dx)
            total = nx * ny * nz
            
            remap_status = "Enabled" if getattr(inputs, "remap_enabled", False) else "Disabled"
            
            self.info_panel.update_info(
                total_cells=total,
                cells_per_dir=f"{nx}×{ny}×{nz}",
                remap_status=remap_status,
                license_status="Active"
            )
    
    def run_3d_process(self):
        """Execute 3D simulation"""
        try:
            inputs = self.tab_3d.get_case_inputs()
            
            if not self.active_case_dir_3d or not self.active_case_initialized_3d:
                self.on_initialize_model_3d(inputs)
                if not self.active_case_dir_3d or not self.active_case_initialized_3d:
                    return
            
            # Get write parameters based on control type
            if inputs.write_control_type == "adjustableRunTime":
                write_param = inputs.write_interval_time
            else:
                write_param = inputs.write_interval_steps
            
            self._update_control_dict_end_time(
                self.active_case_dir_3d,
                inputs.end_time_s,
                write_param,
                inputs.write_control_type
            )
            try:
                probe = build_execution_plan(
                    self.active_case_dir_3d,
                    inputs.cores,
                    ExecutionIntent.RESUME,
                )
                intent = ExecutionIntent.RESUME
            except ExecutionPreparationError as exc:
                if str(exc).startswith("No resumable saved time exists"):
                    intent = ExecutionIntent.INITIALIZED_SOLVER_RUN
                else:
                    raise
            self._start_solver(
                self.active_case_dir_3d,
                cores=inputs.cores,
                mode="3D",
                intent=intent,
            )
            
        except Exception as e:
            self.status_bar.set_status("Error", "#e74c3c")
            QMessageBox.critical(self, "3D Run Error", str(e))
    
    def _update_control_dict_end_time(self, case_dir, new_end_time, write_int, write_control_type="timeStep"):
        """Update controlDict with new end time and write interval"""
        cd_path = os.path.join(case_dir, "system", "controlDict")
        if not os.path.exists(cd_path):
            return
        try:
            with open(cd_path, 'r') as f:
                lines = f.readlines()
            with open(cd_path, 'w') as f:
                for line in lines:
                    if line.strip().startswith("endTime"):
                        f.write(f"endTime {new_end_time};\n")
                    elif line.strip().startswith("writeInterval"):
                        f.write(f"writeInterval {write_int};\n")
                    else:
                        f.write(line)
        except OSError as exc:
            raise RuntimeError(f"Could not update {cd_path}: {exc}") from exc

    def _set_control_dict_one_step(self, case_dir):
        """Set controlDict so the solver runs exactly one more step from the latest time and stops.
        Uses the latest time directory (so each 'exact 1' continues from the previous step).
        Sets startFrom latestTime and endTime = latestTime + deltaT. Returns True on success."""
        cd_path = os.path.join(case_dir, "system", "controlDict")
        if not os.path.exists(cd_path):
            return False
        try:
            delta_t = None
            with open(cd_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines:
                s = line.strip()
                if s.startswith("deltaT"):
                    tokens = s.replace(";", "").split()
                    if len(tokens) >= 2:
                        try:
                            delta_t = float(tokens[1])
                            break
                        except ValueError:
                            pass
            if delta_t is None or delta_t <= 0:
                return False
            # Start from latest time directory so each 'exact 1' continues from the previous step
            execution = build_execution_plan(
                case_dir,
                max(1, int(self.tab_3d.spin_cores.value())),
                ExecutionIntent.ONE_STEP_RESUME,
            )
            start_time = float(execution.latest_time or 0.0)
            one_step_end = start_time + delta_t
            with open(cd_path, "w", encoding="utf-8") as f:
                for line in lines:
                    s = line.strip()
                    if s.startswith("endTime"):
                        f.write(f"endTime {one_step_end};\n")
                    elif s.startswith("startFrom"):
                        f.write("startFrom       latestTime;\n")
                    else:
                        f.write(line)
            return True
        except (OSError, ValueError, ExecutionPreparationError) as exc:
            raise RuntimeError(f"Cannot prepare one-step resume: {exc}") from exc

    def run_3d_process_exact_1(self):
        """Run 3D solver for exactly one time step then stop."""
        try:
            inputs = self.tab_3d.get_case_inputs()
            if not self.active_case_dir_3d or not self.active_case_initialized_3d:
                self.on_initialize_model_3d(inputs)
                if not self.active_case_dir_3d or not self.active_case_initialized_3d:
                    return
            if not self._set_control_dict_one_step(self.active_case_dir_3d):
                QMessageBox.critical(self, "exact 1", "Could not set one-step end time (check system/controlDict startTime and deltaT).")
                return
            self._start_solver(
                self.active_case_dir_3d,
                cores=inputs.cores,
                mode="3D",
                intent=ExecutionIntent.ONE_STEP_RESUME,
            )
        except Exception as e:
            self.status_bar.set_status("Error", "#e74c3c")
            QMessageBox.critical(self, "3D Run Error", str(e))

    def run_3d_process_exact_end(self):
        """Run 3D solver until stop or end time."""
        self.run_3d_process()
    
    def _start_solver(
        self,
        case_dir,
        cores: int = 1,
        mode="1D",
        intent: ExecutionIntent = ExecutionIntent.FRESH_FULL_PIPELINE,
    ):
        """Start solver with monitoring"""
        self.status_bar.set_status("Solver Running...", "#3498db")
        self.status_bar.set_progress(0)
        self.status_bar.start_et_timing()
        
        log_path = os.path.join(case_dir, "log.blastFoam")
        self.tab_jotter.start_monitoring(log_path)
        
        if mode == "3D" and isinstance(self.tabs.currentWidget(), TabGeneral3D):
            self.view_timer.start(1000)
        
        self.runner = SolverRunner(
            case_dir,
            self.openfoam_bashrc,
            project_root=self.project_root,
            cores=cores,
            intent=intent,
        )
        self.runner.data_signal.connect(lambda p, t, s, dt: self.on_new_data(p, t, s, dt, mode))
        self.runner.status_signal.connect(lambda s: self.status_bar.set_status(s, "#3498db"))
        self.runner.progress_signal.connect(self.status_bar.set_progress)
        self.runner.finished_signal.connect(self.on_simulation_finished)
        
        self.runner.start()
    
    def on_stop_request(self):
        """Handle stop/interrupt request"""
        if self.runner:
            self.runner.stop()
            self.status_bar.stop_et_timing()
            self.status_bar.set_status("Interrupted", "#e67e22")
        self.view_timer.stop()
    
    def on_new_data(self, pressures, sim_time_s, step_n, dt_val, mode="1D"):
        """Handle new simulation data; retain prior stage metrics when updating another mode."""
        if mode == "1D":
            self.status_bar.update_1d(step=step_n, tt=sim_time_s, dt=dt_val)
            if self.tabs.currentWidget() == self.tab_1d:
                self.tab_1d.update_graph(pressures, sim_time_s)
        elif mode == "2D":
            # 2D runner path is not implemented yet; keep API wired for future use.
            self.status_bar.update_2d(step=step_n, tt=sim_time_s, dt=dt_val)
        elif mode == "3D":
            self.status_bar.update_3d(step=step_n, tt=sim_time_s, dt=dt_val)
    
    def check_3d_updates(self):
        """Check for 3D viewport updates"""
        if self.tabs.currentWidget() == self.tab_3d:
            self.tab_3d.check_mesh_update()
    
    def on_simulation_finished(self, success):
        """Handle simulation completion"""
        self.view_timer.stop()
        self.tab_jotter.stop_monitoring()
        self.runner = None
        self.status_bar.stop_et_timing()
        
        if success:
            self.status_bar.set_progress(100)
            self.status_bar.set_status("Done", "#2ecc71")
            if self.tabs.currentWidget() == self.tab_3d:
                self.tab_3d.viewer.refresh_view()
        else:
            if "Interrupted" not in self.status_bar.lbl_status.text():
                self.status_bar.set_status("Stopped/Failed", "#e74c3c")

    def _on_3d_initial_dt_changed(self, dt_val):
        """Initial Δt is shown in General 3D Simulation Control (not the status bar)."""
        if hasattr(self.tab_3d, "lbl_initial_dt_display"):
            if dt_val is None:
                self.tab_3d.lbl_initial_dt_display.setText("Initial Δt: — s")
            else:
                self.tab_3d.lbl_initial_dt_display.setText(f"Initial Δt: {dt_val:g} s")


def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = BlastFoamApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
