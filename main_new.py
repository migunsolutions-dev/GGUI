"""
BlastFoam GUI Manager - Refactored UI
Main application window with multi-panel layout following specification.
"""
import sys
import os
import subprocess
from dataclasses import asdict
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QProgressBar, QMessageBox, QToolBar, QAction,
    QSplitter, QScrollArea, QGroupBox, QFormLayout, QStatusBar, QFileDialog,
    QDialog, QTextEdit, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

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
from startup_capture_guard import evaluate_unsafe_capture
from project_io import (
    PROJECT_SUFFIX,
    ProjectFormatError,
    build_project,
    read_project,
    write_project_atomic,
)
from viewer_widget import ObstacleItem
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
    """Bottom full-width status bar with segments for 1D/2D/3D"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("background-color: #34495e; border-top: 2px solid #2c3e50;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(20)
        
        label_style = "color: white; font-family: 'Consolas', monospace; font-size: 10pt; font-weight: bold;"
        
        # 1D Section
        self.lbl_1d_step = QLabel("Step–1D: ——")
        self.lbl_1d_tt = QLabel("Tt–1D: ——")
        self.lbl_1d_dt = QLabel("DT–1D: ——")
        
        # 2D Section (placeholder for future)
        self.lbl_2d_step = QLabel("Step–2D: ——")
        self.lbl_2d_tt = QLabel("Tt–2D: ——")
        self.lbl_2d_dt = QLabel("DT–2D: ——")
        
        # 3D Section
        self.lbl_3d_step = QLabel("Step–3D: ——")
        self.lbl_3d_tt = QLabel("Tt–3D: ——")
        self.lbl_3d_dt = QLabel("DT–3D: ——")
        self.lbl_3d_initial_dt = QLabel("Initial dt: —")
        self.lbl_3d_et = QLabel("ET: ——")
        
        # General status
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 11pt;")
        
        
        # Apply styles
        for lbl in [self.lbl_1d_step, self.lbl_1d_tt, self.lbl_1d_dt,
                    self.lbl_2d_step, self.lbl_2d_tt, self.lbl_2d_dt,
                    self.lbl_3d_step, self.lbl_3d_tt, self.lbl_3d_dt, self.lbl_3d_initial_dt, self.lbl_3d_et]:
            lbl.setStyleSheet(label_style)
        
        # Layout
        layout.addWidget(self.lbl_1d_step)
        layout.addWidget(self.lbl_1d_tt)
        layout.addWidget(self.lbl_1d_dt)
        layout.addWidget(QLabel("│").setStyleSheet("color: #7f8c8d;") or QLabel("│"))
        layout.addWidget(self.lbl_2d_step)
        layout.addWidget(self.lbl_2d_tt)
        layout.addWidget(self.lbl_2d_dt)
        layout.addWidget(QLabel("│").setStyleSheet("color: #7f8c8d;") or QLabel("│"))
        layout.addWidget(self.lbl_3d_step)
        layout.addWidget(self.lbl_3d_tt)
        layout.addWidget(self.lbl_3d_dt)
        layout.addWidget(self.lbl_3d_initial_dt)
        layout.addWidget(self.lbl_3d_et)
        layout.addStretch()
        layout.addWidget(self.lbl_status)
    
    def update_1d(self, step=None, tt=None, dt=None):
        if step is not None:
            self.lbl_1d_step.setText(f"Step–1D: {step}")
        if tt is not None:
            self.lbl_1d_tt.setText(f"Tt–1D: {tt:.5f}")
        if dt is not None:
            self.lbl_1d_dt.setText(f"DT–1D: {dt:.2e}")
    
    def update_3d(self, step=None, tt=None, dt=None, et=None):
        if step is not None:
            self.lbl_3d_step.setText(f"Step–3D: {step}")
        if tt is not None:
            self.lbl_3d_tt.setText(f"Tt–3D: {tt:.5f}")
        if dt is not None:
            self.lbl_3d_dt.setText(f"DT–3D: {dt:.2e}")
        if et is not None:
            self.lbl_3d_et.setText(f"ET: {et:.2f}s")

    def set_3d_initial_dt(self, dt_val):
        """Set the calculated initial dt in the 3D segment (from Execution parameters)."""
        self.lbl_3d_initial_dt.setText(f"Initial dt: {dt_val:.2e}")
    
    def set_status(self, text, color="white"):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11pt;")
    
    def set_progress(self, value):
        pass  # Progress bar removed


class BlastFoamApp(QMainWindow):
    """Main application window with refactored UI layout"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BlastFoam GUI Manager - v4.0 (Refactored UI)")
        self.setGeometry(50, 50, 800, 1000)
        
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
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        
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
        self.tab_3d.initial_dt_changed.connect(self.status_bar.set_3d_initial_dt)
        self.tab_3d._update_calculated_dt_label()
    
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
            inputs = project["inputs"]
            data = asdict(inputs)
            data["charge_radius"] = inputs.cylinder_radius
            self.tab_3d.set_case_inputs(data)
            saved_obstacles = project["gui_state"].get("obstacles")
            if isinstance(saved_obstacles, list):
                self.tab_3d.obstacles = [
                    ObstacleItem(
                        bool(item.get("enabled", True)),
                        str(item["path"]),
                        float(item.get("scale", 1.0)),
                        float(item.get("ox", 0.0)),
                        float(item.get("oy", 0.0)),
                        float(item.get("oz", 0.0)),
                    )
                    for item in saved_obstacles
                    if isinstance(item, dict) and item.get("path")
                ]
            else:
                self.tab_3d.obstacles = [
                    ObstacleItem(
                        True,
                        obstacle.stl_path,
                        obstacle.scale,
                        obstacle.offset_x,
                        obstacle.offset_y,
                        obstacle.offset_z,
                    )
                    for obstacle in inputs.obstacles
                ]
            self.tab_3d._refresh_table()
            self.probes_model.load_dict(project["probes"])
            self.tab_3d.load_project_gui_state(project["gui_state"])
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
            payload = build_project(
                self.tab_3d.get_case_inputs(),
                probes=self.probes_model.to_dict(),
                gui_state={
                    "selected_primary_tab": "General 3D",
                    "sections": [asdict(section) for section in self.tab_3d.sections],
                    "obstacles": [asdict(obstacle) for obstacle in self.tab_3d.obstacles],
                },
            )
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
                if not getattr(inputs, "remap_enabled", False):
                    guard = evaluate_unsafe_capture(inputs)
                    if not guard.safe:
                        QMessageBox.critical(
                            self,
                            "Unsafe charge capture",
                            "Initialization is blocked because no applied internal seed or "
                            "outer refinement band protects capture, and the aligned base mesh "
                            "has no cell centre inside the physical charge.\n\n"
                            "Choose one remedy without changing charge mass:\n"
                            "• reduce the base cell size;\n"
                            "• enable Dyn Mesh and select an internal charge refinement level greater than zero; or\n"
                            "• deliberately enable the advanced outer refinement band.",
                        )
                        self.status_bar.set_status("Init blocked", "#e74c3c")
                        return
                self.status_bar.set_status("Generating 3D Case...", "#f39c12")
                QApplication.processEvents()

                prefix = "Case_3D"
                case_name = self.service.make_case_name(prefix)
                case_dir = self.service.generate_case(case_name, inputs)
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
                if not use_remap:
                    mode_path = os.path.join(case_dir, "case_init_mode.json")
                    try:
                        import json
                        with open(mode_path, "r", encoding="utf-8") as f:
                            mode = json.load(f)
                        if get_charge_cell_count and os.path.isfile(os.path.join(case_dir, "0", "alpha.c4")):
                            charge_cells, _ = get_charge_cell_count(case_dir, "0", 0.5)
                            mode["cells_inside_charge"] = charge_cells
                        mode["retries_used"] = retries_used
                        mode["set_cmd_actual"] = set_cmd_actual
                        with open(mode_path, "w", encoding="utf-8") as f:
                            json.dump(mode, f, indent=2)
                    except (OSError, ValueError, KeyError):
                        pass
                final_mode = mode if not use_remap else None
                
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
            self.status_bar.set_status("Interrupted", "#e67e22")
        self.view_timer.stop()
    
    def on_new_data(self, pressures, sim_time_s, step_n, dt_val, mode="1D"):
        """Handle new simulation data"""
        if mode == "1D":
            self.status_bar.update_1d(step=step_n, tt=sim_time_s, dt=dt_val)
            if self.tabs.currentWidget() == self.tab_1d:
                self.tab_1d.update_graph(pressures, sim_time_s)
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
        
        if success:
            self.status_bar.set_progress(100)
            self.status_bar.set_status("Done", "#2ecc71")
            if self.tabs.currentWidget() == self.tab_3d:
                self.tab_3d.viewer.refresh_view()
        else:
            if "Interrupted" not in self.status_bar.lbl_status.text():
                self.status_bar.set_status("Stopped/Failed", "#e74c3c")


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
