"""
Cylindrical 2D tab - same 2-column resizable layout as 1D/3D.
Left: Input Parameters (top) + Info Panel (bottom).
Right: Viewport (top) + Execution Control (bottom).
Placeholder content until 2D simulation is implemented.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QScrollArea,
    QFrame, QTabWidget, QGroupBox, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt

from ui_metrics import (
    COMPUTATIONAL_LEFT_PANEL_WIDTH,
    COMPUTATIONAL_LEFT_PANEL_MIN,
    EXECUTION_AREA_MIN_HEIGHT,
    EXECUTION_AREA_PREFERRED_HEIGHT,
)


class Tab2D(QWidget):
    """2D simulation tab with consistent 2-column layout (placeholder)."""

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)

        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)
        placeholder_input = QLabel("2D Input Parameters\n(not yet implemented)")
        placeholder_input.setAlignment(Qt.AlignCenter)
        placeholder_input.setStyleSheet("color: #7f8c8d; padding: 20px;")
        input_layout.addWidget(placeholder_input)
        input_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(input_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(COMPUTATIONAL_LEFT_PANEL_MIN)
        left_layout.addWidget(scroll, stretch=1)
        self._left_setup_scroll = scroll

        info_frm = QFrame()
        info_frm.setStyleSheet("background:#eef2f6; border:1px solid #c7d0da; border-radius: 4px;")
        il = QVBoxLayout(info_frm)
        lbl_info = QLabel("Info: —")
        lbl_info.setStyleSheet("font-weight: bold; font-size: 11pt; color: #333;")
        il.addWidget(lbl_info)
        left_layout.addWidget(info_frm)

        splitter.addWidget(left_column)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._right_v_splitter = QSplitter(Qt.Vertical)
        self._right_v_splitter.setChildrenCollapsible(False)

        viewport_placeholder = QLabel("2D Viewport\n(not yet implemented)")
        viewport_placeholder.setAlignment(Qt.AlignCenter)
        viewport_placeholder.setStyleSheet(
            "background: #ecf0f1; color: #7f8c8d; font-size: 14pt; padding: 40px;"
        )
        viewport_placeholder.setMinimumHeight(120)

        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.setMinimumHeight(EXECUTION_AREA_MIN_HEIGHT)
        self.ctrl_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        exec_widget = QWidget()
        exec_layout = QHBoxLayout(exec_widget)
        exec_layout.setContentsMargins(8, 8, 8, 8)
        g = QGroupBox("Simulation Control")
        gh = QHBoxLayout(g)
        self.btn_run = QPushButton("▶ Run Simulation")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setMinimumWidth(180)
        self.btn_run.setStyleSheet(
            "background-color: #2ecc71; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;"
        )
        self.btn_run.setEnabled(False)
        self.btn_run.setToolTip("2D solver is not yet implemented.")
        self.btn_stop = QPushButton("⏸ Interrupt")
        self.btn_stop.setFixedHeight(50)
        self.btn_stop.setMinimumWidth(140)
        self.btn_stop.setStyleSheet(
            "background-color: #e67e22; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("2D solver is not yet implemented.")
        gh.addWidget(self.btn_run)
        gh.addSpacing(20)
        gh.addWidget(self.btn_stop)
        exec_layout.addWidget(g)
        exec_layout.addStretch()
        exec_scroll = QScrollArea()
        exec_scroll.setWidgetResizable(True)
        exec_scroll.setFrameShape(QFrame.NoFrame)
        exec_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        exec_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        exec_scroll.setWidget(exec_widget)
        self._exec_scroll = exec_scroll
        self.ctrl_tabs.addTab(exec_scroll, "Execution Controls")

        self._right_v_splitter.addWidget(viewport_placeholder)
        self._right_v_splitter.addWidget(self.ctrl_tabs)
        self._right_v_splitter.setStretchFactor(0, 1)
        self._right_v_splitter.setStretchFactor(1, 0)
        self._right_v_splitter.setSizes([800, EXECUTION_AREA_PREFERRED_HEIGHT])
        self._2d_exec_splitter_sizes = list(self._right_v_splitter.sizes())
        self._right_v_splitter.splitterMoved.connect(self._on_2d_exec_splitter_moved)

        right_layout.addWidget(self._right_v_splitter)
        splitter.addWidget(right_column)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        left_w = COMPUTATIONAL_LEFT_PANEL_WIDTH
        splitter.setSizes([left_w, max(400, 1200 - left_w)])
        self._main_splitter = splitter
        main_layout.addWidget(splitter)

    def _on_2d_exec_splitter_moved(self, _pos: int = 0, _index: int = 0) -> None:
        self._2d_exec_splitter_sizes = list(self._right_v_splitter.sizes())

    def get_computational_left_width(self) -> int:
        sizes = self._main_splitter.sizes()
        return int(sizes[0]) if sizes else COMPUTATIONAL_LEFT_PANEL_WIDTH

    def set_computational_left_width(self, width: int) -> None:
        width = max(COMPUTATIONAL_LEFT_PANEL_MIN, int(width))
        total = sum(self._main_splitter.sizes()) or (width + 800)
        self._main_splitter.setSizes([width, max(50, total - width)])
