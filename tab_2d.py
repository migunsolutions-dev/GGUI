"""
Cylindrical 2D tab - same 2-column resizable layout as 1D/3D.
Left: Input Parameters (top) + Info Panel (bottom).
Right: Viewport (top) + Execution Control (bottom).
Placeholder content until 2D simulation is implemented.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QScrollArea,
    QGroupBox, QFormLayout, QFrame, QTabWidget
)
from PyQt5.QtCore import Qt


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

        # ===== LEFT COLUMN: Input Parameters (top) + Info Panel (bottom) =====
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
        scroll.setMinimumWidth(260)
        left_layout.addWidget(scroll, stretch=1)

        info_frm = QFrame()
        info_frm.setStyleSheet("background:#eef2f6; border:1px solid #c7d0da; border-radius: 4px;")
        il = QVBoxLayout(info_frm)
        lbl_info = QLabel("Info: —")
        lbl_info.setStyleSheet("font-weight: bold; font-size: 11pt; color: #333;")
        il.addWidget(lbl_info)
        left_layout.addWidget(info_frm)

        splitter.addWidget(left_column)

        # ===== RIGHT COLUMN: Viewport (top) + Execution Control (bottom) =====
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        viewport_placeholder = QLabel("2D Viewport\n(not yet implemented)")
        viewport_placeholder.setAlignment(Qt.AlignCenter)
        viewport_placeholder.setStyleSheet("background: #ecf0f1; color: #7f8c8d; font-size: 14pt; padding: 40px;")
        right_layout.addWidget(viewport_placeholder, stretch=1)

        ctrl_tabs = QTabWidget()
        ctrl_tabs.setMinimumHeight(120)
        ctrl_tabs.setMaximumHeight(160)
        exec_widget = QWidget()
        exec_layout = QHBoxLayout(exec_widget)
        exec_layout.addWidget(QLabel("Execution Controls (2D): not yet implemented"))
        exec_layout.addStretch()
        ctrl_tabs.addTab(exec_widget, "Execution Controls")
        right_layout.addWidget(ctrl_tabs)

        splitter.addWidget(right_column)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 800])
        main_layout.addWidget(splitter)
