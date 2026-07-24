"""Shared UI presentation metrics for computational tabs.

Logical Qt sizes only — do not impose top-level window minimum widths.
"""
from __future__ import annotations

# Default first-show window size (NOT a hard minimum — window remains shrinkable).
DEFAULT_WINDOW_WIDTH = 1685
DEFAULT_WINDOW_HEIGHT = 1060
DEFAULT_WINDOW_WIDTH_TOLERANCE = 20
DEFAULT_WINDOW_HEIGHT_TOLERANCE = 40

# Opening left-panel width shared by Spherical–1D, Cylindrical–2D, General 3D.
COMPUTATIONAL_LEFT_PANEL_WIDTH = 450
COMPUTATIONAL_LEFT_PANEL_TOLERANCE = 10
COMPUTATIONAL_LEFT_PANEL_MIN = 320

# Fixed General 3D Info panel (bottom of left column).
INFO_PANEL_HEIGHT_MIN = 120
INFO_PANEL_HEIGHT_MAX = 150
INFO_PANEL_HEIGHT = 140

# 1D / 2D lower execution region (vertical splitter allocation).
EXECUTION_AREA_MIN_HEIGHT = 180
EXECUTION_AREA_PREFERRED_HEIGHT = 230

# Readable status-bar fonts (fixed; never scaled down with window width).
# Metrics: single non-wrapping row at 9 pt monospace. Ready/Running stays 11 pt.
STATUS_METRICS_POINT_SIZE = 9
STATUS_READY_POINT_SIZE = 11
STATUS_FONT_MIN_POINT_SIZE = 9

# Representative full mode-group string for reserved label width (monospace).
STATUS_REP_MODE_GROUP = "3D: Step=12345678  Tt=1.234567e-04  Δt=1.234e-07"
STATUS_REP_ET = "ET=12345.6 s"

# Primary action buttons (1D Run / Interrupt, etc.). Point size — not px.
ACTION_BUTTON_FONT_PT = 10
# Group titles in computational lower panels (match ordinary label hierarchy).
GROUP_TITLE_FONT_PT = 10

# Form / group presentation (aligned with current General 3D Model Setup scale).
GROUP_MARGINS = (4, 4, 4, 4)
FORM_ROW_SPACING = 5
PANEL_PADDING = 4
CONTROL_MAX_WIDTH_DEFAULT = 144
LABEL_COLUMN_WIDTH = 100

SECONDARY_INFO_STYLE = "font-size: 9pt; color: #555;"
WARNING_STYLE = "color: #c0392b; font-size: 9pt; font-weight: bold;"
INFO_ROW_STYLE = "font-size: 9pt; color: #333; background: transparent; border: none;"
INFO_TITLE_STYLE = "font-weight: bold; font-size: 9pt; color: #333; background: transparent; border: none;"
