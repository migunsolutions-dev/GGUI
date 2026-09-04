"""Create QApplication with OpenGL sharing enabled before any VTK import."""
from __future__ import annotations

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


def prepare_qt_application(argv=None):
    """Return the process QApplication, creating it with shared GL contexts.

    Must run before pyvista/pyvistaqt are imported. Sharing prevents Windows
    from aborting when a raster widget (1D graph, jotter) paints in a window
    that also hosts or previously hosted a VTK OpenGL widget.
    """
    existing = QApplication.instance()
    if existing is not None:
        return existing
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    return QApplication(sys.argv if argv is None else argv)
