"""Qt bootstrap and 1D-path import isolation."""
from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent


def _toplevel_imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()

    def collect(nodes):
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    names.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Try):
                collect(node.body)
                for handler in node.handlers:
                    collect(handler.body)
                collect(node.orelse)
                collect(node.finalbody)
            elif isinstance(node, ast.If):
                collect(node.body)
                collect(node.orelse)

    collect(tree.body)
    return names


class ViewerWidgetImportIsolationTests(unittest.TestCase):
    def test_viewer_widget_does_not_import_vtk_at_module_level(self):
        imported = _toplevel_imported_modules(ROOT / "viewer_widget.py")
        self.assertNotIn("pyvista", imported)
        self.assertNotIn("pyvistaqt", imported)
        self.assertNotIn("vtk", imported)


class LogBufferFlushTests(unittest.TestCase):
    def test_flush_on_idle_interval_or_size(self):
        from tab_log import should_flush_log_buffer

        self.assertFalse(should_flush_log_buffer("", idle=False, elapsed_s=1.0))
        self.assertTrue(should_flush_log_buffer("x", idle=True, elapsed_s=0.0))
        self.assertTrue(should_flush_log_buffer("x", idle=False, elapsed_s=0.2))
        self.assertTrue(should_flush_log_buffer("y" * 9000, idle=False, elapsed_s=0.0))
        self.assertFalse(should_flush_log_buffer("x", idle=False, elapsed_s=0.01))


if __name__ == "__main__":
    unittest.main()
