"""1D source-model radios: JWL default, IG persists, JWL editor disabled on IG."""

from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication

from models import SOURCE_MODEL_IG, SOURCE_MODEL_JWL
from tab_1d import Tab1D


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class Tab1DSourceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_jwl_is_the_initial_default(self):
        tab = Tab1D()
        self.assertEqual(tab.selected_source_model(), SOURCE_MODEL_JWL)
        self.assertTrue(tab.radio_jwl.isChecked())
        inputs = tab.get_case_inputs()
        self.assertEqual(inputs.source_model, SOURCE_MODEL_JWL)

    def test_ig_round_trips_through_get_and_set(self):
        tab = Tab1D()
        tab.radio_ig.setChecked(True)
        data = tab.get_case_inputs().__dict__
        self.assertEqual(data["source_model"], SOURCE_MODEL_IG)
        other = Tab1D()
        other.set_case_inputs(data)
        self.assertEqual(other.selected_source_model(), SOURCE_MODEL_IG)
        self.assertTrue(other.radio_ig.isChecked())

    def test_ig_disables_jwl_coefficient_editor(self):
        tab = Tab1D()
        tab.combo_comp.setCurrentText("Custom")
        tab.radio_jwl.setChecked(True)
        tab.on_source_model_changed()
        self.assertTrue(tab.btn_edit_comp.isEnabled())
        tab.radio_ig.setChecked(True)
        tab.on_source_model_changed()
        self.assertFalse(tab.btn_edit_comp.isEnabled())


if __name__ == "__main__":
    unittest.main()
