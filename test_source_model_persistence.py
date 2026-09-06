"""1D project JSON keeps the source-model discriminator, and legacy files stay JWL."""

from __future__ import annotations

import unittest
from dataclasses import asdict, replace

from models import SOURCE_MODEL_IG, SOURCE_MODEL_JWL
from project_io import ProjectFormatError, _case_inputs_1d_from_dict
from test_generator_1d import _inputs


class SourceModelProjectPersistenceTests(unittest.TestCase):
    def test_missing_source_model_loads_as_jwl(self):
        data = asdict(_inputs("Terminate"))
        data.pop("source_model", None)
        loaded = _case_inputs_1d_from_dict(data)
        self.assertEqual(loaded.source_model, SOURCE_MODEL_JWL)

    def test_ig_source_model_round_trips(self):
        data = asdict(replace(_inputs("Terminate"), source_model=SOURCE_MODEL_IG))
        loaded = _case_inputs_1d_from_dict(data)
        self.assertEqual(loaded.source_model, SOURCE_MODEL_IG)

    def test_unknown_source_model_is_rejected(self):
        data = asdict(_inputs("Terminate"))
        data["source_model"] = "IdealGas"
        with self.assertRaises(ProjectFormatError):
            _case_inputs_1d_from_dict(data)


if __name__ == "__main__":
    unittest.main()
