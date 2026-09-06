"""Physical-quantity to native OpenFOAM field resolution."""

from __future__ import annotations

import unittest

from models import SOURCE_MODEL_IG, SOURCE_MODEL_JWL
from physical_fields import (
    DENSITY,
    EXPLOSIVE_FRACTION,
    PRESSURE,
    REACTION_PROGRESS,
    VELOCITY,
    foam_field,
    foam_fields,
    quantity_available,
    remap_field_names,
)


class SchemaTests(unittest.TestCase):
    def test_shared_primitives_keep_native_names(self):
        for model in (SOURCE_MODEL_JWL, SOURCE_MODEL_IG):
            self.assertEqual(foam_field(PRESSURE, model), "p")
            self.assertEqual(foam_field(VELOCITY, model), "U")

    def test_ig_uses_unsuffixed_rho_and_has_no_phase_fields(self):
        self.assertEqual(foam_field(DENSITY, SOURCE_MODEL_IG), "rho")
        self.assertIsNone(foam_field(EXPLOSIVE_FRACTION, SOURCE_MODEL_IG))
        self.assertIsNone(foam_field(REACTION_PROGRESS, SOURCE_MODEL_IG))
        self.assertFalse(quantity_available(EXPLOSIVE_FRACTION, SOURCE_MODEL_IG))

    def test_jwl_keeps_native_two_phase_fields(self):
        self.assertEqual(foam_field(DENSITY, SOURCE_MODEL_JWL), "rho.air")
        self.assertEqual(foam_field(EXPLOSIVE_FRACTION, SOURCE_MODEL_JWL), "alpha.c4")
        self.assertEqual(foam_field(REACTION_PROGRESS, SOURCE_MODEL_JWL), "lambda.c4")

    def test_ig_remap_list_does_not_invent_alpha_c4(self):
        self.assertNotIn("alpha.c4", remap_field_names(SOURCE_MODEL_IG))
        self.assertIn("alpha.c4", remap_field_names(SOURCE_MODEL_JWL))

    def test_probe_resolution_drops_missing_quantities(self):
        self.assertEqual(
            foam_fields((PRESSURE, EXPLOSIVE_FRACTION), SOURCE_MODEL_IG),
            ("p",),
        )


if __name__ == "__main__":
    unittest.main()
