"""CONWEP interface: KB-backed scalars only; waveform N/A."""
from __future__ import annotations

import unittest

from validation import conwep as conwep_engine
from validation import kingery_bulmash as kb


class ConwepInterfaceTests(unittest.TestCase):
    def test_waveform_is_unavailable(self):
        result = conwep_engine.evaluate(range_m=1.0, mass_kg=1.0)
        self.assertIsNone(result.pressure_history)
        self.assertIsNone(result.impulse_history)
        self.assertIsNone(conwep_engine.pressure_history())
        self.assertIsNone(conwep_engine.impulse_history())
        self.assertIn("N/A", result.waveform_reason)

    def test_scalars_are_kb_backed(self):
        result = conwep_engine.evaluate(
            range_m=1.0, mass_kg=1.0, pressure_type=conwep_engine.PRESSURE_INCIDENT
        )
        kb_p = kb.evaluate(kb.QUANTITY_PEAK_PRESSURE, range_m=1.0, mass_kg=1.0)
        kb_i = kb.evaluate(kb.QUANTITY_INCIDENT_IMPULSE, range_m=1.0, mass_kg=1.0)
        kb_a = kb.evaluate(kb.QUANTITY_ARRIVAL, range_m=1.0, mass_kg=1.0)
        kb_d = kb.evaluate(kb.QUANTITY_DURATION, range_m=1.0, mass_kg=1.0)
        self.assertEqual(result.peak_pressure.kb_quantity, kb.QUANTITY_PEAK_PRESSURE)
        self.assertAlmostEqual(result.peak_pressure.value_si, kb_p.value_si)
        self.assertAlmostEqual(result.positive_impulse.value_si, kb_i.value_si)
        self.assertAlmostEqual(result.arrival_time.value_si, kb_a.value_si)
        self.assertAlmostEqual(result.positive_duration.value_si, kb_d.value_si)
        self.assertIn("Kingery-Bulmash", result.peak_pressure.provenance)
        self.assertIn("ADA526744", result.peak_pressure.citation)

    def test_reflected_family_uses_reflected_kb(self):
        result = conwep_engine.evaluate(
            range_m=1.0, mass_kg=1.0, pressure_type=conwep_engine.PRESSURE_REFLECTED
        )
        kb_p = kb.evaluate(kb.QUANTITY_REFLECTED_PRESSURE, range_m=1.0, mass_kg=1.0)
        self.assertEqual(result.peak_pressure.kb_quantity, kb.QUANTITY_REFLECTED_PRESSURE)
        self.assertAlmostEqual(result.peak_pressure.value_si, kb_p.value_si)

    def test_out_of_range_scalar_is_na_not_extrapolated(self):
        result = conwep_engine.evaluate(range_m=0.01, mass_kg=1.0)
        self.assertIsNone(result.peak_pressure.value_si)
        self.assertIn("outside the published", result.peak_pressure.unavailable_reason)


if __name__ == "__main__":
    unittest.main()
