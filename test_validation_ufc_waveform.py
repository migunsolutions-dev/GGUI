"""UFC Calc modified-Friedlander waveform (not CONWEP)."""
from __future__ import annotations

import unittest

from validation import conwep as conwep_engine
from validation import ufc_airblast as ufc
from validation import ufc_waveform
from validation.ufc_data import load_json
from validation.ufc_units import friedlander_impulse_factor


class UfcWaveformTests(unittest.TestCase):
    def test_not_labeled_conwep(self):
        wave = ufc_waveform.evaluate(
            range_m=1.0,
            mass_kg=1.0,
            burst_type=ufc.BURST_HEMISPHERICAL,
        )
        self.assertTrue(wave.ok)
        self.assertTrue(wave.citation.startswith("UFC Calc.xlsx"))
        self.assertIn("not a CONWEP source", wave.citation)

    def test_peak_at_arrival_zero_at_end(self):
        wave = ufc_waveform.evaluate(
            range_m=2.0,
            mass_kg=1.0,
            burst_type=ufc.BURST_SPHERICAL,
            n_points=101,
        )
        self.assertTrue(wave.ok)
        self.assertAlmostEqual(wave.overpressure_pa[0], wave.peak_pa, places=8)
        self.assertAlmostEqual(wave.overpressure_pa[-1], 0.0, places=8)
        self.assertAlmostEqual(wave.times_s[0], wave.arrival_s, places=12)

    def test_integral_matches_friedlander_identity(self):
        rows = load_json("ufc_3_340_02_fig_2_15.json")["rows"]
        z = rows[80][0]
        wave = ufc_waveform.evaluate(
            range_m=z,
            mass_kg=1.0,
            burst_type=ufc.BURST_HEMISPHERICAL,
            n_points=401,
        )
        pred = wave.peak_pa * wave.duration_s * friedlander_impulse_factor(wave.decay_b)
        self.assertAlmostEqual(wave.impulse_pa_s[-1] / pred, 1.0, places=3)
        self.assertAlmostEqual(pred / wave.impulse_table_pa_s, 1.0, places=6)

    def test_out_of_range_is_na(self):
        wave = ufc_waveform.evaluate(
            range_m=0.01,
            mass_kg=1.0,
            burst_type=ufc.BURST_SPHERICAL,
        )
        self.assertFalse(wave.ok)
        self.assertIn("N/A", wave.unavailable_reason)

    def test_conwep_waveform_remains_unavailable(self):
        result = conwep_engine.evaluate(range_m=1.0, mass_kg=1.0)
        self.assertIsNone(result.pressure_history)
        self.assertIsNone(conwep_engine.pressure_history())
        self.assertIn("N/A", result.waveform_reason)


if __name__ == "__main__":
    unittest.main()
