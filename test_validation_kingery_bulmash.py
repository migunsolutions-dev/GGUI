"""Swisdak 1994 Kingery-Bulmash engine: ranges, no extrapolation, citations."""
from __future__ import annotations

import math
import unittest

from validation import kingery_bulmash as kb


class SwisdakEngineTests(unittest.TestCase):
    def test_citation_metadata_on_success(self):
        ev = kb.evaluate(kb.QUANTITY_PEAK_PRESSURE, range_m=1.0, mass_kg=1.0)
        self.assertTrue(ev.ok)
        self.assertIn("ADA526744", ev.citation)
        self.assertEqual(ev.burst_type, kb.BURST_HEMISPHERICAL)
        self.assertEqual(ev.equation_id, kb.EQUATION)
        self.assertIn("entered charge mass", ev.mass_convention.lower())
        self.assertIsNotNone(ev.z_min)
        self.assertIsNotNone(ev.z_max)

    def test_z_one_uses_leading_coefficient(self):
        ev = kb.evaluate(kb.QUANTITY_PEAK_PRESSURE, range_m=1.0, mass_kg=1.0)
        expected_kpa = math.exp(7.2106)
        self.assertAlmostEqual(ev.value_si / 1000.0, expected_kpa, places=4)

    def test_no_extrapolation_below_or_above(self):
        low = kb.evaluate(kb.QUANTITY_PEAK_PRESSURE, range_m=0.05, mass_kg=1.0)
        high = kb.evaluate(kb.QUANTITY_PEAK_PRESSURE, range_m=200.0, mass_kg=1.0)
        self.assertFalse(low.ok)
        self.assertFalse(high.ok)
        self.assertIn("outside the published", low.unavailable_reason)
        self.assertIn("outside the published", high.unavailable_reason)
        self.assertIsNone(low.value_si)

    def test_breakpoint_uses_near_field_piece(self):
        at = kb.evaluate(kb.QUANTITY_PEAK_PRESSURE, range_m=2.9, mass_kg=1.0)
        self.assertTrue(at.ok)
        self.assertAlmostEqual(at.z_min, 0.2)
        self.assertAlmostEqual(at.z_max, 2.9)
        just_above = kb.evaluate(kb.QUANTITY_PEAK_PRESSURE, range_m=2.9001, mass_kg=1.0)
        self.assertTrue(just_above.ok)
        self.assertAlmostEqual(just_above.z_min, 2.9)

    def test_spherical_is_unavailable(self):
        ev = kb.evaluate(
            kb.QUANTITY_PEAK_PRESSURE,
            range_m=1.0,
            mass_kg=1.0,
            burst_type=kb.BURST_SPHERICAL,
        )
        self.assertFalse(ev.ok)
        self.assertIn("ARBRL-TR-02555", ev.unavailable_reason)
        xr, yr = kb.curve(kb.QUANTITY_PEAK_PRESSURE, mass_kg=1.0, burst_type=kb.BURST_SPHERICAL)
        self.assertEqual(xr, ())
        self.assertEqual(yr, ())

    def test_curve_stays_inside_published_z(self):
        zr, _vals = kb.curve_vs_z(kb.QUANTITY_PEAK_PRESSURE, mass_kg=8.0)
        self.assertGreater(len(zr), 10)
        self.assertGreaterEqual(min(zr), 0.2 - 1e-9)
        self.assertLessEqual(max(zr), 198.5 + 1e-9)

    def test_impulse_scales_with_w_third(self):
        a = kb.evaluate(kb.QUANTITY_INCIDENT_IMPULSE, range_m=1.0, mass_kg=1.0)
        b = kb.evaluate(kb.QUANTITY_INCIDENT_IMPULSE, range_m=2.0, mass_kg=8.0)
        self.assertTrue(a.ok and b.ok)
        self.assertAlmostEqual(a.z, b.z)
        self.assertAlmostEqual(b.value_si / a.value_si, 2.0, places=5)

    def test_unknown_quantity_is_na(self):
        ev = kb.evaluate("not-a-swisdak-quantity", range_m=1.0, mass_kg=1.0)
        self.assertFalse(ev.ok)
        self.assertIn("not in Swisdak 1994", ev.unavailable_reason)


if __name__ == "__main__":
    unittest.main()
