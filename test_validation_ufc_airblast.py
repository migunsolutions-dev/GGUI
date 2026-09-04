"""UFC Figures 2-7 / 2-15 airblast tables, distinct from Swisdak / CONWEP."""
from __future__ import annotations

import unittest

from validation import kingery_bulmash as kb
from validation import ufc_airblast as ufc
from validation.ufc_data import load_json
from validation.ufc_units import cube_root, english_scaled_to_si


class UfcAirblastTests(unittest.TestCase):
    def test_excel_row_reproduces_ps0(self):
        rows = load_json("ufc_3_340_02_fig_2_7.json")["rows"]
        z, _ta, _t0, ps0, _i, _b, _pr, _ir, _br = rows[0]
        ev = ufc.evaluate(
            ufc.QUANTITY_PEAK_PRESSURE,
            range_m=z,
            mass_kg=1.0,
            burst_type=ufc.BURST_SPHERICAL,
        )
        self.assertTrue(ev.ok)
        self.assertAlmostEqual(ev.value_si / 1000.0, ps0, places=9)
        self.assertIn("Figure 2-7", ev.figure)
        self.assertIn("DataSpherical", ev.sheet)
        self.assertNotIn("CONWEP", ev.citation)
        self.assertNotIn("Swisdak", ev.citation)

    def test_last_row_endpoint(self):
        rows = load_json("ufc_3_340_02_fig_2_7.json")["rows"]
        z, _ta, _t0, ps0, *_rest = rows[-1]
        ev = ufc.evaluate(
            ufc.QUANTITY_PEAK_PRESSURE,
            range_m=z * cube_root(8.0),
            mass_kg=8.0,
            burst_type=ufc.BURST_SPHERICAL,
        )
        self.assertTrue(ev.ok)
        self.assertAlmostEqual(ev.value_si / 1000.0, ps0, places=8)

    def test_linear_interp_in_z(self):
        rows = load_json("ufc_3_340_02_fig_2_15.json")["rows"]
        z0, z1 = rows[10][0], rows[11][0]
        p0, p1 = rows[10][3], rows[11][3]
        z_mid = 0.5 * (z0 + z1)
        ev = ufc.evaluate(
            ufc.QUANTITY_PEAK_PRESSURE,
            range_m=z_mid,
            mass_kg=1.0,
            burst_type=ufc.BURST_HEMISPHERICAL,
        )
        self.assertAlmostEqual(ev.value_si / 1000.0, 0.5 * (p0 + p1), places=8)

    def test_no_extrapolation(self):
        sph = load_json("ufc_3_340_02_fig_2_7.json")
        z_lo, z_hi = sph["valid_range_z_si"]
        low = ufc.evaluate(
            ufc.QUANTITY_PEAK_PRESSURE,
            range_m=0.5 * z_lo,
            mass_kg=1.0,
            burst_type=ufc.BURST_SPHERICAL,
        )
        high = ufc.evaluate(
            ufc.QUANTITY_PEAK_PRESSURE,
            range_m=1.2 * z_hi,
            mass_kg=1.0,
            burst_type=ufc.BURST_SPHERICAL,
        )
        self.assertFalse(low.ok)
        self.assertFalse(high.ok)
        self.assertIn("outside", low.unavailable_reason)
        self.assertIsNone(low.value_si)

    def test_spherical_does_not_use_hemispherical_table(self):
        z = 1.0
        sph = ufc.evaluate(
            ufc.QUANTITY_PEAK_PRESSURE,
            range_m=z,
            mass_kg=1.0,
            burst_type=ufc.BURST_SPHERICAL,
        )
        hemi = ufc.evaluate(
            ufc.QUANTITY_PEAK_PRESSURE,
            range_m=z,
            mass_kg=1.0,
            burst_type=ufc.BURST_HEMISPHERICAL,
        )
        self.assertTrue(sph.ok and hemi.ok)
        self.assertNotAlmostEqual(sph.value_si, hemi.value_si, places=3)
        self.assertIn("2-7", sph.figure)
        self.assertIn("2-15", hemi.figure)

    def test_kb_spherical_remains_unavailable(self):
        ev = kb.evaluate(
            kb.QUANTITY_PEAK_PRESSURE,
            range_m=1.0,
            mass_kg=1.0,
            burst_type=kb.BURST_SPHERICAL,
        )
        self.assertFalse(ev.ok)
        self.assertIn("ARBRL-TR-02555", ev.unavailable_reason)

    def test_swisdak_hemi_close_to_ufc_fig_2_15(self):
        z = 1.0
        sw = kb.evaluate(kb.QUANTITY_PEAK_PRESSURE, range_m=z, mass_kg=1.0)
        ufc_ev = ufc.evaluate(
            ufc.QUANTITY_PEAK_PRESSURE,
            range_m=z,
            mass_kg=1.0,
            burst_type=ufc.BURST_HEMISPHERICAL,
        )
        self.assertTrue(sw.ok and ufc_ev.ok)
        rel = abs(sw.value_si - ufc_ev.value_si) / ufc_ev.value_si
        self.assertLess(rel, 0.03)

    def test_fig_2_7_first_z_is_english_0_37(self):
        z_si = english_scaled_to_si(0.37)
        rows = load_json("ufc_3_340_02_fig_2_7.json")["rows"]
        self.assertAlmostEqual(rows[0][0], z_si, places=8)


if __name__ == "__main__":
    unittest.main()
