"""UFC 3-340-02 Figure 2-13 HOB reference: points, interpolation, no extrapolation."""
from __future__ import annotations

import unittest

from validation import ufc_hob
from validation.ufc_data import load_json
from validation.ufc_units import cube_root, english_scaled_to_si


class UfcFigure213Tests(unittest.TestCase):
    def setUp(self):
        self.data = load_json("ufc_3_340_02_fig_2_13.json")
        self.w = 8.0
        self.w13 = cube_root(self.w)

    def _physical(self, scaled_en: float) -> float:
        return english_scaled_to_si(scaled_en) * self.w13

    def test_published_hc_family_is_separate(self):
        self.assertEqual(ufc_hob.published_hc_english(), (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0))
        self.assertEqual(self.data["missing_curves"][0]["hc_published"], 7.0)
        self.assertEqual(self.data["data_kind"], "ufc_dplot_empirical")
        self.assertTrue(self.data["not_analytical"])

    def test_endpoint_on_hc_one_curve(self):
        x0 = self.data["x_published"][0]
        y0 = self.data["curves"][0]["y_published"][0]
        ev = ufc_hob.lookup_mach_stem_height(
            self._physical(x0), hob_m=self._physical(1.0), mass_kg=self.w
        )
        self.assertFalse(ev.unavailable_reason)
        self.assertAlmostEqual(ev.hm_m, self._physical(y0), places=9)
        self.assertAlmostEqual(ev.curve_hc_lo, 1.0)
        self.assertAlmostEqual(ev.curve_hc_hi, 1.0)

    def test_last_published_x_on_hc_six(self):
        x_end = self.data["x_published"][-1]
        y_end = self.data["curves"][-1]["y_published"][-1]
        ev = ufc_hob.lookup_mach_stem_height(
            self._physical(x_end), hob_m=self._physical(6.0), mass_kg=self.w
        )
        self.assertTrue(ev.hm_m is not None)
        self.assertAlmostEqual(ev.hm_m, self._physical(y_end), places=9)

    def test_linear_interp_along_a_curve(self):
        xs = self.data["x_published"]
        ys = self.data["curves"][0]["y_published"]
        x_mid = 0.5 * (xs[10] + xs[11])
        y_mid = 0.5 * (ys[10] + ys[11])
        ev = ufc_hob.lookup_mach_stem_height(
            self._physical(x_mid), hob_m=self._physical(1.0), mass_kg=self.w
        )
        self.assertAlmostEqual(ev.hm_m, self._physical(y_mid), places=9)

    def test_linear_interp_between_hc_curves(self):
        xs = self.data["x_published"]
        y_a = self.data["curves"][0]["y_published"][20]
        y_b = self.data["curves"][1]["y_published"][20]
        ev = ufc_hob.lookup_mach_stem_height(
            self._physical(xs[20]), hob_m=self._physical(1.25), mass_kg=self.w
        )
        self.assertAlmostEqual(ev.hm_m, self._physical(0.5 * (y_a + y_b)), places=9)
        self.assertIsNotNone(ev.band)

    def test_no_extrapolation_in_range_or_hc(self):
        xs = self.data["x_published"]
        low_r = ufc_hob.lookup_mach_stem_height(
            self._physical(xs[0] * 0.5), hob_m=self._physical(2.0), mass_kg=self.w
        )
        high_r = ufc_hob.lookup_mach_stem_height(
            self._physical(xs[-1] * 1.2), hob_m=self._physical(2.0), mass_kg=self.w
        )
        low_h = ufc_hob.lookup_mach_stem_height(
            self._physical(xs[10]), hob_m=self._physical(0.5), mass_kg=self.w
        )
        high_h = ufc_hob.lookup_mach_stem_height(
            self._physical(xs[10]), hob_m=self._physical(7.0), mass_kg=self.w
        )
        for ev in (low_r, high_r, low_h, high_h):
            self.assertIsNone(ev.hm_m)
            self.assertIn("N/A", ev.unavailable_reason)

    def test_reference_curve_uses_current_hc(self):
        xr, yr = ufc_hob.reference_curve(hob_m=self._physical(2.0), mass_kg=self.w)
        self.assertEqual(len(xr), 200)
        self.assertEqual(len(yr), 200)
        ev = ufc_hob.lookup_mach_stem_height(xr[40], hob_m=self._physical(2.0), mass_kg=self.w)
        self.assertAlmostEqual(ev.hm_m, yr[40], places=9)


if __name__ == "__main__":
    unittest.main()
