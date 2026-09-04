"""UFC Figures 2-9 / 2-10 ground-surface applicability and interpolation."""
from __future__ import annotations

import math
import unittest

from validation import ufc_ground
from validation.ufc_data import load_json
from validation.ufc_units import cube_root, english_scaled_to_si


class UfcGroundTests(unittest.TestCase):
    def test_above_ground_is_na(self):
        ev = ufc_ground.lookup(
            ufc_ground.FIGURE_PRESSURE,
            ground_range_m=2.0,
            hob_m=1.0,
            mass_kg=1.0,
            observer_z_m=0.5,
            z_ground_m=0.0,
        )
        self.assertFalse(ev.ok)
        self.assertIn("reflecting surface", ev.unavailable_reason.lower())

    def test_known_point_on_labeled_curve(self):
        data = load_json("ufc_3_340_02_fig_2_09.json")
        curve = next(c for c in data["curves"] if c.get("hc_published") == 0.3)
        npts = data["npts"]
        amax = curve["alpha_max_deg"]
        alpha = amax * 10 / (npts - 1)
        y = curve["y_published"][10]
        hob = english_scaled_to_si(0.3)  # W = 1 kg
        r = hob * math.tan(math.radians(alpha))
        ev = ufc_ground.lookup(
            ufc_ground.FIGURE_PRESSURE,
            ground_range_m=r,
            hob_m=hob,
            mass_kg=1.0,
            observer_z_m=0.0,
            z_ground_m=0.0,
        )
        self.assertTrue(ev.ok)
        self.assertAlmostEqual(ev.value_si / 6894.757293168361, y, places=6)
        self.assertIn("Figure 2-9", ev.figure)

    def test_no_hc_extrapolation(self):
        ev = ufc_ground.lookup(
            ufc_ground.FIGURE_PRESSURE,
            ground_range_m=1.0,
            hob_m=english_scaled_to_si(0.05),
            mass_kg=1.0,
            observer_z_m=0.0,
        )
        self.assertFalse(ev.ok)
        self.assertIn("N/A", ev.unavailable_reason)

    def test_impulse_scales_with_w_third(self):
        hob1 = english_scaled_to_si(0.8)
        r = hob1 * math.tan(math.radians(20.0))
        a = ufc_ground.lookup(
            ufc_ground.FIGURE_IMPULSE,
            ground_range_m=r,
            hob_m=hob1,
            mass_kg=1.0,
            observer_z_m=0.0,
        )
        hob8 = english_scaled_to_si(0.8) * cube_root(8.0)
        r8 = hob8 * math.tan(math.radians(20.0))
        b = ufc_ground.lookup(
            ufc_ground.FIGURE_IMPULSE,
            ground_range_m=r8,
            hob_m=hob8,
            mass_kg=8.0,
            observer_z_m=0.0,
        )
        self.assertTrue(a.ok and b.ok)
        self.assertAlmostEqual(b.value_si / a.value_si, 2.0, places=5)

    def test_angle_too_large_is_na(self):
        hob = english_scaled_to_si(3.0)
        ev = ufc_ground.lookup(
            ufc_ground.FIGURE_PRESSURE,
            ground_range_m=hob * math.tan(math.radians(89.5)),
            hob_m=hob,
            mass_kg=1.0,
            observer_z_m=0.0,
        )
        self.assertFalse(ev.ok)


if __name__ == "__main__":
    unittest.main()
