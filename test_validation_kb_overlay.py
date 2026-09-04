"""Per-series UFC/KB overlay: mixed 1D spherical and 2D hemispherical."""
from __future__ import annotations

import unittest

from validation import kb_overlay
from validation import ufc_airblast as ufc_ab


class KbOverlayTests(unittest.TestCase):
    def test_mixed_1d_spherical_and_2d_hemispherical_use_separate_references(self):
        mass = 1.0
        range_m = 1.0
        sph = kb_overlay.OverlaySample(
            point_id="VAL_1D_001",
            dim="1d",
            mass_kg=mass,
            burst=ufc_ab.BURST_SPHERICAL,
            figure=ufc_ab.figure_id(ufc_ab.BURST_SPHERICAL),
            reference_source=kb_overlay.SOURCE_UFC,
            range_m=range_m,
            scaled_z=ufc_ab.scaled_distance(range_m, mass),
            bf_peak=2.0e5,
            bf_impulse=50.0,
            comparable=True,
            kind="bf",
        )
        hemi = kb_overlay.OverlaySample(
            point_id="VAL_2D_001",
            dim="2d",
            mass_kg=mass,
            burst=ufc_ab.BURST_HEMISPHERICAL,
            figure=ufc_ab.figure_id(ufc_ab.BURST_HEMISPHERICAL),
            reference_source=kb_overlay.SOURCE_UFC,
            range_m=range_m,
            scaled_z=ufc_ab.scaled_distance(range_m, mass),
            bf_peak=2.0e5,
            bf_impulse=50.0,
            comparable=True,
            kind="bf",
        )
        groups = kb_overlay.group_samples((sph, hemi))
        self.assertEqual(len(groups), 2)
        self.assertTrue(kb_overlay.mixed_references(groups))
        ev_sph = kb_overlay.evaluate_sample(sph)
        ev_hemi = kb_overlay.evaluate_sample(hemi)
        self.assertIsNotNone(ev_sph.ref_peak)
        self.assertIsNotNone(ev_hemi.ref_peak)
        self.assertNotAlmostEqual(ev_sph.ref_peak, ev_hemi.ref_peak)
        global_hemi = ufc_ab.evaluate(
            ufc_ab.QUANTITY_PEAK_PRESSURE,
            range_m=range_m,
            mass_kg=mass,
            burst_type=ufc_ab.BURST_HEMISPHERICAL,
        )
        self.assertAlmostEqual(ev_hemi.ref_peak, global_hemi.value_si)
        self.assertNotAlmostEqual(ev_sph.ref_peak, global_hemi.value_si)
        self.assertIsNotNone(ev_sph.error_peak_pct)
        self.assertIsNotNone(ev_hemi.error_peak_pct)

    def test_incompatible_or_invalid_sample_has_no_error_percent(self):
        sample = kb_overlay.OverlaySample(
            point_id="VAL_2D_001",
            dim="2d",
            mass_kg=1.0,
            burst=ufc_ab.BURST_HEMISPHERICAL,
            figure="2-15",
            reference_source=kb_overlay.SOURCE_UFC,
            range_m=1.0,
            scaled_z=1.0,
            bf_peak=2.0e5,
            comparable=False,
            validity_reason="Positive phase is incomplete; UFC/KB comparison is N/A.",
            kind="invalid",
        )
        ev = kb_overlay.evaluate_sample(sample)
        self.assertIsNone(ev.error_peak_pct)
        self.assertIsNone(ev.error_impulse_pct)


if __name__ == "__main__":
    unittest.main()
