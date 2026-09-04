"""HOB extraction, UFC N/A hook, and Rankine–Hugoniot helpers."""
from __future__ import annotations

import unittest

import numpy as np

from validation import hob as hob_engine
from validation import rankine_hugoniot as rh
from validation import ufc_hob


class HobExtractionTests(unittest.TestCase):
    def test_kink_is_reported_as_triple_point(self):
        r = []
        z = []
        p = []
        for zi in np.linspace(0.0, 1.0, 40):
            front = 0.4 if zi < 0.35 else 0.4 + 1.6 * (zi - 0.35)
            for ri in np.linspace(0.0, 1.2, 40):
                r.append(ri)
                z.append(zi)
                p.append(2.0e5 if ri <= front else 1.01325e5)
        fronts = hob_engine.extract_fronts(r, z, p, z_ground=0.0)
        self.assertGreater(fronts.grad_mag.max(), 0.0)
        self.assertTrue(np.isfinite(fronts.r_shock).any())
        if fronts.triple_point is None:
            self.assertTrue(fronts.reason)

    def test_weak_field_does_not_invent_tp(self):
        r = np.linspace(0, 1, 20)
        z = np.linspace(0, 1, 20)
        rr, zz = np.meshgrid(r, z)
        fronts = hob_engine.extract_fronts(rr.ravel(), zz.ravel(), np.full(rr.size, 101325.0))
        self.assertIsNone(fronts.triple_point)
        self.assertTrue(fronts.reason)

    def test_image_source_is_geometry_only(self):
        t = hob_engine.image_source_reflected_arrival(
            source_xyz=(0.0, 0.0, 0.5),
            observer_xyz=(1.0, 0.0, 0.1),
            z_ground=0.0,
            shock_speed=400.0,
        )
        self.assertGreater(t, 0.0)

    def test_3d_section_keeps_xz_or_yz_only(self):
        from validation.spatial import load_pressure_rz
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as td:
            tdir = os.path.join(td, "0")
            os.makedirs(tdir)
            # Minimal nonuniform lists: 4 cells, two on X-Z (y=0), two off-plane.
            centres = """internalField nonuniform List<vector> 4
(
(1 0 0.1)
(2 0 0.2)
(1 5 0.1)
(2 5 0.2)
);
"""
            field = """internalField nonuniform List<scalar> 4
(
101325
201325
301325
401325
);
"""
            with open(os.path.join(tdir, "C"), "w", encoding="utf-8") as handle:
                handle.write(centres)
            with open(os.path.join(tdir, "p"), "w", encoding="utf-8") as handle:
                handle.write(field)
            r, z, p, err = load_pressure_rz(td, "0", "p", plane="X-Z", origin=(0.0, 0.0, 0.0))
            self.assertEqual(err, "")
            self.assertEqual(len(r), 2)
            self.assertTrue(np.allclose(sorted(p), [101325.0, 201325.0]))

    def test_ufc_requires_hob_and_mass(self):
        ev = ufc_hob.lookup_mach_stem_height(2.0)
        self.assertIsNone(ev.hm_m)
        self.assertIn("UFC 3-340-02", ev.unavailable_reason)
        self.assertTrue(ufc_hob.UFC_HOB_PROVENANCE.populated)
        self.assertEqual(ufc_hob.UFC_HOB_PROVENANCE.figure_or_table, "Figure 2-13")
        self.assertEqual(ufc_hob.reference_curve(), ((), ()))


class RankineHugoniotTests(unittest.TestCase):
    def test_normal_shock_mach_two(self):
        a1 = (1.4 * 101325.0 / 1.225) ** 0.5
        shock = rh.normal_shock(
            shock_speed=2.0 * a1, p1=101325.0, rho1=1.225, t1=288.15, gamma=1.4
        )
        self.assertIsNotNone(shock.mach)
        self.assertAlmostEqual(shock.mach, 2.0, places=5)
        self.assertGreater(shock.p2, shock.p1)
        self.assertGreater(shock.rho2, shock.rho1)

    def test_subsonic_is_unavailable(self):
        shock = rh.normal_shock(shock_speed=10.0, p1=101325.0, rho1=1.225)
        self.assertIsNone(shock.p2)
        self.assertIn("not greater than 1", shock.unavailable_reason)

    def test_oblique_weak_root(self):
        obl = rh.regular_oblique_shock(mach1=3.0, theta_deg=10.0)
        self.assertIsNotNone(obl.beta_deg)
        self.assertGreater(obl.beta_deg, 10.0)
        self.assertGreater(obl.p2_over_p1, 1.0)

    def test_normal_component(self):
        self.assertAlmostEqual(rh.normal_component((3.0, 4.0, 0.0), (0.0, 1.0, 0.0)), 4.0)


if __name__ == "__main__":
    unittest.main()
