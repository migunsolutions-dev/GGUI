"""1D VAL radius → probes1d mapping (no nearest-neighbour aliasing)."""
from __future__ import annotations

import math
import os
import tempfile
import unittest

from generator_1d import Generator1D
from models import CaseInputs1D, RecommendedParams1D
from validation.map_1d import (
    KIND_EXACT,
    KIND_INTERP,
    KIND_NONE,
    map_radius,
    mapped_peak_impulse,
    merge_radii,
    radii_close,
)
from validation.probes import parse_probe_history, peak_and_impulse, radii_from_locations
from validation.auto_points import plan_1d
from validation.current_run import RunSnapshot, default_display_dims, histories_available


def _rec() -> RecommendedParams1D:
    return RecommendedParams1D(
        r_min=1.0e-4,
        ignition_point=(0.0, 0.0, 0.0),
        ignition_radius=0.01,
        dt0=1.0e-8,
        maxCo=0.5,
        maxDeltaT=1.0e-5,
    )


def _inputs_1d() -> CaseInputs1D:
    return CaseInputs1D(
        radius=2.0,
        cell_size=0.05,
        p_atm=101325.0,
        t_atm=288.0,
        mass_kg=1.0,
        rho_charge=1601.0,
        energy_j_per_kg=4.52e6,
        material_props={"rho": 1601.0, "A": 1.0, "B": 1.0, "R1": 1.0, "R2": 1.0, "omega": 0.25, "E0": 4.52e6},
        max_cfl=0.5,
        end_time_s=1.0e-3,
        gauge_locations=((0.4, "UserG"),),
        n_probes=20,
    )


class Map1DTests(unittest.TestCase):
    def test_exact_match_not_nearest_neighbour(self):
        radii = [0.10, 0.20, 0.30]
        mapped = map_radius(radii, 0.20)
        self.assertEqual(mapped.kind, KIND_EXACT)
        self.assertEqual(mapped.index_lo, 1)

    def test_adjacent_targets_do_not_share_one_probe(self):
        radii = [0.193586, 0.233654]
        a = map_radius(radii, 0.184929)
        b = map_radius(radii, 0.203567)
        self.assertEqual(a.kind, KIND_NONE)
        self.assertEqual(b.kind, KIND_INTERP)
        self.assertEqual(b.index_lo, 0)
        self.assertEqual(b.index_hi, 1)
        c = map_radius([0.17355, 0.193586, 0.21362], 0.184929)
        d = map_radius([0.17355, 0.193586, 0.21362], 0.203567)
        self.assertEqual(c.kind, KIND_INTERP)
        self.assertEqual(d.kind, KIND_INTERP)
        self.assertNotEqual((c.index_lo, c.index_hi, c.weight), (d.index_lo, d.index_hi, d.weight))

    def test_interpolated_peak_differs_when_histories_differ(self):
        times = [0.0, 1.0, 2.0]
        p_lo = [101325.0, 201325.0, 101325.0]
        p_hi = [101325.0, 301325.0, 101325.0]
        radii = [1.0, 2.0]
        mapping = map_radius(radii, 1.5)
        self.assertEqual(mapping.kind, KIND_INTERP)
        peak, _impl, _t, p_blend = mapped_peak_impulse(mapping, times, [p_lo, p_hi], p_atm=101325.0)
        self.assertAlmostEqual(peak, 150000.0)
        peak_lo, _ = peak_and_impulse(times, p_lo, p_atm=101325.0)
        peak_hi, _ = peak_and_impulse(times, p_hi, p_atm=101325.0)
        self.assertNotAlmostEqual(peak, peak_lo)
        self.assertNotAlmostEqual(peak, peak_hi)

    def test_no_extrapolation(self):
        mapped = map_radius([0.2, 0.4], 0.1)
        self.assertEqual(mapped.kind, KIND_NONE)

    def test_merge_keeps_validation_radii(self):
        merged = merge_radii([0.1, 0.2, 0.3], [0.1849, 0.2036], r_lo=0.05, r_hi=0.5)
        self.assertTrue(any(radii_close(r, 0.1849) for r in merged))
        self.assertTrue(any(radii_close(r, 0.2036) for r in merged))

    def test_generator_inserts_exact_val_radii(self):
        with tempfile.TemporaryDirectory() as td:
            case = Generator1D(td).generate("one", _inputs_1d(), _rec())
            plan = plan_1d(mass_kg=1.0, domain_radius_m=2.0, cell_size=0.05)
            with open(os.path.join(case, "system", "controlDict"), encoding="utf-8") as handle:
                control = handle.read()
            self.assertTrue(plan.points)
            self.assertGreater(control.count("("), 10)
            # Euclidean radius of a written point should match a VAL radius.
            import re

            coords = re.findall(r"\(\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\)", control)
            radii = [math.sqrt(float(x) ** 2 + float(y) ** 2 + float(z) ** 2) for x, y, z in coords]
            hits = sum(1 for pt in plan.points if any(radii_close(r, pt.range_m) for r in radii))
            self.assertEqual(hits, len(plan.points))

    def test_real_case_aliasing_is_resolved_by_interpolation(self):
        case = r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\Case_1D_20260904_122553"
        p_path = os.path.join(case, "postProcessing", "probes1d", "0", "p")
        if not os.path.isfile(p_path):
            self.skipTest("completed 1D case not available")
        locs, times, cols = parse_probe_history(p_path)
        radii = radii_from_locations(locs, dim="1d")
        a = map_radius(radii, 0.18492889377505312)
        b = map_radius(radii, 0.20356741985267576)
        self.assertEqual(a.kind, KIND_INTERP)
        self.assertEqual(b.kind, KIND_INTERP)
        i_path = os.path.join(case, "postProcessing", "probes1d", "0", "impulse")
        impulse_cols = parse_probe_history(i_path)[2] if os.path.isfile(i_path) else None
        pa, ia, _, _ = mapped_peak_impulse(a, times, cols, impulse_cols, p_atm=101325.0)
        pb, ib, _, _ = mapped_peak_impulse(b, times, cols, impulse_cols, p_atm=101325.0)
        self.assertIsNotNone(pa)
        self.assertIsNotNone(pb)
        self.assertNotAlmostEqual(pa, pb, places=1)


class DisplayDefaultTests(unittest.TestCase):
    def test_idle_domain_does_not_select_other_dims(self):
        snap = RunSnapshot(live_mode="1d", domain_radius_1d=2.0, domain_radius_2d=1.5, domain_height_2d=1.5)
        self.assertEqual(default_display_dims(snap), {"1d"})
        self.assertFalse(histories_available(snap, "2d"))

    def test_live_2d_without_histories_is_preview_only(self):
        snap = RunSnapshot(live_mode="2d", domain_radius_2d=1.5, domain_height_2d=1.5)
        self.assertEqual(default_display_dims(snap), {"2d"})

    def test_no_live_mode_without_histories_is_empty(self):
        snap = RunSnapshot(domain_radius_2d=1.5, domain_height_2d=1.5)
        self.assertEqual(default_display_dims(snap), set())


if __name__ == "__main__":
    unittest.main()
