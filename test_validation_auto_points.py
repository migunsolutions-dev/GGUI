"""Automatic Validation Point generation (Qt-free)."""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from dataclasses import replace

from generator_1d import Generator1D
from generator_2d import Generator2D
from models import CaseInputs1D, RecommendedParams1D
from models_2d import CaseInputs2D, ProbePoint2D
from validation.auto_points import (
    DEFAULT_LOGICAL_DPI_X,
    DEFAULT_PLOT_WIDTH_PX,
    MAX_POINTS,
    MIN_POINTS,
    TARGET_SPACING_MM,
    cache_key,
    log_spaced,
    marker_stride,
    pixels_per_mm,
    plan_1d,
    plan_2d,
    point_count,
    target_spacing_px,
)
from validation.sampling_io import LEGACY_NO_VALIDATION_HISTORIES, read_sampling_plan
from validation.ufc_airblast import BURST_SPHERICAL, z_interval
from validation.ufc_units import cube_root as cube_root_units


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
    )


class AutoPointsEngineTests(unittest.TestCase):
    def test_dpi_conversion_and_five_mm_spacing(self):
        self.assertAlmostEqual(pixels_per_mm(96.0), 96.0 / 25.4)
        self.assertAlmostEqual(target_spacing_px(96.0), 5.0 * 96.0 / 25.4)
        n = point_count(DEFAULT_PLOT_WIDTH_PX, DEFAULT_LOGICAL_DPI_X)
        expected = int(round(DEFAULT_PLOT_WIDTH_PX / target_spacing_px(96.0) + 1.0))
        expected = max(MIN_POINTS, min(MAX_POINTS, expected))
        self.assertEqual(n, expected)

    def test_point_count_guards(self):
        self.assertEqual(point_count(1.0, 96.0), MIN_POINTS)
        self.assertEqual(point_count(1.0e9, 96.0), MAX_POINTS)

    def test_log_spacing_is_uniform_in_log_r(self):
        radii = log_spaced(0.2, 10.0, 12)
        self.assertEqual(len(radii), 12)
        self.assertAlmostEqual(radii[0], 0.2)
        self.assertAlmostEqual(radii[-1], 10.0)
        diffs = [math.log(radii[i + 1]) - math.log(radii[i]) for i in range(len(radii) - 1)]
        for d in diffs:
            self.assertAlmostEqual(d, diffs[0], places=9)

    def test_ufc_rmin_is_zmin_times_cube_root_mass(self):
        mass = 8.0
        z_lo, z_hi = z_interval(BURST_SPHERICAL)
        plan = plan_1d(mass_kg=mass, domain_radius_m=50.0, cell_size=0.05)
        self.assertAlmostEqual(plan.r_min, z_lo * cube_root_units(mass))
        self.assertLessEqual(plan.r_max, z_hi * cube_root_units(mass) + 1e-12)

    def test_rmax_is_domain_intersection_no_extrapolation(self):
        mass = 1.0
        z_lo, z_hi = z_interval(BURST_SPHERICAL)
        domain = 0.5
        plan = plan_1d(mass_kg=mass, domain_radius_m=domain, cell_size=0.01)
        self.assertTrue(plan.ok)
        self.assertLess(plan.r_max, domain)
        self.assertGreaterEqual(plan.r_min, z_lo * cube_root_units(mass) - 1e-12)
        self.assertTrue(all(p.range_m <= plan.r_max + 1e-12 for p in plan.points))
        self.assertTrue(all(p.range_m >= plan.r_min - 1e-12 for p in plan.points))
        empty = plan_1d(mass_kg=mass, domain_radius_m=0.01, cell_size=0.001)
        self.assertFalse(empty.ok)
        self.assertEqual(empty.points, ())

    def test_1d_points_are_radial_and_named(self):
        plan = plan_1d(mass_kg=1.0, domain_radius_m=2.0)
        self.assertTrue(plan.ok)
        self.assertTrue(plan.points[0].point_id.startswith("VAL_1D_"))
        self.assertEqual(plan.points[0].purpose, "validation")
        for p in plan.points:
            self.assertEqual(p.x, p.range_m)
            self.assertEqual(p.y, 0.0)
            self.assertEqual(p.z, 0.0)
            self.assertEqual(p.dim, "1d")

    def test_2d_line_through_charge_centre(self):
        hob = 0.5
        plan = plan_2d(
            mass_kg=1.0,
            domain_radius_m=1.5,
            domain_height_m=1.5,
            hob_m=hob,
            cell_size=0.05,
        )
        self.assertTrue(plan.ok)
        self.assertEqual(plan.line_kind, "horizontal_through_charge_centre")
        self.assertAlmostEqual(plan.line_z, hob)
        self.assertTrue(plan.points[0].point_id.startswith("VAL_2D_"))
        for p in plan.points:
            self.assertAlmostEqual(p.y, hob)
            self.assertAlmostEqual(p.x, p.range_m)
            self.assertEqual(p.z, 0.0)

    def test_current_run_cache_key_changes_with_case(self):
        a = cache_key(case_1d="a", case_2d="b", mass_kg=1.0, domain_1d=1.0, domain_2d=1.5, height_2d=1.5, hob_m=0.5)
        b = cache_key(case_1d="a", case_2d="c", mass_kg=1.0, domain_1d=1.0, domain_2d=1.5, height_2d=1.5, hob_m=0.5)
        self.assertNotEqual(a, b)

    def test_marker_stride_does_not_drop_below_one(self):
        self.assertEqual(marker_stride(10, DEFAULT_PLOT_WIDTH_PX, 96.0), 1)
        self.assertGreaterEqual(marker_stride(80, 100.0, 96.0), 1)

    def test_legacy_message_exists(self):
        self.assertIn("VTK", LEGACY_NO_VALIDATION_HISTORIES)

    def test_2d_surface_burst_uses_fig_2_15(self):
        from validation.ufc_airblast import BURST_HEMISPHERICAL

        plan = plan_2d(
            mass_kg=1.0,
            domain_radius_m=2.0,
            domain_height_m=1.0,
            hob_m=0.0,
            cell_size=0.05,
        )
        self.assertEqual(plan.burst_master, BURST_HEMISPHERICAL)
        self.assertEqual(plan.figure, "2-15")
        self.assertTrue(plan.ok)
        z_lo, z_hi = z_interval(BURST_HEMISPHERICAL)
        self.assertAlmostEqual(plan.r_min, z_lo * cube_root_units(1.0))

    def test_remap_points_are_outside_receiving_region(self):
        from validation.auto_points import REMAP_NO_VALID_DOMAIN

        plan = plan_2d(
            mass_kg=1.0,
            domain_radius_m=2.0,
            domain_height_m=1.5,
            hob_m=0.5,
            cell_size=0.05,
            remap_receive_r_max=0.8,
        )
        self.assertTrue(plan.ok)
        self.assertTrue(plan.points)
        self.assertTrue(all(p.range_m > 0.8 for p in plan.points))
        self.assertEqual(plan.extra["remap_region"]["center"], [0.0, 0.5, 0.0])
        self.assertTrue(
            all(
                math.hypot(p.x - plan.charge_center[0], p.y - plan.charge_center[1]) > 0.8
                for p in plan.points
            )
        )
        empty = plan_2d(
            mass_kg=1.0,
            domain_radius_m=0.85,
            domain_height_m=1.5,
            hob_m=0.5,
            cell_size=0.05,
            remap_receive_r_max=0.84,
        )
        self.assertFalse(empty.ok)
        self.assertEqual(empty.points, ())
        self.assertTrue(any(REMAP_NO_VALID_DOMAIN in n for n in empty.notes))

    def test_remap_keeps_user_hob_burst_classification(self):
        from validation.ufc_airblast import BURST_HEMISPHERICAL, BURST_SPHERICAL

        elevated = plan_2d(
            mass_kg=1.0,
            domain_radius_m=2.0,
            domain_height_m=1.5,
            hob_m=0.8,
            cell_size=0.05,
            remap_receive_r_max=0.3,
        )
        self.assertEqual(elevated.burst_master, BURST_SPHERICAL)
        self.assertAlmostEqual(elevated.line_z, 0.8)
        self.assertEqual(elevated.charge_center[1], 0.8)
        self.assertTrue(all(abs(p.y - 0.8) < 1e-12 for p in elevated.points))
        self.assertEqual(elevated.extra["remap_region"]["center"], [0.0, 0.8, 0.0])
        # Exclusion is spherical distance from [0, HOB, 0], not from the origin.
        self.assertTrue(
            all(
                math.hypot(p.x, p.y - 0.8) > 0.3
                for p in elevated.points
            )
        )
        inside_if_origin_centred = [
            p for p in elevated.points if math.hypot(p.x, p.y) <= 0.3
        ]
        self.assertEqual(inside_if_origin_centred, [])

        surface = plan_2d(
            mass_kg=1.0,
            domain_radius_m=2.0,
            domain_height_m=1.5,
            hob_m=0.0,
            cell_size=0.05,
            remap_receive_r_max=0.3,
        )
        self.assertEqual(surface.burst_master, BURST_HEMISPHERICAL)
        self.assertEqual(surface.figure, "2-15")
        self.assertAlmostEqual(surface.line_z, 0.0)

    def test_higher_dpi_reduces_point_count(self):
        n96 = point_count(DEFAULT_PLOT_WIDTH_PX, 96.0)
        n192 = point_count(DEFAULT_PLOT_WIDTH_PX, 192.0)
        self.assertGreater(n96, n192)
        self.assertGreaterEqual(n192, MIN_POINTS)


class GeneratorSamplingTests(unittest.TestCase):
    def test_1d_writes_metadata_not_extra_validation_fo(self):
        with tempfile.TemporaryDirectory() as td:
            case = Generator1D(td).generate("one", _inputs_1d(), _rec())
            with open(os.path.join(case, "system", "controlDict"), encoding="utf-8") as handle:
                control = handle.read()
            self.assertNotIn("validationGauges1d", control)
            self.assertIn("gauges1d", control)
            self.assertIn("probes1d", control)
            plan = read_sampling_plan(case)
            self.assertIsNotNone(plan)
            self.assertTrue(plan.points)
            self.assertEqual(plan.points[0].point_id, "VAL_1D_001")

    def test_2d_user_probes_unchanged_and_validation_fo_separate(self):
        user = (ProbePoint2D("R1", 0.15, 0.5), ProbePoint2D("R2", 0.25, 0.5))
        with tempfile.TemporaryDirectory() as td:
            case = Generator2D(td).generate("two", replace(CaseInputs2D(), probes=user))
            with open(os.path.join(case, "system", "controlDict"), encoding="utf-8") as handle:
                control = handle.read()
            self.assertIn("probes2d", control)
            self.assertIn("0.15", control)
            self.assertIn("0.25", control)
            self.assertIn("validationGauges2d", control)
            self.assertEqual(control.count("probes2d"), 1)
            plan = read_sampling_plan(case)
            self.assertIsNotNone(plan)
            self.assertTrue(plan.points[0].point_id.startswith("VAL_2D_"))
            with open(os.path.join(case, "constant", "dynamicMeshDict"), encoding="utf-8") as handle:
                dyn = handle.read()
            self.assertIn("refineProbes true;", dyn)

    def test_2d_without_user_probes_still_writes_validation_line(self):
        with tempfile.TemporaryDirectory() as td:
            case = Generator2D(td).generate("novaluser", CaseInputs2D())
            with open(os.path.join(case, "system", "controlDict"), encoding="utf-8") as handle:
                control = handle.read()
            self.assertNotIn("probes2d", control)
            self.assertIn("validationGauges2d", control)
            self.assertNotIn("type refineProbes;", control)


if __name__ == "__main__":
    unittest.main()
