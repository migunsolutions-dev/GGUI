"""Fixed 10-cell 1D -> 2D remap handoff (no 8 kPa radius rule)."""
from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from dataclasses import replace

from completion_1d import ARRIVAL_CRITERION, ARRIVAL_OVERPRESSURE_PA, read_completion_record
from generator_1d import Generator1D
from models import (
    BOUNDARY_1D_REFLECT,
    BOUNDARY_1D_TERMINATE,
    CaseInputs1D,
    RecommendedParams1D,
    RUN_MODE_REFLECT,
    RUN_MODE_TERMINATE,
)
from remap_handoff_1d import (
    HANDOFF_CRITERION,
    PRIMARY_SHOCK_COMPRESSION_RATIO,
    REMAP_FRONT_BUFFER_CELLS_1D,
    HandoffGeometryError,
    handoff_plan,
    handoff_radius_m,
    leading_primary_front_radius_m,
    physical_buffer_m,
    primary_shock_arrival_time,
    primary_shock_at_probe,
    read_handoff_metadata,
    uses_remap_handoff,
)
import remap_handoff_1d as handoff_mod


def _inputs(**kwargs) -> CaseInputs1D:
    base = CaseInputs1D(
        radius=1.5,
        cell_size=0.01,
        p_atm=101325.0,
        t_atm=288.0,
        mass_kg=5.0,
        rho_charge=1630.0,
        energy_j_per_kg=4.29e6,
        material_props={
            "rho": 1630.0,
            "A": 371.2e9,
            "B": 3.23e9,
            "R1": 4.15,
            "R2": 0.95,
            "omega": 0.30,
            "E0": 4.29e6,
        },
        max_cfl=0.5,
        end_time_s=0.03,
        right_boundary=BOUNDARY_1D_TERMINATE,
        stop_mode=RUN_MODE_TERMINATE,
        stop_radius_m=1.5,
        remap_for_2d=False,
    )
    return replace(base, **kwargs) if kwargs else base


def _rec() -> RecommendedParams1D:
    return RecommendedParams1D(
        r_min=1.0e-4,
        ignition_point=(0.0, 0.0, 0.0),
        ignition_radius=0.01,
        dt0=1.0e-8,
        maxCo=0.5,
        maxDeltaT=1.0e-5,
    )


class RemapHandoffGeometryTests(unittest.TestCase):
    def test_buffer_is_ten_cells(self):
        self.assertEqual(REMAP_FRONT_BUFFER_CELLS_1D, 10)

    def test_handoff_radius_is_remap_minus_ten_cells(self):
        self.assertAlmostEqual(handoff_radius_m(1.5, 0.01), 1.4)
        self.assertAlmostEqual(physical_buffer_m(0.01), 0.10)

    def test_remap_radius_itself_is_not_reduced(self):
        plan = handoff_plan(1.5, 0.01)
        self.assertAlmostEqual(plan["remap_radius_m"], 1.5)
        self.assertAlmostEqual(plan["handoff_radius_m"], 1.4)
        self.assertLess(plan["handoff_radius_m"], plan["remap_radius_m"])

    def test_no_8kpa_in_handoff_module_or_plan(self):
        source = inspect.getsource(handoff_mod)
        self.assertNotIn("8000", source)
        self.assertNotIn("8 kPa", source)
        self.assertNotIn("8kPa", source.lower().replace(" ", ""))
        plan = handoff_plan(1.5, 0.01)
        blob = str(plan).lower()
        self.assertNotIn("8000", blob)
        self.assertNotIn("8 kpa", blob)
        self.assertNotIn(str(int(ARRIVAL_OVERPRESSURE_PA)), blob)
        self.assertEqual(plan["handoff_criterion"], HANDOFF_CRITERION)
        self.assertNotEqual(HANDOFF_CRITERION, ARRIVAL_CRITERION)
        self.assertNotIn("8000", HANDOFF_CRITERION)
        self.assertGreater(PRIMARY_SHOCK_COMPRESSION_RATIO, 1.0)


class RemapHandoffFrontTriggerTests(unittest.TestCase):
    def test_trailing_plateau_is_not_the_primary_shock(self):
        p_atm = 101325.0
        self.assertFalse(primary_shock_at_probe(p_atm, p_atm))
        self.assertFalse(primary_shock_at_probe(p_atm + 8000.0, p_atm))
        self.assertFalse(primary_shock_at_probe(146000.0, p_atm))

    def test_primary_shock_is_a_compression_not_a_kpa_gate(self):
        p_atm = 101325.0
        self.assertTrue(primary_shock_at_probe(2.0 * p_atm, p_atm))
        self.assertTrue(primary_shock_at_probe(4.0e5, p_atm))
        self.assertGreater(PRIMARY_SHOCK_COMPRESSION_RATIO * p_atm, p_atm + 8000.0)

    def test_arrival_time_waits_for_shock_not_8kpa(self):
        p_atm = 101325.0
        times = (0.0, 1.0e-4, 2.0e-4, 3.0e-4)
        pressures = (p_atm, p_atm + 8000.0, 146000.0, 4.0e5)
        self.assertAlmostEqual(
            primary_shock_arrival_time(times, pressures, p_atm), 3.0e-4
        )

    def test_profile_front_ignores_outer_plateau(self):
        p_atm = 101325.0
        r = (0.07, 0.18, 0.40, 0.59, 0.60)
        p = (3.9e6, 1.2e6, 6.2e4, 1.46e5, 1.46e5)
        front = leading_primary_front_radius_m(r, p, p_atm)
        self.assertAlmostEqual(front, 0.18)
        self.assertLess(front, 0.59)

    def test_gui_and_harness_share_the_same_trigger(self):
        import solver_runner
        from viper_compare import ggui_run

        self.assertIs(solver_runner.primary_shock_at_probe, primary_shock_at_probe)
        self.assertIs(ggui_run.primary_shock_at_probe, primary_shock_at_probe)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(HandoffGeometryError) as ctx:
            handoff_radius_m(0.05, 0.01)
        self.assertIn("too small", str(ctx.exception).lower())
        with self.assertRaises(HandoffGeometryError):
            handoff_radius_m(0.10, 0.01)
        with self.assertRaises(HandoffGeometryError):
            Generator1D(tempfile.mkdtemp(prefix="ggui_bad_handoff_")).generate(
                "too_small",
                _inputs(radius=0.08, cell_size=0.01, remap_for_2d=True),
                _rec(),
            )


class RemapHandoffGeneratorTests(unittest.TestCase):
    def test_remap_terminate_watchdog_at_handoff_not_domain(self):
        root = tempfile.mkdtemp(prefix="ggui_remap_handoff_")
        case_dir = Generator1D(root).generate(
            "case_remap",
            _inputs(remap_for_2d=True),
            _rec(),
        )
        record = read_completion_record(case_dir)
        self.assertIsNotNone(record)
        self.assertTrue(record.remap_for_2d)
        self.assertAlmostEqual(record.remap_radius_m, 1.5)
        self.assertAlmostEqual(record.dr_1d_m, 0.01)
        self.assertEqual(record.remap_front_buffer_cells, 10)
        self.assertAlmostEqual(record.handoff_radius_m, 1.4)
        self.assertAlmostEqual(record.requested_stop_radius_m, 1.4)
        self.assertEqual(record.criterion, HANDOFF_CRITERION)
        self.assertNotIn("8000", record.criterion)
        with open(os.path.join(case_dir, ".watchdog_target_radius"), encoding="utf-8") as handle:
            self.assertAlmostEqual(float(handle.read().strip()), 1.4)
        with open(os.path.join(case_dir, "system", "blockMeshDict"), encoding="utf-8") as handle:
            mesh = handle.read()
        self.assertIn("(1 1 164)", mesh)
        self.assertNotIn("(1 1 140)", mesh)
        meta = read_handoff_metadata(case_dir)
        self.assertIsNotNone(meta)
        self.assertAlmostEqual(meta["remap_radius_m"], 1.5)
        self.assertAlmostEqual(meta["dr_1d_m"], 0.01)
        self.assertEqual(meta["remap_front_buffer_cells"], 10)
        self.assertAlmostEqual(meta["handoff_radius_m"], 1.4)

    def test_non_remap_terminate_still_stops_at_domain_or_stop_radius(self):
        root = tempfile.mkdtemp(prefix="ggui_plain_1d_")
        gen = Generator1D(root)
        domain_dir = gen.generate("plain_domain", _inputs(remap_for_2d=False), _rec())
        record = read_completion_record(domain_dir)
        self.assertFalse(record.remap_for_2d)
        self.assertAlmostEqual(record.requested_stop_radius_m, 1.5)
        self.assertIsNone(record.handoff_radius_m)
        self.assertEqual(record.criterion, ARRIVAL_CRITERION)
        with open(os.path.join(domain_dir, ".watchdog_target_radius"), encoding="utf-8") as handle:
            self.assertAlmostEqual(float(handle.read().strip()), 1.5)
        inner_dir = gen.generate(
            "plain_inner",
            _inputs(remap_for_2d=False, stop_radius_m=0.5),
            _rec(),
        )
        inner = read_completion_record(inner_dir)
        self.assertAlmostEqual(inner.requested_stop_radius_m, 0.5)
        self.assertIsNone(inner.handoff_radius_m)

    def test_remap_reflect_does_not_move_watchdog(self):
        self.assertFalse(
            uses_remap_handoff(
                _inputs(
                    remap_for_2d=True,
                    stop_mode=RUN_MODE_REFLECT,
                    right_boundary=BOUNDARY_1D_REFLECT,
                )
            )
        )
        root = tempfile.mkdtemp(prefix="ggui_remap_reflect_")
        case_dir = Generator1D(root).generate(
            "remap_reflect",
            _inputs(
                remap_for_2d=True,
                stop_mode=RUN_MODE_REFLECT,
                right_boundary=BOUNDARY_1D_REFLECT,
            ),
            _rec(),
        )
        record = read_completion_record(case_dir)
        self.assertAlmostEqual(record.requested_stop_radius_m, 1.5)
        self.assertIsNone(record.handoff_radius_m)
        self.assertFalse(os.path.isfile(os.path.join(case_dir, "ggui_remap_handoff.json")))


if __name__ == "__main__":
    unittest.main()
