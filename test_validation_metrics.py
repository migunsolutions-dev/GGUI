"""Relative error, RMS/MAE, current-run resolver, and display units."""
from __future__ import annotations

import math
import unittest

from validation.current_run import (
    SOURCE_CURRENT,
    SOURCE_MANUAL,
    RunSnapshot,
    case_dir_for_dim,
    primary_case_dir,
    reset_to_current,
    with_manual_case,
)
from validation.metrics import (
    max_absolute_error,
    max_meaningful_relative_error,
    mean_absolute_error,
    relative_error_percent,
    rms_error,
)
from validation.units import fmt, kpa_to_pa, pa_s_to_kpa_ms, pa_to_kpa, s_to_ms


class RelativeErrorTests(unittest.TestCase):
    def test_percent_convention(self):
        self.assertAlmostEqual(relative_error_percent(110.0, 100.0), 10.0)
        self.assertAlmostEqual(relative_error_percent(90.0, 100.0), -10.0)

    def test_near_zero_reference_is_none(self):
        self.assertIsNone(relative_error_percent(1.0, 0.0))
        self.assertIsNone(relative_error_percent(1.0, 1e-40))
        self.assertIsNone(relative_error_percent(None, 1.0))

    def test_series_metrics(self):
        left = [1.0, 2.0, 3.0]
        right = [1.0, 3.0, 3.0]
        self.assertAlmostEqual(rms_error(left, right), math.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(mean_absolute_error(left, right), 1.0 / 3.0)
        self.assertAlmostEqual(max_absolute_error(left, right), 1.0)
        self.assertAlmostEqual(max_meaningful_relative_error(left, right), 1.0 / 3.0)

    def test_relative_near_zero_reference_skipped(self):
        self.assertIsNone(max_meaningful_relative_error([1.0], [0.0]))


class UnitFormatTests(unittest.TestCase):
    def test_si_display_roundtrip(self):
        self.assertAlmostEqual(pa_to_kpa(101325.0), 101.325)
        self.assertAlmostEqual(kpa_to_pa(101.325), 101325.0)
        self.assertAlmostEqual(s_to_ms(0.002), 2.0)
        self.assertAlmostEqual(pa_s_to_kpa_ms(12.5), 12.5)
        self.assertEqual(fmt(None), "N/A")
        self.assertIn("kPa", fmt(12.5, suffix="kPa"))


class CurrentRunResolverTests(unittest.TestCase):
    def test_live_solver_wins(self):
        snap = RunSnapshot(
            live_mode="2D",
            live_case_dir=r"C:\live2d",
            case_2d=r"C:\init2d",
            last_run_2d=r"C:\old2d",
        )
        self.assertEqual(case_dir_for_dim(snap, "2d"), r"C:\live2d")

    def test_initialized_before_last_run(self):
        snap = RunSnapshot(case_2d=r"C:\init2d", last_run_2d=r"C:\old2d")
        self.assertEqual(case_dir_for_dim(snap, "2d"), r"C:\init2d")

    def test_last_run_then_mapping_1d(self):
        snap = RunSnapshot(last_run_1d=r"C:\run1d")
        self.assertEqual(case_dir_for_dim(snap, "1d"), r"C:\run1d")
        empty = RunSnapshot()
        self.assertIsNone(case_dir_for_dim(empty, "1d"))
        self.assertIsNone(primary_case_dir(empty))

    def test_manual_does_not_scan(self):
        snap = RunSnapshot(source=SOURCE_CURRENT, case_2d=r"C:\current")
        manual = with_manual_case(snap, "2d", r"C:\manual")
        self.assertEqual(manual.source, SOURCE_MANUAL)
        self.assertEqual(case_dir_for_dim(manual, "2d"), r"C:\manual")
        restored = reset_to_current(manual)
        self.assertEqual(restored.source, SOURCE_CURRENT)


if __name__ == "__main__":
    unittest.main()
