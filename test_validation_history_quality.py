"""Peak/impulse are N/A unless arrival, positive phase, and endTime checks pass."""
from __future__ import annotations

import unittest

from validation.history_quality import assess_history, comparable_peak_impulse


def _friedlander(n: int = 40, *, complete: bool = True, t_end: float = 0.01):
    times = [t_end * i / (n - 1) for i in range(n)]
    p_atm = 101325.0
    pressure = []
    impulse = []
    acc = 0.0
    peak = 2.0e5
    t_arr = 0.002
    t_pos = 0.007
    dt = times[1] - times[0]
    for t in times:
        if t < t_arr:
            over = 0.0
        elif (not complete) or t <= t_pos:
            frac = (t - t_arr) / max(t_pos - t_arr, 1e-12)
            over = peak * max(0.0, 1.0 - 0.5 * frac)
            if not complete:
                over = peak * 0.8
        else:
            over = -500.0
        pressure.append(p_atm + over)
        if over > 0.0:
            acc += over * dt
        impulse.append(acc)
    return times, pressure, impulse, p_atm


class HistoryQualityTests(unittest.TestCase):
    def test_complete_history_reaching_endtime_is_comparable(self):
        times, pressure, impulse, atm = _friedlander(complete=True, t_end=0.01)
        validity = assess_history(
            times, pressure, impulse, p_atm=atm, end_time_s=0.01, reached_end=True
        )
        self.assertTrue(validity.arrival_detected)
        self.assertTrue(validity.positive_phase_started)
        self.assertTrue(validity.positive_phase_completed)
        self.assertTrue(validity.run_reached_end_time)
        self.assertTrue(validity.comparable)
        peak, impl, reason = comparable_peak_impulse(validity)
        self.assertIsNotNone(peak)
        self.assertIsNotNone(impl)
        self.assertEqual(reason, "")

    def test_incomplete_positive_phase_is_not_compared(self):
        times, pressure, impulse, atm = _friedlander(complete=False, t_end=0.01)
        validity = assess_history(
            times, pressure, impulse, p_atm=atm, end_time_s=0.01, reached_end=True
        )
        self.assertTrue(validity.arrival_detected)
        self.assertFalse(validity.positive_phase_completed)
        self.assertFalse(validity.comparable)
        peak, impl, reason = comparable_peak_impulse(validity)
        self.assertIsNone(peak)
        self.assertIsNone(impl)
        self.assertIn("incomplete", reason.lower())

    def test_run_short_of_endtime_is_not_compared(self):
        times, pressure, impulse, atm = _friedlander(complete=True, t_end=0.01)
        validity = assess_history(
            times, pressure, impulse, p_atm=atm, end_time_s=0.02, reached_end=False
        )
        self.assertTrue(validity.positive_phase_completed)
        self.assertFalse(validity.comparable)
        peak, impl, reason = comparable_peak_impulse(validity)
        self.assertIsNone(peak)
        self.assertIsNone(impl)
        self.assertIn("endtime", reason.lower())

    def test_openfoam_great_sentinel_is_not_compared(self):
        p_atm = 101325.0
        great = -1.79769313486e307
        times = [i * 0.001 for i in range(12)]
        pressure = [great] * 12
        validity = assess_history(
            times, pressure, p_atm=p_atm, end_time_s=0.011, reached_end=True
        )
        self.assertFalse(validity.comparable)
        peak, impl, reason = comparable_peak_impulse(validity)
        self.assertIsNone(peak)
        self.assertIsNone(impl)
        self.assertTrue(reason)

    def test_peak_is_taken_after_arrival_not_the_last_sample(self):
        p_atm = 101325.0
        times = [i * 0.001 for i in range(20)]
        pressure = [p_atm] * 4 + [p_atm + 2.0e5] + [p_atm + 1.0e4] * 4 + [p_atm - 100.0] * 11
        impulse = [0.0] * 20
        validity = assess_history(
            times, pressure, impulse, p_atm=p_atm, end_time_s=0.019, reached_end=True
        )
        self.assertTrue(validity.comparable)
        self.assertAlmostEqual(validity.peak_overpressure_pa, 2.0e5)
        self.assertAlmostEqual(validity.peak_time_s, 0.004)


if __name__ == "__main__":
    unittest.main()
