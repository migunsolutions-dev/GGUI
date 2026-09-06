"""Impulse sampling adequacy is judged from the history, not a fixed step interval."""

from __future__ import annotations

import unittest

from probe_sampling import (
    impulse_converged,
    positive_impulse,
    recommended_write_interval_steps,
    samples_in_positive_phase,
)


def triangle(n=200, t_plus=0.002, pmax=1.0e5):
    dt = t_plus / (n - 1)
    times = [i * dt for i in range(n)]
    # Rise immediately then linear decay to zero at t_plus.
    over = [pmax * max(0.0, 1.0 - t / t_plus) for t in times]
    times = times + [t_plus + dt]
    over = over + [-0.01 * pmax]
    return times, over, dt, t_plus


class ImpulseTests(unittest.TestCase):
    def test_triangle_impulse_is_half_pmax_times_duration(self):
        times, over, _dt, t_plus = triangle()
        self.assertAlmostEqual(positive_impulse(times, over), 0.5 * 1.0e5 * t_plus, places=0)

    def test_dense_history_converges_under_subsampling(self):
        times, over, dt, t_plus = triangle(n=400)
        ok, _base, rows = impulse_converged(times, over, strides=(2, 4), rel_tol=0.02)
        self.assertTrue(ok)
        self.assertGreater(samples_in_positive_phase(times, over), 50)
        self.assertGreaterEqual(recommended_write_interval_steps(dt_s=dt, positive_phase_s=t_plus), 1)

    def test_coarse_history_fails_convergence(self):
        times, over, _dt, _tp = triangle(n=8)
        ok, _base, _rows = impulse_converged(times, over, strides=(2, 4), rel_tol=0.02)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
