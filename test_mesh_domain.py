"""Tests for domain alignment to integer cell counts (requested cell size preserved)."""
from __future__ import annotations

import unittest

from mesh_domain import align_domain_to_cell_size


class TestMeshDomain(unittest.TestCase):
    def test_cell_size_preserved_domain_adjusted(self):
        r = align_domain_to_cell_size((0.0, 0.0, 0.0), (10.03, 1.0, 2.0), 0.5)
        self.assertEqual(r.cell_size, 0.5)
        self.assertEqual(r.nx, 20)
        self.assertAlmostEqual(r.actual_lengths[0], 10.0, places=9)
        self.assertTrue(r.adjusted)

    def test_no_adjustment_when_clean_divide(self):
        r = align_domain_to_cell_size((-5.0, -5.0, 0.0), (5.0, 5.0, 5.0), 0.5)
        self.assertFalse(r.adjusted)
        self.assertEqual(r.nx, 20)
        self.assertEqual(r.ny, 20)
        self.assertEqual(r.nz, 10)

    def test_min_corner_fixed(self):
        r = align_domain_to_cell_size((-2.0, 0.0, 0.0), (3.07, 1.0, 1.0), 0.25)
        self.assertAlmostEqual(r.min_point[0], -2.0)
        self.assertAlmostEqual(r.max_point[0], -2.0 + 5.0, places=6)  # round(5.07/0.25)=20 → 5.0


if __name__ == "__main__":
    unittest.main()
