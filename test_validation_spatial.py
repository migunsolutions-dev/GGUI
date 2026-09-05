"""1D spatial radius uses sqrt(x^2+y^2+z^2) for wedge/spherical geometry."""
from __future__ import annotations

import unittest

import numpy as np

from validation.probes import radial_distance, radii_from_locations
from validation.spatial import spherical_radii


class SpatialRadiusTests(unittest.TestCase):
    def test_spherical_radii_use_euclidean_norm_not_x_alone(self):
        centres = np.array(
            [
                [0.10, 0.02, 0.01],
                [0.20, 0.04, 0.02],
            ],
            dtype=float,
        )
        r = spherical_radii(centres)
        expected = np.sqrt(np.sum(centres**2, axis=1))
        np.testing.assert_allclose(r, expected)
        self.assertFalse(np.allclose(r, centres[:, 0]))

    def test_probes1d_locations_still_use_euclidean_radius(self):
        locs = ["0.10 0.02 0.01", "0.20 0.04 0.02"]
        radii = radii_from_locations(locs, dim="1d")
        self.assertAlmostEqual(radii[0], radial_distance(0.10, 0.02, 0.01))
        self.assertAlmostEqual(radii[1], (0.20**2 + 0.04**2 + 0.02**2) ** 0.5)

    def test_2d_probe_radius_remains_cylindrical_x(self):
        locs = ["0.40 0.50 0.00"]
        radii = radii_from_locations(locs, dim="2d")
        self.assertAlmostEqual(radii[0], 0.40)


if __name__ == "__main__":
    unittest.main()
