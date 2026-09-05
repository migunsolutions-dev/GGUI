"""1D → 2D remap about the user-controlled target HOB."""
from __future__ import annotations

import math
import unittest

import numpy as np

import os
import tempfile

from remap_fields_2d import (
    GROUND_CLIP,
    MAPPING_METHOD,
    _read_1d_data,
    carry_mixture_mass_in_air,
    charge_center_xyz,
    effective_mapped_radius,
    map_fields_to_2d_cells,
    map_scalar_profile,
    remap_region_metadata,
    source_radius_rz,
)


def _peak_profile(r_max=1.0, n=21, p_peak=1.0e7, p_amb=1.01325e5):
    r_1d = np.linspace(0.0, r_max, n)
    p_1d = p_amb + (p_peak - p_amb) * np.exp(-(r_1d / 0.05) ** 2)
    return r_1d, p_1d, p_amb


class RemapFields2DTests(unittest.TestCase):
    def test_source_radius_uses_hob_not_origin(self):
        self.assertAlmostEqual(source_radius_rz(0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(source_radius_rz(0.0, 0.0, 1.25), 1.25)
        self.assertAlmostEqual(source_radius_rz(0.3, 1.25, 1.25), 0.3)
        self.assertAlmostEqual(
            float(source_radius_rz(0.4, 0.9, 0.5)),
            math.hypot(0.4, 0.4),
        )

    def test_hob_zero_field_is_centred_at_ground(self):
        r_1d, p_1d, p_amb = _peak_profile()
        r = np.array([0.0, 0.4, 0.0])
        z = np.array([0.0, 0.0, 0.4])
        p = map_scalar_profile(r, z, 0.0, r_1d, p_1d, ambient=p_amb)
        self.assertGreater(p[0], p[1])
        self.assertGreater(p[0], p[2])
        self.assertAlmostEqual(p[1], p[2], places=6)

    def test_hob_elevated_field_is_centred_at_target_charge(self):
        r_1d, p_1d, p_amb = _peak_profile()
        hob = 1.25
        r = np.array([0.0, 0.0, 0.4, 0.0])
        z = np.array([0.0, hob, hob, hob + 0.4])
        p = map_scalar_profile(r, z, hob, r_1d, p_1d, ambient=p_amb)
        self.assertGreater(p[1], p[0])
        self.assertGreater(p[1], p[2])
        self.assertGreater(p[1], p[3])
        self.assertAlmostEqual(p[2], p[3], places=6)

    def test_raising_hob_moves_the_pressure_hotspot_up(self):
        r_1d, p_1d, p_amb = _peak_profile()
        z = np.array([0.0, 0.6, 1.2])
        r = np.zeros_like(z)
        p_low = map_scalar_profile(r, z, 0.0, r_1d, p_1d, ambient=p_amb)
        p_high = map_scalar_profile(r, z, 1.2, r_1d, p_1d, ambient=p_amb)
        self.assertEqual(int(np.argmax(p_low)), 0)
        self.assertEqual(int(np.argmax(p_high)), 2)

    def test_no_below_ground_mirror_or_duplicate(self):
        r_1d, p_1d, p_amb = _peak_profile()
        hob = 0.5
        # A cell below the ground must stay ambient — never an image charge.
        p_below = map_scalar_profile(
            np.array([0.0]),
            np.array([-0.1]),
            hob,
            r_1d,
            p_1d,
            ambient=p_amb,
        )
        self.assertAlmostEqual(float(p_below[0]), p_amb)
        # Ground cell samples distance-to-HOB, not an origin-centred or mirrored peak.
        r_src = float(source_radius_rz(0.0, 0.0, hob))
        self.assertAlmostEqual(r_src, hob)
        p_ground = map_scalar_profile(
            np.array([0.0]),
            np.array([0.0]),
            hob,
            r_1d,
            p_1d,
            ambient=p_amb,
        )
        p_at_hob = map_scalar_profile(
            np.array([0.0]),
            np.array([hob]),
            hob,
            r_1d,
            p_1d,
            ambient=p_amb,
        )
        self.assertGreater(float(p_at_hob[0]), float(p_ground[0]))

    def test_velocity_points_away_from_target_centre(self):
        r_1d = np.array([0.0, 0.5, 1.0])
        u_mag = np.array([0.0, 100.0, 50.0])
        mapped = map_fields_to_2d_cells(
            np.array([0.3]),
            np.array([1.0]),
            0.7,
            r_1d,
            {"U_mag": u_mag, "p": np.array([1.0, 1.0, 1.0])},
            ambient={"p": 0.0, "U": np.zeros(3)},
        )
        u = mapped["U"][0]
        self.assertGreater(u[0], 0.0)
        self.assertGreater(u[1], 0.0)
        self.assertAlmostEqual(u[2], 0.0)
        self.assertAlmostEqual(u[1] / u[0], (1.0 - 0.7) / 0.3, places=8)

    def test_per_cell_ambient_from_0_orig_does_not_crash(self):
        r_1d, p_1d, p_amb = _peak_profile()
        r = np.array([0.0, 2.0])
        z = np.array([0.0, 0.0])
        n = r.size
        mapped = map_fields_to_2d_cells(
            r,
            z,
            0.0,
            r_1d,
            {
                "p": p_1d,
                "T": np.full_like(p_1d, 300.0),
                "rho.air": np.full_like(p_1d, 1.2),
                "U_mag": np.zeros_like(p_1d),
            },
            ambient={
                "p": np.full(n, p_amb),
                "T": np.full(n, 288.15),
                "rho.air": np.full(n, 1.225),
            },
        )
        self.assertGreater(mapped["p"][0], mapped["p"][1])
        self.assertAlmostEqual(float(mapped["p"][1]), p_amb)
        self.assertEqual(mapped["U"].shape, (n, 3))

    def test_product_cells_keep_mixture_mass_when_he_phase_is_dropped(self):
        mapped = {
            "rho.air": np.array([1e-16, 1e-16, 1.225]),
            "rho.c4": np.array([2.0, 1.0, 1600.0]),
            "alpha.c4": np.array([1.0, 1.0, 0.0]),
            "U": np.array([[800.0, 0.0, 0.0], [400.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
        before_air = mapped["rho.air"].copy()
        carry_mixture_mass_in_air(mapped, unused_rho_c4=1600.0)
        self.assertTrue(np.all(mapped["alpha.c4"] == 0.0))
        self.assertTrue(np.all(mapped["rho.c4"] == 1600.0))
        self.assertAlmostEqual(float(mapped["rho.air"][0]), 2.0)
        self.assertAlmostEqual(float(mapped["rho.air"][1]), 1.0)
        self.assertAlmostEqual(float(mapped["rho.air"][2]), 1.225)
        self.assertGreater(float(mapped["rho.air"][0]), float(before_air[0]))
        self.assertTrue(np.all(mapped["rho.air"] > 0.1))
        self.assertAlmostEqual(float(mapped["U"][0, 0]), 800.0)

        r = np.array([0.0, 0.3, 1.5])
        z = np.array([1.0, 1.0, 1.0])
        r_1d = np.array([0.0, 0.3, 0.6])
        from_source = map_fields_to_2d_cells(
            r,
            z,
            1.0,
            r_1d,
            {
                "p": np.array([2.0e6, 4.0e5, 1.01325e5]),
                "T": np.array([2000.0, 800.0, 288.0]),
                "rho.air": np.array([1e-16, 1e-16, 1.225]),
                "rho.c4": np.array([2.0, 1.0, 1600.0]),
                "alpha.c4": np.array([1.0, 1.0, 0.0]),
                "U_mag": np.array([800.0, 400.0, 0.0]),
            },
            ambient={"p": 101325.0, "T": 288.0, "rho.air": 1.225, "rho.c4": 1600.0},
        )
        carry_mixture_mass_in_air(from_source, unused_rho_c4=1600.0)
        self.assertTrue(np.all(from_source["rho.air"] > 0.1))
        self.assertAlmostEqual(float(from_source["rho.air"][1]), 1.0)
        self.assertAlmostEqual(float(from_source["rho.air"][2]), 1.225)

    def test_read_1d_data_uses_poly_mesh_cell_radii_not_linspace(self):
        with tempfile.TemporaryDirectory() as td:
            mesh = os.path.join(td, "constant", "polyMesh")
            os.makedirs(mesh)
            os.makedirs(os.path.join(td, "0.001"))
            with open(os.path.join(mesh, "points"), "w", encoding="utf-8") as handle:
                handle.write(
                    "8\n(\n"
                    "(0.10 0 0)\n(0.10 0.01 0)\n(0.10 0.01 0.01)\n(0.10 0 0.01)\n"
                    "(1.50 0 0)\n(1.50 0.01 0)\n(1.50 0.01 0.01)\n(1.50 0 0.01)\n"
                    ")\n"
                )
            with open(os.path.join(mesh, "faces"), "w", encoding="utf-8") as handle:
                handle.write("2\n(\n4(4 5 6 7)\n4(0 1 2 3)\n)\n")
            with open(os.path.join(mesh, "owner"), "w", encoding="utf-8") as handle:
                handle.write("2\n(\n0\n1\n)\n")
            with open(os.path.join(td, "0.001", "p"), "w", encoding="utf-8") as handle:
                handle.write(
                    "internalField nonuniform List<scalar>\n2\n(\n 2.0e6\n 1.0e5\n);\n"
                )
            data = _read_1d_data(td, "0.001")
            self.assertIsNotNone(data)
            self.assertGreater(float(data["r"][0]), 1.49)
            self.assertLess(float(data["r"][1]), 0.12)
            self.assertGreater(float(data["p"][0]), float(data["p"][1]))

    def test_read_1d_data_does_not_linspace_when_mesh_radii_missing(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "constant", "polyMesh"))
            os.makedirs(os.path.join(td, "0.001"))
            with open(os.path.join(td, "constant", "polyMesh", "points"), "w", encoding="utf-8") as handle:
                handle.write("2\n(\n(0 0 0)\n(1.50 0 0)\n)\n")
            with open(os.path.join(td, "0.001", "p"), "w", encoding="utf-8") as handle:
                handle.write(
                    "internalField nonuniform List<scalar>\n2\n(\n 2.0e6\n 1.0e5\n);\n"
                )
            self.assertIsNone(_read_1d_data(td, "0.001"))

    def test_he_wipe_without_carry_discards_product_mass(self):
        """1D->3D used to zero HE without moving mass into rho.air."""
        alpha = np.array([1.0, 0.0])
        rho_c4 = np.array([2.5, 1600.0])
        rho_air = np.array([1e-16, 1.225])
        mix = alpha * rho_c4 + (1.0 - alpha) * rho_air
        wiped = np.zeros_like(alpha) * rho_c4 + (1.0 - 0.0) * rho_air
        carried = mix.copy()
        self.assertLess(float(wiped[0]), 1e-10)
        self.assertAlmostEqual(float(carried[0]), 2.5)
        self.assertAlmostEqual(float(carried[1]), float(wiped[1]))

    def test_mapped_radius_clips_source_padding(self):
        r_1d = np.array([0.0, 0.8, 1.6])
        self.assertAlmostEqual(effective_mapped_radius(r_1d, mapped_radius=0.5), 0.5)
        self.assertAlmostEqual(effective_mapped_radius(r_1d, mapped_radius=0.0), 1.6)
        self.assertAlmostEqual(effective_mapped_radius(r_1d, mapped_radius=2.0), 1.6)

    def test_user_radius_is_not_silently_replaced_by_1d_padding(self):
        r_1d = np.array([0.0, 0.3, 0.6, 0.66])
        p_1d = np.array([4.0e6, 2.0e6, 1.5e6, 1.4e6])
        p_amb = 101325.0
        r = np.array([0.55, 0.63])
        z = np.array([1.0, 1.0])
        p = map_scalar_profile(r, z, 1.0, r_1d, p_1d, mapped_radius=0.60, ambient=p_amb)
        self.assertGreater(p[0], p_amb)
        self.assertAlmostEqual(p[1], p_amb)

    def test_region_metadata_agrees_with_target_hob(self):
        meta = remap_region_metadata(1.25, mapped_radius=0.4, source_time="latest", time_mode="latest")
        self.assertEqual(meta["center"], list(charge_center_xyz(1.25)))
        self.assertEqual(meta["mapping_method"], MAPPING_METHOD)
        self.assertEqual(meta["ground_clip"], GROUND_CLIP)
        self.assertEqual(meta["target_time"], "0")


if __name__ == "__main__":
    unittest.main()
