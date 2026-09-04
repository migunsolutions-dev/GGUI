"""1D→2D / 2D→3D remap comparison and conservation helpers."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from validation import remap as remap_engine


class RemapResolveTests(unittest.TestCase):
    def test_missing_target_is_current_run_message(self):
        src, st, tt, msg = remap_engine.resolve_1d_to_2d(target_case="", mapping_source=None)
        self.assertIsNone(src)
        self.assertIn("required validation data", msg)

    def test_metadata_only_no_filename_guess(self):
        with tempfile.TemporaryDirectory() as td:
            src, st, tt, msg = remap_engine.resolve_1d_to_2d(
                target_case=td, mapping_source=""
            )
            self.assertIn("not recorded", msg)

    def test_case_2d_json_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            source = os.path.join(td, "src1d")
            target = os.path.join(td, "tgt2d")
            os.makedirs(os.path.join(source, "0"))
            os.makedirs(os.path.join(target, "0"))
            with open(os.path.join(target, "case_2d.json"), "w", encoding="utf-8") as handle:
                json.dump({"mapping": {"case_path": source, "specific_time": "0.001"}}, handle)
            src, st, tt, msg = remap_engine.resolve_1d_to_2d(
                target_case=target, mapping_source=None
            )
            self.assertEqual(msg, "")
            self.assertEqual(os.path.normpath(src), os.path.normpath(source))
            self.assertEqual(st, "0.001")
            self.assertEqual(tt, "0")

    def test_2d_to_3d_refuses_1d_radial_source(self):
        src, st, tt, msg = remap_engine.resolve_2d_to_3d(
            target_case="anywhere",
            remap_source_type="1D",
            prepare_3d_transfer=None,
        )
        self.assertIsNone(src)
        self.assertIn("1D radial", msg)

    def test_2d_to_3d_requires_prepare_json(self):
        with tempfile.TemporaryDirectory() as td:
            src, st, tt, msg = remap_engine.resolve_2d_to_3d(
                target_case=td,
                remap_source_type="2D",
                prepare_3d_transfer=None,
            )
            self.assertIn("prepare_3d_transfer.json", msg)


class ProfileAndConservationTests(unittest.TestCase):
    def test_compare_profiles_metrics_and_dt_warning(self):
        r = np.linspace(0.1, 1.0, 10)
        src = np.ones_like(r)
        tgt = src + 0.5
        cmp = remap_engine.compare_profiles(
            field="p",
            source_r=r,
            source_v=src,
            target_r=r,
            target_v=tgt,
            r_max=1.0,
            source_time=0.001,
            target_time=0.002,
        )
        self.assertFalse(cmp.synchronized)
        self.assertIn("not synchronized", cmp.message)
        self.assertAlmostEqual(cmp.mae, 0.5)
        self.assertAlmostEqual(cmp.rms, 0.5)
        self.assertAlmostEqual(cmp.max_abs, 0.5)

    def test_relative_near_zero_is_none(self):
        r = np.array([0.1, 0.2])
        cmp = remap_engine.compare_profiles(
            field="p",
            source_r=r,
            source_v=np.array([0.0, 1.0]),
            target_r=r,
            target_v=np.array([1.0, 1.1]),
            r_max=1.0,
            source_time=0.0,
            target_time=0.0,
        )
        self.assertIsNone(cmp.rel_diff[0])
        self.assertIsNotNone(cmp.rel_diff[1])

    def test_spherical_mass_uniform(self):
        r = np.linspace(0.0, 1.0, 21)
        rho = np.ones_like(r)
        mass = remap_engine.spherical_mass(r, rho)
        self.assertAlmostEqual(mass, 4.0 / 3.0 * np.pi, places=2)


if __name__ == "__main__":
    unittest.main()
