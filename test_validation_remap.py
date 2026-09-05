"""1D→2D / 2D→3D remap comparison and conservation helpers."""
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

import numpy as np

from validation import remap as remap_engine
from validation.metrics import rms_error


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

    def test_recorded_handoff_time_beats_widget_latest_and_folder_zero(self):
        with tempfile.TemporaryDirectory() as td:
            source = os.path.join(td, "src1d")
            target = os.path.join(td, "tgt2d")
            leftover = os.path.join(td, "other1d")
            os.makedirs(os.path.join(source, "0"))
            os.makedirs(os.path.join(source, "0.000184005"))
            os.makedirs(os.path.join(leftover, "0"))
            os.makedirs(os.path.join(leftover, "0.005"))
            os.makedirs(os.path.join(target, "0"))
            with open(os.path.join(target, "case_2d.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "mapping": {"case_path": source, "time_mode": "latest"},
                        "remap_timing": {
                            "source_time_label": "0.000184005",
                            "source_physical_time": 0.000184005,
                            "physical_time_offset": 0.000184005,
                            "target_time_label": "0",
                        },
                    },
                    handle,
                )
            src, st, tt, msg = remap_engine.resolve_1d_to_2d(
                target_case=target,
                mapping_source=leftover,
                mapping_time="latest",
            )
            self.assertEqual(msg, "")
            self.assertEqual(os.path.normpath(src), os.path.normpath(source))
            self.assertEqual(st, "0.000184005")
            self.assertEqual(tt, "0")
            src2, st2, tt2, msg2 = remap_engine.resolve_1d_to_2d(
                target_case=target,
                mapping_source=source,
                mapping_time=None,
            )
            self.assertEqual(msg2, "")
            self.assertEqual(st2, "0.000184005")
            self.assertNotEqual(st2, "0")

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
    def _write_scalar_field(self, path, values):
        body = " ".join(f"{float(v)}" for v in values)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                f"internalField   nonuniform List<scalar> {len(values)} ({body});\n"
            )

    def _write_vector_field(self, path, pts):
        chunks = " ".join(f"({p[0]} {p[1]} {p[2]})" for p in pts)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                f"internalField   nonuniform List<vector> {len(pts)} ({chunks});\n"
            )

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

    def test_compare_sorts_target_storage_order_and_uses_absolute_diff(self):
        source_r = np.array([0.1, 0.2, 0.3, 0.4])
        source_v = np.array([1.0, 2.0, 3.0, 4.0])
        target_r = np.array([0.4, 0.1, 0.3, 0.2])
        target_v = np.array([4.5, 1.0, 3.2, 2.1])
        cmp = remap_engine.compare_profiles(
            field="p",
            source_r=source_r,
            source_v=source_v,
            target_r=target_r,
            target_v=target_v,
            r_max=0.4,
            source_time=0.001,
            target_time=0.0,
            physical_time_offset=0.001,
        )
        self.assertTrue(cmp.sorted_in_r)
        self.assertEqual(cmp.n_pairs, 4)
        self.assertTrue(all(cmp.r[i] <= cmp.r[i + 1] for i in range(3)))
        self.assertAlmostEqual(cmp.r[0], 0.1)
        self.assertAlmostEqual(cmp.abs_diff[0], abs(1.0 - 1.0))
        self.assertAlmostEqual(cmp.abs_diff[1], abs(2.1 - 2.0))
        self.assertGreaterEqual(min(cmp.abs_diff), 0.0)
        self.assertAlmostEqual(cmp.remap_radius_m, 0.4)
        self.assertEqual(cmp.field_unit, "Pa")
        self.assertAlmostEqual(cmp.peak_source, 4.0)
        self.assertAlmostEqual(cmp.peak_r_source, 0.4)

    def test_physical_profile_uses_charge_centre_r_not_cylindrical_x(self):
        r, z = np.meshgrid(np.array([0.3, 0.6]), np.array([0.0, 1.0]))
        centres = np.column_stack([r.ravel(), z.ravel(), np.zeros(4)])
        # Values differ by height so cylindrical-x pairing would mix them.
        values = np.array([10.0, 20.0, 30.0, 40.0])
        phys = remap_engine.physical_radius_from_centre(centres, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(phys[0], math.hypot(0.3, 1.0))
        self.assertAlmostEqual(phys[2], 0.3)

    def test_load_physical_profile_keeps_charge_height_ray(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "0"))
            centres = np.array(
                [
                    [0.10, 1.00, 0.0],
                    [0.20, 1.00, 0.0],
                    [0.10, 0.00, 0.0],
                    [0.20, 0.00, 0.0],
                ]
            )
            values = np.array([1.0, 2.0, 9.0, 8.0])
            self._write_vector_field(os.path.join(td, "0", "C"), centres)
            self._write_scalar_field(os.path.join(td, "0", "p"), values)
            r, v, msg = remap_engine.load_physical_radial_profile(
                td,
                "0",
                "p",
                dim="2d",
                charge_centre=(0.0, 1.0, 0.0),
                ray_half_width=0.01,
            )
            self.assertEqual(msg, "")
            self.assertEqual(list(np.round(r, 6)), [0.10, 0.20])
            self.assertEqual(list(v), [1.0, 2.0])
            self.assertTrue(np.all(np.diff(r) >= 0.0))

    def test_charge_height_ray_keeps_only_closest_row(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "0"))
            centres = np.array(
                [
                    [0.10, 1.00, 0.0],
                    [0.20, 1.00, 0.0],
                    [0.10, 1.01, 0.0],
                    [0.20, 1.01, 0.0],
                ]
            )
            values = np.array([1.0, 2.0, 99.0, 88.0])
            self._write_vector_field(os.path.join(td, "0", "C"), centres)
            self._write_scalar_field(os.path.join(td, "0", "p"), values)
            r, v, msg = remap_engine.load_physical_radial_profile(
                td,
                "0",
                "p",
                dim="2d",
                charge_centre=(0.0, 1.0, 0.0),
                ray_half_width=0.01,
            )
            self.assertEqual(msg, "")
            self.assertEqual(list(np.round(r, 6)), [0.10, 0.20])
            self.assertEqual(list(v), [1.0, 2.0])

    def test_compare_clips_to_user_remap_radius(self):
        source_r = np.array([0.1, 0.3, 0.5, 0.7])
        source_v = np.array([4.0, 3.0, 2.0, 1.0])
        target_r = np.array([0.7, 0.5, 0.3, 0.1])
        target_v = np.array([1.1, 2.2, 3.1, 4.0])
        cmp = remap_engine.compare_profiles(
            field="p",
            source_r=source_r,
            source_v=source_v,
            target_r=target_r,
            target_v=target_v,
            r_max=0.6,
            source_time=0.000184,
            target_time=0.0,
            physical_time_offset=0.000184,
        )
        self.assertLessEqual(max(cmp.r), 0.6)
        self.assertEqual(cmp.n_pairs, 3)
        self.assertAlmostEqual(cmp.remap_radius_m, 0.6)
        self.assertEqual(len(cmp.source), len(cmp.r))
        self.assertEqual(len(cmp.target), len(cmp.r))
        self.assertEqual(len(cmp.abs_diff), len(cmp.r))
        expected_rms = rms_error(cmp.target, cmp.source)
        self.assertAlmostEqual(cmp.rms, expected_rms)
        self.assertNotIn(0.7, cmp.r)

    def test_missing_field_is_empty_not_zeros(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "0"))
            centres = np.array([[0.10, 0.0, 0.0], [0.20, 0.0, 0.0]])
            self._write_vector_field(os.path.join(td, "0", "C"), centres)
            r, v, msg = remap_engine.load_physical_radial_profile(td, "0", "T", dim="1d")
            self.assertEqual(list(r), [])
            self.assertEqual(list(v), [])
            self.assertIn("not available", msg)
            cmp = remap_engine.compare_profiles(
                field="T",
                source_r=r,
                source_v=v,
                target_r=r,
                target_v=v,
                r_max=0.6,
                source_time=0.0,
                target_time=0.0,
            )
            self.assertEqual(cmp.n_pairs, 0)
            self.assertIsNone(cmp.rms)
            self.assertIsNone(cmp.mae)
            self.assertIsNone(cmp.max_abs)

    def test_1d_velocity_uses_magnitude_not_wedge_ux(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "0"))
            centres = np.array(
                [
                    [0.30, 0.05, 0.0],
                    [0.60, 0.05, 0.0],
                ]
            )
            u = np.array(
                [
                    [300.0, 50.0, 0.0],
                    [600.0, 80.0, 0.0],
                ]
            )
            self._write_vector_field(os.path.join(td, "0", "C"), centres)
            chunks = " ".join(f"({row[0]} {row[1]} {row[2]})" for row in u)
            with open(os.path.join(td, "0", "U"), "w", encoding="utf-8") as handle:
                handle.write(f"internalField   nonuniform List<vector> {len(u)} ({chunks});\n")
            r, v, msg = remap_engine.load_physical_radial_profile(td, "0", "U", dim="1d")
            self.assertEqual(msg, "")
            self.assertTrue(np.allclose(v, np.linalg.norm(u, axis=1)))
            self.assertFalse(np.allclose(v, u[:, 0]))

    def test_2d_velocity_uses_radial_component_on_charge_height_ray(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "0"))
            centres = np.array(
                [
                    [0.30, 1.00, 0.0],
                    [0.60, 1.00, 0.0],
                    [0.30, 0.00, 0.0],
                ]
            )
            u = np.array(
                [
                    [3.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [99.0, 0.0, 0.0],
                ]
            )
            self._write_vector_field(os.path.join(td, "0", "C"), centres)
            chunks = " ".join(f"({row[0]} {row[1]} {row[2]})" for row in u)
            with open(os.path.join(td, "0", "U"), "w", encoding="utf-8") as handle:
                handle.write(f"internalField   nonuniform List<vector> {len(u)} ({chunks});\n")
            r, v, msg = remap_engine.load_physical_radial_profile(
                td,
                "0",
                "U",
                dim="2d",
                charge_centre=(0.0, 1.0, 0.0),
                ray_half_width=0.01,
            )
            self.assertEqual(msg, "")
            self.assertEqual(list(np.round(r, 6)), [0.30, 0.60])
            self.assertTrue(np.allclose(v, [3.0, 6.0]))


class RemapPhysicalTimeTests(unittest.TestCase):
    def test_target_dir_zero_with_advanced_source_uses_physical_offset(self):
        timing = remap_engine.build_remap_timing(
            source_time_label="0.001",
            target_time_label="0",
        )
        self.assertAlmostEqual(timing.source_physical_time, 0.001)
        self.assertAlmostEqual(timing.target_initial_time, 0.0)
        self.assertAlmostEqual(timing.physical_time_offset, 0.001)
        self.assertAlmostEqual(timing.target_physical(0.0), 0.001)
        r = np.linspace(0.1, 1.0, 10)
        src = np.ones_like(r)
        cmp = remap_engine.compare_profiles(
            field="p",
            source_r=r,
            source_v=src,
            target_r=r,
            target_v=src,
            r_max=1.0,
            source_time=0.001,
            target_time=0.0,
            physical_time_offset=timing.physical_time_offset,
        )
        self.assertTrue(cmp.synchronized)
        self.assertAlmostEqual(cmp.source_physical_time, 0.001)
        self.assertAlmostEqual(cmp.target_physical_time, 0.001)
        self.assertAlmostEqual(cmp.physical_time_offset, 0.001)
        self.assertAlmostEqual(cmp.delta_t, 0.0)

    def test_openfoam_labels_alone_do_not_prove_sync_without_offset(self):
        r = np.linspace(0.1, 1.0, 10)
        cmp = remap_engine.compare_profiles(
            field="p",
            source_r=r,
            source_v=np.ones_like(r),
            target_r=r,
            target_v=np.ones_like(r),
            r_max=1.0,
            source_time=0.001,
            target_time=0.0,
        )
        self.assertFalse(cmp.synchronized)
        self.assertIn("physical times", cmp.message)


if __name__ == "__main__":
    unittest.main()
