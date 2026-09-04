"""Stale ggui_validation_sampling.json must not be reused silently."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from validation.auto_points import live_fingerprint, plan_1d, stamp_plan
from validation.sampling_io import (
    SAMPLING_LEGACY,
    SAMPLING_MISMATCH,
    load_matching_plan,
    read_sampling_plan,
    write_sampling_plan,
)
from validation.ufc_airblast import BURST_SPHERICAL, figure_id


class SamplingFingerprintTests(unittest.TestCase):
    def test_stale_sampling_json_rejected_when_mass_does_not_match(self):
        with tempfile.TemporaryDirectory() as td:
            plan = stamp_plan(
                plan_1d(mass_kg=1.0, domain_radius_m=2.0, cell_size=0.05),
                case_path=td,
                cell_size=0.05,
                hob_m=0.0,
            )
            write_sampling_plan(td, plan)
            expected = live_fingerprint(
                dim="1d",
                case_path=td,
                mass_kg=8.0,
                domain_radius_m=2.0,
                hob_m=0.0,
                charge_center=(0.0, 0.0, 0.0),
                cell_size=0.05,
                burst_mode=BURST_SPHERICAL,
                figure=figure_id(BURST_SPHERICAL),
            )
            loaded, msg = load_matching_plan(td, expected)
            self.assertIsNone(loaded)
            self.assertEqual(msg, SAMPLING_MISMATCH)
            stored = read_sampling_plan(td)
            self.assertIsNotNone(stored)
            self.assertAlmostEqual(stored.mass_kg, 1.0)

    def test_legacy_json_without_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            plan = plan_1d(mass_kg=1.0, domain_radius_m=2.0, cell_size=0.05)
            path = os.path.join(td, "ggui_validation_sampling.json")
            payload = {
                "dim": "1d",
                "burst_master": plan.burst_master,
                "figure": plan.figure,
                "mass_kg": plan.mass_kg,
                "charge_center": [0.0, 0.0, 0.0],
                "r_min": plan.r_min,
                "r_max": plan.r_max,
                "z_min": plan.z_min,
                "z_max": plan.z_max,
                "n_points": plan.n_points,
                "line_kind": plan.line_kind,
                "line_z": 0.0,
                "points": [
                    {
                        "point_id": p.point_id,
                        "dim": p.dim,
                        "index": p.index,
                        "range_m": p.range_m,
                        "x": p.x,
                        "y": p.y,
                        "z": p.z,
                    }
                    for p in plan.points
                ],
                "function_object": plan.function_object,
                "data_source": plan.data_source,
                "notes": list(plan.notes),
                "domain_r_max": plan.domain_r_max,
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            expected = live_fingerprint(
                dim="1d",
                case_path=td,
                mass_kg=1.0,
                domain_radius_m=2.0,
                hob_m=0.0,
                charge_center=(0.0, 0.0, 0.0),
                cell_size=0.05,
                burst_mode=BURST_SPHERICAL,
                figure=figure_id(BURST_SPHERICAL),
            )
            loaded, msg = load_matching_plan(td, expected)
            self.assertIsNone(loaded)
            self.assertEqual(msg, SAMPLING_LEGACY)

    def test_matching_fingerprint_is_reused(self):
        with tempfile.TemporaryDirectory() as td:
            plan = stamp_plan(
                plan_1d(mass_kg=1.0, domain_radius_m=2.0, cell_size=0.05),
                case_path=td,
                cell_size=0.05,
                hob_m=0.0,
            )
            write_sampling_plan(td, plan)
            expected = live_fingerprint(
                dim="1d",
                case_path=td,
                mass_kg=1.0,
                domain_radius_m=2.0,
                hob_m=0.0,
                charge_center=(0.0, 0.0, 0.0),
                cell_size=0.05,
                burst_mode=BURST_SPHERICAL,
                figure=figure_id(BURST_SPHERICAL),
            )
            loaded, msg = load_matching_plan(td, expected)
            self.assertEqual(msg, "")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.n_points, plan.n_points)


if __name__ == "__main__":
    unittest.main()
