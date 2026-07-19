"""Tests for tools.validation_harness (read-only collectors)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.validation_harness.collect import collect_case_metrics, case_metrics_from_dict
from tools.validation_harness.compare import compare_to_reference, evaluate_rules
from tools.validation_harness.models import MetricRule
from tools.validation_harness.report import write_experiment_reports


class ValidationHarnessTests(unittest.TestCase):
    def test_collect_from_case_init_mode_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            case = Path(td) / "c1"
            case.mkdir()
            mode = {
                "charge_shape": "Sphere",
                "base_cell_size": 0.2,
                "base_cell_count": 8000,
                "charge_refinement_requested": 4,
                "charge_refinement_effective": 4,
                "set_cmd": "setRefinedFields",
                "charge_capture": {
                    "mode": "auto",
                    "charge_capture_radius_used_m": 0.26,
                    "physical_charge_radius_m": 0.155,
                    "ratio_capture_to_physical": 1.68,
                },
                "amr_written": {
                    "maxRefinement": 3,
                    "errorEstimator_line": "errorEstimator  densityGradient;",
                },
                "startup_mesh_metadata": {
                    "charge_capture_quality": {"nominal_mass_kg": 25.0},
                    "deep_seeding": {"cells_across_charge_estimate": 6.2},
                    "startup_mesh": {"charge_refine_outer_enabled": False},
                    "validation_profile": {"tags": ["deep_seed"]},
                },
            }
            (case / "case_init_mode.json").write_text(json.dumps(mode), encoding="utf-8")
            m = collect_case_metrics(case)
            self.assertEqual(m.startup.nominal_mass_kg, 25.0)
            self.assertEqual(m.startup.seed_level_effective, 4)
            self.assertEqual(m.runtime.max_refinement, 3)
            self.assertIn("deep_seed", m.validation_profile_tags)

    def test_rules_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            r1 = collect_case_metrics(Path(td) / "missing")  # empty case
            r1.case_id = "a"
            r1.startup.mass_ratio = 0.99
            r2 = collect_case_metrics(Path(td) / "missing2")
            r2.case_id = "b"
            r2.startup.mass_ratio = 0.90
            rules = [MetricRule("startup.mass_ratio", "gte", 0.98)]
            res = evaluate_rules(r2, rules)
            self.assertFalse(res[0].passed)
            cmp = compare_to_reference(r2, r1, ["startup.mass_ratio"])
            self.assertIsNotNone(cmp[0].absolute_diff)
            out = Path(td) / "reports"
            paths = write_experiment_reports(out, "T1", [r1, r2], rule_results=res)
            self.assertTrue(paths["metrics_json"].is_file())
            self.assertTrue(paths["summary_csv"].is_file())
            self.assertTrue(paths["report_md"].is_file())

    def test_roundtrip_dict(self) -> None:
        d = {
            "case_id": "x",
            "case_dir": "/tmp/x",
            "startup": {"mass_ratio": 1.0},
            "runtime": {},
            "blast": {"probes": []},
        }
        cm = case_metrics_from_dict(d)
        self.assertEqual(cm.case_id, "x")
        self.assertEqual(cm.startup.mass_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
