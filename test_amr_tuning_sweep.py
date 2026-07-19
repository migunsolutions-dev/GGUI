"""Tests for developer-only tools/amr_tuning_sweep.py (no WSL)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_sweep():
    root = Path(__file__).resolve().parent
    path = root / "tools" / "amr_tuning_sweep.py"
    name = "ggui_amr_tuning_sweep"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class AmrTuningSweepTests(unittest.TestCase):
    def test_parse_blastfoam_refine_counts(self) -> None:
        m = _load_sweep()
        log = """
Time = 0
Refined from 1000 to 1008 cells.
Unrefined from 1008 to 900 cells.
Time = 1e-6
FOAM FATAL
"""
        p = m.parse_blastfoam_log(log)
        self.assertEqual(p["refine_events"], 1)
        self.assertEqual(p["unrefine_events"], 1)
        self.assertTrue(p["foam_fatal"])
        self.assertEqual(p["peak_cell_count"], 1008)

    def test_final_time_ignores_execution_and_clock_time(self) -> None:
        """final_time must come from `Time = X`, not from `ExecutionTime`/`ClockTime`."""
        m = _load_sweep()
        log = """
Time = 1.2e-07
Refined from 1000 to 1008 cells.
ExecutionTime = 2.5 s  ClockTime = 3 s
Time = 4.9e-05
ExecutionTime = 3359.44 s  ClockTime = 3374 s
End
"""
        p = m.parse_blastfoam_log(log)
        self.assertAlmostEqual(p["final_time"], 4.9e-05, places=10)
        self.assertAlmostEqual(p["last_execution_time_s"], 3359.44, places=4)

    def test_cellcount_vs_time_extraction(self) -> None:
        m = _load_sweep()
        log = """
Time = 0
Refined from 100 to 108 cells.
Time = 1e-7
Unrefined from 108 to 100 cells.
"""
        tv = m.extract_cellcount_vs_time(log)
        self.assertEqual(tv, [(0.0, 108), (1e-7, 100)])

    def test_patch_dynamic_mesh_density_and_pressure(self) -> None:
        m = _load_sweep()
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            p = Path(td) / "dynamicMeshDict"
            p.write_text(
                "HEADER\n"
                "dynamicFvMesh   adaptiveFvMesh;\n"
                "errorEstimator  densityGradient;\n"
                "refineInterval  3;\n"
                "lowerRefineLevel 0.1;\n"
                "unrefineLevel   0.1;\n"
                "nBufferLayers   2;\n"
                "maxRefinement   3;\n"
                "dumpLevel      true;\n",
                encoding="utf-8",
            )
            m.patch_dynamic_mesh_dict(
                p,
                estimator="scaledDelta_p",
                n_buffer_layers=0,
                lower_refine_level=0.2,
                unrefine_level=0.05,
            )
            t = p.read_text(encoding="utf-8")
            self.assertIn("scaledDeltaField p;", t)
            self.assertIn("nBufferLayers   0;", t)
            self.assertIn("lowerRefineLevel 0.2;", t)
            self.assertIn("unrefineLevel   0.05;", t)
            self.assertTrue(t.startswith("HEADER"))


if __name__ == "__main__":
    unittest.main()
