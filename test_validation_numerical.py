"""Numerical diagnostics from logs, checkMesh, and output options."""
from __future__ import annotations

import os
import tempfile
import unittest

from output_options import OutputFileOptions
from validation import numerical as numerical_engine


class LogParseTests(unittest.TestCase):
    def test_parses_courant_deltat_and_refine(self):
        text = """
Time = 0.001
deltaT = 1e-6
Courant Number mean: 0.1 max: 0.4
Refined from 1000 to 1800 cells
ExecutionTime = 12.5 s  ClockTime = 13 s
End
"""
        parsed = numerical_engine.parse_solver_log(text)
        self.assertEqual(parsed["times"], [0.001])
        self.assertEqual(parsed["delta_t"], [1e-6])
        self.assertEqual(parsed["courant_max"], [0.4])
        self.assertEqual(parsed["refine_events"], 1)
        self.assertTrue(parsed["completed"])
        self.assertFalse(parsed["foam_fatal"])

    def test_fatal_and_fpe_flags(self):
        parsed = numerical_engine.parse_solver_log("FOAM FATAL ERROR\nfloating point exception\n")
        self.assertTrue(parsed["foam_fatal"])
        self.assertTrue(parsed["fpe"])


class ReportTests(unittest.TestCase):
    def test_missing_case_is_na_not_pass(self):
        report = numerical_engine.build_report("", dim="2d")
        self.assertEqual(report.run_status, "N/A")
        self.assertIsNone(report.foam_fatal)
        self.assertTrue(any("required validation data" in n for n in report.notes))
        self.assertIsNone(report.checkmesh_ok)

    def test_keep_times_off_is_not_missing(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "system"))
            opts = OutputFileOptions()
            opts.dim2d.output_remap_data = False
            rows = numerical_engine.completeness(
                td, dim="2d", options=opts, keep_openfoam_time_folders=False
            )
            mapped = dict(rows)
            self.assertIn("not requested (Keep OpenFOAM time folders = Off)", mapped["OpenFOAM time folders"])

    def test_report_reads_log(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "system"))
            with open(os.path.join(td, "system", "controlDict"), "w", encoding="utf-8") as handle:
                handle.write("maxCo 0.5;\nendTime 0.01;\nstartTime 0;\n")
            with open(os.path.join(td, "log.blastFoam"), "w", encoding="utf-8") as handle:
                handle.write("Time = 0.001\ndeltaT = 1e-6\nCourant Number mean: 0.1 max: 0.3\nEnd\n")
            with open(os.path.join(td, "log.checkMesh"), "w", encoding="utf-8") as handle:
                handle.write("nCells: 42\nMesh OK\n")
            report = numerical_engine.build_report(td, dim="2d", keep_openfoam_time_folders=True)
            self.assertEqual(report.run_status, "completed")
            self.assertFalse(report.foam_fatal)
            self.assertTrue(report.checkmesh_ok)
            self.assertEqual(report.n_cells, 42)
            self.assertAlmostEqual(report.max_co_configured, 0.5)
            self.assertEqual(report.courant.values, [0.3])


if __name__ == "__main__":
    unittest.main()
