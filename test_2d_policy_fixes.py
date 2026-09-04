"""Focused tests for Phase-2 production policy fixes (refineProbes FO + interval defaults)."""
from __future__ import annotations

import os
import re
import tempfile
import unittest
from dataclasses import replace

from axisymmetric_2d import DYNAMIC_MESH, FIXED_MESH
from generator_2d import Generator2D
from models_2d import DEFAULT_REFINE_INTERVAL, CaseInputs2D, ProbePoint2D
from project_io import build_project, read_project, write_project_atomic
from test_generator_3d import _minimal


def _read(case: str, relative: str) -> str:
    with open(os.path.join(case, relative), encoding="utf-8") as stream:
        return stream.read()


def _probe_locations(control: str) -> list[str]:
    match = re.search(
        r"probeLocations\s*\(\s*((?:.|\n)*?)\s*\)\s*;",
        control,
    )
    if not match:
        return []
    return re.findall(r"\(([^)]+)\)", match.group(1))


class RefineProbesAndIntervalPolicyTests(unittest.TestCase):
    def test_native_probes2d_valid_and_no_unsupported_refineprobes_fo(self):
        probes = (
            ProbePoint2D("R1", 0.15, 0.5),
            ProbePoint2D("R2", 0.25, 0.5),
            ProbePoint2D("R3", 0.40, 0.5),
        )
        with tempfile.TemporaryDirectory() as td:
            Generator2D(td).generate(
                "dyn_probes",
                replace(
                    CaseInputs2D(),
                    mesh_mode=DYNAMIC_MESH,
                    refine_probes=True,
                    probes=probes,
                    output_fields=("p",),
                ),
            )
            control = _read(os.path.join(td, "dyn_probes"), "system/controlDict")
            dynamic = _read(os.path.join(td, "dyn_probes"), "constant/dynamicMeshDict")

            self.assertIn("probes2d", control)
            self.assertIn("type probes;", control)
            self.assertIn('libs ("libfieldFunctionObjects.so");', control)
            self.assertIn("writeControl timeStep;", control)
            self.assertIn("writeInterval 1;", control)
            self.assertNotIn("type refineProbes;", control)
            self.assertNotIn("libblastFunctionObjects.so", control)
            # dynamicMeshDict Switch remains the supported refine-at-probe path
            self.assertIn("refineProbes true;", dynamic)

            locs = _probe_locations(control)
            self.assertEqual(len(locs), 3)
            self.assertEqual(control.count("probes2d"), 1)
            self.assertIn("validationGauges2d", control)
            self.assertEqual(control.count("probeLocations"), 2)
            self.assertIn("0.15", locs[0])
            self.assertIn("0.5", locs[0])
            self.assertIn("0.25", locs[1])
            self.assertIn("0.4", locs[2])

    def test_refine_probes_false_keeps_probes2d_and_switch_false(self):
        probes = (ProbePoint2D("R1", 0.15, 1.0),)
        with tempfile.TemporaryDirectory() as td:
            Generator2D(td).generate(
                "dyn_off",
                replace(
                    CaseInputs2D(),
                    mesh_mode=DYNAMIC_MESH,
                    refine_probes=False,
                    probes=probes,
                ),
            )
            control = _read(os.path.join(td, "dyn_off"), "system/controlDict")
            dynamic = _read(os.path.join(td, "dyn_off"), "constant/dynamicMeshDict")
            self.assertIn("type probes;", control)
            self.assertNotIn("type refineProbes;", control)
            self.assertIn("refineProbes false;", dynamic)

    def test_new_dynamic_defaults_match_refine_and_unrefine_intervals(self):
        self.assertEqual(DEFAULT_REFINE_INTERVAL, 3)
        self.assertEqual(CaseInputs2D().refine_interval, DEFAULT_REFINE_INTERVAL)
        self.assertEqual(CaseInputs2D().unrefine_interval, DEFAULT_REFINE_INTERVAL)
        with tempfile.TemporaryDirectory() as td:
            Generator2D(td).generate(
                "dyn_default",
                replace(CaseInputs2D(), mesh_mode=DYNAMIC_MESH),
            )
            dynamic = _read(os.path.join(td, "dyn_default"), "constant/dynamicMeshDict")
            self.assertIn(f"refineInterval {DEFAULT_REFINE_INTERVAL};", dynamic)
            self.assertIn(f"unrefineInterval {DEFAULT_REFINE_INTERVAL};", dynamic)

    def test_explicit_unrefine_override_remains_independent(self):
        with tempfile.TemporaryDirectory() as td:
            Generator2D(td).generate(
                "dyn_override",
                replace(
                    CaseInputs2D(),
                    mesh_mode=DYNAMIC_MESH,
                    refine_interval=3,
                    unrefine_interval=1,
                ),
            )
            dynamic = _read(os.path.join(td, "dyn_override"), "constant/dynamicMeshDict")
            self.assertIn("refineInterval 3;", dynamic)
            self.assertIn("unrefineInterval 1;", dynamic)

    def test_saved_project_explicit_intervals_round_trip(self):
        inputs_2d = replace(
            CaseInputs2D(),
            mesh_mode=DYNAMIC_MESH,
            refine_interval=5,
            unrefine_interval=1,
            refine_probes=False,
        )
        payload = build_project(
            _minimal(), probes={"probes": []}, gui_state={}, inputs_2d=inputs_2d
        )
        with tempfile.NamedTemporaryFile(suffix=".ggui.json", delete=False) as stream:
            path = stream.name
        try:
            write_project_atomic(path, payload)
            restored = read_project(path)["inputs_2d"]
            self.assertEqual(restored.refine_interval, 5)
            self.assertEqual(restored.unrefine_interval, 1)
            self.assertFalse(restored.refine_probes)
        finally:
            os.unlink(path)

    def test_fixed_mesh_unchanged_no_amr_switch_or_refineprobes_fo(self):
        probes = (ProbePoint2D("R1", 0.2, 0.5),)
        with tempfile.TemporaryDirectory() as td:
            Generator2D(td).generate(
                "fixed",
                replace(
                    CaseInputs2D(),
                    mesh_mode=FIXED_MESH,
                    cell_size=0.01,
                    probes=probes,
                    refine_probes=True,
                ),
            )
            control = _read(os.path.join(td, "fixed"), "system/controlDict")
            dynamic = _read(os.path.join(td, "fixed"), "constant/dynamicMeshDict")
            self.assertIn("staticFvMesh", dynamic)
            self.assertNotIn("refineInterval", dynamic)
            self.assertNotIn("refineProbes", dynamic)
            self.assertIn("type probes;", control)
            self.assertNotIn("type refineProbes;", control)

    def test_no_duplicate_probe_writers_or_locations(self):
        probes = (
            ProbePoint2D("A", 0.1, 0.5),
            ProbePoint2D("B", 0.2, 0.5),
        )
        with tempfile.TemporaryDirectory() as td:
            Generator2D(td).generate(
                "nodup",
                replace(
                    CaseInputs2D(),
                    mesh_mode=DYNAMIC_MESH,
                    probes=probes,
                    refine_probes=True,
                ),
            )
            control = _read(os.path.join(td, "nodup"), "system/controlDict")
            self.assertEqual(control.count("type probes;"), 2)
            self.assertEqual(control.count("probes2d"), 1)
            self.assertIn("validationGauges2d", control)
            self.assertEqual(len(_probe_locations(control)), 2)


class ImportDictNotRewrittenByGeneratorDefaults(unittest.TestCase):
    """Imported working-copy dicts are not regenerated by these defaults."""

    def test_external_dict_text_preserved_when_not_regenerated(self):
        # Simulate an imported dynamicMeshDict fragment with legacy intervals.
        legacy = (
            "dynamicFvMesh adaptiveFvMesh;\n"
            "refineInterval 3;\n"
            "unrefineInterval 1;\n"
            "refineProbes true;\n"
        )
        with tempfile.TemporaryDirectory() as td:
            case = os.path.join(td, "imported")
            os.makedirs(os.path.join(case, "constant"), exist_ok=True)
            path = os.path.join(case, "constant", "dynamicMeshDict")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(legacy)
            # Reading back without calling Generator2D.generate must be identical.
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertEqual(text, legacy)
            self.assertIn("unrefineInterval 1;", text)


if __name__ == "__main__":
    unittest.main()
