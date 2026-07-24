from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace

from axisymmetric_2d import DYNAMIC_MESH, FIXED_MESH, REMAP_SOURCE
from generator_2d import Generator2D
from models_2d import CaseInputs2D, MappingSource2D, ProbePoint2D


def _read(case, relative):
    with open(os.path.join(case, relative), encoding="utf-8") as stream:
        return stream.read()


class Generator2DTests(unittest.TestCase):
    def _generate(self, root, name, **overrides):
        inputs = replace(CaseInputs2D(), **overrides)
        return inputs, Generator2D(root).generate(name, inputs)

    def test_exact_tutorial_wedge_orientation_and_count(self):
        with tempfile.TemporaryDirectory() as td:
            _, case = self._generate(td, "wedge")
            block = _read(case, "system/blockMeshDict")
            self.assertIn("hex (0 1 2 3 0 4 5 3)", block)
            self.assertIn("(30 30 1)", block)
            self.assertEqual(block.count("type wedge;"), 2)
            self.assertIn("outerRadius", block)
            self.assertIn("top", block)

    def test_fixed_sphere_is_static_and_has_no_topology_refinement(self):
        with tempfile.TemporaryDirectory() as td:
            _, case = self._generate(
                td, "fixed_sphere", mesh_mode=FIXED_MESH, cell_size=0.01
            )
            dynamic = _read(case, "constant/dynamicMeshDict")
            fields = _read(case, "system/setFieldsDict")
            command = _read(case, "Allrun")
            self.assertIn("staticFvMesh", dynamic)
            self.assertNotIn("refineInternal", fields)
            self.assertNotIn("setRefinedFields", command)
            self.assertIn("setFields", command)

    def test_fixed_axial_cylinder_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            _, case = self._generate(
                td,
                "fixed_cylinder",
                mesh_mode=FIXED_MESH,
                cell_size=0.01,
                charge_shape="Cylinder",
            )
            fields = _read(case, "system/setFieldsDict")
            self.assertIn("cylinderToCell", fields)
            self.assertIn("p1 (0 ", fields)
            self.assertIn("p2 (0 ", fields)
            self.assertNotIn("refineInternal", fields)

    def test_dynamic_direct_uses_independent_seed_and_runtime_levels(self):
        with tempfile.TemporaryDirectory() as td:
            _, case = self._generate(
                td,
                "dynamic",
                mesh_mode=DYNAMIC_MESH,
                charge_seed_mode="Manual",
                charge_refinement_level=4,
                dyn_refine_max=1,
            )
            fields = _read(case, "system/setFieldsDict")
            dynamic = _read(case, "constant/dynamicMeshDict")
            command = _read(case, "Allrun")
            self.assertIn("level 4", fields)
            self.assertIn("maxRefinement 1", dynamic)
            self.assertIn("unrefineInterval 1", dynamic)
            self.assertIn("errorEstimator densityGradient", dynamic)
            self.assertIn("setRefinedFields", command)

    def test_dynamic_auto_seed_level_zero_uses_setfields_not_setrefinedfields(self):
        """Fine base mesh Auto seed L0 must not invoke setRefinedFields without levels."""
        with tempfile.TemporaryDirectory() as td:
            inputs, case = self._generate(
                td,
                "dynamic_l0",
                mesh_mode=DYNAMIC_MESH,
                radius=1.5,
                height=2.5,
                cell_size=0.005,
                charge_seed_mode="Auto",
            )
            fields = _read(case, "system/setFieldsDict")
            command = Generator2D(td).initialization_command(inputs)
            self.assertNotIn("refineInternal", fields)
            self.assertNotIn("setRefinedFields", command)
            self.assertIn("setFields", command)
            self.assertIn("(300 500 1)", _read(case, "system/blockMeshDict"))

    def test_fixed_mesh_300x500_reports_exact_volume_cells(self):
        with tempfile.TemporaryDirectory() as td:
            _, case = self._generate(
                td,
                "fixed_150k",
                mesh_mode=FIXED_MESH,
                radius=1.5,
                height=2.5,
                cell_size=0.005,
            )
            block = _read(case, "system/blockMeshDict")
            meta = json.loads(_read(case, "case_2d.json"))
            self.assertIn("(300 500 1)", block)
            self.assertEqual(meta["domain"]["radial_cells"], 300)
            self.assertEqual(meta["domain"]["vertical_cells"], 500)
            self.assertEqual(meta["domain"]["total_computational_cells"], 150000)
            self.assertIn("hex (0 1 2 3 0 4 5 3)", block)
            self.assertEqual(block.count("type wedge;"), 2)

    def test_mapping_fixed_and_dynamic_never_seed_direct_charge(self):
        mapping = MappingSource2D(
            case_path="/tmp/source",
            time_mode="specific",
            specific_time="0.1",
            mapped_radius=0.5,
            source_resolution=0.01,
        )
        for mode, expects_refine in ((FIXED_MESH, False), (DYNAMIC_MESH, False)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                inputs, case = self._generate(
                    td,
                    "mapped",
                    initialization_source=REMAP_SOURCE,
                    mapping=mapping,
                    mesh_mode=mode,
                )
                command = Generator2D(td).initialization_command(inputs)
                self.assertIn("rotateFields", command)
                self.assertEqual(" -refine" in command, expects_refine)
                self.assertNotIn("setRefinedFields", command)
                self.assertNotIn("setFields &&", command)
                self.assertIn("regions ();", _read(case, "system/setFieldsDict"))

    def test_probe_maps_radius_height_to_wedge_centre_plane(self):
        with tempfile.TemporaryDirectory() as td:
            _, case = self._generate(
                td,
                "probe",
                probes=(ProbePoint2D("P1", 0.2, 0.7),),
            )
            control = _read(case, "system/controlDict")
            self.assertIn("(0.2 0.7 0)", control)

    def test_mirroring_never_changes_mesh_or_cell_count(self):
        with tempfile.TemporaryDirectory() as td:
            _, mirrored = self._generate(td, "mirrored", mirrored_view=True)
            _, computational = self._generate(
                td, "computational", mirrored_view=False
            )
            self.assertEqual(
                _read(mirrored, "system/blockMeshDict"),
                _read(computational, "system/blockMeshDict"),
            )
            with open(os.path.join(mirrored, "case_2d.json"), encoding="utf-8") as f:
                a = json.load(f)
            with open(os.path.join(computational, "case_2d.json"), encoding="utf-8") as f:
                b = json.load(f)
            self.assertEqual(
                a["domain"]["total_computational_cells"],
                b["domain"]["total_computational_cells"],
            )


if __name__ == "__main__":
    unittest.main()
