from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace

from axisymmetric_2d import (
    DYNAMIC_MESH,
    FIXED_MESH,
    REMAP_SOURCE,
    align_axisymmetric_domain,
    validate_case_inputs_2d,
    validate_mapping_source,
)
from models_2d import CaseInputs2D, MappingSource2D, ProbePoint2D


class Axisymmetric2DValidationTests(unittest.TestCase):
    def test_exact_fixed_cell_count_and_alignment(self):
        # ceil policy: never shrink below the requested minimum dimensions.
        domain = align_axisymmetric_domain(1.03, 2.02, 0.1)
        self.assertEqual((domain.radial_cells, domain.vertical_cells), (11, 21))
        self.assertEqual(domain.total_cells, 231)
        self.assertGreaterEqual(domain.effective_radius, 1.03 - 1e-12)
        self.assertGreaterEqual(domain.effective_height, 2.02 - 1e-12)
        self.assertTrue(domain.adjusted)

    def test_axis_coordinates_are_locked(self):
        result = validate_case_inputs_2d(CaseInputs2D(charge_center_r=0.1))
        self.assertFalse(result.valid)
        self.assertIn("locked to 0", "\n".join(result.errors))
        result = validate_case_inputs_2d(CaseInputs2D(detonation_radius=0.1))
        self.assertFalse(result.valid)

    def test_sphere_and_axial_cylinder(self):
        sphere = validate_case_inputs_2d(CaseInputs2D())
        cylinder = validate_case_inputs_2d(
            replace(CaseInputs2D(), charge_shape="Cylinder", charge_aspect=2.5)
        )
        self.assertTrue(sphere.valid, sphere.errors)
        self.assertTrue(cylinder.valid, cylinder.errors)
        self.assertEqual(cylinder.charge.shape, "Cylinder")
        self.assertGreater(cylinder.charge.length_m, 0)

    def test_fixed_mesh_blocks_underresolved_charge_without_refining(self):
        inputs = replace(CaseInputs2D(), mesh_mode=FIXED_MESH, cell_size=0.05)
        result = validate_case_inputs_2d(inputs)
        self.assertFalse(result.valid)
        self.assertIn("Fixed Mesh cannot resolve", "\n".join(result.errors))

    def test_startup_and_runtime_levels_are_independent(self):
        inputs = replace(
            CaseInputs2D(),
            mesh_mode=DYNAMIC_MESH,
            charge_seed_mode="Manual",
            charge_refinement_level=4,
            dyn_refine_max=1,
        )
        result = validate_case_inputs_2d(inputs)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.seed_plan.level_effective, 4)
        self.assertEqual(inputs.dyn_refine_max, 1)

    def test_probe_coordinates_are_dimension_isolated(self):
        good = replace(
            CaseInputs2D(), probes=(ProbePoint2D("P1", 0.2, 0.3),)
        )
        bad = replace(
            CaseInputs2D(), probes=(ProbePoint2D("P1", -0.2, 0.3),)
        )
        self.assertTrue(validate_case_inputs_2d(good).valid)
        self.assertFalse(validate_case_inputs_2d(bad).valid)

    def test_mapping_missing_fields_is_blocking(self):
        with tempfile.TemporaryDirectory() as source:
            os.makedirs(os.path.join(source, "0.1"))
            os.makedirs(os.path.join(source, "constant"))
            os.makedirs(os.path.join(source, "system"))
            with open(os.path.join(source, "system", "blockMeshDict"), "w") as f:
                f.write("wedge0 { type wedge; }\n")
            with open(os.path.join(source, "constant", "phaseProperties"), "w") as f:
                f.write("phases (c4 air); equationOfState JWL; rho0 1630;\n")
            inputs = replace(
                CaseInputs2D(),
                initialization_source=REMAP_SOURCE,
                mapping=MappingSource2D(
                    case_path=source,
                    time_mode="specific",
                    specific_time="0.1",
                    mapped_radius=0.5,
                    source_resolution=0.01,
                ),
            )
            report = validate_mapping_source(inputs)
            self.assertFalse(report.valid)
            self.assertIn("p", report.missing_fields)
            self.assertFalse(report.conservation_verified)


if __name__ == "__main__":
    unittest.main()
