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
            self.assertIn("set -eo pipefail", command)
            self.assertIn("| tee log.blastFoam", command)
            self.assertIn("PIPESTATUS[0]", command)
            self.assertIn('exit "$solver_rc"', command)

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
            self.assertIn("unrefineInterval 3", dynamic)
            self.assertIn("refineInterval 3", dynamic)
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
                self.assertNotIn("rotateFields", command)
                self.assertIn("postProcess -func writeCellCentres", command)
                self.assertIn("python3 remap_2d.py", command)
                self.assertEqual(" -refine" in command, expects_refine)
                self.assertNotIn("setRefinedFields", command)
                self.assertNotIn("setFields &&", command)
                allrun = _read(case, "Allrun")
                self.assertIn("set -eo pipefail", allrun)
                self.assertIn("| tee log.blastFoam", allrun)
                self.assertIn("PIPESTATUS[0]", allrun)
                self.assertIn("regions ();", _read(case, "system/setFieldsDict"))
                self.assertIn("from remap_fields_2d import run_case_remap", _read(case, "remap_2d.py"))
                self.assertIn("hob=0.5,\n", _read(case, "remap_2d.py"))
                self.assertTrue(os.path.isfile(os.path.join(case, "remap_fields_2d.py")))
                self.assertTrue(os.path.isfile(os.path.join(case, "remap_snapshot_1d.py")))

    def test_remap_metadata_and_validation_plan_use_user_hob(self):
        from models_2d import HOB_SOURCE_USER_TARGET
        from validation.sampling_io import read_sampling_plan
        from validation.ufc_airblast import BURST_HEMISPHERICAL, BURST_SPHERICAL

        mapping = MappingSource2D(
            case_path="/tmp/source",
            time_mode="latest",
            mapped_radius=0.4,
        )
        with tempfile.TemporaryDirectory() as td:
            _, elevated = self._generate(
                td,
                "remap_elevated",
                initialization_source=REMAP_SOURCE,
                mapping=mapping,
                height_of_burst=1.25,
                mass_kg=1.0,
                radius=2.0,
                height=2.0,
            )
            meta = json.loads(_read(elevated, "case_2d.json"))
            self.assertEqual(meta["hob_source"], HOB_SOURCE_USER_TARGET)
            self.assertAlmostEqual(meta["height_of_burst_m"], 1.25)
            self.assertEqual(meta["charge_center"], [0.0, 1.25, 0.0])
            self.assertEqual(meta["remap_region"]["center"], [0.0, 1.25, 0.0])
            self.assertEqual(meta["remap_region"]["mapping_method"], "radial_from_target_charge_center")
            self.assertEqual(meta["remap_region"]["ground_clip"], "domain_z_ge_0_no_mirror")
            self.assertEqual(meta["remap_timing"]["target_time_label"], "0")
            self.assertIn("hob=1.25,\n", _read(elevated, "remap_2d.py"))
            self.assertIn("points ((0 1.25 0));", _read(elevated, "constant/phaseProperties"))
            plan = read_sampling_plan(elevated)
            self.assertIsNotNone(plan)
            self.assertEqual(plan.burst_master, BURST_SPHERICAL)
            self.assertAlmostEqual(plan.line_z, 1.25)
            self.assertEqual(list(plan.charge_center), [0.0, 1.25, 0.0])
            self.assertTrue(all(abs(p.y - 1.25) < 1e-12 for p in plan.points))

            _, surface = self._generate(
                td,
                "remap_surface",
                initialization_source=REMAP_SOURCE,
                mapping=mapping,
                height_of_burst=0.0,
                mass_kg=1.0,
                radius=2.0,
                height=1.5,
            )
            surface_meta = json.loads(_read(surface, "case_2d.json"))
            self.assertAlmostEqual(surface_meta["height_of_burst_m"], 0.0)
            self.assertEqual(surface_meta["hob_source"], HOB_SOURCE_USER_TARGET)
            self.assertEqual(surface_meta["charge_center"], [0.0, 0.0, 0.0])
            self.assertEqual(surface_meta["remap_region"]["center"], [0.0, 0.0, 0.0])
            self.assertIn("hob=0,\n", _read(surface, "remap_2d.py"))
            surface_plan = read_sampling_plan(surface)
            self.assertIsNotNone(surface_plan)
            self.assertEqual(surface_plan.burst_master, BURST_HEMISPHERICAL)
            self.assertEqual(surface_plan.figure, "2-15")
            self.assertAlmostEqual(surface_plan.line_z, 0.0)

    def test_probe_maps_radius_height_to_wedge_centre_plane(self):
        with tempfile.TemporaryDirectory() as td:
            _, case = self._generate(
                td,
                "probe",
                probes=(ProbePoint2D("P1", 0.2, 0.7),),
            )
            control = _read(case, "system/controlDict")
            self.assertIn("(0.2 0.7 0)", control)

    def test_shared_write_frequency_and_retention_cycle(self):
        with tempfile.TemporaryDirectory() as td:
            _, kept = self._generate(
                td,
                "kept",
                write_control_type="timeStep",
                write_interval_steps=37,
                keep_openfoam_time_folders=True,
                cycle_write=6,
                probes=(ProbePoint2D("g1", 0.2, 0.4),),
                enable_impulse=True,
                enable_dynamic_pressure=True,
            )
            control = _read(kept, "system/controlDict")
            self.assertIn("writeControl timeStep;", control)
            self.assertIn("writeInterval 37;", control)
            self.assertIn("purgeWrite 6;", control)
            self.assertIn("executeControl  timeStep;", control)
            self.assertIn("writeControl    writeTime;", control)

            _, normal = self._generate(
                td,
                "normal",
                write_control_type="adjustableRunTime",
                write_interval_time=0.00023,
                keep_openfoam_time_folders=False,
                cycle_write=6,
                probes=(ProbePoint2D("g1", 0.2, 0.4),),
                enable_impulse=True,
                enable_dynamic_pressure=True,
            )
            normal_control = _read(normal, "system/controlDict")
            self.assertIn("writeControl adjustableRunTime;", normal_control)
            self.assertIn("writeInterval 0.00023;", normal_control)
            self.assertIn("purgeWrite 0;", normal_control)
            self.assertIn("executeControl  timeStep;", normal_control)
            self.assertIn("writeControl    writeTime;", normal_control)

    def test_remap_and_gauges_are_independent_of_native_retention(self):
        with tempfile.TemporaryDirectory() as td:
            _, case = self._generate(
                td,
                "selected_outputs",
                keep_openfoam_time_folders=False,
                output_remap_data=True,
                probes=(ProbePoint2D("g1", 0.2, 0.4),),
            )
            control = _read(case, "system/controlDict")
            self.assertIn("remapDump", control)
            self.assertIn("writeControl    onEnd;", control)
            self.assertIn("probes2d", control)
            self.assertIn("writeInterval 1;", control)

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
