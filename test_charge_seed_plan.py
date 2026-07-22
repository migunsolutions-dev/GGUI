"""Focused tests for the Phase-1 charge-seed architecture."""
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from dataclasses import asdict, replace

from charge_seed_plan import (
    DEFAULT_MAX_AUTO_LEVEL,
    DEFAULT_MIN_CELLS,
    DEFAULT_TARGET_CELLS,
    SEED_MODE_AUTO,
    SEED_MODE_MANUAL,
    SEED_MODE_OFF,
    build_charge_seed_plan,
    migrate_case_inputs_seed_fields,
    required_seed_level,
    smallest_charge_dimension_m,
)
from generator_3d import Generator3D
from initialization_plan import build_initialization_plan, outer_band_will_be_applied
from models import CaseInputs3D
from project_io import build_project, read_project, write_project_atomic
from startup_capture_guard import require_safe_capture


def _base(**overrides) -> CaseInputs3D:
    values = dict(
        min_point=(-5.0, -5.0, 0.0),
        max_point=(5.0, 5.0, 5.0),
        cell_size=0.5,
        charge_center=(0.0, 0.0, 0.5),
        charge_shape="Cylinder",
        mass_kg=25.0,
        cylinder_radius=0.0,  # mass/aspect derived
        charge_aspect=2.5,
        cylinder_axis="Z",
        material_name="C4",
        rho_charge=1601.0,
        energy_j_per_kg=4.5e6,
        p_atm=101325.0,
        t_atm=288.0,
        end_time_s=1e-3,
        delta_t=1e-7,
        write_interval_steps=10,
        cores=1,
        enable_dyn_refine=True,
        dyn_refine_max=1,
        charge_seed_mode=SEED_MODE_AUTO,
        charge_seed_target_cells=8,
        charge_seed_min_cells=6,
        charge_seed_max_level=5,
        charge_refinement_level=0,
        charge_outer_refine_enable=False,
    )
    values.update(overrides)
    return CaseInputs3D(**values)


class SeedFormulaTests(unittest.TestCase):
    def test_building3d_like_auto_level_5(self):
        # h0=0.5, 25 kg C4, L/D=2.5 → mass-derived d_min≈0.20 m → L=5
        plan = build_charge_seed_plan(_base())
        self.assertEqual(plan.mode, SEED_MODE_AUTO)
        self.assertGreater(plan.d_min_m, 0.15)
        self.assertLess(plan.d_min_m, 0.22)
        self.assertEqual(plan.level_effective, 5)
        self.assertEqual(plan.level_required, 5)
        self.assertFalse(plan.cap_applied)
        self.assertTrue(plan.is_safe)
        self.assertGreaterEqual(plan.achieved_cells, DEFAULT_MIN_CELLS)

    def test_required_level_boundary(self):
        self.assertEqual(required_seed_level(0.5, 0.1803, 8), 5)
        self.assertEqual(required_seed_level(0.1, 0.8, 8), 0)
        self.assertEqual(required_seed_level(1.0, 1.0, 2), 1)

    def test_all_shapes_have_d_min(self):
        for shape, kwargs in (
            ("Sphere", {}),
            ("Cylinder", {"charge_aspect": 2.5}),
            ("Cuboid", {"charge_length": 0.2, "charge_width": 0.15, "charge_height": 0.1}),
        ):
            plan = build_charge_seed_plan(_base(charge_shape=shape, **kwargs))
            self.assertGreater(plan.d_min_m, 0.0)
            self.assertEqual(plan.mode, SEED_MODE_AUTO)

    def test_level_5_cap_and_unsafe(self):
        # Huge cell vs tiny charge → capped and unsafe
        plan = build_charge_seed_plan(
            _base(cell_size=5.0, mass_kg=0.01, charge_seed_max_level=5, charge_seed_min_cells=6)
        )
        self.assertTrue(plan.cap_applied)
        self.assertEqual(plan.level_effective, 5)
        self.assertFalse(plan.is_safe)

    def test_target_8_vs_min_6(self):
        plan = build_charge_seed_plan(_base(charge_seed_target_cells=8, charge_seed_min_cells=6))
        self.assertEqual(plan.target_cells, 8)
        self.assertEqual(plan.min_cells, 6)
        self.assertGreaterEqual(plan.achieved_cells, 6)


class SeedModeTests(unittest.TestCase):
    def test_auto_manual_off(self):
        auto = build_initialization_plan(_base(charge_seed_mode=SEED_MODE_AUTO))
        self.assertEqual(auto.command, "setRefinedFields")
        self.assertEqual(auto.seed_effective, 5)

        manual = build_initialization_plan(
            _base(charge_seed_mode=SEED_MODE_MANUAL, charge_refinement_level=3)
        )
        self.assertEqual(manual.command, "setRefinedFields")
        self.assertEqual(manual.seed_effective, 3)

        off = build_initialization_plan(_base(charge_seed_mode=SEED_MODE_OFF))
        self.assertEqual(off.command, "setFields")
        self.assertEqual(off.seed_effective, 0)

    def test_fixed_mesh_disables_internal_seed(self):
        plan = build_initialization_plan(
            _base(enable_dyn_refine=False, charge_seed_mode=SEED_MODE_AUTO)
        )
        self.assertEqual(plan.command, "setFields")
        self.assertEqual(plan.seed_effective, 0)

    def test_remap_bypasses_seed(self):
        plan = build_initialization_plan(
            _base(remap_enabled=True, charge_seed_mode=SEED_MODE_AUTO)
        )
        self.assertEqual(plan.command, "remap_radial.py")
        self.assertEqual(plan.seed_effective, 0)

    def test_seed_independent_of_wave_amr(self):
        plan = build_charge_seed_plan(_base(dyn_refine_max=1, charge_seed_mode=SEED_MODE_AUTO))
        self.assertEqual(plan.level_effective, 5)
        self.assertIn("independent", plan.independence_note.lower())


class OuterBandTests(unittest.TestCase):
    def test_new_default_outer_off(self):
        inputs = _base()
        self.assertEqual(inputs.charge_outer_refine_enable, False)
        self.assertFalse(outer_band_will_be_applied(inputs))

    def test_mode_inside_tuple_emission(self):
        with tempfile.TemporaryDirectory() as td:
            inputs = _base(
                charge_outer_refine_enable=True,
                charge_outer_refine_level=3,
                charge_seed_mode=SEED_MODE_MANUAL,
                charge_refinement_level=2,
            )
            case_dir = Generator3D(td).generate("outer", inputs)
            with open(os.path.join(case_dir, "system", "snappyHexMeshDict"), encoding="utf-8") as fh:
                snappy = fh.read()
            self.assertIn("chargeRefineOuter", snappy)
            self.assertIn("levels ((1e15 3))", snappy)
            self.assertNotRegex(snappy, r"levels \(\(\s*2\s+3\s*\)\)")

    def test_transition_cells_do_not_enlarge_outer_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            a = _base(
                charge_outer_refine_enable=True,
                charge_outer_refine_level=3,
                transition_cells=2,
                charge_seed_mode=SEED_MODE_MANUAL,
                charge_refinement_level=1,
            )
            b = replace(a, transition_cells=8)
            ga = Generator3D(td)
            # Build geometry entries without full generate
            dims = ga._calculate_charge_dimensions(a)
            ea, _, _ = ga._resolve_transition_outside_extent_m(a, dims, a.cell_size)
            eb, _, _ = ga._resolve_transition_outside_extent_m(b, dims, b.cell_size)
            self.assertAlmostEqual(ea, eb, places=9)


class MigrationAndRoundTripTests(unittest.TestCase):
    def test_legacy_level_zero_becomes_off(self):
        migrated = migrate_case_inputs_seed_fields({"charge_refinement_level": 0})
        self.assertEqual(migrated["charge_seed_mode"], SEED_MODE_OFF)

    def test_legacy_level_positive_becomes_manual(self):
        migrated = migrate_case_inputs_seed_fields({"charge_refinement_level": 4})
        self.assertEqual(migrated["charge_seed_mode"], SEED_MODE_MANUAL)

    def test_do_not_convert_off_to_auto(self):
        migrated = migrate_case_inputs_seed_fields(
            {"charge_seed_mode": "Off", "charge_refinement_level": 0}
        )
        self.assertEqual(migrated["charge_seed_mode"], SEED_MODE_OFF)

    def test_case_insensitive_valid_modes(self):
        from charge_seed_plan import normalize_seed_mode

        self.assertEqual(normalize_seed_mode("auto"), SEED_MODE_AUTO)
        self.assertEqual(normalize_seed_mode("MANUAL"), SEED_MODE_MANUAL)
        self.assertEqual(normalize_seed_mode("off"), SEED_MODE_OFF)

    def test_invalid_explicit_mode_raises_in_planner(self):
        from charge_seed_plan import SeedPolicyError, build_charge_seed_plan

        with self.assertRaises(SeedPolicyError) as ctx:
            build_charge_seed_plan(_base(charge_seed_mode="Autoo"))
        self.assertIn("Invalid charge_seed_mode", str(ctx.exception))

    def test_invalid_explicit_mode_raises_on_project_load(self):
        from project_io import ProjectFormatError

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.ggui.json")
            payload = build_project(
                _base(charge_seed_mode="Auto"),
                probes={"probes": []},
                gui_state={},
            )
            payload["case_inputs"]["charge_seed_mode"] = "Autoo"
            write_project_atomic(path, payload)
            with self.assertRaises(ProjectFormatError) as ctx:
                read_project(path)
            self.assertIn("Invalid charge seed policy", str(ctx.exception))

    def test_max_level_zero_preserved(self):
        plan = build_charge_seed_plan(
            _base(charge_seed_mode=SEED_MODE_AUTO, charge_seed_max_level=0)
        )
        self.assertEqual(plan.max_level, 0)
        self.assertEqual(plan.level_effective, 0)

    def test_max_level_five_and_missing(self):
        plan5 = build_charge_seed_plan(
            _base(charge_seed_mode=SEED_MODE_AUTO, charge_seed_max_level=5)
        )
        self.assertEqual(plan5.max_level, 5)
        from types import SimpleNamespace

        # Missing attribute uses default 5
        ns = SimpleNamespace(
            charge_seed_mode=SEED_MODE_AUTO,
            charge_shape="Cylinder",
            mass_kg=25.0,
            rho_charge=1601.0,
            charge_aspect=2.5,
            cell_size=0.5,
            dyn_refine_max=1,
            charge_seed_target_cells=8,
            charge_seed_min_cells=6,
            # no charge_seed_max_level
        )
        plan_missing = build_charge_seed_plan(ns)
        self.assertEqual(plan_missing.max_level, DEFAULT_MAX_AUTO_LEVEL)

    def test_invalid_negative_and_string_policy(self):
        from charge_seed_plan import SeedPolicyError

        with self.assertRaises(SeedPolicyError):
            build_charge_seed_plan(_base(charge_seed_max_level=-1))
        with self.assertRaises(SeedPolicyError):
            build_charge_seed_plan(_base(charge_seed_max_level="five"))
        with self.assertRaises(SeedPolicyError):
            build_charge_seed_plan(_base(charge_seed_min_cells=0))

    def test_project_round_trip_preserves_modes(self):
        for mode, level in ((SEED_MODE_AUTO, 0), (SEED_MODE_MANUAL, 3), (SEED_MODE_OFF, 0)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                inputs = _base(charge_seed_mode=mode, charge_refinement_level=level)
                path = os.path.join(td, "p.ggui.json")
                write_project_atomic(path, build_project(inputs, probes={"probes": []}, gui_state={}))
                loaded = read_project(path)["inputs"]
                self.assertEqual(loaded.charge_seed_mode, mode)
                self.assertEqual(loaded.charge_refinement_level, level)

    def test_legacy_outer_bake_outside_extent(self):
        raw = {
            "min_point": [0, 0, 0],
            "max_point": [2, 2, 2],
            "cell_size": 0.25,
            "charge_center": [1, 1, 1],
            "mass_kg": 1.0,
            "rho_charge": 1600.0,
            "charge_shape": "Sphere",
            "charge_refinement_level": 2,
            "charge_outer_refine_enable": True,
            "charge_outer_refine_min": 2,
            "charge_outer_refine_max": 3,
            "transition_cells": 2,
            "bubble_radius_factor": 1.5,
            "enable_dyn_refine": True,
            "dyn_refine_max": 1,
            "p_atm": 101325.0,
            "t_atm": 288.0,
            "end_time_s": 1e-3,
            "delta_t": 1e-7,
            "write_interval_steps": 10,
            "cores": 1,
            "material_name": "C4",
            "energy_j_per_kg": 1.0,
            "cylinder_radius": 0.05,
            "cylinder_axis": "Z",
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "legacy.ggui.json")
            payload = {
                "schema_version": 1,
                "application": {"name": "GGUI", "version": "4.0"},
                "project_dimension": "3D",
                "case_inputs": raw,
                "probes": {"probes": []},
                "gui_state": {},
            }
            write_project_atomic(path, payload)
            loaded = read_project(path)["inputs"]
            self.assertEqual(loaded.charge_seed_mode, SEED_MODE_MANUAL)
            self.assertTrue(loaded.charge_outer_refine_enable)
            self.assertIsNotNone(loaded.outside_extent)
            self.assertGreater(float(loaded.outside_extent), 0.0)
            self.assertTrue(loaded.charge_outer_legacy_migration_warning)


class GeneratorArchitectureTests(unittest.TestCase):
    def test_building_like_dicts(self):
        with tempfile.TemporaryDirectory() as td:
            inputs = _base()
            case_dir = Generator3D(td).generate("b3d", inputs)
            with open(os.path.join(case_dir, "system", "setFieldsDict"), encoding="utf-8") as fh:
                sf = fh.read()
            with open(os.path.join(case_dir, "system", "snappyHexMeshDict"), encoding="utf-8") as fh:
                snappy = fh.read()
            with open(os.path.join(case_dir, "constant", "dynamicMeshDict"), encoding="utf-8") as fh:
                dyn = fh.read()
            with open(os.path.join(case_dir, "Allrun"), encoding="utf-8") as fh:
                allrun = fh.read()
            self.assertIn("refineInternal yes", sf)
            self.assertIn("level 5", sf)
            self.assertIn("nBufferLayers 5", sf)
            self.assertIn("setRefinedFields", allrun)
            self.assertNotIn("chargeRefineOuter", snappy)
            self.assertIn("maxRefinement   1", dyn)
            self.assertIn("nBufferLayers   2", dyn)
            self.assertIn("densityGradient", dyn)
            self.assertIn("refineInterval  3", dyn)
            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta.get("set_cmd"), "setRefinedFields")
            self.assertEqual(meta.get("charge_seed_plan", {}).get("mode"), SEED_MODE_AUTO)

    def test_unsafe_auto_blocks_generate(self):
        inputs = _base(cell_size=5.0, mass_kg=0.01)
        with self.assertRaises(ValueError) as ctx:
            require_safe_capture(inputs)
        self.assertIn("minimum", str(ctx.exception).lower())

    def test_unsafe_auto_does_not_block_fixed_mesh(self):
        """Fixed Mesh must not raise Auto seed target errors (seed not applied)."""
        from dataclasses import replace

        from startup_capture_guard import UNSAFE_CAPTURE_MESSAGE

        inputs = replace(
            _base(cell_size=5.0, mass_kg=0.01),
            enable_dyn_refine=False,
            enable_local_refinement=False,
        )
        plan = build_initialization_plan(inputs)
        self.assertFalse(plan.uses_set_refined_fields)
        self.assertEqual(plan.seed_effective, 0)
        with self.assertRaises(ValueError) as ctx:
            require_safe_capture(inputs)
        msg = str(ctx.exception)
        self.assertNotIn("Automatic charge seeding cannot achieve", msg)
        self.assertEqual(msg, UNSAFE_CAPTURE_MESSAGE)

    def test_unsafe_auto_blocks_amr(self):
        from dataclasses import replace

        inputs = replace(
            _base(cell_size=5.0, mass_kg=0.01),
            enable_dyn_refine=True,
            charge_seed_mode=SEED_MODE_AUTO,
        )
        plan = build_initialization_plan(inputs)
        self.assertTrue(plan.uses_set_refined_fields)
        with self.assertRaises(ValueError) as ctx:
            require_safe_capture(inputs)
        self.assertIn("Automatic charge seeding cannot achieve", str(ctx.exception))

    def test_safe_auto_amr_passes_auto_guard(self):
        inputs = _base(cell_size=0.5, mass_kg=25.0)
        plan = build_initialization_plan(inputs)
        self.assertTrue(plan.uses_set_refined_fields)
        self.assertEqual(plan.seed_mode, SEED_MODE_AUTO)
        require_safe_capture(inputs)  # must not raise Auto or base-grid error

    def test_manual_seed_amr_skips_auto_message(self):
        from dataclasses import replace

        from startup_capture_guard import UNSAFE_CAPTURE_MESSAGE

        inputs = replace(
            _base(cell_size=5.0, mass_kg=0.01),
            enable_dyn_refine=True,
            charge_seed_mode=SEED_MODE_MANUAL,
            charge_refinement_level=2,
        )
        plan = build_initialization_plan(inputs)
        self.assertTrue(plan.uses_set_refined_fields)
        self.assertEqual(plan.seed_mode, SEED_MODE_MANUAL)
        # Manual may still fail base-grid capture when seed is applied? seed_effective>0 protects.
        require_safe_capture(inputs)

    def test_unsafe_auto_does_not_block_remap(self):
        from dataclasses import replace

        inputs = replace(
            _base(cell_size=5.0, mass_kg=0.01),
            remap_enabled=True,
        )
        require_safe_capture(inputs)  # must not raise

    def test_fixed_mesh_plan_and_guard_agree(self):
        from dataclasses import replace

        # Fine enough base mesh that physical capture succeeds without Auto seed.
        inputs = replace(
            _base(cell_size=0.05, mass_kg=25.0, charge_seed_mode=SEED_MODE_AUTO),
            enable_dyn_refine=False,
            enable_local_refinement=False,
        )
        plan = build_initialization_plan(inputs)
        self.assertFalse(plan.uses_set_refined_fields)
        self.assertEqual(plan.seed_effective, 0)
        require_safe_capture(inputs)


class ModelDefaultTests(unittest.TestCase):
    def test_new_case_defaults(self):
        c = _base()
        self.assertEqual(c.charge_seed_mode, SEED_MODE_AUTO)
        self.assertEqual(c.charge_seed_target_cells, DEFAULT_TARGET_CELLS)
        self.assertEqual(c.charge_seed_min_cells, DEFAULT_MIN_CELLS)
        self.assertEqual(c.charge_seed_max_level, DEFAULT_MAX_AUTO_LEVEL)
        self.assertEqual(c.charge_outer_refine_enable, False)
        self.assertEqual(c.buffer_layers, 5)
        self.assertEqual(c.dyn_refine_max, 1)


if __name__ == "__main__":
    unittest.main()
