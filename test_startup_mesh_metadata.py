"""Unit tests for startup_mesh_metadata (observability only)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from generator_3d import Generator3D
from models import CaseInputs3D
from startup_mesh_metadata import (
    RECOMMENDED_AUTO_SEED_L_MAX,
    RECOMMENDED_CELLS_ACROSS_CHARGE_N,
    build_startup_mesh_metadata,
    flatten_warnings_for_charge_warnings,
    recommended_auto_seed_level,
    smallest_charge_dimension_m,
)


class StartupMeshMetadataTests(unittest.TestCase):
    def test_smallest_dimension_cylinder_diameter_binding(self) -> None:
        d, name = smallest_charge_dimension_m("Cylinder", {"radius": 0.1, "length": 0.5})
        self.assertAlmostEqual(d, 0.2)
        self.assertEqual(name, "diameter")

    def test_auto_seed_recommendation_25kg_02m(self) -> None:
        # 25 kg sphere r~0.155, d_min~0.31, dx=0.2 -> log2(6*0.2/0.31) ~ 0.95 -> ceil 1? 
        # Actually 6*0.2/0.31 = 3.87, log2 = 1.95, ceil = 2
        rec = recommended_auto_seed_level(0.2, 0.31)
        self.assertEqual(rec["target_cells_across_charge_N"], RECOMMENDED_CELLS_ACROSS_CHARGE_N)
        self.assertGreaterEqual(rec["recommended_level"], 0)
        self.assertLessEqual(rec["recommended_level"], RECOMMENDED_AUTO_SEED_L_MAX)

    def test_build_metadata_has_required_sections(self) -> None:
        dims = {"radius": 0.155}
        meta = build_startup_mesh_metadata(
            type("Inp", (), {"cell_size": 0.2, "mass_kg": 25.0})(),
            dims,
            charge_capture={
                "mode": "auto",
                "charge_capture_radius_used_m": 0.26,
                "physical_charge_radius_m": 0.155,
                "ratio_capture_to_physical": 1.68,
            },
            base_cell_count=8000,
            base_cell_size_m=0.2,
            charge_shape="Sphere",
            seed_requested=4,
            seed_effective=4,
            uses_set_refined_fields=True,
            set_cmd="setRefinedFields",
            charge_refine_outer_enabled=True,
            outer_snappy_level_min=2,
            outer_snappy_level_max=3,
            amr_written={"maxRefinement": 3, "errorEstimator_line": "errorEstimator  densityGradient;"},
            transition_region={"outside_extent_m": 0.48, "outside_extent_auto": True},
        )
        self.assertEqual(meta["schema_version"], 1)
        self.assertIn("charge_capture_quality", meta)
        self.assertIn("startup_mesh", meta)
        self.assertIn("backup_region", meta)
        self.assertIn("deep_seeding", meta)
        self.assertIn("auto_seed_recommendation", meta)
        self.assertFalse(meta["auto_seed_recommendation"]["applied"])
        self.assertIn("runtime_planning", meta)
        self.assertIn("validation_profile", meta)
        codes = {w["code"] for w in meta["warnings_structured"]}
        self.assertIn("band_plus_deep_seed", codes)

    def test_case_init_mode_includes_startup_mesh_metadata(self) -> None:
        inp = CaseInputs3D(
            min_point=(-2.0, -2.0, -2.0),
            max_point=(2.0, 2.0, 2.0),
            cell_size=0.2,
            charge_center=(0.0, 0.0, 0.0),
            charge_shape="Sphere",
            mass_kg=25.0,
            cylinder_radius=0.1,
            cylinder_axis="Z",
            material_name="C4",
            rho_charge=1601.0,
            energy_j_per_kg=4.5e6,
            p_atm=101325.0,
            t_atm=300.0,
            end_time_s=1e-4,
            delta_t=1e-7,
            write_interval_steps=10,
            cores=1,
            enable_dyn_refine=True,
            charge_refinement_level=4,
            charge_outer_refine_enable=False,
            charge_capture_mode="auto",
            charge_capture_factor=1.0,
        )
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("meta_phase1", inp)
            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                mode = json.load(f)
        smm = mode.get("startup_mesh_metadata")
        self.assertIsInstance(smm, dict)
        self.assertEqual(smm["charge_capture_quality"]["nominal_mass_kg"], 25.0)
        self.assertIn("validation_profile", smm)
        self.assertFalse(smm["auto_seed_recommendation"]["applied"])


if __name__ == "__main__":
    unittest.main()
