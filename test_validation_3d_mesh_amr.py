"""Focused validation tests for 3D domain alignment, transition regions, and AMR dict output."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import unittest

from mesh_domain import align_domain_to_cell_size
from generator_3d import Generator3D
from models import CaseInputs3D


def _case(**kw) -> CaseInputs3D:
    d = dict(
        min_point=(0.0, 0.0, 0.0),
        max_point=(4.0, 4.0, 4.0),
        cell_size=0.5,
        charge_center=(2.0, 2.0, 2.0),
        charge_shape="Sphere",
        mass_kg=25.0,
        cylinder_radius=0.1,
        cylinder_axis="Z",
        material_name="C4",
        rho_charge=1601.0,
        energy_j_per_kg=4.5e6,
        p_atm=101325.0,
        t_atm=300.0,
        end_time_s=1e-3,
        delta_t=1e-7,
        write_interval_steps=10,
        cores=1,
        enable_dyn_refine=True,
        enable_local_refinement=True,
        refine_min=2,
        refine_max=3,
        dyn_refine_max=1,
        charge_refinement_level=2,
        charge_outer_refine_min=2,
        charge_outer_refine_max=3,
        charge_capture_mode="auto",
        charge_capture_factor=1.0,
    )
    d.update(kw)
    return CaseInputs3D(**d)


class DomainAlignmentValidation(unittest.TestCase):
    def test_example_lx_10_cell_0_6(self) -> None:
        r = align_domain_to_cell_size((0.0, 0.0, 0.0), (10.0, 1.0, 1.0), 0.6)
        self.assertEqual(r.nx, 17)
        self.assertAlmostEqual(r.actual_lengths[0], 10.2, places=9)
        self.assertAlmostEqual(r.cell_size, 0.6, places=9)
        self.assertAlmostEqual(r.min_point[0], 0.0)
        self.assertAlmostEqual(r.max_point[0], 10.2)

    def test_blockmesh_matches_alignment(self) -> None:
        inp = _case(min_point=(0.0, 0.0, 0.0), max_point=(10.03, 2.0, 2.0), cell_size=0.5, charge_center=(5.0, 1.0, 1.0))
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("align_bm", inp)
            with open(os.path.join(case_dir, "system", "blockMeshDict"), encoding="utf-8") as f:
                bm = f.read()
            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                mode = json.load(f)
        m = re.search(r"hex\s+\([^)]+\)\s+\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", bm)
        self.assertIsNotNone(m)
        nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
        da = mode.get("domain_alignment") or {}
        self.assertEqual([nx, ny, nz], da.get("n_cells_xyz"))
        self.assertTrue(da.get("domain_adjusted_for_cell_fit"))
        self.assertEqual(mode.get("base_cell_count"), nx * ny * nz)


class SphereTransitionValidation(unittest.TestCase):
    def test_searchable_sphere_outer_radius(self) -> None:
        oe = 0.4
        inp = _case(outside_extent=oe, bubble_radius_factor=99.0)
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("sph_tr", inp)
            with open(os.path.join(case_dir, "system", "snappyHexMeshDict"), encoding="utf-8") as f:
                sn = f.read()
            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                mode = json.load(f)
        self.assertIn("searchableSphere", sn)
        vol = 25.0 / 1601.0
        r = ((3.0 * vol) / (4.0 * math.pi)) ** (1.0 / 3.0)
        expect_outer = r + oe
        m = re.search(
            r"chargeRefineOuter \{ type searchableSphere;.*?radius\s+([\d.eE+-]+);",
            sn,
            re.DOTALL,
        )
        self.assertIsNotNone(m)
        self.assertAlmostEqual(float(m.group(1)), expect_outer, places=5)
        tr = mode.get("transition_region") or {}
        self.assertEqual(tr.get("transition_shape"), "sphere")
        self.assertFalse(tr.get("outside_extent_auto"))
        self.assertAlmostEqual(tr.get("outside_extent_m", 0), oe)
        self.assertAlmostEqual(tr.get("physical_charge_radius_m", 0), r, places=5)
        self.assertAlmostEqual(tr.get("effective_outer_radius_m", 0), expect_outer, places=5)


class CylinderTransitionValidation(unittest.TestCase):
    def test_searchable_cylinder_dimensions(self) -> None:
        oe = 0.25
        inp = _case(
            charge_shape="Cylinder",
            charge_aspect=2.5,
            outside_extent=oe,
            cylinder_axis="Z",
        )
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("cyl_tr", inp)
            with open(os.path.join(case_dir, "system", "snappyHexMeshDict"), encoding="utf-8") as f:
                sn = f.read()
            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                mode = json.load(f)
        self.assertIn("searchableCylinder", sn)
        tr = mode.get("transition_region") or {}
        self.assertEqual(tr.get("transition_shape"), "cylinder")
        self.assertEqual(tr.get("cylinder_axis"), "Z")
        r_i = tr.get("physical_inner_radius_m")
        h_i = tr.get("physical_inner_half_length_m")
        self.assertIsNotNone(r_i)
        self.assertAlmostEqual(tr.get("effective_outer_radius_m", 0), float(r_i) + oe, places=5)
        self.assertAlmostEqual(tr.get("effective_outer_half_length_m", 0), float(h_i) + oe, places=5)


class CuboidTransitionValidation(unittest.TestCase):
    def test_searchable_box_not_sphere(self) -> None:
        oe = 0.1
        inp = _case(
            charge_shape="Cuboid",
            charge_length=0.5,
            charge_width=0.5,
            charge_height=0.5,
            charge_refinement_level=0,
            charge_outer_refine_min=2,
            charge_outer_refine_max=2,
            outside_extent=oe,
        )
        with tempfile.TemporaryDirectory() as td:
            gen = Generator3D(td)
            case_dir = gen.generate("box_tr", inp)
            with open(os.path.join(case_dir, "system", "snappyHexMeshDict"), encoding="utf-8") as f:
                sn = f.read()
            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                mode = json.load(f)
        self.assertIn("searchableBox", sn)
        self.assertNotRegex(sn, r"chargeRefineOuter\s*\{\s*type\s+searchableSphere")
        tr = mode.get("transition_region") or {}
        self.assertEqual(tr.get("transition_shape"), "box")
        hx, hy, hz = tr.get("physical_half_extents_m") or [0, 0, 0]
        bmin = tr.get("effective_box_min_m") or [0, 0, 0]
        bmax = tr.get("effective_box_max_m") or [0, 0, 0]
        self.assertAlmostEqual(bmax[0] - bmin[0], 2.0 * hx + 2.0 * oe, places=5)

    def test_cuboid_inside_level_warns(self) -> None:
        inp = _case(
            charge_shape="Cuboid",
            charge_length=0.4,
            charge_width=0.4,
            charge_height=0.4,
            charge_refinement_level=2,
            charge_outer_refine_min=0,
            charge_outer_refine_max=0,
        )
        with tempfile.TemporaryDirectory() as td:
            gen = Generator3D(td)
            gen.generate("box_warn", inp)
        self.assertTrue(any("Cuboid with Inside refinement" in w for w in gen._charge_warnings))


class OutsideExtentLegacyValidation(unittest.TestCase):
    def test_auto_reports_legacy_in_metadata(self) -> None:
        inp = _case(outside_extent=None, bubble_radius_factor=1.5, transition_cells=2)
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("auto_ext", inp)
            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                mode = json.load(f)
        tr = mode.get("transition_region") or {}
        self.assertTrue(tr.get("outside_extent_auto"))
        self.assertIn("bubble_radius_factor", tr.get("policy_description", ""))


class AmrDictValidation(unittest.TestCase):
    def test_default_density_gradient(self) -> None:
        inp = _case()
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("dg", inp)
            with open(os.path.join(case_dir, "constant", "dynamicMeshDict"), encoding="utf-8") as f:
                dm = f.read()
        self.assertIn("errorEstimator  densityGradient;", dm)
        self.assertNotIn("scaledDelta", dm)

    def test_scaled_delta_pressure_not_pressure_gradient_keyword(self) -> None:
        inp = _case(refine_indicator_field="scaledDelta_p")
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("sd", inp)
            with open(os.path.join(case_dir, "constant", "dynamicMeshDict"), encoding="utf-8") as f:
                dm = f.read()
        self.assertIn("errorEstimator  scaledDelta;", dm)
        self.assertIn("deltaCoeffs", dm)
        self.assertIn("field           p;", dm)
        self.assertNotIn("errorEstimator  pressureGradient;", dm)

    def test_advanced_amr_keys_and_load_balance(self) -> None:
        inp = _case(
            dynamic_max_cells=5000000,
            begin_unrefine=1e-6,
            upper_refine_level=0.5,
            upper_unrefine_level=0.05,
            enable_balancing=True,
            balance_interval=15,
        )
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("adv", inp)
            with open(os.path.join(case_dir, "constant", "dynamicMeshDict"), encoding="utf-8") as f:
                dm = f.read()
        self.assertIn("maxCells       5000000;", dm)
        self.assertIn("beginUnrefine", dm)
        self.assertIn("upperRefineLevel", dm)
        self.assertIn("upperUnrefineLevel", dm)
        self.assertIn("enableBalancing true;", dm)
        self.assertIn("loadBalance", dm)
        self.assertIn("balance yes;", dm)
        self.assertIn("balanceInterval 15;", dm)


if __name__ == "__main__":
    unittest.main()
