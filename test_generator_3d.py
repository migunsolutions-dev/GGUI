"""Generator3D write-path tests (no OpenFOAM runtime)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from generator_3d import Generator3D
from models import CaseInputs3D


def _minimal(**overrides):
    base = dict(
        min_point=(-2.0, -2.0, 0.0),
        max_point=(2.0, 2.0, 4.0),
        cell_size=0.5,
        charge_center=(0.0, 0.0, 2.0),
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
    base.update(overrides)
    return CaseInputs3D(**base)


class Generator3DWriteTests(unittest.TestCase):
    def test_cylinder_outer_snappy_is_searchable_cylinder(self) -> None:
        inp = _minimal(
            charge_shape="Cylinder",
            charge_aspect=2.5,
        )
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("cyl_outer", inp)
            with open(os.path.join(case_dir, "system", "snappyHexMeshDict"), encoding="utf-8") as f:
                sn = f.read()
        self.assertIn("searchableCylinder", sn)
        self.assertIn("chargeRefineOuter", sn)

    def test_cuboid_outer_snappy_is_searchable_box_not_sphere(self) -> None:
        inp = _minimal(
            charge_shape="Cuboid",
            charge_length=0.4,
            charge_width=0.4,
            charge_height=0.4,
            charge_refinement_level=0,
            charge_outer_refine_min=2,
            charge_outer_refine_max=2,
        )
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("box_outer", inp)
            with open(os.path.join(case_dir, "system", "snappyHexMeshDict"), encoding="utf-8") as f:
                sn = f.read()
        self.assertIn("searchableBox", sn)
        self.assertNotIn("chargeRefineOuter { type searchableSphere", sn)

    def test_pressure_amr_writes_scaled_delta_not_bare_pressure_gradient(self) -> None:
        inp = _minimal(refine_indicator_field="scaledDelta_p")
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("p_amr", inp)
            with open(os.path.join(case_dir, "constant", "dynamicMeshDict"), encoding="utf-8") as f:
                dm = f.read()
        self.assertIn("errorEstimator  scaledDelta;", dm)
        self.assertIn("field           p;", dm)
        self.assertNotIn("errorEstimator  pressureGradient;", dm)

    def test_legacy_pressure_gradient_mapped_in_writer(self) -> None:
        inp = _minimal(refine_indicator_field="pressureGradient")
        with tempfile.TemporaryDirectory() as td:
            gen = Generator3D(td)
            case_dir = gen.generate("p_legacy", inp)
            with open(os.path.join(case_dir, "constant", "dynamicMeshDict"), encoding="utf-8") as f:
                dm = f.read()
        self.assertIn("scaledDelta", dm)
        self.assertTrue(any("pressureGradient" in w for w in gen._charge_warnings))

    def test_outside_extent_overrides_bubble_factor_geometry(self) -> None:
        inp = _minimal(
            charge_shape="Sphere",
            outside_extent=0.75,
            bubble_radius_factor=99.0,
        )
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("oe", inp)
            with open(os.path.join(case_dir, "system", "snappyHexMeshDict"), encoding="utf-8") as f:
                sn = f.read()
        # Physical R for 25 kg C4 ~0.155 m; outer radius ~0.155+0.75
        import re

        m = re.search(r"chargeRefineOuter \{ type searchableSphere;.*?radius ([\d.eE+-]+);", sn, re.DOTALL)
        self.assertIsNotNone(m)
        r_out = float(m.group(1))
        self.assertLess(r_out, 2.0)
        self.assertGreater(r_out, 0.8)

    def test_domain_alignment_metadata(self) -> None:
        inp = _minimal(
            min_point=(0.0, 0.0, 0.0),
            max_point=(10.03, 1.0, 2.0),
            cell_size=0.5,
            charge_center=(5.0, 0.5, 1.0),
        )
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("align_meta", inp)
            with open(os.path.join(case_dir, "case_init_mode.json"), encoding="utf-8") as f:
                mode = json.load(f)
        da = mode.get("domain_alignment") or {}
        self.assertTrue(da.get("domain_adjusted_for_cell_fit"))


if __name__ == "__main__":
    unittest.main()
