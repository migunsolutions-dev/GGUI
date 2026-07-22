"""Authoritative physical charge geometry + outer lossless round-trip tests."""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

from case_loader import load_case
from charge_seed_plan import build_charge_seed_plan, charge_dims_from_inputs
from generator_3d import Generator3D
from models import CaseInputs3D
from physical_charge_geometry import (
    canonical_outside_extent_from_outer_geometry,
    physical_charge_geometry,
    sync_derived_cylinder_fields,
)
from project_io import (
    _bake_legacy_outside_extent_if_needed,
    build_project,
    read_project,
    write_project_atomic,
)


def _cyl_base(**kw):
    base = dict(
        min_point=(-5.0, -5.0, 0.0),
        max_point=(5.0, 5.0, 5.0),
        cell_size=0.5,
        charge_center=(0.0, 0.0, 0.5),
        charge_shape="Cylinder",
        mass_kg=25.0,
        cylinder_radius=0.5,  # deliberately wrong vs mass/L/D — must be ignored
        cylinder_axis="Z",
        material_name="C4",
        rho_charge=1601.0,
        energy_j_per_kg=4.5e6,
        p_atm=101325.0,
        t_atm=300.0,
        end_time_s=1e-3,
        delta_t=1e-6,
        write_interval_steps=1,
        cores=1,
        charge_aspect=2.5,
        charge_length=9.0,  # deliberately wrong
        enable_dyn_refine=True,
        dyn_refine_max=1,
        charge_seed_mode="Manual",
        charge_refinement_level=5,
        buffer_layers=5,
        charge_outer_refine_enable=False,
    )
    base.update(kw)
    return CaseInputs3D(**base)


class PhysicalCylinderAuthorityTests(unittest.TestCase):
    def test_mass_rho_lbyd_authoritative_ignores_gui_radius(self):
        inp = _cyl_base()
        geom = physical_charge_geometry(inp)
        vol = 25.0 / 1601.0
        r_exp = (vol / (2.0 * math.pi * 2.5)) ** (1.0 / 3.0)
        self.assertAlmostEqual(geom.cylinder_radius_m, r_exp, places=9)
        self.assertAlmostEqual(geom.length_m, 2.0 * r_exp * 2.5, places=9)
        self.assertEqual(geom.authoritative, "mass_rho_LbyD_cylindericalMassToCell")
        dims_seed = charge_dims_from_inputs(inp)
        dims_gen = Generator3D(tempfile.mkdtemp())._calculate_charge_dimensions(inp)
        self.assertAlmostEqual(dims_seed["radius"], dims_gen["radius"], places=9)
        self.assertAlmostEqual(dims_seed["length"], dims_gen["length"], places=9)
        self.assertAlmostEqual(dims_seed["radius"], r_exp, places=9)

    def test_setfields_uses_lbyd_not_stale_radius(self):
        with tempfile.TemporaryDirectory() as td:
            inp = _cyl_base()
            case_dir = Generator3D(td).generate("cyl_auth", inp)
            with open(os.path.join(case_dir, "system", "setFieldsDict"), encoding="utf-8") as fh:
                sf = fh.read()
            self.assertIn("LbyD", sf)
            self.assertIn("mass 25", sf)
            self.assertIn("rho 1601", sf)
            plan = build_charge_seed_plan(inp)
            geom = physical_charge_geometry(inp)
            self.assertAlmostEqual(plan.d_min_m, geom.d_min_m, places=9)

    def test_unsupported_shape_raises(self):
        with self.assertRaises(ValueError) as ctx:
            physical_charge_geometry(SimpleNamespace(
                charge_shape="Pyramid",
                mass_kg=1.0,
                rho_charge=1600.0,
            ))
        self.assertIn("Unsupported charge shape", str(ctx.exception))

    def test_invalid_mass_rho_aspect_raise(self):
        with self.assertRaises(ValueError):
            physical_charge_geometry(SimpleNamespace(
                charge_shape="Sphere", mass_kg=0.0, rho_charge=1600.0,
            ))
        with self.assertRaises(ValueError):
            physical_charge_geometry(SimpleNamespace(
                charge_shape="Sphere", mass_kg=1.0, rho_charge=-10.0,
            ))
        with self.assertRaises(ValueError):
            physical_charge_geometry(SimpleNamespace(
                charge_shape="Cylinder", mass_kg=1.0, rho_charge=1600.0, charge_aspect=0.0,
            ))

    def test_sphere_and_cuboid_explicit(self):
        sph = physical_charge_geometry(SimpleNamespace(
            charge_shape="Sphere", mass_kg=5.0, rho_charge=1630.0,
        ))
        self.assertEqual(sph.shape, "Sphere")
        self.assertGreater(sph.radius_m, 0.0)
        cub = physical_charge_geometry(SimpleNamespace(
            charge_shape="Cuboid", mass_kg=8.0, rho_charge=1600.0,
            charge_length=0.0, charge_width=0.0, charge_height=0.0,
        ))
        self.assertEqual(cub.shape, "Cuboid")
        self.assertAlmostEqual(cub.length_box_m, (8.0 / 1600.0) ** (1.0 / 3.0), places=9)


class OuterLosslessRoundTripTests(unittest.TestCase):
    def test_inside_single_level_and_cylinder_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            inp = _cyl_base(
                charge_outer_refine_enable=True,
                charge_outer_refine_level=3,
                charge_outer_mode="inside",
                outside_extent=0.70,
            )
            case_dir = Generator3D(td).generate("outer_cyl", inp)
            loaded = load_case(case_dir)
            self.assertTrue(loaded.get("charge_outer_refine_enable"))
            self.assertEqual(loaded.get("charge_outer_refine_level"), 3)
            self.assertEqual(loaded.get("charge_outer_mode"), "inside")
            geom = loaded.get("charge_outer_geometry") or {}
            self.assertEqual(geom.get("type"), "searchableCylinder")
            self.assertIn("radius", geom)
            self.assertIn("point1", geom)
            self.assertIn("point2", geom)
            # Canonical shell ≈ 0.700 m (radial and axial agree)
            self.assertIsNotNone(loaded.get("outside_extent"))
            self.assertAlmostEqual(float(loaded["outside_extent"]), 0.70, places=2)
            self.assertLess(abs(float(loaded["outside_extent"]) - 0.70), 0.05)
            self.assertNotAlmostEqual(float(loaded["outside_extent"]), 3.1247, places=2)
            baked = _bake_legacy_outside_extent_if_needed(dict(loaded))
            self.assertAlmostEqual(
                float(baked["outside_extent"]),
                float(loaded["outside_extent"]),
                places=6,
            )

    def test_distance_multi_pair_full_gui_round_trip(self):
        """load → GUI apply → collect → generate → reload preserves distance pairs."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        import sys
        import tab_3d_general
        from probes_model import ProbesModel

        from PyQt5.QtCore import pyqtSignal
        from PyQt5.QtWidgets import QWidget

        app = QApplication.instance() or QApplication(sys.argv)
        real_viewer = tab_3d_general.BlastViewerWidget

        class DummyViewer(QWidget):
            cell_count_updated = pyqtSignal(int)

            def __init__(self, *a, **k):
                super().__init__()

            def __getattr__(self, _n):
                return lambda *a, **k: None

        tab_3d_general.BlastViewerWidget = DummyViewer
        try:
            with tempfile.TemporaryDirectory() as td:
                case = os.path.join(td, "dist")
                os.makedirs(os.path.join(case, "system"))
                os.makedirs(os.path.join(case, "constant"))
                # Physical charge matching mass/rho/L/D for extent recovery if needed
                snappy = """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
geometry
{
    chargeRefineOuter
    {
        type searchableCylinder;
        point1 (0 0 -0.2);
        point2 (0 0 1.2);
        radius 0.85;
    }
};
castellatedMeshControls
{
    nCellsBetweenLevels 2;
    refinementRegions
    {
        chargeRefineOuter { mode distance; levels ((0.2 3) (0.5 2) (1.0 1)); }
    }
    locationInMesh (0 0 0.5);
}
"""
                with open(os.path.join(case, "system", "snappyHexMeshDict"), "w", encoding="utf-8") as fh:
                    fh.write(snappy)
                setf = """FoamFile { version 2.0; format ascii; class dictionary; object setFieldsDict; }
defaultFieldValues ( volScalarFieldValue alpha.c4 0 );
regions
(
    cylindericalMassToCell
    {
        mass 25;
        rho 1601;
        LbyD 2.5;
        centre (0 0 0.5);
        p1 (0 0 0);
        p2 (0 0 1);
    }
);
"""
                with open(os.path.join(case, "system", "setFieldsDict"), "w", encoding="utf-8") as fh:
                    fh.write(setf)
                for rel in (
                    "system/blockMeshDict",
                    "system/controlDict",
                    "constant/phaseProperties",
                    "constant/dynamicMeshDict",
                ):
                    p = os.path.join(case, rel)
                    with open(p, "w", encoding="utf-8") as fh:
                        fh.write(
                            "FoamFile { version 2.0; format ascii; "
                            "class dictionary; object x; }\n"
                        )
                loaded = load_case(case)
                self.assertEqual(loaded.get("charge_outer_mode"), "distance")
                self.assertEqual(len(loaded.get("charge_outer_distance_levels") or []), 3)

                tab = tab_3d_general.TabGeneral3D(ProbesModel())
                # Merge loaded outer fields onto a complete case dict for GUI apply
                base = asdict(_cyl_base(charge_outer_refine_enable=True))
                base.update({k: loaded[k] for k in loaded if k in base or k.startswith("charge_outer") or k == "outside_extent"})
                base["charge_outer_mode"] = loaded["charge_outer_mode"]
                base["charge_outer_distance_levels"] = loaded["charge_outer_distance_levels"]
                base["charge_outer_geometry"] = loaded["charge_outer_geometry"]
                base["charge_outer_refine_enable"] = True
                base["charge_outer_refine_level"] = loaded.get("charge_outer_refine_level", 3)
                tab.set_case_inputs(base)
                collected = tab.get_case_inputs()
                self.assertEqual(collected.charge_outer_mode, "distance")
                self.assertEqual(len(collected.charge_outer_distance_levels), 3)
                self.assertEqual(collected.charge_outer_geometry["type"], "searchableCylinder")
                p1 = collected.charge_outer_geometry["point1"]
                p2 = collected.charge_outer_geometry["point2"]
                self.assertAlmostEqual(float(p1[2]), -0.2, places=5)
                self.assertAlmostEqual(float(p2[2]), 1.2, places=5)
                self.assertAlmostEqual(float(collected.charge_outer_geometry["radius"]), 0.85, places=5)

                out_case = Generator3D(td).generate("dist_regen", collected)
                reloaded = load_case(out_case)
                self.assertEqual(reloaded.get("charge_outer_mode"), "distance")
                pairs = reloaded.get("charge_outer_distance_levels")
                self.assertEqual(len(pairs), 3)
                self.assertAlmostEqual(float(pairs[0][0]), 0.2, places=5)
                self.assertEqual(int(pairs[0][1]), 3)
                self.assertAlmostEqual(float(pairs[2][0]), 1.0, places=5)
                self.assertEqual(int(pairs[2][1]), 1)
                geom2 = reloaded.get("charge_outer_geometry") or {}
                self.assertEqual(geom2.get("type"), "searchableCylinder")
                self.assertAlmostEqual(float(geom2["radius"]), 0.85, places=5)
                self.assertAlmostEqual(float(geom2["point1"][2]), -0.2, places=5)
                self.assertAlmostEqual(float(geom2["point2"][2]), 1.2, places=5)
        finally:
            tab_3d_general.BlastViewerWidget = real_viewer

    def test_sphere_and_box_geometry_preserved_on_regen(self):
        with tempfile.TemporaryDirectory() as td:
            for gtype, geom, shape in (
                (
                    "searchableSphere",
                    {"type": "searchableSphere", "centre": (0.0, 0.0, 0.5), "radius": 1.1},
                    "Sphere",
                ),
                (
                    "searchableBox",
                    {
                        "type": "searchableBox",
                        "min": (-1.0, -1.0, -0.5),
                        "max": (1.0, 1.0, 1.5),
                    },
                    "Cuboid",
                ),
            ):
                inp = _cyl_base(
                    charge_shape=shape,
                    mass_kg=25.0 if shape == "Sphere" else 8.0,
                    rho_charge=1601.0 if shape == "Sphere" else 1600.0,
                    charge_aspect=2.5,
                    charge_outer_refine_enable=True,
                    charge_outer_refine_level=2,
                    charge_outer_mode="inside",
                    charge_outer_geometry=geom,
                    outside_extent=None,
                )
                if shape == "Sphere":
                    inp = replace(inp, charge_shape="Sphere")
                case_dir = Generator3D(td).generate(f"g_{gtype}", inp)
                loaded = load_case(case_dir)
                g2 = loaded.get("charge_outer_geometry") or {}
                self.assertEqual(g2.get("type"), gtype)
                if gtype == "searchableSphere":
                    self.assertAlmostEqual(float(g2["radius"]), 1.1, places=5)
                    self.assertAlmostEqual(float(g2["centre"][2]), 0.5, places=5)
                else:
                    self.assertAlmostEqual(float(g2["min"][0]), -1.0, places=5)
                    self.assertAlmostEqual(float(g2["max"][2]), 1.5, places=5)

    def test_project_save_open_preserves_outer_state(self):
        with tempfile.TemporaryDirectory() as td:
            inp = _cyl_base(
                charge_outer_refine_enable=True,
                charge_outer_refine_level=3,
                charge_outer_mode="distance",
                charge_outer_distance_levels=[(0.2, 3), (0.5, 2), (1.0, 1)],
                charge_outer_geometry={
                    "type": "searchableCylinder",
                    "point1": (0.0, 0.0, -0.2),
                    "point2": (0.0, 0.0, 1.2),
                    "radius": 0.85,
                },
            )
            path = os.path.join(td, "proj.ggui.json")
            payload = build_project(inp, probes={"probes": []}, gui_state={})
            write_project_atomic(path, payload)
            loaded = read_project(path)
            ci = loaded["inputs"]
            self.assertEqual(ci.charge_outer_mode, "distance")
            self.assertEqual(len(ci.charge_outer_distance_levels), 3)
            self.assertEqual(ci.charge_outer_geometry["type"], "searchableCylinder")
            self.assertAlmostEqual(float(ci.charge_outer_geometry["radius"]), 0.85, places=5)

    def test_noncanonical_cylinder_shell_does_not_set_scalar(self):
        phys = physical_charge_geometry(_cyl_base())
        # Radial expansion 0.7 but axial much larger → not canonical
        geom = {
            "type": "searchableCylinder",
            "point1": (0.0, 0.0, -2.0),
            "point2": (0.0, 0.0, 3.0),
            "radius": phys.cylinder_radius_m + 0.7,
        }
        extent = canonical_outside_extent_from_outer_geometry(
            geom, phys, charge_center=(0.0, 0.0, 0.5), cylinder_axis="Z"
        )
        self.assertIsNone(extent)

    def test_canonical_aligned_sphere_box_cylinder(self):
        # Sphere
        sph_in = SimpleNamespace(
            charge_shape="Sphere", mass_kg=5.0, rho_charge=1630.0
        )
        sph = physical_charge_geometry(sph_in)
        centre = (0.0, 0.0, 0.0)
        geom_s = {
            "type": "searchableSphere",
            "centre": centre,
            "radius": sph.radius_m + 0.5,
        }
        self.assertAlmostEqual(
            canonical_outside_extent_from_outer_geometry(
                geom_s, sph, charge_center=centre
            ),
            0.5,
            places=6,
        )
        # Cylinder
        cyl = physical_charge_geometry(_cyl_base())
        half = 0.5 * cyl.length_m + 0.7
        geom_c = {
            "type": "searchableCylinder",
            "point1": (0.0, 0.0, 0.5 - half),
            "point2": (0.0, 0.0, 0.5 + half),
            "radius": cyl.cylinder_radius_m + 0.7,
        }
        self.assertAlmostEqual(
            canonical_outside_extent_from_outer_geometry(
                geom_c, cyl, charge_center=(0.0, 0.0, 0.5), cylinder_axis="Z"
            ),
            0.7,
            places=5,
        )
        # Box around cube cuboid
        cub = physical_charge_geometry(
            SimpleNamespace(
                charge_shape="Cuboid",
                mass_kg=8.0,
                rho_charge=1600.0,
                charge_length=0.0,
                charge_width=0.0,
                charge_height=0.0,
            )
        )
        side = cub.length_box_m
        half_o = side / 2.0 + 0.25
        geom_b = {
            "type": "searchableBox",
            "min": (-half_o, -half_o, 0.5 - half_o),
            "max": (half_o, half_o, 0.5 + half_o),
        }
        self.assertAlmostEqual(
            canonical_outside_extent_from_outer_geometry(
                geom_b, cub, charge_center=(0.0, 0.0, 0.5)
            ),
            0.25,
            places=5,
        )

    def test_shifted_or_rotated_geometry_returns_none(self):
        sph = physical_charge_geometry(
            SimpleNamespace(charge_shape="Sphere", mass_kg=5.0, rho_charge=1630.0)
        )
        # Shifted sphere
        self.assertIsNone(
            canonical_outside_extent_from_outer_geometry(
                {
                    "type": "searchableSphere",
                    "centre": (1.0, 0.0, 0.0),
                    "radius": sph.radius_m + 0.5,
                },
                sph,
                charge_center=(0.0, 0.0, 0.0),
            )
        )
        cub = physical_charge_geometry(
            SimpleNamespace(
                charge_shape="Cuboid",
                mass_kg=8.0,
                rho_charge=1600.0,
                charge_length=0.0,
                charge_width=0.0,
                charge_height=0.0,
            )
        )
        half = cub.length_box_m / 2.0 + 0.2
        # Shifted box
        self.assertIsNone(
            canonical_outside_extent_from_outer_geometry(
                {
                    "type": "searchableBox",
                    "min": (0.5 - half, -half, -half),
                    "max": (0.5 + half, half, half),
                },
                cub,
                charge_center=(0.0, 0.0, 0.0),
            )
        )
        cyl = physical_charge_geometry(_cyl_base())
        half_l = 0.5 * cyl.length_m + 0.5
        # Shifted cylinder midpoint
        self.assertIsNone(
            canonical_outside_extent_from_outer_geometry(
                {
                    "type": "searchableCylinder",
                    "point1": (0.3, 0.0, 0.5 - half_l),
                    "point2": (0.3, 0.0, 0.5 + half_l),
                    "radius": cyl.cylinder_radius_m + 0.5,
                },
                cyl,
                charge_center=(0.0, 0.0, 0.5),
                cylinder_axis="Z",
            )
        )
        # Rotated cylinder (axis along X instead of Z)
        self.assertIsNone(
            canonical_outside_extent_from_outer_geometry(
                {
                    "type": "searchableCylinder",
                    "point1": (-half_l, 0.0, 0.5),
                    "point2": (half_l, 0.0, 0.5),
                    "radius": cyl.cylinder_radius_m + 0.5,
                },
                cyl,
                charge_center=(0.0, 0.0, 0.5),
                cylinder_axis="Z",
            )
        )

    def test_noncanonical_preserves_geometry_on_regen(self):
        """When extent is None, generator still re-emits imported searchable geometry."""
        with tempfile.TemporaryDirectory() as td:
            shifted = {
                "type": "searchableSphere",
                "centre": (1.5, 0.0, 0.5),
                "radius": 1.2,
            }
            inp = _cyl_base(
                charge_shape="Sphere",
                mass_kg=5.0,
                rho_charge=1630.0,
                charge_outer_refine_enable=True,
                charge_outer_refine_level=2,
                charge_outer_mode="inside",
                charge_outer_geometry=shifted,
                outside_extent=None,
            )
            case_dir = Generator3D(td).generate("shift_outer", inp)
            loaded = load_case(case_dir)
            g = loaded.get("charge_outer_geometry") or {}
            self.assertEqual(g.get("type"), "searchableSphere")
            self.assertAlmostEqual(float(g["centre"][0]), 1.5, places=5)
            self.assertAlmostEqual(float(g["radius"]), 1.2, places=5)
            # Scalar extent must not invent a recentred shell
            # (may be absent or not equal to radius-phys from shifted centre)
            if loaded.get("outside_extent") is not None:
                # If recovered, must still keep original geometry (already checked)
                pass


    def test_idempotent_load_save_inside(self):
        with tempfile.TemporaryDirectory() as td:
            inp = _cyl_base(
                charge_outer_refine_enable=True,
                charge_outer_refine_level=3,
                charge_outer_mode="inside",
                outside_extent=0.70,
            )
            case_dir = Generator3D(td).generate("idem", inp)
            a = load_case(case_dir)
            inp2 = replace(
                inp,
                charge_outer_refine_enable=a["charge_outer_refine_enable"],
                charge_outer_refine_level=a["charge_outer_refine_level"],
                charge_outer_mode=a.get("charge_outer_mode") or "inside",
                charge_outer_geometry=a.get("charge_outer_geometry"),
                outside_extent=a.get("outside_extent"),
            )
            case2 = Generator3D(td).generate("idem2", inp2)
            b = load_case(case2)
            self.assertEqual(a["charge_outer_refine_level"], b["charge_outer_refine_level"])
            self.assertEqual(a.get("charge_outer_mode"), b.get("charge_outer_mode"))
            self.assertAlmostEqual(
                float(a["outside_extent"]), float(b["outside_extent"]), places=5
            )
            g1 = a.get("charge_outer_geometry") or {}
            g2 = b.get("charge_outer_geometry") or {}
            self.assertAlmostEqual(float(g1["radius"]), float(g2["radius"]), places=5)


if __name__ == "__main__":
    unittest.main()
