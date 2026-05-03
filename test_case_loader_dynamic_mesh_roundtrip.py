"""Round-trip: load_case parses dynamicMeshDict → CaseInputs3D → regenerate → preserve AMR keys."""
from __future__ import annotations

import os
import re
import tempfile
import unittest

from case_loader import load_case
from generator_3d import Generator3D
from models import CaseInputs3D


def _full_sphere_case(**kw) -> CaseInputs3D:
    d = dict(
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
        dynamic_max_cells=9_000_000,
        begin_unrefine=2e-7,
        upper_refine_level=0.4,
        upper_unrefine_level=0.04,
        enable_balancing=True,
        balance_interval=12,
        refine_indicator_field="densityGradient",
    )
    d.update(kw)
    return CaseInputs3D(**d)


def _inputs_from_loaded(data: dict) -> CaseInputs3D:
    """Minimal mapping from load_case output to CaseInputs3D (3D sphere baseline)."""
    mp = data.get("min_point") or (-2, -2, 0)
    xp = data.get("max_point") or (2, 2, 4)
    cc = data.get("charge_center") or (0, 0, 2)
    return CaseInputs3D(
        min_point=tuple(float(x) for x in mp[:3]),
        max_point=tuple(float(x) for x in xp[:3]),
        cell_size=float(data.get("cell_size") or 0.5),
        charge_center=tuple(float(x) for x in cc[:3]),
        charge_shape=str(data.get("charge_shape") or "Sphere"),
        mass_kg=float(data.get("mass_kg") or 25.0),
        cylinder_radius=float(data.get("charge_radius") or data.get("cylinder_radius") or 0.1),
        cylinder_axis=str(data.get("cylinder_axis") or "Z"),
        material_name=str(data.get("material_name") or "C4"),
        rho_charge=float(data.get("rho_charge") or 1601.0),
        energy_j_per_kg=float(data.get("energy_j_per_kg") or 4.5e6),
        p_atm=float(data.get("p_atm") or 101325.0),
        t_atm=float(data.get("t_atm") or 300.0),
        end_time_s=float(data.get("end_time_s") or 1e-3),
        delta_t=float(data.get("delta_t") or 1e-7),
        write_interval_steps=int(data.get("write_interval_steps") or 10),
        cores=int(data.get("cores") or 1),
        enable_dyn_refine=bool(data.get("enable_dyn_refine", True)),
        enable_local_refinement=bool(data.get("enable_local_refinement", True)),
        refine_min=int(data.get("refine_min") or 2),
        refine_max=int(data.get("refine_max") or 3),
        dyn_refine_min=int(data.get("dyn_refine_min") or data.get("refine_min") or 2),
        dyn_refine_max=int(data.get("dyn_refine_max") or data.get("refine_max") or 1),
        charge_refinement_level=int(data.get("charge_refinement_level") or 0),
        charge_outer_refine_min=int(data.get("charge_outer_refine_min") or 2),
        charge_outer_refine_max=int(data.get("charge_outer_refine_max") or 3),
        charge_capture_mode=str(data.get("charge_capture_mode") or "auto"),
        charge_capture_factor=float(data.get("charge_capture_factor") or 1.0),
        dynamic_max_cells=int(data.get("dynamic_max_cells") or 200_000_000),
        begin_unrefine=data.get("begin_unrefine"),
        upper_refine_level=data.get("upper_refine_level"),
        upper_unrefine_level=data.get("upper_unrefine_level"),
        enable_balancing=bool(data.get("enable_balancing") or False),
        balance_interval=data.get("balance_interval"),
        refine_indicator_field=str(data.get("refine_indicator_field") or "densityGradient"),
        provenance=dict(data.get("_provenance") or {}),
    )


class DynamicMeshRoundTripTests(unittest.TestCase):
    def test_density_gradient_advanced_fields_roundtrip(self) -> None:
        inp0 = _full_sphere_case()
        with tempfile.TemporaryDirectory() as td:
            c1 = Generator3D(td).generate("gen1", inp0)
            data = load_case(c1)
            inp1 = _inputs_from_loaded(data)
            c2 = os.path.join(td, "gen2")
            Generator3D(td).generate("gen2", inp1)
            p1 = os.path.join(c1, "constant", "dynamicMeshDict")
            p2 = os.path.join(c2, "constant", "dynamicMeshDict")
            with open(p1, encoding="utf-8") as f:
                d1 = f.read()
            with open(p2, encoding="utf-8") as f:
                d2 = f.read()
        for needle in (
            "errorEstimator  densityGradient;",
            "maxCells       9000000;",
            "beginUnrefine",
            "upperRefineLevel",
            "upperUnrefineLevel",
            "balance yes;",
            "balanceInterval 12;",
        ):
            self.assertIn(needle, d1, msg=f"gen1 missing {needle}")
            self.assertIn(needle, d2, msg=f"gen2 missing {needle} after load_case")

    def test_scaled_delta_p_roundtrip(self) -> None:
        inp0 = _full_sphere_case(
            refine_indicator_field="scaledDelta_p",
            begin_unrefine=None,
            upper_refine_level=None,
            upper_unrefine_level=None,
            enable_balancing=False,
            dynamic_max_cells=200_000_000,
        )
        with tempfile.TemporaryDirectory() as td:
            c1 = Generator3D(td).generate("p1", inp0)
            data = load_case(c1)
            self.assertEqual(data.get("refine_indicator_field"), "scaledDelta_p")
            inp1 = _inputs_from_loaded(data)
            self.assertEqual(inp1.refine_indicator_field, "scaledDelta_p")
            c2 = os.path.join(td, "p2")
            Generator3D(td).generate("p2", inp1)
            with open(os.path.join(c2, "constant", "dynamicMeshDict"), encoding="utf-8") as f:
                d2 = f.read()
        self.assertIn("errorEstimator  scaledDelta;", d2)
        self.assertIn("field           p;", d2)
        self.assertNotRegex(d2, r"errorEstimator\s+pressureGradient\s*;")


if __name__ == "__main__":
    unittest.main()
