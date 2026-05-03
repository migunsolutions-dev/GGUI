"""Unit tests for 3D charge capture radius (setRefinedFields backup region)."""
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

from charge_capture import auto_charge_capture_radius_m, resolve_charge_capture_radius_m
from generator_3d import Generator3D
from models import CaseInputs3D


class ChargeCaptureTests(unittest.TestCase):
    def test_manual_exact_no_inflation(self) -> None:
        inp = SimpleNamespace(
            charge_capture_mode="manual",
            charge_capture_radius=0.123,
            charge_capture_factor=99.0,
            charge_backup_radius_override=None,
            cell_size=10.0,
        )
        r, rep = resolve_charge_capture_radius_m(inp, r_phys=0.05)
        self.assertAlmostEqual(r, 0.123)
        self.assertEqual(rep.mode, "manual")

    def test_auto_formula(self) -> None:
        r_phys = 0.1
        dx, dy, dz = 0.4, 0.4, 0.4
        cf = 1.0
        expect = max(1.05 * r_phys, 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz) * cf)
        got = auto_charge_capture_radius_m(r_phys, dx, dy, dz, cf)
        self.assertAlmostEqual(got, expect)

    def test_no_hidden_125_dx_floor_manual(self) -> None:
        """Regression: older code used max(r_user, 1.25*dx); manual must not inflate."""
        inp = SimpleNamespace(
            charge_capture_mode="manual",
            charge_capture_radius=0.01,
            charge_backup_radius_override=None,
            cell_size=1.0,
        )
        r, rep = resolve_charge_capture_radius_m(inp, r_phys=0.05)
        self.assertLess(r, 1.25 * 1.0)
        self.assertAlmostEqual(r, 0.01)
        self.assertTrue(any("smaller than the physical charge radius" in w for w in rep.warnings))
        self.assertTrue(any("half the base-cell diagonal" in w for w in rep.warnings))

    def test_legacy_override_maps_to_manual(self) -> None:
        inp = SimpleNamespace(
            charge_capture_mode="auto",
            charge_capture_radius=None,
            charge_capture_factor=1.0,
            charge_backup_radius_override=0.222,
            cell_size=0.5,
        )
        r, rep = resolve_charge_capture_radius_m(inp, r_phys=0.05)
        self.assertAlmostEqual(r, 0.222)
        self.assertEqual(rep.mode, "manual")

    def test_manual_smaller_than_physical_warning_only(self) -> None:
        inp = SimpleNamespace(
            charge_capture_mode="manual",
            charge_capture_radius=0.02,
            charge_backup_radius_override=None,
            cell_size=0.01,
        )
        _r, rep = resolve_charge_capture_radius_m(inp, r_phys=0.05)
        self.assertAlmostEqual(_r, 0.02)
        self.assertTrue(any("smaller than the physical charge radius" in w for w in rep.warnings))
        self.assertFalse(any("very close to the physical charge radius" in w for w in rep.warnings))

    def test_manual_marginal_close_warning(self) -> None:
        inp = SimpleNamespace(
            charge_capture_mode="manual",
            charge_capture_radius=0.102,
            charge_backup_radius_override=None,
            cell_size=0.01,
        )
        _r, rep = resolve_charge_capture_radius_m(inp, r_phys=0.1)
        self.assertAlmostEqual(_r, 0.102)
        self.assertTrue(any("very close to the physical charge radius" in w for w in rep.warnings))
        self.assertFalse(any("smaller than the physical charge radius" in w for w in rep.warnings))

    def test_manual_geometric_diagonal_warning(self) -> None:
        inp = SimpleNamespace(
            charge_capture_mode="manual",
            charge_capture_radius=0.12,
            charge_backup_radius_override=None,
            cell_size=1.0,
        )
        _r, rep = resolve_charge_capture_radius_m(inp, r_phys=0.1)
        self.assertAlmostEqual(_r, 0.12)
        self.assertTrue(any("half the base-cell diagonal" in w for w in rep.warnings))
        self.assertFalse(any("smaller than the physical charge radius" in w for w in rep.warnings))
        self.assertFalse(any("very close to the physical charge radius" in w for w in rep.warnings))

    def test_manual_physical_and_geometric_warnings_stack(self) -> None:
        inp = SimpleNamespace(
            charge_capture_mode="manual",
            charge_capture_radius=0.01,
            charge_backup_radius_override=None,
            cell_size=1.0,
        )
        _r, rep = resolve_charge_capture_radius_m(inp, r_phys=0.05)
        texts = " ".join(rep.warnings)
        self.assertIn("physical charge radius", texts)
        self.assertIn("base-cell diagonal", texts)

    def test_warning_order_geometric_before_large(self) -> None:
        """Deterministic: manual geometric advisory precedes large-radius (2×) warning."""
        inp = SimpleNamespace(
            charge_capture_mode="manual",
            charge_capture_radius=0.12,
            charge_backup_radius_override=None,
            cell_size=1.0,
        )
        _r, rep = resolve_charge_capture_radius_m(inp, r_phys=0.05)
        self.assertTrue(any("half the base-cell diagonal" in w for w in rep.warnings))
        self.assertTrue(any("2×" in w for w in rep.warnings))
        idx_geom = next(i for i, w in enumerate(rep.warnings) if "base-cell diagonal" in w)
        idx_large = next(i for i, w in enumerate(rep.warnings) if "2×" in w)
        self.assertLess(idx_geom, idx_large)

    def test_warnings_thresholds(self) -> None:
        inp = SimpleNamespace(
            charge_capture_mode="manual",
            charge_capture_radius=0.12,
            charge_backup_radius_override=None,
            cell_size=0.1,
        )
        _r, rep = resolve_charge_capture_radius_m(inp, r_phys=0.05)
        self.assertTrue(any("2×" in w for w in rep.warnings))
        self.assertFalse(any("4×" in w for w in rep.warnings))
        inp2 = SimpleNamespace(
            charge_capture_mode="manual",
            charge_capture_radius=0.3,
            charge_backup_radius_override=None,
            cell_size=0.1,
        )
        _r2, rep2 = resolve_charge_capture_radius_m(inp2, r_phys=0.05)
        self.assertTrue(any("4×" in w for w in rep2.warnings))

    def test_case_init_mode_documents_capture(self) -> None:
        """Generated case_init_mode.json includes charge capture metadata."""
        inp = CaseInputs3D(
            min_point=(-1, -1, 0),
            max_point=(1, 1, 2),
            cell_size=0.5,
            charge_center=(0, 0, 1),
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
            charge_refinement_level=3,
            charge_capture_mode="auto",
            charge_capture_factor=1.0,
        )
        with tempfile.TemporaryDirectory() as td:
            gen = Generator3D(td)
            case_dir = gen.generate("t_capture_meta", inp)
            path = os.path.join(case_dir, "case_init_mode.json")
            self.assertTrue(os.path.isfile(path))
            with open(path, "r", encoding="utf-8") as f:
                mode = json.load(f)
            cap = mode.get("charge_capture") or {}
            self.assertEqual(cap.get("mode"), "auto")
            self.assertIn("charge_capture_radius_used_m", cap)
            self.assertEqual(mode.get("charge_refinement_requested"), 3)
            self.assertEqual(mode.get("charge_refinement_effective"), 3)


if __name__ == "__main__":
    unittest.main()
