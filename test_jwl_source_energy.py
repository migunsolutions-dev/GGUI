"""JWL energy-budget audit: blastFoam semantics, not a JWL rewrite."""

from __future__ import annotations

import unittest

import jwl_activation_energy as jae
import jwl_source_energy as jse
from material_catalog import JWL_PARAMETERS


class BirchMurnaghanTests(unittest.TestCase):
    def test_cold_energy_at_reference_density_is_finite_and_large(self):
        e = jse.birch_murnaghan3_cold_energy(1630.0, 1630.0)
        # 9/16*K0*(K0Prime-6)/rho0  minus pRef/rho0 ~ 5.47 MJ/kg
        self.assertAlmostEqual(e / 5.466901e6, 1.0, places=3)
        self.assertGreater(e, 4e6)

    def test_reactant_initial_e_is_cv_t_plus_cold_curve(self):
        e = jse.reactant_initial_specific_energy(1630.0, 288.0)
        self.assertAlmostEqual(e, 1400.0 * 288.0 + jse.birch_murnaghan3_cold_energy(1630.0, 1630.0))


class InitiationE0Tests(unittest.TestCase):
    def test_blastfoam_divides_E0_by_rho0(self):
        self.assertAlmostEqual(jse.blastfoam_e0_from_initiation(4.29e6, 1630.0), 4.29e6 / 1630.0)
        self.assertAlmostEqual(jse.blastfoam_e0_from_initiation(4.29e9, 1630.0), 4.29e9 / 1630.0)


class ProductionV2BudgetTests(unittest.TestCase):
    def test_v2_writes_rho_times_e_charge_in_every_dimension(self):
        budgets = [
            jse.audit_jwl_energy(
                dimension=dim,
                mass_kg=1.0,
                rho_charge=1630.0,
                energy_j_per_kg=4.29e6,
                material_name="TNT",
            )
            for dim in ("1D", "2D", "3D")
        ]
        expected = 1630.0 * 4.29e6
        for budget in budgets:
            self.assertEqual(budget.jwl_energy_schema, jae.JWL_ENERGY_SCHEMA_V2)
            self.assertAlmostEqual(budget.written_initiation_E0, expected)
            self.assertAlmostEqual(budget.blastfoam_e0_j_per_kg, 4.29e6)
            self.assertAlmostEqual(budget.represented_chemical_over_intended, 1.0)
            self.assertNotAlmostEqual(budget.written_initiation_E0, JWL_PARAMETERS["TNT"]["E0"])
        self.assertEqual(
            {b.written_initiation_E0 for b in budgets},
            {budgets[0].written_initiation_E0},
        )

    def test_c4_v2_does_not_keep_tutorial_e0(self):
        budget = jse.audit_jwl_energy(
            dimension="2D",
            mass_kg=1.0,
            rho_charge=1601.0,
            energy_j_per_kg=4.52e6,
            material_name="C4",
        )
        self.assertAlmostEqual(budget.written_initiation_E0, 1601.0 * 4.52e6)
        self.assertNotAlmostEqual(budget.written_initiation_E0, 9.0e9)
        self.assertEqual(budget.catalog_jwl_E0, 9.0e9)

    def test_ig_equivalence_gate_stays_validation_only(self):
        budget = jse.audit_jwl_energy(
            dimension="1D", mass_kg=1.0, rho_charge=1630.0, energy_j_per_kg=4.29e6
        )
        payload = jse.budget_dict(budget)
        self.assertIs(payload["equivalence_to_ig"]["same_W_rho_E_charge_chemical_add_matches"], True)
        self.assertIs(payload["equivalence_to_ig"]["same_W_rho_E_charge_is_energy_equivalent"], False)
        self.assertIn("validation only", payload["equivalence_to_ig"]["hard_gate"].lower())
        self.assertNotIn("CURRENT LEGACY IMPLEMENTATION", payload["equivalence_to_ig"]["hard_gate"])


class LegacySchemaBudgetTests(unittest.TestCase):
    def test_legacy_1d_writes_gui_energy_into_E0_slot(self):
        b = jse.audit_jwl_energy(
            dimension="1D",
            mass_kg=1.0,
            rho_charge=1630.0,
            energy_j_per_kg=4.29e6,
            material_name="TNT",
            schema=jae.JWL_ENERGY_SCHEMA_LEGACY,
        )
        self.assertEqual(b.written_initiation_E0, 4.29e6)
        self.assertAlmostEqual(b.blastfoam_e0_j_per_kg, 4.29e6 / 1630.0)
        self.assertLess(b.represented_chemical_over_intended, 0.002)
        self.assertGreater(b.reactant_e_init_j_per_kg, b.chemical_energy_added_j_per_kg)

    def test_legacy_2d_writes_catalog_jwl_E0(self):
        b = jse.audit_jwl_energy(
            dimension="2D",
            mass_kg=1.0,
            rho_charge=1630.0,
            energy_j_per_kg=4.29e6,
            material_name="TNT",
            schema=jae.JWL_ENERGY_SCHEMA_LEGACY,
        )
        self.assertEqual(b.written_initiation_E0, 4.29e9)
        self.assertAlmostEqual(b.blastfoam_e0_j_per_kg, 4.29e9 / 1630.0)
        self.assertNotAlmostEqual(b.blastfoam_e0_j_per_kg, 4.29e6)

    def test_legacy_same_gui_inputs_are_not_energy_equivalent_across_dimensions(self):
        a = jse.audit_jwl_energy(
            dimension="1D",
            mass_kg=1.0,
            rho_charge=1630.0,
            energy_j_per_kg=4.29e6,
            schema=jae.JWL_ENERGY_SCHEMA_LEGACY,
        )
        b = jse.audit_jwl_energy(
            dimension="2D",
            mass_kg=1.0,
            rho_charge=1630.0,
            energy_j_per_kg=4.29e6,
            schema=jae.JWL_ENERGY_SCHEMA_LEGACY,
        )
        self.assertNotAlmostEqual(a.chemical_energy_added_j, b.chemical_energy_added_j)

    def test_legacy_3d_writes_the_same_catalog_e0_as_2d(self):
        b2 = jse.audit_jwl_energy(
            dimension="2D",
            mass_kg=1.0,
            rho_charge=1630.0,
            energy_j_per_kg=4.29e6,
            schema=jae.JWL_ENERGY_SCHEMA_LEGACY,
        )
        b3 = jse.audit_jwl_energy(
            dimension="3D",
            mass_kg=1.0,
            rho_charge=1630.0,
            energy_j_per_kg=4.29e6,
            schema=jae.JWL_ENERGY_SCHEMA_LEGACY,
        )
        self.assertEqual(b2.written_initiation_E0, b3.written_initiation_E0)
        self.assertAlmostEqual(b2.chemical_energy_added_j, b3.chemical_energy_added_j)


if __name__ == "__main__":
    unittest.main()
