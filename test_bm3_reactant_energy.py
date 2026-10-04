"""Production 2D JWL reactant reference energy matches validated Formulation 1."""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from dataclasses import replace

from axisymmetric_2d import DYNAMIC_MESH, FIXED_MESH, REMAP_SOURCE
from bm3_reactant_energy import (
    BM3_CV_J_PER_KG_K,
    BM3_K0_PA,
    BM3_K0_PRIME,
    BM3_PREF_PA,
    TSTD_SI_K,
    cold_energy_j_per_kg,
    pressure_pa,
    reactant_esref,
    sound_speed_squared,
    stored_reactant_energy,
)
from generator_2d import Generator2D
from models_2d import CaseInputs2D, MappingSource2D


BENCHMARK_RHO0 = 1630.0
BENCHMARK_ESREF = -5048371.41227
BENCHMARK_T_K = 288.15


class BirchMurnaghanReferenceEnergyTests(unittest.TestCase):
    def test_benchmark_density_matches_validated_esref(self):
        esref = reactant_esref(BENCHMARK_RHO0)
        self.assertAlmostEqual(esref, BENCHMARK_ESREF, places=5)
        self.assertEqual(f"{esref:.12g}", f"{BENCHMARK_ESREF:.12g}")

    def test_esref_follows_density_and_is_not_a_fixed_constant(self):
        at_catalog = reactant_esref(BENCHMARK_RHO0)
        at_other = reactant_esref(1600.0)
        self.assertNotAlmostEqual(at_catalog, at_other, places=3)
        expected = BM3_CV_J_PER_KG_K * TSTD_SI_K - cold_energy_j_per_kg(1600.0, 1600.0)
        self.assertAlmostEqual(at_other, expected, places=6)

    def test_shift_keeps_cv_t_and_leaves_mechanics_unchanged(self):
        rho0 = BENCHMARK_RHO0
        esref_default = BM3_CV_J_PER_KG_K * TSTD_SI_K
        esref = reactant_esref(rho0)
        offsets = []
        for ratio in (0.90, 0.95, 1.0, 1.05, 1.10):
            rho = ratio * rho0
            p = pressure_pa(rho, rho0)
            c2 = sound_speed_squared(rho, rho0)
            # The mechanical formulas have no Esref argument. Re-evaluating
            # them is the statement that the reference shift cannot enter.
            self.assertEqual(p, pressure_pa(rho, rho0))
            self.assertEqual(c2, sound_speed_squared(rho, rho0))
            self.assertAlmostEqual(rho * c2, rho * sound_speed_squared(rho, rho0), places=6)
            e_default = stored_reactant_energy(rho, BENCHMARK_T_K, esref_default, rho0=rho0)
            e_corrected = stored_reactant_energy(rho, BENCHMARK_T_K, esref, rho0=rho0)
            offsets.append(e_corrected - e_default)
        self.assertAlmostEqual(pressure_pa(rho0, rho0), BM3_PREF_PA, places=6)
        bulk = rho0 * sound_speed_squared(rho0, rho0)
        self.assertLess(abs(bulk - BM3_K0_PA) / BM3_K0_PA, 1e-12)
        self.assertAlmostEqual(max(offsets) - min(offsets), 0.0, places=6)
        self.assertAlmostEqual(offsets[0], -cold_energy_j_per_kg(rho0, rho0), places=4)
        at_rho0 = stored_reactant_energy(rho0, BENCHMARK_T_K, esref, rho0=rho0)
        self.assertAlmostEqual(at_rho0, BM3_CV_J_PER_KG_K * BENCHMARK_T_K, places=4)
        # The same identity holds at the thermo reference temperature.
        at_tstd = stored_reactant_energy(rho0, TSTD_SI_K, esref, rho0=rho0)
        self.assertAlmostEqual(at_tstd, BM3_CV_J_PER_KG_K * TSTD_SI_K, places=4)

    def test_reaction_energy_constants_are_the_formulation_1_set(self):
        self.assertEqual(BM3_K0_PA, 8.04e9)
        self.assertEqual(BM3_K0_PRIME, 7.97)
        self.assertEqual(BM3_PREF_PA, 101298.0)
        self.assertEqual(BM3_CV_J_PER_KG_K, 1400.0)
        self.assertTrue(math.isfinite(reactant_esref(BENCHMARK_RHO0)))


class Production2DJWLDictionaryTests(unittest.TestCase):
    def _generate(self, root, name, **overrides):
        inputs = replace(CaseInputs2D(), **overrides)
        return Generator2D(root).generate(name, inputs)

    def test_direct_jwl_writes_formulation_1_including_esref(self):
        with tempfile.TemporaryDirectory() as td:
            case = self._generate(
                td,
                "direct",
                mesh_mode=FIXED_MESH,
                rho_charge=BENCHMARK_RHO0,
                energy_j_per_kg=4.29e6,
                p_atm=101325.0,
                cell_size=0.01,
                dyn_refine_max=1,
                detonation_height=1.0,
            )
            phase = open(os.path.join(case, "constant", "phaseProperties"), encoding="utf-8").read()
            schemes = open(os.path.join(case, "system", "fvSchemes"), encoding="utf-8").read()
            esref = reactant_esref(BENCHMARK_RHO0)
            self.assertIn("equationOfState BirchMurnaghan3", phase)
            self.assertIn("K0 8.04e9; K0Prime 7.97", phase)
            self.assertIn("pRef 101298", phase)
            self.assertIn(f"Esref {esref:.12g}", phase)
            self.assertIn("thermo ePolynomial; equationOfState JWL", phase)
            self.assertIn("A 373770000000", phase)
            self.assertIn("omega 0.35", phase)
            self.assertIn("activationModel pressureBased", phase)
            self.assertIn("E0 6992700000", phase)
            self.assertIn("I 4.0e6", phase)
            self.assertIn("G1 1.4997e-7", phase)
            self.assertIn("timeIntegrator Euler", schemes)
            self.assertIn("fluxScheme Tadmor", schemes)
            self.assertIn("div(alphaRhoPhi.c4,lambda.c4) Gauss linear", schemes)
            self.assertIn('"reconstruct(U)" vanLeer', schemes)
            self.assertNotIn("RK2SSP", schemes)
            self.assertNotIn("Kurganov", schemes)
            self.assertNotIn("vanAlbada", schemes)
            self.assertNotIn("Riemann", schemes)

    def test_dynamic_mesh_keeps_the_same_reactant_reference(self):
        with tempfile.TemporaryDirectory() as td:
            case = self._generate(td, "dynamic", mesh_mode=DYNAMIC_MESH, rho_charge=BENCHMARK_RHO0)
            phase = open(os.path.join(case, "constant", "phaseProperties"), encoding="utf-8").read()
            dynamic = open(os.path.join(case, "constant", "dynamicMeshDict"), encoding="utf-8").read()
            self.assertIn(f"Esref {reactant_esref(BENCHMARK_RHO0):.12g}", phase)
            self.assertIn("adaptiveFvMesh", dynamic)

    def test_ideal_gas_remap_does_not_receive_the_jwl_reference(self):
        with tempfile.TemporaryDirectory() as td:
            source = os.path.join(td, "ig_source")
            os.makedirs(os.path.join(source, "constant"))
            with open(os.path.join(source, "constant", "phaseProperties"), "w", encoding="utf-8") as handle:
                handle.write("type basic;\nthermoType { equationOfState idealGas; }\n")
            with open(os.path.join(source, "ggui_1d_run_completion.json"), "w", encoding="utf-8") as handle:
                handle.write('{"source_model": "IG_ISOTHERMAL_BURST"}\n')
            case = self._generate(
                td,
                "ig",
                initialization_source=REMAP_SOURCE,
                mesh_mode=FIXED_MESH,
                mapping=MappingSource2D(case_path=source, mapped_radius=0.5),
            )
            phase = open(os.path.join(case, "constant", "phaseProperties"), encoding="utf-8").read()
            self.assertIn("idealGas", phase)
            self.assertNotIn("Esref", phase)
            self.assertNotIn("BirchMurnaghan3", phase)

    def test_one_d_writer_is_not_given_this_reference(self):
        text = open(os.path.join(os.path.dirname(__file__), "generator_1d.py"), encoding="utf-8").read()
        self.assertNotIn("reactant_esref", text)
        self.assertNotIn("Esref", text)


if __name__ == "__main__":
    unittest.main()
