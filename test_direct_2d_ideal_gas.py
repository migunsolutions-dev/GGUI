"""Direct 2D Ideal-Gas is a single-phase burst and does not alter JWL or remap."""
import os
import tempfile
import unittest
from dataclasses import replace

from generator_2d import Generator2D
from ig_source_state import GAMMA_IDEAL_GAS, CV_IDEAL_GAS, direct_axisymmetric_burst
from models import SOURCE_MODEL_IG, SOURCE_MODEL_JWL, SOURCE_MODEL_LABELS
from models_2d import CaseInputs2D
from project_io import _case_inputs_2d_from_dict


def _read(case, name):
    with open(os.path.join(case, name), encoding="utf-8") as handle:
        return handle.read()


class DirectIdealGasTests(unittest.TestCase):
    def test_label_and_old_identity(self):
        self.assertEqual(SOURCE_MODEL_LABELS[SOURCE_MODEL_IG], "Ideal-Gas")
        self.assertEqual(SOURCE_MODEL_IG, "IG_ISOTHERMAL_BURST")
        self.assertEqual(SOURCE_MODEL_LABELS[SOURCE_MODEL_JWL], "JWL Detonation")
        loaded = _case_inputs_2d_from_dict({"radius": 1.5, "height": 1.5})
        self.assertEqual(loaded.source_model, SOURCE_MODEL_JWL)

    def test_burst_matches_one_d_energy_rule(self):
        burst = direct_axisymmetric_burst(
            mass_kg=1.0,
            rho_charge=1630.0,
            energy_j_per_kg=4.29e6,
            p_atm=101325.0,
            t_atm=288.15,
        )
        self.assertAlmostEqual(burst.gamma, GAMMA_IDEAL_GAS)
        self.assertAlmostEqual(burst.cv, CV_IDEAL_GAS)
        self.assertAlmostEqual(burst.volume_m3, 1.0 / 1630.0)
        self.assertAlmostEqual(burst.rho_source * burst.volume_m3, 1.0)
        self.assertAlmostEqual(burst.e_source, burst.cv * burst.t_ambient + 4.29e6)
        self.assertAlmostEqual(burst.t_source, burst.e_source / burst.cv)
        self.assertAlmostEqual(
            burst.p_source, (burst.gamma - 1.0) * burst.rho_source * burst.e_source
        )
        self.assertAlmostEqual(
            burst.p_ambient, (burst.gamma - 1.0) * burst.rho_ambient * burst.e_ambient
        )
        self.assertGreater(burst.t_source, burst.t_ambient)

    def test_direct_case_is_single_phase_and_jwl_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            gen = Generator2D(root)
            ig = gen.generate(
                "ig",
                replace(
                    CaseInputs2D(),
                    source_model=SOURCE_MODEL_IG,
                    mesh_mode="Fixed Mesh",
                    cell_size=0.015,
                    radius=0.30,
                    height=0.30,
                    height_of_burst=0.15,
                    mass_kg=1.0,
                    rho_charge=1630.0,
                    energy_j_per_kg=4.29e6,
                ),
            )
            phase = _read(ig, "constant/phaseProperties")
            schemes = _read(ig, "system/fvSchemes")
            fields = _read(ig, "system/setFieldsDict")
            self.assertIn("equationOfState idealGas", phase)
            self.assertIn("gamma 1.4", phase)
            self.assertIn("Cv 718", phase)
            self.assertNotIn("phases", phase)
            for banned in (
                "JWL",
                "BirchMurnaghan3",
                "pressureBased",
                "activationModel",
                "lambda",
                "Esref",
                "c4",
            ):
                self.assertNotIn(banned, phase)
            self.assertIn("timeIntegrator Euler", schemes)
            self.assertIn("fluxScheme Tadmor", schemes)
            self.assertIn('"reconstruct(U)" vanLeer', schemes)
            self.assertNotIn("lambda.c4", schemes)
            self.assertNotIn("alpha.c4", fields)
            self.assertIn("sphereToCell", fields)
            self.assertNotIn("JWL", _read(ig, "0.orig/p"))
            self.assertFalse(os.path.isfile(os.path.join(ig, "0.orig", "alpha.c4")))

            jwl = gen.generate(
                "jwl",
                replace(
                    CaseInputs2D(),
                    mesh_mode="Fixed Mesh",
                    cell_size=0.015,
                    radius=0.30,
                    height=0.30,
                    height_of_burst=0.15,
                    detonation_height=0.15,
                ),
            )
            jwl_phase = _read(jwl, "constant/phaseProperties")
            self.assertIn("BirchMurnaghan3", jwl_phase)
            self.assertIn("pressureBased", jwl_phase)
            self.assertIn("equationOfState JWL", jwl_phase)
            self.assertIn("Esref -5048371.41227", jwl_phase)
            self.assertIn("timeIntegrator Euler", _read(jwl, "system/fvSchemes"))
            self.assertIn("div(alphaRhoPhi.c4,lambda.c4) Gauss linear", _read(jwl, "system/fvSchemes"))


if __name__ == "__main__":
    unittest.main()
