"""Generated 1D IG cases use the single-phase Sedov schema and added-energy state."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace

from generator_1d import Generator1D
from ig_source_state import ENERGY_CONVENTION_DETONATION_ADDED, derive_ig_state
from models import (
    BOUNDARY_1D_TERMINATE,
    CaseInputs1D,
    RecommendedParams1D,
    SOURCE_MODEL_IG,
    SOURCE_MODEL_JWL,
)
from test_generator_1d import _inputs, _rec


class Generator1DIGTests(unittest.TestCase):
    def _generate(self, source_model: str):
        root = tempfile.mkdtemp(prefix="ggui_1d_ig_")
        gen = Generator1D(root)
        inputs = replace(
            _inputs(BOUNDARY_1D_TERMINATE),
            source_model=source_model,
            rho_charge=1630.0,
            energy_j_per_kg=4.29e6,
            cell_size=0.001,
            material_name="TNT",
        )
        return gen.generate("ig_case", inputs, _rec()), inputs

    def _read(self, case_dir: str, *parts: str) -> str:
        with open(os.path.join(case_dir, *parts), encoding="utf-8") as handle:
            return handle.read()

    def test_ig_case_has_no_jwl_or_phase_fields(self):
        case_dir, _inputs_ig = self._generate(SOURCE_MODEL_IG)
        pp = self._read(case_dir, "constant", "phaseProperties")
        self.assertNotIn("phases", pp)
        self.assertNotIn("detonating", pp)
        self.assertNotIn("activationModel", pp)
        self.assertNotIn("initiation", pp)
        self.assertIn("equationOfState idealGas", pp)
        self.assertTrue(os.path.isfile(os.path.join(case_dir, "0.orig", "rho")))
        self.assertFalse(os.path.isfile(os.path.join(case_dir, "0.orig", "alpha.c4")))
        self.assertFalse(os.path.isfile(os.path.join(case_dir, "0.orig", "rho.c4")))
        schemes = self._read(case_dir, "system", "fvSchemes")
        self.assertIn("fluxScheme      Tadmor", schemes)
        self.assertNotIn("alpha.c4", schemes)
        self.assertNotIn("lambda.c4", schemes)
        allrun = self._read(case_dir, "Allrun")
        self.assertIn("check_ig_source.sh", allrun)
        self.assertNotIn("check_alpha_c4.sh", allrun)

    def test_ig_setfields_uses_added_energy_pressure(self):
        case_dir, inputs = self._generate(SOURCE_MODEL_IG)
        rec = _rec()
        state = derive_ig_state(
            mass_kg=inputs.mass_kg,
            rho_charge=inputs.rho_charge,
            energy_j_per_kg=inputs.energy_j_per_kg,
            p_atm=inputs.p_atm,
            t_atm=inputs.t_atm,
            r_min_m=rec.r_min,
            cell_size_m=inputs.cell_size,
        )
        self.assertEqual(state.production_energy_convention, ENERGY_CONVENTION_DETONATION_ADDED)
        sf = self._read(case_dir, "system", "setFieldsDict")
        compact = sf.replace(" ", "")
        self.assertIn(f"volScalarFieldValuerho{state.rho_source:.12g}", compact)
        self.assertIn(f"volScalarFieldValuep{state.p_source:.12g}", compact)
        self.assertNotIn("alpha.c4", sf)

    def test_jwl_case_still_writes_two_phase_dictionaries(self):
        case_dir, inputs_jwl = self._generate(SOURCE_MODEL_JWL)
        pp = self._read(case_dir, "constant", "phaseProperties")
        self.assertIn("phases (c4 air)", pp)
        self.assertIn("type detonating", pp)
        self.assertTrue(os.path.isfile(os.path.join(case_dir, "0.orig", "alpha.c4")))
        self.assertFalse(os.path.isfile(os.path.join(case_dir, "0.orig", "rho")))
        expected_e0 = float(inputs_jwl.rho_charge) * float(inputs_jwl.energy_j_per_kg)
        self.assertIn(f"E0 {expected_e0:.12g}", pp)
        self.assertTrue(os.path.isfile(os.path.join(case_dir, "ggui_jwl_energy.json")))
        self.assertFalse(os.path.isfile(os.path.join(case_dir, "ggui_ig_source_audit.json")))

    def test_ig_case_does_not_write_jwl_energy_sidecar(self):
        case_dir, _inputs_ig = self._generate(SOURCE_MODEL_IG)
        self.assertFalse(os.path.isfile(os.path.join(case_dir, "ggui_jwl_energy.json")))
        self.assertTrue(os.path.isfile(os.path.join(case_dir, "ggui_ig_source_audit.json")))


if __name__ == "__main__":
    unittest.main()
