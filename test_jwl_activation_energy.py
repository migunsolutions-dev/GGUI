"""JWL V2 activation energy: E0 = rho0 * E_charge in every dimension."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest

from dataclasses import replace

import jwl_activation_energy as jae
from generator_1d import Generator1D
from generator_2d import Generator2D
from generator_3d import Generator3D
from material_catalog import JWL_PARAMETERS
from models import SOURCE_MODEL_IG
from models_2d import CaseInputs2D
from test_generator_1d import _inputs, _rec
from test_generator_3d import _minimal


TNT_RHO = 1630.0
TNT_E = 4.29e6
TNT_E0_V2 = TNT_RHO * TNT_E
C4_RHO = 1601.0
C4_E = 4.52e6
C4_E0_V2 = C4_RHO * C4_E
C4_TUTORIAL_E0 = 9.0e9


def _initiation_e0(phase_properties: str) -> float:
    match = re.search(
        r"initiation\s*\{[^}]*?\bE0\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
        phase_properties,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("initiation E0 not found")
    return float(match.group(1))


def _products_eos(phase_properties: str) -> dict:
    match = re.search(
        r"products\s*\{.*?equationOfState\s*\{([^}]*)\}",
        phase_properties,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("products equationOfState not found")
    block = match.group(1)
    out = {}
    for key in ("A", "B", "R1", "R2", "omega"):
        found = re.search(rf"\b{key}\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", block)
        if found is None:
            raise AssertionError(f"{key} not found in products EOS")
        out[key] = float(found.group(1))
    return out


def _read(case_dir: str, *parts: str) -> str:
    with open(os.path.join(case_dir, *parts), encoding="utf-8") as handle:
        return handle.read()


class ConversionTests(unittest.TestCase):
    def test_tnt_identity(self):
        act = jae.v2_activation(TNT_E, TNT_RHO, material_name="TNT", dimension="1D")
        self.assertEqual(act.schema, jae.JWL_ENERGY_SCHEMA_V2)
        self.assertEqual(act.e0_j_per_kg, TNT_E)
        self.assertAlmostEqual(act.E0_pa, TNT_E0_V2)
        self.assertAlmostEqual(act.E0_pa, 6.9927e9)
        self.assertEqual(act.catalog_E0_legacy_pa, JWL_PARAMETERS["TNT"]["E0"])
        self.assertNotAlmostEqual(act.E0_pa, JWL_PARAMETERS["TNT"]["E0"])

    def test_c4_uses_rho_times_e_charge_not_tutorial(self):
        act = jae.v2_activation(C4_E, C4_RHO, material_name="C4", dimension="2D")
        self.assertAlmostEqual(act.E0_pa, C4_E0_V2)
        self.assertAlmostEqual(act.e0_j_per_kg, C4_E)
        self.assertNotAlmostEqual(act.E0_pa, C4_TUTORIAL_E0)
        self.assertEqual(act.catalog_E0_legacy_pa, C4_TUTORIAL_E0)

    def test_same_inputs_same_result_in_every_dimension(self):
        values = [
            jae.v2_activation(TNT_E, TNT_RHO, material_name="TNT", dimension=dim).E0_pa
            for dim in ("1D", "2D", "3D")
        ]
        self.assertEqual(len(set(values)), 1)
        self.assertAlmostEqual(values[0], TNT_E0_V2)

    def test_rejects_non_positive(self):
        with self.assertRaises(jae.JwlActivationEnergyError):
            jae.v2_activation(0.0, TNT_RHO)
        with self.assertRaises(jae.JwlActivationEnergyError):
            jae.v2_activation(TNT_E, -1.0)

    def test_legacy_writer_still_diverges_across_dimensions(self):
        one = jae.legacy_written_E0_pa(
            dimension="1D", energy_j_per_kg=TNT_E, material_name="TNT"
        )
        two = jae.legacy_written_E0_pa(
            dimension="2D", energy_j_per_kg=TNT_E, material_name="TNT"
        )
        self.assertEqual(one, TNT_E)
        self.assertEqual(two, JWL_PARAMETERS["TNT"]["E0"])
        self.assertNotAlmostEqual(one, two)

    def test_missing_sidecar_is_legacy(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(jae.read_jwl_energy_schema(td), jae.JWL_ENERGY_SCHEMA_LEGACY)

    def test_written_sidecar_is_v2(self):
        act = jae.v2_activation(TNT_E, TNT_RHO, material_name="TNT", dimension="1D")
        with tempfile.TemporaryDirectory() as td:
            jae.write_jwl_energy_audit(td, act, mass_kg=1.0)
            self.assertEqual(jae.read_jwl_energy_schema(td), jae.JWL_ENERGY_SCHEMA_V2)
            with open(os.path.join(td, jae.JWL_ENERGY_AUDIT_FILENAME), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertAlmostEqual(payload["blastfoam_initiation"]["E0_Pa"], TNT_E0_V2)
            self.assertAlmostEqual(payload["blastfoam_initiation"]["e0_J_kg"], TNT_E)


class GeneratedDictionaryTests(unittest.TestCase):
    def test_1d_2d_3d_write_the_same_v2_e0(self):
        with tempfile.TemporaryDirectory() as td:
            rec = _rec()
            inputs_1d = replace(
                _inputs("Terminate"),
                material_name="TNT",
                rho_charge=TNT_RHO,
                energy_j_per_kg=TNT_E,
                material_props={
                    "rho": TNT_RHO,
                    "A": JWL_PARAMETERS["TNT"]["A"],
                    "B": JWL_PARAMETERS["TNT"]["B"],
                    "R1": JWL_PARAMETERS["TNT"]["R1"],
                    "R2": JWL_PARAMETERS["TNT"]["R2"],
                    "omega": JWL_PARAMETERS["TNT"]["omega"],
                    "E0": TNT_E,
                },
            )
            case_1d = Generator1D(os.path.join(td, "one")).generate("tnt", inputs_1d, rec)
            case_2d = Generator2D(os.path.join(td, "two")).generate(
                "tnt",
                CaseInputs2D(
                    material_name="TNT",
                    rho_charge=TNT_RHO,
                    energy_j_per_kg=TNT_E,
                    mass_kg=1.0,
                    cell_size=0.05,
                ),
            )
            case_3d = Generator3D(os.path.join(td, "three")).generate(
                "tnt",
                _minimal(
                    material_name="TNT",
                    rho_charge=TNT_RHO,
                    energy_j_per_kg=TNT_E,
                    mass_kg=1.0,
                ),
            )
            e0_1d = _initiation_e0(_read(case_1d, "constant", "phaseProperties"))
            e0_2d = _initiation_e0(_read(case_2d, "constant", "phaseProperties"))
            e0_3d = _initiation_e0(_read(case_3d, "constant", "phaseProperties"))
            self.assertAlmostEqual(e0_1d, TNT_E0_V2, places=3)
            self.assertAlmostEqual(e0_2d, TNT_E0_V2, places=3)
            self.assertAlmostEqual(e0_3d, TNT_E0_V2, places=3)
            self.assertAlmostEqual(e0_1d, e0_2d, places=3)
            self.assertAlmostEqual(e0_2d, e0_3d, places=3)
            self.assertNotAlmostEqual(e0_2d, JWL_PARAMETERS["TNT"]["E0"])
            self.assertNotAlmostEqual(e0_1d, TNT_E)
            self.assertEqual(jae.read_jwl_energy_schema(case_1d), jae.JWL_ENERGY_SCHEMA_V2)
            self.assertEqual(jae.read_jwl_energy_schema(case_2d), jae.JWL_ENERGY_SCHEMA_V2)
            self.assertEqual(jae.read_jwl_energy_schema(case_3d), jae.JWL_ENERGY_SCHEMA_V2)
            eos_2d = _products_eos(_read(case_2d, "constant", "phaseProperties"))
            eos_3d = _products_eos(_read(case_3d, "constant", "phaseProperties"))
            catalog = JWL_PARAMETERS["TNT"]
            for key in ("A", "B", "R1", "R2", "omega"):
                self.assertAlmostEqual(eos_2d[key], float(catalog[key]))
            # 3D still writes A/B with the pre-V2 %.4g format; do not retune it.
            self.assertAlmostEqual(eos_3d["A"], float(f"{catalog['A']:.4g}"))
            self.assertAlmostEqual(eos_3d["B"], float(f"{catalog['B']:.4g}"))
            self.assertAlmostEqual(eos_3d["R1"], float(catalog["R1"]))
            self.assertAlmostEqual(eos_3d["R2"], float(catalog["R2"]))
            self.assertAlmostEqual(eos_3d["omega"], float(catalog["omega"]))

    def test_c4_generated_e0_is_not_tutorial_9e9(self):
        with tempfile.TemporaryDirectory() as td:
            case = Generator2D(td).generate(
                "c4",
                CaseInputs2D(
                    material_name="C4",
                    rho_charge=C4_RHO,
                    energy_j_per_kg=C4_E,
                    mass_kg=1.0,
                    cell_size=0.05,
                ),
            )
            written = _initiation_e0(_read(case, "constant", "phaseProperties"))
            self.assertAlmostEqual(written, C4_E0_V2, places=3)
            self.assertNotAlmostEqual(written, C4_TUTORIAL_E0)

    def test_custom_uses_the_same_conversion_not_catalog_or_raw_e0(self):
        rho = 1600.0
        energy = 4.50e6
        props = {
            "rho": rho,
            "energy": energy,
            "A": 300.0e9,
            "B": 3.0e9,
            "R1": 4.0,
            "R2": 1.0,
            "omega": 0.30,
            "E0": 9.0e9,
        }
        with tempfile.TemporaryDirectory() as td:
            case = Generator3D(td).generate(
                "custom",
                _minimal(
                    material_name="Custom",
                    rho_charge=rho,
                    energy_j_per_kg=energy,
                    material_props=props,
                    mass_kg=1.0,
                ),
            )
            written = _initiation_e0(_read(case, "constant", "phaseProperties"))
            self.assertAlmostEqual(written, rho * energy, places=3)
            self.assertNotAlmostEqual(written, 9.0e9)
            self.assertNotAlmostEqual(written, energy)

    def test_ig_case_does_not_write_jwl_energy_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            inputs = replace(_inputs("Terminate"), source_model=SOURCE_MODEL_IG)
            case = Generator1D(td).generate("ig", inputs, _rec())
            self.assertFalse(
                os.path.isfile(os.path.join(case, jae.JWL_ENERGY_AUDIT_FILENAME))
            )
            self.assertEqual(jae.read_jwl_energy_schema(case), jae.JWL_ENERGY_SCHEMA_LEGACY)
            self.assertFalse(os.path.isfile(os.path.join(case, "0.orig", "alpha.c4")))


if __name__ == "__main__":
    unittest.main()
