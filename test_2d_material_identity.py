"""Loading a generated 2D case must not treat the phase name c4 as material C4."""
import unittest

from imported_case_mapping_2d import ImportMappingResult, map_material
from material_catalog import JWL_PARAMETERS, MATERIALS


class MaterialIdentityTests(unittest.TestCase):
    def test_tnt_jwl_is_not_renamed_c4(self):
        tnt = JWL_PARAMETERS["TNT"]
        phases = {
            "path": "constant/phaseProperties",
            "phases": ["c4", "air"],
            "rho0": 1630.0,
            "E0": 1630.0 * MATERIALS["TNT"]["energy"],
            "A": tnt["A"],
            "B": tnt["B"],
            "R1": tnt["R1"],
            "R2": tnt["R2"],
            "omega": tnt["omega"],
        }
        result = ImportMappingResult()
        map_material(result, {"phase_field": "alpha.c4"}, phases, [])
        self.assertEqual(result.gui_values.get("material_name"), "TNT")
        self.assertAlmostEqual(
            result.gui_values.get("energy_j_per_kg"),
            MATERIALS["TNT"]["energy"],
        )

    def test_c4_jwl_stays_c4(self):
        c4 = JWL_PARAMETERS["C4"]
        phases = {
            "path": "constant/phaseProperties",
            "phases": ["c4", "air"],
            "rho0": 1601.0,
            "A": c4["A"],
            "B": c4["B"],
            "R1": c4["R1"],
            "R2": c4["R2"],
            "omega": c4["omega"],
        }
        result = ImportMappingResult()
        map_material(result, {"phase_field": "alpha.c4"}, phases, [])
        self.assertEqual(result.gui_values.get("material_name"), "C4")

    def test_phase_name_still_used_when_jwl_is_absent(self):
        phases = {
            "path": "constant/phaseProperties",
            "phases": ["c4", "air"],
            "rho0": 1601.0,
        }
        result = ImportMappingResult()
        map_material(result, {"phase_field": "alpha.c4"}, phases, [])
        self.assertEqual(result.gui_values.get("material_name"), "C4")


if __name__ == "__main__":
    unittest.main()
