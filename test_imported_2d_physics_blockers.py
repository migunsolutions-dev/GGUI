"""Phase 1: block silent physics substitution for imported Cylindrical–2D cases."""
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication, QMessageBox

from domain_errors import (
    IncompleteMaterialError,
    MissingRequiredInputError,
    UnknownMaterialError,
)
from imported_case_mapping_2d import FieldProvenance, MappedField, map_imported_case_to_gui
from material_catalog import JWL_PARAMETERS, jwl_parameters
from material_validation import (
    REQUIRED_CUSTOM_MATERIAL_KEYS,
    validate_material_definition,
    validate_required_values,
)
from models_2d import CaseInputs2D
from project_io import build_project, read_project, write_project_atomic
from test_external_working_copy_2d import _make_unmeshed_source
from test_generator_3d import _minimal

app = QApplication.instance() or QApplication([])


def _complete_custom(**overrides):
    props = {
        "rho": 1600.0,
        "energy": 4.5e6,
        "A": 300.0e9,
        "B": 3.0e9,
        "R1": 4.0,
        "R2": 1.0,
        "omega": 0.30,
    }
    props.update(overrides)
    return props


class MaterialCatalogFallbackTests(unittest.TestCase):
    def test_recognized_catalog_material_c4(self):
        params = jwl_parameters("C4")
        self.assertEqual(params["A"], JWL_PARAMETERS["C4"]["A"])
        self.assertNotEqual(params["A"], JWL_PARAMETERS["TNT"]["A"])

    def test_unrecognized_material_raises_unknown(self):
        with self.assertRaises(UnknownMaterialError):
            jwl_parameters("NotARealExplosive")

    def test_no_fallback_to_c4(self):
        with self.assertRaises(UnknownMaterialError) as ctx:
            jwl_parameters("mysteryHE")
        self.assertNotIn("C4", str(ctx.exception).split("Unknown")[0])
        self.assertIn("substitute", str(ctx.exception).lower())

    def test_no_fallback_to_tnt(self):
        with self.assertRaises(UnknownMaterialError):
            jwl_parameters("mysteryHE")
        # Must not return TNT parameters.
        self.assertNotEqual(
            JWL_PARAMETERS["TNT"]["A"],
            JWL_PARAMETERS["C4"]["A"],
        )

    def test_no_fallback_to_any_catalog_material(self):
        for name in ("ghost", "RDX", "HMX", ""):
            with self.assertRaises(UnknownMaterialError):
                jwl_parameters(name)

    def test_complete_custom_material(self):
        props = _complete_custom()
        params = jwl_parameters("Custom", props)
        self.assertAlmostEqual(params["A"], props["A"])
        self.assertAlmostEqual(params["E0"], props["energy"])

    def test_custom_missing_each_required_key(self):
        for key in ("A", "B", "R1", "R2", "omega", "rho", "energy"):
            props = _complete_custom()
            if key == "rho":
                # rho is enforced by validate_material_definition / required values.
                issues = validate_material_definition(
                    "Custom", {k: v for k, v in props.items() if k != "rho"}
                )
                self.assertTrue(any(key in i.message or "rho" in i.message for i in issues))
                continue
            if key == "energy":
                props.pop("energy")
            else:
                props.pop(key)
            with self.assertRaises(IncompleteMaterialError):
                jwl_parameters("Custom", props)

    def test_custom_missing_e0_and_energy(self):
        props = _complete_custom()
        props.pop("energy")
        with self.assertRaises(IncompleteMaterialError) as ctx:
            jwl_parameters("Custom", props)
        self.assertIn("E0", str(ctx.exception))

    def test_non_finite_custom_material_value(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            props = _complete_custom(A=bad)
            with self.assertRaises(IncompleteMaterialError):
                jwl_parameters("Custom", props)
            issues = validate_material_definition("Custom", props)
            self.assertTrue(issues)


class RequiredValuesValidationTests(unittest.TestCase):
    def test_multiple_missing_values_reported_together(self):
        inputs = replace(
            CaseInputs2D(),
            material_name="",
            rho_charge=None,
            energy_j_per_kg=None,
            mass_kg=None,
            undefined_keys=("material_name", "rho_charge", "energy_j_per_kg", "mass_kg"),
        )
        result = validate_required_values(
            inputs,
            undefined_keys=inputs.undefined_keys,
            require_imported_physics=True,
        )
        self.assertFalse(result.ok)
        joined = "\n".join(result.errors)
        self.assertIn("Material", joined)
        self.assertIn("density", joined.lower())
        self.assertIn("energy", joined.lower())
        self.assertIn("mass", joined.lower())
        self.assertGreaterEqual(len(result.issues), 4)

    def test_unsupported_source_representation_reported_specifically(self):
        inputs = replace(CaseInputs2D(), material_name="", undefined_keys=("material_name",))
        meta = {
            "material_name": MappedField(
                key="material_name",
                displayed_value=None,
                provenance=FieldProvenance.CASE_DEFINED,
                reason="phase 'pbx9502' has no GGUI catalog entry — case-defined",
            )
        }
        result = validate_required_values(
            inputs,
            undefined_keys=("material_name",),
            imported_field_meta=meta,
            require_imported_physics=True,
        )
        self.assertFalse(result.ok)
        self.assertTrue(any(i.code == "unsupported_source" for i in result.issues))
        self.assertIn("Unsupported source representation", "\n".join(result.errors))

    def test_raise_if_invalid_uses_domain_errors(self):
        inputs = replace(CaseInputs2D(), material_name="ghostHE")
        with self.assertRaises(UnknownMaterialError):
            validate_required_values(inputs).raise_if_invalid()


class ImportedMappingPhysicsTests(unittest.TestCase):
    def test_recognized_import_still_maps_c4(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            mapping = map_imported_case_to_gui(str(source))
            self.assertEqual(mapping.gui_values.get("material_name"), "C4")
            self.assertNotIn("material_name", mapping.not_recovered_keys)

    def test_unrecognized_material_not_substituted(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            phase = (source / "constant" / "phaseProperties").read_text(encoding="utf-8")
            phase = phase.replace("phases (c4 air);", "phases (pbx9502 air);")
            phase = phase.replace("\nc4\n", "\npbx9502\n", 1)
            (source / "constant" / "phaseProperties").write_text(phase, encoding="utf-8")
            alpha = source / "0.orig" / "alpha.c4"
            if alpha.is_file():
                alpha.rename(source / "0.orig" / "alpha.pbx9502")
            setfields = (source / "system" / "setFieldsDict").read_text(encoding="utf-8")
            setfields = setfields.replace("alpha.c4", "alpha.pbx9502")
            (source / "system" / "setFieldsDict").write_text(setfields, encoding="utf-8")
            mapping = map_imported_case_to_gui(str(source))
            self.assertNotIn("material_name", mapping.gui_values)
            self.assertIsNone(mapping.fields["material_name"].displayed_value)
            self.assertNotEqual(mapping.fields["material_name"].displayed_value, "C4")
            self.assertNotEqual(mapping.fields["material_name"].displayed_value, "TNT")
            self.assertTrue(mapping.unsupported_features)


class ImportedGuiUndefinedTests(unittest.TestCase):
    def _make_tab_with_import(self, mutate_source=None):
        from case_loader_2d import inspect_imported_axisymmetric_case
        from external_case_workflow_2d import ImportMode2D
        from tab_2d import Tab2D

        td = tempfile.TemporaryDirectory()
        source = _make_unmeshed_source(Path(td.name) / "axisymmetricCharge")
        if mutate_source:
            mutate_source(source)
        state = inspect_imported_axisymmetric_case(str(source))
        state.mode = ImportMode2D.IMPORTED_2D_UNINITIALIZED
        tab = Tab2D()
        tab.load_imported_case(state, apply_mapping=True)
        return tab, td

    def test_combo_default_not_mistaken_for_recovered_material(self):
        def mutate(source: Path):
            phase = (source / "constant" / "phaseProperties").read_text(encoding="utf-8")
            phase = phase.replace("phases (c4 air);", "phases (pbx9502 air);")
            phase = phase.replace("\nc4\n", "\npbx9502\n", 1)
            (source / "constant" / "phaseProperties").write_text(phase, encoding="utf-8")
            setfields = (source / "system" / "setFieldsDict").read_text(encoding="utf-8")
            setfields = setfields.replace("alpha.c4", "alpha.pbx9502")
            (source / "system" / "setFieldsDict").write_text(setfields, encoding="utf-8")

        tab, td = self._make_tab_with_import(mutate)
        try:
            from tab_2d import MATERIAL_UNDEFINED_PLACEHOLDER

            self.assertEqual(tab.cmb_material.currentText(), MATERIAL_UNDEFINED_PLACEHOLDER)
            inputs = tab.get_case_inputs()
            self.assertEqual(inputs.material_name, "")
            self.assertIn("material_name", inputs.undefined_keys)
            self.assertNotEqual(inputs.material_name, "TNT")
            self.assertNotEqual(inputs.material_name, "C4")
        finally:
            td.cleanup()

    def test_numeric_widget_default_not_mistaken_for_recovered_density(self):
        def mutate(source: Path):
            # Remove recoverable density sources.
            phase = (source / "constant" / "phaseProperties").read_text(encoding="utf-8")
            phase = phase.replace("phases (c4 air);", "phases (pbx9502 air);")
            phase = phase.replace("\nc4\n", "\npbx9502\n", 1)
            # Drop rho0 lines.
            lines = [ln for ln in phase.splitlines() if "rho0" not in ln]
            (source / "constant" / "phaseProperties").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            setfields = (source / "system" / "setFieldsDict").read_text(encoding="utf-8")
            setfields = setfields.replace("alpha.c4", "alpha.pbx9502")
            # Remove rho from mass regions if present.
            setfields = setfields.replace("rho 1601;", "")
            setfields = setfields.replace("rho 1601", "")
            (source / "system" / "setFieldsDict").write_text(setfields, encoding="utf-8")

        tab, td = self._make_tab_with_import(mutate)
        try:
            inputs = tab.get_case_inputs()
            self.assertIn("rho_charge", inputs.undefined_keys)
            self.assertIsNone(inputs.rho_charge)
            self.assertNotEqual(inputs.rho_charge, 1630.0)
        finally:
            td.cleanup()

    def test_undefined_values_survive_save_and_reload(self):
        inputs = replace(
            CaseInputs2D(),
            material_name="",
            rho_charge=None,
            energy_j_per_kg=None,
            undefined_keys=("material_name", "rho_charge", "energy_j_per_kg"),
        )
        payload = build_project(_minimal(), probes={"probes": []}, gui_state={}, inputs_2d=inputs)
        with tempfile.NamedTemporaryFile(suffix=".ggui.json", delete=False) as stream:
            path = stream.name
        try:
            write_project_atomic(path, payload)
            restored = read_project(path)["inputs_2d"]
            self.assertEqual(restored.material_name, "")
            self.assertIsNone(restored.rho_charge)
            self.assertIsNone(restored.energy_j_per_kg)
            self.assertEqual(
                set(restored.undefined_keys),
                {"material_name", "rho_charge", "energy_j_per_kg"},
            )
        finally:
            os.unlink(path)

    def test_get_case_inputs_does_not_complete_undefined_widgets(self):
        def mutate(source: Path):
            phase = (source / "constant" / "phaseProperties").read_text(encoding="utf-8")
            phase = phase.replace("phases (c4 air);", "phases (pbx9502 air);")
            phase = phase.replace("\nc4\n", "\npbx9502\n", 1)
            (source / "constant" / "phaseProperties").write_text(phase, encoding="utf-8")
            setfields = (source / "system" / "setFieldsDict").read_text(encoding="utf-8")
            setfields = setfields.replace("alpha.c4", "alpha.pbx9502")
            (source / "system" / "setFieldsDict").write_text(setfields, encoding="utf-8")

        tab, td = self._make_tab_with_import(mutate)
        try:
            tab._mark_control_undefined("energy_j_per_kg")
            inputs = tab.get_case_inputs()
            self.assertEqual(inputs.material_name, "")
            self.assertIsNone(inputs.energy_j_per_kg)
            self.assertTrue(inputs.undefined_keys)
        finally:
            td.cleanup()


class InitialiseModelBlockedTests(unittest.TestCase):
    def test_initialise_blocked_and_generator_not_called(self):
        from case_loader_2d import inspect_imported_axisymmetric_case
        from external_case_workflow_2d import ImportMode2D
        from main_new import BlastFoamApp
        from tab_2d import MATERIAL_UNDEFINED_PLACEHOLDER, Tab2D

        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            phase = (source / "constant" / "phaseProperties").read_text(encoding="utf-8")
            phase = phase.replace("phases (c4 air);", "phases (pbx9502 air);")
            phase = phase.replace("\nc4\n", "\npbx9502\n", 1)
            (source / "constant" / "phaseProperties").write_text(phase, encoding="utf-8")
            setfields = (source / "system" / "setFieldsDict").read_text(encoding="utf-8")
            setfields = setfields.replace("alpha.c4", "alpha.pbx9502")
            (source / "system" / "setFieldsDict").write_text(setfields, encoding="utf-8")

            win = BlastFoamApp.__new__(BlastFoamApp)
            win.tab_2d = Tab2D()
            win.status_bar = mock.Mock()
            win.service = mock.Mock()
            win.service.generate_case = mock.Mock(
                side_effect=AssertionError("generator called")
            )
            win._resolved_case_root = mock.Mock(return_value=td)
            win.active_case_initialized_2d = False
            win._prep_phase = "idle"
            win._prep_worker = None
            win._force_sync_prep = True
            win.openfoam_bashrc = ""

            state = inspect_imported_axisymmetric_case(str(source))
            state.mode = ImportMode2D.IMPORTED_2D_UNINITIALIZED
            win.tab_2d.load_imported_case(state, apply_mapping=True)
            self.assertEqual(
                win.tab_2d.cmb_material.currentText(), MATERIAL_UNDEFINED_PLACEHOLDER
            )

            with mock.patch.object(
                QMessageBox, "critical", return_value=QMessageBox.Ok
            ) as crit:
                win.on_initialize_imported_model_2d()
                self.assertTrue(crit.called)
            win.service.generate_case.assert_not_called()
            self.assertFalse(win.active_case_initialized_2d)
            self.assertNotEqual(
                win.tab_2d.import_mode, ImportMode2D.IMPORTED_2D_READY
            )

    def test_valid_imported_case_still_passes_required_validation(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            mapping = map_imported_case_to_gui(str(source))
            inputs = replace(
                CaseInputs2D(),
                **{
                    k: v
                    for k, v in mapping.gui_values.items()
                    if k in CaseInputs2D.__dataclass_fields__
                },
            )
            result = validate_required_values(
                inputs,
                imported_field_meta=mapping.fields,
                unsupported_features=mapping.unsupported_features,
                require_imported_physics=True,
            )
            self.assertTrue(result.ok, result.errors)

    def test_generator_rejects_undefined_without_writing_case(self):
        from generator_2d import Generator2D

        with tempfile.TemporaryDirectory() as td:
            gen = Generator2D(td)
            inputs = replace(
                CaseInputs2D(),
                material_name="",
                undefined_keys=("material_name",),
            )
            with self.assertRaises(
                (MissingRequiredInputError, UnknownMaterialError, IncompleteMaterialError)
            ):
                gen.generate("blocked", inputs)
            self.assertFalse((Path(td) / "blocked").exists())


class NewWorkflowUnchangedTests(unittest.TestCase):
    def test_native_new_2d_defaults_still_complete(self):
        from tab_2d import Tab2D

        tab = Tab2D()
        inputs = tab.get_case_inputs()
        self.assertEqual(inputs.material_name, "TNT")
        self.assertEqual(inputs.undefined_keys, ())
        self.assertIsNotNone(inputs.rho_charge)
        result = validate_required_values(inputs)
        self.assertTrue(result.ok, result.errors)

    def test_required_custom_keys_match_catalog_expectations(self):
        self.assertEqual(
            set(REQUIRED_CUSTOM_MATERIAL_KEYS),
            {"rho", "energy", "A", "B", "R1", "R2", "omega"},
        )


if __name__ == "__main__":
    unittest.main()
