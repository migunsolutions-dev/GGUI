"""Focused tests: BF import → editable GGUI model, VIPER mass, generate path."""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication, QMessageBox

from axisymmetric_2d import BOUNDARY_SLIP, DYNAMIC_MESH, validate_case_inputs_2d
from case_loader_2d import inspect_imported_axisymmetric_case
from external_case_workflow_2d import inventory_case
from generator_2d import Generator2D
from imported_case_mapping_2d import FieldProvenance, full_sphere_mass_kg, map_imported_case_to_gui
from models_2d import CaseInputs2D, ProbePoint2D
from physical_charge_geometry import physical_charge_geometry
from test_external_working_copy_2d import _make_unmeshed_source
from main_new import BlastFoamApp

app = QApplication.instance() or QApplication([])
REPO = Path(__file__).resolve().parent


class ViperMassConventionTests(unittest.TestCase):
    def test_radius_from_mass_200kg_1600(self):
        geom = physical_charge_geometry(
            replace(CaseInputs2D(), mass_kg=200.0, rho_charge=1600.0, charge_shape="Sphere")
        )
        self.assertAlmostEqual(geom.radius_m, 0.310175245, places=8)

    def test_radius_from_mass_50kg_1600(self):
        geom = physical_charge_geometry(
            replace(CaseInputs2D(), mass_kg=50.0, rho_charge=1600.0, charge_shape="Sphere")
        )
        self.assertAlmostEqual(geom.radius_m, 0.195398160, places=8)

    def test_mass_from_radius_025_1601(self):
        mass = full_sphere_mass_kg(0.25, 1601.0)
        self.assertAlmostEqual(mass, 104.785, places=2)

    def test_hob_zero_does_not_halve_mass_or_radius(self):
        inputs = replace(
            CaseInputs2D(),
            mass_kg=104.785,
            rho_charge=1601.0,
            height_of_burst=0.0,
            bottom_boundary=BOUNDARY_SLIP,
            radius=20.0,
            height=20.0,
            cell_size=1.0,
            mesh_mode=DYNAMIC_MESH,
            charge_seed_mode="Manual",
            charge_refinement_level=5,
        )
        geom = physical_charge_geometry(inputs)
        self.assertAlmostEqual(geom.radius_m, 0.25, places=3)
        # Half-mass would yield ~0.198 m — must not match that.
        half_mass_r = (3.0 * (104.785 / 2) / (4.0 * math.pi * 1601.0)) ** (1.0 / 3.0)
        self.assertGreater(abs(geom.radius_m - half_mass_r), 0.04)
        result = validate_case_inputs_2d(inputs)
        self.assertTrue(result.valid, result.errors)


class ImportEditableModelTests(unittest.TestCase):
    def test_axisymmetric_charge_mapping_viper_mass_and_editable(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            mapping = map_imported_case_to_gui(str(source))
            self.assertEqual(mapping.gui_values.get("radius"), 20.0)
            self.assertEqual(mapping.gui_values.get("height"), 20.0)
            self.assertEqual(mapping.gui_values.get("material_name"), "C4")
            self.assertEqual(mapping.gui_values.get("height_of_burst"), 0.0)
            self.assertEqual(mapping.gui_values.get("dyn_refine_max"), 4)
            self.assertEqual(mapping.gui_values.get("charge_refinement_level"), 5)
            self.assertEqual(mapping.gui_values.get("buffer_layers"), 5)
            self.assertAlmostEqual(mapping.gui_values["mass_kg"], 104.785, places=2)
            self.assertEqual(mapping.fields["mass_kg"].provenance, FieldProvenance.DERIVED)
            self.assertTrue(mapping.fields["mass_kg"].editable)
            self.assertIn("mass_kg", mapping.editable_keys)
            self.assertEqual(mapping.gui_values.get("bottom_boundary"), "Reflecting slip wall")
            # Notes may mention the BF source remaining read-only; controls are not.
            self.assertFalse(
                any("controls" in n.lower() and "read-only" in n.lower() for n in mapping.notes)
            )

    @unittest.skipUnless(
        Path(
            r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\blastfoam"
            r"\tutorials\blastFoam\axisymmetricCharge\system"
        ).is_dir(),
        "official axisymmetricCharge tutorial unavailable",
    )
    def test_official_tutorial_ground_is_reflecting_from_slip(self):
        source = (
            r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\blastfoam"
            r"\tutorials\blastFoam\axisymmetricCharge"
        )
        mapping = map_imported_case_to_gui(source)
        self.assertEqual(mapping.gui_values.get("bottom_boundary"), "Reflecting slip wall")
        self.assertAlmostEqual(mapping.gui_values["mass_kg"], 104.785, places=2)
        self.assertEqual(mapping.gui_values.get("dyn_refine_max"), 4)

    def test_generated_setfields_uses_full_sphere_mass(self):
        with tempfile.TemporaryDirectory() as td:
            mass = full_sphere_mass_kg(0.25, 1601.0)
            inputs = replace(
                CaseInputs2D(),
                mass_kg=mass,
                rho_charge=1601.0,
                material_name="C4",
                height_of_burst=0.0,
                detonation_height=0.0,
                bottom_boundary=BOUNDARY_SLIP,
                radius=20.0,
                height=20.0,
                cell_size=1.0,
                mesh_mode=DYNAMIC_MESH,
                charge_seed_mode="Manual",
                charge_refinement_level=5,
                buffer_layers=5,
                dyn_refine_max=4,
                refine_interval=1,
                unrefine_interval=1,
                lower_refine_threshold=0.05,
                unrefine_threshold=0.05,
                n_buffer_layers_dynamic=1,
            )
            case = Generator2D(td).generate("ground_burst", inputs)
            fields = Path(case, "system", "setFieldsDict").read_text(encoding="utf-8")
            control = Path(case, "system", "controlDict").read_text(encoding="utf-8")
            dynamic = Path(case, "constant", "dynamicMeshDict").read_text(encoding="utf-8")
            self.assertIn("centre (0 0 0)", fields)
            self.assertIn("sphericalMassToCell", fields)
            self.assertRegex(fields, r"mass\s+104\.785")
            self.assertRegex(fields, r"rho\s+1601")
            # VIPER: mass is full-sphere; half-mass ~52.39 must not appear.
            self.assertNotRegex(fields, r"mass\s+52\.3")
            self.assertNotIn("type refineProbes;", control)
            self.assertIn("refineProbes", dynamic)
            self.assertIn("maxRefinement 4", dynamic)


class GeneratedCaseRoundTripTests(unittest.TestCase):
    def test_reference_sphere_generated_case_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            source_before = inventory_case(str(source))
            source_mapping = map_imported_case_to_gui(str(source))
            values = source_mapping.gui_values
            inputs = replace(
                CaseInputs2D(),
                radius=values["radius"],
                height=values["height"],
                cell_size=values["cell_size"],
                mass_kg=values["mass_kg"],
                rho_charge=values["rho_charge"],
                material_name=values["material_name"],
                charge_shape=values["charge_shape"],
                height_of_burst=values["height_of_burst"],
                detonation_height=values["detonation_height"],
                bottom_boundary=values["bottom_boundary"],
                outer_boundary=values["outer_boundary"],
                top_boundary=values["top_boundary"],
                mesh_mode=values["mesh_mode"],
                charge_seed_mode=values["charge_seed_mode"],
                charge_refinement_level=values["charge_refinement_level"],
                buffer_layers=values["buffer_layers"],
                dyn_refine_max=values["dyn_refine_max"],
                refine_interval=values["refine_interval"],
                lower_refine_threshold=values["lower_refine_threshold"],
                unrefine_threshold=values["unrefine_threshold"],
                n_buffer_layers_dynamic=values["n_buffer_layers_dynamic"],
            )
            generated = Generator2D(td).generate("round_trip_sphere", inputs)
            reopened = map_imported_case_to_gui(generated)

            for key, expected, tolerance in (
                ("radius", 20.0, 1e-8),
                ("height", 20.0, 1e-10),
                ("cell_size", 1.0, 1e-8),
                ("mass_kg", inputs.mass_kg, 1e-8),
                ("rho_charge", 1601.0, 1e-10),
                ("height_of_burst", 0.0, 1e-12),
                ("detonation_height", 0.0, 1e-12),
            ):
                self.assertAlmostEqual(
                    reopened.gui_values[key], expected, delta=tolerance, msg=key
                )
            self.assertEqual(reopened.gui_values["charge_shape"], "Sphere")
            self.assertEqual(reopened.gui_values["material_name"], "C4")
            self.assertEqual(reopened.gui_values["charge_refinement_level"], 5)
            self.assertEqual(reopened.gui_values["buffer_layers"], 5)
            self.assertEqual(reopened.gui_values["dyn_refine_max"], 4)
            self.assertEqual(reopened.gui_values["bottom_boundary"], BOUNDARY_SLIP)
            for key in (
                "charge_shape",
                "mass_kg",
                "rho_charge",
                "height_of_burst",
                "cell_size",
            ):
                self.assertNotEqual(
                    reopened.fields[key].provenance,
                    FieldProvenance.NOT_RECOVERED,
                    key,
                )
            self.assertEqual(source_before, inventory_case(str(source)))

    def test_generated_mass_based_cylinder_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            inputs = replace(
                CaseInputs2D(),
                radius=12.0,
                height=15.0,
                cell_size=0.5,
                mass_kg=50.0,
                rho_charge=1600.0,
                material_name="C4",
                charge_shape="Cylinder",
                charge_aspect=2.5,
                height_of_burst=4.0,
                detonation_height=4.0,
                mesh_mode=DYNAMIC_MESH,
                charge_seed_mode="Manual",
                charge_refinement_level=4,
                buffer_layers=3,
                dyn_refine_max=3,
                cycle_write=4,
                cores=3,
                probes=(ProbePoint2D("near", 0.5, 4.0),),
                output_fields=("p", "alpha.c4"),
            )
            generated = Generator2D(td).generate("round_trip_cylinder", inputs)
            set_fields = Path(generated, "system", "setFieldsDict").read_text(
                encoding="utf-8"
            )
            self.assertIn("cylindericalMassToCell", set_fields)
            reopened = map_imported_case_to_gui(generated)
            self.assertEqual(reopened.gui_values["charge_shape"], "Cylinder")
            self.assertAlmostEqual(reopened.gui_values["mass_kg"], 50.0)
            self.assertAlmostEqual(reopened.gui_values["rho_charge"], 1600.0)
            self.assertAlmostEqual(reopened.gui_values["charge_aspect"], 2.5)
            self.assertAlmostEqual(reopened.gui_values["height_of_burst"], 4.0)
            self.assertAlmostEqual(reopened.gui_values["detonation_height"], 4.0)
            self.assertAlmostEqual(reopened.gui_values["cell_size"], 0.5, delta=1e-8)
            self.assertEqual(reopened.gui_values["cycle_write"], 4)
            self.assertEqual(reopened.gui_values["cores"], 3)
            self.assertEqual(reopened.gui_values["output_fields"], ("p", "alpha.c4"))
            self.assertEqual(
                reopened.gui_values["probes"],
                [{"name": "P1", "radius": 0.5, "height": 4.0}],
            )
            self.assertNotIn("mass_kg", reopened.not_recovered_keys)

    def test_generated_fixed_mesh_round_trip_stays_fixed(self):
        with tempfile.TemporaryDirectory() as td:
            inputs = replace(
                CaseInputs2D(),
                radius=2.0,
                height=3.0,
                cell_size=0.01,
                mass_kg=2.0,
                rho_charge=1600.0,
                material_name="C4",
                charge_shape="Sphere",
                height_of_burst=1.0,
                detonation_height=1.0,
                mesh_mode="Fixed Mesh",
            )
            generated = Generator2D(td).generate("round_trip_fixed", inputs)
            reopened = map_imported_case_to_gui(generated)
            self.assertEqual(reopened.gui_values["mesh_mode"], "Fixed Mesh")
            self.assertEqual(reopened.gui_values["charge_seed_mode"], "Off")
            self.assertAlmostEqual(reopened.gui_values["cell_size"], 0.01, delta=1e-8)
            self.assertAlmostEqual(reopened.gui_values["mass_kg"], 2.0, delta=1e-6)

    def test_same_session_runtime_attach_preserves_validated_controls(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            state = inspect_imported_axisymmetric_case(str(source))
            tab = self._make_tab()
            try:
                tab.spin_mass.setValue(50.0)
                tab.spin_density.setValue(1600.0)
                tab.load_imported_case(state, apply_mapping=False)
                self.assertAlmostEqual(tab.spin_mass.value(), 50.0)
                self.assertAlmostEqual(tab.spin_density.value(), 1600.0)
            finally:
                tab.close()

    @staticmethod
    def _make_tab():
        from tab_2d import Tab2D

        return Tab2D()


class ImportUiEditableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._patches = [
            mock.patch.object(QMessageBox, "information", return_value=QMessageBox.Ok),
            mock.patch.object(QMessageBox, "warning", return_value=QMessageBox.Ok),
            mock.patch.object(QMessageBox, "critical", return_value=QMessageBox.Ok),
            mock.patch.object(BlastFoamApp, "_show_load_summary_dialog_2d", autospec=True),
            mock.patch.object(BlastFoamApp, "_show_load_summary_dialog", autospec=True),
        ]
        for p in cls._patches:
            p.start()
        cls.win = BlastFoamApp()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.win.close()
        except Exception:
            pass
        for p in cls._patches:
            p.stop()

    def test_import_enables_controls_and_builds_preview(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            case_root = Path(td) / "Work"
            case_root.mkdir()
            with mock.patch.object(self.win, "base_projects_path", str(case_root)):
                self.win.open_openfoam_case_path(str(source))
            tab = self.win.tab_2d
            self.assertTrue(tab.is_imported_mode)
            self.assertTrue(tab.spin_mass.isEnabled())
            self.assertTrue(tab.spin_radius.isEnabled())
            self.assertTrue(tab.spin_refine_interval.isEnabled())
            self.assertGreater(tab.spin_mass.value(), 100.0)
            self.assertAlmostEqual(tab.spin_mass.value(), 104.785, places=1)
            # Setup preview must exist without polyMesh / without red mesh-missing clear.
            self.assertIsNotNone(tab.viewer._last_preview_data)
            self.assertFalse(tab.viewer.is_simulating)
            self.assertFalse(tab.cmb_field.isEnabled())
            # Edit updates preview radius via VIPER mass→radius.
            tab.spin_mass.setValue(50.0)
            tab.spin_density.setValue(1600.0)
            tab._refresh_derived()
            self.assertIn("0.195", tab.lbl_charge_r.text())


class ImportInitUsesGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._patches = [
            mock.patch.object(QMessageBox, "information", return_value=QMessageBox.Ok),
            mock.patch.object(QMessageBox, "warning", return_value=QMessageBox.Ok),
            mock.patch.object(QMessageBox, "critical", return_value=QMessageBox.Ok),
            mock.patch.object(BlastFoamApp, "_show_load_summary_dialog_2d", autospec=True),
            mock.patch.object(BlastFoamApp, "_show_load_summary_dialog", autospec=True),
        ]
        for p in cls._patches:
            p.start()
        cls.win = BlastFoamApp()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.win.close()
        except Exception:
            pass
        for p in cls._patches:
            p.stop()

    def test_initialize_imported_calls_generator_not_prepare(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            case_root = Path(td) / "Work"
            case_root.mkdir()
            with mock.patch.object(self.win, "base_projects_path", str(case_root)):
                self.win.open_openfoam_case_path(str(source))
            generated = []

            def fake_generate(name, inputs):
                out = Path(td) / "generated" / name
                Generator2D(str(out.parent)).generate(name, inputs)
                generated.append(str(out))
                return str(out)

            def fake_run(case_dir, command):
                Path(case_dir, "log.checkMesh").write_text(
                    "Mesh OK\n", encoding="utf-8"
                )
                Path(case_dir, "log.initialize").write_text(
                    f"Command: {command}\nMesh OK\n", encoding="utf-8"
                )
                mesh = Path(case_dir, "constant", "polyMesh")
                mesh.mkdir(parents=True, exist_ok=True)
                (mesh / "owner").write_text(
                    'FoamFile { note "nCells: 400"; }\n4\n(0\n1\n2\n3\n)\n',
                    encoding="utf-8",
                )
                return True

            with mock.patch.object(
                self.win.service, "generate_case", side_effect=fake_generate
            ) as gen, mock.patch.object(
                self.win, "_run_wsl_commands", side_effect=fake_run
            ) as run, mock.patch(
                "main_new.prepare_working_copy"
            ) as prepare, mock.patch.object(
                self.win.tab_2d.viewer, "load_case", autospec=True
            ), mock.patch.object(
                self.win.tab_2d.viewer, "set_field", autospec=True
            ):
                self.win._force_sync_prep = True
                self.win.on_initialize_imported_model_2d()
                gen.assert_called_once()
                run.assert_called_once()
                prepare.assert_not_called()
            self.assertTrue(generated)
            control = Path(generated[0], "system", "controlDict").read_text(encoding="utf-8")
            self.assertNotIn("type refineProbes;", control)


if __name__ == "__main__":
    unittest.main()
