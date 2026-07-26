from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from dataclasses import asdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

import tab_3d_general
from generator_3d import Generator3D
from models import CaseInputs3D, ObstacleData
from probes_model import ProbesModel
from project_io import (
    apply_project_payload,
    capture_project_payload,
    read_project,
    write_project_atomic,
)
from simulation_service import SimulationService
from startup_capture_guard import UNSAFE_CAPTURE_MESSAGE
from viewer_widget import ObstacleItem


class DummyViewer(QWidget):
    cell_count_updated = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.current_case_dir = None
        self.show_mesh_lines = False
        self.show_boundaries = False
        self.show_obstacles = False
        self.show_obstacles_wireframe_only = False
        self.show_tracers = False

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class Tab3DGuiCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.real_viewer = tab_3d_general.BlastViewerWidget
        tab_3d_general.BlastViewerWidget = DummyViewer

    @classmethod
    def tearDownClass(cls):
        tab_3d_general.BlastViewerWidget = cls.real_viewer

    def make_tab(self):
        return tab_3d_general.TabGeneral3D(ProbesModel())

    def test_fresh_display_matches_generated_amr_max(self):
        tab = self.make_tab()
        self.assertEqual(tab.spin_refine_max.value(), 1)
        inputs = tab.get_case_inputs()
        self.assertEqual(inputs.dyn_refine_max, 1)
        # Force Dyn Mesh only to make dynamicMeshDict active; do not alter max.
        tab.rad_dyn_mesh.setChecked(True)
        inputs = tab.get_case_inputs()
        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("gui", inputs)
            with open(
                os.path.join(case_dir, "constant", "dynamicMeshDict"),
                encoding="utf-8",
            ) as stream:
                text = stream.read()
        self.assertRegex(text, r"maxRefinement\s+1\s*;")

    def test_edit_and_load_keep_one_runtime_value(self):
        tab = self.make_tab()
        tab.spin_refine_max.setValue(4)
        self.assertEqual(tab._dyn_refine_max, 4)
        self.assertEqual(tab.get_case_inputs().dyn_refine_max, 4)
        tab.set_case_inputs({"dyn_refine_max": 2, "enable_dyn_refine": True})
        self.assertEqual(tab.spin_refine_max.value(), 2)
        self.assertEqual(tab._dyn_refine_max, 2)
        self.assertEqual(tab.get_case_inputs().dyn_refine_max, 2)

    def test_seed_and_outer_band_defaults_remain_protected(self):
        tab = self.make_tab()
        self.assertEqual(tab.combo_charge_seed_mode.currentText(), "Auto")
        self.assertEqual(tab.spin_charge_refine.value(), 0)
        self.assertFalse(tab.chk_charge_outer_enable.isChecked())
        self.assertEqual(tab.spin_charge_outer_level.value(), 3)
        # Legacy min/max spins mirror the single outer level.
        self.assertEqual(tab.spin_charge_outer_min.value(), 3)
        self.assertEqual(tab.spin_charge_outer_max.value(), 3)

    def test_service_and_generator_share_unsafe_capture_guard(self):
        """Direct service/generator entry points must block unsafe seed-0/no-band cases."""
        unsafe = CaseInputs3D(
            min_point=(0.0, 0.0, 0.0),
            max_point=(1.0, 1.0, 1.0),
            cell_size=0.2,
            charge_center=(0.4, 0.4, 0.4),
            charge_shape="Sphere",
            mass_kg=0.001,
            cylinder_radius=0.05,
            cylinder_axis="Z",
            material_name="C4",
            rho_charge=1601.0,
            energy_j_per_kg=4.5e6,
            p_atm=101325.0,
            t_atm=288.0,
            end_time_s=1e-3,
            delta_t=1e-7,
            write_interval_steps=10,
            cores=1,
            charge_seed_mode="Off",
            charge_refinement_level=0,
            charge_outer_refine_enable=False,
        )
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                Generator3D(td).generate("unsafe_gui", unsafe)
            self.assertEqual(str(ctx.exception), UNSAFE_CAPTURE_MESSAGE)
            service = SimulationService(base_projects_path=td, openfoam_bashrc="/opt/openfoam9/etc/bashrc")
            with self.assertRaises(ValueError) as ctx2:
                service.generate_case("unsafe_svc", unsafe)
            self.assertEqual(str(ctx2.exception), UNSAFE_CAPTURE_MESSAGE)

    def test_optional_amr_none_does_not_coerce_user_choice(self):
        tab = self.make_tab()
        tab.rad_dyn_mesh.setChecked(True)
        tab.spin_refine_max.setValue(3)
        tab.set_case_inputs(
            {
                "dyn_refine_min": None,
                "dyn_refine_max": None,
                "enable_dyn_refine": None,
            }
        )
        self.assertTrue(tab.rad_dyn_mesh.isChecked())
        self.assertEqual(tab.spin_refine_max.value(), 3)

    def test_project_gui_round_trip_preserves_custom_and_obstacles(self):
        """CaseInputs3D → GUI → save → open → GUI → get_case_inputs → regenerate."""
        custom_props = {
            "rho": 1555.0,
            "energy": 5.12e6,
            "A": 310.0e9,
            "B": 3.5e9,
            "R1": 4.2,
            "R2": 1.1,
            "omega": 0.28,
        }
        with tempfile.TemporaryDirectory() as td:
            stl_enabled = os.path.join(td, "enabled-wall.stl")
            stl_disabled = os.path.join(td, "disabled-wall.stl")
            for path in (stl_enabled, stl_disabled):
                with open(path, "w", encoding="ascii") as f:
                    f.write(
                        "solid wall\nfacet normal 0 0 1\nouter loop\n"
                        "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
                        "endloop\nendfacet\nendsolid wall\n"
                    )

            original = CaseInputs3D(
                min_point=(0.0, 0.0, 0.0),
                max_point=(2.0, 2.0, 2.0),
                cell_size=0.25,
                charge_center=(1.0, 1.0, 1.0),
                charge_shape="Sphere",
                mass_kg=1.0,
                cylinder_radius=0.08,
                cylinder_axis="Z",
                material_name="Custom",
                rho_charge=1555.0,
                energy_j_per_kg=custom_props["energy"],
                p_atm=101325.0,
                t_atm=288.0,
                end_time_s=1e-3,
                delta_t=1e-7,
                write_interval_steps=10,
                cores=1,
                material_props={**custom_props, "E0": custom_props["energy"]},
                enable_dyn_refine=True,
                dyn_refine_max=4,
                enable_obstacle_refine=True,
                charge_refinement_level=2,
                charge_outer_refine_enable=False,
                charge_outer_refine_min=0,
                charge_outer_refine_max=0,
                obstacles=[
                    ObstacleData(stl_enabled, "enabled-wall.stl", 0.002, 0.1, 0.2, 0.3, 2),
                ],
            )

            probes = ProbesModel()
            tab = tab_3d_general.TabGeneral3D(probes)
            # Stale OpenFOAM provenance must not survive a subsequent project apply.
            tab._provenance["enable_dyn_refine"] = "UNSET"
            tab._provenance["enable_obstacle_refine"] = "UNSET"

            data = asdict(original)
            data["charge_radius"] = original.cylinder_radius
            tab.set_case_inputs(data)
            tab.obstacles = [
                ObstacleItem(True, stl_enabled, 0.002, 0.1, 0.2, 0.3),
                ObstacleItem(False, stl_disabled, 0.003, 1.0, 2.0, 3.0),
            ]
            tab._refresh_table()
            probes.add_probe("P1", 0.5, 0.6, 0.7)
            tab.load_project_gui_state(
                {
                    "sections": [
                        {
                            "enabled": True,
                            "name": "cutA",
                            "normal": [0, 0, 1],
                            "position_m": 0.25,
                            "opacity": 0.4,
                        }
                    ]
                }
            )

            before_inputs = tab.get_case_inputs()
            self.assertEqual(before_inputs.material_name, "Custom")
            for key in ("rho", "energy", "A", "B", "R1", "R2", "omega"):
                self.assertEqual(before_inputs.material_props[key], custom_props[key])
            self.assertTrue(before_inputs.enable_dyn_refine)
            self.assertEqual(before_inputs.dyn_refine_max, 4)
            self.assertTrue(before_inputs.enable_obstacle_refine)

            payload = capture_project_payload(tab, probes)
            project_path = os.path.join(td, "roundtrip.ggui.json")
            write_project_atomic(project_path, payload)

            case_before = Generator3D(td).generate("before", before_inputs)

            probes2 = ProbesModel()
            tab2 = tab_3d_general.TabGeneral3D(probes2)
            # Simulate previously opened OpenFOAM case leaving UNSET provenance.
            tab2._provenance["enable_dyn_refine"] = "UNSET"
            tab2._provenance["enable_obstacle_refine"] = "UNSET"
            tab2.materials_db["Custom"] = {
                "rho": 1600,
                "energy": 4.50e6,
                "A": 300.0e9,
                "B": 3.0e9,
                "R1": 4.0,
                "R2": 1.0,
                "omega": 0.30,
            }

            loaded = read_project(project_path)
            apply_project_payload(tab2, probes2, loaded)
            after_inputs = tab2.get_case_inputs()

            self.assertEqual(after_inputs.material_name, "Custom")
            for key in ("rho", "energy", "A", "B", "R1", "R2", "omega"):
                self.assertEqual(after_inputs.material_props[key], custom_props[key])
            self.assertTrue(after_inputs.enable_dyn_refine)
            self.assertEqual(after_inputs.dyn_refine_max, 4)
            self.assertTrue(after_inputs.enable_obstacle_refine)
            self.assertNotEqual(tab2._provenance.get("enable_dyn_refine"), "UNSET")

            self.assertEqual(len(tab2.obstacles), 2)
            enabled = next(o for o in tab2.obstacles if o.enabled)
            disabled = next(o for o in tab2.obstacles if not o.enabled)
            self.assertEqual(enabled.path, stl_enabled)
            self.assertEqual(enabled.scale, 0.002)
            self.assertEqual((enabled.ox, enabled.oy, enabled.oz), (0.1, 0.2, 0.3))
            self.assertEqual(disabled.path, stl_disabled)
            self.assertEqual(disabled.scale, 0.003)
            self.assertEqual((disabled.ox, disabled.oy, disabled.oz), (1.0, 2.0, 3.0))

            probe_dicts = probes2.to_dict()["probes"]
            self.assertEqual(len(probe_dicts), 1)
            self.assertEqual(probe_dicts[0]["name"], "P1")
            self.assertEqual((probe_dicts[0]["x"], probe_dicts[0]["y"], probe_dicts[0]["z"]), (0.5, 0.6, 0.7))

            self.assertEqual(len(tab2.sections), 1)
            self.assertEqual(tab2.sections[0].name, "cutA")
            self.assertAlmostEqual(tab2.sections[0].position_m, 0.25)

            case_after = Generator3D(td).generate("after", after_inputs)
            for rel in (
                "constant/dynamicMeshDict",
                "system/controlDict",
                "system/setFieldsDict",
                "system/snappyHexMeshDict",
                "constant/phaseProperties",
            ):
                with open(os.path.join(case_before, rel), encoding="utf-8") as f:
                    a = f.read()
                with open(os.path.join(case_after, rel), encoding="utf-8") as f:
                    b = f.read()
                self.assertEqual(a, b, rel)

    def _visible_summary_texts(self, tab) -> list[str]:
        texts = []
        for lbl in tab.findChildren(QLabel):
            if not lbl.isVisible():
                continue
            t = (lbl.text() or "").strip()
            if t:
                texts.append(t)
        return texts

    def test_mesh_plan_populated_before_initialize(self):
        tab = self.make_tab()
        tab.sx1.setValue(-5)
        tab.sx2.setValue(5)
        tab.sy1.setValue(-5)
        tab.sy2.setValue(5)
        tab.sz1.setValue(0)
        tab.sz2.setValue(5)
        tab.scell.setValue(0.5)
        tab.rad_dyn_mesh.setChecked(True)
        tab.spin_refine_max.setValue(2)
        tab.combo_charge_seed_mode.setCurrentText("Manual")
        tab.spin_charge_refine.setValue(4)
        tab.chk_charge_outer_enable.setChecked(True)
        tab.spin_charge_outer_level.setValue(3)
        tab._update_mesh_plan_display()

        plan = tab.lbl_plan_block_mesh.text()
        self.assertIn("Total cells before refinement:    4,000", plan)
        self.assertIn("Base grid:    20 × 20 × 10", plan)
        self.assertNotIn("h0", plan)
        self.assertNotIn("→", plan)
        self.assertIn("Mesh mode:    AMR", plan)
        self.assertIn("Wave AMR level:    2", plan)
        self.assertIn("Planned initialization:    setRefinedFields", plan)
        self.assertIn("Charge seed mode:    Manual", plan)
        self.assertIn("Charge seed level:    4", plan)
        self.assertIn("Charge seed status:", plan)
        self.assertIn("Startup outer region:    On", plan)
        self.assertIn("Startup outer level:    3", plan)
        # First data row is total cells; base grid is second (no duplicate cell-count row).
        lines = [ln for ln in plan.splitlines() if ln.strip()]
        self.assertTrue(lines[0].startswith("Total cells before refinement:"))
        self.assertTrue(lines[1].startswith("Base grid:"))
        self.assertEqual(sum(1 for ln in lines if "Total cells" in ln), 1)
        self.assertFalse(tab.grp_init_results.isVisible())
        # Horizontal scrolling must remain available (not disabled).
        from PyQt5.QtCore import Qt as _Qt
        self.assertNotEqual(
            tab._left_setup_scroll.horizontalScrollBarPolicy(),
            _Qt.ScrollBarAlwaysOff,
        )
        # Title is a separate QLabel with layout gap (not QGroupBox title / newlines).
        self.assertEqual(tab.lbl_mesh_plan_title.text(), "Mesh Plan — Before Initialize")
        self.assertIsInstance(tab.lbl_mesh_plan_title, QLabel)
        self.assertNotIn("Mesh Plan", plan)

    def test_mesh_plan_fixed_and_remap_seed_not_applied(self):
        tab = self.make_tab()
        tab.sx1.setValue(-5)
        tab.sx2.setValue(5)
        tab.sy1.setValue(-5)
        tab.sy2.setValue(5)
        tab.sz1.setValue(0)
        tab.sz2.setValue(5)
        tab.scell.setValue(0.5)
        tab.combo_charge_seed_mode.setCurrentText("Auto")
        tab.spin_charge_seed_target.setValue(8)
        # Fixed Mesh
        tab.rad_fixed_mesh.setChecked(True)
        tab._update_charge_seed_controls_enabled()
        tab._update_mesh_plan_display()
        seed_txt = tab.lbl_plan_charge_seed.text()
        self.assertIn("Not applied — Fixed Mesh", seed_txt)
        self.assertNotIn("Charge seed level:    5", seed_txt)
        self.assertFalse(tab.spin_charge_seed_target.isEnabled())
        # Remap
        tab.rad_dyn_mesh.setChecked(True)
        tab.rad_init_remap.setChecked(True)
        tab._update_mesh_plan_display()
        seed_txt = tab.lbl_plan_charge_seed.text()
        self.assertIn("Not applied — Remap", seed_txt)

    def test_seed_target_enabled_only_in_auto(self):
        tab = self.make_tab()
        tab.rad_dyn_mesh.setChecked(True)
        tab.combo_charge_seed_mode.setCurrentText("Auto")
        tab._update_charge_seed_controls_enabled()
        self.assertTrue(tab.spin_charge_seed_target.isEnabled())
        self.assertFalse(tab.spin_charge_refine.isEnabled())
        tab.combo_charge_seed_mode.setCurrentText("Manual")
        tab._update_charge_seed_controls_enabled()
        self.assertFalse(tab.spin_charge_seed_target.isEnabled())
        self.assertTrue(tab.spin_charge_refine.isEnabled())
        tab.combo_charge_seed_mode.setCurrentText("Off")
        tab._update_charge_seed_controls_enabled()
        self.assertFalse(tab.spin_charge_seed_target.isEnabled())
        self.assertFalse(tab.spin_charge_refine.isEnabled())

    def test_cylinder_mass_rho_ld_widgets_drive_same_geometry(self):
        """Offscreen Qt: mass/density/L/D widgets → display, collect, seed, setFieldsDict."""
        import math

        from charge_seed_plan import build_charge_seed_plan
        from physical_charge_geometry import physical_charge_geometry

        tab = self.make_tab()
        tab.c_shape.setCurrentText("Cylinder")
        tab._on_shape_changed("Cylinder")
        tab.c_mass.setValue(25.0)
        tab.c_rho.setValue(1601.0)
        tab.c_aspect.setValue(2.5)
        tab._update_cylinder_derived_geometry()
        self.app.processEvents()

        self.assertTrue(tab.c_radius.isReadOnly())
        self.assertTrue(tab.c_length.isReadOnly())
        vol = 25.0 / 1601.0
        r_exp = (vol / (2.0 * math.pi * 2.5)) ** (1.0 / 3.0)
        L_exp = 2.0 * r_exp * 2.5
        # SpinBoxes use 4 decimal places — compare at display precision.
        self.assertAlmostEqual(tab.c_radius.value(), r_exp, places=4)
        self.assertAlmostEqual(tab.c_length.value(), L_exp, places=4)

        inputs = tab.get_case_inputs()
        geom = physical_charge_geometry(inputs)
        self.assertAlmostEqual(inputs.cylinder_radius, r_exp, places=6)
        self.assertAlmostEqual(inputs.charge_length, L_exp, places=6)
        self.assertAlmostEqual(geom.cylinder_radius_m, r_exp, places=6)
        plan = build_charge_seed_plan(inputs)
        self.assertAlmostEqual(plan.d_min_m, geom.d_min_m, places=9)

        with tempfile.TemporaryDirectory() as td:
            case_dir = Generator3D(td).generate("cyl_qt", inputs)
            with open(os.path.join(case_dir, "system", "setFieldsDict"), encoding="utf-8") as fh:
                sf = fh.read()
            self.assertIn("cylindericalMassToCell", sf)
            self.assertIn("mass 25", sf)
            self.assertIn("rho 1601", sf)
            self.assertRegex(sf, r"LbyD\s+2\.5")

    def test_seed_policy_hidden_fields_survive_project_round_trip(self):
        """Dialog-free: non-default min/max seed policy preserved Save→Open→collect."""
        from project_io import (
            apply_project_payload,
            build_project,
            capture_project_payload,
            read_project,
            write_project_atomic,
        )

        tab = self.make_tab()
        tab.rad_dyn_mesh.setChecked(True)
        tab.combo_charge_seed_mode.setCurrentText("Auto")
        tab.spin_charge_seed_target.setValue(7)
        tab._charge_seed_min_cells = 4
        tab._charge_seed_max_level = 3
        tab._charge_outer_legacy_migration_warning = "legacy bake test warning"
        collected = tab.get_case_inputs()
        self.assertEqual(collected.charge_seed_target_cells, 7)
        self.assertEqual(collected.charge_seed_min_cells, 4)
        self.assertEqual(collected.charge_seed_max_level, 3)
        self.assertEqual(
            collected.charge_outer_legacy_migration_warning, "legacy bake test warning"
        )

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "seed_policy.ggui.json")
            payload = build_project(
                collected, probes={"probes": []}, gui_state={}
            )
            write_project_atomic(path, payload)
            loaded = read_project(path)
            tab2 = self.make_tab()
            apply_project_payload(tab2, ProbesModel(), loaded)
            again = tab2.get_case_inputs()
            self.assertEqual(again.charge_seed_mode, "Auto")
            self.assertEqual(again.charge_seed_target_cells, 7)
            self.assertEqual(again.charge_seed_min_cells, 4)
            self.assertEqual(again.charge_seed_max_level, 3)
            self.assertEqual(
                again.charge_outer_legacy_migration_warning, "legacy bake test warning"
            )
            # Second collect after save/open still matches
            payload2 = capture_project_payload(tab2, ProbesModel())
            self.assertEqual(payload2["case_inputs"]["charge_seed_min_cells"], 4)
            self.assertEqual(payload2["case_inputs"]["charge_seed_max_level"], 3)
    def test_initialization_results_hidden_until_metadata(self):
        tab = self.make_tab()
        self.assertTrue(tab.grp_init_results.isHidden())
        self.assertFalse(tab._init_results_available)
        for lbl in (
            tab.lbl_result_total_cells,
            tab.lbl_result_init_command,
            tab.lbl_result_charge_cells,
            tab.lbl_result_ignition_cells,
        ):
            self.assertFalse(bool((lbl.text() or "").strip()) and not lbl.isHidden())

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "case_init_mode.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "set_cmd": "setFields",
                        "base_cell_count": 12345,
                        "cells_inside_charge": 88,
                        "cells_in_ignition_region": 12,
                        "charge_clipped_by_domain": False,
                    },
                    f,
                )
            tab.update_charge_cells_display(td)

        # Tall Initialization Results group stays hidden; compact Info panel shows actuals.
        self.assertTrue(tab.grp_init_results.isHidden())
        self.assertTrue(tab._init_results_available)
        self.assertIn("12345", tab.lbl_result_total_cells.text().replace(",", ""))
        self.assertIn("setFields", tab.lbl_result_init_command.text())
        self.assertIn("88", tab.lbl_result_charge_cells.text())
        self.assertIn("12", tab.lbl_result_ignition_cells.text())
        self.assertIn("12345", tab.lbl_info_total_cells.text().replace(",", ""))
        self.assertIn("88", tab.lbl_info_charge_cells.text())
        self.assertTrue(tab.lbl_info_total_cells.text().startswith("Current cells:"))

    def test_duplicate_summary_rows_removed(self):
        tab = self.make_tab()
        # Permanent summary texts on Mesh Plan / Results (ignore hidden compatibility labels).
        permanent = [
            tab.lbl_plan_base_grid.text(),
            tab.lbl_plan_mesh_mode.text(),
            tab.lbl_plan_init_command.text(),
            tab.lbl_plan_charge_seed.text(),
            tab.lbl_plan_charge_capture.text(),
            tab.lbl_plan_startup_outer.text(),
            tab.lbl_plan_initiation.text(),
            tab.lbl_result_total_cells.text(),
            tab.lbl_result_init_command.text(),
            tab.lbl_result_charge_cells.text(),
            tab.lbl_result_ignition_cells.text(),
        ]
        joined = "\n".join(permanent)
        for banned in (
            "Charge fraction (%)",
            "Cells inside charge (post-refinement)",
            "Obstacle refine:",
            "Expected .eMesh",
            "Charge clipped by domain",
        ):
            self.assertNotIn(banned, joined)

    def test_clipping_shown_only_as_warning_when_true(self):
        tab = self.make_tab()
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "case_init_mode.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "set_cmd": "setFields",
                        "base_cell_count": 100,
                        "charge_clipped_by_domain": False,
                    },
                    f,
                )
            tab.update_charge_cells_display(td)
        self.assertNotIn("clipped", (tab.lbl_charge_resolution_warning.text() or "").lower())
        joined = "\n".join(self._visible_summary_texts(tab))
        self.assertNotIn("Charge clipped by domain", joined)

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "case_init_mode.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "set_cmd": "setFields",
                        "base_cell_count": 100,
                        "charge_clipped_by_domain": True,
                    },
                    f,
                )
            tab.update_charge_cells_display(td)
        self.assertIn("clipped", (tab.lbl_charge_resolution_warning.text() or "").lower())
        self.assertTrue(tab.lbl_charge_resolution_warning.wordWrap())

    def test_exact_buttons_and_no_visible_run_resume(self):
        tab = self.make_tab()
        self.assertEqual(tab.btn_exact_1.text(), "exact 1")
        self.assertEqual(tab.btn_exact_end.text(), "exact END")
        self.assertFalse(tab.btn_exact_1.isHidden())
        self.assertFalse(tab.btn_exact_end.isHidden())
        self.assertFalse(hasattr(tab, "btn_run"))
        self.assertFalse(hasattr(tab, "btn_save_remap"))
        for child in tab.findChildren(QWidget):
            if hasattr(child, "text") and callable(child.text):
                t = child.text()
                if t is None:
                    continue
                self.assertNotIn("Run / Resume", t)
                self.assertNotIn("Save 3D remap", t)

        class Catch(QObject):
            def __init__(self):
                super().__init__()
                self.exact_1 = 0
                self.exact_end = 0

            def on_1(self):
                self.exact_1 += 1

            def on_end(self):
                self.exact_end += 1

        catch = Catch()
        tab.sig_request_run_exact_1.connect(catch.on_1)
        tab.sig_request_run_exact_end.connect(catch.on_end)
        tab.btn_exact_1.click()
        tab.btn_exact_end.click()
        self.assertEqual(catch.exact_1, 1)
        self.assertEqual(catch.exact_end, 1)

    def test_obstacle_refine_and_cycle_write_remain_in_sim_control(self):
        tab = self.make_tab()
        self.assertFalse(tab.chk_obstacle_refine.isHidden())
        self.assertFalse(tab.spin_obstacle_refine_min.isHidden())
        self.assertFalse(tab.spin_obstacle_refine_max.isHidden())
        self.assertFalse(tab.spin_cycle_write.isHidden())
        self.assertEqual(tab.spin_cycle_write.value(), 0)
        # Mesh Properties moved next to Cell Size (Domain/Grid), not Simulation Control.
        self.assertFalse(tab.btn_mesh_properties.isHidden())
        self.assertEqual(tab.btn_mesh_properties.text(), "Mesh Properties…")
        self.assertIn("Wave AMR", " ".join(
            lbl.text() for lbl in tab.findChildren(QLabel) if not lbl.isHidden() and lbl.text()
        ))

    def test_relocated_advanced_mesh_values_survive_project_apply(self):
        tab = self.make_tab()
        tab.rad_dyn_mesh.setChecked(True)
        tab.combo_charge_seed_mode.setCurrentText("Manual")
        tab.spin_charge_refine.setValue(3)
        tab.chk_charge_outer_enable.setChecked(True)
        tab.spin_charge_outer_level.setValue(4)
        tab.spin_transition_cells.setValue(5)
        tab._enable_post_processing = True
        tab._fast_run_mode = False

        with tempfile.TemporaryDirectory() as td:
            probes = ProbesModel()
            payload = capture_project_payload(tab, probes)
            path = os.path.join(td, "adv.ggui.json")
            write_project_atomic(path, payload)

            tab2 = self.make_tab()
            apply_project_payload(tab2, ProbesModel(), read_project(path))
            self.assertEqual(tab2.combo_charge_seed_mode.currentText(), "Manual")
            self.assertEqual(tab2.spin_charge_refine.value(), 3)
            self.assertTrue(tab2.chk_charge_outer_enable.isChecked())
            self.assertEqual(tab2.spin_charge_outer_level.value(), 4)
            self.assertEqual(tab2.spin_charge_outer_max.value(), 4)
            self.assertEqual(tab2.spin_transition_cells.value(), 5)
            self.assertTrue(tab2._enable_post_processing)
            self.assertFalse(tab2._fast_run_mode)
            inputs = tab2.get_case_inputs()
            self.assertEqual(inputs.charge_seed_mode, "Manual")
            self.assertEqual(inputs.charge_refinement_level, 3)
            self.assertEqual(inputs.charge_outer_refine_level, 4)
            self.assertEqual(inputs.transition_cells, 5)

    def test_warning_label_uses_word_wrap(self):
        tab = self.make_tab()
        self.assertTrue(tab.lbl_charge_resolution_warning.wordWrap())
        tab.lbl_charge_resolution_warning.setText(
            "Warning: Charge resolution is too low. Blast may fail. "
            "This long warning must wrap instead of forcing horizontal scroll."
        )
        self.assertTrue(tab.lbl_charge_resolution_warning.wordWrap())
        self.assertGreaterEqual(tab.lbl_charge_resolution_warning.minimumHeight(), 36)

    def test_relocated_charge_seed_spins_do_not_float_visible(self):
        """Regression: advanced seed spins must stay hidden hosts, not cover the tab."""
        tab = self.make_tab()
        tab.resize(1200, 800)
        tab.show()
        self.app.processEvents()
        host = tab._charge_seed_host
        self.assertEqual(host.objectName(), "chargeSeedAdvancedHost")
        self.assertTrue(host.isHidden())
        self.assertFalse(host.isVisible())
        self.assertTrue(host.testAttribute(Qt.WA_DontShowOnScreen))
        for w in (
            tab.spin_charge_refine,
            tab.spin_charge_outer_min,
            tab.spin_charge_outer_max,
            tab.spin_transition_cells,
        ):
            self.assertTrue(host.isAncestorOf(w))
            self.assertFalse(w.isVisible())
            self.assertFalse(w.isWindow())
        # Horizontal scrolling remains enabled (AsNeeded/AlwaysOn — not AlwaysOff).
        self.assertNotEqual(
            tab._left_setup_scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarAlwaysOff,
        )
        # No visible QSpinBox/QAbstractItemView may float as a direct oversized overlay.
        from PyQt5.QtWidgets import QAbstractItemView, QSpinBox

        for child in tab.findChildren((QSpinBox, QAbstractItemView)):
            if not child.isVisible():
                continue
            if child.parentWidget() is tab:
                g = child.geometry()
                if g.width() >= 200 or g.height() >= 200:
                    self.fail(
                        f"Visible floating overlay: {type(child).__name__} "
                        f"geom={g.getRect()} objectName={child.objectName()!r}"
                    )


if __name__ == "__main__":
    unittest.main()
