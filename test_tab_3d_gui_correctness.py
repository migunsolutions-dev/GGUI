from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from dataclasses import asdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget

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
        self.assertEqual(tab.spin_charge_refine.value(), 0)
        self.assertEqual(tab.spin_charge_outer_min.value(), 2)
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


if __name__ == "__main__":
    unittest.main()
