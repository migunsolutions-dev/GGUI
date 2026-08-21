"""Tests for automatic imported axisymmetric working-case workflow (Cylindrical–2D)."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication, QMessageBox, QStackedWidget

from case_loader_2d import inspect_imported_axisymmetric_case
from case_topology import CaseDimension, classify_case_topology
from external_case_workflow_2d import (
    ImportMode2D,
    create_automatic_working_copy,
    create_working_copy,
    gui_values_to_control_updates,
    inventory_case,
    make_imported_working_case_name,
    parse_allrun_preprocess_sequence,
    preparation_commands_for_case,
    prepare_working_copy,
    restore_zero_orig_fields,
    validate_working_copy_destination,
    write_control_dict_entries,
)
from imported_case_mapping_2d import FieldProvenance, map_imported_case_to_gui
from main_new import BlastFoamApp


app = QApplication.instance() or QApplication([])
REPO = Path(__file__).resolve().parent
TUTORIAL = REPO / "axisymmetricCharge"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_allrun(*, refined=True) -> str:
    setter = "setRefinedFields" if refined else "setFields"
    return f"""
#!/bin/sh
cd ${{0%/*}} || exit 1
. $WM_PROJECT_DIR/bin/tools/RunFunctions
paraFoam -builtin -touch
runApplication blockMesh
runApplication {setter}
runApplication $(getApplication)
"""


def _wedge_blockmesh() -> str:
    return """
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
H 20.0; R 20.0; cellSize 1;
nx  #calc "round($R / $cellSize)";
ny  #calc "round($H / $cellSize)";
vertices
(
    (0 0 0)
    (19.924 0 -1.736)
    (19.924 20 -1.736)
    (0 20 0)
    (19.924 0 1.736)
    (19.924 20 1.736)
);
blocks ( hex (0 1 2 3 0 4 5 3) ($nx $ny 1) simpleGrading (1 1 1) );
boundary
(
    ground { type wall; faces ((0 1 4 0)); }
    outer { type patch; faces ((1 2 5 4)); }
    frontWedge { type wedge; faces ((0 1 2 3)); }
    backWedge { type wedge; faces ((0 4 5 3)); }
);
"""


def _polymesh_boundary_wedge() -> str:
    return """
FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
4
(
frontWedge { type wedge; nFaces 10; startFace 0; }
backWedge { type wedge; nFaces 10; startFace 10; }
ground { type wall; nFaces 5; startFace 20; }
outer { type patch; nFaces 5; startFace 25; }
)
"""


def _make_unmeshed_source(root: Path) -> Path:
    _write(root / "Allrun", _minimal_allrun(refined=True))
    _write(root / "system" / "blockMeshDict", _wedge_blockmesh())
    _write(
        root / "system" / "controlDict",
        "application blastFoam;\nendTime 0.025;\ndeltaT 1e-8;\nmaxCo 0.5;\n"
        "writeControl adjustableRunTime;\nwriteInterval 0.0005;\nadjustTimeStep yes;\n",
    )
    _write(
        root / "system" / "setFieldsDict",
        "fields (alpha.c4);\nnBufferLayers 5;\n"
        "regions ( sphereToCell { centre (0 0 0); radius 0.25; "
        "backup { centre (0 0 0); radius 1; } level 5; "
        "fieldValues ( volScalarFieldValue alpha.c4 1 ); } );\n",
    )
    _write(
        root / "constant" / "phaseProperties",
        "phases (c4 air);\nc4 { type detonating; reactants { equationOfState { rho0 1601; } } "
        "initiation { E0 9e9; points ((0 0 0)); } }\n",
    )
    _write(
        root / "constant" / "dynamicMeshDict",
        "dynamicFvMesh adaptiveFvMesh;\nerrorEstimator densityGradient;\n"
        "refineInterval 1;\nlowerRefineLevel 0.05;\nunrefineLevel 0.05;\n"
        "nBufferLayers 1;\nmaxRefinement 4;\ndumpLevel true;\n",
    )
    _write(root / "0" / "alpha.c4.orig", "dimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\n")
    _write(root / "0" / "p.orig", "dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 101298;\n")
    _write(root / "0" / "T.orig", "dimensions [0 0 0 1 0 0 0];\ninternalField uniform 288;\n")
    return root


class AllrunSequenceTests(unittest.TestCase):
    def test_sequence_matches_inspected_case_pattern(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_unmeshed_source(Path(td))
            seq = parse_allrun_preprocess_sequence(str(root))
            self.assertEqual(tuple(c.utility for c in seq), ("blockMesh", "setRefinedFields"))
            cmds = preparation_commands_for_case(str(root))
            self.assertEqual(
                tuple(c.display() for c in cmds),
                ("blockMesh", "setRefinedFields", "checkMesh"),
            )

    def test_arbitrary_allrun_commands_never_executed(self):
        from allrun_commands import AllrunParseError

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root / "Allrun",
                "#!/bin/sh\nrunApplication blockMesh\nrunApplication curl http://evil\n",
            )
            with self.assertRaises(AllrunParseError):
                parse_allrun_preprocess_sequence(str(root))


class WorkingCopyFilesystemTests(unittest.TestCase):
    def test_destination_inside_repository_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            dest = REPO / "_should_not_create_wc_here"
            with self.assertRaises(ValueError):
                validate_working_copy_destination(str(source), str(dest), str(REPO))

    def test_destination_already_exists_nonempty_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            dest = Path(td) / "dst"
            dest.mkdir()
            (dest / "noise.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_working_copy_destination(str(source), str(dest), str(REPO))

    def test_source_and_working_paths_distinct_and_hashes_match(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            dest = Path(td) / "wc"
            before = inventory_case(str(source))
            paths = create_working_copy(str(source), str(dest), str(REPO))
            self.assertNotEqual(
                os.path.normpath(paths.source_dir),
                os.path.normpath(paths.working_copy_dir),
            )
            after_src = inventory_case(str(source))
            self.assertEqual(before, after_src)
            self.assertEqual(before, inventory_case(paths.working_copy_dir))

    def test_copy_failure_cleanup_narrowly_scoped(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            dest = Path(td) / "wc"
            from external_case_workflow_2d import CaseInventory, CopyVerificationError

            with mock.patch(
                "external_case_workflow_2d.inventory_case",
                side_effect=[
                    inventory_case(str(source)),
                    CaseInventory(files={"x": "1"}, dirs=(), entries={}),
                ],
            ):
                with self.assertRaises(CopyVerificationError):
                    create_working_copy(str(source), str(dest), str(REPO))
            self.assertFalse(dest.exists())
            self.assertTrue(source.exists())
            # Staging leftovers must also be gone.
            self.assertFalse(any(Path(td).glob("wc.incomplete*")))

    def test_automatic_naming_unique(self):
        when = datetime(2026, 7, 24, 12, 0, 0)
        a = make_imported_working_case_name(r"C:\x\axisymmetricCharge", when=when)
        b = make_imported_working_case_name(r"C:\x\axisymmetricCharge", when=when)
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("Case_2D_imported_axisymmetricCharge_"))
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            root = Path(td) / "cases"
            root.mkdir()
            p1 = create_automatic_working_copy(str(source), str(root), str(REPO), when=when)
            p2 = create_automatic_working_copy(str(source), str(root), str(REPO), when=when)
            self.assertNotEqual(p1.working_copy_dir, p2.working_copy_dir)
            self.assertTrue(os.path.isdir(p1.working_copy_dir))
            self.assertTrue(os.path.isdir(p2.working_copy_dir))

    def test_restore_zero_orig_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_unmeshed_source(Path(td))
            restored = restore_zero_orig_fields(str(root))
            self.assertIn("alpha.c4", restored)
            self.assertTrue((root / "0" / "alpha.c4").is_file())
            self.assertTrue((root / "0" / "alpha.c4.orig").is_file())


class MappingTests(unittest.TestCase):
    def test_imported_values_replace_native_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            mapping = map_imported_case_to_gui(str(source))
            self.assertEqual(mapping.gui_values.get("radius"), 20.0)
            self.assertEqual(mapping.gui_values.get("height"), 20.0)
            self.assertEqual(mapping.gui_values.get("material_name"), "C4")
            self.assertEqual(mapping.gui_values.get("height_of_burst"), 0.0)
            self.assertEqual(mapping.fields["charge_radius"].displayed_value, 0.25)
            self.assertNotEqual(mapping.gui_values.get("material_name"), "TNT")
            # VIPER: HOB=0 sphere uses full-sphere mass (not half / unrecovered).
            self.assertEqual(
                mapping.fields["mass_kg"].provenance, FieldProvenance.DERIVED
            )
            self.assertAlmostEqual(mapping.gui_values["mass_kg"], 104.785, places=2)
            self.assertTrue(mapping.fields["mass_kg"].editable)
            self.assertEqual(mapping.fields["radial_cells"].displayed_value, 20)
            self.assertEqual(mapping.fields["vertical_cells"].displayed_value, 20)
            self.assertEqual(mapping.gui_values.get("cell_size"), 1.0)
            self.assertIn("end_time_s", mapping.editable_keys)
            self.assertTrue(mapping.fields["end_time_s"].editable)
            self.assertIn("mass_kg", mapping.editable_keys)

    def test_graded_grid_marked_case_defined(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            text = (source / "system" / "blockMeshDict").read_text(encoding="utf-8")
            text = text.replace("simpleGrading (1 1 1)", "simpleGrading (2 1 1)")
            (source / "system" / "blockMeshDict").write_text(text, encoding="utf-8")
            mapping = map_imported_case_to_gui(str(source))
            self.assertEqual(
                mapping.fields["cell_size"].provenance, FieldProvenance.CASE_DEFINED
            )
            self.assertNotIn("cell_size", mapping.gui_values)

    def test_control_dict_writer_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            updates = gui_values_to_control_updates(
                {
                    "end_time_s": 1e-6,
                    "delta_t": 1e-9,
                    "max_co": 0.25,
                    "write_control_type": "adjustableRunTime",
                    "write_interval_time": 1e-7,
                }
            )
            result = write_control_dict_entries(str(source), updates)
            self.assertTrue(result.ok)
            self.assertIn("endTime", result.changed)
            self.assertEqual(float(result.readback["endTime"]), 1e-6)

    def test_unsupported_control_keys_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            result = write_control_dict_entries(str(source), {"startFrom": "latestTime"})
            self.assertFalse(result.ok)

    @unittest.skipUnless(TUTORIAL.is_dir(), "axisymmetricCharge fixture missing")
    def test_real_tutorial_mapping(self):
        mapping = map_imported_case_to_gui(str(TUTORIAL))
        self.assertEqual(mapping.gui_values.get("material_name"), "C4")
        self.assertEqual(mapping.fields["charge_radius"].displayed_value, 0.25)
        self.assertEqual(mapping.gui_values.get("radius"), 20.0)
        self.assertEqual(mapping.gui_values.get("height"), 20.0)
        self.assertAlmostEqual(mapping.gui_values["mass_kg"], 104.785, places=2)


class PrepareWorkflowTests(unittest.TestCase):
    @staticmethod
    def _utility_name(command):
        return getattr(command, "utility", str(command))

    def _runner_ok(self, case_dir, command):
        utility = self._utility_name(command)
        if utility == "blockMesh":
            mesh = Path(case_dir) / "constant" / "polyMesh"
            for name in ("points", "faces", "owner", "neighbour", "boundary"):
                if name == "boundary":
                    _write(mesh / name, _polymesh_boundary_wedge())
                elif name == "owner":
                    _write(
                        mesh / name,
                        "FoamFile { note \"nCells: 400\"; }\n4\n(0\n1\n2\n3\n)\n",
                    )
                else:
                    _write(mesh / name, "ok\n")
            return 0, "End blockMesh"
        if utility == "setRefinedFields":
            owner = Path(case_dir) / "0" / "polyMesh" / "owner"
            owner.parent.mkdir(parents=True, exist_ok=True)
            _write(
                owner,
                "FoamFile { note \"nCells: 712\"; }\n4\n(0\n1\n2\n3\n)\n",
            )
            for name in ("points", "faces", "neighbour", "boundary"):
                if name == "boundary":
                    _write(owner.parent / name, _polymesh_boundary_wedge())
                else:
                    _write(owner.parent / name, "ok\n")
            return 0, "Selected 52 cells, 10 faces\nEnd setRefinedFields"
        if utility == "checkMesh":
            return 0, "Mesh OK.\nEnd checkMesh"
        return 1, "unexpected"

    def test_preparation_runs_only_in_working_copy(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            dest = Path(td) / "wc"
            create_working_copy(str(source), str(dest), str(REPO))
            seen = []

            def runner(case_dir, command):
                seen.append((os.path.normpath(case_dir), self._utility_name(command)))
                return self._runner_ok(case_dir, command)

            result = prepare_working_copy(str(dest), str(source), runner)
            self.assertTrue(result.ok)
            self.assertTrue(all(c == os.path.normpath(str(dest)) for c, _ in seen))
            self.assertEqual(
                [u for _, u in seen],
                ["blockMesh", "setRefinedFields", "checkMesh"],
            )
            self.assertEqual(result.cell_count, 712)
            self.assertEqual(result.cell_count_source, "time_polyMesh")
            self.assertIn("0", result.mesh_owner_path.replace("\\", "/"))

    def test_failure_stops_later_utilities(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            dest = Path(td) / "wc"
            create_working_copy(str(source), str(dest), str(REPO))
            seen = []

            def runner(case_dir, command):
                utility = self._utility_name(command)
                seen.append(utility)
                if utility == "blockMesh":
                    return 1, "FAILED"
                return 0, "ok"

            result = prepare_working_copy(str(dest), str(source), runner)
            self.assertFalse(result.ok)
            self.assertEqual(seen, ["blockMesh"])
            self.assertEqual(result.mode, ImportMode2D.IMPORTED_2D_FAILED)

    def test_prepare_working_copy_helper_does_not_call_generator(self):
        """Legacy prepare helper remains non-generative; UI uses generator_2d."""
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            dest = Path(td) / "wc"
            create_working_copy(str(source), str(dest), str(REPO))
            with mock.patch("generator_2d.Generator2D.generate") as gen:
                result = prepare_working_copy(
                    str(dest), str(source), self._runner_ok
                )
                self.assertTrue(result.ok)
                gen.assert_not_called()


class ImportedUiTests(unittest.TestCase):
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
        cls.win._force_sync_prep = True

    @classmethod
    def tearDownClass(cls):
        try:
            cls.win.close()
        except Exception:
            pass
        for p in cls._patches:
            p.stop()

    def test_opening_wedge_auto_creates_working_case_and_normal_ui(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "axisymmetricCharge")
            before = inventory_case(str(source))
            case_root = Path(td) / "Work"
            case_root.mkdir()
            with mock.patch.object(self.win, "base_projects_path", str(case_root)):
                self.win.open_openfoam_case_path(str(source))
            ext = self.win.tab_2d._imported_case
            self.assertIsNotNone(ext)
            self.assertEqual(ext.mode, ImportMode2D.IMPORTED_2D_UNINITIALIZED)
            self.assertTrue(self.win.tab_2d.is_imported_mode)
            self.assertTrue(os.path.isdir(ext.working_copy_dir))
            self.assertNotEqual(
                os.path.normpath(ext.source_dir),
                os.path.normpath(ext.working_copy_dir),
            )
            self.assertEqual(before, inventory_case(str(source)))
            # Normal controls visible — no replacement stack.
            self.assertFalse(hasattr(self.win.tab_2d, "_left_stack") and
                             isinstance(getattr(self.win.tab_2d, "_left_stack", None), QStackedWidget)
                             and self.win.tab_2d._left_stack.currentIndex() == 1)
            self.assertFalse(hasattr(self.win.tab_2d, "txt_external_summary"))
            self.assertEqual(self.win.tab_2d.input_tabs.count(), 3)
            self.assertEqual(self.win.tab_2d.input_tabs.tabText(0), "Setup")
            self.assertEqual(self.win.tab_2d.input_tabs.tabText(1), "Mesh & AMR")
            self.assertEqual(self.win.tab_2d.input_tabs.tabText(2), "Output & Probes")
            self.assertTrue(self.win.tab_2d.lbl_import_banner.text())
            self.assertIn("Editable GGUI model", self.win.tab_2d.lbl_import_banner.text())
            # Banner is shown when a case is loaded (may not be visible offscreen).
            self.assertFalse(self.win.tab_2d.lbl_import_banner.isHidden())
            # Populated values
            self.assertEqual(self.win.tab_2d.cmb_material.currentText(), "C4")
            self.assertAlmostEqual(self.win.tab_2d.spin_radius.value(), 20.0)
            self.assertAlmostEqual(self.win.tab_2d.spin_height.value(), 20.0)
            self.assertAlmostEqual(self.win.tab_2d.spin_hob.value(), 0.0)
            self.assertIn("0.25", self.win.tab_2d.lbl_charge_r.text())
            self.assertNotEqual(self.win.tab_2d.cmb_material.currentText(), "TNT")
            # Buttons
            self.assertEqual(self.win.tab_2d.btn_initialize.text(), "Initialise Model")
            self.assertTrue(self.win.tab_2d.btn_initialize.isEnabled())
            self.assertFalse(self.win.tab_2d.btn_exact_end.isEnabled())
            self.assertFalse(self.win.tab_2d.btn_stop.isEnabled())
            self.assertIn("ready to initialise", self.win.tab_2d.lbl_state.text().lower())

    def test_imported_exact_end_launches_blastfoam_not_generator(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            dest = Path(td) / "wc"
            create_working_copy(str(source), str(dest), str(REPO))
            # Fake prepared mesh + checkMesh
            mesh = dest / "constant" / "polyMesh"
            for name in ("points", "faces", "owner", "neighbour", "boundary"):
                if name == "boundary":
                    _write(mesh / name, _polymesh_boundary_wedge())
                else:
                    _write(mesh / name, "ok\n")
            state = inspect_imported_axisymmetric_case(
                str(dest),
                source_dir=str(source),
                working_copy_dir=str(dest),
                mode=ImportMode2D.IMPORTED_2D_READY,
            )
            state.check_mesh_ok = True
            state.mesh_present = True
            self.win.tab_2d.load_imported_case(state)
            self.win.active_case_dir_2d = str(dest)
            self.win.active_case_initialized_2d = True

            started = {}

            def fake_start(self_runner):
                started["case"] = self_runner.win_case_dir
                started["intent"] = str(self_runner.intent)
                started["cmd_plan"] = True

            with mock.patch("generator_2d.Generator2D.generate") as gen, mock.patch(
                "main_new.SolverRunner.start", fake_start
            ), mock.patch("main_new.SolverRunner.isRunning", return_value=False):
                # Avoid real QThread issues: patch _start_solver body partially
                with mock.patch.object(self.win, "_start_solver") as start:
                    self.win.run_imported_2d_exact_end()
                    start.assert_called_once()
                    args, kwargs = start.call_args
                    self.assertEqual(os.path.normpath(args[0]), os.path.normpath(str(dest)))
                    self.assertEqual(kwargs.get("mode"), "2D")
                    gen.assert_not_called()
            # controlDict endTime written from GUI
            text = (dest / "system" / "controlDict").read_text(encoding="utf-8")
            self.assertIn("endTime", text)

    def test_native_2d_workflow_unchanged_when_cleared(self):
        self.win.tab_2d.clear_imported_case()
        self.assertFalse(self.win.tab_2d.is_imported_mode)
        self.assertEqual(self.win.tab_2d.btn_initialize.text(), "Initialise Model")
        self.assertTrue(self.win.tab_2d.btn_initialize.isEnabled())
        self.assertFalse(self.win.tab_2d.lbl_import_banner.isVisible())

    def test_planar_and_3d_routing_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            planar = Path(td) / "planar"
            _write(
                planar / "system" / "blockMeshDict",
                """
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
boundary ( front { type empty; faces ((0 1 2 3)); } back { type empty; faces ((4 5 6 7)); } );
""",
            )
            before = self.win.tabs.currentWidget()
            self.assertEqual(
                self.win.open_openfoam_case_path(str(planar)),
                CaseDimension.PLANAR_2D_EMPTY.value,
            )
            self.assertIs(self.win.tabs.currentWidget(), before)

            box = Path(td) / "box"
            _write(
                box / "system" / "blockMeshDict",
                """
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
boundary ( minX { type wall; faces ((0 1 2 3)); } );
""",
            )
            with mock.patch(
                "main_new.load_case",
                return_value={
                    "_load_summary": {
                        "filled": [],
                        "not_filled": [],
                        "unsupported": {},
                        "notes": [],
                    }
                },
            ):
                with mock.patch.object(self.win.tab_3d, "set_case_inputs"):
                    self.assertEqual(
                        self.win.open_openfoam_case_path(str(box)),
                        CaseDimension.GENERAL_3D.value,
                    )


if __name__ == "__main__":
    unittest.main()
