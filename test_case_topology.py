"""Tests for dimension-aware OpenFOAM case topology classification and routing."""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication, QMessageBox

from case_loader_2d import format_external_case_report_2d, inspect_external_axisymmetric_case
from case_topology import CaseDimension, classify_case_topology
from main_new import BlastFoamApp


app = QApplication.instance() or QApplication([])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_wedge_blockmesh(
    *,
    names=("frontWedge", "backWedge"),
    angle=5.0,
    include_calc=False,
) -> str:
    n0, n1 = names
    if include_calc:
        return f"""
FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
H   2.0;
R   1.5;
x #calc "$R * cos({angle} * Foam::constant::mathematical::pi/180)";
z #calc "$R * sin({angle} * Foam::constant::mathematical::pi/180)";
nz #calc "-$R * sin({angle} * Foam::constant::mathematical::pi/180)";
vertices
(
    (0 0 0)
    ($x 0 $nz)
    ($x $H  $nz)
    (0 $H 0)
    ($x 0 $z)
    ($x $H  $z)
);
blocks ( hex (0 1 2 3 0 4 5 3) (10 20 1) simpleGrading (1 1 1) );
boundary
(
    ground {{ type wall; faces ((0 1 4 0)); }}
    outer {{ type patch; faces ((1 2 5 4)); }}
    {n0} {{ type wedge; faces ((0 1 2 3)); }}
    {n1} {{ type wedge; faces ((0 4 5 3)); }}
);
"""
    # Numeric vertices equivalent to ±5° at R=1.5, H=2
    import math

    x = 1.5 * math.cos(math.radians(angle))
    z = 1.5 * math.sin(math.radians(angle))
    return f"""
FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices
(
    (0 0 0)
    ({x} 0 {-z})
    ({x} 2 {-z})
    (0 2 0)
    ({x} 0 {z})
    ({x} 2 {z})
);
blocks ( hex (0 1 2 3 0 4 5 3) (10 20 1) simpleGrading (1 1 1) );
boundary
(
    ground {{ type wall; faces ((0 1 4 0)); }}
    outer {{ type patch; faces ((1 2 5 4)); }}
    {n0} {{ type wedge; faces ((0 1 2 3)); }}
    {n1} {{ type wedge; faces ((0 4 5 3)); }}
);
"""


def _minimal_polymesh_boundary(*, wedge_names=("wedgeA", "wedgeB"), empty=False) -> str:
    if empty:
        body = """
front { type empty; nFaces 10; startFace 0; }
back { type empty; nFaces 10; startFace 10; }
outer { type patch; nFaces 20; startFace 20; }
"""
    else:
        n0, n1 = wedge_names
        body = f"""
{n0} {{ type wedge; nFaces 100; startFace 0; }}
{n1} {{ type wedge; nFaces 100; startFace 100; }}
ground {{ type wall; nFaces 10; startFace 200; }}
"""
    return f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}}
4
(
{body}
)
"""


def _minimal_empty_blockmesh() -> str:
    return """
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
vertices ( (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1) );
blocks ( hex (0 1 2 3 4 5 6 7) (10 10 1) simpleGrading (1 1 1) );
boundary
(
    front { type empty; faces ((0 1 2 3)); }
    back { type empty; faces ((4 5 6 7)); }
    walls { type wall; faces ((0 1 5 4)); }
);
"""


def _minimal_3d_blockmesh() -> str:
    return """
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
vertices ( (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1) );
blocks ( hex (0 1 2 3 4 5 6 7) (5 5 5) simpleGrading (1 1 1) );
boundary
(
    minX { type wall; faces ((0 3 7 4)); }
    maxX { type patch; faces ((1 2 6 5)); }
    minY { type wall; faces ((0 1 5 4)); }
    maxY { type patch; faces ((3 2 6 7)); }
    minZ { type wall; faces ((0 1 2 3)); }
    maxZ { type patch; faces ((4 5 6 7)); }
);
"""


class TopologyClassifierTests(unittest.TestCase):
    def test_unmeshed_axisymmetric_from_blockmesh_with_macros(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "wedge case with spaces"
            _write(root / "system" / "blockMeshDict", _minimal_wedge_blockmesh(include_calc=True))
            result = classify_case_topology(str(root))
            self.assertEqual(result.classification, CaseDimension.AXISYMMETRIC_WEDGE)
            self.assertEqual(result.evidence.source, "blockMeshDict")
            self.assertEqual(result.evidence.wedge_patch_names, ("frontWedge", "backWedge"))
            self.assertAlmostEqual(result.evidence.wedge_half_angle_deg or 0.0, 5.0)

    def test_meshed_axisymmetric_from_polymesh_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "meshed"
            _write(root / "system" / "controlDict", "application blastFoam;\n")
            _write(
                root / "constant" / "polyMesh" / "boundary",
                _minimal_polymesh_boundary(wedge_names=("alphaWedge", "betaWedge")),
            )
            # Also provide agreeing blockMesh
            _write(
                root / "system" / "blockMeshDict",
                _minimal_wedge_blockmesh(names=("alphaWedge", "betaWedge")),
            )
            result = classify_case_topology(str(root))
            self.assertEqual(result.classification, CaseDimension.AXISYMMETRIC_WEDGE)
            self.assertIn(result.evidence.source, ("polyMesh/boundary", "both"))
            self.assertEqual(set(result.evidence.wedge_patch_names), {"alphaWedge", "betaWedge"})

    def test_wedge_names_need_not_be_wedge0_wedge1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root / "system" / "blockMeshDict",
                _minimal_wedge_blockmesh(names=("leftSector", "rightSector")),
            )
            result = classify_case_topology(str(root))
            self.assertEqual(result.classification, CaseDimension.AXISYMMETRIC_WEDGE)
            self.assertEqual(result.evidence.wedge_patch_names, ("leftSector", "rightSector"))

    def test_planar_empty_case(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "system" / "blockMeshDict", _minimal_empty_blockmesh())
            result = classify_case_topology(str(root))
            self.assertEqual(result.classification, CaseDimension.PLANAR_2D_EMPTY)

    def test_general_3d_case(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "system" / "blockMeshDict", _minimal_3d_blockmesh())
            result = classify_case_topology(str(root))
            self.assertEqual(result.classification, CaseDimension.GENERAL_3D)

    def test_conflicting_polymesh_and_blockmesh(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root / "constant" / "polyMesh" / "boundary",
                _minimal_polymesh_boundary(empty=True),
            )
            _write(root / "system" / "blockMeshDict", _minimal_wedge_blockmesh())
            result = classify_case_topology(str(root))
            self.assertEqual(result.classification, CaseDimension.AMBIGUOUS_OR_INVALID)
            self.assertIn("Conflicting", result.reason)

    def test_missing_topology(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "system" / "controlDict", "application blastFoam;\n")
            result = classify_case_topology(str(root))
            self.assertEqual(result.classification, CaseDimension.AMBIGUOUS_OR_INVALID)


class RoutingAndExternalLoadTests(unittest.TestCase):
    """GUI routing tests — one shared BlastFoamApp; dialogs mocked (no modal exec_)."""

    @classmethod
    def setUpClass(cls):
        cls._msg_patches = [
            mock.patch.object(QMessageBox, "information", return_value=QMessageBox.Ok),
            mock.patch.object(QMessageBox, "warning", return_value=QMessageBox.Ok),
            mock.patch.object(QMessageBox, "critical", return_value=QMessageBox.Ok),
            mock.patch.object(
                BlastFoamApp, "_show_load_summary_dialog_2d", autospec=True
            ),
            mock.patch.object(
                BlastFoamApp, "_show_load_summary_dialog", autospec=True
            ),
        ]
        for p in cls._msg_patches:
            p.start()
        cls.win = BlastFoamApp()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.win.close()
        except Exception:
            pass
        for p in cls._msg_patches:
            p.stop()

    def setUp(self):
        # Reset routing-related state between tests without rebuilding the app.
        self.win.tab_2d.clear_external_case()
        self.win.active_case_dir_2d = None
        self.win.active_case_dir_3d = None
        self.win.tabs.setCurrentIndex(0)

    def test_axisymmetric_routing_selects_cylindrical_2d(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ax"
            _write(root / "system" / "blockMeshDict", _minimal_wedge_blockmesh(include_calc=True))
            _write(root / "system" / "controlDict", "application blastFoam;\nendTime 1;\n")
            _write(root / "Allrun", "#!/bin/sh\nrunApplication blockMesh\nrunApplication setRefinedFields\n")
            case_root = Path(td) / "Work"
            case_root.mkdir()
            with mock.patch.object(self.win, "base_projects_path", str(case_root)):
                value = self.win.open_openfoam_case_path(str(root))
            self.assertEqual(value, CaseDimension.AXISYMMETRIC_WEDGE.value)
            self.assertIs(self.win.tabs.currentWidget(), self.win.tab_2d)
            self.assertIsNotNone(self.win.tab_2d._imported_case)
            self.assertEqual(self.win.tab_2d.btn_initialize.text(), "Initialise Model")
            self.assertTrue(self.win.tab_2d.btn_initialize.isEnabled())
            self.assertFalse(self.win.tab_2d.btn_exact_end.isEnabled())
            self.assertFalse(hasattr(self.win.tab_2d, "txt_external_summary"))
            report = format_external_case_report_2d(self.win.tab_2d._imported_case)
            self.assertIn("Axisymmetric wedge", report)
            self.assertIn("Working case path", report)
            self.assertNotIn("Fields filled from case (LOADED)", report)

    def test_planar_not_silently_routed_to_3d(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "system" / "blockMeshDict", _minimal_empty_blockmesh())
            before = self.win.tabs.currentWidget()
            before_3d_case = self.win.active_case_dir_3d
            value = self.win.open_openfoam_case_path(str(root))
            self.assertEqual(value, CaseDimension.PLANAR_2D_EMPTY.value)
            self.assertIs(self.win.tabs.currentWidget(), before)
            self.assertEqual(self.win.active_case_dir_3d, before_3d_case)

    def test_failed_classification_does_not_overwrite_active_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "system" / "controlDict", "application blastFoam;\n")
            self.win.active_case_dir_3d = "/tmp/keep_me"
            self.win.tabs.setCurrentWidget(self.win.tab_1d)
            value = self.win.open_openfoam_case_path(str(root))
            self.assertEqual(value, CaseDimension.AMBIGUOUS_OR_INVALID.value)
            self.assertEqual(self.win.active_case_dir_3d, "/tmp/keep_me")
            self.assertIs(self.win.tabs.currentWidget(), self.win.tab_1d)

    def test_loading_does_not_change_source_case_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "immutable"
            _write(root / "system" / "blockMeshDict", _minimal_wedge_blockmesh(include_calc=True))
            _write(root / "system" / "controlDict", "application blastFoam;\n")
            _write(root / "Allrun", "#!/bin/sh\nrunApplication blockMesh\nrunApplication setRefinedFields\n")
            before = {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*")
                if path.is_file()
            }
            case_root = Path(td) / "Work"
            case_root.mkdir()
            with mock.patch.object(self.win, "base_projects_path", str(case_root)):
                self.win.open_openfoam_case_path(str(root))
            after = {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(set(before), set(after))

    def test_general_3d_still_selects_general_3d(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "box3d"
            _write(root / "system" / "blockMeshDict", _minimal_3d_blockmesh())
            _write(
                root / "system" / "controlDict",
                "application blastFoam;\nstartTime 0;\nendTime 1;\ndeltaT 1e-6;\n",
            )
            summary = {
                "filled": ["domain_Lx"],
                "not_filled": [("domain_Ly", "unset")],
                "unsupported": {},
                "notes": [],
            }
            with mock.patch(
                "main_new.load_case",
                return_value={"_load_summary": summary, "domain_Lx": 1.0},
            ) as mocked_load:
                with mock.patch.object(self.win.tab_3d, "set_case_inputs") as set_inputs:
                    value = self.win.open_openfoam_case_path(str(root))
            self.assertEqual(value, CaseDimension.GENERAL_3D.value)
            self.assertIs(self.win.tabs.currentWidget(), self.win.tab_3d)
            self.assertEqual(self.win.active_case_dir_3d, os.path.normpath(str(root)))
            mocked_load.assert_called_once()
            set_inputs.assert_called_once()

    def test_external_report_and_initialize_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "system" / "blockMeshDict", _minimal_wedge_blockmesh(include_calc=True))
            _write(root / "Allrun", "#!/bin/sh\nrunApplication blockMesh\nrunApplication setRefinedFields\n")
            state = inspect_external_axisymmetric_case(str(root))
            # Uninitialized imported case may initialise; exact END remains blocked.
            self.assertTrue(state.initialize_allowed)
            self.assertFalse(state.runnable)
            text = format_external_case_report_2d(state)
            self.assertIn("Initialise Model", text)
            self.assertIn("Wedge patches", text)
            self.assertIn("generator_2d fresh case", text)
            self.assertIn("editable GGUI model", text)

    def test_mirrored_external_view_remains_display_only(self):
        from axisymmetric_viewer import mirror_meridional
        import pyvista as pv

        half = pv.Plane(i_size=1, j_size=1, i_resolution=2, j_resolution=2)
        half.point_data["p"] = [1.0] * half.n_points
        mirrored = mirror_meridional(half)
        # Display merge may add points; computational cell count must stay on half mesh.
        self.assertGreaterEqual(mirrored.n_points, half.n_points)
        # Cell-count helper reads owner file, not mirrored display mesh.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "system" / "blockMeshDict", _minimal_wedge_blockmesh(include_calc=True))
            state = inspect_external_axisymmetric_case(str(root))
            self.assertIsNone(state.cell_count)  # unmeshed: no invented cells from mirror

    def test_native_2d_orientation_unchanged_by_external_loader(self):
        # Clearing external case restores Initialise Model enablement path.
        self.win.tab_2d.clear_external_case()
        self.win.tab_2d.set_simulation_state(self.win.tab_2d.simulation_state)
        self.assertTrue(self.win.tab_2d.btn_initialize.isEnabled())
        # Native domain spins remain authoritative for GGUI Y-axial Radius–Height.
        self.assertGreater(self.win.tab_2d.spin_radius.value(), 0.0)
        self.assertGreater(self.win.tab_2d.spin_height.value(), 0.0)


class RealTutorialClassificationTests(unittest.TestCase):
    def test_axisymmetric_charge_fixture_classifies_as_wedge(self):
        case = Path(r"C:\Users\migun\Desktop\GGUI\axisymmetricCharge")
        if not case.is_dir():
            self.skipTest("axisymmetricCharge fixture not present")
        result = classify_case_topology(str(case))
        self.assertEqual(result.classification, CaseDimension.AXISYMMETRIC_WEDGE)
        self.assertEqual(result.evidence.source, "blockMeshDict")
        self.assertGreaterEqual(len(result.evidence.wedge_patch_names), 2)
        for name in result.evidence.wedge_patch_names:
            # Names may be wedge0/wedge1, but classification must be by type.
            self.assertTrue(name)
        self.assertAlmostEqual(result.evidence.wedge_half_angle_deg or 0.0, 5.0)
        self.assertEqual(result.evidence.symmetry_axis, "Y")


if __name__ == "__main__":
    unittest.main()
