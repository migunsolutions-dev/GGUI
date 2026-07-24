"""Focused tests for Windows→WSL imported working-case copy."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from external_case_workflow_2d import (
    CaseInventory,
    CopyVerificationError,
    InventoryEntry,
    _is_wsl_unc_path,
    _is_zone_identifier_rel,
    compare_inventories,
    create_working_copy,
    inventory_case,
)
from solver_runner import SolverRunner


REPO = Path(__file__).resolve().parent
WSL_ROOT = r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work"
WSL_ROOT_ALT = r"\\wsl$\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _mini_case(root: Path) -> Path:
    _write(root / "system" / "controlDict", "application blastFoam;\nendTime 1;\n")
    _write(root / "system" / "blockMeshDict", "convertToMeters 1;\n")
    _write(root / "0" / "p.orig", "internalField uniform 1;\n")
    _write(root / "Allrun", "#!/bin/sh\nrunApplication blockMesh\n")
    return root


def _wsl_work_available() -> bool:
    try:
        return os.path.isdir(WSL_ROOT)
    except OSError:
        return False


class PathConversionTests(unittest.TestCase):
    def test_is_wsl_unc_variants(self):
        self.assertTrue(_is_wsl_unc_path(WSL_ROOT))
        self.assertTrue(_is_wsl_unc_path(WSL_ROOT_ALT))
        self.assertFalse(_is_wsl_unc_path(r"C:\Users\migun\GGUI_imported_cases"))
        self.assertFalse(_is_wsl_unc_path("/home/naor/OpenFOAM/naor-9/run/Work"))

    def test_solver_runner_unc_and_linux_forms(self):
        distro, linux = SolverRunner._win_unc_to_wsl_path_and_distro(WSL_ROOT)
        self.assertEqual(distro, "Ubuntu-20.04")
        self.assertEqual(linux, "/home/naor/OpenFOAM/naor-9/run/Work")
        distro2, linux2 = SolverRunner._win_unc_to_wsl_path_and_distro(WSL_ROOT_ALT)
        self.assertEqual(distro2, "Ubuntu-20.04")
        self.assertEqual(linux2, linux)
        _, mnt = SolverRunner._win_unc_to_wsl_path_and_distro(
            r"C:\Users\migun\Desktop\GGUI\axisymmetricCharge"
        )
        self.assertEqual(mnt, "/mnt/c/Users/migun/Desktop/GGUI/axisymmetricCharge")

    def test_zone_identifier_detection(self):
        self.assertTrue(_is_zone_identifier_rel("Allrun\uf03aZone.Identifier"))
        self.assertTrue(_is_zone_identifier_rel("system/controlDict:Zone.Identifier"))
        self.assertFalse(_is_zone_identifier_rel("system/controlDict"))


class InventorySemanticsTests(unittest.TestCase):
    def test_relative_inventory_ignores_zone_identifier(self):
        with tempfile.TemporaryDirectory() as td:
            root = _mini_case(Path(td) / "src")
            # Simulate 9P ADS materialization.
            junk = root / "Allrun\uf03aZone.Identifier"
            junk.write_bytes(b"[ZoneTransfer]\nZoneId=3\n")
            inv = inventory_case(str(root))
            self.assertNotIn("Allrun\uf03aZone.Identifier", inv.files)
            self.assertIn("Allrun", inv.files)

    def test_timestamps_and_permissions_not_in_identity(self):
        a = InventoryEntry(rel="f", kind="file", size=1, sha256="abc")
        b = InventoryEntry(rel="f", kind="file", size=1, sha256="abc")
        self.assertEqual(a, b)

    def test_content_mismatch_rejected(self):
        src = CaseInventory(
            files={"a": "1"},
            dirs=(),
            entries={"a": InventoryEntry("a", "file", 1, "1")},
        )
        dst = CaseInventory(
            files={"a": "2"},
            dirs=(),
            entries={"a": InventoryEntry("a", "file", 1, "2")},
        )
        cmp = compare_inventories(src, dst, write_report=False)
        self.assertFalse(cmp.ok)
        self.assertEqual(cmp.first.category, "sha256_mismatch")

    def test_missing_and_extra_rejected(self):
        src = CaseInventory(
            files={"a": "1"},
            dirs=(),
            entries={"a": InventoryEntry("a", "file", 1, "1")},
        )
        dst = CaseInventory(
            files={"b": "1"},
            dirs=(),
            entries={"b": InventoryEntry("b", "file", 1, "1")},
        )
        cmp = compare_inventories(src, dst, write_report=False)
        cats = {m.category for m in cmp.mismatches}
        self.assertIn("missing_from_destination", cats)
        self.assertIn("extra_in_destination", cats)


class LocalCopyTests(unittest.TestCase):
    def test_windows_copytree_still_works_locally(self):
        with tempfile.TemporaryDirectory() as td:
            source = _mini_case(Path(td) / "src space")
            dest = Path(td) / "wc"
            before = inventory_case(str(source))
            paths = create_working_copy(str(source), str(dest), str(REPO))
            self.assertEqual(paths.copy_method, "windows_copytree")
            self.assertEqual(before, inventory_case(str(source)))
            self.assertEqual(before, inventory_case(paths.working_copy_dir))

    def test_unicode_path_component(self):
        with tempfile.TemporaryDirectory() as td:
            source = _mini_case(Path(td) / "casé_α")
            dest = Path(td) / "wc_α"
            paths = create_working_copy(str(source), str(dest), str(REPO))
            self.assertTrue(os.path.isdir(paths.working_copy_dir))

    def test_existing_destination_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            source = _mini_case(Path(td) / "src")
            dest = Path(td) / "wc"
            dest.mkdir()
            (dest / "keep.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                create_working_copy(str(source), str(dest), str(REPO))
            self.assertTrue((dest / "keep.txt").is_file())

    def test_copy_failure_cleanup_narrow(self):
        with tempfile.TemporaryDirectory() as td:
            source = _mini_case(Path(td) / "src")
            dest = Path(td) / "wc"
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
            self.assertFalse(any(Path(td).glob("wc.incomplete*")))


@unittest.skipUnless(_wsl_work_available(), "WSL Work UNC root not available")
class WslProductionCopyTests(unittest.TestCase):
    def test_windows_source_to_wsl_destination(self):
        source = REPO / "axisymmetricCharge"
        if not source.is_dir():
            self.skipTest("axisymmetricCharge fixture missing")
        dest = os.path.join(WSL_ROOT, "_ggui_test_imported_copy")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        before = inventory_case(str(source))
        try:
            paths = create_working_copy(str(source), dest, str(REPO))
            self.assertEqual(paths.copy_method, "wsl_cp")
            self.assertEqual(paths.distro, "Ubuntu-20.04")
            self.assertTrue(paths.source_linux.startswith("/mnt/c/"))
            self.assertTrue(paths.dest_linux.startswith("/home/"))
            self.assertEqual(before, inventory_case(str(source)))
            self.assertEqual(before, inventory_case(paths.working_copy_dir))
            # No Zone.Identifier pollution
            for name in Path(paths.working_copy_dir).rglob("*"):
                self.assertFalse(_is_zone_identifier_rel(name.name))
        finally:
            if os.path.exists(dest):
                shutil.rmtree(dest, ignore_errors=True)

    def test_wsl_dollar_unc_equivalent(self):
        with tempfile.TemporaryDirectory() as td:
            source = _mini_case(Path(td) / "src")
            dest = os.path.join(WSL_ROOT_ALT, "_ggui_test_imported_copy_alt")
            if os.path.exists(dest):
                shutil.rmtree(dest)
            try:
                paths = create_working_copy(str(source), dest, str(REPO))
                self.assertEqual(paths.copy_method, "wsl_cp")
                self.assertEqual(inventory_case(str(source)), inventory_case(dest))
            finally:
                if os.path.exists(dest):
                    shutil.rmtree(dest, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
