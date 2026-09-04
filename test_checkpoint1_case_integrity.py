"""Checkpoint 1: Allrun args, inventory fail-closed, topology, domain ceil."""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from allrun_commands import (
    AllrunParseError,
    parse_allrun_preprocess_sequence,
    preparation_commands_for_case,
    validate_utility_arguments,
)
from axisymmetric_2d import align_axisymmetric_domain, validate_mapping_source
from case_topology import CaseDimension, classify_case_topology
from external_case_workflow_2d import (
    CopyVerificationError,
    INVENTORY_HASH_CHUNK_SIZE,
    _sha256_file_chunked,
    compare_inventories,
    create_working_copy,
    inventory_case,
)
from models_2d import CaseInputs2D, MappingSource2D
from test_external_working_copy_2d import REPO, _make_unmeshed_source, _write


class AllrunArgumentTests(unittest.TestCase):
    def _allrun(self, root: Path, body: str) -> None:
        _write(root / "Allrun", "#!/bin/sh\n" + body)

    def test_plain_supported_utilities(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._allrun(
                root,
                "runApplication blockMesh\nrunApplication setFields\nrunApplication checkMesh\n",
            )
            seq = parse_allrun_preprocess_sequence(str(root))
            self.assertEqual(
                [c.display() for c in seq],
                ["blockMesh", "setFields", "checkMesh"],
            )

    def test_blockmesh_dict_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._allrun(
                root,
                "runApplication blockMesh -dict system/blockMeshDict.custom\n"
                "runApplication setFields\n",
            )
            seq = parse_allrun_preprocess_sequence(str(root))
            self.assertEqual(seq[0].utility, "blockMesh")
            self.assertEqual(seq[0].arguments, ("-dict", "system/blockMeshDict.custom"))
            self.assertNotEqual(seq[0].display(), "blockMesh")

    def test_setfields_dict_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._allrun(
                root,
                "runApplication blockMesh\n"
                "runApplication setFields -dict system/setFieldsDict.alt\n",
            )
            seq = parse_allrun_preprocess_sequence(str(root))
            self.assertEqual(seq[1].arguments, ("-dict", "system/setFieldsDict.alt"))

    def test_checkmesh_supported_flags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._allrun(
                root,
                "runApplication blockMesh\n"
                "runApplication checkMesh -allGeometry -allTopology\n",
            )
            seq = parse_allrun_preprocess_sequence(str(root))
            self.assertEqual(
                seq[1].arguments, ("-allGeometry", "-allTopology")
            )

    def test_unsupported_utility_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._allrun(root, "runApplication blockMesh\nrunApplication curl evil\n")
            with self.assertRaises(AllrunParseError) as ctx:
                parse_allrun_preprocess_sequence(str(root))
            self.assertIn("curl", str(ctx.exception))

    def test_unsupported_argument_rejected_without_bare_fallback(self):
        cmd = validate_utility_arguments(
            "blockMesh",
            ["-parallel"],
            source_line="runApplication blockMesh -parallel",
        )
        self.assertFalse(cmd.valid)
        self.assertIn("-parallel", cmd.rejection_reason)
        self.assertIn("blockMesh", cmd.rejection_reason)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._allrun(root, "runApplication blockMesh -parallel\n")
            with self.assertRaises(AllrunParseError):
                parse_allrun_preprocess_sequence(str(root))

    def test_missing_dict_value_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._allrun(root, "runApplication blockMesh -dict\n")
            with self.assertRaises(AllrunParseError) as ctx:
                parse_allrun_preprocess_sequence(str(root))
            self.assertIn("-dict", str(ctx.exception))
            self.assertIn("missing", str(ctx.exception).lower())

    def test_shell_operator_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._allrun(root, "runApplication blockMesh > /tmp/x\n")
            with self.assertRaises(AllrunParseError) as ctx:
                parse_allrun_preprocess_sequence(str(root))
            self.assertIn("Shell", str(ctx.exception))

    def test_preparation_appends_checkmesh(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._allrun(root, "runApplication blockMesh\nrunApplication setFields\n")
            cmds = preparation_commands_for_case(str(root))
            self.assertEqual(cmds[-1].utility, "checkMesh")


class InventoryIntegrityTests(unittest.TestCase):
    def test_unreadable_source_file_fails_verification(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            secret = source / "constant" / "secret.bin"
            secret.write_bytes(b"abc")
            inv = inventory_case(str(source))
            self.assertTrue(inv.is_valid)

            with mock.patch(
                "external_case_workflow_2d._sha256_file_chunked",
                side_effect=OSError("Permission denied"),
            ):
                bad = inventory_case(str(source))
            self.assertFalse(bad.is_valid)
            self.assertTrue(any("secret.bin" in e.rel for e in bad.read_errors))
            comparison = compare_inventories(bad, inv)
            self.assertFalse(comparison.ok)
            self.assertTrue(
                any(m.category == "inventory_read_error" for m in comparison.mismatches)
            )

    def test_deleted_file_during_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            vanishing = source / "constant" / "vanishing.dat"
            vanishing.write_bytes(b"x" * 10)

            real_chunked = _sha256_file_chunked

            def flaky(path, *, expected_size=None, cancel_token=None):
                if path.name == "vanishing.dat":
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    raise OSError("No such file")
                return real_chunked(
                    path, expected_size=expected_size, cancel_token=cancel_token
                )

            with mock.patch(
                "external_case_workflow_2d._sha256_file_chunked", side_effect=flaky
            ):
                inv = inventory_case(str(source))
            self.assertFalse(inv.is_valid)
            self.assertTrue(any("vanishing.dat" in e.rel for e in inv.read_errors))

    def test_changed_file_during_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "mutating.bin"
            target.write_bytes(b"a" * 100)

            def mutating_open(path, mode="rb"):
                handle = open(path, mode)
                original_read = handle.read

                def read(size=-1):
                    data = original_read(size)
                    if data:
                        path.write_bytes(b"b" * 200)
                    return data

                handle.read = read  # type: ignore[method-assign]
                return handle

            with mock.patch.object(Path, "open", mutating_open):
                with self.assertRaises(RuntimeError):
                    _sha256_file_chunked(target, expected_size=100)

    def test_chunked_hash_equals_standard_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "large.bin"
            # Larger than one chunk to exercise the streaming path.
            payload = os.urandom(INVENTORY_HASH_CHUNK_SIZE + 12345)
            path.write_bytes(payload)
            digest, total = _sha256_file_chunked(path, expected_size=len(payload))
            self.assertEqual(total, len(payload))
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_staging_cleanup_after_verification_failure(self):
        from external_case_workflow_2d import InventoryReadError

        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            dest = Path(td) / "wc"
            good = inventory_case(str(source))
            bad = inventory_case(str(source))
            bad.read_errors = (
                InventoryReadError(
                    rel="constant/phaseProperties",
                    exception_type="OSError",
                    message="simulated unreadable",
                ),
            )
            with mock.patch(
                "external_case_workflow_2d.inventory_case",
                side_effect=[good, bad],
            ):
                with self.assertRaises(CopyVerificationError) as ctx:
                    create_working_copy(str(source), str(dest), str(REPO))
            self.assertIn("constant/phaseProperties", str(ctx.exception))
            self.assertFalse(dest.exists())
            leftovers = list(Path(td).glob("wc.incomplete*"))
            self.assertEqual(leftovers, [])


class TopologyClassifierCallerTests(unittest.TestCase):
    def test_valid_wedge_pair(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            result = classify_case_topology(str(source))
            self.assertEqual(result.classification, CaseDimension.AXISYMMETRIC_WEDGE)

    def test_one_wedge_only(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            boundary = (source / "system" / "blockMeshDict").read_text(encoding="utf-8")
            boundary = boundary.replace(
                "backWedge { type wedge; faces ((0 4 5 3)); }",
                "backWedge { type patch; faces ((0 4 5 3)); }",
            )
            (source / "system" / "blockMeshDict").write_text(boundary, encoding="utf-8")
            result = classify_case_topology(str(source))
            self.assertEqual(result.classification, CaseDimension.AMBIGUOUS_OR_INVALID)

    def test_mixed_wedge_and_empty(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            text = (source / "system" / "blockMeshDict").read_text(encoding="utf-8")
            text = text.replace(
                "outer { type patch; faces ((1 2 5 4)); }",
                "frontAndBack { type empty; faces ((1 2 5 4)); }",
            )
            (source / "system" / "blockMeshDict").write_text(text, encoding="utf-8")
            result = classify_case_topology(str(source))
            self.assertEqual(result.classification, CaseDimension.AMBIGUOUS_OR_INVALID)

    def test_wedge_text_in_comments_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root / "system" / "blockMeshDict",
                "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n"
                "// type wedge appears only in this comment\n"
                "boundary\n(\n"
                "  left { type patch; faces ((0 1 2 3)); }\n"
                "  right { type patch; faces ((4 5 6 7)); }\n"
                ");\n",
            )
            result = classify_case_topology(str(root))
            self.assertNotEqual(result.classification, CaseDimension.AXISYMMETRIC_WEDGE)

    def test_mapping_source_uses_classifier(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            inputs = CaseInputs2D(
                initialization_source="From 1D",
                mapping=MappingSource2D(case_path=str(source), mapped_radius=0.5),
            )
            with mock.patch(
                "case_topology.classify_case_topology",
                wraps=classify_case_topology,
            ) as spy:
                report = validate_mapping_source(inputs)
                self.assertTrue(spy.called)
            self.assertFalse(
                any("not verified as an axisymmetric wedge" in e for e in report.errors)
            )

    def test_mapping_source_rejects_non_wedge_via_classifier(self):
        with tempfile.TemporaryDirectory() as td:
            source = _make_unmeshed_source(Path(td) / "src")
            text = (source / "system" / "blockMeshDict").read_text(encoding="utf-8")
            text = text.replace("type wedge;", "type patch;")
            (source / "system" / "blockMeshDict").write_text(text, encoding="utf-8")
            inputs = CaseInputs2D(
                initialization_source="From 1D",
                mapping=MappingSource2D(case_path=str(source), mapped_radius=0.5),
            )
            report = validate_mapping_source(inputs)
            self.assertTrue(
                any("axisymmetric wedge" in e for e in report.errors)
            )


class DomainCeilPolicyTests(unittest.TestCase):
    def test_requested_radius_not_reduced(self):
        domain = align_axisymmetric_domain(10.03, 5.0, 0.5)
        self.assertGreaterEqual(domain.effective_radius, 10.03 - 1e-12)
        self.assertEqual(domain.radial_cells, 21)
        self.assertAlmostEqual(domain.effective_radius, 10.5, places=9)

    def test_requested_height_not_reduced(self):
        domain = align_axisymmetric_domain(5.0, 10.03, 0.5)
        self.assertGreaterEqual(domain.effective_height, 10.03 - 1e-12)
        self.assertEqual(domain.vertical_cells, 21)

    def test_already_aligned_dimensions_unchanged(self):
        domain = align_axisymmetric_domain(20.0, 20.0, 1.0)
        self.assertFalse(domain.adjusted)
        self.assertEqual(domain.radial_cells, 20)
        self.assertEqual(domain.vertical_cells, 20)
        self.assertAlmostEqual(domain.effective_radius, 20.0)
        self.assertAlmostEqual(domain.effective_height, 20.0)


if __name__ == "__main__":
    unittest.main()
