"""Focused tests for context-aware Load Summary classification and presentation."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict

from case_loader import UI_FIELD_KEYS, load_case
from load_summary_report import (
    ALTERNATIVE_MAPPING,
    CONFLICT_AMBIGUOUS,
    DERIVED,
    LOADED_CONVERTED,
    LOADED_EXACT,
    NOT_APPLICABLE,
    PRESERVED_UNCHANGED,
    SOLVER_DEFAULT,
    UNSUPPORTED_LOST,
    classify_load_fields,
    format_load_summary_text,
    strip_report_metadata,
    _jwl_emit_4g,
    _BUILTIN_JWL,
)


def _base_out(**overrides: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "_case_dir": "",
        "_load_notes": [],
        "charge_shape": "Sphere",
        "material_name": "C4",
        "enable_dyn_refine": False,
        "enable_local_refinement": False,
        "write_control_type": "adjustableRunTime",
        "write_interval_time": 1e-4,
        "write_interval_raw": 1e-4,
        "charge_outer_refine_enable": False,
        "activation_model": "pressureBased",
        "mass_kg": 1.0,
        "rho_charge": 1601.0,
    }
    out.update(overrides)
    return out


def _classes(report) -> Dict[str, str]:
    return {f.gui_key: f.classification for f in report.fields}


def _entry(report, key: str):
    for f in report.fields:
        if f.gui_key == key:
            return f
    raise KeyError(key)


class ContextAwareLoadSummaryTests(unittest.TestCase):
    def test_sphere_hides_cylinder_cuboid_only_fields(self) -> None:
        report = classify_load_fields(_base_out(charge_shape="Sphere"), UI_FIELD_KEYS)
        cls = _classes(report)
        self.assertEqual(cls["charge_width"], NOT_APPLICABLE)
        self.assertEqual(cls["charge_height"], NOT_APPLICABLE)
        self.assertEqual(cls["charge_lbyd"], NOT_APPLICABLE)
        self.assertEqual(cls["cylinder_axis"], NOT_APPLICABLE)
        self.assertEqual(cls["charge_length"], NOT_APPLICABLE)
        text = format_load_summary_text(report.to_load_summary_dict(), "case")
        self.assertNotIn("charge_width:", text)
        self.assertNotIn("charge_lbyd:", text)

    def test_cylinder_hides_cuboid_only_fields(self) -> None:
        report = classify_load_fields(
            _base_out(
                charge_shape="Cylinder",
                charge_lbyd=2.5,
                mass_kg=25.0,
                rho_charge=1601.0,
            ),
            UI_FIELD_KEYS,
        )
        cls = _classes(report)
        self.assertEqual(cls["charge_width"], NOT_APPLICABLE)
        self.assertEqual(cls["charge_height"], NOT_APPLICABLE)
        self.assertEqual(cls["charge_length"], DERIVED)

    def test_fixed_mesh_hides_amr_only_fields(self) -> None:
        report = classify_load_fields(
            _base_out(enable_dyn_refine=False, enable_local_refinement=False),
            UI_FIELD_KEYS,
        )
        cls = _classes(report)
        self.assertEqual(cls["refine_interval"], NOT_APPLICABLE)
        self.assertEqual(cls["upper_refine_level"], NOT_APPLICABLE)
        text = format_load_summary_text(report.to_load_summary_dict(), "case")
        self.assertNotIn("upper_refine_level:", text)

    def test_amr_reports_applicable_amr_fields(self) -> None:
        report = classify_load_fields(
            _base_out(
                enable_dyn_refine=True,
                enable_local_refinement=True,
                refine_interval=3,
                lower_refine_threshold=0.1,
                unrefine_threshold=0.1,
                refine_max=1,
                dyn_refine_max=1,
            ),
            UI_FIELD_KEYS,
        )
        cls = _classes(report)
        self.assertEqual(cls["refine_interval"], LOADED_EXACT)
        self.assertEqual(cls["upper_refine_level"], SOLVER_DEFAULT)
        self.assertEqual(cls["dyn_refine_min"], NOT_APPLICABLE)
        self.assertEqual(cls["refine_min"], NOT_APPLICABLE)

    def test_adjustable_runtime_shows_time_alternative(self) -> None:
        report = classify_load_fields(
            _base_out(
                write_control_type="adjustableRunTime",
                write_interval_time=0.0001,
                write_interval_raw=0.0001,
            ),
            UI_FIELD_KEYS,
        )
        cls = _classes(report)
        self.assertEqual(cls["write_interval_steps"], NOT_APPLICABLE)
        self.assertEqual(cls["write_interval_time"], ALTERNATIVE_MAPPING)
        text = format_load_summary_text(report.to_load_summary_dict(), "case")
        self.assertIn("write_interval_time", text)
        self.assertIn("Resolved field entries", text)
        self.assertNotIn("Loaded-field count", text)

    def test_timestep_shows_step_alternative(self) -> None:
        out = _base_out(
            write_control_type="timeStep",
            write_interval_steps=50,
            write_interval_raw=50,
        )
        out.pop("write_interval_time", None)
        report = classify_load_fields(out, UI_FIELD_KEYS)
        cls = _classes(report)
        self.assertEqual(cls["write_interval_time"], NOT_APPLICABLE)
        self.assertEqual(cls["write_interval_steps"], ALTERNATIVE_MAPPING)

    def test_write_control_exact_vs_converted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "system"), exist_ok=True)
            with open(os.path.join(td, "system", "controlDict"), "w", encoding="utf-8") as f:
                f.write("writeControl    adjustableRunTime;\nwriteInterval   1e-4;\n")
            report = classify_load_fields(
                _base_out(
                    _case_dir=td,
                    write_control_type="adjustableRunTime",
                    write_interval_time=1e-4,
                ),
                UI_FIELD_KEYS,
            )
            self.assertEqual(_classes(report)["write_control_type"], LOADED_EXACT)

        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "system"), exist_ok=True)
            with open(os.path.join(td, "system", "controlDict"), "w", encoding="utf-8") as f:
                f.write("writeControl    runTime;\nwriteInterval   1e-4;\n")
            report = classify_load_fields(
                _base_out(
                    _case_dir=td,
                    write_control_type="adjustableRunTime",
                    write_interval_time=1e-4,
                ),
                UI_FIELD_KEYS,
            )
            self.assertEqual(_classes(report)["write_control_type"], LOADED_CONVERTED)

    def test_derived_charge_radius_and_backup_factor(self) -> None:
        import math

        mass, rho, lbyd = 25.0, 1601.0, 2.5
        vol = mass / rho
        r = (vol / (2.0 * math.pi * lbyd)) ** (1.0 / 3.0)
        backup = 1.0
        factor = backup / r
        report = classify_load_fields(
            _base_out(
                charge_shape="Cylinder",
                mass_kg=mass,
                rho_charge=rho,
                charge_lbyd=lbyd,
                charge_radius=r,
                charge_backup_radius_override=backup,
                charge_capture_radius=backup,
                charge_backup_radius_factor=factor,
                charge_capture_mode="manual",
            ),
            UI_FIELD_KEYS,
        )
        self.assertEqual(_classes(report)["charge_radius"], DERIVED)
        self.assertEqual(_classes(report)["charge_backup_radius_factor"], DERIVED)

    def test_builtin_c4_hides_custom_when_floats_match(self) -> None:
        report = classify_load_fields(
            _base_out(
                material_name="C4",
                jwl_A=_BUILTIN_JWL["C4"]["A"],
                jwl_B=_BUILTIN_JWL["C4"]["B"],
                jwl_R1=_BUILTIN_JWL["C4"]["R1"],
                jwl_R2=_BUILTIN_JWL["C4"]["R2"],
                jwl_omega=_BUILTIN_JWL["C4"]["omega"],
            ),
            UI_FIELD_KEYS,
        )
        self.assertEqual(_classes(report)["custom_material_props"], NOT_APPLICABLE)

    def test_builtin_material_mismatch_shows_alternative(self) -> None:
        report = classify_load_fields(
            _base_out(
                material_name="C4",
                jwl_A=1.0e12,  # differs from built-in C4 A
                jwl_B=_BUILTIN_JWL["C4"]["B"],
                jwl_R1=_BUILTIN_JWL["C4"]["R1"],
                jwl_R2=_BUILTIN_JWL["C4"]["R2"],
                jwl_omega=_BUILTIN_JWL["C4"]["omega"],
            ),
            UI_FIELD_KEYS,
        )
        entry = _entry(report, "custom_material_props")
        self.assertEqual(entry.classification, ALTERNATIVE_MAPPING)
        self.assertIn("A", str(entry.source_value))

    def test_outer_off_inactive_fields_not_exact(self) -> None:
        report = classify_load_fields(
            _base_out(
                charge_outer_refine_enable=False,
                charge_outer_refine_level=0,
                charge_outer_refine_min=0,
                charge_outer_refine_max=0,
            ),
            UI_FIELD_KEYS,
        )
        cls = _classes(report)
        self.assertEqual(cls["outside_extent"], NOT_APPLICABLE)
        self.assertEqual(cls["transition_cells"], NOT_APPLICABLE)
        self.assertEqual(cls["charge_outer_refine_level"], NOT_APPLICABLE)
        self.assertEqual(cls["charge_outer_refine_min"], NOT_APPLICABLE)
        self.assertEqual(cls["charge_outer_refine_max"], NOT_APPLICABLE)
        # enable flag is structure-inferred (present/absent chargeRefineOuter), not a raw leaf copy
        self.assertEqual(cls["charge_outer_refine_enable"], LOADED_CONVERTED)
        # Inactive levels must not inflate exact-load counts
        exact = [f for f in report.fields if f.classification == LOADED_EXACT]
        self.assertFalse(any(f.gui_key == "charge_outer_refine_level" for f in exact))
        text = format_load_summary_text(report.to_load_summary_dict(), "case")
        self.assertNotIn("outside_extent", text)
        self.assertNotIn("charge_outer_refine_level:", text)

    def test_real_conflict_ambiguous(self) -> None:
        report = classify_load_fields(
            _base_out(enable_dyn_refine=True),
            UI_FIELD_KEYS,
            ambiguous_keys=["cfl_value"],
        )
        self.assertEqual(_classes(report)["cfl_value"], CONFLICT_AMBIGUOUS)
        text = format_load_summary_text(report.to_load_summary_dict(), "case")
        self.assertIn("Conflicts / ambiguous:", text)
        self.assertIn("AMBIGUOUS", text)
        self.assertIn("cfl_value", text)
        # Must not say Needs review: 0 while Needs attention is non-empty
        self.assertNotIn("Needs review: 0", text)

    def test_real_preserved_unchanged(self) -> None:
        report = classify_load_fields(
            _base_out(end_time_s=0.0025),
            UI_FIELD_KEYS,
            preserved_keys=["end_time_s"],
            regen_proof={
                "end_time_s": {
                    "unchanged": True,
                    "original": 0.0025,
                    "regenerated": 0.0025,
                    "source_file": "system/controlDict",
                    "source_path": "endTime",
                    "destination": "system/controlDict/endTime",
                }
            },
        )
        self.assertEqual(_classes(report)["end_time_s"], PRESERVED_UNCHANGED)
        # Non-GUI leaf via regen_proof only
        report2 = classify_load_fields(
            _base_out(),
            UI_FIELD_KEYS,
            regen_proof={
                "[parity] controlDict/startTime": {
                    "unchanged": True,
                    "original": 0,
                    "regenerated": 0,
                    "source_file": "system/controlDict",
                    "source_path": "startTime",
                    "destination": "system/controlDict/startTime",
                }
            },
        )
        self.assertEqual(
            _classes(report2)["[parity] controlDict/startTime"], PRESERVED_UNCHANGED
        )

    def test_absence_alone_not_preserved_without_case_proof(self) -> None:
        # No case_dir / empty dir -> cannot prove source-key absence -> not PRESERVED.
        report = classify_load_fields(
            _base_out(
                _case_dir="",
                enable_dyn_refine=True,
                enable_local_refinement=True,
            ),
            UI_FIELD_KEYS,
        )
        self.assertEqual(_classes(report)["dynamic_max_cells"], CONFLICT_AMBIGUOUS)
        self.assertEqual(_classes(report)["begin_unrefine"], CONFLICT_AMBIGUOUS)
        self.assertEqual(_classes(report)["enable_balancing"], CONFLICT_AMBIGUOUS)

    def test_balancing_context_for_balance_interval(self) -> None:
        # Balancing absent/disabled -> NOT_APPLICABLE
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "constant"), exist_ok=True)
            with open(
                os.path.join(td, "constant", "dynamicMeshDict"), "w", encoding="utf-8"
            ) as f:
                f.write("dynamicFvMesh adaptiveFvMesh;\nmaxRefinement 1;\n")
            report = classify_load_fields(
                _base_out(
                    _case_dir=td,
                    enable_dyn_refine=True,
                    enable_local_refinement=True,
                ),
                UI_FIELD_KEYS,
            )
            self.assertEqual(_classes(report)["balance_interval"], NOT_APPLICABLE)
            self.assertEqual(_classes(report)["enable_balancing"], PRESERVED_UNCHANGED)
            self.assertEqual(_classes(report)["dynamic_max_cells"], PRESERVED_UNCHANGED)
            self.assertEqual(_classes(report)["begin_unrefine"], PRESERVED_UNCHANGED)
            self.assertEqual(_entry(report, "dynamic_max_cells").source_path, "maxCells")
            self.assertEqual(_entry(report, "begin_unrefine").source_path, "beginUnrefine")
            self.assertEqual(
                _entry(report, "enable_balancing").source_path, "enableBalancing"
            )
            self.assertEqual(
                _entry(report, "balance_interval").source_path,
                "loadBalance/balanceInterval",
            )

        # Balancing explicitly enabled, interval absent -> CONFLICT_AMBIGUOUS
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "constant"), exist_ok=True)
            with open(
                os.path.join(td, "constant", "dynamicMeshDict"), "w", encoding="utf-8"
            ) as f:
                f.write(
                    "dynamicFvMesh adaptiveFvMesh;\n"
                    "enableBalancing true;\n"
                    "maxRefinement 1;\n"
                )
            report = classify_load_fields(
                _base_out(
                    _case_dir=td,
                    enable_dyn_refine=True,
                    enable_local_refinement=True,
                    enable_balancing=True,
                ),
                UI_FIELD_KEYS,
            )
            self.assertEqual(_classes(report)["balance_interval"], CONFLICT_AMBIGUOUS)
            # Explicitly present optional key classified from source, not omission parity
            self.assertEqual(_classes(report)["enable_balancing"], LOADED_EXACT)

    def test_derived_cylinder_length_shows_source_rule(self) -> None:
        report = classify_load_fields(
            _base_out(
                charge_shape="Cylinder",
                mass_kg=25.0,
                rho_charge=1601.0,
                charge_lbyd=2.5,
            ),
            UI_FIELD_KEYS,
        )
        entry = _entry(report, "charge_length")
        self.assertEqual(entry.classification, DERIVED)
        self.assertEqual(entry.value_origin, "report_interpretation")
        text = format_load_summary_text(report.to_load_summary_dict(), "case")
        self.assertIn("charge_length", text)

    def test_parent_containers_not_falsely_unsupported(self) -> None:
        report = classify_load_fields(_base_out(), UI_FIELD_KEYS)
        summary = report.to_load_summary_dict()
        self.assertEqual(summary.get("unsupported"), {})
        text = format_load_summary_text(summary, "case", include_technical=True)
        for bad in (
            "blocks:",
            "geometry:",
            "regions:",
            "refinementSurfaces:",
            "equationOfState:",
            "activationModel:",
        ):
            self.assertNotIn(bad, text)

    def test_copy_output_follows_context_rules(self) -> None:
        report = classify_load_fields(
            _base_out(
                charge_shape="Cylinder",
                mass_kg=25.0,
                rho_charge=1601.0,
                charge_lbyd=2.5,
                material_name="C4",
                jwl_A=_BUILTIN_JWL["C4"]["A"],
                jwl_B=_BUILTIN_JWL["C4"]["B"],
                jwl_R1=_BUILTIN_JWL["C4"]["R1"],
                jwl_R2=_BUILTIN_JWL["C4"]["R2"],
                jwl_omega=_BUILTIN_JWL["C4"]["omega"],
                write_control_type="adjustableRunTime",
                write_interval_time=5e-5,
                charge_outer_refine_enable=False,
            ),
            UI_FIELD_KEYS,
        )
        copied = format_load_summary_text(
            report.to_load_summary_dict(), "case", include_technical=True
        )
        self.assertIn("Derived / alternative mappings", copied)
        self.assertIn("Total needing attention:", copied)
        self.assertNotIn("custom_material_props", copied)
        self.assertNotIn("outside_extent", copied)

    def test_load_case_payload_unchanged_by_report(self) -> None:
        out = _base_out(
            charge_shape="Cylinder",
            mass_kg=25.0,
            rho_charge=1601.0,
            charge_lbyd=2.5,
            write_interval_time=5e-5,
        )
        before = copy.deepcopy(out)
        classify_load_fields(out, UI_FIELD_KEYS)
        self.assertEqual(out, before)


@unittest.skipUnless(
    os.path.isdir(r"C:/Users/migun/Desktop/building3D/building3D/building3D"),
    "external building3D reference case not present",
)
class Building3dLoadSummaryIntegrationTests(unittest.TestCase):
    CASE = r"C:/Users/migun/Desktop/building3D/building3D/building3D"

    def test_building3d_parity_leaves(self) -> None:
        data = load_case(self.CASE)
        by = {c["gui_key"]: c for c in data["_load_summary"]["classifications"]}
        # JWL A replacement visible
        self.assertIn("[parity] phaseProperties/products/equationOfState/A", by)
        a = by["[parity] phaseProperties/products/equationOfState/A"]
        self.assertEqual(a["classification"], ALTERNATIVE_MAPPING)
        self.assertEqual(a["difference_kind"], "numeric")
        self.assertIn("6.098e", str(a["regenerated_value"]))
        # pMin replacement
        self.assertIn("[parity] phaseProperties/initiation/pMin", by)
        p = by["[parity] phaseProperties/initiation/pMin"]
        self.assertEqual(p["classification"], ALTERNATIVE_MAPPING)
        self.assertEqual(p["difference_kind"], "numeric")
        # feature edge level
        self.assertIn("[parity] snappyHexMeshDict/features/*/level", by)
        f = by["[parity] snappyHexMeshDict/features/*/level"]
        self.assertEqual(f["classification"], ALTERNATIVE_MAPPING)
        self.assertEqual(int(f["source_value"]), 0)
        self.assertEqual(int(f["regenerated_value"]), 1)

    def test_building3d_context_classifications(self) -> None:
        data = load_case(self.CASE)
        summary = data["_load_summary"]
        by = {c["gui_key"]: c["classification"] for c in summary["classifications"]}
        self.assertEqual(by["charge_width"], NOT_APPLICABLE)
        self.assertEqual(by["write_interval_steps"], NOT_APPLICABLE)
        self.assertEqual(by["write_interval_time"], ALTERNATIVE_MAPPING)
        self.assertEqual(by["charge_length"], DERIVED)
        self.assertEqual(by["charge_radius"], DERIVED)
        self.assertEqual(by["charge_backup_radius_factor"], DERIVED)
        self.assertEqual(by["write_control_type"], LOADED_EXACT)
        self.assertEqual(by["custom_material_props"], NOT_APPLICABLE)
        self.assertEqual(by["outside_extent"], NOT_APPLICABLE)
        self.assertEqual(by["charge_outer_refine_level"], NOT_APPLICABLE)
        self.assertEqual(by["ignition_mode"], ALTERNATIVE_MAPPING)
        self.assertEqual(by["dyn_refine_min"], NOT_APPLICABLE)
        self.assertEqual(by["upper_refine_level"], SOLVER_DEFAULT)
        # Omission-parity optional AMR fields
        self.assertEqual(by["dynamic_max_cells"], PRESERVED_UNCHANGED)
        self.assertEqual(by["begin_unrefine"], PRESERVED_UNCHANGED)
        self.assertEqual(by["enable_balancing"], PRESERVED_UNCHANGED)
        self.assertEqual(by["balance_interval"], NOT_APPLICABLE)
        counts = summary["counts"]
        self.assertEqual(counts.get(CONFLICT_AMBIGUOUS, 0), 0)
        self.assertEqual(counts.get(UNSUPPORTED_LOST, 0), 1)
        self.assertEqual(
            counts.get(CONFLICT_AMBIGUOUS, 0) + counts.get(UNSUPPORTED_LOST, 0), 1
        )
        text = format_load_summary_text(summary, self.CASE, include_technical=True)
        self.assertIn("Resolved field entries", text)
        self.assertIn("Total needing attention: 1", text)
        self.assertIn("Conflicts / ambiguous: 0", text)
        self.assertIn("Unsupported / lost: 1", text)
        self.assertNotIn("Loaded-field count", text)
        self.assertNotIn("Needs review: 0", text)
        # Only searchable cylinder under Needs attention in default view body
        default = format_load_summary_text(summary, self.CASE, include_technical=False)
        needs_block = default.split("Needs attention")[1].split("Derived /")[0]
        self.assertIn("snappy:cylinder", needs_block)
        self.assertNotIn("dynamic_max_cells", needs_block)
        self.assertNotIn("enable_balancing", needs_block)

    def test_building3d_optional_amr_source_paths(self) -> None:
        data = load_case(self.CASE)
        by = {c["gui_key"]: c for c in data["_load_summary"]["classifications"]}
        self.assertEqual(by["dynamic_max_cells"]["source_path"], "maxCells")
        self.assertEqual(by["begin_unrefine"]["source_path"], "beginUnrefine")
        self.assertEqual(by["enable_balancing"]["source_path"], "enableBalancing")
        self.assertEqual(
            by["balance_interval"]["source_path"], "loadBalance/balanceInterval"
        )
        tech = format_load_summary_text(
            data["_load_summary"], self.CASE, include_technical=True
        )
        self.assertIn("maxCells", tech)
        self.assertIn("beginUnrefine", tech)
        self.assertIn("enableBalancing", tech)

    def test_building3d_repeatability(self) -> None:
        a = load_case(self.CASE)
        b = load_case(self.CASE)
        self.assertEqual(strip_report_metadata(a), strip_report_metadata(b))


@unittest.skipUnless(
    os.path.isdir(r"C:/Users/migun/Desktop/building3D/building3D/building3D"),
    "external building3D reference case not present",
)
class BaselineVersusPreviewPayloadTests(unittest.TestCase):
    """True isolated comparison: main 182c169 load_case vs preview load_case."""

    CASE = r"C:/Users/migun/Desktop/building3D/building3D/building3D"
    REPO = os.path.dirname(os.path.abspath(__file__))
    BASELINE_COMMIT = "182c169bb20414293251a0b64f83d5b22db47ef5"

    def test_baseline_versus_preview_payload_equality(self) -> None:
        script = r"""
import json, os, sys
case = sys.argv[1]
from case_loader import load_case
d = load_case(case)
# strip report-only
skip = {'_load_summary'}
payload = {k: v for k, v in d.items() if k not in skip}
summary = d.get('_load_summary') or {}
out = {
  'payload': payload,
  'filled': list(summary.get('filled') or []),
  'not_filled': [list(x) for x in (summary.get('not_filled') or [])],
  'provenance': dict(d.get('_provenance') or {}),
}
def conv(o):
  if isinstance(o, tuple): return list(o)
  if isinstance(o, set): return sorted(conv(x) for x in o)
  if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
  if isinstance(o, list): return [conv(x) for x in o]
  return o
json.dump(conv(out), sys.stdout)
"""
        with tempfile.TemporaryDirectory() as td:
            # Extract baseline tree from the commit (isolated from preview modules).
            baseline_root = os.path.join(td, "baseline")
            os.makedirs(baseline_root, exist_ok=True)
            archive = subprocess.run(
                ["git", "archive", self.BASELINE_COMMIT],
                cwd=self.REPO,
                check=True,
                capture_output=True,
            )
            # untar via Python
            import io
            import tarfile

            with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tf:
                try:
                    tf.extractall(baseline_root, filter="data")
                except TypeError:
                    tf.extractall(baseline_root)

            env = os.environ.copy()
            # Baseline process: only baseline tree on PYTHONPATH
            env["PYTHONPATH"] = baseline_root
            base_proc = subprocess.run(
                [sys.executable, "-c", script, self.CASE],
                cwd=baseline_root,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                base_proc.returncode,
                0,
                msg=f"baseline load failed: {base_proc.stderr}",
            )
            baseline = json.loads(base_proc.stdout)

            # Preview process: only preview tree on PYTHONPATH
            env2 = os.environ.copy()
            env2["PYTHONPATH"] = self.REPO
            prev_proc = subprocess.run(
                [sys.executable, "-c", script, self.CASE],
                cwd=self.REPO,
                env=env2,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                prev_proc.returncode,
                0,
                msg=f"preview load failed: {prev_proc.stderr}",
            )
            preview = json.loads(prev_proc.stdout)

        payload_equal = baseline["payload"] == preview["payload"]
        filled_equal = baseline["filled"] == preview["filled"]
        nf_equal = baseline["not_filled"] == preview["not_filled"]
        prov_equal = baseline["provenance"] == preview["provenance"]

        # Persist comparison artifact for the review ZIP (best-effort).
        review = os.path.join(self.REPO, "_review_load_report")
        os.makedirs(review, exist_ok=True)
        report = {
            "payload_equal": payload_equal,
            "filled_equal": filled_equal,
            "not_filled_equal": nf_equal,
            "provenance_equal": prov_equal,
            "baseline_filled_count": len(baseline["filled"]),
            "preview_filled_count": len(preview["filled"]),
            "filled_only_baseline": sorted(set(baseline["filled"]) - set(preview["filled"])),
            "filled_only_preview": sorted(set(preview["filled"]) - set(baseline["filled"])),
            "not_filled_only_baseline": [
                x for x in baseline["not_filled"] if x not in preview["not_filled"]
            ],
            "not_filled_only_preview": [
                x for x in preview["not_filled"] if x not in baseline["not_filled"]
            ],
            "provenance_only_baseline": {
                k: baseline["provenance"][k]
                for k in baseline["provenance"]
                if preview["provenance"].get(k) != baseline["provenance"][k]
            },
            "provenance_only_preview": {
                k: preview["provenance"][k]
                for k in preview["provenance"]
                if baseline["provenance"].get(k) != preview["provenance"][k]
            },
        }
        # Payload key diffs (shallow)
        bp, pp = baseline["payload"], preview["payload"]
        report["payload_keys_only_baseline"] = sorted(set(bp) - set(pp))
        report["payload_keys_only_preview"] = sorted(set(pp) - set(bp))
        value_diffs = []
        for k in sorted(set(bp) & set(pp)):
            if bp[k] != pp[k]:
                value_diffs.append({"key": k, "baseline": bp[k], "preview": pp[k]})
        report["payload_value_diffs"] = value_diffs
        with open(os.path.join(review, "BASELINE_VS_PREVIEW.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Assert equality — fail with measured diffs if not.
        self.assertTrue(payload_equal, msg=f"payload diffs: {value_diffs[:20]}")
        self.assertTrue(filled_equal, msg=report["filled_only_baseline"] or report["filled_only_preview"])
        self.assertTrue(nf_equal, msg=str(report["not_filled_only_baseline"] or report["not_filled_only_preview"]))
        self.assertTrue(prov_equal, msg=str(report["provenance_only_baseline"] or report["provenance_only_preview"]))


if __name__ == "__main__":
    unittest.main()
