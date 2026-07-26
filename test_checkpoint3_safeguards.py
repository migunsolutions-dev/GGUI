"""Checkpoint 3 — material policy, schema v2, state machine, logging, deps."""
from __future__ import annotations

import ast
import importlib
import json
import logging
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from charge_seed_plan import smallest_charge_dimension_m as seed_smallest
from domain_errors import IncompleteMaterialError
from ggui_logging import configure_logging, get_logger, log_operation
from material_catalog import REQUIRED_CUSTOM_JWL_KEYS, jwl_parameters
from material_validation import (
    REQUIRED_CUSTOM_MATERIAL_KEYS,
    validate_material_definition,
    validate_required_values,
)
from models_2d import CaseInputs2D, SimulationState2D
from project_io import (
    SCHEMA_VERSION,
    ProjectFormatError,
    build_project,
    read_project,
    write_project_atomic,
)
from state_machine_2d import (
    InvalidStateTransition,
    apply_transition,
    can_run,
    state_after_input_edit,
)
import startup_mesh_metadata as smm
from test_3d_correctness import case as _minimal_3d


class MaterialResponsibilityTests(unittest.TestCase):
    def test_jwl_keys_exclude_rho(self):
        self.assertNotIn("rho", REQUIRED_CUSTOM_JWL_KEYS)
        self.assertIn("rho", REQUIRED_CUSTOM_MATERIAL_KEYS)

    def test_jwl_parameters_does_not_accept_incomplete_jwl(self):
        props = {
            "A": 1.0,
            "B": 1.0,
            "R1": 1.0,
            "R2": 1.0,
            "omega": 0.3,
            # missing energy/E0
            "rho": 1600.0,
        }
        with self.assertRaises(IncompleteMaterialError):
            jwl_parameters("Custom", props)

    def test_higher_level_validator_requires_rho(self):
        props = {
            "A": 1.0,
            "B": 1.0,
            "R1": 1.0,
            "R2": 1.0,
            "omega": 0.3,
            "energy": 4e6,
        }
        issues = validate_material_definition("Custom", props)
        self.assertTrue(any("rho" in i.message for i in issues))

    def test_generator_callers_cannot_skip_required_values(self):
        """Direct jwl_parameters callers in generators are preceded by validate_required_values."""
        root = Path(__file__).resolve().parent
        for rel in ("generator_2d.py", "generator_3d.py"):
            text = (root / rel).read_text(encoding="utf-8")
            self.assertIn("validate_required_values", text)
            self.assertIn("jwl_parameters(", text)
            # Last validate_required_values before the jwl_parameters call site.
            jwl_at = text.index("jwl_parameters(")
            validate_at = text.rfind("validate_required_values", 0, jwl_at)
            self.assertGreaterEqual(validate_at, 0, rel)

    def test_incomplete_custom_blocked_before_generation(self):
        inputs = replace(
            CaseInputs2D(),
            material_name="Custom",
            material_props={
                "A": 1.0,
                "B": 1.0,
                "R1": 1.0,
                "R2": 1.0,
                "omega": 0.3,
                "energy": 4e6,
                # rho missing
            },
            rho_charge=None,
        )
        result = validate_required_values(inputs)
        self.assertFalse(result.ok)
        with self.assertRaises(IncompleteMaterialError):
            result.raise_if_invalid()


class SharedHelperTests(unittest.TestCase):
    def test_smallest_charge_dimension_single_source(self):
        dims = {"radius": 0.1, "length": 0.5}
        a = seed_smallest("Cylinder", dims)
        b = smm.smallest_charge_dimension_m("Cylinder", dims)
        self.assertEqual(a, b)
        self.assertIs(smm.smallest_charge_dimension_m, seed_smallest)


class SchemaMigrationTests(unittest.TestCase):
    def test_schema_version_is_2(self):
        self.assertEqual(SCHEMA_VERSION, 2)

    def test_v1_to_v2_migration(self):
        inputs = _minimal_3d()
        payload_v1 = {
            "schema_version": 1,
            "project_dimension": "3D",
            "case_inputs": asdict(inputs),
            "probes": {"probes": []},
            "gui_state": {"sections": [], "obstacles": []},
            "dimensions": {
                "2D": {
                    "model": "axisymmetric-rz-wedge",
                    "case_inputs": asdict(
                        replace(
                            CaseInputs2D(),
                            rho_charge=None,
                            undefined_keys=("rho_charge",),
                        )
                    ),
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "legacy.ggui.json")
            write_project_atomic(path, payload_v1)
            loaded = read_project(path)
        self.assertEqual(loaded["payload"]["schema_version"], 2)
        self.assertIn("3D", loaded["payload"]["dimensions"])
        self.assertIn("2D", loaded["payload"]["dimensions_available"])
        self.assertIn("rho_charge", loaded["inputs_2d"].undefined_keys)
        self.assertIsNone(loaded["inputs_2d"].rho_charge)

    def test_undefined_round_trip(self):
        inputs_3d = _minimal_3d()
        inputs_2d = replace(
            CaseInputs2D(),
            material_name="",
            rho_charge=None,
            energy_j_per_kg=None,
            undefined_keys=("material_name", "rho_charge", "energy_j_per_kg"),
        )
        payload = build_project(
            inputs_3d,
            probes={"probes": []},
            gui_state={"sections": [], "selected_primary_tab": "Cylindrical 2D"},
            inputs_2d=inputs_2d,
        )
        self.assertEqual(payload["schema_version"], 2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "u.ggui.json")
            write_project_atomic(path, payload)
            loaded = read_project(path)["inputs_2d"]
        self.assertEqual(
            set(loaded.undefined_keys),
            {"material_name", "rho_charge", "energy_j_per_kg"},
        )
        self.assertIsNone(loaded.rho_charge)
        self.assertEqual(loaded.material_name, "")

    def test_unsupported_schema_version(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.ggui.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": 99, "project_dimension": "3D"}, handle)
            with self.assertRaises(ProjectFormatError) as ctx:
                read_project(path)
            self.assertIn("Unsupported schema_version", str(ctx.exception))


class StateMachine2DTests(unittest.TestCase):
    def test_all_documented_transitions(self):
        cases = [
            (SimulationState2D.DRAFT, "validate_ok", SimulationState2D.VALIDATED),
            (SimulationState2D.VALIDATED, "initialize_start", SimulationState2D.INITIALIZING),
            (SimulationState2D.INITIALIZING, "initialize_ok", SimulationState2D.INITIALIZED),
            (SimulationState2D.INITIALIZING, "initialize_fail", SimulationState2D.FAILED),
            (SimulationState2D.INITIALIZING, "initialize_cancel", SimulationState2D.FAILED),
            (SimulationState2D.INITIALIZED, "run_start", SimulationState2D.RUNNING),
            (SimulationState2D.RUNNING, "run_complete", SimulationState2D.COMPLETED),
            (SimulationState2D.RUNNING, "run_interrupt", SimulationState2D.INTERRUPTED),
            (SimulationState2D.RUNNING, "run_fail", SimulationState2D.FAILED),
            (SimulationState2D.INITIALIZED, "edit_inputs", SimulationState2D.STALE),
            (SimulationState2D.STALE, "initialize_start", SimulationState2D.INITIALIZING),
        ]
        for source, event, target in cases:
            self.assertEqual(apply_transition(source, event, strict=True), target)

    def test_stale_after_case_defining_edits(self):
        for state in (
            SimulationState2D.INITIALIZED,
            SimulationState2D.INTERRUPTED,
            SimulationState2D.COMPLETED,
            SimulationState2D.STALE,
        ):
            self.assertEqual(state_after_input_edit(state), SimulationState2D.STALE)
        self.assertFalse(can_run(SimulationState2D.STALE))

    def test_failed_init_not_runnable(self):
        state = apply_transition(
            SimulationState2D.INITIALIZING, "initialize_fail", strict=True
        )
        self.assertEqual(state, SimulationState2D.FAILED)
        self.assertFalse(can_run(state))

    def test_strict_unknown_transition(self):
        with self.assertRaises(InvalidStateTransition):
            apply_transition(SimulationState2D.DRAFT, "run_start", strict=True)


class LoggingAndImportTests(unittest.TestCase):
    def test_logging_caught_refresh_error(self):
        configure_logging(logging.DEBUG)
        logger = get_logger("viewer_refresh")
        with self.assertLogs("ggui.viewer_refresh", level="ERROR") as captured:
            try:
                raise RuntimeError("refresh boom")
            except RuntimeError as exc:
                log_operation(
                    "viewer_refresh",
                    "timer_refresh",
                    exc=exc,
                    level=logging.ERROR,
                )
        self.assertTrue(any("refresh boom" in line for line in captured.output))

    def test_pure_modules_have_no_pyqt(self):
        for name in ("execution_plan", "wsl_runtime", "state_machine_2d", "ggui_logging"):
            mod = importlib.import_module(name)
            tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(
                    any(n.startswith("PyQt5") for n in names if n),
                    msg=f"{name} imports PyQt5",
                )

    def test_dependency_imports(self):
        for name in ("PyQt5", "numpy", "matplotlib", "pyvista"):
            importlib.import_module(name)

    def test_ci_workflow_exists_and_mentions_offscreen(self):
        ci = Path(__file__).resolve().parent / ".github" / "workflows" / "ci.yml"
        self.assertTrue(ci.is_file())
        text = ci.read_text(encoding="utf-8")
        self.assertIn("QT_QPA_PLATFORM: offscreen", text)
        self.assertIn("unittest discover", text)
        self.assertIn("OpenFOAM", text)

    def test_artifact_policy_debug_summary_untracked(self):
        import subprocess

        tracked = subprocess.check_output(
            ["git", "ls-files", "debug_summary.txt"], text=True
        ).strip()
        self.assertEqual(tracked, "")


class MappingDecompositionTests(unittest.TestCase):
    def test_mapping_steps_exported(self):
        from imported_case_mapping_2d import (
            build_mapping_result,
            map_boundaries,
            map_charge,
            map_domain,
            map_material,
            map_mesh,
            map_output,
            map_solver_controls,
        )

        for fn in (
            map_domain,
            map_charge,
            map_material,
            map_boundaries,
            map_solver_controls,
            map_mesh,
            map_output,
            build_mapping_result,
        ):
            self.assertTrue(callable(fn))


if __name__ == "__main__":
    unittest.main()
