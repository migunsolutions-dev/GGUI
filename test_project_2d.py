from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

from models_2d import CaseInputs2D, MappingSource2D, ProbePoint2D
from project_io import build_project, read_project, write_project_atomic
from test_generator_3d import _minimal


class Project2DRoundTripTests(unittest.TestCase):
    def test_legacy_3d_project_loads_without_2d_state(self):
        payload = build_project(_minimal(), probes={"probes": []}, gui_state={})
        with tempfile.NamedTemporaryFile(suffix=".ggui.json", delete=False) as stream:
            path = stream.name
        try:
            write_project_atomic(path, payload)
            loaded = read_project(path)
            self.assertIsNone(loaded["inputs_2d"])
            self.assertEqual(loaded["inputs"].charge_shape, "Sphere")
        finally:
            import os
            os.unlink(path)

    def test_direct_dynamic_2d_round_trip_preserves_disabled_values_and_probes(self):
        inputs_2d = replace(
            CaseInputs2D(),
            charge_shape="Cylinder",
            charge_refinement_level=4,
            charge_seed_mode="Auto",
            probes=(ProbePoint2D("axis", 0.0, 0.2),),
            mirrored_view=False,
        )
        payload = build_project(
            _minimal(), probes={"probes": []}, gui_state={}, inputs_2d=inputs_2d
        )
        with tempfile.NamedTemporaryFile(suffix=".ggui.json", delete=False) as stream:
            path = stream.name
        try:
            write_project_atomic(path, payload)
            restored = read_project(path)["inputs_2d"]
            self.assertEqual(restored, inputs_2d)
            self.assertEqual(restored.charge_refinement_level, 4)
            self.assertEqual(restored.probes[0].radius, 0.0)
        finally:
            import os
            os.unlink(path)

    def test_remap_provenance_round_trip(self):
        mapping = MappingSource2D(
            case_path="/source/case",
            time_mode="specific",
            specific_time="0.0001",
            mapped_radius=0.8,
            source_resolution=0.002,
            source_case_id="case-123",
        )
        inputs_2d = replace(
            CaseInputs2D(), initialization_source="From 1D", mapping=mapping
        )
        payload = build_project(
            _minimal(), probes={"probes": []}, gui_state={}, inputs_2d=inputs_2d
        )
        with tempfile.NamedTemporaryFile(suffix=".ggui.json", delete=False) as stream:
            path = stream.name
        try:
            write_project_atomic(path, payload)
            restored = read_project(path)["inputs_2d"]
            self.assertEqual(restored.mapping, mapping)
        finally:
            import os
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
