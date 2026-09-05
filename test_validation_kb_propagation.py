"""Common KB propagation graph: physical R and remap exclusion."""
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

from validation.auto_points import plan_1d, plan_2d
from validation.current_run import RunSnapshot, default_display_dims, histories_available
from validation.kb_overlay import OverlaySample, SOURCE_UFC
from validation.kb_propagation import (
    CLASS_INSIDE,
    CLASS_ON_BOUNDARY,
    CLASS_OUTSIDE,
    classify_vs_remap,
    copied_1d2d_radius_m,
    copied_2d3d_radius_m,
    first_independent_r_m,
    kb_propagation_eligible,
    physical_standoff_m,
    scaled_z_from_r,
    series_label,
)
from validation.probes import EXISTING_1D_GRAPH_FO, VALIDATION_FO
from validation.ufc_airblast import BURST_SPHERICAL, scaled_distance


def _write_handoff(case_dir: str, payload: dict) -> None:
    os.makedirs(case_dir, exist_ok=True)
    path = os.path.join(case_dir, "ggui_remap_handoff.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _touch_probe(case_dir: str, fo_name: str) -> None:
    path = os.path.join(case_dir, "postProcessing", fo_name, "0", "p")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Probe 0 (0.5 0 0)\n0 101325\n")


class PhysicalStandoffTests(unittest.TestCase):
    def test_1d_uses_euclidean_radius_from_origin(self):
        r = physical_standoff_m("1d", (0.30, 0.04, 0.03), (0.0, 1.5, 0.0))
        self.assertAlmostEqual(r, math.sqrt(0.30**2 + 0.04**2 + 0.03**2))

    def test_2d_uses_euclidean_distance_from_charge_centre(self):
        hob = 1.5
        r = physical_standoff_m("2d", (2.0, hob, 0.0), (0.0, hob, 0.0))
        self.assertAlmostEqual(r, 2.0)
        off_line = physical_standoff_m("2d", (1.0, 0.5, 0.0), (0.0, hob, 0.0))
        self.assertAlmostEqual(off_line, math.hypot(1.0, 0.5 - hob))

    def test_3d_uses_euclidean_distance_from_actual_centre(self):
        centre = (1.0, 2.0, 3.0)
        r = physical_standoff_m("3d", (4.0, 6.0, 3.0), centre)
        self.assertAlmostEqual(r, math.sqrt(3.0**2 + 4.0**2 + 0.0**2))

    def test_z_is_only_an_x_transform_of_the_same_r(self):
        r = 2.5
        mass = 5.0
        z_val = scaled_z_from_r(r, mass)
        self.assertAlmostEqual(z_val, r / (mass ** (1.0 / 3.0)))
        self.assertAlmostEqual(z_val, scaled_distance(r, mass))

    def test_no_x_offset_from_previous_series(self):
        last_1d = 1.49
        first_2d = physical_standoff_m("2d", (1.80, 1.5, 0.0), (0.0, 1.5, 0.0))
        self.assertAlmostEqual(first_2d, 1.80)
        self.assertNotAlmostEqual(first_2d, last_1d)
        self.assertGreater(abs(first_2d - last_1d), 0.2)


class RemapExclusionTests(unittest.TestCase):
    def test_authoritative_handoff_uses_user_radius_not_field_padding(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "1d")
            tgt = os.path.join(td, "2d")
            _write_handoff(
                src,
                {
                    "remap_radius_m": 1.5,
                    "field_r_max_m": 1.6422,
                    "handoff_radius_m": 1.4,
                },
            )
            _write_handoff(
                tgt,
                {
                    "remap_radius_m": 1.5,
                    "field_r_max_m": 1.6422,
                    "handoff_radius_m": 1.4,
                    "actual_remap_geometry": {
                        "copied_radius_m": 1.6422,
                        "field_r_max_m": 1.6422,
                        "requested_mapped_radius_m": 1.5,
                    },
                },
            )
            found = copied_1d2d_radius_m(
                target_2d_case=tgt,
                source_1d_case=src,
                widget_mapped_radius=0.5,
            )
            self.assertAlmostEqual(found, 1.5)
            self.assertAlmostEqual(first_independent_r_m(found, 0.01), 1.51)

    def test_old_copied_padding_does_not_override_user_radius(self):
        with tempfile.TemporaryDirectory() as td:
            tgt = os.path.join(td, "2d")
            _write_handoff(
                tgt,
                {
                    "remap_radius_m": 0.6,
                    "field_r_max_m": 0.658,
                    "actual_remap_geometry": {
                        "copied_radius_m": 0.658,
                        "field_r_max_m": 0.658,
                    },
                },
            )
            found = copied_1d2d_radius_m(target_2d_case=tgt, widget_mapped_radius=0.5)
            self.assertAlmostEqual(found, 0.6)
            self.assertAlmostEqual(first_independent_r_m(found, 0.01), 0.61)

    def test_2d_plan_excludes_inside_and_boundary_and_one_cell_guard(self):
        receive = 1.6422
        dx = 0.05
        plan = plan_2d(
            mass_kg=5.0,
            domain_radius_m=10.0,
            domain_height_m=10.0,
            hob_m=1.5,
            cell_size=dx,
            remap_receive_r_max=receive,
        )
        self.assertTrue(plan.ok)
        self.assertTrue(plan.points)
        limit = first_independent_r_m(receive, dx)
        self.assertTrue(all(p.range_m > receive for p in plan.points))
        self.assertTrue(all(p.range_m > limit for p in plan.points))
        self.assertGreater(plan.points[0].range_m, limit)
        self.assertEqual(plan.extra["remap_region"]["radius_m"], receive)
        self.assertAlmostEqual(plan.extra["remap_region"]["first_independent_r_m"], limit)
        classes = [classify_vs_remap(p.range_m, receive, dx) for p in plan.points]
        self.assertTrue(classes)
        self.assertTrue(all(c == CLASS_OUTSIDE for c in classes))

    def test_comparison_exclusion_is_remap_plus_one_2d_cell(self):
        self.assertAlmostEqual(first_independent_r_m(0.60, 0.01), 0.61)
        self.assertFalse(kb_propagation_eligible(0.60, 0.60, 0.01))
        self.assertFalse(kb_propagation_eligible(0.605, 0.60, 0.01))
        self.assertTrue(kb_propagation_eligible(0.70, 0.60, 0.01))

    def test_inside_and_boundary_gauges_are_not_kb_eligible(self):
        receive = 1.5
        dx = 0.05
        self.assertEqual(classify_vs_remap(1.2, receive, dx), CLASS_INSIDE)
        self.assertEqual(classify_vs_remap(1.5, receive, dx), CLASS_ON_BOUNDARY)
        self.assertEqual(classify_vs_remap(1.52, receive, dx), CLASS_ON_BOUNDARY)
        self.assertEqual(classify_vs_remap(1.55, receive, dx), CLASS_ON_BOUNDARY)
        self.assertEqual(classify_vs_remap(1.5500001, receive, dx), CLASS_OUTSIDE)
        self.assertFalse(kb_propagation_eligible(1.2, receive, dx))
        self.assertFalse(kb_propagation_eligible(1.5, receive, dx))
        self.assertTrue(kb_propagation_eligible(1.56, receive, dx))

    def test_3d_inside_remap_excluded_outside_included(self):
        receive = copied_2d3d_radius_m(
            prepare_payload={
                "copied_radius_m": 2.0,
                "centre": [0.0, 1.5, 0.0],
            }
        )
        self.assertAlmostEqual(receive, 2.0)
        centre = (0.0, 1.5, 0.0)
        inside = physical_standoff_m("3d", (1.0, 1.5, 0.0), centre)
        outside = physical_standoff_m("3d", (3.0, 1.5, 0.0), centre)
        self.assertEqual(classify_vs_remap(inside, receive, None), CLASS_INSIDE)
        self.assertEqual(classify_vs_remap(outside, receive, None), CLASS_OUTSIDE)
        leftover = copied_2d3d_radius_m(
            prepare_payload={"mapped_radius": 0.5, "mapped_height": 10.0}
        )
        self.assertIsNone(leftover)

    def test_3d_does_not_infer_volume_from_domain_size_alone(self):
        self.assertIsNone(copied_2d3d_radius_m(prepare_payload={"domain_radius": 20.0}))


class CommonGraphTests(unittest.TestCase):
    def test_one_series_label_per_dimension(self):
        self.assertEqual(series_label("1d"), "BF 1D")
        self.assertEqual(series_label("2d"), "BF 2D")
        self.assertEqual(series_label("3d"), "BF 3D")

    def test_simultaneous_1d_2d_3d_selection_keeps_independent_x(self):
        samples = (
            OverlaySample(
                point_id="VAL_1D_001",
                dim="1d",
                mass_kg=5.0,
                burst=BURST_SPHERICAL,
                figure="2-7",
                reference_source=SOURCE_UFC,
                range_m=1.2,
                scaled_z=scaled_z_from_r(1.2, 5.0),
                kind="bf",
            ),
            OverlaySample(
                point_id="VAL_2D_001",
                dim="2d",
                mass_kg=5.0,
                burst=BURST_SPHERICAL,
                figure="2-7",
                reference_source=SOURCE_UFC,
                range_m=1.80,
                scaled_z=scaled_z_from_r(1.80, 5.0),
                kind="bf",
            ),
            OverlaySample(
                point_id="G3",
                dim="3d",
                mass_kg=5.0,
                burst=BURST_SPHERICAL,
                figure="2-7",
                reference_source=SOURCE_UFC,
                range_m=4.0,
                scaled_z=scaled_z_from_r(4.0, 5.0),
                kind="bf",
            ),
        )
        xs_r = {s.dim: s.range_m for s in samples}
        xs_z = {s.dim: s.scaled_z for s in samples}
        self.assertAlmostEqual(xs_r["2d"], 1.80)
        self.assertNotAlmostEqual(xs_r["2d"], xs_r["1d"])
        self.assertAlmostEqual(xs_z["1d"], xs_r["1d"] / (5.0 ** (1.0 / 3.0)))
        self.assertEqual(list(xs_r.keys()), ["1d", "2d", "3d"])
        # Switching R→Z does not change point identity.
        self.assertEqual(samples[1].point_id, "VAL_2D_001")
        self.assertAlmostEqual(samples[1].range_m, 1.80)

    def test_plan_2d_range_is_physical_r_not_appended_1d_curve(self):
        plan1 = plan_1d(mass_kg=5.0, domain_radius_m=1.5, cell_size=0.01)
        plan2 = plan_2d(
            mass_kg=5.0,
            domain_radius_m=10.0,
            domain_height_m=10.0,
            hob_m=1.5,
            cell_size=0.05,
            remap_receive_r_max=1.6422,
        )
        self.assertTrue(plan1.points)
        self.assertTrue(plan2.points)
        last_1d = plan1.points[-1].range_m
        first_2d = plan2.points[0].range_m
        self.assertGreater(first_2d, last_1d)
        self.assertAlmostEqual(first_2d, plan2.points[0].x)
        self.assertAlmostEqual(plan2.points[0].y, 1.5)

    def test_completed_1d_and_2d_both_selected_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            c1 = os.path.join(td, "1d")
            c2 = os.path.join(td, "2d")
            _touch_probe(c1, EXISTING_1D_GRAPH_FO)
            _touch_probe(c2, VALIDATION_FO["2d"])
            snap = RunSnapshot(
                live_mode="2d",
                case_1d=c1,
                case_2d=c2,
                domain_radius_1d=1.5,
                domain_radius_2d=10.0,
                domain_height_2d=10.0,
            )
            self.assertTrue(histories_available(snap, "1d"))
            self.assertTrue(histories_available(snap, "2d"))
            self.assertFalse(histories_available(snap, "3d"))
            self.assertEqual(default_display_dims(snap), {"1d", "2d"})

    def test_missing_dimension_is_not_treated_as_computed(self):
        snap = RunSnapshot(live_mode="1d", domain_radius_1d=1.5, domain_radius_2d=10.0)
        self.assertEqual(default_display_dims(snap), {"1d"})
        self.assertFalse(histories_available(snap, "2d"))
        self.assertFalse(histories_available(snap, "3d"))

    def test_free_air_2d_sampling_stays_spherical(self):
        plan = plan_2d(
            mass_kg=5.0,
            domain_radius_m=10.0,
            domain_height_m=10.0,
            hob_m=1.5,
            cell_size=0.05,
            remap_receive_r_max=1.5,
        )
        self.assertEqual(plan.burst_master, BURST_SPHERICAL)
        self.assertEqual(plan.figure, "2-7")
        self.assertTrue(all(p.burst == BURST_SPHERICAL for p in plan.points))


class TabCommonGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        import sys

        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_r_to_z_keeps_point_identity_and_free_air(self):
        from tab_validation import MODE_KB, TabValidation

        snap = RunSnapshot(
            live_mode="1d",
            mass_kg=5.0,
            hob_m=1.5,
            domain_radius_1d=1.5,
            domain_radius_2d=10.0,
            domain_height_2d=10.0,
            domain_cell_2d=0.05,
            mapped_radius=0.5,
        )
        tab = TabValidation()
        tab.set_source_provider(context=lambda: snap, gauges_1d=lambda: (), probes_2d=lambda: (), probes_3d=lambda: ())
        tab.show()
        tab.combo_mode.setCurrentText(MODE_KB)
        tab._display_sync_key = tab._snapshot_cache_key()
        tab.radio_kb_sph.setChecked(True)
        tab.combo_kb_source.setCurrentText("UFC 3-340-02")
        tab.chk_show_1d.setChecked(True)
        tab.chk_show_2d.setChecked(True)
        tab.chk_show_3d.setChecked(True)
        tab.radio_kb_range.setChecked(True)
        tab._redraw()
        self.assertTrue(tab.radio_kb_sph.isChecked())
        self.assertFalse(tab.radio_kb_hemi.isChecked())
        ids_r = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
        ranges_r = [tab.table.item(r, 3).text() for r in range(tab.table.rowCount())]
        tab.radio_kb_z.setChecked(True)
        tab._redraw()
        ids_z = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
        ranges_z = [tab.table.item(r, 3).text() for r in range(tab.table.rowCount())]
        self.assertEqual(ids_r, ids_z)
        self.assertEqual(ranges_r, ranges_z)
        self.assertTrue(tab.radio_kb_sph.isChecked())
        legend = []
        if tab.plot_canvas.axes.get_legend():
            legend = [t.get_text() for t in tab.plot_canvas.axes.get_legend().get_texts()]
        self.assertFalse(any("2-15" in t for t in legend))
        sources = [tab.table.item(r, 2).text() for r in range(tab.table.rowCount())]
        dims = [tab.table.item(r, 1).text() for r in range(tab.table.rowCount())]
        self.assertFalse(any(d == "3D" and s == "BF" for d, s in zip(dims, sources)))


if __name__ == "__main__":
    unittest.main()
