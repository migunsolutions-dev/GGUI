"""2D probe order / missing probe must not attach blastFoam values by index."""
from __future__ import annotations

import os
import tempfile
import unittest

from validation.auto_points import ValidationPoint
from validation.probes import (
    PROBE_MISMATCH,
    PROBE_MISSING,
    match_probe_to_point,
    parse_probe_history,
    series_for_index,
)


def _write_probe(path: str, headers, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for i, xyz in enumerate(headers):
            handle.write(f"# Probe {i} ({xyz[0]} {xyz[1]} {xyz[2]})\n")
        handle.write("# Time\n")
        for t, values in rows:
            handle.write(f"{t} " + " ".join(str(v) for v in values) + "\n")


class ProbeMatchTests(unittest.TestCase):
    def test_reordered_probes_match_by_coordinates_not_index(self):
        point = ValidationPoint(
            point_id="VAL_2D_001",
            dim="2d",
            index=0,
            range_m=0.4,
            x=0.4,
            y=0.5,
            z=0.0,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "p")
            _write_probe(
                path,
                headers=((0.8, 0.5, 0.0), (0.4, 0.5, 0.0)),
                rows=((0.0, (101325.0, 201325.0)),),
            )
            locs, _times, _cols = parse_probe_history(path)
            idx, reason = match_probe_to_point(locs, (point.x, point.y, point.z))
            self.assertEqual(reason, "")
            self.assertEqual(idx, 1)
            self.assertNotEqual(idx, point.index)

    def test_missing_probe_is_invalid(self):
        locs = ["0.8 0.5 0"]
        idx, reason = match_probe_to_point(locs, (0.4, 0.5, 0.0))
        self.assertIsNone(idx)
        self.assertEqual(reason, PROBE_MISMATCH)

    def test_empty_probe_file_is_missing(self):
        idx, reason = match_probe_to_point([], (0.4, 0.5, 0.0))
        self.assertIsNone(idx)
        self.assertEqual(reason, PROBE_MISSING)

    def test_openfoam_great_samples_are_dropped(self):
        times = [0.0, 0.001, 0.002]
        cols = [[101325.0, -1.79769313486e307, 2.0e5]]
        t, v = series_for_index(times, cols, 0)
        self.assertEqual(t, [0.0, 0.002])
        self.assertEqual(v, [101325.0, 2.0e5])


if __name__ == "__main__":
    unittest.main()
