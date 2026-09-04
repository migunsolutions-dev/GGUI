"""1D outer-radius boundary mapping into blastFoam outlet BCs."""
from __future__ import annotations

import os
import tempfile
import unittest

from dataclasses import replace
from generator_1d import Generator1D
from models import (
    BOUNDARY_1D_REFLECT,
    BOUNDARY_1D_TERMINATE,
    BOUNDARY_1D_TRANSMIT,
    CaseInputs1D,
    RecommendedParams1D,
)


def _inputs(right: str) -> CaseInputs1D:
    return CaseInputs1D(
        radius=1.0,
        cell_size=0.05,
        p_atm=101325.0,
        t_atm=288.0,
        mass_kg=1.0,
        rho_charge=1601.0,
        energy_j_per_kg=4.52e6,
        material_props={
            "rho": 1601.0,
            "A": 609.77e9,
            "B": 12.95e9,
            "R1": 4.50,
            "R2": 1.40,
            "omega": 0.25,
            "E0": 4.52e6,
        },
        max_cfl=0.5,
        end_time_s=1.0e-3,
        right_boundary=right,
    )


def _rec() -> RecommendedParams1D:
    return RecommendedParams1D(
        r_min=1.0e-4,
        ignition_point=(0.0, 0.0, 0.0),
        ignition_radius=0.01,
        dt0=1.0e-8,
        maxCo=0.5,
        maxDeltaT=1.0e-5,
    )


class Generator1DRightBoundaryTests(unittest.TestCase):
    def _generate(self, right: str) -> str:
        root = tempfile.mkdtemp(prefix="ggui_1d_bc_")
        gen = Generator1D(root)
        return gen.generate(f"case_{right.lower()}", _inputs(right), _rec())

    def _read(self, case_dir: str, *parts: str) -> str:
        with open(os.path.join(case_dir, *parts), encoding="utf-8") as handle:
            return handle.read()

    def test_default_and_transmit_keep_wave_transmissive_outlet(self):
        default_dir = self._generate(BOUNDARY_1D_TRANSMIT)
        p_text = self._read(default_dir, "0.orig", "p")
        u_text = self._read(default_dir, "0.orig", "U")
        mesh = self._read(default_dir, "system", "blockMeshDict")
        self.assertIn("pressureWaveTransmissive", p_text)
        self.assertNotIn("type slip", u_text)
        self.assertIn("outlet     { type patch;", mesh)

    def test_terminate_uses_zero_gradient_outflow(self):
        case_dir = self._generate(BOUNDARY_1D_TERMINATE)
        p_text = self._read(case_dir, "0.orig", "p")
        mesh = self._read(case_dir, "system", "blockMeshDict")
        self.assertNotIn("pressureWaveTransmissive", p_text)
        self.assertIn("outlet { type zeroGradient; }", p_text.replace("\n", " ").replace("  ", " "))
        self.assertIn("outlet     { type patch;", mesh)

    def test_reflect_uses_slip_wall(self):
        case_dir = self._generate(BOUNDARY_1D_REFLECT)
        p_text = self._read(case_dir, "0.orig", "p")
        u_text = self._read(case_dir, "0.orig", "U")
        mesh = self._read(case_dir, "system", "blockMeshDict")
        self.assertNotIn("pressureWaveTransmissive", p_text)
        self.assertIn("type slip", u_text)
        self.assertIn("outlet     { type wall;", mesh)


class Recommended1DInnerMeshTests(unittest.TestCase):
    def test_r_min_stays_inside_charge_but_not_half_a_cell(self):
        from profiles import compute_recommended_1d, get_profile

        rec = compute_recommended_1d(
            radius=5.0,
            cell_size=0.001,
            charge_radius=0.31,
            profile=get_profile("Balanced"),
            max_cfl_from_ui=0.5,
        )
        self.assertGreaterEqual(rec.r_min, 0.01)
        self.assertLessEqual(rec.r_min, 0.2 * 0.31)
        self.assertAlmostEqual(rec.r_min, 0.05 * 0.31, places=6)

    def test_small_charge_caps_r_min_at_20_percent(self):
        from profiles import compute_recommended_1d, get_profile

        rec = compute_recommended_1d(
            radius=1.0,
            cell_size=0.005,
            charge_radius=0.05,
            profile=get_profile("Balanced"),
            max_cfl_from_ui=0.5,
        )
        self.assertAlmostEqual(rec.r_min, 0.2 * 0.05, places=6)

    def test_generator_floors_axis_sliver(self):
        import math
        import re

        root = tempfile.mkdtemp(prefix="ggui_1d_axis_")
        gen = Generator1D(root)
        inputs = replace(_inputs(BOUNDARY_1D_TRANSMIT), axis_epsilon=1e-3)
        case_dir = gen.generate("case_axis", inputs, _rec())
        with open(os.path.join(case_dir, "system", "blockMeshDict"), encoding="utf-8") as handle:
            mesh = handle.read()
        verts = [
            tuple(float(v) for v in m)
            for m in re.findall(
                r"\(\s*([-eE0-9.]+)\s+([-eE0-9.]+)\s+([-eE0-9.]+)\s*\)",
                mesh,
            )
        ]
        thetas = []
        for x, y, z in verts[:8]:
            r = math.sqrt(x * x + y * y + z * z)
            if r <= 0:
                continue
            thetas.append(math.acos(max(-1.0, min(1.0, x / r))))
        self.assertGreaterEqual(min(thetas), 0.08)


class Generator1DIgnitionInWedgeTests(unittest.TestCase):
    def test_detonation_point_is_off_axis_inside_the_wedge(self):
        import math
        import re

        from profiles import compute_recommended_1d, get_profile

        root = tempfile.mkdtemp(prefix="ggui_1d_ign_")
        gen = Generator1D(root)
        inputs = replace(_inputs(BOUNDARY_1D_TRANSMIT), cell_size=0.001, end_time_s=0.03)
        rec = compute_recommended_1d(
            radius=inputs.radius,
            cell_size=inputs.cell_size,
            charge_radius=gen.calculate_charge_radius(inputs.mass_kg, inputs.rho_charge),
            profile=get_profile("Balanced"),
            max_cfl_from_ui=inputs.max_cfl,
        )
        case_dir = gen.generate("case_ign", inputs, rec)
        with open(
            os.path.join(case_dir, "constant", "phaseProperties"), encoding="utf-8"
        ) as handle:
            text = handle.read()
        match = re.search(r"points\s+\(\(([^)]+)\)\)", text)
        self.assertIsNotNone(match)
        x, y, z = (float(v) for v in match.group(1).split())
        self.assertGreater(abs(y), 1.0e-9)
        r = math.sqrt(x * x + y * y + z * z)
        theta = math.acos(max(-1.0, min(1.0, x / r)))
        axis_eps, cone_half, _ = Generator1D.wedge_angles(inputs)
        self.assertGreaterEqual(theta, axis_eps - 1e-9)
        self.assertLessEqual(theta, cone_half + 1e-9)
        with open(os.path.join(case_dir, "Allrun"), encoding="utf-8") as handle:
            allrun = handle.read()
        self.assertIn("PIPESTATUS[0]", allrun)

    def test_impulse_function_object_does_not_write_every_timestep(self):
        text = Generator1DSparseOutputTests()._read_control(enable_impulse=True)
        impulse_block = text.split("impulse", 1)[1].split("probes1d", 1)[0]
        self.assertIn("executeControl  timeStep;", impulse_block)
        self.assertIn("writeControl    writeTime;", impulse_block)
        self.assertNotIn("writeInterval   1;", impulse_block)


class Generator1DSparseOutputTests(unittest.TestCase):
    def _read_control(self, **kwargs) -> str:
        root = tempfile.mkdtemp(prefix="ggui_1d_write_")
        gen = Generator1D(root)
        inputs = replace(_inputs(BOUNDARY_1D_TRANSMIT), **kwargs)
        case_dir = gen.generate("case_write", inputs, _rec())
        with open(os.path.join(case_dir, "system", "controlDict"), encoding="utf-8") as handle:
            return handle.read()

    def test_default_writes_fields_only_at_end_time(self):
        text = self._read_control()
        self.assertIn("purgeWrite      1;", text)
        self.assertNotIn("writeInterval   1e-05;", text)
        self.assertNotIn("writeInterval   1e-5;", text)
        # radius=1 m → safety endTime = (1/300)*2
        self.assertIn("endTime         0.006666666667;", text)
        self.assertIn("writeInterval   0.006666666667;", text)

    def test_explicit_write_interval_is_kept(self):
        text = self._read_control(write_interval_s=1e-5)
        self.assertIn("writeInterval   1e-05;", text)
        self.assertIn("purgeWrite      1;", text)


class SolverWriteNowStopTests(unittest.TestCase):
    def test_request_solver_write_and_stop_sets_write_now(self):
        from solver_runner import request_solver_write_and_stop

        root = tempfile.mkdtemp(prefix="ggui_1d_stop_")
        sys_dir = os.path.join(root, "system")
        os.makedirs(sys_dir)
        cd_path = os.path.join(sys_dir, "controlDict")
        with open(cd_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("stopAt          endTime;\nwriteInterval   0.03;\n")
        self.assertTrue(request_solver_write_and_stop(root))
        with open(cd_path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("stopAt          writeNow;", text)
        self.assertNotIn("stopAt          endTime;", text)


class ProbeWriteIntervalParseTests(unittest.TestCase):
    def test_reads_probes1d_not_watchdog(self):
        from solver_runner import probe_write_interval_from_control_dict

        text = """
writeInterval   0.03;
functions
{
    probes1d
    {
        type            probes;
        writeControl    timeStep;
        writeInterval   25;
    }
    watchdog_probe
    {
        writeInterval   1;
    }
}
"""
        self.assertEqual(probe_write_interval_from_control_dict(text), 25)

    def test_generated_default_probe_interval_is_25(self):
        root = tempfile.mkdtemp(prefix="ggui_1d_probe_")
        gen = Generator1D(root)
        case_dir = gen.generate("case_probe", _inputs(BOUNDARY_1D_TRANSMIT), _rec())
        with open(os.path.join(case_dir, "system", "controlDict"), encoding="utf-8") as handle:
            text = handle.read()
        from solver_runner import probe_write_interval_from_control_dict

        self.assertEqual(probe_write_interval_from_control_dict(text), 25)
        self.assertIn("writeInterval   25;", text)
        self.assertEqual(text.count("writeInterval   25;"), 2)


class WatchdogStopLogicTests(unittest.TestCase):
    def test_missed_149kpa_peak_stops_after_fall(self):
        from solver_runner import WatchdogState, watchdog_should_stop

        state = WatchdogState()
        self.assertFalse(watchdog_should_stop(101325.0, state))
        self.assertFalse(watchdog_should_stop(148811.0, state))
        self.assertTrue(watchdog_should_stop(133000.0, state))

    def test_strong_jump_stops_immediately(self):
        from solver_runner import WatchdogState, watchdog_should_stop

        state = WatchdogState()
        self.assertFalse(watchdog_should_stop(101325.0, state))
        self.assertTrue(watchdog_should_stop(1.6e5, state))

    def test_small_noise_does_not_stop(self):
        from solver_runner import WatchdogState, watchdog_should_stop

        state = WatchdogState()
        self.assertFalse(watchdog_should_stop(101325.0, state))
        self.assertFalse(watchdog_should_stop(105000.0, state))
        self.assertFalse(watchdog_should_stop(101400.0, state))


if __name__ == "__main__":
    unittest.main()
