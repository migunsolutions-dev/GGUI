"""1D outer-radius boundary mapping into blastFoam outlet BCs."""
from __future__ import annotations

import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
