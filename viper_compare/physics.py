"""Single physical test definition for VIPER vs GGUI/blastFoam comparison."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from remap_handoff_1d import REMAP_FRONT_BUFFER_CELLS_1D, handoff_radius_m
from validation.kb_propagation import classify_vs_remap


def spherical_charge_radius_m(mass_kg: float, rho: float) -> float:
    return (3.0 * float(mass_kg) / (4.0 * math.pi * float(rho))) ** (1.0 / 3.0)


@dataclass(frozen=True)
class MatchRow:
    parameter: str
    viper: str
    ggui: str
    exact: str
    reason: str


@dataclass(frozen=True)
class TestDefinition:
    mass_kg: float = 1.0
    rho_kg_m3: float = 1600.0
    energy_j_kg: float = 4.52e6
    p_atm: float = 101325.0
    t_atm: float = 288.0
    cfl_1d: float = 0.5
    cfl_2d: float = 0.4
    dx_1d: float = 0.001
    dx_2d: float = 0.01
    domain_1d_m: float = 1.0
    domain_2d_r_m: float = 2.0
    domain_2d_h_m: float = 2.0
    hob_m: float = 1.0
    r_remap_m: float = 0.60
    end_time_1d_s: float = 0.005
    end_time_2d_s: float = 0.008
    jwl_a: float = 371.2e9
    jwl_b: float = 3.231e9
    jwl_r1: float = 4.15
    jwl_r2: float = 0.95
    jwl_omega: float = 0.30
    # Common physical R for 1D and the 2D radial line z = HOB.
    r_gauges_1d: Tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.70, 0.85)
    r_gauges_2d: Tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 1.00, 1.20)

    @property
    def charge_radius_m(self) -> float:
        return spherical_charge_radius_m(self.mass_kg, self.rho_kg_m3)

    @property
    def cells_through_charge_1d(self) -> float:
        return self.charge_radius_m / self.dx_1d

    @property
    def cells_through_charge_2d(self) -> float:
        return self.charge_radius_m / self.dx_2d

    @property
    def n_cells_1d(self) -> int:
        return max(1, int(math.ceil(self.domain_1d_m / self.dx_1d - 1e-15)))

    @property
    def n_cells_2d_r(self) -> int:
        return max(1, int(math.ceil(self.domain_2d_r_m / self.dx_2d - 1e-15)))

    @property
    def n_cells_2d_z(self) -> int:
        return max(1, int(math.ceil(self.domain_2d_h_m / self.dx_2d - 1e-15)))

    @property
    def n_cells_2d(self) -> int:
        return self.n_cells_2d_r * self.n_cells_2d_z

    @property
    def r_handoff_ggui_m(self) -> float:
        return handoff_radius_m(self.r_remap_m, self.dx_1d, REMAP_FRONT_BUFFER_CELLS_1D)

    @property
    def remap_guard_m(self) -> float:
        return self.r_remap_m + self.dx_2d

    def classify_remap(self, r_m: float, receive_r_max: float | None = None) -> str:
        return classify_vs_remap(
            r_m,
            self.r_remap_m if receive_r_max is None else receive_r_max,
            self.dx_2d,
        )

    def match_table(self) -> List[MatchRow]:
        return [
            MatchRow("charge mass", "1 kg", "1 kg", "yes", ""),
            MatchRow("density", "1600 kg/m3", "1600 kg/m3", "yes", "VIPER charge_density, not JWL rho=1540"),
            MatchRow("energy", "4.52e6 J/kg", "4.52e6 J/kg", "yes", "VIPER charge_energy_od; JWL e0=5.4e6 not used by GGUI"),
            MatchRow(
                "EOS / explosive model",
                "Method 1, jwlflag_od=0",
                "blastFoam detonating JWL + pressureBased",
                "no",
                "Different source models; comparison not forced agreement",
            ),
            MatchRow("atmosphere P", "101325 Pa", "101325 Pa", "yes", ""),
            MatchRow("atmosphere T", "288 K", "288 K", "yes", ""),
            MatchRow("CFL 1D", "0.5", "0.5", "yes", ""),
            MatchRow("CFL 2D", "0.4", "0.4", "yes", ""),
            MatchRow("1D cell size", "0.001 m", "0.001 m", "yes", ""),
            MatchRow(
                "2D cell size",
                "0.01 m (JSON override)",
                "0.01 m",
                "yes",
                "TEST2.vip native 2D dx=0.05 is ~1 cell through the charge; not used",
            ),
            MatchRow("charge centre 2D", "r=0, z=HOB=1.0 m", "r=0, z=1.0 m", "yes", ""),
            MatchRow("1D domain", "1.0 m", "1.0 m", "yes", ""),
            MatchRow("2D domain", "2.0 x 2.0 m (JSON override)", "2.0 x 2.0 m", "yes", "TEST2 native 5x5 m at dx=0.01 would be 250k cells"),
            MatchRow(
                "2D boundaries",
                "left=0 axis; right/top/bottom transmitting (JSON bottom=1)",
                "axisymmetric; outer/top/bottom Open",
                "yes",
                "TEST2 native boun_bottom=0 is treated as reflecting; overridden for free-air",
            ),
            MatchRow(
                "2D remap identity",
                "direct: remapflag=0,shape=1; remap: remapflag=1,shape=0",
                "initialization_source remap vs direct",
                "n/a",
                "VIPER shape is a remap-mode companion, not a JSON sphere override. "
                "twodremapoption=1 is present on both and is not a discriminator.",
            ),
        ]

    def material_props(self) -> Dict[str, float]:
        return {
            "rho": self.rho_kg_m3,
            "A": self.jwl_a,
            "B": self.jwl_b,
            "R1": self.jwl_r1,
            "R2": self.jwl_r2,
            "omega": self.jwl_omega,
            "E0": self.energy_j_kg,
        }

    def viper_json(self, template_json_path: str, *, domain_1d_m: float | None = None) -> dict:
        with open(template_json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        r1 = float(self.domain_1d_m if domain_1d_m is None else domain_1d_m)
        data.setdefault("params_1d", {}).update(
            {
                "atmosp_od": self.p_atm,
                "atmost_od": self.t_atm,
                "cellsize_od": self.dx_1d,
                "cfl_od": self.cfl_1d,
                "charge_density_od": self.rho_kg_m3,
                "charge_energy_od": self.energy_j_kg,
                "charge_mass_od": self.mass_kg,
                "composition_od": 0,
                "domain_radius_od": r1,
                "jwlflag_od": 0,
                "method_od": 1,
            }
        )
        data.setdefault("params_2d", {}).update(
            {
                "atmosp": self.p_atm,
                "atmost": self.t_atm,
                "boun_bottom": 1,
                "boun_left": 0,
                "boun_right": 1,
                "boun_top": 1,
                "cellsize": self.dx_2d,
                "cfl": self.cfl_2d,
                "charge_density": self.rho_kg_m3,
                "charge_energy": self.energy_j_kg,
                "charge_hob": self.hob_m,
                "charge_mass": self.mass_kg,
                "composition": 0,
                "det_2d_height": self.hob_m,
                "det_2d_radius": 0,
                "domain_height": self.domain_2d_h_m,
                "domain_radius": self.domain_2d_r_m,
                "endtime_2d": self.end_time_2d_s,
                "method": 4,
            }
        )
        # Do not overwrite params_2d.shape. Direct VIP/JSON use 1; remap uses 0.
        return data

    def gauges_1d(self) -> List[Tuple[float, str]]:
        return [(r, f"G1D_R{r:.2f}") for r in self.r_gauges_1d]

    def gauges_2d(self) -> List[Tuple[float, float, str]]:
        return [(r, self.hob_m, f"G2D_R{r:.2f}") for r in self.r_gauges_2d]

    def cost_report(self) -> str:
        rc = self.charge_radius_m
        return (
            f"charge radius {rc:.5f} m\n"
            f"1D cells through charge {self.cells_through_charge_1d:.2f}\n"
            f"2D cells through charge {self.cells_through_charge_2d:.2f}\n"
            f"1D total cells {self.n_cells_1d}\n"
            f"2D total cells {self.n_cells_2d_r} x {self.n_cells_2d_z} = {self.n_cells_2d}\n"
            f"GGUI R_handoff {self.r_handoff_ggui_m:.4f} m "
            f"(R_remap={self.r_remap_m} - {REMAP_FRONT_BUFFER_CELLS_1D}*dx_1d)\n"
            f"independent 2D remap gauges require R > {self.remap_guard_m:.4f} m\n"
            "2D blastFoam estimate: 40k serial cells, endTime 0.008 s; "
            "if dt~1e-6 then ~8e3 steps; prior 40k runs were ~0.14 s/step "
            "when T was extreme (~20 min). If dt~2e-6, ~10 min. "
            "VIPER 2D GPU should finish in seconds to a few minutes.\n"
            "TEST2 native 2D dx=0.05 m gives ~1.06 cells through the charge "
            "and is rejected as under-resolved. TEST2 5 m domain at dx=0.01 "
            "would be 250k cells and is not used."
        )


def default_test() -> TestDefinition:
    return TestDefinition()
