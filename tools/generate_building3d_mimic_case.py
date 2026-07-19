#!/usr/bin/env python3
"""Generate a GGUI 3D case tuned to mirror building3D/building3D for audit/diff workflows.

Run from repo root:  python tools/generate_building3d_mimic_case.py

Does not modify the reference case; writes a new folder under _audit_building3d_ggui/.
"""
from __future__ import annotations

import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from generator_3d import Generator3D
from models import CaseInputs3D, ObstacleData


def building3d_mimic_inputs() -> CaseInputs3D:
    """Match manual building3D/building3D as closely as CaseInputs3D allows."""
    rho = 1601.0
    mass_kg = 25.0
    lbyd = 2.5
    vol = mass_kg / rho
    r_cyl = (vol / (2.0 * math.pi * lbyd)) ** (1.0 / 3.0)
    length = 2.0 * r_cyl * lbyd

    stl = os.path.join(_REPO, "building3D", "building3D", "constant", "triSurface", "L_Wall.stl")
    if not os.path.isfile(stl):
        raise FileNotFoundError(f"Reference STL not found: {stl}")

    return CaseInputs3D(
        min_point=(-5.0, -5.0, 0.0),
        max_point=(5.0, 5.0, 5.0),
        cell_size=0.5,
        charge_center=(0.0, 0.0, 0.5),
        charge_shape="Cylinder",
        mass_kg=mass_kg,
        cylinder_radius=r_cyl,
        cylinder_axis="Z",
        material_name="C4",
        rho_charge=rho,
        energy_j_per_kg=4.5e6,
        p_atm=101298.0,
        t_atm=300.0,
        end_time_s=0.0025,
        delta_t=1e-7,
        write_interval_steps=1,
        cores=4,
        charge_aspect=lbyd,
        charge_length=length,
        obstacles=[
            ObstacleData(
                stl_path=stl,
                name="walls",
                scale=0.001,
                refinement_level=1,
            )
        ],
        enable_local_refinement=True,
        enable_dyn_refine=True,
        enable_obstacle_refine=True,
        refine_min=2,
        refine_max=3,
        obstacle_refine_min=1,
        obstacle_refine_max=1,
        charge_outer_refine_min=2,
        charge_outer_refine_max=2,
        outside_extent=0.35,
        transition_cells=2,
        charge_refinement_level=5,
        buffer_layers=5,
        charge_capture_mode="manual",
        charge_capture_radius=1.0,
        charge_capture_factor=1.0,
        dyn_refine_max=1,
        refine_interval=3,
        lower_refine_threshold=0.1,
        unrefine_threshold=0.1,
        n_buffer_layers_dynamic=2,
        refine_indicator_field="densityGradient",
        write_control_type="adjustableRunTime",
        write_interval_time=5e-5,
        cfl_value=0.5,
        enable_post_processing=True,
        fast_run_mode=True,
        mesh_resolve_feature_angle=30,
        mesh_implicit_feature_snap=True,
        mesh_explicit_feature_snap=False,
        mesh_multi_region_feature_snap=False,
        decomposition_method="scotch",
        decomposition_simple_n=(2, 2, 1),
        decomposition_simple_delta=0.001,
        ignition_mode="Center of Charge",
    )


def main() -> int:
    out_root = os.path.join(_REPO, "_audit_building3d_ggui")
    os.makedirs(out_root, exist_ok=True)
    inp = building3d_mimic_inputs()
    case_dir = Generator3D(out_root).generate("mimic_building3d", inp)
    print(case_dir)
    print(
        "Audit diff: python tools/compare_building3d_reference.py",
        f'"{os.path.join(_REPO, "building3D", "building3D")}"',
        f'"{case_dir}"',
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
