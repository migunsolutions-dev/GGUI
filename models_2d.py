"""Dimension-specific models for the axisymmetric Cylindrical–2D workflow."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from charge_seed_plan import (
    DEFAULT_MAX_AUTO_LEVEL,
    DEFAULT_MIN_CELLS,
    DEFAULT_TARGET_CELLS,
    SEED_MODE_AUTO,
)

# Authoritative native Cylindrical–2D runtime AMR interval default.
# New Dynamic cases use the same value for refine and unrefine scheduling.
DEFAULT_REFINE_INTERVAL = 3


class SimulationState2D(str, Enum):
    DRAFT = "Draft"
    VALIDATED = "Validated"
    INITIALIZED = "Initialized"
    RUNNING = "Running"
    INTERRUPTED = "Interrupted"
    COMPLETED = "Completed"
    STALE = "Stale"
    FAILED = "Failed"


@dataclass(frozen=True)
class ProbePoint2D:
    """A probe in user-facing axisymmetric coordinates."""

    name: str
    radius: float
    height: float


@dataclass(frozen=True)
class MappingSource2D:
    case_path: str = ""
    time_mode: str = "latest"
    specific_time: str = ""
    mapped_radius: float = 0.0
    source_resolution: Optional[float] = None
    source_case_id: str = ""


@dataclass(frozen=True)
class CaseInputs2D:
    # Domain. Radius and height are requested values; generation records effective values.
    radius: float = 1.5
    height: float = 1.5
    cell_size: float = 0.05

    # Initialization.
    initialization_source: str = "Direct Charge"  # Direct Charge | From 1D
    charge_shape: str = "Sphere"  # Sphere | Cylinder
    charge_center_r: float = 0.0  # locked; validated to exactly zero
    height_of_burst: float = 0.5
    detonation_radius: float = 0.0  # locked; validated to exactly zero
    detonation_height: float = 0.5
    charge_aspect: float = 2.5  # cylinder L/D
    mass_kg: float = 1.0
    material_name: str = "TNT"
    rho_charge: float = 1630.0
    energy_j_per_kg: float = 4.29e6
    material_props: Dict[str, Any] = field(default_factory=dict)

    # Atmosphere.
    p_atm: float = 101325.0
    t_atm: float = 288.15

    # Logical physical boundaries. Wedge faces and the axis are never user editable.
    outer_boundary: str = "Open"
    top_boundary: str = "Open"
    bottom_boundary: str = "Reflecting slip wall"

    # Solver controls.
    max_co: float = 0.5
    end_time_s: float = 1.0e-3
    delta_t: float = 1.0e-8
    adjust_time_step: bool = True
    write_control_type: str = "adjustableRunTime"
    write_interval_time: float = 1.0e-5
    write_interval_steps: int = 100
    cycle_write: int = 0
    cores: int = 1

    # Mesh mode.
    mesh_mode: str = "Dynamic Mesh (AMR)"  # exact axisymmetricCharge tutorial path

    # Startup direct-charge refinement; canonical defaults come from charge_seed_plan.py.
    charge_seed_mode: str = SEED_MODE_AUTO
    charge_refinement_level: int = 0
    charge_seed_target_cells: int = DEFAULT_TARGET_CELLS
    charge_seed_min_cells: int = DEFAULT_MIN_CELLS
    charge_seed_max_level: int = DEFAULT_MAX_AUTO_LEVEL
    buffer_layers: int = 5

    # Runtime AMR; defaults follow the approved 3D canonical model except
    # unrefine_interval, which matches refine_interval for native 2D cases
    # (Phase-2 churn evidence). Explicit saved/imported values are preserved.
    refine_indicator_field: str = "densityGradient"
    dyn_refine_max: int = 1
    refine_interval: int = DEFAULT_REFINE_INTERVAL
    unrefine_interval: int = DEFAULT_REFINE_INTERVAL
    lower_refine_threshold: float = 0.1
    upper_refine_level: Optional[float] = None
    unrefine_threshold: float = 0.1
    upper_unrefine_level: Optional[float] = None
    begin_unrefine: Optional[float] = None
    n_buffer_layers_dynamic: int = 2
    dynamic_max_cells: int = 200000000
    dump_level: bool = True
    # dynamicMeshDict Switch: force error=1 at cells containing probes/blastProbes.
    # Not a controlDict function object (no type refineProbes exists in blastFoam).
    refine_probes: bool = True
    enable_balancing: bool = False
    balance_interval: Optional[int] = None

    # Mapping and output.
    mapping: MappingSource2D = field(default_factory=MappingSource2D)
    probes: Tuple[ProbePoint2D, ...] = field(default_factory=tuple)
    output_fields: Tuple[str, ...] = ("p", "rho", "T", "U", "alpha.c4")

    # Display-only state. It is persisted but never consumed by generation.
    mirrored_view: bool = True
    show_mesh: bool = False
    show_probes: bool = True
    log_scale: bool = False
