from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

# Provenance for 3D optional fields: LOADED=from case, USER=user edited, UNSET=not in case (do not override)
ProvenanceMap = Dict[str, str]

Vec3 = Tuple[float, float, float]

@dataclass(frozen=True)
class RecommendedParams1D:
    r_min: float
    ignition_point: Vec3
    ignition_radius: float
    dt0: float
    maxCo: float
    maxDeltaT: float

@dataclass(frozen=True)
class GenerationResult1D:
    case_dir: str
    log_path: str
    charge_radius: float
    rec: RecommendedParams1D

@dataclass(frozen=True)
class CaseInputs1D:
    radius: float
    cell_size: float
    p_atm: float
    t_atm: float
    mass_kg: float
    rho_charge: float
    energy_j_per_kg: float
    material_props: Dict[str, Any]
    max_cfl: float
    end_time_s: float
    write_interval_s: float = 1e-5
    n_probes: int = 1000
    probe_write_interval_steps: int = 100
    wedge_angle_deg: float = 5.0
    cone_half_angle_deg: float = 12.0
    axis_epsilon: float = 1e-3

@dataclass(frozen=True)
class ObstacleData:
    """Data for a single STL obstacle"""
    stl_path: str
    name: str
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    refinement_level: int = 1

@dataclass(frozen=True)
class CaseInputs3D:
    # --- Geometry & Grid ---
    min_point: Vec3
    max_point: Vec3
    cell_size: float
    
    # --- Charge ---
    charge_center: Vec3
    charge_shape: str
    mass_kg: float
    
    # --- Cylinder Params ---
    cylinder_radius: float
    cylinder_axis: str

    # --- Material ---
    material_name: str
    rho_charge: float
    energy_j_per_kg: float
    
    # --- Physics ---
    p_atm: float
    t_atm: float
    
    # --- Simulation Control ---
    end_time_s: float
    delta_t: float
    write_interval_steps: int
    cores: int
    cfl_value: float = 0.5
    
    # --- Obstacles ---
    obstacles: list[ObstacleData] = field(default_factory=list)
    
    # --- Boundaries ---
    boundaries: Dict[str, str] = field(default_factory=dict)
    
    # --- Extra ---
    material_props: Dict[str, Any] = None

    # --- Mesh Refinement ---
    enable_local_refinement: bool = True  # legacy: when loading old state, maps to both enable_dyn_refine and enable_obstacle_refine
    refine_min: int = 2
    refine_max: int = 3
    # Dynamic (AMR): written to dynamicMeshDict only when enable_dyn_refine
    enable_dyn_refine: Optional[bool] = None   # None = use enable_local_refinement for backward compat
    dyn_refine_min: Optional[int] = None      # legacy load compatibility; not generated
    dyn_refine_max: Optional[int] = 1         # runtime dynamicMeshDict maxRefinement
    # Obstacle surface (snappy): refinementSurfaces levels only when enable_obstacle_refine
    enable_obstacle_refine: Optional[bool] = None  # None = use enable_local_refinement for backward compat
    obstacle_refine_min: Optional[int] = None     # None = use refine_min
    obstacle_refine_max: Optional[int] = None     # None = use refine_max
    # Manual seed level storage. Auto mode computes level via charge_seed_plan (do not overload 0).
    charge_refinement_level: int = 0
    # Explicit seed mode: "Auto" | "Manual" | "Off" (new cases default Auto).
    charge_seed_mode: str = "Auto"
    charge_seed_target_cells: int = 8   # Auto: cells across smallest charge dimension
    charge_seed_min_cells: int = 6      # Auto: minimum acceptable cells across d_min
    charge_seed_max_level: int = 5      # Auto: maximum automatic seed level
    # --- Charge capture (setRefinedFields backup { radius ... } only; not snappy transition) ---
    charge_capture_mode: str = "auto"  # "auto" | "manual"
    charge_capture_factor: float = 1.0  # auto: multiplier on 0.5*sqrt(dx²+dy²+dz²) term
    charge_capture_radius: Optional[float] = None  # manual: exact radius [m]; no hidden minimum
    # Legacy keys (loaders / old scripts): still honored in charge_capture.resolve_charge_capture_radius_m
    charge_backup_radius_factor: float = 1.0  # deprecated; do not use for new cases
    charge_backup_radius_override: Optional[float] = None  # deprecated alias for manual capture radius
    charge_backup_length_override: Optional[float] = None  # absolute backup axial length for cylinder setRefinedFields
    outside_extent: Optional[float] = None    # [m] thickness of outside pre-refinement region; None/0 = use bubble_radius_factor
    
    # --- Charge Geometry (optional, with defaults) ---
    charge_aspect: float = 2.5  # L/D ratio (Sphere/Cylinder only)
    charge_length: float = 0.0  # explicit length (Cylinder/Cuboid)
    charge_width: float = 0.0   # explicit width (Cuboid)
    charge_height: float = 0.0  # explicit height (Cuboid)

    # --- controlDict write options ---
    write_control_type: str = "timeStep"  # "timeStep" | "adjustableRunTime"
    write_interval_time: float = 5e-5  # seconds; used when write_control_type == "adjustableRunTime"
    cycle_write: int = 0  # cycleWrite in controlDict (0 = off)

    # --- Remap from pre-cursor (radial remap, Autodyn-style) ---
    remap_enabled: bool = False
    remap_post_detonation: bool = False     # UNUSED: Removed from UI, always False. Reserved for future.
    remap_source_type: str = "1D"          # UNUSED: Always "1D" (hardcoded). Reserved for future 2D remap support.
    remap_case_path: str = ""              # Path to source case root directory
    remap_origin: Vec3 = (0.0, 0.0, 0.0)   # Origin (x,y,z) for radial mapping
    remap_time_mode: str = "latest"        # "latest" | "specific"
    remap_specific_time: str = "1e-4"      # e.g. "1e-4" when remap_time_mode == "specific"

    # --- Initiation (3D detonation) ---
    initiation_point: Optional[Vec3] = None  # if None or ignition_mode "Center of Charge", use charge_center
    ignition_mode: str = "Center of Charge"   # "Center of Charge" | "Manual" (manual uses initiation_point)
    use_seed_bubble: bool = True             # if True and estimated charge cells low, run topoSet + refineMesh bubble
    # Snappy outer transition + topoSet seed sphere radius = R_charge * this (independent of charge capture)
    bubble_radius_factor: float = 1.5

    # --- setFields/setRefinedFields ---
    # building3D-style startup buffer (setFieldsDict only — not runtime dynamicMeshDict).
    buffer_layers: int = 5

    # --- Run mode (Allrun + controlDict speed-vs-verbosity tradeoffs) ---
    # Defaults are tuned for FAST runs: skip optional post-processing and verbose
    # stage verification. Enable explicitly when debugging or when impulse/
    # overpressure/fieldMinMax fields are required for downstream analysis.
    enable_post_processing: bool = False     # write functions { impulse; overpressure; fieldMinMax; } in controlDict
    fast_run_mode: bool = True               # skip stage_check/log.stageVerification/checkMesh/check_charge_region/check_internal_patch in Allrun

    # --- Charge outer refinement (snappyHexMesh refinement region); expert/legacy only ---
    # New cases: Off. Legacy None on load is migrated to True in project_io.
    charge_outer_refine_enable: Optional[bool] = False
    # Canonical mode-inside level (second tuple value). Legacy min/max kept for migration only.
    charge_outer_refine_level: Optional[int] = None
    charge_outer_refine_min: Optional[int] = None      # legacy load only; not a true "min level"
    charge_outer_refine_max: Optional[int] = None      # legacy load only
    # snappy mode: "inside" | "distance" (preserved on load; never convert distance→inside)
    charge_outer_mode: Optional[str] = None
    # For mode distance: list of (distance_m, level) pairs; lossless round-trip
    charge_outer_distance_levels: Optional[list] = None
    # Parsed searchable* geometry for chargeRefineOuter (sphere/cylinder/box params)
    charge_outer_geometry: Optional[dict] = None
    # Unsupported outer config preserved verbatim; blocks regeneration if set with warning
    charge_outer_raw_refinement: Optional[str] = None
    # Global snappy nCellsBetweenLevels (not outer-extent metres). UI: "Snappy cells between levels".
    transition_cells: int = 2
    # Set when a legacy auto outside_extent was baked to an explicit metres value on load.
    charge_outer_legacy_migration_warning: Optional[str] = None

    # --- dynamicMeshDict advanced (building3D-style; used when enable_dyn_refine) ---
    refine_interval: int = 3
    lower_refine_threshold: float = 0.1
    unrefine_threshold: float = 0.1          # unrefineLevel in dict (building3D default 0.1)
    n_buffer_layers_dynamic: int = 2
    enable_balancing: bool = False
    dynamic_max_cells: int = 200000000
    refine_indicator_field: str = "densityGradient"  # densityGradient | scaledDelta_p (p) | scaledDelta_rho (rho)
    begin_unrefine: Optional[float] = None  # beginUnrefine in dynamicMeshDict
    upper_refine_level: Optional[float] = None  # upperRefineLevel
    upper_unrefine_level: Optional[float] = None  # upperUnrefineLevel
    balance_interval: Optional[int] = None  # loadBalance { balanceInterval ... }
    # --- Obstacle/snappy advanced (surfaceFeaturesDict + snappyHexMeshDict) ---
    obstacle_feature_angle: int = 120       # includedAngle (building3D: 120)
    obstacle_cells_between_levels: int = 2  # nCellsBetweenLevels (building3D: 2)
    obstacle_snap_iter: int = 100           # nSolveIter (building3D: 100)
    obstacle_feature_snap_iter: int = 15    # nFeatureSnapIter (building3D: 15)
    # --- Ignition (non-remap); None = auto from charge ---
    ignition_radius: Optional[float] = None   # user override; effective = max(this, 0.5*smallest_cell)

    # --- Geometry & Mesh Quality (snappyHexMeshDict / surfaceFeaturesDict) ---
    mesh_included_angle: Optional[int] = None          # surfaceFeatures includedAngle (deg)
    mesh_n_smooth_patch: Optional[int] = None          # snapControls nSmoothPatch
    mesh_snap_tolerance: Optional[float] = None         # snapControls tolerance
    mesh_n_solve_iter: Optional[int] = None             # snapControls nSolveIter
    mesh_n_relax_iter: Optional[int] = None            # snapControls nRelaxIter
    mesh_n_feature_snap_iter: Optional[int] = None     # snapControls nFeatureSnapIter
    mesh_explicit_feature_snap: Optional[bool] = None  # snapControls explicitFeatureSnap
    mesh_implicit_feature_snap: Optional[bool] = None  # snapControls implicitFeatureSnap
    mesh_multi_region_feature_snap: Optional[bool] = None  # snapControls multiRegionFeatureSnap
    mesh_n_cells_between_levels: Optional[int] = None  # castellatedMeshControls nCellsBetweenLevels
    mesh_resolve_feature_angle: Optional[int] = None   # castellatedMeshControls resolveFeatureAngle (deg)
    mesh_max_non_ortho: Optional[float] = None         # meshQualityControls maxNonOrtho
    mesh_max_boundary_skewness: Optional[float] = None
    mesh_max_internal_skewness: Optional[float] = None
    mesh_max_concave: Optional[float] = None
    mesh_min_vol: Optional[float] = None
    mesh_min_tet_quality: Optional[float] = None
    mesh_min_twist: Optional[float] = None
    mesh_min_determinant: Optional[float] = None
    mesh_min_face_weight: Optional[float] = None
    mesh_min_vol_ratio: Optional[float] = None
    mesh_n_smooth_scale: Optional[int] = None
    mesh_error_reduction: Optional[float] = None
    mesh_relaxed_max_non_ortho: Optional[float] = None  # relaxed { maxNonOrtho }

    # --- Charge: EOS / activation / thermo (phaseProperties) ---
    eos_model: Optional[str] = None           # equationOfState model name (products; e.g. JWL)
    activation_model_ui: Optional[str] = None  # activationModel (pressureBased / none)
    thermo_model: Optional[str] = None        # thermo model (eConst / ePolynomial)
    thermo_model_air: Optional[str] = None   # air thermodynamics type (e.g. eConst)

    # --- Decomposition (decomposeParDict) ---
    decomposition_method: Optional[str] = None   # method (scotch / simple / hierarchical / manual)
    decomposition_simple_n: Optional[Tuple[int, int, int]] = None  # simpleCoeffs n (n1 n2 n3)
    decomposition_simple_delta: Optional[float] = None            # simpleCoeffs delta

    # Provenance per optional key: "LOADED" | "USER" | "UNSET". Only write/activate when LOADED or USER.
    provenance: ProvenanceMap = field(default_factory=dict)

    estimated_charge_cells: float = 0.0

CaseInputs = Union[CaseInputs1D, CaseInputs3D]