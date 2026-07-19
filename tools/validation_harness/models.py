"""Data models for validation harness (no OpenFOAM / GUI side effects)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StartupMetrics:
    nominal_mass_kg: Optional[float] = None
    captured_mass_kg: Optional[float] = None
    mass_ratio: Optional[float] = None
    backup_radius_m: Optional[float] = None
    backup_to_charge_ratio: Optional[float] = None
    seed_level_requested: Optional[int] = None
    seed_level_effective: Optional[int] = None
    charge_cells: Optional[int] = None
    startup_cell_count: Optional[int] = None
    base_cell_count_blockmesh: Optional[int] = None
    set_cmd: Optional[str] = None
    charge_refine_outer_enabled: Optional[bool] = None
    cells_across_charge_estimate: Optional[float] = None
    projected_startup_cells_estimate: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeMetrics:
    peak_cell_count: Optional[int] = None
    final_cell_count: Optional[int] = None
    release_ratio: Optional[float] = None
    refine_events: Optional[int] = None
    unrefine_events: Optional[int] = None
    refinement_persistence: Optional[bool] = None
    amr_indicator: Optional[str] = None
    max_refinement: Optional[int] = None
    final_sim_time_s: Optional[float] = None
    wall_execution_time_s: Optional[float] = None
    cell_series: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeMetrics:
    location: str
    shock_arrival_time_s: Optional[float] = None
    peak_pressure_pa: Optional[float] = None
    peak_pressure_time_s: Optional[float] = None
    peak_overpressure_bar: Optional[float] = None


@dataclass
class BlastMetrics:
    probes: List[ProbeMetrics] = field(default_factory=list)
    arrival_factor: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseMetrics:
    case_id: str
    case_dir: str
    experiment_id: Optional[str] = None
    track: Optional[str] = None  # "EX" | "B" | None
    collected_at_utc: Optional[str] = None
    startup: StartupMetrics = field(default_factory=StartupMetrics)
    runtime: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    blast: BlastMetrics = field(default_factory=BlastMetrics)
    validation_profile_tags: List[str] = field(default_factory=list)
    case_init_mode: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricRule:
    """Pass/fail rule; thresholds supplied by experiment scripts, not hard-coded here."""

    metric: str  # dot path, e.g. "startup.mass_ratio"
    op: str  # gte | lte | eq | within_pct_of_ref | within_abs_of_ref
    value: float
    label: Optional[str] = None


@dataclass
class ComparisonResult:
    case_id: str
    reference_id: str
    metric: str
    reference_value: Optional[float]
    case_value: Optional[float]
    relative_diff: Optional[float]
    absolute_diff: Optional[float]
    passed: Optional[bool]
    rule: Optional[str] = None
    note: Optional[str] = None


@dataclass
class ExperimentManifest:
    experiment_id: str
    description: str = ""
    track: str = ""  # "A" | "B" | ""
    cases: Dict[str, str] = field(default_factory=dict)
    metrics_files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
