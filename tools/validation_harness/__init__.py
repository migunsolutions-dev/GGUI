"""
3D startup / runtime validation harness (infrastructure only).

Does not change GUI defaults, case generation logic, or OpenFOAM dictionaries.
Experiment scripts (EX-1 … EX-5, B1 … B5) should import this package.
"""
from tools.validation_harness.collect import collect_case_metrics
from tools.validation_harness.compare import compare_to_reference, evaluate_rules
from tools.validation_harness.experiment import ExperimentSession
from tools.validation_harness.models import (
    CaseMetrics,
    ComparisonResult,
    ExperimentManifest,
    MetricRule,
)
from tools.validation_harness.report import write_experiment_reports

__all__ = [
    "CaseMetrics",
    "ComparisonResult",
    "ExperimentManifest",
    "MetricRule",
    "ExperimentSession",
    "collect_case_metrics",
    "compare_to_reference",
    "evaluate_rules",
    "write_experiment_reports",
]
