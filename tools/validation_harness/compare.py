"""Compare case metrics and evaluate pass/fail rules (thresholds supplied by caller)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from tools.validation_harness.io_case import get_nested
from tools.validation_harness.models import CaseMetrics, ComparisonResult, MetricRule


def metrics_as_dict(record: CaseMetrics) -> dict:
    return record.to_dict()


def _resolve_metric(record: CaseMetrics, dot_path: str) -> Any:
    d = metrics_as_dict(record)
    return get_nested(d, dot_path)


def compare_to_reference(
    case: CaseMetrics,
    reference: CaseMetrics,
    metrics: List[str],
) -> List[ComparisonResult]:
    """Relative/absolute differences for named dot-path metrics (no pass/fail)."""
    out: List[ComparisonResult] = []
    for m in metrics:
        rv = _resolve_metric(reference, m)
        cv = _resolve_metric(case, m)
        rel = abs_diff = None
        passed = None
        try:
            if rv is not None and cv is not None:
                rf, cf = float(rv), float(cv)
                abs_diff = cf - rf
                rel = abs_diff / rf if abs(rf) > 1e-30 else None
        except (TypeError, ValueError):
            pass
        out.append(
            ComparisonResult(
                case_id=case.case_id,
                reference_id=reference.case_id,
                metric=m,
                reference_value=_num(rv),
                case_value=_num(cv),
                relative_diff=rel,
                absolute_diff=abs_diff,
                passed=passed,
            )
        )
    return out


def evaluate_rules(
    record: CaseMetrics,
    rules: List[MetricRule],
    *,
    reference: Optional[CaseMetrics] = None,
) -> List[ComparisonResult]:
    """Evaluate metric rules; reference required for within_pct_of_ref / within_abs_of_ref."""
    out: List[ComparisonResult] = []
    for rule in rules:
        cv = _resolve_metric(record, rule.metric)
        rv = _resolve_metric(reference, rule.metric) if reference else None
        passed: Optional[bool] = None
        note = None
        try:
            cf = float(cv) if cv is not None else None
        except (TypeError, ValueError):
            cf = None
        if cf is None:
            passed = False
            note = "metric missing"
        elif rule.op == "gte":
            passed = cf >= float(rule.value)
        elif rule.op == "lte":
            passed = cf <= float(rule.value)
        elif rule.op == "eq":
            passed = abs(cf - float(rule.value)) < 1e-9
        elif rule.op in ("within_pct_of_ref", "within_abs_of_ref"):
            if reference is None or rv is None:
                passed = False
                note = "reference required"
            else:
                rf = float(rv)
                if rule.op == "within_pct_of_ref":
                    tol = abs(rf) * float(rule.value)
                    passed = abs(cf - rf) <= tol
                else:
                    passed = abs(cf - rf) <= float(rule.value)
        else:
            passed = None
            note = f"unknown op {rule.op}"

        rel = abs_diff = None
        if rv is not None and cf is not None:
            try:
                rf = float(rv)
                abs_diff = cf - rf
                rel = abs_diff / rf if abs(rf) > 1e-30 else None
            except (TypeError, ValueError):
                pass

        out.append(
            ComparisonResult(
                case_id=record.case_id,
                reference_id=reference.case_id if reference else "",
                metric=rule.metric,
                reference_value=_num(rv),
                case_value=_num(cf),
                relative_diff=rel,
                absolute_diff=abs_diff,
                passed=passed,
                rule=rule.label or f"{rule.op} {rule.value}",
                note=note,
            )
        )
    return out


def _num(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
