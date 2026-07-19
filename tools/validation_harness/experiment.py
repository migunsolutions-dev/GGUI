"""Experiment session: register cases, collect metrics, write reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tools.validation_harness.collect import collect_case_metrics, collect_many
from tools.validation_harness.compare import compare_to_reference, evaluate_rules
from tools.validation_harness.execution import run_allrun, run_init_only
from tools.validation_harness.models import (
    CaseMetrics,
    ComparisonResult,
    ExperimentManifest,
    MetricRule,
)
from tools.validation_harness.report import write_experiment_reports


class ExperimentSession:
    """
    Orchestrates validation workflows without embedding experiment-specific logic.

    Case generation is injected via ``register_generator`` or manual ``register_case``.
    """

    def __init__(
        self,
        work_root: Path,
        experiment_id: str,
        *,
        description: str = "",
        track: str = "",
    ):
        self.work_root = Path(work_root).resolve()
        self.experiment_id = experiment_id
        self.description = description
        self.track = track
        self.cases: Dict[str, Path] = {}
        self._generator: Optional[Callable[[str], Path]] = None
        self.work_root.mkdir(parents=True, exist_ok=True)

    def register_generator(self, fn: Callable[[str], Path]) -> None:
        """fn(case_id) -> case_dir path (caller runs Generator3D or copies cases)."""
        self._generator = fn

    def register_case(self, case_id: str, case_dir: Path) -> None:
        self.cases[case_id] = Path(case_dir).resolve()

    def generate_case(self, case_id: str) -> Path:
        if not self._generator:
            raise RuntimeError("No generator registered")
        path = self._generator(case_id)
        self.register_case(case_id, path)
        return path

    def run_init_only(self, case_id: str, timeout: int = 3600) -> tuple:
        return run_init_only(self.cases[case_id], timeout=timeout)

    def run_solver(self, case_id: str, timeout: int = 7200) -> tuple:
        return run_allrun(self.cases[case_id], timeout=timeout)

    def collect(self, case_id: str) -> CaseMetrics:
        return collect_case_metrics(
            self.cases[case_id],
            case_id=case_id,
            experiment_id=self.experiment_id,
            track=self.track,
        )

    def collect_all(self) -> List[CaseMetrics]:
        return collect_many(
            self.cases,
            experiment_id=self.experiment_id,
            track=self.track,
        )

    def compare(
        self,
        case_id: str,
        reference_id: str,
        metrics: List[str],
    ) -> List[ComparisonResult]:
        records = {r.case_id: r for r in self.collect_all()}
        return compare_to_reference(
            records[case_id],
            records[reference_id],
            metrics,
        )

    def evaluate(
        self,
        rules: List[MetricRule],
        *,
        reference_id: Optional[str] = None,
    ) -> List[ComparisonResult]:
        records = self.collect_all()
        by_id = {r.case_id: r for r in records}
        ref = by_id.get(reference_id) if reference_id else None
        out: List[ComparisonResult] = []
        for r in records:
            out.extend(evaluate_rules(r, rules, reference=ref))
        return out

    def write_reports(
        self,
        out_dir: Optional[Path] = None,
        *,
        comparisons: Optional[List[ComparisonResult]] = None,
        rule_results: Optional[List[ComparisonResult]] = None,
    ) -> Dict[str, Path]:
        out_dir = out_dir or (self.work_root / "reports")
        records = self.collect_all()
        manifest = ExperimentManifest(
            experiment_id=self.experiment_id,
            description=self.description,
            track=self.track,
            cases={k: str(v) for k, v in self.cases.items()},
        )
        paths = write_experiment_reports(
            out_dir,
            self.experiment_id,
            records,
            comparisons=comparisons,
            rule_results=rule_results,
            manifest=manifest,
            description=self.description,
            track=self.track,
        )
        manifest.metrics_files = [str(p) for p in paths.values()]
        (out_dir / f"{self.experiment_id}_manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )
        return paths

    def save_metrics_snapshot(self, path: Optional[Path] = None) -> Path:
        path = path or (self.work_root / f"{self.experiment_id}_metrics_snapshot.json")
        payload = {
            "experiment_id": self.experiment_id,
            "cases": [r.to_dict() for r in self.collect_all()],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
