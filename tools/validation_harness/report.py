"""Human-readable and machine-readable experiment reports."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.validation_harness.models import CaseMetrics, ComparisonResult, ExperimentManifest


def write_experiment_reports(
    out_dir: Path,
    experiment_id: str,
    records: List[CaseMetrics],
    *,
    comparisons: Optional[List[ComparisonResult]] = None,
    rule_results: Optional[List[ComparisonResult]] = None,
    manifest: Optional[ExperimentManifest] = None,
    description: str = "",
    track: str = "",
) -> Dict[str, Path]:
    """Write JSON, CSV summary, and Markdown report. Returns paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    metrics_json = out_dir / f"{experiment_id}_metrics.json"
    payload = {
        "experiment_id": experiment_id,
        "description": description,
        "track": track,
        "cases": [r.to_dict() for r in records],
    }
    metrics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["metrics_json"] = metrics_json

    if comparisons:
        cmp_path = out_dir / f"{experiment_id}_comparisons.json"
        cmp_path.write_text(
            json.dumps([c.__dict__ for c in comparisons], indent=2),
            encoding="utf-8",
        )
        paths["comparisons_json"] = cmp_path

    if rule_results:
        rules_path = out_dir / f"{experiment_id}_rule_results.json"
        rules_path.write_text(
            json.dumps([r.__dict__ for r in rule_results], indent=2),
            encoding="utf-8",
        )
        paths["rule_results_json"] = rules_path

    csv_path = out_dir / f"{experiment_id}_summary.csv"
    _write_summary_csv(csv_path, records)
    paths["summary_csv"] = csv_path

    md_path = out_dir / f"{experiment_id}_report.md"
    md_path.write_text(
        _markdown_report(
            experiment_id,
            description,
            track,
            records,
            comparisons=comparisons,
            rule_results=rule_results,
        ),
        encoding="utf-8",
    )
    paths["report_md"] = md_path

    if manifest:
        man_path = out_dir / f"{experiment_id}_manifest.json"
        man_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        paths["manifest_json"] = man_path

    return paths


def _write_summary_csv(path: Path, records: List[CaseMetrics]) -> None:
    rows: List[Dict[str, Any]] = []
    for r in records:
        s, rt, b = r.startup, r.runtime, r.blast
        p0 = b.probes[0] if b.probes else None
        rows.append(
            {
                "case_id": r.case_id,
                "nominal_mass_kg": s.nominal_mass_kg,
                "captured_mass_kg": s.captured_mass_kg,
                "mass_ratio": s.mass_ratio,
                "backup_radius_m": s.backup_radius_m,
                "backup_to_charge_ratio": s.backup_to_charge_ratio,
                "seed_level": s.seed_level_effective,
                "charge_cells": s.charge_cells,
                "startup_cell_count": s.startup_cell_count,
                "peak_cell_count": rt.peak_cell_count,
                "final_cell_count": rt.final_cell_count,
                "release_ratio": rt.release_ratio,
                "amr_indicator": rt.amr_indicator,
                "max_refinement": rt.max_refinement,
                "shock_arrival_s": p0.shock_arrival_time_s if p0 else None,
                "peak_p_bar": p0.peak_overpressure_bar if p0 else None,
                "errors": "; ".join(r.errors),
            }
        )
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _markdown_report(
    experiment_id: str,
    description: str,
    track: str,
    records: List[CaseMetrics],
    *,
    comparisons: Optional[List[ComparisonResult]] = None,
    rule_results: Optional[List[ComparisonResult]] = None,
) -> str:
    lines = [
        f"# Validation report: {experiment_id}",
        "",
        f"**Track:** {track or '—'}",
        f"**Description:** {description or '—'}",
        "",
        "## Cases",
        "",
        "| case | mass_ratio | startup_cells | peak_cells | release_ratio | arrival@p0 | peak_p (bar) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        s, rt, b = r.startup, r.runtime, r.blast
        p0 = b.probes[0] if b.probes else None
        lines.append(
            f"| {r.case_id} | {_fmt(s.mass_ratio)} | {_fmt(s.startup_cell_count)} | "
            f"{_fmt(rt.peak_cell_count)} | {_fmt(rt.release_ratio)} | "
            f"{_fmt(p0.shock_arrival_time_s if p0 else None)} | "
            f"{_fmt(p0.peak_overpressure_bar if p0 else None)} |"
        )
    if rule_results:
        lines.extend(["", "## Rule evaluation", ""])
        for rr in rule_results:
            status = "PASS" if rr.passed else ("FAIL" if rr.passed is False else "—")
            lines.append(
                f"- **{rr.case_id}** `{rr.metric}`: {status} "
                f"(value={_fmt(rr.case_value)}, rule={rr.rule})"
            )
    if comparisons:
        lines.extend(["", "## Comparisons vs reference", ""])
        for c in comparisons:
            lines.append(
                f"- **{c.case_id}** vs **{c.reference_id}** `{c.metric}`: "
                f"Δ={_fmt(c.absolute_diff)} rel={_fmt(c.relative_diff)}"
            )
    lines.append("")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
