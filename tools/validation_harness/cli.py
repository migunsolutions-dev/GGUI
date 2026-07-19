#!/usr/bin/env python3
"""CLI for the 3D validation harness (infrastructure only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.validation_harness.collect import collect_case_metrics  # noqa: E402
from tools.validation_harness.compare import compare_to_reference, evaluate_rules  # noqa: E402
from tools.validation_harness.models import MetricRule  # noqa: E402
from tools.validation_harness.report import write_experiment_reports  # noqa: E402


def cmd_collect(args: argparse.Namespace) -> int:
    records = []
    if args.case:
        records.append(
            collect_case_metrics(
                Path(args.case),
                experiment_id=args.experiment_id,
                track=args.track,
            )
        )
    elif args.cases_dir:
        cases_dir = Path(args.cases_dir)
        for sub in sorted(cases_dir.iterdir()):
            if not sub.is_dir():
                continue
            if (sub / "system").is_dir() or (sub / "case_init_mode.json").is_file():
                records.append(
                    collect_case_metrics(sub, experiment_id=args.experiment_id, track=args.track)
                )
    else:
        print("Provide --case or --cases-dir", file=sys.stderr)
        return 2
    if args.out:
        out = Path(args.out)
    elif args.cases_dir:
        out = Path(args.cases_dir) / "harness_metrics.json"
    else:
        out = Path(args.case) / "harness_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": args.experiment_id or "adhoc",
        "track": args.track or "",
        "cases": [r.to_dict() for r in records],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} case(s) -> {out}")
    if args.report_dir:
        write_experiment_reports(
            Path(args.report_dir),
            args.experiment_id or "adhoc",
            records,
            description=args.description or "",
            track=args.track or "",
        )
        print(f"Reports -> {args.report_dir}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    metrics_path = Path(args.metrics)
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in data.get("cases", [])}

    from tools.validation_harness.collect import case_metrics_from_dict

    ref = case_metrics_from_dict(by_id[args.reference])
    metric_list = [m.strip() for m in args.metrics_list.split(",") if m.strip()]
    results = []
    for cid in args.cases.split(","):
        cid = cid.strip()
        if cid == args.reference or cid not in by_id:
            continue
        results.extend(
            compare_to_reference(case_metrics_from_dict(by_id[cid]), ref, metric_list)
        )
    out = Path(args.out) if args.out else metrics_path.parent / "comparisons.json"
    out.write_text(json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} comparison(s) -> {out}")
    return 0


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description="3D startup mesh validation harness")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect", help="Collect metrics from case directory(ies)")
    c.add_argument("--cases-dir", help="Parent directory containing case subfolders")
    c.add_argument("--case", help="Single case directory")
    c.add_argument("--out", help="Output JSON path")
    c.add_argument("--experiment-id", default="adhoc")
    c.add_argument("--track", default="", help="A or B")
    c.add_argument("--description", default="")
    c.add_argument("--report-dir", help="Also write CSV + Markdown reports")
    c.set_defaults(func=cmd_collect)

    cmp = sub.add_parser("compare", help="Compare cases from a metrics JSON file")
    cmp.add_argument("--metrics", required=True)
    cmp.add_argument("--reference", required=True)
    cmp.add_argument("--cases", required=True, help="Comma-separated case ids")
    cmp.add_argument("--metrics-list", required=True, help="Comma-separated dot paths")
    cmp.add_argument("--out")
    cmp.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
