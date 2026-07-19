# 3D Validation Harness

Infrastructure-only package for Track A (EX) and Track B (B) experiments. Does **not** change GUI defaults, `generator_3d` behavior, or OpenFOAM dictionaries.

## Architecture

```
Experiment script (EX-2, B2, …)
        │
        ▼
ExperimentSession  ──register_case / register_generator
        │
        ├── execution.run_init_only / run_allrun  (optional WSL)
        │
        ├── collect.collect_case_metrics  (read-only)
        │        ├── metrics_startup
        │        ├── metrics_runtime  (uses tools.amr_tuning_sweep.parse_blastfoam_log)
        │        └── metrics_blast
        │
        ├── compare.evaluate_rules / compare_to_reference
        │
        └── report.write_experiment_reports
                 ├── {id}_metrics.json
                 ├── {id}_summary.csv
                 ├── {id}_report.md
                 └── {id}_manifest.json
```

## CLI

From repo root:

```bash
python -m tools.validation_harness collect --cases-dir _val_m2 --experiment-id M2 --track A --report-dir _val_m2/harness_reports

python -m tools.validation_harness compare --metrics _val_m2/harness_metrics.json --reference C4new --cases B4 --metrics-list startup.peak_cell_count,runtime.peak_cell_count
```

## Programmatic use

```python
from pathlib import Path
from tools.validation_harness import ExperimentSession, MetricRule, collect_case_metrics

session = ExperimentSession(Path("_experiments/EX-2"), "EX-2", track="A")
session.register_case("sph_02", Path("_experiments/EX-2/sph_02"))
records = session.collect_all()
session.write_reports()

rules = [MetricRule("startup.mass_ratio", "gte", 0.98, label="capture")]
results = session.evaluate(rules)
```

Thresholds and case matrices belong in experiment scripts, not in this package.
