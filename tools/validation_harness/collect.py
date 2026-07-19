"""Unified per-case metric collection."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.validation_harness.io_case import load_case_init_mode
from tools.validation_harness.metrics_blast import collect_blast_metrics
from tools.validation_harness.metrics_runtime import collect_runtime_metrics
from tools.validation_harness.metrics_startup import collect_startup_metrics
from tools.validation_harness.models import CaseMetrics


def collect_case_metrics(
    case_dir: Path,
    case_id: Optional[str] = None,
    *,
    experiment_id: Optional[str] = None,
    track: Optional[str] = None,
    p_atm: float = 101325.0,
    arrival_factor: float = 1.5,
) -> CaseMetrics:
    """
    Collect startup, runtime, and blast metrics for one case directory.
    Read-only; does not modify the case or regenerate dictionaries.
    """
    case_dir = Path(case_dir).resolve()
    cid = case_id or case_dir.name
    errors: List[str] = []
    mode: Dict[str, Any] = {}
    try:
        mode = load_case_init_mode(case_dir)
        if not mode:
            errors.append("case_init_mode.json missing or empty")
    except Exception as exc:
        errors.append(f"case_init_mode load failed: {exc}")

    try:
        startup = collect_startup_metrics(case_dir, mode)
    except Exception as exc:
        errors.append(f"startup metrics failed: {exc}")
        startup = collect_startup_metrics(case_dir, {})

    try:
        runtime = collect_runtime_metrics(case_dir, mode)
    except Exception as exc:
        errors.append(f"runtime metrics failed: {exc}")
        runtime = collect_runtime_metrics(case_dir, {})

    try:
        blast = collect_blast_metrics(
            case_dir, p_atm=p_atm, arrival_factor=arrival_factor, mode=mode
        )
    except Exception as exc:
        errors.append(f"blast metrics failed: {exc}")
        blast = collect_blast_metrics(case_dir)

    tags: List[str] = []
    smm = mode.get("startup_mesh_metadata") or {}
    vp = smm.get("validation_profile") or {}
    if isinstance(vp.get("tags"), list):
        tags = list(vp["tags"])

    return CaseMetrics(
        case_id=cid,
        case_dir=str(case_dir),
        experiment_id=experiment_id,
        track=track,
        collected_at_utc=datetime.now(timezone.utc).isoformat(),
        startup=startup,
        runtime=runtime,
        blast=blast,
        validation_profile_tags=tags,
        case_init_mode=mode,
        errors=errors,
    )


def collect_many(
    case_dirs: Dict[str, Path],
    **kwargs,
) -> List[CaseMetrics]:
    return [
        collect_case_metrics(path, case_id=cid, **kwargs)
        for cid, path in sorted(case_dirs.items())
    ]


def case_metrics_from_dict(d: dict) -> CaseMetrics:
    """Rehydrate CaseMetrics from harness JSON snapshot."""
    from tools.validation_harness.models import BlastMetrics, ProbeMetrics, RuntimeMetrics, StartupMetrics

    def _probes(raw: list) -> list:
        out = []
        for p in raw or []:
            if isinstance(p, dict):
                out.append(ProbeMetrics(**{k: v for k, v in p.items() if k in ProbeMetrics.__dataclass_fields__}))
        return out

    st = d.get("startup") or {}
    rt = d.get("runtime") or {}
    bl = d.get("blast") or {}
    return CaseMetrics(
        case_id=d.get("case_id", "?"),
        case_dir=d.get("case_dir", ""),
        experiment_id=d.get("experiment_id"),
        track=d.get("track"),
        collected_at_utc=d.get("collected_at_utc"),
        startup=StartupMetrics(**{k: v for k, v in st.items() if k in StartupMetrics.__dataclass_fields__}),
        runtime=RuntimeMetrics(**{k: v for k, v in rt.items() if k in RuntimeMetrics.__dataclass_fields__}),
        blast=BlastMetrics(
            probes=_probes(bl.get("probes")),
            arrival_factor=bl.get("arrival_factor"),
            extra=bl.get("extra") or {},
        ),
        validation_profile_tags=list(d.get("validation_profile_tags") or []),
        case_init_mode=dict(d.get("case_init_mode") or {}),
        errors=list(d.get("errors") or []),
    )
