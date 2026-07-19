"""Collect blast-wave probe metrics from postProcessing."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.validation_harness.io_case import load_case_init_mode
from tools.validation_harness.models import BlastMetrics, ProbeMetrics

DEFAULT_P_ATM = 101325.0
DEFAULT_ARRIVAL_FACTOR = 1.5


def parse_probe_pressure_file(pfile: Path) -> Tuple[List[str], List[Tuple[float, List[float]]]]:
    locs: List[str] = []
    rows: List[Tuple[float, List[float]]] = []
    for line in pfile.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#"):
            if "Probe" in line:
                m = re.findall(r"\(([^)]+)\)", line)
                if m:
                    locs.append(m[-1].strip())
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            ps = [float(x) for x in parts[1:]]
            rows.append((t, ps))
        except ValueError:
            continue
    return locs, rows


def collect_blast_metrics(
    case_dir: Path,
    *,
    p_atm: float = DEFAULT_P_ATM,
    arrival_factor: float = DEFAULT_ARRIVAL_FACTOR,
    mode: Optional[Dict[str, Any]] = None,
) -> BlastMetrics:
    pdir = case_dir / "postProcessing" / "probes"
    if not pdir.is_dir():
        return BlastMetrics(arrival_factor=arrival_factor, extra={"probes_found": False})

    pfile = None
    for tdir in sorted(pdir.iterdir()):
        cand = tdir / "p"
        if cand.is_file():
            pfile = cand
            break
    if not pfile:
        return BlastMetrics(arrival_factor=arrival_factor, extra={"probes_found": False})

    locs, rows = parse_probe_pressure_file(pfile)
    threshold = arrival_factor * p_atm
    probes: List[ProbeMetrics] = []
    n_probes = len(rows[0][1]) if rows else 0
    for i in range(n_probes):
        loc = locs[i] if i < len(locs) else f"probe{i}"
        ta = pk = tpk = None
        for t, ps in rows:
            if i >= len(ps):
                continue
            p = ps[i]
            if ta is None and p > threshold:
                ta = t
            if pk is None or p > pk:
                pk = p
                tpk = t
        probes.append(
            ProbeMetrics(
                location=loc,
                shock_arrival_time_s=ta,
                peak_pressure_pa=pk,
                peak_pressure_time_s=tpk,
                peak_overpressure_bar=(pk / 1e5) if pk is not None else None,
            )
        )
    return BlastMetrics(
        probes=probes,
        arrival_factor=arrival_factor,
        extra={"probe_file": str(pfile), "n_samples": len(rows)},
    )
