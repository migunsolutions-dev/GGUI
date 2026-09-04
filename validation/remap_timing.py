"""Physical-time offset for remap validation. No generator imports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from validation.metrics import is_finite_number


@dataclass(frozen=True)
class RemapTiming:
    source_physical_time: Optional[float]
    target_initial_time: Optional[float]
    physical_time_offset: Optional[float]
    source_time_label: str = ""
    target_time_label: str = ""

    def target_physical(self, of_time: Optional[float]) -> Optional[float]:
        if not is_finite_number(of_time):
            return None
        if is_finite_number(self.physical_time_offset):
            return float(of_time) + float(self.physical_time_offset)
        return float(of_time)

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "source_physical_time": self.source_physical_time,
            "target_initial_time": self.target_initial_time,
            "physical_time_offset": self.physical_time_offset,
            "source_time_label": self.source_time_label,
            "target_time_label": self.target_time_label,
        }


def parse_time_label(label: Optional[str]) -> Optional[float]:
    if label is None:
        return None
    try:
        return float(str(label).strip())
    except (TypeError, ValueError):
        return None


def build_remap_timing(
    *,
    source_time_label: Optional[str],
    target_time_label: Optional[str] = "0",
) -> RemapTiming:
    source = parse_time_label(source_time_label)
    target = parse_time_label(target_time_label)
    offset = None
    if is_finite_number(source) and is_finite_number(target):
        offset = float(source) - float(target)
    return RemapTiming(
        source_physical_time=source,
        target_initial_time=target,
        physical_time_offset=offset,
        source_time_label=str(source_time_label or ""),
        target_time_label=str(target_time_label or ""),
    )


def remap_timing_from_mapping(
    *,
    mapping_time: Optional[str],
    mapping_time_mode: Optional[str] = None,
    target_time_label: Optional[str] = "0",
) -> RemapTiming:
    label = mapping_time
    if str(mapping_time_mode or "").strip().lower() == "latest":
        label = mapping_time if mapping_time and mapping_time not in ("latest",) else None
    return build_remap_timing(source_time_label=label, target_time_label=target_time_label)


def physical_times_synchronized(
    source_physical: Optional[float],
    target_of_time: Optional[float],
    *,
    offset: Optional[float] = None,
    tol: float = 1e-12,
) -> bool:
    if not is_finite_number(source_physical) or not is_finite_number(target_of_time):
        return False
    target_physical = float(target_of_time)
    if is_finite_number(offset):
        target_physical += float(offset)
    return abs(float(source_physical) - target_physical) <= float(tol)
