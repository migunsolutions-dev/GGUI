"""Display formatting. Internal Validation calculations stay SI."""
from __future__ import annotations

from typing import Optional

from validation.metrics import is_finite_number

NA = "N/A"


def pa_to_kpa(value_pa: float) -> float:
    return float(value_pa) / 1000.0


def kpa_to_pa(value_kpa: float) -> float:
    return float(value_kpa) * 1000.0


def s_to_ms(value_s: float) -> float:
    return float(value_s) * 1000.0


def ms_to_s(value_ms: float) -> float:
    return float(value_ms) / 1000.0


def pa_s_to_kpa_ms(value_pa_s: float) -> float:
    return float(value_pa_s)  # 1 Pa·s = 1 kPa·ms


def kpa_ms_to_pa_s(value_kpa_ms: float) -> float:
    return float(value_kpa_ms)


def fmt(value: Optional[float], *, digits: int = 4, suffix: str = "") -> str:
    if not is_finite_number(value):
        return NA
    number = float(value)
    text = f"{number:.{digits}g}"
    return f"{text} {suffix}".strip() if suffix else text
