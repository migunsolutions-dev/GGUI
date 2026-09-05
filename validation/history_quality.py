"""Peak/impulse completeness so truncated runs are not compared to UFC/KB."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from validation.metrics import is_finite_number
from validation.probes import is_physical_probe_value

ARRIVAL_OVERPRESSURE_PA = 1000.0
MIN_PROBE_SAMPLES = 8
END_TIME_REL_TOL = 1.0e-4
END_TIME_ABS_TOL = 1.0e-9
PEAK_NEAR_END_FRACTION = 0.90
RESIDUAL_POSITIVE_FRACTION = 0.05

STATUS_UNKNOWN = "unknown"
STATUS_TRUNCATED = "truncated"
STATUS_INVALID = "invalid"
STATUS_OK = "ok"


@dataclass(frozen=True)
class HistoryValidity:
    arrival_detected: bool = False
    positive_phase_started: bool = False
    positive_phase_completed: bool = False
    run_reached_end_time: Optional[bool] = None
    probe_complete: bool = False
    truncated: bool = False
    invalid: bool = False
    unknown: bool = False
    comparable: bool = False
    reason: str = ""
    peak_overpressure_pa: Optional[float] = None
    impulse_pa_s: Optional[float] = None
    arrival_time_s: Optional[float] = None
    peak_time_s: Optional[float] = None
    positive_phase_end_time_s: Optional[float] = None
    status: str = STATUS_UNKNOWN

    @property
    def ok(self) -> bool:
        return self.comparable and self.status == STATUS_OK


def _finite_series(
    times: Sequence[float], values: Sequence[float]
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    out_t = []
    out_v = []
    for raw_t, raw_v in zip(times, values):
        if not is_finite_number(raw_t) or not is_physical_probe_value(raw_v):
            continue
        out_t.append(float(raw_t))
        out_v.append(float(raw_v))
    return tuple(out_t), tuple(out_v)


def run_reached_end_time(
    *,
    last_time_s: Optional[float],
    end_time_s: Optional[float],
    reached_end: Optional[bool] = None,
) -> Optional[bool]:
    if reached_end is True:
        return True
    if not is_finite_number(end_time_s):
        return None if reached_end is None else bool(reached_end)
    end = float(end_time_s)
    tol = max(END_TIME_ABS_TOL, abs(end) * END_TIME_REL_TOL)
    if is_finite_number(last_time_s) and float(last_time_s) >= end - tol:
        return True
    if reached_end is False:
        return False
    if reached_end is None:
        return False if is_finite_number(last_time_s) else None
    return bool(reached_end)


def assess_history(
    times: Sequence[float],
    pressure_pa: Sequence[float],
    impulse_pa_s: Optional[Sequence[float]] = None,
    *,
    p_atm: float = 101325.0,
    end_time_s: Optional[float] = None,
    reached_end: Optional[bool] = None,
    arrival_pa: float = ARRIVAL_OVERPRESSURE_PA,
) -> HistoryValidity:
    """Classify a pressure history before any UFC/KB error percentage is computed.

    Peak is the maximum overpressure *after arrival*, not the global maximum
    (which may be the t=0 charge). Impulse for comparison is the sample at
    positive-phase completion, not the last file value.
    """
    t_s, p_s = _finite_series(times, pressure_pa)
    if len(t_s) < 2 or len(p_s) < 2:
        return HistoryValidity(
            unknown=True,
            invalid=True,
            reason="Probe data is not complete enough to assess peak or impulse.",
            status=STATUS_UNKNOWN,
        )
    atm = float(p_atm) if is_finite_number(p_atm) else 101325.0
    over = tuple(v - atm for v in p_s)
    last_t = t_s[-1]
    reached = run_reached_end_time(last_time_s=last_t, end_time_s=end_time_s, reached_end=reached_end)
    probe_complete = len(t_s) >= MIN_PROBE_SAMPLES

    threshold = float(arrival_pa) if is_finite_number(arrival_pa) and float(arrival_pa) > 0.0 else ARRIVAL_OVERPRESSURE_PA
    arrival_i = None
    for i, value in enumerate(over):
        if value >= threshold:
            arrival_i = i
            break
    if arrival_i is None:
        return HistoryValidity(
            arrival_detected=False,
            run_reached_end_time=reached,
            probe_complete=probe_complete,
            invalid=True,
            truncated=reached is False,
            unknown=reached is None,
            reason="Arrival was not detected; UFC/KB comparison is N/A.",
            status=STATUS_INVALID,
        )

    window = over[arrival_i:]
    peak_local = max(range(len(window)), key=lambda j: window[j])
    peak_i = arrival_i + peak_local
    peak_over = over[peak_i]
    started = peak_over > 0.0
    completion_thr = min(0.0, RESIDUAL_POSITIVE_FRACTION * peak_over) if peak_over > 0.0 else 0.0
    complete_i = None
    if started:
        for j in range(peak_i + 1, len(over)):
            if over[j] <= completion_thr:
                complete_i = j
                break
    completed = complete_i is not None
    near_end = peak_i >= int(math.floor(PEAK_NEAR_END_FRACTION * (len(over) - 1)))
    still_positive = over[-1] > max(threshold * 0.1, RESIDUAL_POSITIVE_FRACTION * max(peak_over, 0.0))
    truncated = (not completed) and (near_end or still_positive or reached is False)

    impulse_val = None
    if impulse_pa_s:
        aligned = []
        for raw in impulse_pa_s:
            if is_finite_number(raw):
                aligned.append(float(raw))
        if complete_i is not None and complete_i < len(aligned):
            impulse_val = aligned[complete_i]
        elif completed and aligned:
            impulse_val = aligned[min(len(aligned) - 1, complete_i or (len(aligned) - 1))]

    unknown = reached is None
    invalid = (not started) or (not probe_complete)
    comparable = bool(
        started
        and completed
        and reached is True
        and probe_complete
        and not truncated
        and not unknown
    )
    if not probe_complete:
        reason = "Probe data is not complete enough; UFC/KB comparison is N/A."
        status = STATUS_INVALID
    elif reached is False:
        reason = "Run did not reach configured endTime; UFC/KB comparison is N/A."
        status = STATUS_TRUNCATED
    elif not completed:
        reason = "Positive phase is incomplete; UFC/KB comparison is N/A."
        status = STATUS_TRUNCATED
    elif unknown:
        reason = "Run completion is unknown; UFC/KB comparison is N/A."
        status = STATUS_UNKNOWN
    elif not comparable:
        reason = "History is not valid for a UFC/KB comparison."
        status = STATUS_INVALID
    else:
        reason = ""
        status = STATUS_OK

    return HistoryValidity(
        arrival_detected=True,
        positive_phase_started=started,
        positive_phase_completed=completed,
        run_reached_end_time=reached,
        probe_complete=probe_complete,
        truncated=truncated,
        invalid=invalid and not comparable,
        unknown=unknown,
        comparable=comparable,
        reason=reason,
        peak_overpressure_pa=float(peak_over) if started else None,
        impulse_pa_s=impulse_val if comparable else None,
        arrival_time_s=t_s[arrival_i],
        peak_time_s=t_s[peak_i],
        positive_phase_end_time_s=None if complete_i is None else t_s[complete_i],
        status=status,
    )


def comparable_peak_impulse(validity: HistoryValidity) -> Tuple[Optional[float], Optional[float], str]:
    if not validity.comparable:
        return None, None, validity.reason or "N/A"
    return validity.peak_overpressure_pa, validity.impulse_pa_s, ""
