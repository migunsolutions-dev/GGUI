"""Normal and regular-oblique Rankine-Hugoniot relations (ideal gas).

Textbook gas dynamics; not a blast-reference table. Pre-shock state must be
the actual ambient (or sampled pre-shock) state. Velocity comparisons use the
shock-normal component only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from validation.metrics import is_finite_number

AIR_GAMMA = 1.4


@dataclass(frozen=True)
class NormalShockState:
    gamma: float
    p1: float
    rho1: float
    t1: Optional[float]
    shock_speed: float
    mach: Optional[float]
    p2: Optional[float]
    rho2: Optional[float]
    u2_normal: Optional[float]
    t2: Optional[float]
    unavailable_reason: str = ""


def _sound_speed(gamma: float, p: float, rho: float) -> Optional[float]:
    if p <= 0.0 or rho <= 0.0:
        return None
    return math.sqrt(gamma * p / rho)


def normal_shock(
    *,
    shock_speed: float,
    p1: float,
    rho1: float,
    t1: Optional[float] = None,
    gamma: float = AIR_GAMMA,
    pre_shock_normal_velocity: float = 0.0,
) -> NormalShockState:
    """RH jump for a shock moving at ``shock_speed`` into gas with velocity ``u1``."""
    reason = ""
    if not all(is_finite_number(v) for v in (shock_speed, p1, rho1, gamma)):
        reason = "Pre-shock state or shock speed is non-finite."
    elif float(p1) <= 0.0 or float(rho1) <= 0.0 or float(gamma) <= 1.0:
        reason = "Pre-shock p, rho, and gamma must be physical."
    if reason:
        return NormalShockState(
            gamma=float(gamma) if is_finite_number(gamma) else AIR_GAMMA,
            p1=float(p1) if is_finite_number(p1) else 0.0,
            rho1=float(rho1) if is_finite_number(rho1) else 0.0,
            t1=float(t1) if is_finite_number(t1) else None,
            shock_speed=float(shock_speed) if is_finite_number(shock_speed) else 0.0,
            mach=None,
            p2=None,
            rho2=None,
            u2_normal=None,
            t2=None,
            unavailable_reason=reason,
        )
    g = float(gamma)
    a1 = _sound_speed(g, float(p1), float(rho1))
    u_rel = float(shock_speed) - float(pre_shock_normal_velocity)
    mach = abs(u_rel) / a1 if a1 and a1 > 0.0 else None
    if mach is None or mach <= 1.0:
        return NormalShockState(
            gamma=g,
            p1=float(p1),
            rho1=float(rho1),
            t1=float(t1) if is_finite_number(t1) else None,
            shock_speed=float(shock_speed),
            mach=mach,
            p2=None,
            rho2=None,
            u2_normal=None,
            t2=None,
            unavailable_reason="Shock Mach number is unavailable or not greater than 1.",
        )
    m2 = mach * mach
    p2 = float(p1) * (1.0 + 2.0 * g / (g - 1.0) * (m2 - 1.0) / ((g + 1.0) / (g - 1.0)))
    # Standard: p2/p1 = (2*gamma*M^2 - (gamma-1)) / (gamma+1)
    p2 = float(p1) * (2.0 * g * m2 - (g - 1.0)) / (g + 1.0)
    rho2 = float(rho1) * ((g + 1.0) * m2) / ((g - 1.0) * m2 + 2.0)
    u2_rel = u_rel * float(rho1) / rho2
    u2_lab = float(shock_speed) - u2_rel
    t2 = None
    if is_finite_number(t1) and float(t1) > 0.0:
        t2 = float(t1) * p2 / float(p1) * float(rho1) / rho2
    return NormalShockState(
        gamma=g,
        p1=float(p1),
        rho1=float(rho1),
        t1=float(t1) if is_finite_number(t1) else None,
        shock_speed=float(shock_speed),
        mach=mach,
        p2=p2,
        rho2=rho2,
        u2_normal=u2_lab,
        t2=t2,
    )


def normal_component(velocity: Tuple[float, float, float], normal: Tuple[float, float, float]) -> Optional[float]:
    nx, ny, nz = (float(normal[0]), float(normal[1]), float(normal[2]))
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag <= 0.0:
        return None
    nx, ny, nz = nx / mag, ny / mag, nz / mag
    return float(velocity[0]) * nx + float(velocity[1]) * ny + float(velocity[2]) * nz


@dataclass(frozen=True)
class ObliqueShock:
    incidence_deg: float
    mach1: float
    beta_deg: Optional[float]
    theta_deg: Optional[float]
    p2_over_p1: Optional[float]
    unavailable_reason: str = ""


def regular_oblique_shock(
    *,
    mach1: float,
    theta_deg: float,
    gamma: float = AIR_GAMMA,
) -> ObliqueShock:
    """Wave angle beta from the theta-beta-M relation (weak regular solution)."""
    if not is_finite_number(mach1) or not is_finite_number(theta_deg) or float(mach1) <= 1.0:
        return ObliqueShock(
            incidence_deg=float(theta_deg) if is_finite_number(theta_deg) else 0.0,
            mach1=float(mach1) if is_finite_number(mach1) else 0.0,
            beta_deg=None,
            theta_deg=float(theta_deg) if is_finite_number(theta_deg) else None,
            p2_over_p1=None,
            unavailable_reason="Regular oblique shock requires M1 > 1 and a finite deflection angle.",
        )
    theta = math.radians(float(theta_deg))
    g = float(gamma)
    # Scan beta from mu to 90 deg for the weak root.
    mu = math.asin(min(1.0, 1.0 / float(mach1)))
    best = None
    m2 = float(mach1) ** 2
    for i in range(1, 400):
        beta = mu + (math.pi / 2.0 - mu) * i / 400.0
        sb = math.sin(beta)
        num = 2.0 * (sb * sb - 1.0 / m2) / math.tan(beta)
        den = g + math.cos(2.0 * beta) + 2.0 / m2
        if den == 0.0:
            continue
        th = math.atan(num / den)
        err = abs(th - theta)
        if best is None or err < best[0]:
            mn = float(mach1) * sb
            p2p1 = (2.0 * g * mn * mn - (g - 1.0)) / (g + 1.0)
            best = (err, beta, th, p2p1)
    if best is None or best[0] > math.radians(2.0):
        return ObliqueShock(
            incidence_deg=float(theta_deg),
            mach1=float(mach1),
            beta_deg=None,
            theta_deg=float(theta_deg),
            p2_over_p1=None,
            unavailable_reason="No regular-reflection (weak) theta-beta-M root found.",
        )
    return ObliqueShock(
        incidence_deg=math.degrees(best[1]),
        mach1=float(mach1),
        beta_deg=math.degrees(best[1]),
        theta_deg=math.degrees(best[2]),
        p2_over_p1=best[3],
    )
