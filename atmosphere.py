"""EOS-consistent atmosphere shared by 1D/2D/3D (R-091).

BlastFoam idealGas::p = (gamma-1)*rho*e; eConst with Hf=0 gives e=Cv*T.
Thus use the written gamma/Cv, not a molWeight-derived or generic R.
Local solver source verified at 4e6ee07a0c1fc4629ee7206804f4f1fe802ec64c.
Inputs are supplied per dimension; this module contains no shared user state.
"""
from dataclasses import dataclass
from typing import Any
import math

GAMMA_IDEAL_GAS = 1.4
CV_IDEAL_GAS = 718.0

class AtmosphereError(ValueError):
    """Invalid atmosphere for the supported eConst/idealGas pair."""

def _require_positive(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AtmosphereError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise AtmosphereError(f"{name} must be finite and > 0, got {number!r}")
    return number


@dataclass(frozen=True)
class AmbientState:
    """Uniform far-field gas state, derived so it is exactly consistent with the EOS."""

    p_atm: float
    t_atm: float
    gamma: float
    cv: float
    r_specific: float
    rho: float
    e: float


def ambient_state(
    p_atm: float,
    t_atm: float,
    gamma: float = GAMMA_IDEAL_GAS,
    cv: float = CV_IDEAL_GAS,
) -> AmbientState:
    """Ambient density from the EOS rather than a hard-coded 1.225 kg/m^3.

    ``rho = p/(R*T)`` is approximately 1.225 at 101325 Pa / 288 K. Writing a constant
    at any other ambient seeds a spurious starting wave, because blastFoam derives
    ``e`` from the ``p`` and ``rho`` it is given.
    """
    p = _require_positive("p_atm", p_atm)
    t = _require_positive("t_atm", t_atm)
    g = _require_positive("gamma", gamma)
    c = _require_positive("cv", cv)
    if g <= 1.0:
        raise AtmosphereError(f"gamma must be > 1, got {g!r}")
    r_specific = (g - 1.0) * c
    return AmbientState(
        p_atm=p,
        t_atm=t,
        gamma=g,
        cv=c,
        r_specific=r_specific,
        rho=p / (r_specific * t),
        e=c * t,
    )


