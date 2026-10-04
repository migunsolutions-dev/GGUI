"""BirchMurnaghan3 reactant reference energy for production 2D JWL.

blastFoam ``eConst`` stores

    Es = Cv*(T - Tref) + Esref + EOS::E(rho)

``Tref`` defaults to ``Tstd`` and, when ``Esref`` is omitted, ``Esref``
defaults to ``Cv*Tref`` (eConstBlastThermo.C). ``EOS::E(rho0)`` is the
BirchMurnaghan3 integration constant. Removing only that constant leaves

    Es(rho0, T) = Cv*T

at every temperature, including the case initial temperature. Pressure,
sound speed, and bulk modulus do not depend on ``Esref``.

``Tstd`` is the SI value in OpenFOAM ``etc/controlDict`` (298.15 K). GGUI
cases are SI, so that is the ``Tref`` blastFoam uses when the dictionary
does not set ``Tref``.
"""

from __future__ import annotations

import math

TSTD_SI_K = 298.15

# Constants written into the 2D JWL reactant block. Esref is derived from
# these values and the active charge density; it is not a fitted number.
BM3_K0_PA = 8.04e9
BM3_K0_PRIME = 7.97
BM3_PREF_PA = 101298.0
BM3_CV_J_PER_KG_K = 1400.0


def _positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return number


def cold_energy_j_per_kg(
    rho: float,
    rho0: float,
    *,
    k0: float = BM3_K0_PA,
    k0_prime: float = BM3_K0_PRIME,
    p_ref: float = BM3_PREF_PA,
) -> float:
    """BirchMurnaghan3::E(rho). Independent of temperature and Esref."""
    rho_n = _positive("rho", rho)
    rho0_n = _positive("rho0", rho0)
    ratio = rho_n / rho0_n
    return (
        9.0
        / 16.0
        * float(k0)
        * (
            ratio ** (5.0 / 3.0)
            * (
                ratio ** (2.0 / 3.0) * (14.0 - 3.0 * float(k0_prime))
                + 3.0 * float(k0_prime)
                - 16.0
            )
            + ratio ** 3 * (float(k0_prime) - 4.0)
        )
        - float(p_ref)
    ) / max(rho_n, 1e-10)


def reactant_esref(
    rho0: float,
    *,
    k0: float = BM3_K0_PA,
    k0_prime: float = BM3_K0_PRIME,
    p_ref: float = BM3_PREF_PA,
    cv: float = BM3_CV_J_PER_KG_K,
    t_ref: float = TSTD_SI_K,
) -> float:
    """Esref that cancels E(rho0) and keeps the Cv*T sensible energy.

    With the blastFoam default ``Tref``, this is ``Cv*Tref - E(rho0)``.
    """
    rho0_n = _positive("rho0", rho0)
    cv_n = _positive("Cv", cv)
    t_ref_n = float(t_ref)
    if not math.isfinite(t_ref_n) or t_ref_n <= 0.0:
        raise ValueError(f"Tref must be finite and > 0, got {t_ref!r}")
    return cv_n * t_ref_n - cold_energy_j_per_kg(
        rho0_n, rho0_n, k0=k0, k0_prime=k0_prime, p_ref=p_ref
    )


def stored_reactant_energy(
    rho: float,
    temperature: float,
    esref: float,
    *,
    rho0: float,
    k0: float = BM3_K0_PA,
    k0_prime: float = BM3_K0_PRIME,
    p_ref: float = BM3_PREF_PA,
    cv: float = BM3_CV_J_PER_KG_K,
    t_ref: float = TSTD_SI_K,
) -> float:
    """eConst Es(rho, T) for the BirchMurnaghan3 reactant."""
    return (
        float(cv) * (float(temperature) - float(t_ref))
        + float(esref)
        + cold_energy_j_per_kg(rho, rho0, k0=k0, k0_prime=k0_prime, p_ref=p_ref)
    )


def pressure_pa(
    rho: float,
    rho0: float,
    *,
    k0: float = BM3_K0_PA,
    k0_prime: float = BM3_K0_PRIME,
    p_ref: float = BM3_PREF_PA,
) -> float:
    """BirchMurnaghan3::pRhoT. No dependence on energy or Esref."""
    rho_n = _positive("rho", rho)
    rho0_n = _positive("rho0", rho0)
    ratio = rho_n / rho0_n
    return float(p_ref) + 1.5 * float(k0) * (
        ratio ** (7.0 / 3.0) - ratio ** (5.0 / 3.0)
    ) * (1.0 + 0.75 * (float(k0_prime) - 4.0) * (ratio ** (2.0 / 3.0) - 1.0))


def sound_speed_squared(
    rho: float,
    rho0: float,
    *,
    k0: float = BM3_K0_PA,
    k0_prime: float = BM3_K0_PRIME,
) -> float:
    """BirchMurnaghan3::cSqr. No dependence on energy or Esref."""
    rho_n = _positive("rho", rho)
    rho0_n = _positive("rho0", rho0)
    ratio = rho_n / rho0_n
    return (
        float(k0)
        * ratio ** (5.0 / 3.0)
        * (
            6.0 * ratio ** (2.0 / 3.0) * (float(k0_prime) - 4.0) * (ratio ** (2.0 / 3.0) - 1.0)
            + (7.0 * ratio ** (2.0 / 3.0) - 5.0)
            * (3.0 * (float(k0_prime) - 4.0) * (ratio ** (2.0 / 3.0) - 1.0) + 4.0)
        )
        / (8.0 * max(rho_n, 1e-10))
    )
