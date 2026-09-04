"""UFC 3-340-02 English blast-chart units → SI.

These are metrology conversions, not blast-physics models.
Pound is avoirdupois (exact 0.45359237 kg). Foot and inch are exact SI.
"""
from __future__ import annotations

import math

LB_TO_KG = 0.45359237
FT_TO_M = 0.3048
IN_TO_M = 0.0254
LBF_TO_N = 4.4482216152605  # 1 lbf = gc * 1 lb, with g = 9.80665 m/s²
PSI_TO_PA = LBF_TO_N / (IN_TO_M ** 2)
PSI_TO_KPA = PSI_TO_PA / 1000.0

# 1 ft / lb**(1/3) = 0.3048 / (0.45359237**(1/3)) m / kg**(1/3)
FT_LB13_TO_M_KG13 = FT_TO_M / (LB_TO_KG ** (1.0 / 3.0))

# time scaled: (ms / lb**(1/3)) → (ms / kg**(1/3))
MS_LB13_TO_MS_KG13 = 1.0 / (LB_TO_KG ** (1.0 / 3.0))

# impulse scaled: (psi·ms / lb**(1/3)) → (kPa·ms / kg**(1/3))
PSI_MS_LB13_TO_KPA_MS_KG13 = PSI_TO_KPA / (LB_TO_KG ** (1.0 / 3.0))

# shock speed: ft/ms → m/s
FT_PER_MS_TO_M_PER_S = FT_TO_M / 0.001


def cube_root(value: float) -> float:
    return float(value) ** (1.0 / 3.0)


def scaled_si(length_m: float, mass_kg: float) -> float:
    return float(length_m) / cube_root(float(mass_kg))


def unscale_si(scaled: float, mass_kg: float) -> float:
    return float(scaled) * cube_root(float(mass_kg))


def si_scaled_to_english(scaled_m_kg13: float) -> float:
    return float(scaled_m_kg13) / FT_LB13_TO_M_KG13


def english_scaled_to_si(scaled_ft_lb13: float) -> float:
    return float(scaled_ft_lb13) * FT_LB13_TO_M_KG13


def friedlander_impulse_factor(decay: float) -> float:
    """I / (P * t0) = (b - 1 + exp(-b)) / b**2. Limit 1/2 as b → 0."""
    b = float(decay)
    if abs(b) < 1e-12:
        return 0.5
    return (b - 1.0 + math.exp(-b)) / (b * b)
