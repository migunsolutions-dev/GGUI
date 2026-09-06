"""Central mapping from physical quantities to native OpenFOAM field names.

GGUI code should ask for a physical quantity (pressure, density, velocity,
temperature, explosive fraction, reaction progress). This module resolves
those names onto the real field schema of the selected source model.

JWL keeps its two-phase fields. IG keeps the single-phase Sedov schema.
No fake ``alpha.c4`` is invented for IG.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

from models import SOURCE_MODEL_IG, SOURCE_MODEL_JWL, normalize_source_model

PRESSURE = "pressure"
DENSITY = "density"
VELOCITY = "velocity"
TEMPERATURE = "temperature"
EXPLOSIVE_FRACTION = "explosive_fraction"
REACTION_PROGRESS = "reaction_progress"
INTERNAL_ENERGY = "internal_energy"
TOTAL_ENERGY = "total_energy"
IMPULSE = "impulse"
OVERPRESSURE = "overpressure"

PHYSICAL_QUANTITIES: Tuple[str, ...] = (
    PRESSURE,
    DENSITY,
    VELOCITY,
    TEMPERATURE,
    EXPLOSIVE_FRACTION,
    REACTION_PROGRESS,
    INTERNAL_ENERGY,
    TOTAL_ENERGY,
    IMPULSE,
    OVERPRESSURE,
)

# Native OpenFOAM names, in preference order, for each source-model schema.
# An empty tuple means the quantity does not exist for that model.
_SCHEMA: Dict[str, Dict[str, Tuple[str, ...]]] = {
    SOURCE_MODEL_JWL: {
        PRESSURE: ("p",),
        DENSITY: ("rho.air", "rho"),
        VELOCITY: ("U",),
        TEMPERATURE: ("T",),
        EXPLOSIVE_FRACTION: ("alpha.c4",),
        REACTION_PROGRESS: ("lambda.c4",),
        INTERNAL_ENERGY: ("e",),
        TOTAL_ENERGY: ("rhoE",),
        IMPULSE: ("impulse",),
        OVERPRESSURE: ("overpressure", "p"),
    },
    SOURCE_MODEL_IG: {
        PRESSURE: ("p",),
        DENSITY: ("rho",),
        VELOCITY: ("U",),
        TEMPERATURE: ("T",),
        EXPLOSIVE_FRACTION: (),
        REACTION_PROGRESS: (),
        INTERNAL_ENERGY: ("e",),
        TOTAL_ENERGY: ("rhoE",),
        IMPULSE: ("impulse",),
        OVERPRESSURE: ("overpressure", "p"),
    },
}


class UnknownPhysicalQuantity(KeyError):
    """Raised when a caller asks for a quantity this schema does not know."""


def schema_for(source_model: Optional[str] = None) -> Dict[str, Tuple[str, ...]]:
    return dict(_SCHEMA[normalize_source_model(source_model)])


def foam_candidates(
    quantity: str, source_model: Optional[str] = None
) -> Tuple[str, ...]:
    """Native field names that may hold ``quantity``, preferred first."""
    schema = schema_for(source_model)
    if quantity not in schema:
        raise UnknownPhysicalQuantity(quantity)
    return schema[quantity]


def foam_field(quantity: str, source_model: Optional[str] = None) -> Optional[str]:
    """Primary native field, or None if the model has no such quantity."""
    names = foam_candidates(quantity, source_model)
    return names[0] if names else None


def quantity_available(quantity: str, source_model: Optional[str] = None) -> bool:
    return bool(foam_candidates(quantity, source_model))


def foam_fields(
    quantities: Sequence[str], source_model: Optional[str] = None
) -> Tuple[str, ...]:
    """Resolve physical quantities to native names, dropping unavailable ones."""
    out = []
    seen = set()
    for quantity in quantities:
        name = foam_field(quantity, source_model)
        if name is None or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def remap_field_names(source_model: Optional[str] = None) -> Tuple[str, ...]:
    """Fields a remap snapshot may transfer for this source model."""
    names = [
        PRESSURE,
        DENSITY,
        VELOCITY,
        TEMPERATURE,
    ]
    if quantity_available(EXPLOSIVE_FRACTION, source_model):
        names.append(EXPLOSIVE_FRACTION)
    return foam_fields(names, source_model)


def probe_field_names(
    quantities: Iterable[str], source_model: Optional[str] = None
) -> Tuple[str, ...]:
    return foam_fields(tuple(quantities), source_model)
