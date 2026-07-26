"""Pure material / imported-case required-value validation (no UI)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from domain_errors import (
    IncompleteMaterialError,
    MissingRequiredInputError,
    UnknownMaterialError,
    UnsupportedSourceRepresentationError,
)
from material_catalog import JWL_PARAMETERS, MATERIALS

# Canonical Custom / case material keys used by generators and catalogs.
REQUIRED_CUSTOM_MATERIAL_KEYS = ("rho", "energy", "A", "B", "R1", "R2", "omega")

# Imported Direct-Charge physics that must be recovered or explicitly supplied.
REQUIRED_IMPORTED_PHYSICS_KEYS = (
    "material_name",
    "rho_charge",
    "energy_j_per_kg",
    "mass_kg",
)

# CASE_DEFINED provenance on these keys means the source cannot be regenerated
# identically without an unsupported approximation.
UNSUPPORTED_IMPORT_KEYS = frozenset(
    {
        "material_name",
        "rho_charge",
        "energy_j_per_kg",
        "mass_kg",
        "cell_size",
        "charge_shape",
        "mesh_mode",
        "outer_boundary",
        "top_boundary",
        "bottom_boundary",
    }
)

_POSITIVE_CUSTOM_KEYS = frozenset({"rho", "energy", "A", "B", "R1", "R2", "omega", "E0"})


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str
    message: str


@dataclass
class RequiredValuesResult:
    ok: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[str]:
        return [issue.message for issue in self.issues]

    def raise_if_invalid(self) -> None:
        if self.ok:
            return
        first = self.issues[0]
        joined = "\n".join(self.errors)
        if first.code == "unknown_material":
            raise UnknownMaterialError(joined)
        if first.code == "incomplete_material":
            raise IncompleteMaterialError(joined)
        if first.code == "unsupported_source":
            raise UnsupportedSourceRepresentationError(joined)
        raise MissingRequiredInputError(joined)


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def resolve_custom_material_props(material_props: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize Custom props using canonical keys only (rho, energy, A…)."""
    return dict(material_props or {})


def validate_material_definition(
    material_name: Any,
    material_props: Optional[Mapping[str, Any]] = None,
    *,
    rho_charge: Any = None,
    energy_j_per_kg: Any = None,
) -> List[ValidationIssue]:
    """Validate catalog / Custom material definitions without UI side effects."""
    issues: List[ValidationIssue] = []
    name = str(material_name or "").strip()
    if not name:
        issues.append(
            ValidationIssue(
                "missing_required",
                "material_name",
                "Material is undefined. Select a catalog material or complete a Custom definition.",
            )
        )
        return issues

    if name == "Custom":
        props = resolve_custom_material_props(material_props)
        if "rho" not in props and rho_charge is not None:
            props["rho"] = rho_charge
        if "energy" not in props and "E0" not in props and energy_j_per_kg is not None:
            props["energy"] = energy_j_per_kg

        missing: List[str] = []
        for key in REQUIRED_CUSTOM_MATERIAL_KEYS:
            if key == "energy":
                if "energy" not in props and "E0" not in props:
                    missing.append("energy (or E0)")
                continue
            if key not in props:
                missing.append(key)
        if missing:
            issues.append(
                ValidationIssue(
                    "incomplete_material",
                    "material_props",
                    "Custom material is incomplete; missing required parameter(s): "
                    + ", ".join(missing)
                    + ".",
                )
            )

        for key in ("rho", "A", "B", "R1", "R2", "omega"):
            if key not in props:
                continue
            number = _finite_number(props.get(key))
            if number is None:
                issues.append(
                    ValidationIssue(
                        "incomplete_material",
                        key,
                        f"Custom material parameter {key!r} is missing, malformed, or non-finite.",
                    )
                )
            elif key in _POSITIVE_CUSTOM_KEYS and number <= 0:
                issues.append(
                    ValidationIssue(
                        "incomplete_material",
                        key,
                        f"Custom material parameter {key!r} must be > 0.",
                    )
                )
        if "energy" in props or "E0" in props:
            number = _finite_number(props.get("E0", props.get("energy")))
            if number is None:
                issues.append(
                    ValidationIssue(
                        "incomplete_material",
                        "energy",
                        "Custom material energy/E0 is missing, malformed, or non-finite.",
                    )
                )
            elif number <= 0:
                issues.append(
                    ValidationIssue(
                        "incomplete_material",
                        "energy",
                        "Custom material energy/E0 must be > 0.",
                    )
                )
        return issues

    if name not in JWL_PARAMETERS or name not in MATERIALS:
        issues.append(
            ValidationIssue(
                "unknown_material",
                "material_name",
                f"Unknown material {name!r}. GGUI will not substitute another catalog material.",
            )
        )
        return issues

    # Known catalog materials are complete in MATERIALS/JWL_PARAMETERS. Only
    # validate caller-supplied density/energy overrides — never invent them here
    # when the caller passed an explicit undefined (None) without a catalog fill.
    if rho_charge is not None:
        rho = _finite_number(rho_charge)
        if rho is None or rho <= 0:
            issues.append(
                ValidationIssue(
                    "missing_required",
                    "rho_charge",
                    f"Density for material {name!r} is undefined or invalid.",
                )
            )
    if energy_j_per_kg is not None:
        energy = _finite_number(energy_j_per_kg)
        if energy is None or energy <= 0:
            issues.append(
                ValidationIssue(
                    "missing_required",
                    "energy_j_per_kg",
                    f"Energy for material {name!r} is undefined or invalid.",
                )
            )
    return issues


def _issue_for_undefined(field_name: str) -> ValidationIssue:
    labels = {
        "material_name": "Material composition",
        "rho_charge": "Charge density",
        "energy_j_per_kg": "Charge energy",
        "mass_kg": "Charge mass",
        "cell_size": "Base cell size",
    }
    label = labels.get(field_name, field_name)
    return ValidationIssue(
        "missing_required",
        field_name,
        f"{label} is undefined. Recovered source data did not supply it and no explicit "
        "user value has been provided. Initialise Model is blocked.",
    )


def _issue_for_unsupported(feature: Mapping[str, Any]) -> ValidationIssue:
    source = str(feature.get("source_feature") or feature.get("field") or "source feature")
    why = str(feature.get("reason") or "it cannot be represented identically in GGUI")
    affected = str(feature.get("affected") or "the regenerated model")
    message = (
        f"Unsupported source representation: {source}. {why} "
        f"Affected: {affected}. Initialise Model is blocked."
    )
    return ValidationIssue(
        "unsupported_source",
        str(feature.get("field") or "source"),
        message,
    )


def _input_value_missing(inputs: Any, key: str) -> bool:
    value = getattr(inputs, key, None)
    if key == "material_name":
        return not str(value or "").strip()
    number = _finite_number(value)
    return number is None or number <= 0


def validate_required_values(
    inputs: Any,
    *,
    undefined_keys: Optional[Iterable[str]] = None,
    imported_field_meta: Optional[Mapping[str, Any]] = None,
    unsupported_features: Optional[Sequence[Mapping[str, Any]]] = None,
    require_imported_physics: bool = False,
) -> RequiredValuesResult:
    """Collect every blocking reason for imported / material required values.

    Pure function: does not touch Qt or write cases. Does not stop at the first
    problem. Provenance metadata only blocks while the corresponding value is
    still undefined / missing — an explicit user value unblocks regeneration of
    a new GGUI model (not a claim of source identity).
    """
    issues: List[ValidationIssue] = []
    undefined: Set[str] = {str(k) for k in (undefined_keys or ())}
    meta = dict(imported_field_meta or {})

    for feature in unsupported_features or ():
        if not isinstance(feature, Mapping):
            continue
        field_name = str(feature.get("field") or "")
        if field_name and field_name not in undefined and not _input_value_missing(
            inputs, field_name
        ):
            continue
        issues.append(_issue_for_unsupported(feature))

    if require_imported_physics or undefined or meta:
        checked_keys = list(REQUIRED_IMPORTED_PHYSICS_KEYS)
        for key in UNSUPPORTED_IMPORT_KEYS:
            if key not in checked_keys:
                checked_keys.append(key)

        for key in checked_keys:
            if key in REQUIRED_IMPORTED_PHYSICS_KEYS:
                missing = key in undefined or _input_value_missing(inputs, key)
            else:
                missing = key in undefined
            if not missing:
                continue
            mf = meta.get(key)
            provenance = getattr(mf, "provenance", None) if mf is not None else None
            prov_value = getattr(provenance, "value", provenance)
            if prov_value == "CASE_DEFINED":
                reason = ""
                if mf is not None:
                    reason = getattr(mf, "reason", "") or ""
                reason = reason or (
                    "the source value cannot be represented identically in GGUI"
                )
                affected = (
                    "Direct Charge physics used by Initialise Model"
                    if key in REQUIRED_IMPORTED_PHYSICS_KEYS
                    else "the regenerated mesh / boundary model"
                )
                issues.append(
                    _issue_for_unsupported(
                        {
                            "field": key,
                            "source_feature": f"{key} in the imported blastFoam case",
                            "reason": reason,
                            "affected": affected,
                        }
                    )
                )
            elif key in REQUIRED_IMPORTED_PHYSICS_KEYS:
                issues.append(_issue_for_undefined(key))

    if "material_name" not in undefined and not (
        require_imported_physics and _input_value_missing(inputs, "material_name")
    ):
        issues.extend(
            validate_material_definition(
                getattr(inputs, "material_name", None),
                getattr(inputs, "material_props", None),
                rho_charge=getattr(inputs, "rho_charge", None),
                energy_j_per_kg=getattr(inputs, "energy_j_per_kg", None),
            )
        )

    seen: Set[str] = set()
    unique: List[ValidationIssue] = []
    for issue in issues:
        token = f"{issue.code}|{issue.field}|{issue.message}"
        if token in seen:
            continue
        seen.add(token)
        unique.append(issue)
    return RequiredValuesResult(ok=not unique, issues=unique)
