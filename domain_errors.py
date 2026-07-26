"""Shared domain-specific errors for GGUI validation and generation."""
from __future__ import annotations


class UnknownMaterialError(ValueError):
    """Raised when a material name is not present in the GGUI catalog."""


class IncompleteMaterialError(ValueError):
    """Raised when a Custom (or incomplete) material lacks required parameters."""


class MissingRequiredInputError(ValueError):
    """Raised when a required physical input is undefined or invalid."""


class UnsupportedSourceRepresentationError(ValueError):
    """Raised when a source-case feature cannot be represented identically in GGUI."""
