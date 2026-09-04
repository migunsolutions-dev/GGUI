"""Read/write automatic Validation sampling metadata next to an OpenFOAM case."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

from validation.auto_points import SamplingPlan, plan_from_dict, plan_to_dict
from validation.fingerprint import fingerprints_match

SAMPLING_FILENAME = "ggui_validation_sampling.json"

LEGACY_NO_VALIDATION_HISTORIES = (
    "Automatic validation histories were not generated for this run and cannot "
    "be reconstructed from sparse VTK frames. Peak/impulse at Validation Points "
    "are N/A."
)

PLANNED_NOT_RUN = "Planned validation points — simulation not run yet"

THREE_D_HEMI_NA = (
    "Hemispherical/reflected reference is not applied to an arbitrary 3D gauge "
    "without surface/orientation data."
)

SAMPLING_MISMATCH = (
    "Stored sampling plan does not match the current case/configuration "
    "and was not reused."
)

SAMPLING_LEGACY = (
    "Stored sampling plan has no provenance fingerprint and was not reused."
)


def sampling_path(case_dir: str) -> str:
    return os.path.join(case_dir or "", SAMPLING_FILENAME)


def write_sampling_plan(case_dir: str, plan: SamplingPlan) -> str:
    path = sampling_path(case_dir)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(plan_to_dict(plan), handle, indent=2)
        handle.write("\n")
    return path


def read_sampling_plan(case_dir: str) -> Optional[SamplingPlan]:
    path = sampling_path(case_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return plan_from_dict(data)


def load_matching_plan(
    case_dir: str,
    expected: Optional[Dict] = None,
) -> Tuple[Optional[SamplingPlan], str]:
    """Return the stored plan only when its fingerprint matches the current case."""
    loaded = read_sampling_plan(case_dir)
    if loaded is None:
        return None, ""
    stored = dict(loaded.fingerprint or {})
    if not stored:
        return None, SAMPLING_LEGACY
    if expected is None:
        return None, SAMPLING_MISMATCH
    if not fingerprints_match(stored, expected):
        return None, SAMPLING_MISMATCH
    return loaded, ""
