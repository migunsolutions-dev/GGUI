"""Helpers for case_init_mode.json metadata updates (dialog-independent)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def record_set_cmd_actual(
    case_dir: str,
    set_cmd_actual: str,
    *,
    retries_used: int = 0,
    cells_inside_charge: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist the command actually executed during initialization.

    Used by the GUI init path after a successful setFields/setRefinedFields run.
    Returns the updated mode dict.
    """
    mode_path = os.path.join(case_dir, "case_init_mode.json")
    with open(mode_path, "r", encoding="utf-8") as stream:
        mode = json.load(stream)
    if not isinstance(mode, dict):
        raise ValueError("case_init_mode.json root must be a JSON object")
    if cells_inside_charge is not None:
        mode["cells_inside_charge"] = int(cells_inside_charge)
    mode["retries_used"] = int(retries_used)
    mode["set_cmd_actual"] = str(set_cmd_actual)
    with open(mode_path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(mode, stream, indent=2)
        stream.write("\n")
    return mode
