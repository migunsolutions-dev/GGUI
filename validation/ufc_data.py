"""Load version-controlled UFC 3-340-02 JSON tables."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=16)
def load_json(filename: str) -> Dict[str, Any]:
    path = DATA_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))
