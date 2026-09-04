"""VIPER nogui command construction (no shell)."""
from __future__ import annotations

from typing import List, Sequence


def viper_argv(
    exe: str,
    vip_path: str,
    json_path: str,
    stages: Sequence[str],
) -> List[str]:
    argv = [str(exe), "nogui", f"file={vip_path}", f"json={json_path}"]
    argv.extend(str(s) for s in stages)
    return argv
