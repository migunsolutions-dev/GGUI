"""Small scope-aware helpers for OpenFOAM root dictionary scalar entries."""
from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping


def _line_depths(text: str) -> list[int]:
    """Return brace depth at each line start, ignoring comments and strings."""
    depths: list[int] = []
    depth = 0
    in_block_comment = False
    in_string = False
    escaped = False
    for line in text.splitlines(keepends=True):
        depths.append(depth)
        i = 0
        while i < len(line):
            ch = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue
            if ch == "/" and nxt == "/":
                break
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth = max(0, depth - 1)
            i += 1
    return depths


def _entry_pattern(keys: Iterable[str]) -> re.Pattern[str]:
    alternatives = "|".join(
        re.escape(str(key)) for key in sorted(set(keys), key=len, reverse=True)
    )
    return re.compile(
        rf"^(\s*)({alternatives})(\s+)([^;]*)(;[^\r\n]*)(\r?\n)?$"
    )


def read_top_level_entries(text: str, keys: Iterable[str]) -> Dict[str, str]:
    wanted = tuple(dict.fromkeys(str(key) for key in keys))
    if not wanted:
        return {}
    pattern = _entry_pattern(wanted)
    values: Dict[str, str] = {}
    lines = text.splitlines(keepends=True)
    for depth, line in zip(_line_depths(text), lines):
        if depth != 0:
            continue
        match = pattern.match(line)
        if match and match.group(2) not in values:
            values[match.group(2)] = match.group(4).strip()
    return values


def format_foam_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def update_top_level_entries(
    text: str,
    updates: Mapping[str, object],
) -> tuple[str, tuple[str, ...]]:
    """Update existing root entries, never same-named entries in nested dicts."""
    remaining = dict(updates)
    if not remaining:
        return text, ()
    pattern = _entry_pattern(remaining)
    lines = text.splitlines(keepends=True)
    depths = _line_depths(text)
    changed: list[str] = []
    output: list[str] = []
    for depth, line in zip(depths, lines):
        match = pattern.match(line) if depth == 0 else None
        key = match.group(2) if match else ""
        if match and key in remaining:
            ending = match.group(6) or ""
            output.append(
                f"{match.group(1)}{key}{match.group(3)}"
                f"{format_foam_scalar(remaining.pop(key))}{match.group(5)}{ending}"
            )
            changed.append(key)
        else:
            output.append(line)
    if remaining:
        raise KeyError(f"Top-level keys not found: {sorted(remaining)}")
    return "".join(output), tuple(changed)
