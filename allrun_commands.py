"""Structured, allowlisted Allrun preprocess command parsing.

Preserves utility arguments. Rejects unknown utilities, unsupported flags,
missing values, unsafe paths, and shell metacharacters. Never falls back to
running a bare utility name when arguments were present but invalid.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple


class CommandValidationStatus(str, Enum):
    OK = "OK"
    REJECTED = "REJECTED"


WHITELISTED_UTILITIES = frozenset(
    {"blockMesh", "setFields", "setRefinedFields", "checkMesh"}
)

# Per-utility allowlist. Only demonstrably safe / required flags are accepted.
# setRefinedFields: project fixtures invoke it with no arguments.
_UTILITY_FLAGS: Dict[str, FrozenSet[str]] = {
    "blockMesh": frozenset(),
    "setFields": frozenset(),
    "setRefinedFields": frozenset(),
    "checkMesh": frozenset({"-allGeometry", "-allTopology"}),
}

_UTILITY_FLAGS_WITH_VALUE: Dict[str, FrozenSet[str]] = {
    "blockMesh": frozenset({"-dict"}),
    "setFields": frozenset({"-dict"}),
    "setRefinedFields": frozenset(),
    "checkMesh": frozenset(),
}

_SHELL_META_RE = re.compile(
    r"""(?:
        ;|&&|\|\||   # chaining
        \| |         # pipe
        >|>>|<|      # redirection
        ` |          # backtick
        \$\(|        # command substitution
        \$\{|        # parameter expansion
        \$[A-Za-z_]| # variable expansion
        &(?!\S)      # backgrounding (lone &)
    )""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class AllrunCommand:
    """One validated preprocess utility invocation from Allrun."""

    utility: str
    arguments: Tuple[str, ...]
    source_line: str
    status: CommandValidationStatus = CommandValidationStatus.OK
    rejection_reason: str = ""

    @property
    def valid(self) -> bool:
        return self.status == CommandValidationStatus.OK

    def argv(self) -> Tuple[str, ...]:
        return (self.utility,) + self.arguments

    def display(self) -> str:
        return " ".join(self.argv())


class AllrunParseError(ValueError):
    """Raised when Allrun cannot be safely converted to a command sequence."""


def _is_safe_relative_dict_path(value: str) -> bool:
    if not value or value.startswith("-"):
        return False
    if os.path.isabs(value):
        return False
    # Reject Windows drive / UNC forms.
    if re.match(r"^[A-Za-z]:", value) or value.startswith("\\\\") or value.startswith("//"):
        return False
    parts = Path(value.replace("\\", "/")).parts
    if ".." in parts:
        return False
    if any(p.startswith("~") for p in parts):
        return False
    return True


def _reject(utility: str, source_line: str, reason: str, arguments: Sequence[str] = ()) -> AllrunCommand:
    return AllrunCommand(
        utility=utility,
        arguments=tuple(arguments),
        source_line=source_line,
        status=CommandValidationStatus.REJECTED,
        rejection_reason=reason,
    )


def validate_utility_arguments(
    utility: str,
    tokens: Sequence[str],
    *,
    source_line: str,
) -> AllrunCommand:
    """Validate argv tokens after the utility name against the allowlist."""
    if utility not in WHITELISTED_UTILITIES:
        return _reject(
            utility,
            source_line,
            f"Unsupported utility {utility!r}. "
            f"Allowed utilities: {sorted(WHITELISTED_UTILITIES)}.",
            tokens,
        )

    allowed_flags = _UTILITY_FLAGS[utility]
    allowed_valued = _UTILITY_FLAGS_WITH_VALUE[utility]
    accepted: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in allowed_valued:
            if i + 1 >= len(tokens):
                return _reject(
                    utility,
                    source_line,
                    f"Utility {utility!r}: option {tok!r} is missing its value.",
                    tokens,
                )
            value = tokens[i + 1]
            if tok == "-dict" and not _is_safe_relative_dict_path(value):
                return _reject(
                    utility,
                    source_line,
                    f"Utility {utility!r}: argument {tok!r} has unsupported path "
                    f"{value!r}. Only relative case paths without '..' are allowed.",
                    tokens,
                )
            accepted.extend([tok, value])
            i += 2
            continue
        if tok in allowed_flags:
            accepted.append(tok)
            i += 1
            continue
        return _reject(
            utility,
            source_line,
            f"Utility {utility!r}: unsupported argument {tok!r}.",
            tokens,
        )

    return AllrunCommand(
        utility=utility,
        arguments=tuple(accepted),
        source_line=source_line,
        status=CommandValidationStatus.OK,
    )


def _is_solver_launch_line(stripped: str) -> bool:
    """True for Allrun solver launch lines that preprocess parsing must ignore."""
    return bool(
        re.search(
            r"runApplication\s+(?:\$\(getApplication\)|getApplication|blastFoam)(?:\s|$)",
            stripped,
        )
    )


def parse_run_application_line(line: str) -> Optional[AllrunCommand]:
    """Parse one non-comment Allrun line containing runApplication.

    Returns ``None`` for ignored solver-launch lines (``getApplication`` / blastFoam).
    """
    stripped = line.strip()
    if _is_solver_launch_line(stripped):
        return None

    if _SHELL_META_RE.search(stripped):
        m = re.search(r"runApplication\s+(\S+)", stripped)
        utility = m.group(1) if m else ""
        return _reject(
            utility or "<unknown>",
            stripped,
            f"Shell operators, redirections, pipes, substitutions, or chained "
            f"commands are not allowed in Allrun preprocess lines: {stripped!r}",
        )

    m = re.match(r"runApplication\s+(.+)$", stripped)
    if not m:
        return _reject(
            "<unknown>",
            stripped,
            f"Not a runApplication preprocess line: {stripped!r}",
        )
    try:
        tokens = shlex.split(m.group(1), posix=True)
    except ValueError as exc:
        return _reject(
            "<unknown>",
            stripped,
            f"Could not parse runApplication arguments: {exc}",
        )
    if not tokens:
        return _reject("<unknown>", stripped, "runApplication is missing a utility name.")
    utility = tokens[0]
    if utility.startswith("$") or utility in {"getApplication", "blastFoam", "paraFoam"}:
        return None
    return validate_utility_arguments(utility, tokens[1:], source_line=stripped)


def parse_allrun_preprocess_sequence(case_dir: str) -> Tuple[AllrunCommand, ...]:
    """Extract allowlisted preprocess commands from Allrun, preserving arguments."""
    allrun = Path(case_dir) / "Allrun"
    if not allrun.is_file():
        raise AllrunParseError(f"No Allrun found in {case_dir}")
    text = allrun.read_text(encoding="utf-8", errors="ignore")

    commands: List[AllrunCommand] = []
    seen: set[Tuple[str, ...]] = set()
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            continue
        if "runApplication" not in stripped:
            # Non-runApplication body may still contain forbidden shell forms for
            # preprocess — ignore helper functions; only validate runApplication.
            continue
        cmd = parse_run_application_line(stripped)
        if cmd is None:
            continue
        if not cmd.valid:
            raise AllrunParseError(cmd.rejection_reason)
        key = cmd.argv()
        if key in seen:
            continue
        seen.add(key)
        commands.append(cmd)

    if not commands:
        raise AllrunParseError(
            "Allrun contains no whitelisted preprocessing utilities "
            f"(expected one of {sorted(WHITELISTED_UTILITIES)})"
        )
    return tuple(commands)


def preparation_commands_for_case(case_dir: str) -> Tuple[AllrunCommand, ...]:
    """Proven preprocess sequence from Allrun, plus checkMesh validation."""
    seq = list(parse_allrun_preprocess_sequence(case_dir))
    if not any(cmd.utility == "checkMesh" for cmd in seq):
        seq.append(
            AllrunCommand(
                utility="checkMesh",
                arguments=(),
                source_line="runApplication checkMesh",
            )
        )
    return tuple(seq)
