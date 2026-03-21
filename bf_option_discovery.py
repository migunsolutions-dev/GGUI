"""
Discover blastFoam/OpenFOAM dropdown options locally (no web).
Used by tab_3d_general for EOS, activation, thermo, decomposition method.
Merge: discovered options + tutorials + loaded-case tokens (never discard unknown from case).
Cache at runtime to avoid slow UI.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Set, Tuple

log = logging.getLogger(__name__)

# Built-in options from blastFoam/OpenFOAM (phaseProperties and decomposeParDict)
# Used when discovery finds nothing or as base list; loaded-case tokens are always added.
EOS_BUILTIN: List[str] = ["JWL", "BirchMurnaghan3", "idealGas"]
ACTIVATION_BUILTIN: List[str] = ["pressureBased", "none"]
THERMO_BUILTIN: List[str] = ["eConst", "ePolynomial"]
DECOMPOSITION_METHOD_BUILTIN: List[str] = ["scotch", "simple", "hierarchical", "manual", "metis"]

_cache: dict = {}


def _scan_file_for_tokens(path: str, patterns: List[Tuple[str, str]]) -> Set[str]:
    """Scan file at path for regex patterns; return set of first group matches."""
    out: Set[str] = set()
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return out
    for name, pat in patterns:
        for m in re.finditer(pat, text):
            if m.lastindex and m.group(1):
                out.add(m.group(1).strip())
    return out


def _discover_phase_tokens(root_dirs: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Scan phaseProperties files for equationOfState, activationModel, thermo tokens."""
    eos_set: Set[str] = set(EOS_BUILTIN)
    act_set: Set[str] = set(ACTIVATION_BUILTIN)
    thermo_set: Set[str] = set(THERMO_BUILTIN)
    for root in root_dirs:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for f in filenames:
                if f != "phaseProperties":
                    continue
                path = os.path.join(dirpath, f)
                try:
                    text = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                for m in re.finditer(r"equationOfState\s+(\w+)", text):
                    eos_set.add(m.group(1))
                for m in re.finditer(r"activationModel\s+(\w+)", text):
                    act_set.add(m.group(1))
                for m in re.finditer(r"thermo\s+(\w+)", text):
                    w = m.group(1)
                    if w not in ("const", "transport", "type"):
                        thermo_set.add(w)
    return (sorted(eos_set), sorted(act_set), sorted(thermo_set))


def _discover_decomposition_methods(root_dirs: List[str]) -> List[str]:
    """Scan decomposeParDict for method tokens."""
    methods: Set[str] = set(DECOMPOSITION_METHOD_BUILTIN)
    for root in root_dirs:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for f in filenames:
                if f != "decomposeParDict":
                    continue
                path = os.path.join(dirpath, f)
                found = _scan_file_for_tokens(path, [("method", r"method\s+(\w+)")])
                methods.update(found)
    return sorted(methods)


def get_discovery_roots() -> List[str]:
    """Roots to scan: project building3D/mappedBuilding3D, env OPENFOAM_PROJECT_DIR / BLASTFOAM_TUTORIALS."""
    roots: List[str] = []
    try:
        gui_root = os.path.dirname(os.path.abspath(__file__))
        for sub in ("building3D", "mappedBuilding3D", "building3D/building3D", "mappedBuilding3D/building3D"):
            p = os.path.join(gui_root, sub)
            if os.path.isdir(p):
                roots.append(p)
        roots.append(gui_root)
    except Exception:
        pass
    for env in ("OPENFOAM_PROJECT_DIR", "BLASTFOAM_TUTORIALS", "WM_PROJECT_DIR"):
        v = os.environ.get(env, "").strip()
        if v and os.path.isdir(v):
            roots.append(v)
    return roots


def get_eos_options(loaded_tokens: List[str] | None = None) -> List[str]:
    """EOS model options: discovered + built-in + loaded-case tokens (no duplicate, unknown from case included)."""
    key = "eos"
    if key not in _cache:
        roots = get_discovery_roots()
        eos_list, _, _ = _discover_phase_tokens(roots)
        if not eos_list:
            eos_list = list(EOS_BUILTIN)
            log.debug("bf_option_discovery: EOS list is built-in only (discovery found none)")
        _cache[key] = eos_list
    out = list(_cache[key])
    if loaded_tokens:
        for t in loaded_tokens:
            if t and t not in out:
                out.append(t)
    return out


def get_activation_options(loaded_tokens: List[str] | None = None) -> List[str]:
    """Activation model options: discovered + built-in + loaded-case tokens."""
    key = "activation"
    if key not in _cache:
        roots = get_discovery_roots()
        _, act_list, _ = _discover_phase_tokens(roots)
        if not act_list:
            act_list = list(ACTIVATION_BUILTIN)
            log.debug("bf_option_discovery: activation list is built-in only")
        _cache[key] = act_list
    out = list(_cache[key])
    if loaded_tokens:
        for t in loaded_tokens:
            if t and t not in out:
                out.append(t)
    return out


def get_thermo_options(loaded_tokens: List[str] | None = None) -> List[str]:
    """Thermo/energy model options: discovered + built-in + loaded-case tokens."""
    key = "thermo"
    if key not in _cache:
        roots = get_discovery_roots()
        _, _, thermo_list = _discover_phase_tokens(roots)
        if not thermo_list:
            thermo_list = list(THERMO_BUILTIN)
            log.debug("bf_option_discovery: thermo list is built-in only")
        _cache[key] = thermo_list
    out = list(_cache[key])
    if loaded_tokens:
        for t in loaded_tokens:
            if t and t not in out:
                out.append(t)
    return out


def get_decomposition_method_options(loaded_tokens: List[str] | None = None) -> List[str]:
    """Decomposition method options: discovered + built-in + loaded-case tokens."""
    key = "decomposition"
    if key not in _cache:
        roots = get_discovery_roots()
        methods = _discover_decomposition_methods(roots)
        if not methods:
            methods = list(DECOMPOSITION_METHOD_BUILTIN)
            log.debug("bf_option_discovery: decomposition list is built-in only")
        _cache[key] = methods
    out = list(_cache[key])
    if loaded_tokens:
        for t in loaded_tokens:
            if t and t not in out:
                out.append(t)
    return out


def clear_cache() -> None:
    """Clear discovery cache (e.g. when switching project)."""
    _cache.clear()
