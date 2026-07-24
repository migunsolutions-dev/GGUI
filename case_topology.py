"""Dimension-aware OpenFOAM/blastFoam topology classification.

Isolated from tab_2d / tab_3d_general. Classifies cases by mesh or blockMesh
boundary *types* (wedge / empty), never by directory name or field names.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class CaseDimension(str, Enum):
    AXISYMMETRIC_WEDGE = "AXISYMMETRIC_WEDGE"
    PLANAR_2D_EMPTY = "PLANAR_2D_EMPTY"
    GENERAL_3D = "GENERAL_3D"
    AMBIGUOUS_OR_INVALID = "AMBIGUOUS_OR_INVALID"


@dataclass(frozen=True)
class PatchTypeRecord:
    name: str
    patch_type: str


@dataclass
class TopologyEvidence:
    source: str  # "polyMesh/boundary" | "blockMeshDict" | "both" | "none"
    patches: Tuple[PatchTypeRecord, ...] = ()
    wedge_patch_names: Tuple[str, ...] = ()
    empty_patch_names: Tuple[str, ...] = ()
    wedge_half_angle_deg: Optional[float] = None
    symmetry_axis: Optional[str] = None  # "X" | "Y" | "Z" when inferred
    radius_extent_m: Optional[float] = None
    height_extent_m: Optional[float] = None
    notes: Tuple[str, ...] = ()
    conflict_detail: Optional[str] = None


@dataclass
class ClassificationResult:
    classification: CaseDimension
    evidence: TopologyEvidence
    reason: str = ""
    case_dir: str = ""

    @property
    def is_axisymmetric(self) -> bool:
        return self.classification == CaseDimension.AXISYMMETRIC_WEDGE

    @property
    def is_planar_2d(self) -> bool:
        return self.classification == CaseDimension.PLANAR_2D_EMPTY

    @property
    def is_general_3d(self) -> bool:
        return self.classification == CaseDimension.GENERAL_3D


_COMMENT_RE = re.compile(r"/\*.*?\*/|//.*?$", re.DOTALL | re.MULTILINE)
_INCLUDE_RE = re.compile(r"#\s*include(?:IfPresent)?\s+[\"'<]")


def _strip_foam_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


def _read_text(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    except OSError:
        return None


def _extract_paren_block_after(text: str, keyword: str) -> Optional[str]:
    idx = text.find(keyword)
    if idx < 0:
        return None
    rest = text[idx:]
    open_paren = rest.find("(")
    if open_paren < 0:
        return None
    start = idx + open_paren + 1
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


def _parse_named_type_entries(block: str) -> List[PatchTypeRecord]:
    """Parse `name { ... type xxx; ... }` entries from a boundary block."""
    records: List[PatchTypeRecord] = []
    for match in re.finditer(
        r"([A-Za-z_][\w]*)\s*\{([^{}]*?)\}",
        block,
        flags=re.DOTALL,
    ):
        name = match.group(1)
        body = match.group(2)
        type_m = re.search(r"\btype\s+(\w+)\s*;", body)
        if not type_m:
            continue
        # Skip FoamFile-like objects accidentally matched
        if name in {"FoamFile", "version", "format", "class", "object", "location"}:
            continue
        records.append(PatchTypeRecord(name=name, patch_type=type_m.group(1)))
    return records


def parse_polymesh_boundary(case_dir: str) -> Optional[List[PatchTypeRecord]]:
    path = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    text = _read_text(path)
    if text is None:
        return None
    cleaned = _strip_foam_comments(text)
    if _INCLUDE_RE.search(cleaned):
        raise ValueError(
            f"constant/polyMesh/boundary uses #include; cannot classify reliably ({path})"
        )
    # polyMesh/boundary is a list: nPatches ( patch { type ...; } ... )
    block = _extract_paren_block_after(cleaned, "(")
    # Prefer the outermost content after the integer count
    records = _parse_named_type_entries(cleaned)
    # Filter out FoamFile if parser grabbed it via braces in header — already skipped.
    # Require at least one patch with a known OF boundary type word.
    if not records:
        return []
    return records


def parse_blockmesh_boundary(case_dir: str) -> Optional[List[PatchTypeRecord]]:
    path = os.path.join(case_dir, "system", "blockMeshDict")
    text = _read_text(path)
    if text is None:
        return None
    cleaned = _strip_foam_comments(text)
    # #calc / #codeStream macros are allowed for coordinates; patch *types* must be literal.
    bounds = _extract_paren_block_after(cleaned, "boundary")
    if bounds is None:
        return []
    return _parse_named_type_entries(bounds)


def _summarize_patches(patches: Sequence[PatchTypeRecord]) -> TopologyEvidence:
    wedges = tuple(p.name for p in patches if p.patch_type == "wedge")
    empties = tuple(p.name for p in patches if p.patch_type == "empty")
    return TopologyEvidence(
        source="pending",
        patches=tuple(patches),
        wedge_patch_names=wedges,
        empty_patch_names=empties,
    )


def _classify_from_patches(
    patches: Sequence[PatchTypeRecord], source: str
) -> ClassificationResult:
    evidence = _summarize_patches(patches)
    evidence.source = source
    n_wedge = len(evidence.wedge_patch_names)
    # OpenFOAM may auto-create leftover defaultFaces (type empty) for unused
    # block faces. That must not override a valid wedge pair from an axisymmetric
    # tutorial mesh.
    meaningful_empty = tuple(
        name for name in evidence.empty_patch_names if name != "defaultFaces"
    )
    n_empty = len(meaningful_empty)
    n_default_empty = len(evidence.empty_patch_names) - n_empty

    if n_wedge and n_empty:
        evidence.conflict_detail = (
            f"Mixed wedge ({list(evidence.wedge_patch_names)}) and empty "
            f"({list(meaningful_empty)}) patches"
        )
        return ClassificationResult(
            CaseDimension.AMBIGUOUS_OR_INVALID,
            evidence,
            reason=evidence.conflict_detail,
        )

    if n_wedge >= 2:
        notes = list(evidence.notes)
        if n_default_empty:
            notes.append(
                f"Ignoring {n_default_empty} OpenFOAM defaultFaces empty patch(es) "
                "alongside a valid wedge pair"
            )
            evidence.notes = tuple(notes)
        return ClassificationResult(
            CaseDimension.AXISYMMETRIC_WEDGE,
            evidence,
            reason=(
                f"Found {n_wedge} type-wedge patches: "
                f"{', '.join(evidence.wedge_patch_names)}"
            ),
        )

    if n_wedge == 1:
        return ClassificationResult(
            CaseDimension.AMBIGUOUS_OR_INVALID,
            evidence,
            reason="Only one type-wedge patch found; a wedge pair is required",
        )

    if n_empty >= 2:
        return ClassificationResult(
            CaseDimension.PLANAR_2D_EMPTY,
            evidence,
            reason=(
                f"Found {n_empty} type-empty patches: "
                f"{', '.join(meaningful_empty)}"
            ),
        )

    if n_empty == 1:
        return ClassificationResult(
            CaseDimension.AMBIGUOUS_OR_INVALID,
            evidence,
            reason="Only one type-empty patch found; a planar front/back pair is required",
        )

    # Sole defaultFaces empty without wedges → not dimensional evidence.
    return ClassificationResult(
        CaseDimension.GENERAL_3D,
        evidence,
        reason="No wedge or empty dimensional pair detected",
    )


def _enrich_blockmesh_geometry(case_dir: str, evidence: TopologyEvidence) -> TopologyEvidence:
    """Best-effort geometry notes from blockMeshDict without executing #calc."""
    path = os.path.join(case_dir, "system", "blockMeshDict")
    text = _read_text(path)
    if text is None:
        return evidence
    cleaned = _strip_foam_comments(text)
    notes = list(evidence.notes)

    # Scalar assignments: H 20.0; R 20.0;
    h_m = re.search(r"\bH\s+([0-9]+(?:\.[0-9]+)?)\s*;", cleaned)
    r_m = re.search(r"\bR\s+([0-9]+(?:\.[0-9]+)?)\s*;", cleaned)
    if h_m:
        evidence.height_extent_m = float(h_m.group(1))
    if r_m:
        evidence.radius_extent_m = float(r_m.group(1))

    # Angle inside cos/sin(# deg): cos(5.0 * ...pi/180)
    ang = re.search(
        r"(?:cos|sin)\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\*\s*[^)]*pi\s*/\s*180",
        cleaned,
        flags=re.IGNORECASE,
    )
    if ang:
        evidence.wedge_half_angle_deg = float(ang.group(1))
        notes.append(f"wedge half-angle {evidence.wedge_half_angle_deg:g}° from blockMeshDict trig")

    # Numeric vertices (skip #calc macros)
    vblock = re.search(r"vertices\s*\((.+?)\)\s*;", cleaned, re.DOTALL)
    if vblock:
        verts = re.findall(
            r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
            vblock.group(1),
        )
        if verts:
            xs = [float(v[0]) for v in verts]
            ys = [float(v[1]) for v in verts]
            zs = [float(v[2]) for v in verts]
            spans = {
                "X": max(xs) - min(xs),
                "Y": max(ys) - min(ys),
                "Z": max(zs) - min(zs),
            }
            # Axis is the direction with near-zero radial extent at origin-like points.
            # Prefer the largest span among axes that have vertices with (near) zero
            # in the other two coords — common wedge pattern.
            axis_candidates = []
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                near0 = sum(1 for c in (x, y, z) if abs(c) < 1e-9)
                if near0 >= 2:
                    axis_candidates.append((x, y, z))
            if len(axis_candidates) >= 2:
                # Direction between axis points
                a = axis_candidates[0]
                b = axis_candidates[-1]
                d = (abs(b[0] - a[0]), abs(b[1] - a[1]), abs(b[2] - a[2]))
                axis = ("X", "Y", "Z")[max(range(3), key=lambda i: d[i])]
                evidence.symmetry_axis = axis
                notes.append(f"symmetry axis inferred as global {axis} from axis vertices")
            elif evidence.height_extent_m and evidence.radius_extent_m:
                # Tutorial-style macros without numeric vertices: GGUI-compatible Y-axis wedge
                # is the usual blastFoam axisymmetricCharge layout; record as inferred.
                evidence.symmetry_axis = "Y"
                notes.append(
                    "symmetry axis inferred as global Y from R/H macros "
                    "(blastFoam axisymmetric wedge layout)"
                )

    evidence.notes = tuple(notes)
    return evidence


def _enrich_polymesh_geometry(case_dir: str, evidence: TopologyEvidence) -> TopologyEvidence:
    points_path = os.path.join(case_dir, "constant", "polyMesh", "points")
    text = _read_text(points_path)
    if text is None:
        return evidence
    cleaned = _strip_foam_comments(text)
    coords = re.findall(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
        cleaned,
    )
    if len(coords) < 8:
        return evidence
    xs = [float(c[0]) for c in coords]
    ys = [float(c[1]) for c in coords]
    zs = [float(c[2]) for c in coords]
    notes = list(evidence.notes)
    # Axis: coordinate with smallest unique radial footprint among points near origin.
    # Use span of points with r_perp ~ 0.
    for axis_name, a, b, c in (
        ("Y", xs, zs, ys),
        ("X", ys, zs, xs),
        ("Z", xs, ys, zs),
    ):
        radial = [math.hypot(u, v) for u, v in zip(a, b)]
        on_axis = [w for w, r in zip(c, radial) if r < 1e-6]
        if len(on_axis) >= 2:
            evidence.symmetry_axis = axis_name
            evidence.height_extent_m = max(on_axis) - min(on_axis)
            evidence.radius_extent_m = max(radial)
            notes.append(
                f"symmetry axis {axis_name} from polyMesh points "
                f"(H≈{evidence.height_extent_m:g} m, R≈{evidence.radius_extent_m:g} m)"
            )
            break
    # Wedge half-angle from max |atan2| of points
    if evidence.symmetry_axis == "Y":
        angles = [abs(math.degrees(math.atan2(z, x))) for x, z in zip(xs, zs) if math.hypot(x, z) > 1e-9]
        if angles:
            evidence.wedge_half_angle_deg = max(angles)
            notes.append(f"wedge half-angle ≈ {evidence.wedge_half_angle_deg:g}° from mesh points")
    evidence.notes = tuple(notes)
    return evidence


def _same_dimensional_class(a: ClassificationResult, b: ClassificationResult) -> bool:
    return a.classification == b.classification


def classify_case_topology(case_dir: str) -> ClassificationResult:
    """Classify an OpenFOAM/blastFoam case root by dimensional topology."""
    case_dir = os.path.normpath(case_dir)
    if not os.path.isdir(case_dir):
        return ClassificationResult(
            CaseDimension.AMBIGUOUS_OR_INVALID,
            TopologyEvidence(source="none"),
            reason=f"Case directory does not exist: {case_dir}",
            case_dir=case_dir,
        )
    if not os.path.isdir(os.path.join(case_dir, "system")):
        return ClassificationResult(
            CaseDimension.AMBIGUOUS_OR_INVALID,
            TopologyEvidence(source="none"),
            reason="Missing system/ — not a valid OpenFOAM case root",
            case_dir=case_dir,
        )

    poly_err: Optional[str] = None
    block_err: Optional[str] = None
    poly_patches: Optional[List[PatchTypeRecord]] = None
    block_patches: Optional[List[PatchTypeRecord]] = None

    try:
        poly_patches = parse_polymesh_boundary(case_dir)
    except ValueError as exc:
        poly_err = str(exc)

    try:
        block_patches = parse_blockmesh_boundary(case_dir)
    except ValueError as exc:
        block_err = str(exc)

    if poly_err and block_err:
        return ClassificationResult(
            CaseDimension.AMBIGUOUS_OR_INVALID,
            TopologyEvidence(source="none", notes=(poly_err, block_err)),
            reason="Cannot read topology from polyMesh/boundary or blockMeshDict",
            case_dir=case_dir,
        )

    if poly_err and block_patches is None:
        return ClassificationResult(
            CaseDimension.AMBIGUOUS_OR_INVALID,
            TopologyEvidence(source="none", notes=(poly_err,)),
            reason=poly_err,
            case_dir=case_dir,
        )

    # polyMesh present (file existed and parsed — even if empty list)
    poly_path = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    has_poly = os.path.isfile(poly_path) and poly_patches is not None and poly_err is None
    has_block = block_patches is not None and block_err is None

    if has_poly and has_block:
        from_poly = _classify_from_patches(poly_patches or [], "polyMesh/boundary")
        from_block = _classify_from_patches(block_patches or [], "blockMeshDict")
        if not _same_dimensional_class(from_poly, from_block):
            detail = (
                f"polyMesh/boundary -> {from_poly.classification.value} "
                f"({from_poly.reason}); blockMeshDict -> {from_block.classification.value} "
                f"({from_block.reason})"
            )
            evidence = TopologyEvidence(
                source="both",
                patches=from_poly.evidence.patches,
                wedge_patch_names=from_poly.evidence.wedge_patch_names,
                empty_patch_names=from_poly.evidence.empty_patch_names,
                conflict_detail=detail,
                notes=(
                    f"blockMesh wedges={list(from_block.evidence.wedge_patch_names)} "
                    f"empty={list(from_block.evidence.empty_patch_names)}",
                ),
            )
            return ClassificationResult(
                CaseDimension.AMBIGUOUS_OR_INVALID,
                evidence,
                reason=f"Conflicting topology sources: {detail}",
                case_dir=case_dir,
            )
        # Agree — prefer generated mesh evidence, enrich from both
        result = from_poly
        result.evidence.source = "both"
        result.evidence = _enrich_polymesh_geometry(case_dir, result.evidence)
        result.evidence = _enrich_blockmesh_geometry(case_dir, result.evidence)
        result.case_dir = case_dir
        return result

    if has_poly:
        result = _classify_from_patches(poly_patches or [], "polyMesh/boundary")
        result.evidence = _enrich_polymesh_geometry(case_dir, result.evidence)
        result.case_dir = case_dir
        return result

    if has_block:
        result = _classify_from_patches(block_patches or [], "blockMeshDict")
        result.evidence = _enrich_blockmesh_geometry(case_dir, result.evidence)
        result.case_dir = case_dir
        return result

    reason = poly_err or block_err or "No constant/polyMesh/boundary or system/blockMeshDict found"
    return ClassificationResult(
        CaseDimension.AMBIGUOUS_OR_INVALID,
        TopologyEvidence(source="none"),
        reason=reason,
        case_dir=case_dir,
    )
