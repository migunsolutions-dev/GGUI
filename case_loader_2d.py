"""Inspect and report imported axisymmetric cases for Cylindrical–2D."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from case_topology import CaseDimension, ClassificationResult, classify_case_topology
from axisymmetric_viewer import AxisymmetricViewerWidget
from external_case_workflow_2d import ImportMode2D, import_mode_label, preparation_commands_for_case
from imported_case_mapping_2d import (
    FieldProvenance,
    ImportMappingResult,
    map_imported_case_to_gui,
)


@dataclass
class ImportedCase2DState:
    """Provenance and mode for an imported wedge working case."""

    case_dir: str  # active working path
    classification: ClassificationResult
    mode: ImportMode2D = ImportMode2D.IMPORTED_2D_UNINITIALIZED
    source_dir: str = ""
    working_copy_dir: Optional[str] = None
    mapping: Optional[ImportMappingResult] = None
    mesh_present: bool = False
    time_dirs: Tuple[str, ...] = ()
    latest_time: Optional[str] = None
    fields: Tuple[str, ...] = ()
    cell_count: Optional[int] = None
    cell_count_source: str = "none"
    mesh_owner_path: str = ""
    viewable: bool = False
    runnable: bool = False
    display_compatible: bool = False
    compatibility_notes: Tuple[str, ...] = ()
    prepare_commands: Tuple[str, ...] = ()
    prepare_results: Dict[str, int] = field(default_factory=dict)
    check_mesh_ok: bool = False
    charge_cell_count: Optional[int] = None
    source_inventory_hash: str = ""
    dirty_control_keys: Tuple[str, ...] = ()
    unsupported_pending_edits: Tuple[str, ...] = ()

    # Compatibility aliases for older attribute names.
    @property
    def lifecycle(self) -> ImportMode2D:
        return self.mode

    @lifecycle.setter
    def lifecycle(self, value: ImportMode2D) -> None:
        self.mode = value

    @property
    def lifecycle_label(self) -> str:
        return import_mode_label(self.mode)

    @property
    def active_case_path(self) -> str:
        return self.working_copy_dir or self.case_dir

    @property
    def radius_m(self) -> Optional[float]:
        if not self.mapping:
            return None
        mf = self.mapping.get("radius")
        return None if mf is None else mf.displayed_value

    @property
    def height_m(self) -> Optional[float]:
        if not self.mapping:
            return None
        mf = self.mapping.get("height")
        return None if mf is None else mf.displayed_value

    @property
    def symmetry_axis(self) -> Optional[str]:
        if not self.mapping:
            return self.classification.evidence.symmetry_axis
        mf = self.mapping.get("symmetry_axis")
        if mf and mf.displayed_value:
            return str(mf.displayed_value)
        return self.classification.evidence.symmetry_axis

    @property
    def wedge_half_angle_deg(self) -> Optional[float]:
        if not self.mapping:
            return self.classification.evidence.wedge_half_angle_deg
        mf = self.mapping.get("wedge_half_angle_deg")
        return None if mf is None else mf.displayed_value

    @property
    def parameters_from_case(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not self.mapping:
            return out
        for key, mf in self.mapping.fields.items():
            if mf.provenance == FieldProvenance.DIRECT and mf.displayed_value is not None:
                out[key] = str(mf.displayed_value)
        return out

    @property
    def parameters_derived(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not self.mapping:
            return out
        for key, mf in self.mapping.fields.items():
            if mf.provenance == FieldProvenance.DERIVED and mf.displayed_value is not None:
                out[key] = str(mf.displayed_value)
        return out

    @property
    def parameters_not_recovered(self) -> Tuple[str, ...]:
        if not self.mapping:
            return ()
        return self.mapping.not_recovered_keys

    @property
    def initialize_allowed(self) -> bool:
        return self.mode in (
            ImportMode2D.IMPORTED_2D_UNINITIALIZED,
            ImportMode2D.IMPORTED_2D_FAILED,
            ImportMode2D.IMPORTED_2D_READY,
        )

    @property
    def provenance(self) -> str:
        return "imported_working_case"


# Alias kept for imports during transition.
ExternalCase2DState = ImportedCase2DState


_COMMENT_RE = re.compile(r"/\*.*?\*/|//.*?$", re.DOTALL | re.MULTILINE)


def _strip(text: str) -> str:
    return _COMMENT_RE.sub("", text)


def _list_time_dirs(case_dir: str) -> List[str]:
    from openfoam_times_2d import list_numeric_time_labels

    return list_numeric_time_labels(case_dir)


def _discover_fields(case_dir: str, time_name: Optional[str]) -> List[str]:
    candidates: List[str] = []
    search_dirs = []
    if time_name:
        search_dirs.append(os.path.join(case_dir, time_name))
    search_dirs.append(os.path.join(case_dir, "0"))
    search_dirs.append(os.path.join(case_dir, "0.orig"))
    seen = set()
    skip = {"phi", "points", "faces", "owner", "neighbour", "boundary"}
    for folder in search_dirs:
        if not os.path.isdir(folder):
            continue
        try:
            for name in os.listdir(folder):
                path = os.path.join(folder, name)
                if not os.path.isfile(path):
                    continue
                base = name[:-5] if name.endswith(".orig") else name
                if base in skip or base.startswith("."):
                    continue
                if base not in seen:
                    seen.add(base)
                    candidates.append(base)
        except OSError:
            continue
        if candidates:
            break
    preferred = ["p", "alpha.c4", "rho", "T", "U", "rho.air", "rho.c4"]
    ordered = [f for f in preferred if f in seen]
    ordered.extend(sorted(f for f in candidates if f not in ordered))
    return ordered


def inspect_imported_axisymmetric_case(
    case_dir: str,
    classification: Optional[ClassificationResult] = None,
    *,
    source_dir: str = "",
    working_copy_dir: Optional[str] = None,
    mode: ImportMode2D = ImportMode2D.IMPORTED_2D_UNINITIALIZED,
) -> ImportedCase2DState:
    """Inspect an AXISYMMETRIC_WEDGE working case without modifying files."""
    case_dir = os.path.normpath(case_dir)
    classification = classification or classify_case_topology(case_dir)
    if classification.classification != CaseDimension.AXISYMMETRIC_WEDGE:
        raise ValueError(
            f"Expected AXISYMMETRIC_WEDGE, got {classification.classification.value}: "
            f"{classification.reason}"
        )

    mapping = map_imported_case_to_gui(case_dir, classification)
    evidence = classification.evidence
    time_dirs = _list_time_dirs(case_dir)
    latest = time_dirs[-1] if time_dirs else None

    mesh_owner = os.path.join(case_dir, "constant", "polyMesh", "owner")
    mesh_present = os.path.isfile(mesh_owner)
    if not mesh_present and latest:
        mesh_present = os.path.isfile(
            os.path.join(case_dir, latest, "polyMesh", "owner")
        )

    cell_count = None
    cell_source = "none"
    owner_path = ""
    time_owner = (
        os.path.join(case_dir, latest, "polyMesh", "owner") if latest else ""
    )
    const_owner = os.path.join(case_dir, "constant", "polyMesh", "owner")
    if latest and os.path.isfile(time_owner):
        owner_path = time_owner
        cell_source = "time_polyMesh"
    elif os.path.isfile(const_owner):
        owner_path = const_owner
        cell_source = "constant_polyMesh"
    if owner_path:
        try:
            text = open(owner_path, encoding="utf-8", errors="ignore").read()
            note_cells = re.search(r"\bnCells:\s*(\d+)", text)
            if note_cells:
                cell_count = int(note_cells.group(1))
            else:
                cell_count = AxisymmetricViewerWidget.count_owner_cells(
                    os.path.dirname(owner_path)
                )
        except OSError:
            cell_count = None

    fields = tuple(_discover_fields(case_dir, latest))
    axis = evidence.symmetry_axis
    display_compatible = axis in (None, "Y")
    notes: List[str] = []
    if axis and axis != "Y":
        notes.append(
            f"Symmetry axis is global {axis}; GGUI Cylindrical–2D viewer currently "
            "supports Y-axial wedges only for meridional display."
        )
        display_compatible = False
    else:
        display_compatible = True
        axis = axis or "Y"
        if evidence.symmetry_axis is None:
            notes.append(
                "Symmetry axis not measured from numeric vertices; "
                "assuming blastFoam Y-axial wedge for display when mesh appears."
            )

    notes.extend(mapping.notes)

    prepare_cmds: Tuple[str, ...] = ()
    try:
        prepare_cmds = preparation_commands_for_case(case_dir)
    except Exception:
        prepare_cmds = ()

    wc = working_copy_dir or case_dir
    src = source_dir or case_dir
    viewable = bool(mesh_present and display_compatible and fields)
    runnable = mode == ImportMode2D.IMPORTED_2D_READY and viewable

    return ImportedCase2DState(
        case_dir=wc,
        classification=classification,
        mode=mode,
        source_dir=src,
        working_copy_dir=wc if working_copy_dir else None,
        mapping=mapping,
        mesh_present=mesh_present,
        time_dirs=tuple(time_dirs),
        latest_time=latest,
        fields=fields,
        cell_count=cell_count,
        cell_count_source=cell_source,
        mesh_owner_path=owner_path,
        viewable=viewable,
        runnable=runnable,
        display_compatible=display_compatible,
        compatibility_notes=tuple(notes),
        prepare_commands=prepare_cmds,
    )


# Backward-compatible name.
def inspect_external_axisymmetric_case(
    case_dir: str,
    classification: Optional[ClassificationResult] = None,
) -> ImportedCase2DState:
    return inspect_imported_axisymmetric_case(case_dir, classification)


def format_imported_case_report_2d(state: ImportedCase2DState) -> str:
    """Structured 2D load report including source/working paths and mapping."""
    ev = state.classification.evidence
    mapping = state.mapping
    lines: List[str] = [
        f"Loaded working case: {state.active_case_path}",
        f"Mode: {state.lifecycle_label}",
        f"Source path (unchanged): {state.source_dir or '(unknown)'}",
        f"Working case path: {state.working_copy_dir or state.case_dir}",
        "",
        "STATEMENT: The selected source directory is never modified.",
        "",
        "1. Case classification",
        "  Classification: Axisymmetric wedge",
        f"  Evidence source: {ev.source}",
        f"  Wedge patches (type wedge): {', '.join(ev.wedge_patch_names) or '(none)'}",
    ]
    if state.wedge_half_angle_deg is not None:
        lines.append(f"  Wedge half-angle: {state.wedge_half_angle_deg:g}°")
    if state.symmetry_axis:
        lines.append(f"  Symmetry axis: global {state.symmetry_axis}")
    lines.extend(["", "2. Case availability"])
    lines.append(f"  Mesh present: {'yes' if state.mesh_present else 'no'}")
    lines.append(
        f"  Available times: {', '.join(state.time_dirs) if state.time_dirs else '(none)'}"
    )
    lines.append(f"  Latest time: {state.latest_time or '(none)'}")
    lines.append(
        f"  Fields found: {', '.join(state.fields) if state.fields else '(none)'}"
    )
    if state.cell_count is not None:
        lines.append(
            f"  Computational cells: {state.cell_count:,} ({state.cell_count_source})"
        )
        if state.mesh_owner_path:
            lines.append(f"  Owner file: {state.mesh_owner_path}")
    else:
        lines.append("  Computational cells: (unavailable until initialise)")
    if state.charge_cell_count is not None:
        lines.append(f"  Charge cells (alpha.c4 > 0): {state.charge_cell_count:,}")
    if state.prepare_commands:
        lines.append(f"  Preparation plan: {' && '.join(state.prepare_commands)}")
    if state.prepare_results:
        lines.append(
            "  Prepare exit codes: "
            + ", ".join(f"{k}={v}" for k, v in state.prepare_results.items())
        )
    if state.check_mesh_ok:
        lines.append("  checkMesh: Mesh OK")

    lines.extend(["", "3. GUI field mapping (provenance)"])
    if mapping:
        for key in sorted(mapping.fields):
            mf = mapping.fields[key]
            val = "—" if mf.displayed_value is None else mf.displayed_value
            if mf.editable:
                edit = "editable"
            elif mf.provenance == FieldProvenance.CASE_DEFINED:
                edit = "case-defined"
            elif mf.provenance == FieldProvenance.NOT_RECOVERED:
                edit = "unrecovered"
            else:
                edit = "info"
            lines.append(
                f"  {key}: {val}  [{mf.provenance.value}|{edit}]"
                + (f"  src={mf.source_file}" if mf.source_file else "")
                + (f"  ({mf.reason})" if mf.reason else "")
            )
    else:
        lines.append("  (no mapping)")

    lines.extend(["", "4. Populated GUI controls"])
    if mapping and mapping.gui_values:
        for key, val in sorted(mapping.gui_values.items()):
            lines.append(f"  {key} = {val}")
    else:
        lines.append("  (none)")

    lines.extend(["", "5. Case-defined (kept for report; controls remain editable)"])
    if mapping and mapping.case_defined_keys:
        for key in mapping.case_defined_keys:
            lines.append(f"  - {key}")
    else:
        lines.append("  (none)")

    lines.extend(
        ["", "6. Not recovered from source (permanent GUI defaults remain available)"]
    )
    if mapping and mapping.not_recovered_keys:
        for key in mapping.not_recovered_keys:
            mf = mapping.fields.get(key)
            reason = mf.reason if mf else ""
            lines.append(f"  - {key}" + (f": {reason}" if reason else ""))
    else:
        lines.append("  (none)")

    lines.extend(["", "7. Initialization / solver compatibility"])
    lines.append(
        f"  Initialise Model: "
        f"{'allowed (GGUI generator path)' if state.initialize_allowed else 'blocked'}"
    )
    lines.append(
        f"  exact END: "
        f"{'allowed after ready' if state.mode == ImportMode2D.IMPORTED_2D_READY else 'disabled until initialized'}"
    )
    lines.append(
        "  Architecture: BF source → editable GGUI model → generator_2d fresh case"
    )
    lines.append("  Source directory: never modified")
    lines.append("  Solver path: blastFoam in the generated GGUI case only")
    lines.append("  Topology: canonical GGUI ±5° wedge (not user-editable)")
    for note in state.compatibility_notes:
        lines.append(f"  - {note}")
    if mapping and mapping.notes:
        lines.extend(["", "8. Mapping notes"])
        for note in mapping.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def format_external_case_report_2d(state: ImportedCase2DState) -> str:
    return format_imported_case_report_2d(state)
