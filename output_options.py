"""Shared Output File Options for 1D / 2D / 3D (blastFoam gauges and VTK writes).

ASII / HVEL / RISK are omitted (no GGUI equivalent). remap2d.ggui, arrival times,
and obstacle IDs are kept where the screenshots require them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, List, Sequence, Tuple


# Dialog row key -> OpenFOAM field name sampled by probes.
GAUGE_FOAM_FIELDS: Dict[str, str] = {
    "overpressure": "p",
    "pressure": "p",
    "impulse": "impulse",
    "density": "rho",
    "velocity": "U",
    "mass_fractions": "alpha.c4",
    "temperature": "T",
    "energy": "rhoE",
    "dynamic_pressure": "dynamicPressure",
}

GAUGE_LABELS_1D: Tuple[Tuple[str, str], ...] = (
    ("overpressure", "Overpressure"),
    ("impulse", "Impulse"),
    ("density", "Density"),
    ("velocity", "Velocity"),
    ("mass_fractions", "Mass fractions"),
    ("temperature", "Temperature"),
    ("energy", "Energy"),
    ("dynamic_pressure", "Dynamic pressure"),
)

GAUGE_LABELS_2D: Tuple[Tuple[str, str], ...] = (
    ("pressure", "Pressure"),
    ("impulse", "Impulse"),
    ("density", "Density"),
    ("velocity", "Velocity"),
    ("mass_fractions", "Mass fractions"),
    ("temperature", "Temperature"),
    ("energy", "Energy"),
    ("dynamic_pressure", "Dynamic pressure"),
)

VTK_KEYS_2D: Tuple[str, ...] = (
    "pressure",
    "density",
    "velocity",
    "mass_fractions",
    "temperature",
    "energy",
)

REMAP_2D_FILENAME = "remap2d.ggui"

VTK_TO_OUTPUT_CHECK: Dict[str, str] = {
    "pressure": "p",
    "density": "rho",
    "temperature": "T",
    "velocity": "U",
    "mass_fractions": "alpha.c4",
    "energy": "rhoE",
}

COL_GAUGES = "gauges"
COL_SECTIONS = "sections"
COL_OBSTACLES = "obstacles"
COL_VOLUMES = "volumes"
QUANTITY_COLUMNS_3D: Tuple[str, ...] = (COL_GAUGES, COL_SECTIONS, COL_OBSTACLES, COL_VOLUMES)
QUANTITY_COLUMN_LABELS_3D: Dict[str, str] = {
    COL_GAUGES: "Gauges",
    COL_SECTIONS: "Sections",
    COL_OBSTACLES: "Obstacles",
    COL_VOLUMES: "Volumes",
}

# Row key, label, columns that have a checkbox (screenshot matrix).
QUANTITY_ROWS_3D: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("pressure", "Pressure", QUANTITY_COLUMNS_3D),
    ("impulse", "Impulse", (COL_GAUGES,)),
    ("peak_overpressure", "Peak overpressure", QUANTITY_COLUMNS_3D),
    ("peak_impulse", "Peak impulse", QUANTITY_COLUMNS_3D),
    ("density", "Density", QUANTITY_COLUMNS_3D),
    ("velocity", "Velocity", QUANTITY_COLUMNS_3D),
    ("mass_fractions", "Mass fractions", QUANTITY_COLUMNS_3D),
    ("temperature", "Temperature", QUANTITY_COLUMNS_3D),
    ("energy", "Energy", QUANTITY_COLUMNS_3D),
    ("dynamic_pressure", "Dynamic pressure", (COL_GAUGES,)),
    ("arrival_initial", "Arrival time (initial)", (COL_OBSTACLES,)),
    ("arrival_peak", "Arrival time (peak)", (COL_OBSTACLES,)),
    ("obstacle_id", "Obstacle ID", (COL_OBSTACLES,)),
)

QUANTITY_FOAM: Dict[str, str] = {
    "pressure": "p",
    "impulse": "impulse",
    "peak_overpressure": "overpressure",
    "peak_impulse": "impulse",
    "density": "rho",
    "velocity": "U",
    "mass_fractions": "alpha.c4",
    "temperature": "T",
    "energy": "rhoE",
    "dynamic_pressure": "dynamicPressure",
    "arrival_initial": "overpressure",
    "arrival_peak": "overpressure",
    "obstacle_id": "obstacleId",
}

DEFAULT_3D_QUANTITY_ON: frozenset = frozenset(
    {
        ("pressure", COL_GAUGES),
        ("pressure", COL_SECTIONS),
        ("pressure", COL_OBSTACLES),
        ("impulse", COL_GAUGES),
        ("peak_overpressure", COL_SECTIONS),
        ("peak_overpressure", COL_OBSTACLES),
        ("peak_impulse", COL_SECTIONS),
        ("peak_impulse", COL_OBSTACLES),
    }
)


def default_3d_quantities() -> Dict[str, Dict[str, bool]]:
    table: Dict[str, Dict[str, bool]] = {}
    for key, _label, cols in QUANTITY_ROWS_3D:
        table[key] = {col: ((key, col) in DEFAULT_3D_QUANTITY_ON) for col in cols}
    return table


def foam_fields_for_column(quantities: Dict[str, Dict[str, bool]], column: str) -> Tuple[str, ...]:
    names: List[str] = []
    seen = set()
    for key, _label, cols in QUANTITY_ROWS_3D:
        if column not in cols:
            continue
        if not quantities.get(key, {}).get(column, False):
            continue
        foam = QUANTITY_FOAM.get(key)
        if not foam or foam in seen:
            continue
        if key == "obstacle_id":
            continue
        seen.add(foam)
        names.append(foam)
    return tuple(names) if names else (("p",) if column == COL_GAUGES else ())


@dataclass
class GaugeFlags:
    overpressure: bool = True
    pressure: bool = True
    impulse: bool = True
    density: bool = False
    velocity: bool = False
    mass_fractions: bool = False
    temperature: bool = False
    energy: bool = False
    dynamic_pressure: bool = False
    peak_overpressure: bool = False
    peak_impulse: bool = False

    def foam_probe_fields(self, *, always_p: bool = False) -> Tuple[str, ...]:
        names: List[str] = []
        if always_p or self.overpressure or self.pressure:
            names.append("p")
        if self.impulse or self.peak_impulse:
            names.append("impulse")
        if self.density:
            names.append("rho")
        if self.velocity:
            names.append("U")
        if self.mass_fractions:
            names.append("alpha.c4")
        if self.temperature:
            names.append("T")
        if self.energy:
            names.append("rhoE")
        if self.dynamic_pressure:
            names.append("dynamicPressure")
        # Preserve order, drop duplicates.
        seen = set()
        out: List[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return tuple(out) if out else ("p",)


@dataclass
class Dim1DOutput:
    gauges: GaugeFlags = field(default_factory=GaugeFlags)


@dataclass
class Dim2DOutput:
    keep_openfoam_time_folders: bool = False
    cycle_write: int = 0
    vtk_by_time: bool = True
    vtk_time_s: float = 0.001
    vtk_steps: int = 25
    gauges: GaugeFlags = field(default_factory=GaugeFlags)
    vtk: GaugeFlags = field(
        default_factory=lambda: GaugeFlags(
            pressure=True,
            impulse=False,
            density=False,
            velocity=False,
            mass_fractions=False,
            temperature=False,
            energy=False,
            dynamic_pressure=False,
        )
    )
    output_remap_data: bool = False


@dataclass
class Dim3DOutput:
    keep_openfoam_time_folders: bool = False
    cycle_write: int = 0
    write_surfaces: bool = True
    surface_by_time: bool = True
    surface_time_s: float = 0.001
    surface_steps: int = 25
    write_volumes: bool = False
    vtk_by_time: bool = True
    vtk_time_s: float = 0.001
    vtk_steps: int = 25
    quantities: Dict[str, Dict[str, bool]] = field(default_factory=default_3d_quantities)

    def fields_for(self, column: str) -> Tuple[str, ...]:
        return foam_fields_for_column(self.quantities, column)

    def checked(self, key: str, column: str) -> bool:
        return bool(self.quantities.get(key, {}).get(column, False))

    def any_peaks(self) -> bool:
        for key in ("peak_overpressure", "peak_impulse"):
            if any(self.quantities.get(key, {}).values()):
                return True
        return False


@dataclass
class OutputFileOptions:
    dim1d: Dim1DOutput = field(default_factory=Dim1DOutput)
    dim2d: Dim2DOutput = field(default_factory=Dim2DOutput)
    dim3d: Dim3DOutput = field(default_factory=Dim3DOutput)


def _known_values(cls, data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


def output_file_options_from_dict(
    data: Any,
    *,
    legacy_cycle_write_2d: int = 0,
    legacy_cycle_write_3d: int = 0,
) -> OutputFileOptions:
    """Restore persisted dialog state while tolerating missing legacy keys.

    Output-options payloads written before native-time retention was explicit
    are migrated to ``keep_openfoam_time_folders=True``.  That preserves the
    old behavior; new ``OutputFileOptions`` instances still default to False.
    """
    if not isinstance(data, dict):
        return OutputFileOptions()
    raw_1d = data.get("dim1d") if isinstance(data.get("dim1d"), dict) else {}
    raw_2d = data.get("dim2d") if isinstance(data.get("dim2d"), dict) else {}
    raw_3d = data.get("dim3d") if isinstance(data.get("dim3d"), dict) else {}
    values_2d = _known_values(Dim2DOutput, raw_2d)
    values_3d = _known_values(Dim3DOutput, raw_3d)
    if "keep_openfoam_time_folders" not in raw_2d:
        values_2d["keep_openfoam_time_folders"] = True
    if "keep_openfoam_time_folders" not in raw_3d:
        values_3d["keep_openfoam_time_folders"] = True
    if "cycle_write" not in raw_2d:
        values_2d["cycle_write"] = int(legacy_cycle_write_2d)
    if "cycle_write" not in raw_3d:
        values_3d["cycle_write"] = int(legacy_cycle_write_3d)
    return OutputFileOptions(
        dim1d=Dim1DOutput(
            gauges=GaugeFlags(**_known_values(GaugeFlags, raw_1d.get("gauges")))
        ),
        dim2d=Dim2DOutput(
            **{
                **values_2d,
                "gauges": GaugeFlags(
                    **_known_values(GaugeFlags, raw_2d.get("gauges"))
                ),
                "vtk": GaugeFlags(**_known_values(GaugeFlags, raw_2d.get("vtk"))),
            }
        ),
        dim3d=Dim3DOutput(**values_3d),
    )


def extra_function_objects(
    *,
    p_atm: float,
    impulse: bool,
    overpressure: bool,
    dynamic_pressure: bool,
    peaks: bool,
) -> str:
    """controlDict functions entries beyond probes (blastFoam field FOs)."""
    chunks: List[str] = []
    if impulse or peaks:
        chunks.append(
            f"""    impulse
    {{
        type            impulse;
        libs            ("libfieldFunctionObjects.so");
        executeControl  timeStep;
        executeInterval 1;
        writeControl    writeTime;
        pRef            {float(p_atm):.10g};
    }}"""
        )
    if overpressure or peaks:
        chunks.append(
            f"""    overpressure
    {{
        type            overpressure;
        libs            ("libfieldFunctionObjects.so");
        executeControl  timeStep;
        executeInterval 1;
        writeControl    writeTime;
        store           yes;
        pRef            {float(p_atm):.10g};
    }}"""
        )
    if dynamic_pressure:
        chunks.append(
            """    dynamicPressure
    {
        type            pressure;
        libs            ("libfieldFunctionObjects.so");
        mode            dynamic;
        result          dynamicPressure;
        executeControl  timeStep;
        executeInterval 1;
        writeControl    writeTime;
    }"""
        )
    if peaks:
        chunks.append(
            """    maxPImpulse
    {
        type            fieldMinMax;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        fields
        (
            overpressure
            impulse
        );
    }"""
        )
    if not chunks:
        return ""
    return "\n".join(chunks) + "\n"


def foam_ident(name: str, prefix: str = "s") -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name).strip()) or prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def section_plane_point(
    normal: Sequence[float],
    position_m: float,
    min_point: Sequence[float],
    max_point: Sequence[float],
) -> Tuple[float, float, float]:
    """Origin of a GGUI XY/XZ/YZ section, clamped to the domain box."""
    xmin, ymin, zmin = (float(v) for v in min_point)
    xmax, ymax, zmax = (float(v) for v in max_point)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    pos = float(position_m)
    nx, ny, nz = (float(normal[0]), float(normal[1]), float(normal[2]))
    if abs(nz) >= 0.9:
        return (cx, cy, min(max(pos, zmin), zmax))
    if abs(ny) >= 0.9:
        return (cx, min(max(pos, ymin), ymax), cz)
    return (min(max(pos, xmin), xmax), cy, cz)


def surfaces_vtk_block(
    *,
    name: str = "surfacesVTK",
    by_time: bool = True,
    interval_time: float = 0.001,
    interval_steps: int = 25,
    fields: Iterable[str],
    planes: Sequence[Tuple[str, float, float, float, float, float, float]] = (),
    patches: Sequence[str] = (),
) -> str:
    """controlDict surfaces FO writing VTK for cutting planes and obstacle walls."""
    field_list = " ".join(dict.fromkeys(fields)) or "p"
    write_control = "adjustableRunTime" if by_time else "timeStep"
    write_interval = f"{float(interval_time):.10g}" if by_time else str(int(interval_steps))
    surface_chunks: List[str] = []
    used_names = set()
    for name, px, py, pz, nx, ny, nz in planes:
        ident = foam_ident(name, "section")
        base = ident
        n = 2
        while ident in used_names:
            ident = f"{base}_{n}"
            n += 1
        used_names.add(ident)
        surface_chunks.append(
            f"""        {ident}
        {{
            type            cuttingPlane;
            planeType       pointAndNormal;
            pointAndNormalDict
            {{
                point       ({float(px):.10g} {float(py):.10g} {float(pz):.10g});
                normal      ({float(nx):.10g} {float(ny):.10g} {float(nz):.10g});
            }}
            interpolate     true;
        }}"""
        )
    for patch in patches:
        ident = foam_ident(patch, "wall")
        base = ident
        n = 2
        while ident in used_names:
            ident = f"{base}_{n}"
            n += 1
        used_names.add(ident)
        surface_chunks.append(
            f"""        {ident}
        {{
            type            patch;
            patches         ({patch});
            interpolate     true;
        }}"""
        )
    if not surface_chunks:
        return ""
    joined = "\n".join(surface_chunks)
    ident = foam_ident(name, "surfacesVTK")
    return f"""    {ident}
    {{
        type            surfaces;
        libs            ("libsampling.so");
        writeControl    {write_control};
        writeInterval   {write_interval};
        surfaceFormat   vtk;
        interpolationScheme cellPoint;
        fields          ({field_list});
        surfaces
        (
{joined}
        );
    }}
"""


def obstacle_monitor_block(patches: Sequence[str]) -> str:
    """Time history of max overpressure on each obstacle patch (arrival)."""
    chunks: List[str] = []
    for patch in patches:
        ident = foam_ident(f"arr_{patch}", "arr")
        chunks.append(
            f"""    {ident}
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        regionType      patch;
        name            {patch};
        operation       max;
        fields          (overpressure);
        writeFields     false;
    }}"""
        )
    if not chunks:
        return ""
    return "\n".join(chunks) + "\n"


def remap2d_snapshot_contents(time_value: str = "latest") -> str:
    return (
        f"/* GGUI 2D remap snapshot ({REMAP_2D_FILENAME}) */\n"
        f"time            {time_value};\n"
        'sourceCase      ".";\n'
        "fields          (p rho U T alpha.c4);\n"
    )
