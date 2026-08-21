"""Shared Output File Options for 1D / 2D / 3D (blastFoam gauges and VTK writes).

Viper-only products are omitted: remap2d.vip, ASII/HVEL/RISK, arrival/ignition
times, obstacle IDs, and VTK decimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


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
    "impulse",
    "density",
    "velocity",
    "mass_fractions",
    "temperature",
    "energy",
)

VTK_TO_OUTPUT_CHECK: Dict[str, str] = {
    "pressure": "p",
    "density": "rho",
    "temperature": "T",
    "velocity": "U",
    "mass_fractions": "alpha.c4",
}


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


@dataclass
class Dim3DOutput:
    write_surfaces: bool = True
    surface_by_time: bool = True
    surface_time_s: float = 0.001
    surface_steps: int = 25
    write_volumes: bool = True
    vtk_by_time: bool = True
    vtk_time_s: float = 0.001
    vtk_steps: int = 25
    gauges: GaugeFlags = field(default_factory=GaugeFlags)
    peak_overpressure: bool = False
    peak_impulse: bool = False


@dataclass
class OutputFileOptions:
    dim1d: Dim1DOutput = field(default_factory=Dim1DOutput)
    dim2d: Dim2DOutput = field(default_factory=Dim2DOutput)
    dim3d: Dim3DOutput = field(default_factory=Dim3DOutput)


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
        writeControl    timeStep;
        writeInterval   1;
        pRef            {float(p_atm):.10g};
    }}"""
        )
    if overpressure or peaks:
        chunks.append(
            f"""    overpressure
    {{
        type            overpressure;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
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
        writeControl    timeStep;
        writeInterval   1;
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
    by_time: bool,
    interval_time: float,
    interval_steps: int,
    fields: Iterable[str],
    planes: Sequence[Tuple[str, float, float, float, float, float, float]],
    patches: Sequence[str],
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
    return f"""    surfacesVTK
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
