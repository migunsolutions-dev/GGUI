"""Authoritative physical charge geometry (Qt/generator independent).

blastFoam ``cylindericalMassToCell`` uses mass, density and L/D — not a GUI radius
that may disagree. All consumers (Auto Seed d_min, Generator3D, setFields,
capture guard, outer geometry, Mesh Plan, metadata, tests) must use this module.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

SUPPORTED_SHAPES = ("Sphere", "Cylinder", "Cuboid")


@dataclass(frozen=True)
class PhysicalChargeGeometry:
    shape: str
    volume_m3: float
    # Sphere
    radius_m: float = 0.0
    # Cylinder
    cylinder_radius_m: float = 0.0
    length_m: float = 0.0
    aspect_ld: float = 0.0
    # Cuboid
    length_box_m: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0
    # Policy notes
    authoritative: str = ""
    d_min_m: float = 0.0
    d_min_name: str = ""

    def to_dims_dict(self) -> Dict[str, float]:
        """Generator / seed-plan dims dict."""
        s = self.shape
        if s == "Sphere":
            return {"radius": self.radius_m}
        if s == "Cuboid":
            if self.length_box_m > 0 and self.width_m > 0 and self.height_m > 0:
                return {
                    "length": self.length_box_m,
                    "width": self.width_m,
                    "height": self.height_m,
                }
            side = self.volume_m3 ** (1.0 / 3.0)
            return {"side": side, "length": side, "width": side, "height": side}
        return {"radius": self.cylinder_radius_m, "length": self.length_m}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _shape(inputs: Any) -> str:
    return str(getattr(inputs, "charge_shape", "Sphere") or "Sphere").strip()


def _require_positive(name: str, value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be > 0 (got {value!r})")
    return v


def physical_charge_geometry(inputs: Any) -> PhysicalChargeGeometry:
    """Derive the physical charge that blastFoam initialization will create.

    Raises ValueError for unsupported shapes or invalid physical inputs.
    Does not silently substitute defaults for non-positive mass/density/L/D.
    """
    shape = _shape(inputs)
    if shape not in SUPPORTED_SHAPES:
        raise ValueError(
            f"Unsupported charge shape {shape!r}. "
            f"Supported shapes: {', '.join(SUPPORTED_SHAPES)}."
        )

    mass = _require_positive("mass_kg", getattr(inputs, "mass_kg", None))
    rho = _require_positive("rho_charge", getattr(inputs, "rho_charge", None))
    vol = mass / rho

    if shape == "Sphere":
        r = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
        return PhysicalChargeGeometry(
            shape=shape,
            volume_m3=vol,
            radius_m=r,
            authoritative="mass_rho_sphere",
            d_min_m=max(2.0 * r, 1e-12),
            d_min_name="diameter",
        )

    if shape == "Cuboid":
        L = float(getattr(inputs, "charge_length", 0.0) or 0.0)
        W = float(getattr(inputs, "charge_width", 0.0) or 0.0)
        H = float(getattr(inputs, "charge_height", 0.0) or 0.0)
        if (
            L > 0
            and W > 0
            and H > 0
            and abs(L * W * H - vol) <= 0.02 * vol
        ):
            dmin = max(min(L, W, H), 1e-12)
            return PhysicalChargeGeometry(
                shape=shape,
                volume_m3=vol,
                length_box_m=L,
                width_m=W,
                height_m=H,
                authoritative="explicit_box_matching_mass",
                d_min_m=dmin,
                d_min_name="min_edge",
            )
        side = vol ** (1.0 / 3.0)
        return PhysicalChargeGeometry(
            shape=shape,
            volume_m3=vol,
            length_box_m=side,
            width_m=side,
            height_m=side,
            authoritative="mass_rho_cube",
            d_min_m=max(side, 1e-12),
            d_min_name="cube_side",
        )

    # Cylinder — mass + ρ + L/D are authoritative for cylindericalMassToCell.
    aspect = _require_positive(
        "charge_aspect (L/D)", getattr(inputs, "charge_aspect", None)
    )
    # V = π r² L with L = 2 r (L/D) ⇒ r³ = V / (2 π aspect)
    r = (vol / (2.0 * math.pi * aspect)) ** (1.0 / 3.0)
    length = 2.0 * r * aspect
    dmin = max(min(2.0 * r, length), 1e-12)
    dname = "diameter" if 2.0 * r <= length else "length"
    return PhysicalChargeGeometry(
        shape="Cylinder",
        volume_m3=vol,
        cylinder_radius_m=r,
        length_m=length,
        aspect_ld=aspect,
        authoritative="mass_rho_LbyD_cylindericalMassToCell",
        d_min_m=dmin,
        d_min_name=dname,
    )


def sync_derived_cylinder_fields(inputs: Any) -> Any:
    """Return inputs with derived cylinder radius/length (frozen-dataclass safe).

    Does not change mass, rho, or L/D. Uses dataclasses.replace when frozen.
    """
    if _shape(inputs) != "Cylinder":
        return inputs
    geom = physical_charge_geometry(inputs)
    r = float(geom.cylinder_radius_m)
    L = float(geom.length_m)
    try:
        return replace(inputs, cylinder_radius=r, charge_length=L)
    except Exception:
        try:
            inputs.cylinder_radius = r
            inputs.charge_length = L
        except Exception:
            pass
        return inputs


def dims_dict_from_inputs(inputs: Any) -> Dict[str, float]:
    return physical_charge_geometry(inputs).to_dims_dict()


def _vec3(value: Any) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


def _near(a: float, b: float, *, scale: float, tol: float) -> bool:
    return abs(a - b) <= max(tol * max(abs(scale), 1e-9), 1e-9)


def _vec_near(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    *,
    scale: float,
    tol: float,
) -> bool:
    return all(_near(a[i], b[i], scale=scale, tol=tol) for i in range(3))


def canonical_outside_extent_from_outer_geometry(
    geom_info: Dict[str, Any],
    phys: PhysicalChargeGeometry,
    *,
    charge_center: Optional[Any] = None,
    cylinder_axis: str = "Z",
    tol: float = 1e-4,
) -> Optional[float]:
    """Return uniform shell thickness [m] only for a canonical GGUI outer shell.

    Requires matching centre (and cylinder axis). Non-canonical imported geometry
    returns None so callers preserve searchable* parameters unchanged.
    """
    if not geom_info:
        return None
    centre = _vec3(charge_center)
    if centre is None:
        # Without a charge centre, refuse to invent a scalar extent.
        return None

    gtype = geom_info.get("type")
    if gtype == "searchableSphere" and "radius" in geom_info:
        gc = _vec3(geom_info.get("centre"))
        if gc is None:
            return None
        scale = max(float(phys.radius_m), float(geom_info["radius"]), 1e-9)
        if not _vec_near(gc, centre, scale=scale, tol=tol):
            return None
        return max(0.0, float(geom_info["radius"]) - float(phys.radius_m))

    if gtype == "searchableCylinder" and "radius" in geom_info:
        p1 = _vec3(geom_info.get("point1"))
        p2 = _vec3(geom_info.get("point2"))
        if p1 is None or p2 is None:
            return None
        mid = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2]))
        axis = str(cylinder_axis or "Z").upper()
        axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(axis, 2)
        direction = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        length = math.sqrt(sum(v * v for v in direction))
        if length <= 1e-15:
            return None
        # Axis must be parallel to the configured charge axis (only that component nonzero).
        unit = [abs(direction[i] / length) for i in range(3)]
        if unit[axis_idx] < 1.0 - max(tol, 1e-6):
            return None
        for i in range(3):
            if i == axis_idx:
                continue
            if unit[i] > max(tol, 1e-6):
                return None
        scale = max(float(phys.cylinder_radius_m), float(phys.length_m), float(geom_info["radius"]), 1e-9)
        if not _vec_near(mid, centre, scale=scale, tol=tol):
            return None
        half_outer = 0.5 * length
        extent_r = float(geom_info["radius"]) - float(phys.cylinder_radius_m)
        extent_a = half_outer - 0.5 * float(phys.length_m)
        scale_e = max(abs(extent_r), abs(extent_a), 1e-9)
        if abs(extent_r - extent_a) > max(tol * scale_e, 1e-9):
            return None
        return max(0.0, 0.5 * (extent_r + extent_a))

    if gtype == "searchableBox" and "min" in geom_info and "max" in geom_info:
        mn = _vec3(geom_info["min"])
        mx = _vec3(geom_info["max"])
        if mn is None or mx is None:
            return None
        box_c = (0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), 0.5 * (mn[2] + mx[2]))
        half_outer = [0.5 * (mx[i] - mn[i]) for i in range(3)]
        if phys.shape == "Cuboid":
            half_phys = [
                phys.length_box_m / 2.0,
                phys.width_m / 2.0,
                phys.height_m / 2.0,
            ]
        elif phys.shape == "Sphere":
            r = phys.radius_m
            half_phys = [r, r, r]
        else:
            # Box outer around a cylinder is not a canonical GGUI shell.
            return None
        scale = max(max(half_phys), max(half_outer), 1e-9)
        if not _vec_near(box_c, centre, scale=scale, tol=tol):
            return None
        extents = [half_outer[i] - half_phys[i] for i in range(3)]
        if min(extents) < -1e-9:
            return None
        lo, hi = min(extents), max(extents)
        scale_e = max(abs(hi), 1e-9)
        if (hi - lo) > max(tol * scale_e, 1e-9):
            return None
        return max(0.0, 0.5 * (lo + hi))

    return None


def format_searchable_outer_geometry(geom: Dict[str, Any]) -> str:
    """Emit a snappy geometry{} entry from preserved searchable* parameters."""
    gtype = str(geom.get("type") or "")
    if gtype == "searchableSphere":
        c = _vec3(geom.get("centre"))
        r = float(geom["radius"])
        if c is None:
            raise ValueError("Imported searchableSphere is missing centre")
        return (
            f"    chargeRefineOuter {{ type searchableSphere; "
            f"centre ({c[0]:.6g} {c[1]:.6g} {c[2]:.6g}); radius {r:.6g}; }}\n"
        )
    if gtype == "searchableCylinder":
        p1 = _vec3(geom.get("point1"))
        p2 = _vec3(geom.get("point2"))
        r = float(geom["radius"])
        if p1 is None or p2 is None:
            raise ValueError("Imported searchableCylinder is missing point1/point2")
        return (
            f"    chargeRefineOuter {{\n"
            f"        type searchableCylinder;\n"
            f"        point1 ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g});\n"
            f"        point2 ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g});\n"
            f"        radius {r:.6g};\n"
            f"    }}\n"
        )
    if gtype == "searchableBox":
        mn = _vec3(geom.get("min"))
        mx = _vec3(geom.get("max"))
        if mn is None or mx is None:
            raise ValueError("Imported searchableBox is missing min/max")
        return (
            f"    chargeRefineOuter {{\n"
            f"        type searchableBox;\n"
            f"        min ({mn[0]:.6g} {mn[1]:.6g} {mn[2]:.6g});\n"
            f"        max ({mx[0]:.6g} {mx[1]:.6g} {mx[2]:.6g});\n"
            f"    }}\n"
        )
    raise ValueError(
        f"Unsupported or incomplete imported chargeRefineOuter geometry type {gtype!r}"
    )
