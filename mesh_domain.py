"""Align 3D blockMesh domain extents to an integer number of cells at the requested cell size.

Keeps the minimum corner fixed and adjusts each axis maximum so
``L_actual = n * cell_size`` with ``n = max(1, round(L_requested / cell_size))``.
The physical cell size stays equal to the user-requested value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class DomainAlignmentResult:
    requested_min_point: Vec3
    requested_max_point: Vec3
    min_point: Vec3
    max_point: Vec3
    cell_size: float
    nx: int
    ny: int
    nz: int
    requested_lengths: Vec3
    actual_lengths: Vec3
    adjusted: bool
    info_messages: Tuple[str, ...]

    def to_case_metadata(self) -> dict:
        return {
            "requested_domain_min": list(self.requested_min_point),
            "requested_domain_max": list(self.requested_max_point),
            "actual_domain_min": list(self.min_point),
            "actual_domain_max": list(self.max_point),
            "requested_cell_size_m": self.cell_size,
            "actual_cell_size_m": self.cell_size,
            "n_cells_xyz": [self.nx, self.ny, self.nz],
            "requested_lengths_m": list(self.requested_lengths),
            "actual_lengths_m": list(self.actual_lengths),
            "domain_adjusted_for_cell_fit": self.adjusted,
            "info_messages": list(self.info_messages),
        }


def align_domain_to_cell_size(min_point: Vec3, max_point: Vec3, cell_size: float) -> DomainAlignmentResult:
    dx = max(1e-12, float(cell_size))
    min_x, min_y, min_z = float(min_point[0]), float(min_point[1]), float(min_point[2])
    max_x, max_y, max_z = float(max_point[0]), float(max_point[1]), float(max_point[2])
    lx_req = abs(max_x - min_x)
    ly_req = abs(max_y - min_y)
    lz_req = abs(max_z - min_z)

    def _axis_len_cells(l_req: float) -> tuple:
        n = max(1, int(round(l_req / dx)))
        l_act = n * dx
        return n, l_act

    nx, ax = _axis_len_cells(lx_req)
    ny, ay = _axis_len_cells(ly_req)
    nz, az = _axis_len_cells(lz_req)

    # Preserve min corner; extend/shrink max along positive span direction.
    max_x_a = min_x + ax if max_x >= min_x else min_x - ax
    max_y_a = min_y + ay if max_y >= min_y else min_y - ay
    max_z_a = min_z + az if max_z >= min_z else min_z - az

    req = (lx_req, ly_req, lz_req)
    act = (ax, ay, az)
    adjusted = any(abs(req[i] - act[i]) > 1e-9 * max(1.0, req[i]) for i in range(3))
    msgs: List[str] = []
    if adjusted:
        msgs.append(
            "The domain length was adjusted to match the requested cell size exactly. "
            f"Requested lengths (x,y,z) [m]: {lx_req:.6g}, {ly_req:.6g}, {lz_req:.6g}. "
            f"Actual lengths [m]: {ax:.6g}, {ay:.6g}, {az:.6g}. "
            f"Cell size [m]: {dx:.6g}. Cells (nx,ny,nz): {nx}, {ny}, {nz}."
        )
    return DomainAlignmentResult(
        requested_min_point=(min_x, min_y, min_z),
        requested_max_point=(max_x, max_y, max_z),
        min_point=(min_x, min_y, min_z),
        max_point=(max_x_a, max_y_a, max_z_a),
        cell_size=dx,
        nx=nx,
        ny=ny,
        nz=nz,
        requested_lengths=req,
        actual_lengths=act,
        adjusted=adjusted,
        info_messages=tuple(msgs),
    )
