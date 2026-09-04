"""VIPER TEST2.vip HDF5 gauge schema (read-only inspection).

Verified from C:\\Users\\migun\\Desktop\\1TEST\\TEST2.vip. Do not guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

GROUP = "vipermodel"

NUM_1D = "numthloc_1d"
NUM_2D = "numthloc_2d"
NUM_3D = "numthloc_3d"
X_1D = "thlocx_1d"
X_2D = "thlocx_2d"
Y_2D = "thlocy_2d"
X_3D = "thlocx_3d"
Y_3D = "thlocy_3d"
Z_3D = "thlocz_3d"
LABEL_1D = "th1dlabel_{i}"
LABEL_2D = "th2dlabel_{i}"

FLAG_1D_P = "outputQuantitiesFlag_1D_Gauges_Pressure"
FLAG_1D_I = "outputQuantitiesFlag_1D_Gauges_Impulse"
FLAG_2D_P = "outputQuantitiesFlag_2D_Gauges_Pressure"
FLAG_2D_I = "outputQuantitiesFlag_2D_Gauges_Impulse"

GAUGE_DATASETS_1D = (NUM_1D, X_1D)
GAUGE_DATASETS_2D = (NUM_2D, X_2D, Y_2D)


@dataclass(frozen=True)
class GaugeSchema:
    """Observed TEST2 gauge layout."""

    numthloc_1d: int
    thlocx_1d: List[float]
    labels_1d: List[str]
    numthloc_2d: int
    thlocx_2d: List[float]
    thlocy_2d: List[float]
    labels_2d: List[str]
    pressure_1d: int
    impulse_1d: int
    pressure_2d: int
    impulse_2d: int
    extra: Dict[str, Any] = field(default_factory=dict)


def _scalar_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tobytes") and getattr(value, "shape", ()) == ():
        raw = bytes(value)
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def _arr(value: Any) -> List[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        data = value.tolist()
    else:
        data = list(value)
    if not isinstance(data, list):
        data = [data]
    return [float(x) for x in data]


def _i(value: Any) -> int:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        value = value[0]
    return int(value)


def extract_gauge_schema(path: str) -> GaugeSchema:
    import h5py

    with h5py.File(path, "r") as handle:
        g = handle[GROUP]
        n1 = _i(g[NUM_1D][()])
        n2 = _i(g[NUM_2D][()])
        labels_1d = []
        for i in range(n1):
            name = LABEL_1D.format(i=i)
            if name not in g:
                raise KeyError(f"missing {name}")
            labels_1d.append(_scalar_str(g[name][()]))
        labels_2d = []
        for i in range(n2):
            name = LABEL_2D.format(i=i)
            if name not in g:
                raise KeyError(f"missing {name}")
            labels_2d.append(_scalar_str(g[name][()]))
        return GaugeSchema(
            numthloc_1d=n1,
            thlocx_1d=_arr(g[X_1D][()]),
            labels_1d=labels_1d,
            numthloc_2d=n2,
            thlocx_2d=_arr(g[X_2D][()]),
            thlocy_2d=_arr(g[Y_2D][()]),
            labels_2d=labels_2d,
            pressure_1d=_i(g[FLAG_1D_P][()]),
            impulse_1d=_i(g[FLAG_1D_I][()]),
            pressure_2d=_i(g[FLAG_2D_P][()]),
            impulse_2d=_i(g[FLAG_2D_I][()]),
        )


def schema_report() -> str:
    return """
VIPER TEST2.vip HDF5 gauge schema (group /vipermodel)

1D gauges
  numthloc_1d                 int32[1]
  thlocx_1d                   float32[N]  (fixed maxshape=N; recreate to change N)
  th1dlabel_{i}               scalar |S*  one dataset per gauge, i = 0 .. N-1

2D gauges
  numthloc_2d                 int32[1]
  thlocx_2d                   float32[N]
  thlocy_2d                   float32[N]
  th2dlabel_{i}               scalar |S*  one dataset per gauge

Output flags (int32[1], 1=enabled)
  outputQuantitiesFlag_1D_Gauges_Pressure
  outputQuantitiesFlag_1D_Gauges_Impulse
  outputQuantitiesFlag_2D_Gauges_Pressure
  outputQuantitiesFlag_2D_Gauges_Impulse

Changing gauges requires ALL of:
  count, coordinate arrays, per-index label datasets, and output flags.
Coordinate-only edits are not sufficient if N or labels change.
""".strip()
