"""Copy a VIPER .vip and replace 1D/2D gauges using the verified TEST2 schema."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from viper_compare.schema import (
    FLAG_1D_I,
    FLAG_1D_P,
    FLAG_2D_I,
    FLAG_2D_P,
    GROUP,
    LABEL_1D,
    LABEL_2D,
    NUM_1D,
    NUM_2D,
    X_1D,
    X_2D,
    Y_2D,
    extract_gauge_schema,
)

Gauge1D = Tuple[float, str]
Gauge2D = Tuple[float, float, str]


def copy_vip(src: str, dest: str) -> str:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_path)
    return str(dest_path)


def _bytes_scalar(text: str) -> np.ndarray:
    encoded = str(text).encode("ascii", errors="replace")
    return np.array(encoded, dtype=f"|S{max(1, len(encoded))}")


def _replace_array(group, name: str, values: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=values)


def _replace_labels(group, pattern: str, labels: Sequence[str]) -> None:
    prefix = pattern.split("{", 1)[0]
    matcher = re.compile(r"^" + re.escape(prefix) + r"\d+$")
    existing = [key for key in group.keys() if matcher.match(key)]
    for key in existing:
        del group[key]
    for i, label in enumerate(labels):
        group.create_dataset(f"{prefix}{i}", data=_bytes_scalar(label))


def _flag(group, name: str, enabled: bool) -> None:
    _replace_array(group, name, np.array([1 if enabled else 0], dtype=np.int32))


def set_gauges(
    vip_path: str,
    *,
    gauges_1d: Sequence[Gauge1D],
    gauges_2d: Sequence[Gauge2D],
    pressure: bool = True,
    impulse: bool = True,
) -> None:
    import h5py

    xs1 = np.asarray([float(g[0]) for g in gauges_1d], dtype=np.float32)
    labels1 = [str(g[1]) for g in gauges_1d]
    xs2 = np.asarray([float(g[0]) for g in gauges_2d], dtype=np.float32)
    ys2 = np.asarray([float(g[1]) for g in gauges_2d], dtype=np.float32)
    labels2 = [str(g[2]) for g in gauges_2d]
    with h5py.File(vip_path, "r+") as handle:
        g = handle[GROUP]
        _replace_array(g, NUM_1D, np.array([len(gauges_1d)], dtype=np.int32))
        _replace_array(g, X_1D, xs1)
        _replace_labels(g, LABEL_1D, labels1)
        _replace_array(g, NUM_2D, np.array([len(gauges_2d)], dtype=np.int32))
        _replace_array(g, X_2D, xs2)
        _replace_array(g, Y_2D, ys2)
        _replace_labels(g, LABEL_2D, labels2)
        _flag(g, FLAG_1D_P, pressure)
        _flag(g, FLAG_1D_I, impulse)
        _flag(g, FLAG_2D_P, pressure)
        _flag(g, FLAG_2D_I, impulse)


def validate_gauges(
    vip_path: str,
    *,
    gauges_1d: Sequence[Gauge1D],
    gauges_2d: Sequence[Gauge2D],
    pressure: bool = True,
    impulse: bool = True,
    atol: float = 1.0e-6,
) -> None:
    schema = extract_gauge_schema(vip_path)
    if schema.numthloc_1d != len(gauges_1d):
        raise AssertionError(f"1D count {schema.numthloc_1d} != {len(gauges_1d)}")
    if schema.numthloc_2d != len(gauges_2d):
        raise AssertionError(f"2D count {schema.numthloc_2d} != {len(gauges_2d)}")
    for got, (x, label) in zip(schema.thlocx_1d, gauges_1d):
        if abs(got - float(x)) > atol:
            raise AssertionError(f"1D x {got} != {x}")
    if schema.labels_1d != [g[1] for g in gauges_1d]:
        raise AssertionError(f"1D labels {schema.labels_1d}")
    for (gx, gy), (x, y, _label) in zip(
        zip(schema.thlocx_2d, schema.thlocy_2d), gauges_2d
    ):
        if abs(gx - float(x)) > atol or abs(gy - float(y)) > atol:
            raise AssertionError(f"2D xy {(gx, gy)} != {(x, y)}")
    if schema.labels_2d != [g[2] for g in gauges_2d]:
        raise AssertionError(f"2D labels {schema.labels_2d}")
    if bool(schema.pressure_1d) != pressure or bool(schema.pressure_2d) != pressure:
        raise AssertionError("pressure flag mismatch")
    if bool(schema.impulse_1d) != impulse or bool(schema.impulse_2d) != impulse:
        raise AssertionError("impulse flag mismatch")


def reset_execution_state(vip_path: str) -> None:
    """Clear saved solver clocks so a fresh nogui run is not a loaded snapshot."""
    import h5py

    zeros = {
        "dt": 0.0,
        "step": 0,
        "tt": 0.0,
        "tt_2d": 0.0,
        "tt_3d": 0.0,
    }
    with h5py.File(vip_path, "r+") as handle:
        g = handle[GROUP]
        for name, value in zeros.items():
            if name not in g:
                continue
            dtype = g[name].dtype
            del g[name]
            g.create_dataset(name, data=np.array([value], dtype=dtype))


def build_model(
    src_vip: str,
    dest_vip: str,
    *,
    gauges_1d: Sequence[Gauge1D],
    gauges_2d: Sequence[Gauge2D],
    reset_clocks: bool = True,
) -> str:
    from viper_compare.vip_diff import assert_remap_identity, read_remap_identity

    before = read_remap_identity(src_vip)
    copy_vip(src_vip, dest_vip)
    set_gauges(dest_vip, gauges_1d=gauges_1d, gauges_2d=gauges_2d)
    if reset_clocks:
        reset_execution_state(dest_vip)
    validate_gauges(dest_vip, gauges_1d=gauges_1d, gauges_2d=gauges_2d)
    assert_remap_identity(
        dest_vip,
        remapflag=int(before["remapflag"]),
        shape=int(before["shape"]),
    )
    return dest_vip
