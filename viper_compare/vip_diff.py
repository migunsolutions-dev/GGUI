"""Structural HDF5 diff of two VIPER .vip files."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

GROUP = "vipermodel"

EXECUTION_STATE = frozenset(
    {
        "dt",
        "step",
        "tt",
        "tt_2d",
        "tt_3d",
        "step_2d",
        "step_3d",
        "dt_2d",
        "dt_3d",
    }
)
REMAP_IDENTITY = frozenset({"remapflag", "shape"})
NOT_REMAP_DISCRIMINATORS = frozenset(
    {"twodremapoption", "remapFlags_2D", "numremapsources"}
)


def _scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    if hasattr(value, "tobytes") and getattr(value, "shape", ()) == ():
        raw = bytes(value)
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        data = value.tolist()
        if isinstance(data, list) and len(data) == 1:
            return data[0]
        return data
    return value


def _read_group(path: str) -> Tuple[Dict[str, Any], Dict[str, dict]]:
    import h5py

    values: Dict[str, Any] = {}
    meta: Dict[str, dict] = {}
    with h5py.File(path, "r") as handle:
        g = handle[GROUP]
        for name in g.keys():
            ds = g[name]
            raw = ds[()]
            values[name] = raw
            meta[name] = {
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
            }
    return values, meta


def classify_name(name: str) -> str:
    if name in EXECUTION_STATE:
        return "execution-state"
    if name in REMAP_IDENTITY:
        return "model-definition"
    if name in NOT_REMAP_DISCRIMINATORS:
        return "same-in-pair / not a remap discriminator"
    return "model-definition"


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.shape != bb.shape or aa.dtype != bb.dtype:
            return False
        if aa.dtype.kind in ("S", "O", "U"):
            return np.array_equal(aa, bb)
        if np.issubdtype(aa.dtype, np.floating):
            return np.allclose(aa, bb, rtol=0.0, atol=0.0, equal_nan=True)
        return np.array_equal(aa, bb)
    return a == b


def diff_vips(left_path: str, right_path: str) -> dict:
    left, left_meta = _read_group(left_path)
    right, right_meta = _read_group(right_path)
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    changed = []
    same_named = []
    for name in sorted(set(left) & set(right)):
        equal = values_equal(left[name], right[name])
        rec = {
            "name": name,
            "class": classify_name(name),
            "left": _scalar(left[name]),
            "right": _scalar(right[name]),
            "left_meta": left_meta[name],
            "right_meta": right_meta[name],
        }
        if equal:
            if name in REMAP_IDENTITY or name in NOT_REMAP_DISCRIMINATORS or name in EXECUTION_STATE:
                same_named.append(rec)
        else:
            changed.append(rec)
    return {
        "left": left_path,
        "right": right_path,
        "n_left": len(left),
        "n_right": len(right),
        "only_left": only_left,
        "only_right": only_right,
        "changed": changed,
        "checked_same": same_named,
        "remap_enabling": {
            name: {
                "left": _scalar(left[name]) if name in left else None,
                "right": _scalar(right[name]) if name in right else None,
            }
            for name in sorted(REMAP_IDENTITY | NOT_REMAP_DISCRIMINATORS)
        },
    }


def read_remap_identity(path: str) -> dict:
    values, _meta = _read_group(path)
    out = {}
    for name in sorted(REMAP_IDENTITY | NOT_REMAP_DISCRIMINATORS | EXECUTION_STATE):
        if name in values:
            out[name] = _scalar(values[name])
    return out


def assert_remap_identity(path: str, *, remapflag: int, shape: int) -> dict:
    ident = read_remap_identity(path)
    got_flag = int(ident.get("remapflag"))
    got_shape = int(ident.get("shape"))
    if got_flag != int(remapflag):
        raise AssertionError(f"{path}: remapflag={got_flag} != {remapflag}")
    if got_shape != int(shape):
        raise AssertionError(f"{path}: shape={got_shape} != {shape}")
    return ident
