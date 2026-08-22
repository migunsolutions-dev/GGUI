from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from PyQt5.QtCore import QObject, pyqtSignal


@dataclass
class ProbePoint:
    name: str
    x: float
    y: float
    z: float
    remap: bool = False
    terminate: bool = False


class ProbesModel(QObject):
    """
    Shared probes storage between tabs.
    Keeps UI simple: tabs edit this model; 3D viewer listens to changes.
    """
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._probes: List[ProbePoint] = []

    def probes(self) -> List[ProbePoint]:
        return list(self._probes)

    def add_probe(
        self,
        name: str,
        x: float,
        y: float,
        z: float,
        *,
        remap: bool = False,
        terminate: bool = False,
    ) -> None:
        self._probes.append(
            ProbePoint(
                name=name,
                x=float(x),
                y=float(y),
                z=float(z),
                remap=bool(remap),
                terminate=bool(terminate),
            )
        )
        self.changed.emit()

    def remove_probe(self, index: int) -> None:
        if 0 <= index < len(self._probes):
            self._probes.pop(index)
            self.changed.emit()

    def update_probe(
        self,
        index: int,
        *,
        name: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        remap: Optional[bool] = None,
        terminate: Optional[bool] = None,
    ) -> None:
        if not (0 <= index < len(self._probes)):
            return
        p = self._probes[index]
        self._probes[index] = ProbePoint(
            name=p.name if name is None else str(name),
            x=p.x if x is None else float(x),
            y=p.y if y is None else float(y),
            z=p.z if z is None else float(z),
            remap=p.remap if remap is None else bool(remap),
            terminate=p.terminate if terminate is None else bool(terminate),
        )
        self.changed.emit()

    def replace_all(self, probes: List[ProbePoint]) -> None:
        self._probes = list(probes)
        self.changed.emit()

    def clear(self) -> None:
        self._probes.clear()
        self.changed.emit()

    def to_dict(self) -> dict:
        return {"probes": [p.__dict__ for p in self._probes]}

    def load_dict(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise ValueError("Probe data must be a JSON object")
        probes = data.get("probes", [])
        if not isinstance(probes, list):
            raise ValueError("'probes' must be a list")
        loaded: List[ProbePoint] = []
        for i, d in enumerate(probes):
            try:
                if not isinstance(d, dict):
                    raise TypeError("entry must be an object")
                point = ProbePoint(
                    name=str(d.get("name", f"P{i+1}")),
                    x=float(d.get("x", 0.0)),
                    y=float(d.get("y", 0.0)),
                    z=float(d.get("z", 0.0)),
                    remap=bool(d.get("remap", False)),
                    terminate=bool(d.get("terminate", False)),
                )
                if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
                    raise ValueError("coordinates must be finite")
                loaded.append(point)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid probe at index {i}: {exc}") from exc
        self._probes = loaded
        self.changed.emit()
