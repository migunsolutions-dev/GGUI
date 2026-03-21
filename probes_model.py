from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PyQt5.QtCore import QObject, pyqtSignal


@dataclass
class ProbePoint:
    name: str
    x: float
    y: float
    z: float


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

    def add_probe(self, name: str, x: float, y: float, z: float) -> None:
        self._probes.append(ProbePoint(name=name, x=float(x), y=float(y), z=float(z)))
        self.changed.emit()

    def remove_probe(self, index: int) -> None:
        if 0 <= index < len(self._probes):
            self._probes.pop(index)
            self.changed.emit()

    def update_probe(self, index: int, *, name: Optional[str] = None,
                     x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None) -> None:
        if not (0 <= index < len(self._probes)):
            return
        p = self._probes[index]
        self._probes[index] = ProbePoint(
            name=p.name if name is None else str(name),
            x=p.x if x is None else float(x),
            y=p.y if y is None else float(y),
            z=p.z if z is None else float(z),
        )
        self.changed.emit()

    def clear(self) -> None:
        self._probes.clear()
        self.changed.emit()

    def to_dict(self) -> dict:
        return {"probes": [p.__dict__ for p in self._probes]}

    def load_dict(self, data: dict) -> None:
        probes = data.get("probes", [])
        self._probes = []
        for i, d in enumerate(probes):
            try:
                self._probes.append(ProbePoint(
                    name=str(d.get("name", f"P{i+1}")),
                    x=float(d.get("x", 0.0)),
                    y=float(d.get("y", 0.0)),
                    z=float(d.get("z", 0.0)),
                ))
            except Exception:
                continue
        self.changed.emit()
