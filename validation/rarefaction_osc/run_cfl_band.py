"""Two extra Ideal-Gas CFL points. Same case as the 1 mm baseline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "rarefaction_osc"))

import run_ab

def main() -> None:
    for name, cfl in (("OscIG_cfl025", 0.25), ("OscIG_cfl030", 0.30)):
        case = run_ab._ig(name, 0.001, "vanLeer", "Euler", cfl)
        print("PATH", name, case, flush=True)
    print("CFL_DONE", flush=True)

if __name__ == "__main__":
    main()
