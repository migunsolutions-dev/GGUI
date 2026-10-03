from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation" / "jwl_internal"))
sys.path.insert(0, str(ROOT / "validation" / "rarefaction_osc"))
import run_models as jm
from diagnose_raw import times

def budget(case: Path):
    available = times(case)
    t0 = available[0][1]
    late = [folder for value, folder in available if abs(value - 0.00025) < 3e-5][0]
    names = sorted(p.name for p in t0.iterdir())
    print(case.name, "t0", t0.name, names)
    radius = None
    if (t0 / "rho").is_file():
        rho0 = jm._foam_scalar(t0 / "rho")
    else:
        rho0 = jm._foam_scalar(t0 / "rho.c4") + jm._foam_scalar(t0 / "rho.air")
    rho1 = jm._foam_scalar(late / "rho")
    e1 = jm._foam_scalar(late / "rhoE")
    radius = jm._cell_radii(case, rho0.size)
    dvol = jm._shell_volumes(radius, 1.0)
    m0 = float(np.sum(rho0 * dvol))
    m1 = float(np.sum(rho1 * dvol))
    # products energy is not in rhoE at t0; compare only mass
    print(f"  mass0={m0:.4f} mass1={m1:.4f} change={(m1-m0)/m0:+.4%}")

base = Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work")
for name in ("OscJWL_rk2", "OscJWL_upwind", "JwlFlux_Tadmor"):
    budget(base / name)
