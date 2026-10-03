"""Shock-rise width and ideal-gas energy change. Reads finished cases only."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation" / "jwl_internal"))
sys.path.insert(0, str(ROOT / "validation" / "rarefaction_osc"))
import run_models as jm
from diagnose_raw import load, times
from post_ab import choose_folder

P = 101325.0
WORK = Path(r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work")
CASES = {
    "IG_1mm": WORK / "OscIG_dx1mm_20261003_152640",
    "IG_rk2": WORK / "OscIG_rk2_20261003_152729",
    "IG_upwind": WORK / "OscIG_upwind_20261003_152713",
    "IG_cfl01": WORK / "OscIG_cfl01_20261003_152739",
    "IG_0p5": WORK / "OscIG_dx0p5mm_20261003_152645",
    "JWL_tadmor": WORK / "JwlFlux_Tadmor",
    "JWL_rk2": WORK / "OscJWL_rk2",
    "JWL_upwind": WORK / "OscJWL_upwind",
}

def rise(data):
    p, r = data["p"], data["r"]
    outer = int(np.where(p > 2 * P)[0][-1])
    level = float(np.median(p[outer - 12 : outer - 4]))
    hi = P + 0.9 * (level - P)
    lo = P + 0.1 * (level - P)
    i_lo = i_hi = None
    for i in range(p.size - 1, 0, -1):
        if i_lo is None and p[i] >= lo:
            i_lo = i
        if i_lo is not None and p[i] >= hi:
            i_hi = i
            break
    return i_lo - i_hi, (r[i_lo] - r[i_hi]) * 1e3, level / 1e6

def energy(case):
    available = times(case)
    t0 = available[0][1]
    t1 = [folder for value, folder in available if abs(value - 0.00025) < 2e-5]
    if not t1 or not (t0 / "p").is_file():
        return None
    p0 = jm._foam_scalar(t0 / "p")
    rho0 = jm._foam_scalar(t0 / "rho")
    # Ideal gas at rest: rhoE = p / (gamma-1)
    e0 = p0 / 0.4
    radius = jm._cell_radii(case, p0.size)
    dvol = jm._shell_volumes(radius, 1.0)
    m0 = float(np.sum(rho0 * dvol))
    u0 = float(np.sum(e0 * dvol))
    folder = t1[0]
    rho1 = jm._foam_scalar(folder / "rho")
    e1 = jm._foam_scalar(folder / "rhoE")
    m1 = float(np.sum(rho1 * dvol))
    u1 = float(np.sum(e1 * dvol))
    return (m1 - m0) / m0, (u1 - u0) / u0

for name, case in CASES.items():
    data = load(case, choose_folder(case))
    cells, mm, level = rise(data)
    extra = ""
    if name.startswith("IG"):
        got = energy(case)
        if got:
            extra = f" mass={got[0]:+.4%} energy={got[1]:+.4%}"
    print(f"{name:12} rise={cells} cells ({mm:.2f} mm) post={level:.2f} MPa{extra}")
