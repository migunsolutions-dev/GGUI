import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "rarefaction_osc"))
import run_ab

def main() -> None:
    import shutil
    dest = run_ab.WORK / "OscJWL_rk2"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(run_ab.BASE_JWL, dest)
    control = (dest / "system" / "controlDict").read_text(encoding="utf-8")
    control = control.replace("endTime         0.005;", f"endTime         {run_ab.END:.8g};")
    control = control.replace("writeInterval   0.005;", f"writeInterval   {run_ab.END:.8g};")
    control = control.replace("purgeWrite      1;", "purgeWrite      0;")
    (dest / "system" / "controlDict").write_text(control, encoding="utf-8", newline="\n")
    run_ab._patch_time(dest, "RK2SSP")
    run_ab._run(dest, "OscJWL_rk2")
    print("JWL_RK2_DONE", flush=True)

if __name__ == "__main__":
    main()
