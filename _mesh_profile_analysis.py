import json
import pathlib
import re
import statistics


CASES = {
    "direct": pathlib.Path(
        r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\building3D_direct_20260428_163855"
    ),
    "gui": pathlib.Path(
        r"\\wsl.localhost\Ubuntu-20.04\home\naor\OpenFOAM\naor-9\run\Work\building3D_gui_20260428_163855"
    ),
}

BASE_DX = 0.5
Y0 = 0.0
Z0 = 0.5
YZ_TOL = 0.22
D_MAX = 5.0
STEP = 0.05
WINDOW = 0.08
SAMPLE = [i * STEP for i in range(int(D_MAX / STEP) + 1)]


def parse_list(path: pathlib.Path, is_vector: bool):
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"internalField\s+nonuniform\s+List<[^>]+>\s*(\d+)\s*\((.*?)\)\s*;",
        txt,
        re.S,
    )
    if not m:
        return []

    if is_vector:
        vals = []
        for ln in m.group(2).strip().splitlines():
            ln = ln.strip()
            if ln.startswith("(") and ln.endswith(")"):
                a = ln[1:-1].split()
                if len(a) == 3:
                    vals.append((float(a[0]), float(a[1]), float(a[2])))
        return vals

    vals = []
    for tok in m.group(2).split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def get_times(case: pathlib.Path):
    ts = []
    for d in (case / "processor0").iterdir():
        if d.is_dir():
            try:
                float(d.name)
                ts.append(d.name)
            except ValueError:
                pass
    return sorted(ts, key=lambda x: float(x))


def parse_runtime(case: pathlib.Path):
    p = case / "log.blastFoam"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="replace")
    m = list(re.finditer(r"ExecutionTime\s*=\s*([0-9.]+)\s*s", txt))
    return float(m[-1].group(1)) if m else None


def main():
    out = {
        "meta": {
            "line": "from charge center towards -X to 5m",
            "y0": Y0,
            "z0": Z0,
            "yz_tol": YZ_TOL,
            "sample_step": STEP,
        },
        "cases": {},
    }

    for name, case in CASES.items():
        times = get_times(case)
        series = []
        for t in times:
            d_to_sizes = {s: [] for s in SAMPLE}
            for proc in sorted(case.glob("processor*")):
                cpath = proc / t / "C"
                lpath = proc / t / "cellLevel"
                if (not cpath.exists()) or (not lpath.exists()):
                    continue
                pts = parse_list(cpath, True)
                lv = parse_list(lpath, False)
                n = min(len(pts), len(lv))
                for i in range(n):
                    x, y, z = pts[i]
                    if x > 1e-9 or x < -D_MAX - 0.3:
                        continue
                    if abs(y - Y0) > YZ_TOL or abs(z - Z0) > YZ_TOL:
                        continue
                    d = -x
                    size = BASE_DX / (2 ** int(round(lv[i])))
                    k0 = max(0, int((d - WINDOW) / STEP))
                    k1 = min(len(SAMPLE) - 1, int((d + WINDOW) / STEP))
                    for k in range(k0, k1 + 1):
                        if abs(SAMPLE[k] - d) <= WINDOW:
                            d_to_sizes[SAMPLE[k]].append(size)

            profile = []
            for s in SAMPLE:
                arr = d_to_sizes[s]
                profile.append(statistics.median(arr) if arr else None)
            valid = [v for v in profile if v is not None]
            series.append(
                {
                    "time": float(t),
                    "profile": profile,
                    "meanSize": (sum(valid) / len(valid) if valid else None),
                    "minSize": (min(valid) if valid else None),
                }
            )

        out["cases"][name] = {
            "path": str(case),
            "runtime_s": parse_runtime(case),
            "n_times": len(times),
            "times": [float(t) for t in times],
            "profiles": series,
        }

    common = sorted(
        set(out["cases"]["direct"]["times"]).intersection(set(out["cases"]["gui"]["times"]))
    )
    out["common_times"] = common

    out_path = pathlib.Path(r"c:\Users\migun\Desktop\GGUI\_mesh_profile_analysis.json")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(out_path)
    print(
        "direct times",
        out["cases"]["direct"]["n_times"],
        "gui times",
        out["cases"]["gui"]["n_times"],
        "common",
        len(common),
    )
    print("runtime", out["cases"]["direct"]["runtime_s"], out["cases"]["gui"]["runtime_s"])


if __name__ == "__main__":
    main()
