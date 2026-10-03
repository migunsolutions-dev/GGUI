"""List or remove local generated solver trees.

Dry-run unless --apply is passed. This is a developer utility. GGUI does not
call it, and it does not delete a user's saved analysis on its own.

It never touches .git, tracked source, or the VIPER model files under
_ig_plan_scratch/phase_d/viper/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Relative to the repository root. Each entry is a generated run tree.
TARGETS = (
    "_amr_tuning_run01/variants",
    "_val_b4/b4a",
    "_val_b4/b4b_archive_invalid",
    "_val_b4/b4b_fine",
    "_val_b4/b4b_coarse",
    "_audit_building3d_ggui/mimic_building3d",
    "_audit_building3d_roundtrip",
    "_ig_plan_scratch/phase_d/mass",
    "_ig_plan_scratch/phase_d/matrix",
    "_ig_plan_scratch/phase_d/resolution",
    "_ig_plan_scratch/Sedov_2D",
    "validation/viper_1d/cases",
    "_gui_verify_1d",
    "_gui_verify_freeair_kb",
    "_gui_verify_remap_2d",
)


def _case_dirs(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    found = []
    for child in parent.iterdir():
        if child.is_dir() and ((child / "constant").is_dir() or (child / "system").is_dir()):
            found.append(child)
    return found


def candidates() -> list[Path]:
    paths = [ROOT / rel for rel in TARGETS]
    for parent_name in ("_val_ex1", "_val_ex2", "_val_ex2_followup", "_val_m1", "_val_m2",
                        "_val_b1", "_val_b2", "_val_band_seed", "_val_release_spatial",
                        "_wsl_solver_val", "_gui_verify_1d", "_gui_verify_freeair_kb",
                        "_gui_verify_remap_2d"):
        paths.extend(_case_dirs(ROOT / parent_name))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        if ROOT.resolve() not in resolved.parents:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete the listed trees")
    args = parser.parse_args(argv)
    found = candidates()
    if not found:
        print("nothing to remove")
        return 0
    for path in found:
        action = "remove" if args.apply else "would remove"
        print(f"{action}  {path.relative_to(ROOT)}")
        if args.apply:
            shutil.rmtree(path)
    if not args.apply:
        print("dry-run only; pass --apply to delete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
