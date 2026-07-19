#!/usr/bin/env python3
"""
Developer/audit: load manual building3D (or any 3D case) via case_loader + TabGeneral3D,
regenerate with Generator3D, optionally diff against reference.

  python tools/roundtrip_building3d_case.py [reference_case_dir] [output_parent_dir]

Defaults:
  reference:  building3D/building3D
  output:     _audit_building3d_roundtrip/roundtrip_from_loader

Requires PyQt5 (same as GUI). Does not run OpenFOAM.
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load case → Tab3D → Generator3D round-trip audit.")
    parser.add_argument(
        "reference_case",
        nargs="?",
        default=os.path.join(_REPO, "building3D", "building3D"),
    )
    parser.add_argument(
        "output_parent",
        nargs="?",
        default=os.path.join(_REPO, "_audit_building3d_roundtrip"),
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip tools/compare_building3d_reference.py report",
    )
    args = parser.parse_args()

    ref = os.path.abspath(args.reference_case)
    if not os.path.isdir(ref):
        print(f"ERROR: reference case not found: {ref}", file=sys.stderr)
        return 1

    from PyQt5.QtWidgets import QApplication

    from case_loader import load_case
    from generator_3d import Generator3D
    from probes_model import ProbesModel
    from tab_3d_general import TabGeneral3D

    data = load_case(ref)
    summary = data.get("_load_summary") or {}

    app = QApplication.instance() or QApplication(sys.argv)
    tab = TabGeneral3D(ProbesModel())
    tab.set_case_inputs(data, summary)
    inputs = tab.get_case_inputs()

    parent = os.path.abspath(args.output_parent)
    os.makedirs(parent, exist_ok=True)
    case_dir = Generator3D(parent).generate("roundtrip_from_loader", inputs)
    print(case_dir)

    if not args.no_compare:
        import importlib.util

        cmp_path = os.path.join(_REPO, "tools", "compare_building3d_reference.py")
        spec = importlib.util.spec_from_file_location("compare_building3d_reference", cmp_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        print("\n")
        mod.print_report(ref, case_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
