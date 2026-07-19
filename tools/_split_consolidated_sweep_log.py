#!/usr/bin/env python3
"""Developer helper: split a consolidated AMR sweep log into per-variant log.blastFoam files.

The amr_tuning_sweep --collect-only mode reads ``variants/<vid>/log.blastFoam``
per case. When ``Allrun`` is re-run (or the case is Allcleaned) the per-case
``log.blastFoam`` gets removed, but the consolidated console log captured by
``run_all.sh`` retains the full output. This helper extracts each variant's
section by the ``========== <variant_id> ==========`` markers emitted by
``run_all.sh`` and writes/refreshes ``variants/<vid>/log.blastFoam`` so
``--collect-only`` can produce a comparable table.

Existing per-case logs are preserved as ``log.blastFoam.preexisting`` to avoid
silently overwriting data from a longer / different run.

Usage:
    python tools/_split_consolidated_sweep_log.py \
        --sweep-root _amr_tuning_run01 \
        --consolidated _amr_tuning_run01/sweep_rerun_console.log

The script is developer-only and does not affect GUI defaults or test cases.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict


_HEADER_RE = re.compile(r"^========== ([A-Za-z0-9_]+) ==========\s*$", re.MULTILINE)


def split_consolidated(text: str) -> Dict[str, str]:
    """Return {variant_id: section_text} from a consolidated sweep log.

    Section markers are produced by ``run_all.sh`` as
    ``========== <variant_id> ==========``. The trailing
    ``========== SWEEP DONE: failures=N ==========`` marker is ignored.
    """
    matches = list(_HEADER_RE.finditer(text))
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        vid = m.group(1)
        if vid.upper().startswith("SWEEP"):
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[vid] = text[start:end].strip("\n")
    return sections


def main() -> int:
    ap = argparse.ArgumentParser(description="Split consolidated AMR sweep log per variant.")
    ap.add_argument("--sweep-root", type=Path, required=True, help="Path to the sweep root directory")
    ap.add_argument(
        "--consolidated",
        type=Path,
        required=True,
        help="Path to consolidated console log (e.g. sweep_rerun_console.log)",
    )
    args = ap.parse_args()

    sweep_root: Path = args.sweep_root.resolve()
    consolidated: Path = args.consolidated.resolve()
    if not consolidated.is_file():
        print(f"ERROR: consolidated log not found: {consolidated}", file=sys.stderr)
        return 2
    variants_root = sweep_root / "variants"
    if not variants_root.is_dir():
        print(f"ERROR: variants directory missing: {variants_root}", file=sys.stderr)
        return 2

    raw = consolidated.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw[3:].decode("utf-8", errors="replace")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-16", errors="replace")
    sections = split_consolidated(text)
    if not sections:
        print("ERROR: no variant sections found in consolidated log.", file=sys.stderr)
        return 1

    written = 0
    skipped = 0
    for vid, body in sections.items():
        vdir = variants_root / vid
        if not vdir.is_dir():
            print(f"  skip {vid}: variant dir missing")
            skipped += 1
            continue
        target = vdir / "log.blastFoam"
        if target.is_file():
            backup = vdir / "log.blastFoam.preexisting"
            if not backup.is_file():
                backup.write_bytes(target.read_bytes())
        target.write_text(body + "\n", encoding="utf-8", newline="\n")
        written += 1
        print(f"  wrote {target.relative_to(sweep_root)} ({len(body)} chars)")
    print(f"Done. wrote={written} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
