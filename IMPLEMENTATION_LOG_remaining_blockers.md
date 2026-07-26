# Implementation log — `fix/cylindrical-2d-remaining-blockers`

## Branch setup

- Phase 1 merge on `main`: `048c5eb6800225219c7a37a4d69302d1fcc32756`
- Branch base: same SHA
- Remote tracking: pushed

## Checkpoint 1 — complete

Commit: `Preserve imported case commands and integrity`

Decisions:
- Allrun parsing moved to `allrun_commands.py` with structured `AllrunCommand` + per-utility allowlist.
- Solver launch lines (`$(getApplication)` / blastFoam) are ignored, not treated as preprocess.
- Inventory uses 4 MiB chunked SHA-256; read errors fail closed in `compare_inventories`.
- `validate_mapping_source` uses `classify_case_topology` (no ad-hoc wedge substring).
- Axisymmetric domain alignment uses `math.ceil` so requested R/H are never reduced.
- 3D `mesh_domain.align_domain_to_cell_size` left unchanged (unrelated mesh-spacing).

Tests: full suite 368 OK after Checkpoint 1.
