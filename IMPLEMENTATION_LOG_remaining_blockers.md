# Implementation log — `fix/cylindrical-2d-remaining-blockers`

## Branch setup

- Phase 1 merge on `main`: `048c5eb6800225219c7a37a4d69302d1fcc32756`
- Branch base: same SHA
- Remote tracking: pushed

## Checkpoint 1 — complete

Commit: `867c553` — `Preserve imported case commands and integrity`

Decisions:
- Allrun parsing moved to `allrun_commands.py` with structured `AllrunCommand` + per-utility allowlist.
- Solver launch lines (`$(getApplication)` / blastFoam) are ignored, not treated as preprocess.
- Inventory uses 4 MiB chunked SHA-256; read errors fail closed in `compare_inventories`.
- `validate_mapping_source` uses `classify_case_topology` (no ad-hoc wedge substring).
- Axisymmetric domain alignment uses `math.ceil` so requested R/H are never reduced.
- 3D `mesh_domain.align_domain_to_cell_size` left unchanged (unrelated mesh-spacing).

Tests: full suite 368 OK after Checkpoint 1.

## Checkpoint 2 — complete

Commit: `393a31a` — `Centralize WSL execution and async preparation`

Decisions:
- `wsl_runtime.py` is the single non-Qt WSL/path/quoting/execution module.
- `execution_plan.py` holds pure solver planning (no PyQt).
- `preparation_worker_qt.py` runs long prep off the GUI thread.
- `solver_worker_qt.py` is a thin alias around `SolverRunner`.
- Imported 2D init uses `PreparationWorker`; tests may set `_force_sync_prep = True`.

Tests: full suite 387 OK after Checkpoint 2.

## Checkpoint 3 — complete

Commit message: `Complete 2D validation and project safeguards`

Tests: full suite **407** OK after Checkpoint 3.

Decisions:
- **Material `rho` responsibility:** `jwl_parameters()` validates JWL-specific Custom keys only
  (`A`, `B`, `R1`, `R2`, `omega`, `E0`/`energy`). Density is enforced by
  `material_validation.validate_material_definition` /
  `validate_required_values` before every generator call site (`generator_2d`,
  `generator_3d`). No ambiguous `if "rho" not in custom: pass`.
- **`smallest_charge_dimension_m`:** canonical in `charge_seed_plan.py`;
  `startup_mesh_metadata` re-exports it.
- **Mapping decomposition:** `map_domain`, `map_charge`, `map_material`,
  `map_atmosphere`, `map_boundaries`, `map_solver_controls`, `map_mesh`,
  `map_output`, `build_mapping_result`.
- **Schema:** `SCHEMA_VERSION = 2` with deterministic v1→v2 migration; undefined
  2D keys preserved; no solver results in project files.
- **State machine:** `state_machine_2d.py` + `SimulationState2D.INITIALIZING`;
  edits after init → `STALE`; fail/cancel init → `FAILED` (not initialized).
- **Logging:** `ggui_logging.py`; bare `except:` removed in `tab_1d`.
- **Deps/CI:** `pyproject.toml`, `constraints.txt`, `.github/workflows/ci.yml`.
- **Cleanup:** untracked `debug_summary.txt`; `.gitignore` updated.

Unresolved:
- Not every `except Exception` in viewers was narrowed (targeted 2D workflow
  paths + logging for preview refresh). Broader cleanup remains debt.
- Manual OpenFOAM/WSL scenarios remain environment-dependent.
