# 3D Startup Meshing Architecture — Design Specification

**Status:** Draft for senior CFD / blast-engineering review
**Scope:** 3D blastFoam case generation only (startup meshing + charge capture). Runtime AMR is referenced only at its interface. 1D and 2D workflows are explicitly out of scope.
**Authority:** Once approved, this document is the design authority for all future 3D startup-meshing development.
**Constraint at time of writing:** No production defaults have been changed. `chargeRefineOuter` remains enabled by default. The Milestone-1 capture fix is the only code change in effect.

---

## 1. Executive Summary

The 3D startup meshing pipeline must deliver two guarantees to the solver: **(1) mass-conservative charge capture** for charges that may be smaller than a single base cell, and **(2) sufficient near-field resolution** to form the initial blast wave correctly. Historically these were achieved indirectly through a volumetric `chargeRefineOuter` refinement band. Investigation has shown that the band was, in effect, a workaround for a charge-capture defect, and that it is neither necessary for capture nor for early-wave formation once the capture defect is fixed. Worse, when combined with deep seeding the band produces explosive, intractable mesh growth.

This specification establishes a new architecture in which:

- **Charge capture** is guaranteed by a correctly sized, mass-conserving **backup region** (fixed in Milestone 1) — independent of any refinement band.
- **Near-field resolution** is provided by **deep `refineInternal` seeding** of a tight region around the charge, with an **automatically computed seed level** tied to charge size and base cell size.
- **`chargeRefineOuter` becomes an advanced, opt-in tool**, off by default, with explicit guardrails against the band×seed explosion. *(Clarification: this is the approved **target** state. The current **shipping default remains `chargeRefineOuter` ON with `charge_refinement_level = 0`**; the band-OFF target is **not yet implemented as the production default** and is gated on the default-transition plan — see `docs/DEFAULT_TRANSITION_PLAN.md` and `docs/LIVING_VALIDATION_ASSESSMENT.md`.)*
- **Startup meshing and runtime AMR are treated as independent systems** with a single, well-defined hand-off contract.

The architecture is shape-agnostic (sphere, cylinder, box), scales across coarse and fine base meshes, and handles small charges (e.g. 5 kg) by automatic seed-level escalation with a safety clamp. It aligns the GUI with the official blastFoam `building3D` / `freeField` philosophy (deep seed, lean near field, runtime AMR follows the wave) while retaining the GUI's general-purpose flexibility.

---

## 2. Problem Statement

The 3D GUI must generate blastFoam cases that are correct and efficient across a wide parameter space: three charge shapes, charge masses spanning at least 5–25 kg, base cell sizes from fine (0.2 m) to coarse (0.5 m), free-field and obstacle cases, with and without runtime AMR. The desired end state is: coarse far field, reliable charge representation, sufficient charge-region refinement to form and propagate the early wave, smooth refinement transitions, independent obstacle refinement, runtime AMR that follows the wave, and a good quality/cost balance.

> **Validation correction (efficiency narrative).** Earlier wording implied runtime AMR practically *releases* (coarsens) cells behind the wave. Validation (B2/B2 Extension + the release spatial analysis) showed that, within the practical propagation window, **release behind the wave was not observed**: the refined region remained associated with the physical detonation-product/fireball region (real ρ/p gradients), not an AMR defect. Demonstrated efficiency therefore comes from **leaner startup meshing, removal of unnecessary volumetric band refinement, bounded maximum refinement, and careful domain/time sizing** — not from runtime release. See `docs/LIVING_VALIDATION_ASSESSMENT.md` (P4–P7, D1, R1–R2) and `docs/ARCHITECTURE_DECISIONS.md` (AD-004).

Two specific failure modes motivated this work:

1. **Sub-cell charge capture failure.** When a charge is smaller than a base cell and the refinement band is disabled, `setRefinedFields` could select **zero** cells, seeding **no explosive** — a silent, catastrophic correctness failure.
2. **Uncontrolled cell growth.** Certain combinations of refinement mechanisms produce initial meshes large enough to make the case computationally intractable, defeating the efficiency goal.

The architecture must eliminate both failure modes by construction, not by tuning.

---

## 3. Investigation Summary

The investigation proceeded from system understanding through controlled experiments:

- **System tracing** established the actual startup pipeline (blockMesh → snappyHexMesh → setRefinedFields/refineInternal → solver) and the runtime AMR pipeline (dynamicMeshDict, `densityGradient`), and identified where mesh-quality decisions are made.
- **Reference comparison** against the official `building3D` and `freeField` tutorials showed the GUI had diverged toward a volumetric band + high runtime `maxRefinement`, whereas the tutorials use deep seeding + lean near field + modest runtime AMR.
- **History reconstruction** indicated the band and elevated `maxRefinement` were introduced incrementally, plausibly to compensate for capture and early-formation problems rather than as a deliberate physical choice.
- **Milestone 1 (charge capture)** isolated and fixed the sub-cell capture defect, then validated capture generation-only across sphere/cylinder/box × {0.2, 0.5 m} × charge masses.
- **Milestone 2 (early propagation)** compared band-ON vs band-OFF at equal deep-seed depth, measuring initial/peak cell counts, cell-count-vs-time, 0.5 m pressure history, shock arrival, peak pressure, and wall-clock cost.

Production defaults observed during the investigation: `charge_refinement_level = 0` (no deep seed; `setFields` only), `chargeRefineOuter` enabled (`None`), `charge_capture_mode = auto`, `charge_capture_factor = 1.0`, cylinder `charge_aspect = 2.5` (L/D). In other words, **today's default relies on the band as the de-facto charge-refinement mechanism and performs no deep seeding.**

---

## 4. Root Causes Identified

**RC-1 — Backup radius equal to the worst-case cell-centre distance (capture failure).**
The automatic capture radius was computed as `0.5 · diag · capture_factor` with `capture_factor = 1.0`, i.e. exactly half the base-cell diagonal. For a charge smaller than a cell placed at a worst-case offset, the nearest cell centre lies *at* this distance, not strictly inside it. `setRefinedFields` selects cells whose centres are strictly within the region, so it could select zero cells and seed no explosive. *Why it matters:* this is a silent correctness failure, not a quality issue. *Solved by:* Milestone 1 (Topic 10).

**RC-2 — The band masked RC-1.** With `chargeRefineOuter` on, the near field is pre-refined, so cell centres exist inside even a tight backup radius. The band therefore "fixed" capture as a side effect, which is why the defect went unnoticed and why the band became load-bearing for correctness. *Why it matters:* a correctness guarantee was resting on a refinement convenience. *Solved by:* removing capture's dependence on the band (Topics 9–10).

**RC-3 — Multiplicative interaction of band and deep seed (cell explosion).** `refineInternal` builds resolution in level-by-level passes, with 2:1 grading and buffer layers. When a large region is already pre-refined by the band, the grading/buffer cascade during seeding propagates outward through that pre-refined shell, inflating the seeded volume far beyond the charge. *Why it matters:* it makes deep-seed + band cases intractable. *Solved by:* default band-OFF and band×seed guardrails (Topics 11, 14, 19).

**RC-4 — Conflation of startup meshing and runtime AMR.** Treating the two as one system led to using startup mechanisms (band, high seed) to chase runtime goals (wave tracking) and vice-versa. *Why it matters:* it produced both over-refinement and fragile defaults. *Solved by:* explicit separation with a hand-off contract (Topics 8, 13).

---

## 5. Key Findings

Grounded in the controlled experiments (free-field sphere, 25 kg, domain ±2 m, base cell 0.2 m, `densityGradient` AMR, probes at 0.5/0.75/1.0 m):

- **F-1 (capture, M1).** After the backup fix, sphere, cylinder, and box charges all capture mass-conservatively on coarse meshes **with the band off**. No zero-capture cases remained.
- **F-2 (coarse sphere, M1 open issue).** Increasing seed from level 4 to level 5 on the coarse sphere (cell 0.5 m) improved captured-mass ratio from **0.938 → 0.985** (charge cells 480 → 4,032). Deeper seed improves coarse-mesh capture *accuracy* (geometry resolution), confirming the seed-level lever.
- **F-3 (early wave, M2 — band OFF).** Band-OFF + deep seed (level 4) produced a clean, physical early wave: initial cells **30,792**, smooth growth to **~119k**, shock arrival at 0.5 m **3.94e-5 s**, peak **163.5 bar @ 5.53e-5 s**, wall-clock **65 s** for the early window.
- **F-4 (cell explosion, M2 — band ON).** Band-ON at the *same* seed depth produced **4,891,200** initial cells (4,733,568 at level 4) — about **159×** more than band-OFF — and was computationally intractable for the early window.
- **F-5 (depth vs volume).** Both M2 cases reached the **same maximum cell level (4)**. The explosion was driven by the **refined volume**, not the seed depth: band-OFF kept level-4 cells local (~0.04 m³, ≈ backup radius), band-ON inflated the level-4 zone to ~9 m³ (≈ r = 1.3 m). *Architectural consequence:* control the seeded **volume**, not just the level.
- **F-6 (no band needed for early formation).** Because band-OFF alone forms and propagates the early wave cleanly, `chargeRefineOuter` is **not required** for early blast formation in the tested regime.

---

## 6. Design Principles

- **P-1 Correctness before refinement.** Charge mass capture must be guaranteed by the capture mechanism itself, never as a side effect of refinement. *Solves RC-1/RC-2.*
- **P-2 Separation of concerns.** Startup meshing and runtime AMR are independent systems with one explicit hand-off contract. *Solves RC-4.*
- **P-3 Control volume, not just level.** The cost driver is the refined volume at a given level; the architecture sizes refined regions tightly. *Solves RC-3/F-5.*
- **P-4 Physics-tied automation.** Resolution is tied to "cells across the smallest charge dimension," not to arbitrary fixed levels. *Addresses generality across mass/shape/mesh.*
- **P-5 Safe defaults, expert overrides.** Defaults are robust and lean; dangerous levers (band, manual seed) are advanced and guarded with warnings.
- **P-6 No hidden geometry inflation.** The mesh must reflect the user's charge; capture backups and seed regions must be transparent and reported.
- **P-7 Alignment with blastFoam reference philosophy** where it does not compromise generality (deep seed, lean near field, runtime AMR follows the wave).

---

## 7. Architecture Goals

1. Guarantee mass-conservative capture for charges smaller than the base cell, for all shapes.
2. Minimize initial cell count consistent with correct early-wave formation.
3. Remove capture's dependence on `chargeRefineOuter`.
4. Preserve `chargeRefineOuter` as a justified advanced tool only.
5. Avoid hidden geometry inflation; report all derived regions.
6. Keep obstacle refinement fully independent of charge capture.
7. Remain aligned with official blastFoam philosophy where appropriate.
8. Scale automatically across charge mass, shape, and base mesh, with explicit safety clamps and warnings.

---

## 8. Startup Meshing Architecture

**Recommendation.** A linear, transparent pipeline with clearly separated responsibilities:

1. **Base mesh** (blockMesh): uniform coarse hexahedral background.
2. **Geometry refinement** (snappyHexMesh): obstacle and (optional, advanced) charge-surface refinement only. *No volumetric charge band by default.*
3. **Charge seeding** (setRefinedFields/refineInternal + mass-conserving source with backup): capture + near-field resolution.
4. **Hand-off** to the solver with documented mesh metadata.

**Why it exists / problem solved.** A clean separation makes each guarantee (capture, resolution, obstacle handling) independently verifiable and prevents one mechanism from silently compensating for another (RC-2, RC-4).

**Alternatives rejected.** (a) Band-centric near field (current default) — rejected: load-bearing for correctness, explosive with seed (F-4). (b) Pure snappy volumetric refinement of the near field — rejected: couples charge resolution to snapping, fragile for tiny charges.

**Risks / tradeoffs.** Requires the capture mechanism to be provably robust on its own (addressed in Topics 9–10). A lean near field shifts more responsibility to runtime AMR's first refinement step (addressed in Topic 13).

---

## 9. Charge Capture Architecture

**Recommendation.** Capture via a **mass-conserving source** (`sphericalMassToCell`, `cylindericalMassToCell`, `boxToCell` with an explicit `backup` region) that distributes the nominal charge mass exactly into the selected cells, with the backup guaranteeing a non-empty selection. Band-OFF by default.

**Why it exists / problem solved.** Mass conservation by construction (P-1) plus a backup that always selects ≥1 cell eliminates the silent zero-capture failure (RC-1) and removes the dependence on the band (RC-2). M1 validated this for all three shapes (F-1).

**Alternatives rejected.** (a) Geometric `boxToCell`/region fills without mass conservation — rejected: mis-conserve mass for sub-cell charges. (b) Relying on the band for capture — rejected: RC-2.

**Risks / tradeoffs.** If a backup is mis-configured to select zero cells, capture fails silently; therefore the backup floor (Topic 10) and a mandatory mass-ratio metadata check (Topic 18) are required safety nets.

---

## 10. Backup Region Architecture

**Recommendation.** Size the backup so it **strictly encloses the nearest cell centre in the worst case**: `backup = max(1.05·r_charge, 0.5·diag·max(capture_factor, SAFETY))`, where `diag = base-cell diagonal` and `SAFETY = 1.5` (the Milestone-1 floor). For cylinders, apply the same floor to the **axial** extent; for boxes, provide an explicit `backup{ box ... }` whose half-extents respect the same floor.

**Why it exists / problem solved.** Directly fixes RC-1: with `capture_factor = 1.0`, the prior radius equalled the worst-case cell-centre distance, a boundary case that selected zero cells. The `SAFETY = 1.5` floor moves the nearest centre strictly inside. Validated in M1 (F-1), including the previously failing coarse cylinder (axial) and box (no backup) cases.

**Alternatives rejected.** (a) Keep `capture_factor = 1.0` and document the limitation — rejected: leaves a silent correctness hole. (b) Make the backup huge to be safe — rejected: inflates the seeded volume on coarse meshes (P-3) and risks hidden geometry inflation (P-6).

**Risks / tradeoffs.** On coarse meshes the backup grows with cell size (≈ 1.3 × base cell). Because the backup currently also bounds the seeded region, a large backup inflates cell count — motivating the **decoupling** of capture-backup from seed-region in Topic 11. User-supplied capture factors above the floor are honoured; below the floor they are raised to the floor (with a metadata note).

---

## 11. Deep Seed Architecture

**Recommendation.** Resolve the charge with `refineInternal` over a **tight seed region — the charge bounding volume plus a thin margin (~1.2×)** — at the computed seed level (Topic 12). **Decouple** this seed region from the capture backup: the backup need only guarantee capture (≥1 cell), while the seed region governs resolution. Band-OFF by default.

**Why it exists / problem solved.** Implements P-3 (control the refined *volume*). M2 showed the cost driver is the seeded volume, not the seed level (F-5). Keeping the seed region tight delivers the resolution needed for early-wave formation (F-3) at minimum cell count, and avoids the band×seed explosion entirely (F-4).

**Alternatives rejected.** (a) Seed the entire (possibly large) capture backup — rejected: inflates cells on coarse meshes. (b) No deep seed (current default) — rejected: leaves charge resolution to the band, which is being retired. (c) Band + deep seed — rejected: F-4 (intractable).

**Risks / tradeoffs.** Too thin a margin under-resolves the very first expansion; runtime AMR must capture the wave within the first refine interval (Topic 13). The margin factor is itself a parameter to validate (Topic 22).

---

## 12. Seed-Level Policy

**Recommendation.** **Automatic** seed level with manual override:

```
L_seed = clamp( ceil( log2( N · base_cell / d_min ) ), 0, L_max )
```

where `d_min` is the smallest charge dimension (sphere diameter; cylinder diameter for L/D ≥ 1; box smallest edge), `N` is the target number of cells across `d_min`, default **N = 6**, and **L_max = 5**.

**Why it exists / problem solved.** Implements P-4: resolution is tied to physics and auto-scales with charge size and base mesh, so the same policy serves 25 kg/fine and 5 kg/coarse without manual tuning. F-2 confirms the seed-level lever controls coarse-mesh accuracy.

**Resolution tradeoffs.** N = 4: marginal shape resolution; N = 6: recommended balance of accuracy and cost; N = 8: higher accuracy at ~2–3× the cells with little additional fidelity once mass is conserved; N = 10: diminishing returns and frequent L_max clamping on coarse meshes.

**Alternatives rejected.** (a) Fixed default level — rejected: cannot serve the mass/mesh range. (b) User-defined only — rejected: novice footgun; retained only as override.

**Risks / tradeoffs.** Tiny charges on coarse meshes can demand levels beyond L_max; the clamp bounds cost but under-resolves — this must trigger a warning and a "use a finer base mesh" suggestion (Topics 15, 18, 19).

---

## 13. Runtime AMR Interaction

**Recommendation.** Startup hands the solver a tightly seeded, mass-correct charge; runtime AMR (`densityGradient`, **modest** `maxRefinement`, sensible `refineInterval`/buffer/unrefine) then forms-and-follows the wave. Startup seed level and runtime `maxRefinement` are **independent** parameters, but the architecture enforces a consistency relation: **`L_seed ≤ runtime maxRefinement`** so the solver does not immediately unrefine the seed.

**Why it exists / problem solved.** Implements P-2/RC-4. M2 demonstrated a clean smooth cell trajectory when startup is lean and runtime AMR does the tracking (F-3). It also prevents the historical pattern of using startup mechanisms to chase runtime goals.

**Alternatives rejected.** (a) High startup refinement to "pre-empt" the wave — rejected: over-refines static regions, defeats efficiency. (b) Low/zero seed relying entirely on runtime AMR to build charge resolution from scratch — rejected: risks under-resolving the first expansion before the first refine interval.

**Risks / tradeoffs.** If `L_seed > maxRefinement`, runtime unrefines the seed at t≈0 — must warn. The lean near field places a fidelity dependency on the first refinement step; validation must confirm early-wave comparability (Topic 20).

---

## 14. chargeRefineOuter Policy

**Recommendation.** **Advanced-only, default OFF.** Never auto-enabled. The GUI may *suggest* (not enable) it only in the narrow case where `L_seed` is clamped at `L_max` and the resolution target is still unmet — and even then the preferred remedy is a finer base mesh. When enabled, enforce the guardrail **`L_band + L_seed ≤ L_max`** and display a projected initial cell count before the run.

**Why it exists / problem solved.** The band is a legitimate tool for a deliberately thicker uniformly-refined near field, but it is not needed for capture (F-1) or early formation (F-6), and it is dangerous with deep seed (F-4). Advanced-only status preserves the capability while protecting the default user (P-5, RC-3).

**Suggested help/warning text (for UX, not implementation).** "Outer charge refinement band (advanced). Not required for charge capture or early-wave formation — robust seeding handles both. Enabling it together with deep seeding can multiply cell count dramatically (observed ~159×). Combined band + seed refinement is capped at the maximum level."

**Alternatives rejected.** (a) Remove entirely — rejected: loses a valid advanced use case. (b) Keep enabled by default — rejected: RC-2, RC-3, F-4. (c) Auto-enable on conditions — rejected: any silent enable can reintroduce the explosion.

**Risks / tradeoffs.** Even advanced use can be misused; the guardrail and cell-count projection are mandatory mitigations (Topic 19).

---

## 15. Small-Charge Strategy

**Recommendation.** Handle small charges through automatic seed-level escalation under the L_max clamp, band-OFF, with warnings when the clamp binds. Worked envelope (C4, ρ = 1601; `d_min` = cylinder diameter, the binding shape; backup ≈ 1.3 × base cell):

| Case | d_min (cyl Ø) | L_seed (N=6) | L_seed (N=8) | backup radius | band needed? |
|---|---|---|---|---|---|
| 25 kg, base 0.2 m | 0.200 m | 3 | 3 | 0.260 m | No |
| 25 kg, base 0.5 m | 0.200 m | 4 | 5 | 0.650 m | No |
| 5 kg, base 0.2 m | 0.117 m | 4 | 4 | 0.260 m | No |
| 5 kg, base 0.5 m | 0.117 m | 5 (clamp) | 6 → clamp 5 | 0.650 m | No (warn) |

(Spheres need roughly one level less than cylinders because their smallest dimension is larger; boxes fall between.)

**Why it exists / problem solved.** Demonstrates the policy degrades gracefully: the hardest case (5 kg, coarse mesh, N=8) hits the clamp and is handled by warning + finer-mesh guidance rather than by silently enabling the band (which would explode).

**Alternatives rejected.** (a) Auto-enable band for small charges — rejected: RC-3. (b) Unbounded seed escalation — rejected: cost blow-up; the L_max clamp is required.

**Risks / tradeoffs.** Clamped cases under-resolve relative to target N; the user must be informed and offered the finer-base-mesh remedy (P-5, Topic 18).

---

## 16. Sphere / Cylinder / Box Handling

**Recommendation.** A single shape-agnostic policy parameterised by `d_min` and the appropriate mass-conserving source:

- **Sphere:** `sphericalMassToCell`; `d_min` = diameter; spherical backup with the radial floor.
- **Cylinder:** `cylindericalMassToCell`; `d_min` = diameter (for L/D ≥ 1, default L/D = 2.5); backup floor applied to **both radial and axial** extents.
- **Box:** `boxToCell` with explicit `backup{ box }`; `d_min` = smallest edge; backup half-extents respect the floor on each axis.

**Why it exists / problem solved.** One policy, three sources, unified by `d_min`, keeps behaviour predictable and verifiable across shapes. M1 confirmed each source captures correctly with the floor applied (the cylinder axial and box-backup gaps were specifically closed).

**Alternatives rejected.** (a) Shape-specific ad-hoc heuristics — rejected: divergent behaviour, hard to validate. (b) Approximating cylinders/boxes as spheres — rejected: mis-resolves the binding (smallest) dimension.

**Risks / tradeoffs.** Highly anisotropic charges (very large L/D, or slab-like boxes) make `d_min` very small, driving high seed levels — the L_max clamp and warnings apply (Topic 15).

---

## 17. User Experience Philosophy

**Recommendation.** Two-tier UI (P-5). **Visible defaults:** charge shape/mass, base cell size, target cells-across-charge (N, default 6), free-field/obstacle. **Advanced (collapsed):** manual seed level, manual capture factor/radius, `chargeRefineOuter` toggle, optional snappy charge-surface band, runtime AMR levels. Dangerous levers live in Advanced and carry warnings.

**Why it exists / problem solved.** Novices get robust, lean, correct cases without tuning; experts retain full control. Hides the footguns (band, manual seed) responsible for the explosion behind explicit opt-in.

**Alternatives rejected.** (a) Expose everything flat — rejected: footgun surface. (b) Hide everything (full auto, no overrides) — rejected: removes legitimate expert control and reproducibility of reference cases.

**Risks / tradeoffs.** Auto behaviour must be transparent (Topic 18) so experts can audit and reproduce; otherwise "automatic" becomes "opaque."

---

## 18. Metadata Requirements

Every generated case must record and surface:

- captured mass and **mass ratio** (warn if < 0.98);
- chosen `L_seed`, target `N`, and whether `L_seed` was **clamped** at `L_max`;
- backup radius/extents and **backup-vs-charge ratio** (flag potential hidden inflation, P-6);
- **projected and actual initial cell count** (warn above a budget, e.g. > 200k);
- `chargeRefineOuter` state and, if on, `L_band + L_seed` vs `L_max`;
- relation `L_seed` vs runtime `maxRefinement`.

**Why it exists / problem solved.** Makes correctness (mass ratio) and cost (cell count) auditable, supports reproducibility, and turns the silent failures (RC-1) and silent explosions (RC-3) into visible, actionable warnings.

**Alternatives rejected.** Minimal/no metadata — rejected: reintroduces silent failure modes.

**Risks / tradeoffs.** Metadata must be accurate; a wrong projected cell count is worse than none. Projections require validation against measured counts (Topic 20).

---

## 19. Safety Requirements

- **S-1** Capture must never select zero cells (enforced by the backup floor, Topic 10) and must be verified by the mass-ratio check (Topic 18).
- **S-2** `chargeRefineOuter` is never auto-enabled; when enabled it is guarded by `L_band + L_seed ≤ L_max` and a pre-run cell-count projection (Topic 14).
- **S-3** Seed level is clamped at `L_max`; binding clamps raise a warning (Topic 15).
- **S-4** `L_seed > runtime maxRefinement` raises a warning to prevent immediate unrefinement (Topic 13).
- **S-5** Obstacle refinement is independent of charge logic and must not be altered by capture/seed changes.

**Why it exists / problem solved.** Encodes the lessons of RC-1 and RC-3 as hard constraints, so neither silent zero-capture nor mesh explosion can recur through normal use.

**Alternatives rejected.** Relying on user discipline — rejected: the historical defaults show how silent failures persist.

**Risks / tradeoffs.** Guardrails can occasionally block a legitimate expert configuration; overrides must be possible but explicit and logged.

---

## 20. Validation Philosophy

**Recommendation.** Two-stage, evidence-first validation, generation-only before any solver runs, and never changing production defaults until validation passes.

- **Stage 1 (generation-only, fast):** for every shape × {5, 25 kg} × {0.2, 0.5 m} base cell, band OFF — verify mass ratio ≥ 0.98, no zero-capture, recorded `L_seed`, backup, and initial cell count within budget.
- **Stage 2 (short early-window solver, subset):** verify 0.5 m shock arrival and peak pressure, smooth bounded cell-count-vs-time, and wall-clock cost, comparing against the band-OFF reference established in M2 (arrival 3.94e-5 s, peak 163.5 bar, smooth 29k→119k cells, 65 s).

**Why it exists / problem solved.** Generation-only catches correctness/cost regressions cheaply; the short solver window confirms early-wave fidelity without multi-hour runs. This mirrors how M1/M2 were actually conducted.

**Alternatives rejected.** (a) Full-duration solver validation for every case — rejected: prohibitively expensive (the band-ON case was intractable). (b) Trusting generation metrics alone — rejected: does not confirm physical early-wave behaviour.

**Risks / tradeoffs.** The short window validates only early formation, not full propagation/late unrefinement; those remain runtime-AMR concerns tracked separately (Topic 22).

---

## 21. Risks

- **R-1 Lean near field under-resolves the first expansion** before runtime AMR engages. *Mitigation:* seed margin + `L_seed ≤ maxRefinement`; validate early peak (Topic 20).
- **R-2 Coarse-mesh backups inflate the seeded volume** if capture-backup and seed-region are not decoupled. *Mitigation:* Topic 11 decoupling.
- **R-3 Advanced band misuse reintroduces explosion.** *Mitigation:* guardrails + projection (Topics 14, 19).
- **R-4 Clamped small-charge cases under-resolve.** *Mitigation:* warning + finer-mesh guidance (Topic 15).
- **R-5 Default change risk.** Flipping defaults (band OFF, auto-seed ON) could surprise existing users / break saved cases. *Mitigation:* staged rollout, validation gate, and explicit change communication (Topic 23).
- **R-6 Anisotropic charges** drive very high seed levels. *Mitigation:* L_max clamp + warnings.

---

## 22. Open Questions

- **OQ-1** Optimal seed-region margin factor (the ~1.2× in Topic 11) — needs a small sensitivity study against early-wave fidelity vs cell count.
- **OQ-2** Optimal target `N` per use case — is N = 6 right for obstacle cases and for cylinders/boxes specifically?
- **OQ-3** Default runtime `maxRefinement` under the new lean-startup regime — does a lower value suffice now that capture/early-formation no longer rely on startup over-refinement?
- **OQ-4** Late-time behaviour (full propagation, unrefinement behind the wave) — not exercised by the short window; requires separate runtime-AMR validation.
- **OQ-5** Obstacle + charge interaction near obstacles — confirm independence holds when a charge sits close to a refined obstacle.
- **OQ-6** Whether the optional snappy charge-surface band (sharp interface) provides measurable fidelity benefit to justify its advanced inclusion.

---

## 23. Final Recommended Architecture

1. **Compute charge geometry** → smallest dimension `d_min` per shape.
2. **Auto seed level** `L_seed = clamp(ceil(log2(N·base_cell/d_min)), 0, L_max)`, defaults **N = 6, L_max = 5**, manual override available.
3. **Capture backup** `= max(1.05·r_charge, 0.5·diag·max(capture_factor, 1.5))`, applied per-axis for cylinder (radial + axial) and box; strictly encloses the nearest cell centre (M1 floor).
4. **Deep seed** via `refineInternal` over a **tight charge region (charge bbox × ~1.2)**, decoupled from the capture backup, **band OFF by default**.
5. **`chargeRefineOuter`** advanced-only, default OFF, never auto-enabled, guarded by `L_band + L_seed ≤ L_max` with pre-run cell-count projection.
6. **Obstacle refinement** is an independent snappy stage, unaffected by charge logic.
7. **Runtime AMR** independent (`densityGradient`, modest `maxRefinement`), with `L_seed ≤ maxRefinement` enforced; startup only forms the charge, runtime follows the wave. *(Release behind the wave was not observed in the practical validation window — the refined region tracks the physical detonation-product region; see the efficiency-narrative correction in §2 and `docs/LIVING_VALIDATION_ASSESSMENT.md`.)*
8. **Metadata + warnings** on mass ratio, `L_seed`/clamp, backup ratio, projected/actual cell count, band state, and seed-vs-maxRefinement.

**Staged rollout (no defaults changed until each gate passes):** auto seed-level behind manual override → seed/backup decoupling → metadata/warnings → flip defaults (band OFF, auto-seed ON) → optional charge-surface band → obstacle-independence regression.

**Explicitly NOT changed at this stage:** production defaults (`charge_refinement_level = 0`, band enabled), the existence of `chargeRefineOuter`, the 1D/2D workflows, obstacle refinement behaviour, runtime AMR defaults, and the Milestone-1 capture formula / `CAPTURE_CELL_SAFETY` floor.

---

## 24. Validation Status & Limitations

This section records the evidence level behind the architecture so reviewers do not mistake reasoned design for demonstrated fact. Evidence levels: **Proven** (directly measured), **Partial** (measured in a narrow case, generalized by reasoning), **Hypothesis** (plausible, untested), **Assumed** (taken as given, not examined).

| # | Claim / recommendation | Section | Evidence level | Basis / gap |
|---|---|---|---|---|
| V-1 | Sub-cell capture defect (RC-1) and its fix via the backup floor | §4, §10 | **Proven** | M1 generation-only, all shapes, 25 kg, base 0.2/0.5 m |
| V-2 | Capture works band-OFF for sphere/cyl/box | §9, §16 | **Proven (25 kg)** / **Partial (other masses)** | M1 at 25 kg; 5 kg not generated |
| V-3 | Deeper seed improves coarse-mesh capture (0.938→0.985) | §5 F-2, §12 | **Proven** | M1 coarse sphere, seed 4→5 |
| V-4 | Band-OFF + deep seed forms a clean early wave | §5 F-3 | **Proven (one config)** | M2: free-field sphere, 25 kg, base 0.2 m only |
| V-5 | Band×seed cell explosion magnitude (~159×) | §5 F-4 | **Proven (one config)** | M2 measured; same single config |
| V-6 | Band×seed causal mechanism (grading/buffer cascade) | §4 RC-3 | **Hypothesis** | Inferred from level/volume data, not isolated |
| V-7 | chargeRefineOuter not needed for early formation | §5 F-6, §14 | **Hypothesis (one-sided)** | C4-new clean; B4 produced no matched 0.5 m peak |
| V-8 | Early-wave behavior generalizes to cyl/box/5 kg/coarse | §15, §16 | **Assumed** | Not measured for any non-sphere / non-25 kg / coarse case |
| V-9 | Seed/backup decoupling controls cost | §11 | **Hypothesis** | M2 used backup-as-seed-region; decoupling untested |
| V-10 | Seed margin ~1.2× | §11 | **Hypothesis** | Guessed value (OQ-1) |
| V-11 | Auto-seed defaults N=6, L_max=5 and N tradeoff table | §12 | **Hypothesis** | Reasoned, not swept |
| V-12 | `L_seed ≤ maxRefinement` prevents immediate unrefinement | §13 | **Hypothesis** | Failure mode not observed |
| V-13 | Thresholds: mass ratio ≥0.98, ~200k cell budget | §18 | **Assumed** | Not derived from data |
| V-14 | Small-charge table (incl. 5 kg) | §15 | **Partial (analytic)** | Geometry/formula only; not generated/run |
| V-15 | Lean startup + runtime AMR beyond the early window | §13 | **Hypothesis** | Late-time propagation/unrefinement not exercised (OQ-4) |
| V-16 | Modest runtime maxRefinement suffices | §13 | **Hypothesis** | OQ-3 |
| V-17 | Obstacle refinement independent of charge logic | §16, S-5 | **Assumed** | Not tested with charge near a refined obstacle (OQ-5) |
| V-18 | Graceful small-charge clamp/warn degradation | §15 | **Hypothesis** | Design intent, not an observed run |
| V-19 | Alignment with building3D/freeField philosophy | §3, §6 | **Partial (qualitative)** | Comparative reading, not quantitative |
| V-20 | Optional snappy charge-surface band benefit | §9, OQ-6 | **Hypothesis** | No fidelity benefit demonstrated |

**Interpretation for reviewers.** The two *correctness* foundations (V-1, V-3) are proven. The central *efficiency/architecture* claims (V-4–V-9, V-11, V-12) rest on a single measured configuration plus reasoning; they are strong but not yet generalized. No production default should be flipped until the items below are closed.

---

## 25. Prioritized Validation Experiment Plan (design only — not yet run)

All experiments use temporary case generators and modified generated cases only; **no production code or defaults are changed**, and generation-only stages precede any solver run. Priority is by information gain per unit effort.

**Priority 1 — close the biggest gaps**
- **EX-1 Matched band comparison (closes V-7).** Run a *runnable* band-ON variant (production-like shallow/zero seed, ~10²k cells) vs band-OFF at equivalent effective near-field resolution; short early window. *Proves/falsifies:* whether band-OFF matches band-ON arrival time and peak pressure at 0.5 m. *Cost:* low–moderate (short solver, 2 cases).
- **EX-2 Multi-shape/mass/mesh capture matrix (closes V-2, V-8 partial, V-14).** Generation-only: sphere/cyl/box × {5, 25 kg} × {0.2, 0.5 m}, band OFF. *Proves:* capture generality, initial cell counts, realized `L_seed`, small-charge table. *Cost:* low (no solver).

**Priority 2 — validate the load-bearing design choices**
- **EX-3 Seed/backup decoupling + margin sweep (closes V-9, V-10).** Generation-only initial-cell comparison of backup-as-seed-region vs decoupled tight seed at margins {1.1, 1.2, 1.5}; short-solver subset for early-wave fidelity. *Proves:* the cost-control claim and selects the margin.
- **EX-4 Target-N sweep (closes V-11, V-13).** N ∈ {4, 6, 8, 10}: generation-only cell counts + short-solver early-wave on one geometry. *Proves:* the N tradeoff table; derives the mass-ratio/cell-budget thresholds from data.
- **EX-5 Seed-vs-maxRefinement contract (closes V-12).** Short solver: `L_seed > maxRefinement` vs `L_seed ≤ maxRefinement`. *Proves/falsifies:* the immediate-unrefinement failure mode.

**Priority 3 — confirm mechanisms and independence**
- **EX-6 Band×seed mechanism (closes V-6).** Generation-only: fix seed, vary band radius/level; measure level-4 volume growth to confirm the grading-cascade explanation.
- **EX-7 Obstacle independence (closes V-17).** Generation-only + short solver: charge near a refined obstacle, band OFF; verify capture and obstacle refinement are mutually unaffected.

**Priority 4 — expensive, separable**
- **EX-8 Late-time propagation & unrefinement (closes V-15, V-16).** Longer solver on the band-OFF reference; measure unrefinement behind the wave, cell trajectory, and whether modest `maxRefinement` suffices. *Cost:* high — run only after P1–P3 justify it.
- **EX-9 Charge-surface band benefit (closes V-20).** Optional advanced study; lowest priority.

**Gating rule.** Defaults remain unchanged until EX-1, EX-2, and (for the cost claim) EX-3 pass; EX-4/EX-5 should pass before the auto-seed default and the seed/maxRefinement contract are adopted.

---

### Appendix A — Evidence base (selected quantitative results)

| Quantity | Band OFF (C4-new) | Band ON (B4) |
|---|---|---|
| Initial cells | 30,792 | 4,891,200 |
| Cells at max level (L4) | 20,480 | 4,733,568 |
| Max cell level | 4 | 4 |
| 0.5 m shock arrival | 3.94e-5 s | not reached (intractable) |
| 0.5 m peak pressure | 163.5 bar @ 5.53e-5 s | — |
| Cell trajectory | smooth 29k → 119k | — |
| Wall-clock (early window) | 65 s | intractable |

| Coarse sphere (cell 0.5 m), capture | seed 4 | seed 5 |
|---|---|---|
| Mass ratio | 0.938 | 0.985 |
| Charge cells | 480 | 4,032 |

(Free-field sphere, 25 kg, domain ±2 m, base cell 0.2 m unless noted, `densityGradient` AMR, probes at 0.5/0.75/1.0 m.)
