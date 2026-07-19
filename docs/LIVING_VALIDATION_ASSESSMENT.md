# Living Validation Assessment — 3D Startup Meshing Program

**Status:** AUTHORITATIVE technical status of the 3D startup-meshing validation program.
**Scope:** 3D blastFoam case generation (startup meshing + charge capture) and its interface to runtime AMR.
**Constraint:** No production code, defaults, architecture, or AMR settings have been changed by this validation work. All experiments live under `_val_*/` and are temporary.
**Supersedes:** the working copy at `_val_release_spatial/LIVING_VALIDATION_ASSESSMENT.md` (kept for raw history).
**Last updated:** after EX-1 / EX-1B / seed-default policy check.

---

## 1. Executive Summary

The 3D startup pipeline (`blockMesh → snappyHexMesh → setFields/setRefinedFields(refineInternal) → blastFoam` with runtime AMR) must guarantee two things: **mass-conservative charge capture** (even for charges smaller than a base cell) and **sufficient near-field resolution** to form the early blast wave. Historically both were achieved indirectly by a volumetric `chargeRefineOuter` refinement band.

This program has established, with controlled experiments and source review, that:

- Capture is now guaranteed by a mass-conserving **backup region** (Milestone 1) and is robust **without** the band — *when a seed level > 0 is used*.
- The band has **zero isolated effect on the early wave** (identical arrival/peak/converged mesh with vs without it), at both a moderate and a deeply sub-cell charge. It is pure startup overhead (+27 % to +240 % startup cells).
- The "runtime AMR releases cells behind the wave" efficiency premise is **disproven** for free-field blast in the practical propagation window — the refined region is a solid expanding product-gas ball; persistence is physical, not an AMR bug, and not indicator-specific.
- A **hidden coupling** was found and confirmed at source: the current shipping default is `seed = 0 + band ON`, the M1 mass-conserving floor is wired **only** on the `setRefinedFields` (seed > 0) path, and the band currently **masks a sub-cell zero-capture failure** of the seed-0 default. Retiring the band as a default is therefore **not a no-op**: it must be a coordinated change.

**Bottom line:** retiring `chargeRefineOuter` as a default is supported for fidelity and cost, **gated on** (a) a coordinated capture-protection change for the seed-0 path, and (b) one remaining correctness anchor (B4) against an external physical reference.

---

## 2. Current Architecture Status

- **Approved design (`docs/3D_Startup_Meshing_Architecture.md`):** capture via mass-conserving backup independent of the band; near-field via **auto-computed deep `refineInternal` seed**; `chargeRefineOuter` becomes advanced opt-in (off by default); startup meshing and runtime AMR are separate systems with a defined hand-off.
- **In-effect production code:** only the Milestone-1 capture fix is live. **Defaults are NOT yet flipped.**
- **Actual current production defaults (verified in source):** `charge_refinement_level = 0` (no deep seed, `setFields` path), `chargeRefineOuter` enabled by default (band levels 2/3), `charge_capture_mode = auto`. Auto-seed is computed for **recommendation/metadata only** — it is *not* wired into the default seed.
- **Gap between approved design and shipping defaults:** the architecture targets *band-OFF + auto-seed*; production still ships *seed-0 + band-ON*. Closing this gap is the subject of `docs/DEFAULT_TRANSITION_PLAN.md`.

---

## 3. Proven Conclusions

- **P1 — Capture is robust without the band (seed > 0).** M1 mass-conserving backup; EX-2 matrix had 0 zero-capture across sphere/cylinder/box × {5, 25} kg × {0.2, 0.5} m. (M1, EX-2)
- **P2 — Marginal capture is a seed-policy lever, not a capture defect.** EX-2 follow-up: `sph_25_05` 0.938→0.985 (seed 4→5); `cyl_5_02` 0.961→1.021 (seed 3→4). (EX-2 follow-up)
- **P3 — Early blast metrics are weakly indicator-sensitive.** Arrival/peak @0.5 m agree within ~0.4 % across `densityGradient` / `scaledDelta_p` / `scaledDelta_rho`. (B1)
- **P4 — Refinement tracks the wave.** The refinement front advances with the shock in all runs. (B2, B2ext, spatial)
- **P5 — No release behind the wave within the propagation window, and it is REAL (not a cell-count artifact).** Interior (r < 0.75 m) stays at max level at all times; the refined region is a solid r³ ball. (Release Spatial Analysis)
- **P6 — The persistence is physical (product-gas fireball), not an AMR bug.** Refined interior coincides with genuine ρ/p gradients in the detonation products; no quiescent zone exists behind the front yet. (Release Spatial Analysis)
- **P7 — Persistence is not indicator-specific.** `densityGradient` ≈ `scaledDelta_p` spatially and in total cells (832k vs 820k). (Release Spatial Analysis / `sdp_medium`)
- **P8 — The band has ZERO isolated early-wave effect (moderate charge).** At fixed seed, band ON vs OFF give identical arrival (3.588e-5 s), peak (176.15 bar @0.5 m), converged cells (111,768) and refine count (38) to solver precision. Band adds ~27 % startup cells (10,520→13,320). New policy (seed2/OFF) reproduces old default (seed0/ON) early wave identically. (EX-1)
- **P9 — The band was masking a sub-cell zero-capture failure for the seed-0 default.** `seed0 + band OFF` → blastFoam FATAL "No mass": base 0.2 m mesh has no cell-centre within the physical charge radius 0.155 m (nearest ≈ 0.173 m). The M1 mass-conserving backup is on the `setRefinedFields` (seed > 0) path only; `setFields` (seed = 0) uses the bare physical radius with no floor. (EX-1, source-confirmed)
- **P10 — Band redundancy holds at the band's HARDEST case (deeply sub-cell, r/dx = 0.30, auto-seed 4).** Isolated seed-4 band ON vs OFF: early wave identical (arrival Δ = 0.0 %, peak Δ ≈ 0.0 %); band adds 3.4× startup cells (14,440→48,768). (EX-1B)
- **P11 — The new default CAPTURES BETTER than legacy for sub-cell charges.** band-OFF + auto-seed-4 mass ratio = 1.0006 (mass-conserving `sphericalMassToCell`); legacy band-ON + seed-0 mass ratio = 0.934 (~6.6 % mass deficit; plain `sphereToCell` marks whole cells only). (EX-1B)
- **P12 — The shipping default is `seed = 0 + band ON`, and auto-seed is not wired in.** Verified: `models.py:109` (`charge_refinement_level: int = 0`); `tab_3d_general.py:2951-2952` (seed from spinbox, band enabled when levels ≠ 0); `startup_mesh_metadata.recommended_auto_seed_level` docstring "recommendation only — not applied to cases"; M1 floor gated on seed > 0 at `generator_3d.py:2121`. Analytic capture gap: fixed seed-0 fails 12/28 sub-cell combos; auto-seed-as-default fails 0/28. (Seed-policy check)

---

## 4. Disproven Conclusions

- **D1 — "Lean startup is cheap because AMR releases cells behind the wave."** Disproven for free-field blast in the propagation window; release is a late-time phenomenon beyond practical domains/windows. (P5–P7)
- **D2 — "densityGradient may be the limiting factor for release."** Disproven; indicator choice does not change interior persistence. (P7)
- **D3 — "Total cell count peaking ⇒ release failed (B2/B2ext)."** Corrected: the *spatial* picture, not total count, is the right metric; total count grows mainly because the refined ball expands.
- **D4 — "Retiring the band is a no-op for the early wave."** Corrected/nuanced: isolated band toggle IS a no-op (P8/P10), but flipping the *default* (which also changes seed 0→auto) is NOT a no-op for deeply sub-cell charges — it changes near-field resolution and mass capture (P11). The change appears to be an improvement, but correctness is unconfirmed (U2).

---

## 5. Remaining Unknowns

- **U1 — Early-wave equivalence band ON vs OFF — RESOLVED (EX-1/EX-1B).** No isolated early-wave effect, including the deeply sub-cell case. Minor caveat: only near-field probes are loaded in the 7e-5 s window.
- **U2 — Physical/correctness anchor (B4) — OPEN, highest priority.** Does band-OFF + auto-seed reproduce a known free-air blast scaling? Current probes (0.3–1.0 m) sit *inside the fireball* (α_c4 > 0.9), so existing data cannot anchor air-blast scaling; needs probes beyond the contact surface (larger domain / longer run). This is the last gate for retirement sign-off.
- **U3 — Near-wall / surface-burst behavior (B3) — OPEN.** Untested.
- **U4 — Late-time release (EX-8) — OPEN.** When (if) the interior finally releases; expensive, low current decision value.
- **U5 — Extreme-coarse regime (B5) — OPEN.** P9 makes this important: any B5 must use seed ≥ 1 / a setFields floor, or it will hit the same no-mass failure.

---

## 6. Current Risks

- **R1 (high) — Runtime cost scales as the refined product-gas ball (~(R_front/Δx)³·8^L), band-independent.** Long-propagation / large-domain runs are expensive regardless of startup strategy. This is the dominant cost driver, not startup meshing.
- **R2 (medium) — The architecture's efficiency narrative overstates runtime release.** Should be corrected to: savings come from leaner startup + bounded `maxRefinement` + domain/time sizing, not from release behind the wave.
- **R3 (high) — Correctness not yet anchored to an external physical reference (U2/B4).** Self-consistency (band ON ≈ OFF) is not the same as correctness.
- **R4 (high, CONFIRMED) — Retiring the band ALONE (seed-0 default unchanged) reintroduces sub-cell zero-capture (P9/P12).** The band is currently load-bearing for capture in the shipping default. Band-OFF is safe only when paired with auto-seed-as-default (closes 12/28 → 0/28 capture gap) OR a setFields/seed-0 backup-radius floor.

---

## 7. Validation History

| Milestone | Objective | Result | Key finding | Status |
|---|---|---|---|---|
| **M1 — Charge capture fix** | Eliminate sub-cell zero-capture defect (RC-1) | Mass-conserving backup (`sphericalMassToCell` + floored radius via `CAPTURE_CELL_SAFETY`) added on the seed>0 path | Capture decoupled from the band when seed>0 | **PASS** |
| **EX-2 — Capture matrix** | Confirm capture across shapes/masses/meshes, band OFF | 0 zero-capture across sphere/cyl/box × {5,25} kg × {0.2,0.5} m | Capture is robust without the band (seed>0) | **PASS** |
| **EX-2 Follow-up** | Diagnose 2 marginal cases (seed vs deeper limitation) | `sph_25_05` 0.938→0.985 (seed4→5); `cyl_5_02` 0.961→1.021 (seed3→4) | Marginal capture is a seed-level lever, not a defect | **PASS** |
| **B1 — Indicator comparison** | Is runtime behavior indicator-dependent? | Arrival/peak @0.5 m within ~0.4 % across 3 indicators | Early behavior weakly indicator-sensitive; `densityGradient` fine | **PASS** |
| **B2 — Release efficiency** | Validate "cells released behind the wave" | Cell count still rising at window end (1.5e-4 s); built `parse_amr_history` (harness misorders AMR events) | Release not observed in window; needed accurate history + extension | **PARTIAL** |
| **B2 Extension** | Continue to 2.5e-4 s | Mesh kept growing; no sustained release | Confirmed persistence; total-count metric is misleading | **PARTIAL** |
| **Release Spatial Analysis** | Is "no release" real, an AMR bug, or indicator-specific? | Interior stays max-level; refined region a solid ball; coincides with real ρ/p gradients; `densityGradient`≈`scaledDelta_p` | Persistence is PHYSICAL (product-gas fireball), not a bug, not indicator-specific | **PASS** |
| **EX-1 — Isolate the band** | Isolated band effect on early wave + capture coupling | 2×2 seed×band: 3 arms identical to solver precision; `seed0+bandOFF` FATAL no-mass | Band = pure startup overhead; band masked seed-0 capture failure | **PASS** |
| **EX-1B — Sub-cell hard case** | Retirement test where band mattered most (r/dx=0.30, auto-seed 4) | Isolated seed-4 band ON≈OFF (identical); new default mass ratio 1.0006 vs legacy 0.934 | Band redundant even here; new default captures *better*; new-vs-legacy divergence is seed-driven | **PASS** |
| **Seed-default policy check (#1)** | Is band retirement "free"? (source + analytic) | Default = seed0+bandON; auto-seed not wired; analytic gap 12/28 → 0/28 with auto-seed | Retirement requires coordinated capture protection (R4) | **PASS** |

---

## 8. Current Recommendation

Retire `chargeRefineOuter` as a **default** (keep it as an advanced opt-in), **but only as a coordinated change** that protects seed-0 capture, and **only after** the B4 correctness anchor confirms the band-OFF + auto-seed early/near-field solution against an external physical reference. See `docs/DEFAULT_TRANSITION_PLAN.md` for the two coordinated options (A: auto-seed-as-default; B: extend the M1 floor to the seed-0 path). Do **not** implement the default flip yet.

---

## 9. Highest-Value Next Action

**B4 — physical correctness anchor.** Run band-OFF + auto-seed (and a band-ON / legacy comparator) with probes **beyond the contact surface** (α_c4 ≈ 0), in a domain/time window large enough to compare far-field peak overpressure and arrival against a known free-air blast scaling (and/or a finer-mesh reference). This is the only remaining gate (U2/R3) before a retirement sign-off. It needs a larger-domain / longer run (higher cost); design it with a bounded charge/domain to control R1 cost.

---

## 10. What Must NOT Be Changed Yet

- Do **not** flip the production default for `chargeRefineOuter` (still ON) or for `charge_refinement_level` (still 0) before the coordinated capture protection is approved.
- Do **not** wire auto-seed into the default seed, or extend the M1 floor to the `setFields` path, without explicit approval (both are production changes; analyzed in the transition plan).
- Do **not** change AMR defaults (`densityGradient`, `maxRefinement`, `refineInterval`, thresholds) — out of scope for this program.
- Do **not** alter the M1 capture code, capture factors, or the architecture hand-off contract.
- Keep all validation artifacts confined to `_val_*/`.

---

## 11. Project Timeline (problem → present)

| Step | Lesson learned |
|---|---|
| **Problem discovered** | Sub-cell charges could capture **zero** mass when the band was off (silent correctness failure); some band×seed combos exploded the mesh. |
| **→ M1** | A mass-conserving backup region fixes capture independent of refinement — *on the seed>0 path*. |
| **→ EX-2** | Capture is robust band-OFF across shapes/masses/meshes; marginal cases are a seed lever. |
| **→ B1** | Runtime/early behavior is only weakly indicator-dependent; `densityGradient` is a fine default indicator. |
| **→ B2** | "Release behind the wave" is not visible in the propagation window; the harness's AMR-event ordering is unreliable (built `parse_amr_history`). |
| **→ B2 Extension** | Extending time doesn't produce release; total cell count is a misleading metric. |
| **→ Release Analysis** | The "no release" is **physical** (a solid product-gas fireball with real gradients), not a bug or an indicator artifact — correcting the efficiency narrative. |
| **→ EX-1 / EX-1B** | The band has **no** isolated early-wave effect (even sub-cell); it was **masking** a seed-0 capture failure; the new default captures *better*. Retirement is sound but must be coordinated; correctness still needs an external anchor (B4). |

---

*Cross-references: `docs/3D_Startup_Meshing_Architecture.md` (approved design), `docs/ARCHITECTURE_DECISIONS.md` (decision log), `docs/DEFAULT_TRANSITION_PLAN.md` (default-flip planning). Raw experiment artifacts under `_val_m1/`, `_val_ex2/`, `_val_ex2_followup/`, `_val_b1/`, `_val_b2/`, `_val_release_spatial/`, `_val_ex1/`.*
