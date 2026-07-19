# Senior Technical Approval Board — Review & Decision

**Package under review:** Architecture Specification, Validation Status & Limitations (§24), Prioritized Validation Experiment Plan (§25), Implementation Roadmap (two-track).
**Mandate:** assess readiness, validation sufficiency, and decision quality. Not a redesign or red-team.

> **Immutable project record.** This document records the Approval Board decision exactly as rendered at the time of review. It is not to be modified, summarized, or reinterpreted.

---

## Review of the 15 required areas

| # | Area | Evidence status | Board view |
|---|---|---|---|
| 1 | Charge capture | **Proven (25 kg, gen-only)** | Production-grade foundation (M1) |
| 2 | Backup region | **Proven** (floor strictly encloses centre) | Sound; coarse-mesh inflation is a known tradeoff |
| 3 | Deep seeding | **Proven (1 config)** | Effective; volume control is the open lever |
| 4 | Auto seed-level | **Unproven** (reasoned) | Sensible; needs EX-4 |
| 5 | Seed/backup decoupling | **Unproven** | *Load-bearing cost claim* — untested (EX-3) |
| 6 | chargeRefineOuter policy | **Partly proven** (harm shown; "not needed" one-sided) | Demotion justified; equivalence not yet clean |
| 7 | Startup mesh philosophy | **Sound** | Endorsed |
| 8 | Runtime AMR philosophy | **Unproven** | Premise (AMR carries the wave) untested beyond early window |
| 9 | AMR indicator selection | **Unaddressed until B1** | Highest blind spot; all results densityGradient-specific |
| 10 | AMR release efficiency | **Unproven** | Core efficiency premise; only now a Pre-Default gate (B2) |
| 11 | Obstacle interaction | **Unproven** | EX-7 narrow (independence only) |
| 12 | Surface-burst / near-wall | **Unaddressed until B3** | Most common real case; zero coverage to date |
| 13 | Extremely coarse mesh | **Analytic only** | B5 cheap; do early |
| 14 | Validation completeness | **Improved, not complete** | Two-track plan closes the structural gaps |
| 15 | Production readiness | **Correctness-ready; efficiency/runtime not** | No default change yet |

---

## A–J Board determinations

- **A. Sufficiently proven:** sub-cell capture defect + fix (V-1), backup floor, deeper-seed capture gain (V-3), capture generality at 25 kg (V-2), and *that band+deep-seed explodes* (V-5). The correctness foundation is solid.
- **B. Unproven:** early-wave/cost generalization (V-8), decoupling (V-9/10), auto-seed defaults (V-11/13), seed↔maxRef contract (V-12), release efficiency (RG-2), indicator independence (RG-3), near-wall/surface (RG-4), physical correctness (RG-5), late-time (V-15/16), obstacle interaction depth (V-17).
- **C. Most dangerous assumptions:** (1) **release efficiency holds** (RG-2) — the whole rationale for retiring the band; (2) **conclusions are not indicator-locked** (RG-3); (3) **single-config results generalize** (V-8). If any is false, the efficiency case collapses.
- **D. Highest information gain:** B2 (release), B1 (indicator), EX-1 (clean band isolation), B4 (physical anchor), EX-2 (cheap, broad capture).
- **E. Low information gain (now):** EX-9 surface band, EX-6 mechanism (informational), EX-8 long tail, out-of-range mass extremes.
- **F. Safe to implement immediately:** metadata + warnings (mass ratio, L_seed/clamp, backup ratio, projected cells, band state, seed-vs-maxRef) — additive, no mesh change; plus flag-gated, off-by-default prototypes.
- **G. Must remain experimental:** auto-seed calc, seed/backup decoupling, contract enforcement, optional surface band — all opt-in until gated.
- **H. Must NOT change yet:** production defaults (`charge_refinement_level=0`, band ON), `chargeRefineOuter` existence, runtime AMR defaults / indicator, the M1 capture formula, 1D/2D.
- **I. Most likely future failure mode:** ship lean startup (band OFF), then discover runtime AMR with the default indicator **does not release cells behind the wave** (RG-2/RG-1) → no efficiency gain or worse cost than the band era, discovered post-rollout.
- **J. Most likely incorrect conclusion the team may be making:** *"chargeRefineOuter is unnecessary"* generalized from a **one-sided, single-config** result — C4-new was clean but B4 produced no matched peak, and EX-1 as originally specced confounds band with seed depth. The team may also conflate **band-OFF ≈ band-ON** with **band-OFF ≈ correct** (no physical anchor yet).

---

## Risk Assessment (major remaining risks)

| Risk | Sev | Like | Impact if late | Mitigation |
|---|---|---|---|---|
| R1 Release efficiency fails / indicator-dependent | **Critical** | Med | Efficiency premise void after rollout; rework defaults | B2 + B1 as Pre-Default gates (now in roadmap) |
| R2 Single-config generalization wrong (shape/5 kg/coarse) | High | Med | Multiple defaults blocked/regressed | EX-2 (cheap) + early-wave subset across shapes |
| R3 Band "not needed" is one-sided / EX-1 confounded | High | Med | Band-OFF default unjustified | Add isolated band arm to EX-1 |
| R4 No physical/correctness anchor | High | Med | Both configs could be wrong vs reality | B4 physical/convergence anchor |
| R5 Near-wall/surface-burst uncovered | High | High | Most common real case fails post-ship | B3 as Pre-Default gate |
| R6 Decoupling gives no benefit / hurts wave | High | Med | Cost-control claim void | EX-3 before decoupling default |
| R7 Extreme-coarse silent under-capture/explosion | Med | Med | Field correctness failure | B5 (cheap) + warnings |
| R8 Obstacle interaction beyond independence | Med-High | Med | Obstacle cases regress | Broaden EX-7 |
| R9 Default flipped too early (process) | High | Low–Med | Production regression | Two-gate enforcement; flag-gating |
| R10 Auto-seed clamp surprises users | Med | Med | Under-resolution unnoticed | Metadata/warnings + V-18 demo |

---

## Validation Prioritization (ranked)

**By information gain × risk reduction ÷ cost:**
1. **EX-2** — broad capture, generation-only, very cheap (closes V-2/8-capture/14/18).
2. **B2** — release efficiency; closes the Critical risk.
3. **B1** — indicator decision; de-risks every other runtime result.
4. **EX-1 (isolated arm)** — clean band-vs-no-band (closes V-7 properly).
5. **B4** — physical/correctness anchor (insurance against systemic error).
6. **EX-3** — decoupling cost claim.
7. **B3** — near-wall/surface burst (high real-world likelihood).
8. **EX-4 / EX-5** — auto-seed defaults / contract.
9. **B5** — extreme coarse (cheap, do opportunistically).
10. **EX-7 (broadened)** — obstacle interaction.
- *Lowest value now:* EX-6 (informational), EX-8 long tail, EX-9 surface band.

**Singletons:**
- **Highest-value experiment:** B2 (release efficiency).
- **Highest-risk assumption:** runtime AMR releases cells behind the wave at the default indicator (RG-2/RG-3).
- **Highest-risk validation gap:** near-wall / surface-burst (RG-4) — common and uncovered.
- **Highest-risk production decision:** flipping `chargeRefineOuter` → OFF by default.

---

## Implementation Readiness

**READY NOW**
1. Metadata + warnings (mass ratio, L_seed/clamp, backup ratio, projected/actual cells, band state, seed-vs-maxRef).
2. Validation harness/tooling (temporary `_val_*` generators, collectors).
3. Flag-gated auto-seed *calculator* (off by default; used by EX-2/EX-4).
4. Unit tests for the auto-seed/clamp pure functions.
5. Keeping the M1 capture fix as the recurring correctness smoke check.

**NOT READY (experimental / opt-in only)**
1. Seed/backup decoupling.
2. Auto-seed as a default.
3. Seed↔maxRef contract enforcement (warning-only first).
4. Optional snappy charge-surface band.
5. Any runtime-AMR indicator/maxRefinement default change.

**BLOCKERS (must validate before any production-default change)**
- Pre-Default gate (both tracks): EX-1 (isolated), EX-2, EX-3 + B1, B2, B3, B4 — and B5/EX-4/V-18 for auto-seed.

---

## FINAL BOARD DECISION

**1. Executive Summary.** The package is technically coherent, honest about its evidence, and unusually well-instrumented with validation gating. Its *correctness* foundation (charge capture) is proven and production-grade. Its *efficiency and runtime* claims — the reasons to retire the band and go lean — rest on a single measured configuration plus reasoning and are not yet validated, especially on the runtime side (release, indicator, walls, physical truth).

**2. Overall Technical Assessment.** Architecture: **sound and approved for planning.** Validation: **structurally complete after the two-track revision, but not yet executed.** Production readiness: **correctness-ready, efficiency/runtime-unvalidated.** The roadmap correctly forbids default changes until both tracks pass, and correctly classifies additive metadata as immediately shippable.

**3. Top 10 Remaining Risks:** R1–R10 above.

**4. Top 10 Validation Activities (ranked):** EX-2, B2, B1, EX-1(isolated), B4, EX-3, B3, EX-4/EX-5, B5, EX-7(broadened).

**5. Top 5 Safe To Implement Now:** metadata/warnings; validation harness; flag-gated auto-seed calculator; unit tests; M1 smoke check.

**6. Top 5 Must NOT Implement Yet:** band→OFF default; auto-seed→ON default; decoupling as default; contract hard-enforcement; any runtime indicator/maxRef default change.

**7. Earliest Point Defaults May Change:** only after the **Pre-Default Validation Gate** clears — Track A (EX-1 isolated + EX-2 [+ EX-3]) **and** Track B (B1 + B2 + B3 + B4) for the band-OFF flip; auto-seed adds EX-4/V-18 + B5. Additive metadata/warnings may ship now (no gate).

**8. Recommended Next Action.** Execute Phase A for both tracks, then launch the cheap/high-value cluster in parallel: **EX-2 ∥ EX-1(with the isolated band arm) ∥ B1 ∥ B5**, and stand up **B2** on the B1-selected indicator. Ship metadata/warnings immediately. Change no defaults.

**9. Final Approval Status:**

## ✅ APPROVED FOR PROTOTYPE IMPLEMENTATION ONLY

**Justification.** The board approves (a) immediate implementation of additive metadata/warnings and validation tooling, and (b) flag-gated, off-by-default prototypes (auto-seed calculator, decoupling, contract-warning) **solely to enable validation**. The board does **not** approve any production-default change, removal of `chargeRefineOuter`, or runtime-AMR default change. This status is chosen over:
- *Approved for Implementation* — rejected: the Critical risk (release efficiency / indicator dependence) and the absence of a physical anchor mean efficiency/runtime claims are unproven; shipping defaults now would risk a post-rollout reversal.
- *Limited Experimental Development Only* — judged too conservative: the correctness foundation is proven and the prototypes are safely flag-gated, so productive prototype work and validation can proceed in parallel.
- *Not Ready* — inapplicable: the package is well-formed and the foundation is solid.

**Conditions of approval:** prototypes remain off by default; both-track Pre-Default gate is mandatory before any default flip; EX-1 must be run with an isolated single-variable band arm; B4 (physical/correctness anchor) is required before the band-OFF default is even considered.

---

*This is an approval-board assessment only — no architecture redesign, no code, no new architecture proposed.*
