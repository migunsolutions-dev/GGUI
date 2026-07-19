# 3D Startup Meshing Architecture — Implementation Roadmap

**Role of this document:** Technical program management plan for implementing the approved architecture in `docs/3D_Startup_Meshing_Architecture.md`.
**Authoritative inputs:** the Architecture Specification (§1–§23), Validation Status & Limitations (§24), and the Prioritized Validation Experiment Plan (§25) of that document.
**Standing constraints (non-negotiable):** the architecture is approved for *planning* only; it is not re-designed here. No production code, patches, or default changes are proposed ahead of their validation gates. No validation step may be skipped. `chargeRefineOuter` is not removed or defaulted-off before its gate. Auto-seed defaults are not adopted before their gate. 1D/2D workflows are untouched throughout.

Throughout, items are labelled by class: **[HYP]** architecture hypothesis, **[VAL]** validated finding, **[PROTO]** prototype implementation, **[ACT]** validation activity, **[PROD]** production change.

---

## 1. Executive Summary

The architecture replaces band-driven near-field meshing with (a) a floored, mass-conserving capture backup (already fixed and proven in M1), and (b) automatically-leveled deep seeding over a tight charge region, with `chargeRefineOuter` demoted to an advanced opt-in. Two correctness foundations are proven (V-1, V-3); the cost/efficiency architecture rests on a single measured configuration plus reasoning and is therefore **not yet cleared for default changes**.

This roadmap sequences the work as five phases (A planning/validation-prep → B prototype-behind-flags → C experimental validation → D default-transition readiness review → E staged production rollout), executed as **two parallel validation tracks** — **Track A (Startup Meshing)** and **Track B (Runtime)** — that converge at a **Pre-Default Validation Gate** and a **Pre-Production Validation Gate**. Every **[HYP]/[Partial]/[Assumed]** item from §24, plus the runtime/coverage review gaps **RG-1…RG-6** (runtime fidelity, release efficiency, indicator dependence, near-wall/surface burst, physical/convergence anchor, extreme-coarse mesh), is tracked to a validation activity, an implementation dependency, and a production dependency. Hard gates prevent any production default change until **both tracks** pass: Track A (EX-1 early-wave equivalence, EX-2 capture generality, EX-3 decoupling) **and** Track B (B1 indicator decision, B2 release efficiency, B3 near-wall/surface burst, B4 physical anchor, B5 extreme-coarse), with EX-4/EX-5 and B5 gating the auto-seed default. These runtime risks are now **Pre-Default gates rather than post-rollout discoveries**, which is the central change in this revision.

The single highest-value, lowest-cost first action is the **EX-2 generation-only capture matrix** (broad, cheap, closes capture-generality gaps), executed in parallel with **EX-1** (the matched band comparison that closes the central one-sided hypothesis V-7).

---

## 2. Current State Assessment

| Area | State | Class |
|---|---|---|
| M1 capture backup fix (floor, cyl axial, box backup) | Implemented, unit-tested, in effect | [VAL] |
| Capture generality at 25 kg (sphere/cyl/box, 0.2/0.5) | Generation-only validated | [VAL] |
| Deep-seed early-wave (band OFF) | Measured for one config only | [VAL]/[Partial] |
| Auto seed-level policy | Designed, not implemented | [HYP] |
| Seed/backup decoupling, margin | Designed, not implemented, untested | [HYP] |
| `chargeRefineOuter` | Enabled by default (unchanged) | as-is |
| Production defaults | `charge_refinement_level=0`, band ON | unchanged |
| Validation harness | Temporary generators exist (`_val_m1`, `_val_m2`) | [ACT] tooling |

**Conclusion:** correctness is in good standing; efficiency architecture is pre-validation. No production behavior has changed beyond the M1 capture fix.

---

## 3. Validated Findings ([VAL])

- **V-1** Sub-cell capture defect identified and fixed (backup floor). *Proven (M1).*
- **V-3** Deeper seed improves coarse-mesh capture (0.938→0.985). *Proven (M1).*
- **V-2 (25 kg)** Band-OFF capture works for sphere/cyl/box at 25 kg. *Proven (M1); 5 kg pending.*
- **V-4/V-5 (one config)** Band-OFF forms a clean early wave (30,792 cells; arrival 3.94e-5 s; 163.5 bar; 65 s); band-ON at equal seed depth explodes (~159×, intractable). *Proven for the single M2 config.*

These are inputs to planning and are not re-validated, except where generalization is claimed (V-2 5 kg, V-4/V-5 other configs → tracked as open).

---

## 4. Open Hypotheses (must be tracked; not validated)

Tracked items with their closing experiment, implementation dependency, and production dependency.

| V-item | Claim | Validation required | Impl. dependency | Production dependency |
|---|---|---|---|---|
| **V-7** | Band not needed for early formation | EX-1 (matched band ON vs OFF) | none (uses generated cases) | Gate for band→OFF default |
| **V-8** | Early-wave/cost generalizes to cyl/box/5 kg/coarse | EX-2 (capture) + EX-1/EX-3 short-solver subset (early wave) | none | Gate for any default change |
| **V-9** | Seed/backup decoupling controls cost | EX-3 | [PROTO] decoupling flag | Gate for decoupling in defaults |
| **V-10** | Seed margin ~1.2× | EX-3 (margin sweep) | [PROTO] margin param | Gate for margin default |
| **V-11** | Auto-seed N=6, L_max=5 + N tradeoff | EX-4 | [PROTO] auto-seed calc | Gate for auto-seed default |
| **V-12** | `L_seed ≤ maxRefinement` contract | EX-5 | [PROTO] contract check/warn | Gate for contract enforcement |
| **V-13** | Thresholds (mass≥0.98, ~200k budget) | EX-4/EX-2 (derive from data) | [PROTO] metadata thresholds | Gate for warning thresholds |
| **V-14** | Small-charge table incl. 5 kg | EX-2 | none | Gate for small-charge claims |
| **V-15** | Lean startup + AMR beyond early window | EX-8 (late-time) | none | Post-rollout / maxRef default |
| **V-16** | Modest runtime maxRefinement suffices | EX-8 | none | Gate for maxRef default change |
| **V-17** | Obstacle refinement independent of charge | EX-7 | none | Gate for obstacle-case defaults |
| **V-18** | Graceful small-charge clamp/warn | EX-2 (observe clamp) + [PROTO] | [PROTO] clamp/warn | Gate for auto-seed default |
| **V-19** | building3D/freeField alignment | [ACT] documented comparison (no run) | none | Informational; not a hard gate |
| **V-20** | Charge-surface band benefit | EX-9 | [PROTO] optional band | Advanced feature only; never a default |
| **V-6** | Band×seed mechanism explanation | EX-6 | none | Informational; supports V-7 |

---

## 5. Validation Dependency Graph

The program now runs as **two parallel tracks** (Track A — Startup Meshing Validation; Track B — Runtime Validation) that converge at two gates: the **Pre-Default Validation Gate** (before any default flip) and the **Pre-Production Validation Gate** (before production rollout is declared complete). Both tracks must pass the Pre-Default gate before any production-default change.

```
                            [M1 fix: V-1, V-3]  (DONE)
                                     |
        ===================== PARALLEL TRACKS =====================
        |                                                         |
  TRACK A (startup meshing)                          TRACK B (runtime)
        |                                                         |
   EX-2 capture matrix  ----.                         B1 indicator comparison (RG-3)
   (V-2 5kg,V-8 cap,V-14)    |                              |
   EX-1 matched band (V-7) --+                         B2 release efficiency (RG-1,RG-2)
   EX-6 mechanism (V-6)      |                              |
        |                    |                         B3 near-wall / surface burst (RG-4)
   EX-3 decoupling+margin (V-9,V-10) -- early-wave -.       |
        |                    |              (-> V-8) |  B4 physical / convergence anchor (RG-5)
   EX-4 N-sweep (V-11,V-13)  |                       |       |
   EX-5 seed<=maxRef (V-12)  |                       |  B5 extreme-coarse mesh (RG-6, V-18/R-2)
        |                    |                       |       |
        +---------- TRACK A subtotal ---------+      +--- TRACK B subtotal ---+
                           |                                 |
                           +============ JOIN ===============+
                                          |
                          [ PRE-DEFAULT VALIDATION GATE ]
                          (Track A gates AND Track B gates)
                                          |
                            Phase D readiness review
                                          |
                          Phase E staged default rollout
                                          |
                          [ PRE-PRODUCTION VALIDATION GATE ]
                                          |
              EX-8 long-duration tail (V-15) + EX-9 surface band (V-20) [advanced/late]
```

EX-6 (mechanism, V-6) and the V-19 documented comparison are informational and do not gate rollout. Track B's release-efficiency / runtime-fidelity checks (B2, formerly the gating portion of EX-8) are **promoted to Pre-Default gates**; only the long-duration propagation tail remains late.

---

## 5-bis. Two-Track Validation Structure & Why the Architecture Is Unchanged

**Track A — Startup Meshing Validation** covers what startup produces: charge capture, seed level, seed/backup decoupling, the auto-seed policy, and the seed↔maxRefinement contract. Milestones: EX-1…EX-7, EX-9 (§7).

**Track B — Runtime Validation** covers what the solver does with that mesh: AMR indicator behavior, refinement/unrefinement (release) efficiency, near-wall/surface-burst reflection handling, a physical/convergence anchor, and the extreme-coarse regime. Milestones: B1…B5 (§7B).

**Review-gap (RG) identifiers** introduced for the runtime/coverage findings (tracked here in the roadmap, *not* added to the architecture's §24 evidence table):
- **RG-1** runtime AMR fidelity beyond the early window;
- **RG-2** AMR release efficiency (unrefinement behind the wave);
- **RG-3** indicator-dependence of all conclusions (densityGradient vs alternatives);
- **RG-4** near-wall / reflecting-surface / boundary cases;
- **RG-5** physical-reference / mesh-convergence anchor (correctness, not just band-ON≈band-OFF);
- **RG-6** extremely coarse mesh (charge ≪ base cell, clamp-binding).

**Why the architecture remains unchanged.** Every item above is a *validation-coverage* gap, not an architectural defect. None of them proposes a different capture mechanism, seed policy, backup rule, band role, or startup/runtime separation than what §1–§23 already specify; they ask whether the *already-chosen* design holds under conditions not yet measured (other indicators, release behavior, walls, coarse meshes, physical truth). The architecture's design decisions — mass-conserving capture + floored backup, tight deep seed with auto level, advanced-only band, startup/runtime separation with the `L_seed ≤ maxRefinement` contract — are unaffected. Track B therefore *tests* the architecture's runtime-interaction assumptions (notably §13 and §6 P-2) rather than altering them. If a Track B gate fails, the response is to hold the relevant default change and escalate, **not** to silently redesign.

---

## 6. Development Phases

| Phase | Theme | Gate to exit |
|---|---|---|
| **A** | Planning & validation preparation | Harness + metrics + experiment specs ready, reviewed |
| **B** | Prototype implementation behind flags | Prototypes generate cases; defaults unchanged; regression-clean |
| **C** | Experimental validation (Track A **and** Track B in parallel) | Track A: EX-1..EX-7 executed; Track B: B1..B5 executed; results recorded |
| **D** | Default-transition readiness review | Go/No-Go on V-7/8/9/10/11/12/13/14/17 **and** RG-1..RG-6 (Pre-Default gate) |
| **E** | Staged production rollout | Defaults flipped per gates; Pre-Production gate + post-rollout validation green |

Track A and Track B both proceed during Phase C and **both must pass the Pre-Default Validation Gate (Phase D)** before Phase E begins. Neither track alone is sufficient for a default change.

(Detailed entry/exit/risks/rollback in §"Implementation Strategy" below.)

---

## 7. Track A — Startup Meshing Validation Milestones (EX-1 … EX-9)

Each milestone is generation-only first; solver stages use the short early-propagation window (to the 0.5 m probe) unless stated. No production code or defaults are touched by any experiment.

### EX-1 — Matched band comparison (closes V-7; supports V-8 wave)
- **Objective:** obtain a matched band-ON vs band-OFF early-wave comparison that B4 could not provide.
- **Expected outcome:** band-OFF matches band-ON 0.5 m arrival/peak at far lower cells/runtime.
- **Information gained:** whether the band contributes anything to early formation at runnable refinement.
- **Implementation required:** none (temporary generated cases; band-ON at production-like shallow seed so it is runnable).
- **Estimated cost:** low–moderate (2 short solver runs).
- **Success:** arrival within ~10% and peak within ~15% between cases; band-OFF cheaper.
- **Failure:** band-ON yields materially earlier/stronger or qualitatively different early wave.
- **Follow-up:** if failure → band retains an early-formation role; revisit §14 before any default change.

### EX-2 — Multi-shape/mass/mesh capture matrix (closes V-2 5 kg, V-8 capture, V-14, partial V-18)
- **Objective:** prove capture generality band-OFF across the full envelope.
- **Expected outcome:** mass ratio ≥0.98 for all; no zero-capture; bounded initial cells.
- **Information gained:** realized `L_seed`, backup, cell counts; empirical small-charge table; clamp behavior.
- **Implementation required:** temporary generator with the auto-seed formula (prototype calc, not production).
- **Estimated cost:** low (generation-only, 12 cases: 3 shapes × 2 masses × 2 cells).
- **Success:** all capture ≥0.98; clamp cases warn rather than fail.
- **Failure:** any zero/low capture or unbounded cells.
- **Follow-up:** feed thresholds (V-13) and small-charge table (V-14/§15).

### EX-3 — Seed/backup decoupling + margin sweep (closes V-9, V-10; V-8 wave subset)
- **Objective:** validate the cost-control claim and select the seed margin.
- **Expected outcome:** decoupled tight seed gives materially fewer initial cells than backup-as-seed with equivalent early-wave fidelity.
- **Information gained:** margin {1.1,1.2,1.5} vs cells vs early-wave fidelity.
- **Implementation required:** [PROTO] decoupling + margin parameter (flagged).
- **Estimated cost:** moderate (generation-only sweep + short-solver subset).
- **Success:** ≥ target cell reduction at no measurable early-wave degradation; a margin chosen.
- **Failure:** decoupling degrades early wave or yields no cell benefit.
- **Follow-up:** lock margin default candidate; gate for decoupling in defaults.

### EX-4 — Target-N sweep (closes V-11, V-13)
- **Objective:** validate auto-seed defaults and derive thresholds from data.
- **Expected outcome:** N=6 balances fidelity/cost; N≥8 adds little once mass conserved.
- **Information gained:** fidelity/cost vs N; data-based mass-ratio/cell-budget thresholds.
- **Implementation required:** [PROTO] auto-seed calc (flagged).
- **Estimated cost:** moderate (gen-only counts + short solver on one geometry).
- **Success:** a defensible default N and L_max; thresholds derived.
- **Failure:** no N gives acceptable fidelity within cell budget → revisit policy (escalate, do not silently change architecture).
- **Follow-up:** set auto-seed default candidate; gate for auto-seed default.

### EX-5 — Seed-vs-maxRefinement contract (closes V-12)
- **Objective:** confirm the immediate-unrefinement failure mode and the protective contract.
- **Expected outcome:** `L_seed > maxRef` → seed unrefined at t≈0; `L_seed ≤ maxRef` → retained.
- **Information gained:** validity of the `L_seed ≤ maxRefinement` rule.
- **Implementation required:** [PROTO] contract check + warning (flagged).
- **Estimated cost:** low (2 short solver runs).
- **Success:** failure mode reproduced and prevented by the contract.
- **Failure:** seed retained regardless → contract unnecessary; relax warning.
- **Follow-up:** finalize warning behavior.

### EX-6 — Band×seed mechanism (closes V-6)
- **Objective:** confirm the grading-cascade explanation for the explosion.
- **Expected outcome:** level-4 volume grows with band radius/level at fixed seed.
- **Information gained:** causal confirmation for §4 RC-3.
- **Implementation required:** none (generation-only parameter variation).
- **Estimated cost:** low.
- **Success:** monotonic volume growth consistent with cascade.
- **Failure:** explosion independent of band extent → mechanism mis-stated (informational only).
- **Follow-up:** annotate §4 with evidence; does not gate rollout.

### EX-7 — Obstacle independence (closes V-17)
- **Objective:** verify charge capture/seed and obstacle refinement are mutually independent.
- **Expected outcome:** charge near a refined obstacle captures correctly; obstacle refinement unchanged.
- **Information gained:** independence holds (or interaction exists) for obstacle cases.
- **Implementation required:** none beyond existing obstacle path.
- **Estimated cost:** moderate (gen-only + one short solver).
- **Success:** capture ≥0.98; obstacle refinement bytes/levels unchanged vs no-charge baseline.
- **Failure:** interaction observed → flag for architecture follow-up before obstacle defaults.
- **Follow-up:** gate obstacle-case defaults on result.

### EX-8 — Late-time propagation & unrefinement (closes V-15, V-16)
- **Objective:** confirm lean startup + runtime AMR behaves well beyond the early window and that modest maxRefinement suffices.
- **Expected outcome:** wave tracked, cells released behind it, bounded totals.
- **Information gained:** late-time fidelity; default maxRefinement guidance.
- **Implementation required:** none.
- **Estimated cost:** **high** (long solver) — run only after P1–P3 justify.
- **Success:** acceptable late-time fidelity + unrefinement at modest maxRef.
- **Failure:** poor tracking/release → maxRef default unchanged; flag runtime-AMR work (separate track).
- **Follow-up:** informs maxRef default (post-rollout or pre-rollout for that single default).
- **Reframing under the two-track model:** EX-8's *release-efficiency / runtime-fidelity* content is promoted into **Track B (B2)** as a Pre-Default gate; only the **long-duration propagation tail** remains here as a late, Pre-Production / post-rollout activity.

### EX-9 — Charge-surface band benefit (closes V-20)
- **Objective:** assess whether the optional snappy surface band improves fidelity enough to justify inclusion.
- **Expected outcome:** marginal benefit; advanced-only.
- **Information gained:** keep/drop the advanced surface band.
- **Implementation required:** [PROTO] optional surface band (flagged, advanced).
- **Estimated cost:** moderate; lowest priority.
- **Success/Failure:** measurable fidelity gain vs cost (keep) / none (drop).
- **Follow-up:** advanced feature only; never a default.

---

## 7B. Track B — Runtime Validation Milestones (B1 … B5)

Track B validates the solver's behavior on the meshes Track A produces. It runs **in parallel** with Track A during Phase C. Each milestone uses the band-OFF deep-seed reference configuration unless noted, and the medium window is "until the wave has clearly propagated and begun releasing cells behind it" (longer than the Track A early window, shorter than the EX-8 tail). No production code or defaults are touched.

### B1 — AMR Indicator Comparison (closes RG-3; informs RG-1)
- **Objective:** determine whether the program's conclusions are indicator-dependent by comparing the supported error estimators (`densityGradient`, `scaledDelta`, `scaledDelta(p)`, and any others supported by the GUI's `errorEstimator` options).
- **Expected outcome:** one or more indicators give comparable or better tracking/release at similar/lower cost; conclusions are not critically indicator-locked — or, if they are, this is surfaced *before* defaults change.
- **Information gained:** shock-tracking quality, peak cell count, release behavior, runtime cost per indicator.
- **Implementation required:** none beyond selecting existing indicator options in generated cases (no new indicator code).
- **Estimated cost:** moderate (one geometry × N indicators, medium window).
- **Success:** the recommended default indicator is identified and justified; Track A early-wave conclusions reproduce under it.
- **Failure:** results swing materially by indicator and no single choice is defensible → hold all efficiency-motivated defaults; escalate indicator decision.
- **Follow-up:** record the chosen indicator as the validation baseline for B2/EX-8.

### B2 — Release Efficiency Validation (closes RG-2; informs RG-1, V-15)
- **Objective:** verify refined cells are released behind the propagating wave (the core efficiency premise behind retiring the band).
- **Expected outcome:** peak cell count is followed by meaningful unrefinement; final ≪ peak.
- **Information gained:** peak cells, final cells, **release ratio** (final/peak), refinement persistence vs time/distance.
- **Implementation required:** none (metrics from logs; reuse collectors).
- **Estimated cost:** moderate (medium-window solver on the reference config; repeated for the B1-selected indicator).
- **Success:** release ratio meets an agreed target (e.g. cells released as the wave passes; no permanent over-refinement of vacated regions).
- **Failure:** refinement persists behind the wave (consistent with the earlier Phase-4.5 concern) → efficiency premise unproven; **block band-OFF/auto-seed efficiency claims and defaults**; escalate to runtime-AMR work.
- **Follow-up:** feed maxRefinement/refineInterval guidance; gate the efficiency-motivated default flips.

### B3 — Near-Wall / Surface-Burst Validation (closes RG-4; complements EX-7)
- **Objective:** validate startup capture and AMR when the charge is near a wall, on a reflecting surface (hemispherical surface burst), or near a domain boundary.
- **Expected outcome:** capture remains mass-conservative with the backup/seed region adjacent to the surface; early wave and reflection form correctly; AMR tracks the reflected wave.
- **Information gained:** capture robustness near boundaries, early-wave formation, reflection handling, AMR behavior near surfaces.
- **Implementation required:** none beyond generating wall/symmetry/ground-plane cases (existing case options).
- **Estimated cost:** moderate (generation-only capture checks + medium-window solver on 1–2 configs).
- **Success:** capture ≥0.98 near the surface (no clipping/leak); reflection physics qualitatively correct; no AMR instability at the surface.
- **Failure:** capture clips/leaks at the boundary or reflection mishandled → block defaults for wall/surface cases; escalate.
- **Follow-up:** extend the small-charge / matrix tables with the surface-burst class.

### B4 — Physical Reference / Convergence Anchor (closes RG-5)
- **Objective:** ensure the program is anchored to *correctness*, not only band-ON≈band-OFF self-consistency.
- **Expected outcome:** the band-OFF reference reproduces a recognized physical benchmark (e.g. free-air scaled overpressure/arrival from Hopkinson–Cranz / UFC-3-340 scaling) **and/or** peak pressure/arrival converges under base-mesh refinement.
- **Information gained:** absolute confidence that band-OFF ≈ correct, plus a convergence baseline.
- **Information required (define at least one):** (a) a physical-reference comparison at one scaled distance, or (b) a base-mesh convergence series on the reference config.
- **Implementation required:** none (analysis of solver output vs reference).
- **Estimated cost:** moderate–high (convergence series implies several runs).
- **Success:** agreement within an agreed tolerance with the physical reference and/or demonstrated convergence.
- **Failure:** band-OFF deviates from physical reference beyond tolerance → correctness concern that overrides self-consistency; **block defaults**; escalate (may indicate a resolution/indicator issue, not an architecture issue).
- **Follow-up:** record tolerance and the converged baseline as the validation ground truth.

### B5 — Extremely Coarse Mesh Validation (closes RG-6; V-18, R-2)
- **Objective:** validate the regime charge ≪ base cell (e.g. 5 kg charge, base cell 0.8–1.0 m).
- **Expected outcome:** capture remains robust (mass-conserving) via the floored backup; `L_seed` clamps at `L_max` with a warning; backup inflation is bounded and reported; degradation is graceful (warn + finer-mesh suggestion), never a silent failure or explosion.
- **Information gained:** capture robustness, seed-level clamping behavior, backup-region behavior, graceful-degradation behavior at the extreme.
- **Implementation required:** prototype auto-seed clamp/warn (already flagged for Track A); no new architecture.
- **Estimated cost:** low (generation-only) + optional one short solver.
- **Success:** capture ≥0.98 or an explicit, correct warning; no zero-capture; no cell explosion; backup-vs-charge ratio reported.
- **Failure:** silent under-capture or unbounded cells at the extreme → strengthen warnings before any auto-seed default; do not change defaults for this regime.
- **Follow-up:** finalize clamp/warn thresholds; feed §15/§18 metadata.

---

## 8. Deliverables Per Milestone

| Milestone | Deliverables |
|---|---|
| **Phase A** | Validation harness (reuses `_val_*` temp generators), metric definitions, per-experiment specs, reviewed test plan |
| **EX-1** | Matched comparison report (arrival/peak/cells/runtime), V-7 disposition |
| **EX-2** | Capture matrix table (mass ratio, cells, L_seed, backup), empirical small-charge table, V-2/8/14/18 disposition |
| **EX-3** | Decoupling cost report, chosen margin candidate, V-9/10 disposition |
| **EX-4** | N-sweep fidelity/cost report, default N/L_max candidate, derived thresholds, V-11/13 disposition |
| **EX-5** | Contract report, warning spec, V-12 disposition |
| **EX-6** | Mechanism note appended to §4 evidence, V-6 disposition |
| **EX-7** | Obstacle-independence report, V-17 disposition |
| **EX-8** | Late-time report, maxRef guidance, V-15/16 disposition |
| **EX-9** | Surface-band assessment, V-20 disposition |
| **B1** | Indicator comparison report (tracking/peak cells/release/cost), chosen default indicator, RG-3 disposition |
| **B2** | Release-efficiency report (peak/final cells, release ratio, persistence), RG-2 disposition |
| **B3** | Near-wall/surface-burst report (capture, early wave, reflection, AMR), RG-4 disposition |
| **B4** | Physical-reference/convergence report + ground-truth tolerance, RG-5 disposition |
| **B5** | Extreme-coarse report (capture, clamp/warn, backup behavior), RG-6/V-18 disposition |
| **Phase D** | Readiness review packet (all gating V-items **and** RG-items green/red), Go/No-Go record |
| **Phase E** | Staged default-change record + post-rollout validation report |

---

## 9. Validation Requirements Per Milestone

**Track A (startup):**
- **Generation-only experiments (EX-2, EX-6, parts of EX-3/EX-4, B5):** mass ratio, initial cell count, realized `L_seed`, backup geometry, clamp/warn events — all recorded; golden-dict regression vs prior generation to prove no unintended change.
- **Short-solver experiments (EX-1, EX-5, EX-7, subsets of EX-3/EX-4):** 0.5 m arrival, peak pressure, cell-count-vs-time, wall-clock; compared to the M2 band-OFF reference.

**Track B (runtime):**
- **Medium-window experiments (B1, B2, B3):** shock-tracking quality, **peak cell count, final cell count, release ratio (final/peak), refinement persistence vs time/distance**, per-indicator runtime cost, and (B3) reflection behavior near walls/surfaces.
- **Reference/convergence (B4):** comparison vs a physical benchmark and/or peak-pressure/arrival convergence under base-mesh refinement.
- **Long-duration tail (EX-8):** late-time cell trajectory, sustained unrefinement behind the wave, total cells, stability.

- **All (both tracks):** results logged with exact case parameters; the chosen B1 indicator is the baseline for B2/EX-8; no production default touched.

---

## 10. Acceptance Criteria Per Milestone

| Milestone | Accept when |
|---|---|
| Phase A | Harness reproduces M1/M2 reference numbers; specs reviewed |
| EX-1 | Band-OFF matches band-ON early wave within tolerance at lower cost (V-7 PASS) |
| EX-2 | All envelope cases capture ≥0.98, no zero-capture, bounded cells (V-2/8/14 PASS) |
| EX-3 | Decoupling reduces cells at no early-wave loss; margin selected (V-9/10 PASS) |
| EX-4 | Default N/L_max defensible; thresholds derived (V-11/13 PASS) |
| EX-5 | Contract validated or shown unnecessary (V-12 resolved) |
| EX-6 | Mechanism confirmed or corrected (V-6 resolved) |
| EX-7 | Independence holds (V-17 PASS) |
| EX-8 | Late-time acceptable at modest maxRef (V-15/16 PASS) |
| B1 | Defensible default indicator chosen; Track A conclusions reproduce (RG-3) |
| B2 | Release ratio meets target; final ≪ peak (RG-2 PASS) |
| B3 | Capture ≥0.98 near surface; reflection correct; AMR stable (RG-4 PASS) |
| B4 | Agreement with physical reference and/or demonstrated convergence (RG-5 PASS) |
| B5 | Extreme-coarse capture robust with correct clamp/warn (RG-6 PASS) |
| EX-9 | Clear keep/drop decision (V-20 resolved) |

---

## 11. Failure Criteria Per Milestone

- **EX-1 FAIL:** band materially improves early wave → V-7 not closed; band cannot be defaulted off; halt band default track.
- **EX-2 FAIL:** any zero/low capture or unbounded cells → capture generality not established; halt all default changes; return to capture/seed design review (do not alter architecture unilaterally — escalate).
- **EX-3 FAIL:** decoupling gives no benefit or degrades wave → keep coupled behavior; do not default decoupling.
- **EX-4 FAIL:** no acceptable N within budget → auto-seed default not adopted; keep manual.
- **EX-5 FAIL:** contract spurious → drop the hard contract; keep informational warning only.
- **EX-7 FAIL:** obstacle interaction → no obstacle-case default change; flag follow-up.
- **EX-8 FAIL:** poor late-time/maxRef → maxRef default unchanged.
- **B1 FAIL:** results swing by indicator with no defensible choice → hold all efficiency defaults; escalate indicator decision.
- **B2 FAIL:** refinement persists behind the wave → efficiency premise unproven; block band-OFF/auto-seed efficiency defaults; escalate to runtime-AMR work.
- **B3 FAIL:** capture clips/leaks at boundary or reflection mishandled → block defaults for wall/surface cases.
- **B4 FAIL:** band-OFF deviates from physical reference beyond tolerance → correctness concern overrides self-consistency; block defaults.
- **B5 FAIL:** silent under-capture or unbounded cells at the extreme → strengthen warnings; no auto-seed default for the regime.
- Any FAIL (Track A **or** Track B) halts the dependent downstream milestones per §5 graph and blocks the Pre-Default gate.

---

## 12. Rollback Criteria Per Milestone

All experiment work is in temporary cases / flagged prototypes, so rollback = disable the flag / discard the temp cases (no production impact). For production phases:
- **Phase B (prototypes):** rollback = remove/disable feature flag; defaults already unchanged, so no behavioral rollback needed.
- **Phase E (defaults):** rollback = revert the specific default flip (band ON, auto-seed off) if post-rollout validation regresses; staged per-default so each is independently revertible.

---

## 13. Regression Risks Per Milestone

| Milestone | Regression risk | Mitigation |
|---|---|---|
| EX-2/Phase B | Prototype auto-seed alters generated dicts for existing cases | Flag-gated; golden-dict regression; defaults off |
| EX-3 | Decoupling changes seed region for existing cases | Flag-gated; compare against coupled baseline |
| EX-5/contract | Warning misfires on legitimate configs | Warning-only first; no hard block in prototype |
| Phase E band→OFF | Existing saved cases relied on band for capture | M1 fix makes capture band-independent; verify on saved-case sample before flip |
| Phase E auto-seed ON | Surprises users expecting level 0 | Staged rollout + metadata transparency + release note |

---

## 14. Testing Strategy

- **Unit tests:** maintain/extend existing capture-formula tests (already present for the M1 fix). Add unit tests for the prototype auto-seed calc and clamp (pure functions, no solver).
- **Generation-only regression:** golden comparison of generated dictionaries for a fixed case set; any diff must be intentional and flag-gated.
- **Short-solver smoke:** the M2 band-OFF reference case as a recurring smoke check (arrival/peak/cells must reproduce).
- **Experiment harness:** reuse the temporary `_val_*` generators/collectors; keep them out of production paths.
- **No production default is exercised as "default" in tests until Phase D passes.**

---

## 15. Branching Strategy

- **main:** production; defaults unchanged until Phase E gates pass.
- **feature flags:** all prototypes (auto-seed, decoupling, margin, contract-warning, surface band) land behind off-by-default flags so they can merge without changing behavior.
- **per-milestone branches:** one branch per prototype/experiment; merged to a long-lived integration branch for validation; promoted to main only when its gate passes.
- **No default-flip commit may merge to main before its Phase D Go decision.** (Git config is never modified; no force-push to protected branches.)

---

## 16. Production Rollout Strategy

Staged, one default at a time, each independently revertible, gated by the relevant Track A (V-items) **and** Track B (RG-items). No efficiency-motivated default flip occurs until **both tracks** clear the Pre-Default gate.

1. **Additive-safe first (no gate):** metadata + warnings (mass ratio, L_seed/clamp, backup ratio, projected/actual cells, band state, seed-vs-maxRef). No mesh behavior change.
2. **Auto-seed available (opt-in, gate EX-2/EX-4):** ship the auto-seed calc as opt-in; default still level 0.
3. **Decoupling available (opt-in, gate EX-3):** ship decoupled seed/backup as opt-in.
4. **Flip band → OFF default (Track A gate EX-1/V-7, EX-2/V-8 AND Track B gate B2 release/RG-2, B1 indicator/RG-3, B3 near-wall/RG-4, B4 anchor/RG-5):** only after matched comparison, capture generality, demonstrated release efficiency, indicator decision, surface-burst robustness, and physical anchor all pass. Band remains available advanced-only.
5. **Flip auto-seed ON default (gate EX-2/EX-4/V-11/V-13/V-18 AND B5/RG-6):** with validated N/L_max, thresholds, and extreme-coarse degradation confirmed.
6. **maxRefinement default guidance (gate B2/EX-8 tail, V-15/V-16):** adjust only if release-efficiency (B2) and late-time (EX-8 tail) validation support it; otherwise leave unchanged.

---

## 17. Post-Rollout Validation Strategy

- Run the generation-only matrix + short-solver smoke as a post-flip regression on a sample of representative saved cases.
- Monitor metadata warnings in generated cases (capture ratio, cell budget, clamp frequency) as field telemetry.
- Keep the M2 reference smoke check green.
- Hold a rollback trigger: if post-flip capture ratio or early-wave smoke regresses, revert that single default.

---

## 18. Recommended Execution Order (Two Parallel Tracks)

**Phase A (both tracks):** stand up harness + metrics + specs (now including runtime metrics: release ratio, persistence, indicator settings, reference/convergence definition).

**Phase C — run Track A and Track B concurrently:**

| Step | Track A (startup) | Track B (runtime) |
|---|---|---|
| 1 | EX-2 capture matrix ∥ EX-1 matched band (P1) | B1 indicator comparison (start early — gates the others' baseline) |
| 2 | EX-6 mechanism (cheap, informational) | B2 release efficiency (on B1-selected indicator) |
| 3 | EX-3 decoupling+margin → EX-4 N-sweep → EX-5 contract | B3 near-wall / surface burst |
| 4 | EX-7 obstacle independence | B4 physical / convergence anchor; B5 extreme-coarse (cheap, can run anytime) |

**JOIN → Pre-Default Validation Gate (Phase D):** requires Track A gates **and** Track B gates (see §19).

**Phase E:** staged rollout (band OFF, then auto-seed ON), each independently revertible.

**Pre-Production Gate → late/advanced:** EX-8 long-duration tail (maxRef default decision) and EX-9 surface band (advanced-only, never a default).

Sequencing notes: **B1 should start early** because it sets the indicator baseline used by B2/EX-8; **B5** is generation-only and can run any time; **B4** (convergence series) is the most expensive Track B item and should begin once B1's indicator is chosen.

---

## 19. Validation Gates & Go/No-Go Criteria

Two explicit gates now govern progression. **Both tracks must clear the Pre-Default gate** before any production-default change.

### Pre-Default Validation Gate (Phase D entry → required before ANY efficiency-motivated default flip)

**Track A (startup) — ALL of:**
- EX-1 PASS (V-7): band-OFF early wave matches band-ON at lower cost (with the isolated band arm).
- EX-2 PASS (V-2 5 kg, V-8 capture, V-14): capture ≥0.98 across the full envelope, no zero-capture, bounded cells.
- EX-3 PASS (V-9, V-10): decoupling reduces cells without early-wave loss; margin chosen.
- (auto-seed default) EX-4 PASS (V-11, V-13) and V-18 clamp/warn demonstrated.
- (contract enforcement) EX-5 resolved (V-12).
- (obstacle-case defaults) EX-7 PASS (V-17).

**Track B (runtime) — ALL of:**
- B1 resolved (RG-3): a defensible default indicator chosen; Track A conclusions reproduce under it.
- B2 PASS (RG-2; informs RG-1, V-15): demonstrated release behind the wave (final ≪ peak) at the chosen indicator.
- B3 PASS (RG-4): robust capture + correct reflection for near-wall / surface-burst cases.
- B4 PASS (RG-5): band-OFF matches a physical reference and/or converges under mesh refinement.
- B5 resolved (RG-6; V-18, R-2): extreme-coarse capture robust with correct clamp/warn (no silent failure/explosion).

**Cross-cutting (both tracks):** additive metadata/warnings shipped and regression-clean.

**No-Go:** if **any** Track A *or* Track B gate above fails, **no production default is changed.** The failing item is held and escalated; the architecture is not redesigned in response (per §5-bis).

### Pre-Production Validation Gate (before rollout is declared complete)
- Post-rollout regression on representative saved cases green (capture ratio, early-wave smoke).
- EX-8 long-duration tail acceptable for the maxRefinement default decision (V-15/V-16); **maxRefinement default changes only on this PASS.**
- **Surface band (EX-9) never becomes a default** regardless of outcome (advanced-only).

### Default-by-default summary
| Default change | Track A gate | Track B gate |
|---|---|---|
| Metadata/warnings (additive) | none | none |
| Auto-seed opt-in | EX-2/EX-4 | — |
| Decoupling opt-in | EX-3 | — |
| **Band → OFF default** | EX-1, EX-2 | B1, B2, B3, B4 |
| **Auto-seed → ON default** | EX-2, EX-4, V-18 | B5 |
| maxRefinement default | — | B2 + EX-8 tail |

---

## Final Output

1. **Recommended next action:** execute **Phase A** for both tracks (stand up the harness from the existing `_val_*` temp generators; lock specs/metrics including runtime metrics — release ratio, refinement persistence, indicator settings, and the physical-reference/convergence definition), then launch Track A **EX-2 ∥ EX-1** together with Track B **B1** (indicator comparison) and **B5** (cheap extreme-coarse). No production code or defaults change.
2. **Highest-value validation:** **EX-1** (Track A) and **B2 release efficiency** (Track B) — EX-1 closes the band-vs-no-band early-formation gate (V-7); B2 closes the core *efficiency* premise (RG-2) that justifies retiring the band. **B4** (physical anchor) is the highest-value *correctness* insurance.
3. **Highest-risk assumption:** jointly **V-8** (single-config early-wave generalization) and **RG-2/RG-3** (that release efficiency holds and is not indicator-locked). If release efficiency fails or is indicator-dependent, the efficiency rationale for the architecture's defaults is unproven even if startup validation passes.
4. **Earliest point production defaults may change:** only after the **Pre-Default Validation Gate** clears — i.e. Track A (EX-1+EX-2[+EX-3]) **and** Track B (B1+B2+B3+B4) for the band-OFF flip; auto-seed adds EX-4/V-18 and B5. Additive metadata/warnings may ship immediately (no gate).
5. **Overall readiness assessment:** **Correctness-ready (startup), efficiency- and runtime-unvalidated.** The M1 capture foundation is production-grade; the cost/efficiency architecture and its runtime-AMR interaction are pre-validation across **both** tracks and must not drive default changes until the Pre-Default gate clears. Risk is well-contained: all prototypes are flag-gated, all experiments are temporary-case-only, and runtime risks (release, indicator, walls, physical truth, coarse mesh) are now explicit Pre-Default gates rather than late discoveries.

---

*This roadmap is execution planning only. It does not redesign or modify the architecture, contains no code or implementation detail, and proposes no production-default change ahead of its validation gate.*
