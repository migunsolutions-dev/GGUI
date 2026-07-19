# Architecture Decision Log — 3D Startup Meshing

Permanent record of major architecture/validation decisions for the 3D startup-meshing program. Each entry is append-only; supersession is noted explicitly.

**Conventions:** Status ∈ {Proposed, Accepted, Accepted (not yet default), Implemented, Open}. "Accepted (not yet default)" means the design is approved but production defaults have **not** been flipped.

---

## AD-001 — `chargeRefineOuter` remains available as an Advanced option
- **Decision:** Keep `chargeRefineOuter` (the volumetric near-field refinement band) in the product as an **advanced, opt-in** tool; do not remove it.
- **Status:** Accepted.
- **Rationale:** The band still has legitimate advanced uses; removing it entirely would reduce flexibility. The design goal is to retire it as a *default*, not delete it.
- **Evidence:** `docs/3D_Startup_Meshing_Architecture.md` §1; band redundancy for the default path shown in EX-1/EX-1B (P8/P10).
- **Open issues:** Needs explicit band×seed guardrails (see AD-009) to prevent the explosion mode when advanced users combine band + deep seed.

## AD-002 — Charge capture MUST NOT depend on `chargeRefineOuter`
- **Decision:** Charge capture must be guaranteed by a mass-conserving backup region, independent of any refinement band.
- **Status:** Accepted; **Implemented for the seed > 0 path only** (Milestone 1).
- **Rationale:** A correctness guarantee (mass capture) must not rest on a refinement convenience. Historically the band masked a capture defect (RC-2).
- **Evidence:** M1 fix (`sphericalMassToCell`/`cylindericalMassToCell` + floored `backup` radius via `CAPTURE_CELL_SAFETY`, `generator_3d.py` ~2149–2222); EX-2 matrix 0 zero-capture; EX-1B new-default mass ratio 1.0006.
- **Open issues:** The floor is **not** applied on the seed-0 `setFields` path (`generator_3d.py:2121` gates on `charge_refine > 0`). This violates the spirit of AD-002 for the current default. Closed only by AD-007 Option A or B.

## AD-003 — Lean-startup philosophy retained
- **Decision:** Prefer a lean startup mesh (coarse far field, tight near-field seed, smooth transitions) over a heavy volumetric startup band; let runtime AMR build resolution.
- **Status:** Accepted.
- **Rationale:** Aligns with official blastFoam `building3D`/`freeField` philosophy; avoids startup over-refinement and the band×seed explosion.
- **Evidence:** `docs/3D_Startup_Meshing_Architecture.md` §1, §5; EX-1 band overhead +27 %; EX-1B band overhead 3.4×.
- **Open issues:** None for startup; see AD-004 regarding the *reason* lean startup is efficient.

## AD-004 — Runtime release is NOT the primary efficiency mechanism
- **Decision:** Do not justify the architecture on the basis that runtime AMR releases (coarsens) cells behind the blast wave. Efficiency comes from leaner startup, bounded `maxRefinement`, and domain/time sizing.
- **Status:** Accepted (narrative correction).
- **Rationale:** Spatial analysis disproved release within the practical propagation window; the refined region is a solid expanding product-gas ball.
- **Evidence:** Release Spatial Analysis (P5–P7); D1; risk R1/R2 in `docs/LIVING_VALIDATION_ASSESSMENT.md`.
- **Open issues:** The architecture document's efficiency rationale should be edited to reflect this (tracked as R2; not yet edited).

## AD-005 — Current default configuration is `seed = 0 + band ON`
- **Decision:** Record the *actual* shipping default as the baseline for any transition: `charge_refinement_level = 0`, `chargeRefineOuter` enabled (band levels 2/3), `charge_capture_mode = auto`; auto-seed computed for recommendation only.
- **Status:** Accepted (statement of fact).
- **Rationale:** Any default change must be measured against the true current default, not the aspirational architecture.
- **Evidence:** `models.py:109`; `tab_3d_general.py:388-402, 2951-2952`; `startup_mesh_metadata.recommended_auto_seed_level` docstring; verified in seed-policy check (P12).
- **Open issues:** This default relies on the band as the de-facto charge-refinement mechanism (RC-2); see AD-006.

## AD-006 — `band OFF + seed = 0` is UNSAFE for sub-cell charges
- **Decision:** Treat the `band OFF + seed 0` configuration as unsafe; it must never become a reachable default without capture protection.
- **Status:** Accepted (hard constraint).
- **Rationale:** With neither seed nor band, `setFields` uses the bare physical charge radius; on a coarse mesh no cell-centre lies inside it → zero capture → blastFoam FATAL "No mass".
- **Evidence:** EX-1 `seed0_bandOFF` FATAL no-mass (P9); analytic gap 12/28 sub-cell combos fail (P12).
- **Open issues:** Directly motivates the coordinated transition (AD-007).

## AD-007 — Band retirement requires coordinated capture protection
- **Decision:** Retiring the band as a default must be shipped **together** with capture protection for the seed-0 path. Two viable options:
  - **Option A:** Wire auto-seed as the default `charge_refinement_level` (≥ 1 for sub-cell charges).
  - **Option B:** Extend the M1 mass-conserving floor to the seed-0 `setFields` path.
- **Status:** Proposed (no implementation; see `docs/DEFAULT_TRANSITION_PLAN.md`).
- **Rationale:** Prevents AD-006's failure when the band default is flipped off.
- **Evidence:** Analytic: auto-seed-as-default closes the gap 12/28 → 0/28 (P12); EX-1B new default captures 1.0006.
- **Open issues:** Requires approval; final correctness still gated on B4 (U2).

## AD-008 — Startup meshing and runtime AMR are independent systems
- **Decision:** Treat startup meshing and runtime AMR as separate systems with a single defined hand-off; do not use startup mechanisms (band, high seed) to chase runtime goals or vice-versa.
- **Status:** Accepted.
- **Rationale:** Conflation (RC-4) produced over-refinement and fragile defaults.
- **Evidence:** `docs/3D_Startup_Meshing_Architecture.md` §4 (RC-4), hand-off contract.
- **Open issues:** None.

## AD-009 — Guard against the band × deep-seed explosion
- **Decision:** When the band is used as an advanced option, guard against combining it with deep seeding, which multiplies startup cells.
- **Status:** Accepted (guardrail required for the advanced path).
- **Rationale:** `refineInternal` grading/buffer cascades through a band-pre-refined shell, inflating the seeded volume (RC-3).
- **Evidence:** Milestone 2 (band+deep-seed intractable); EX-1B band+auto-seed-4 = 3.4× the band-OFF startup mesh.
- **Open issues:** Guardrail behavior for advanced band use is specified in the architecture but should be verified if/when the advanced path is exercised.

## AD-010 — `densityGradient` is an acceptable default AMR indicator
- **Decision:** Retain `densityGradient` as the default runtime AMR indicator.
- **Status:** Accepted.
- **Rationale:** Early metrics are only weakly indicator-sensitive; interior persistence is indicator-independent.
- **Evidence:** B1 (~0.4 % spread across indicators, P3); release spatial `densityGradient` ≈ `scaledDelta_p` (P7).
- **Open issues:** None for the default; indicator choice is not a release lever (D2).

---

### Appendix — EX-2 and EX-1 finding capsules (for quick reference)

- **EX-2 findings:** Band-OFF capture is robust across sphere/cylinder/box × {5, 25} kg × {0.2, 0.5} m (0 zero-capture). Two marginal mass ratios were resolved by raising the seed (EX-2 follow-up), confirming seed level — not a capture defect — governs coarse-mesh capture accuracy. Supports AD-002, AD-003.
- **EX-1 findings:** With everything else fixed, toggling the band has no measurable effect on the early wave (identical arrival/peak/converged mesh; P8/P10) and only adds startup cells. The band was, however, *masking* a seed-0 zero-capture failure (P9). The new default (`band OFF + auto-seed`) captures sub-cell charges more accurately than the legacy default (1.0006 vs 0.934; P11). Supports AD-001, AD-005, AD-006, AD-007.
