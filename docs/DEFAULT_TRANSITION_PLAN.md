# Default Transition Plan — `chargeRefineOuter` Retirement

**Purpose:** Describe the path from the current shipping defaults to the proposed future defaults for 3D startup meshing. **Planning document only — does NOT recommend or authorize implementation.** No code or defaults are changed by this document.

**Related:** `docs/LIVING_VALIDATION_ASSESSMENT.md` (status/evidence), `docs/ARCHITECTURE_DECISIONS.md` (AD-005 … AD-007), `docs/3D_Startup_Meshing_Architecture.md` (approved design).

---

## 1. Current Default Configuration (baseline)

Verified in source (AD-005 / P12):

| Setting | Current default | Source |
|---|---|---|
| `charge_refinement_level` (seed) | **0** (no deep seed; `setFields` path) | `models.py:109` |
| `chargeRefineOuter` (band) | **ON** (enabled; levels 2/3) | `tab_3d_general.py:2952`, `:395`,`:400` |
| `charge_capture_mode` | `auto` | `models.py:111` |
| auto-seed | computed for **recommendation only**, not applied | `startup_mesh_metadata.recommended_auto_seed_level` |
| M1 mass-conserving floor | applied **only** when seed > 0 (`setRefinedFields`) | `generator_3d.py:2121` |

**Implication:** the band is currently *load-bearing for capture*. The near field is refined by the band, which lets `setFields` capture sub-cell charges; remove the band with seed still 0 and capture fails (AD-006, P9).

---

## 2. Proposed Default Configuration (target)

Per the approved architecture (`docs/3D_Startup_Meshing_Architecture.md`):

| Setting | Target default |
|---|---|
| `chargeRefineOuter` (band) | **OFF** by default (advanced opt-in only) |
| Near-field resolution | **auto-computed deep `refineInternal` seed** (seed ≥ 1 for sub-cell charges) |
| Charge capture | mass-conserving backup, **independent of band and independent of seed path** |

The transition is *not* a single toggle: flipping the band off must be paired with capture protection for the seed-0 path. Two options achieve this.

---

## 3. Option A — band OFF + auto-seed as the default

Make `charge_refinement_level` default to the auto-seed value (clamped, e.g. `clamp(ceil(log2(N·dx/d_min)), 0, L_max)`, N = 6, L_max = 5) instead of fixed 0, and set the band OFF by default.

**Advantages**
- Closes the capture gap completely: analytic check shows fixed seed-0 fails 12/28 sub-cell combos, auto-seed-as-default fails 0/28.
- Routes all sub-cell charges through the existing, validated M1 mass-conserving `setRefinedFields` path (better capture: 1.0006 vs legacy 0.934 in EX-1B).
- Smallest behavioural surface: auto-seed already exists and is validated; the change is "promote recommendation → default".
- Improves near-field resolution and consistency (P11), aligning the GUI with `building3D`/`freeField`.

**Disadvantages**
- Changes the early/near-field solution for sub-cell charges (not a no-op; D4) — users will see different (finer) near-field behavior than the legacy default.
- Slightly higher startup cost than fixed seed-0 (deep seed adds local cells), though far cheaper than band + seed.

**Risks**
- R3/U2: the new near-field solution's correctness is not yet anchored to an external reference (B4 pending).
- Auto-seed clamp (L_max) behavior on extreme charges should be confirmed not to over-refine startup.

**Validation completed**
- Capture gap closed analytically (P12); EX-1B confirms capture + early-wave at a deeply sub-cell charge; EX-1 confirms band redundancy.

**Validation still required**
- B4 external correctness anchor (U2).
- A spot check that auto-seed never over-refines startup at the L_max clamp for the largest practical charges.

---

## 4. Option B — band OFF + M1 backup protection for seed = 0

Extend the M1 mass-conserving capture (`sphericalMassToCell` / `cylindericalMassToCell` + floored `backup` radius) to the seed-0 `setFields` path, so capture is robust regardless of seed; keep the band OFF by default and leave the default seed at 0 (or auto).

**Advantages**
- Most principled fix: fully decouples capture correctness from refinement (satisfies AD-002 universally, not just seed > 0).
- Preserves the leanest possible default startup mesh (seed 0) for well-resolved charges while still protecting sub-cell charges.
- Defends against *any* future configuration reaching `band OFF + seed 0` (defence-in-depth), even if a user manually sets seed 0.

**Disadvantages**
- Larger code change in the capture path (`generator_3d._build_set_fields_dict_3d`): the `use_refined = charge_refine > 0` gate must be generalized; needs care to keep the non-refined `setFields` semantics correct.
- A seed-0 captured charge is still coarsely *resolved* (capture ≠ resolution); the early near-field wave may be under-resolved for sub-cell charges unless auto-seed is also used. I.e. Option B fixes *capture* but not *resolution*.

**Risks**
- Regression risk in the widely-used `setFields` path; requires capture regression tests across shapes/masses/meshes.
- Without auto-seed, sub-cell charges remain under-resolved near-field (the EX-1B legacy under-resolution issue, P11, would persist for seed-0 users).

**Validation completed**
- Root cause and mechanism fully characterized (P9/P12); the fix target (`generator_3d.py:2121` gate) is identified.

**Validation still required**
- A generation-only capture regression matrix on the modified seed-0 path (mirroring EX-2).
- B4 external correctness anchor (U2).

---

## 5. Option comparison (summary)

| Dimension | Option A (auto-seed default) | Option B (seed-0 floor) |
|---|---|---|
| Fixes capture gap | Yes (12/28 → 0/28) | Yes |
| Fixes near-field *resolution* for sub-cell | Yes | No (capture only) |
| Code surface | Small (promote existing auto-seed) | Larger (generalize capture gate) |
| Regression risk | Low | Medium (touches `setFields` path) |
| Defence-in-depth vs manual seed-0 | Partial | Full |
| Aligns with architecture intent | Directly | Partially (needs auto-seed too) |

These options are **not mutually exclusive**; A + B together give both correct capture and correct resolution with defence-in-depth. No recommendation is made here.

---

## 6. Pre-conditions before ANY default flip (gating checklist)

1. **B4 correctness anchor passed** (U2/R3) — band-OFF + auto-seed reproduces an external free-air blast reference beyond the contact surface.
2. **Coordinated capture protection chosen and implemented** (Option A and/or B) with a generation-only capture regression matrix (EX-2 style) at 0 zero-capture.
3. **Architecture efficiency narrative corrected** (R2/AD-004).
4. **Advanced band × seed guardrail verified** (AD-009) so the retained advanced band cannot silently explode the mesh.
5. **Explicit owner approval** to change production defaults.

Until all five hold, defaults remain `seed = 0 + band ON` (AD-005), and `band OFF + seed 0` must remain unreachable as a default (AD-006).
