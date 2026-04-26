# 3D Initialize Validation Matrix

This checklist validates transparent initialize behavior in the 3D tab.

## How to use

1. Open `General 3D` tab.
2. Configure the scenario inputs below.
3. Click `Initialize Model (Step 0)`.
4. Verify:
   - Preflight Summary appears.
   - Effective plan confirmation appears (standard mode).
   - Initialize Summary appears after success.
   - `initialize_summary.txt` is appended in the generated case directory.

## Scenario A: Small charge (still allowed)

- Configure a very small charge relative to base cell size.
- Expected:
  - Preflight opens with info and plan.
  - Initialize can continue after user confirmation.
  - If capture is impossible in generated mode, initialize is blocked with a clear error.

## Scenario B: Charge clipped

- Move charge center so geometry crosses domain boundary.
- Expected:
  - Preflight still appears.
  - If generated mode determines capture is impossible, initialize is blocked with a clear error.

## Scenario C: Remap missing time/path

- Enable remap mode.
- Leave source path empty OR set specific time with empty value.
- Expected:
  - Initialize blocked with explicit remap message.
  - No hidden fallback to default remap time.

## Pass criteria

- No silent value changes.
- Blocking conditions show explicit error messages.
- Requested vs Effective confirmation reflects generated mode values.
- Post-init summary is visible and persisted to `initialize_summary.txt`.
