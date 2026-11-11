<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Calibrating `kLeftHandOffset` / `kRightHandOffset`

These two constants in [`../core/manus_hand_tracking_plugin.cpp`](../core/manus_hand_tracking_plugin.cpp)
place the MANUS hand skeleton relative to the controller's **aim** pose. They are
used only on the controller-fallback path (`get_controller_wrist_pose`), so the
headset's own optical hand tracking must be **off** or the runtime will prefer it
and your edits will have no effect at all.

Calibration is: hold a pose whose true geometry is known, record what the plugin
reports, and solve for the offset that reconciles the two.

## Quick start

```bash
cd <repo root>
source ~/.cloudxr/run/cloudxr.env          # sets NV_CXR_RUNTIME_DIR
./install/plugins/manus/manus_hand_plugin &    # MANUS Core must be CLOSED

CAL=src/plugins/manus/calibration
.venv/bin/python $CAL/cal_capture.py --step palms    --no-launch-cloudxr-runtime
.venv/bin/python $CAL/cal_capture.py --step point    --no-launch-cloudxr-runtime
.venv/bin/python $CAL/cal_capture.py --step tipsweep --no-launch-cloudxr-runtime

.venv/bin/python $CAL/cal_solve.py --steps palms,point,tipsweep        # inspect
.venv/bin/python $CAL/cal_solve.py --steps palms,point,tipsweep --apply  # write it

cmake --build build --target manus_hand_plugin --parallel
cmake --install build --component manus
```

Use the repo venv's `python` directly. `uv run --active` only works when
`VIRTUAL_ENV` already points at `.venv`, and otherwise silently builds an
environment with no `isaaccapture` in it.

Each capture is ~10 s and **appends**, so re-run a step to add more data. On a
flaky link that means a dropout costs one short run, not the session.

## The steps, and what each one uniquely measures

They are not interchangeable — each closes a gap the others cannot see.

| step | measures | notes |
|---|---|---|
| `palms` | **Translation**, model-free. The wrist joint's MANUS-local position is ~0, so the wrist gap depends only on the offset and the aim pose — no hand model involved. Palm-plane parallelism pins 2 of 3 rotation axes. | **Required.** |
| `point` | The **third rotation axis** — spin *within* the palm plane. Palms-together is blind to it: press your palms flat and rotate the pair, and they stay flat the whole way. | Required. Keep the two fingers *straight and collinear*; direction is measured knuckle→tip, so a bent finger reads as splay. |
| `tipsweep` | Hand **configuration** diversity. Metagloves track fingertips magnetically and directly, so tips are a primary measurement. Sweeping the fingers varies the local joint geometry in a way rotating the hands cannot. | Sharpens precision; carries little independent signal on its own. |
| `pinkies` / `thumbs` | The **lateral** (thumb↔pinky) axis, which `palms` constrains most weakly. | Only if the mount lets you keep the hands **untilted**. A tilted capture is worse than none — it fights the good data. |

**During every capture, rotate the pair** through as many orientations as you can
while holding the contact. A wrong offset produces an error fixed in the hand's
frame; comparing it across orientations is what separates a translation error
from a rotation error. Sliding the hands around *without* rotating adds nothing —
the geometry between them is unchanged. Aim for >100° of total spread.

## Reading the solver output

`cal_solve.py` prints, per step, current → fitted for every constraint, plus:

- **precision per axis** — if an axis has a large standard error, the steps you
  supplied do not pin it down and the fitted value for it is not trustworthy.
- **per-finger agreement** (when `tipsweep` is present) — each finger solves for
  the offset alone. If they scatter far beyond their individual precision, the
  hand model is internally inconsistent and no single rigid offset can satisfy
  them. This is the check that exposed a miscalibrated glove.

Useful flags: `--palm-bump <cm>` raises the palms contact targets (see
*Contact targets* below), `--reg` regularises rotation toward the current value,
`--steps` selects which captures to use.

## Gotchas that have actually bitten

- **MANUS Core steals the dongle.** Both it and the plugin use Core Integrated,
  and only one can hold the dongle. Symptom: `Error claiming interface 0:
  Resource busy`. Close MANUS Core, then restart the plugin.
- **A dropped glove leaves the plugin publishing a frozen pose.** It keeps
  emitting joints at the last known values, so captures look fine and are
  worthless. Tell-tale: contact gaps with `sd = 0.00`, identical values across
  steps. `cal_capture` shows a live contact number — **if it does not move as you
  move, stop.**
- **Capture-time offsets.** Joint positions only mean something relative to the
  offset that produced them. Every `.npz` now stamps the offsets compiled in at
  record time and the solver reads them per file, so captures from different
  builds mix safely and nothing needs rebasing. Appending to a file recorded
  under different offsets is refused rather than silently mixed.
- **Recalibrating the gloves invalidates every existing capture** — the hand
  model changed. Delete the `step_*.npz` files and start over.
- **Weighting.** Terms are normalised by `sqrt(#residuals)` so a per-frame term
  cannot outvote a per-pose one. Without this the palm-plane constraint (one
  scalar against thousands of gap residuals) simply collapses.

## Contact targets, and what the solver cannot know

The solver matches each pose's measured gaps against assumed separations in
`TARGET` (cal_solve.py) — how far apart two joint *centres* sit when the surfaces
touch. The gloves report centres, not surfaces, so that depth is an assumption,
not a measurement. If the hands visibly clip when the palms meet, raise
`--palm-bump`; if they float apart, lower it. **This one number is legitimately
calibrated by eye**, unlike rotation and alignment which come from pose geometry.

## Known limitation: the thumb

The thumb has been the persistent outlier across every round — the largest
per-finger disagreement, and the landmark an unconstrained solver discards. It is
misplaced *relative to the rest of the hand*, and a rigid wrist offset moves the
whole hand together, so **the offset cannot correct it**. Measured trade-off:
clearing the thumb to 2.5 cm forces the knuckles to ~4.2–5.1 cm, which reads as
the palms floating apart. Rotation is no better a lever — about 0.06 cm of thumb
clearance per degree, while widening the knuckles at the same time.

If the thumb matters more than it currently does, the fix is upstream in MANUS
Core, where the thumb's position relative to the hand can actually be changed.
A plausible contributing factor on this hardware: Metagloves track fingertips
magnetically, and the controllers bolted to the backs of both gloves sit close to
that field. Worth testing with the controllers temporarily unmounted.

## Unrelated bug worth knowing

`XR_HAND_JOINT_PALM_EXT` (joint 0) and `XR_HAND_JOINT_LITTLE_TIP_EXT` (joint 25)
both read MANUS node 24, so the palm joint is a duplicate of the little fingertip
rather than a palm. See the `manus_index` mapping in `inject_hand_data`. These
tools skip joint 0 for that reason. Fixing it is independent of calibration.

## Current state (2026-11-04, revB gloves + Quest controllers)

```cpp
kLeftHandOffset  = { { -0.705f, -0.555f, -0.130f, 0.423f }, { -0.094f, 0.001f, 0.012f } };
kRightHandOffset = { { -0.705f,  0.555f,  0.130f, 0.423f }, {  0.094f, 0.001f, 0.012f } };
```

Reached from `--steps palms,point,tipsweep`, then widened for visual clearance
and nudged 3° — so it does **not** correspond to a plain solver run at default
targets. Re-running `cal_solve` with no `--palm-bump` will pull the hands ~1.4 cm
closer than this and clip; start from `--palm-bump 1.5` to land near here.

Achieved: palm plane 2.9°, knuckles ~4.1 cm, worst non-thumb joint ~3.0 cm,
middle-finger pointing ~165° (ideal 180°), thumb ~2.0 cm (still marginal — see
*Known limitation* above). The captures behind these numbers are the
`step_*.npz` files in this directory, each stamped with its capture-time offsets.
