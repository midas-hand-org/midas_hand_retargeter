# Changelog

## 0.2.0 (unreleased)

**Milestone: this retargeting drove the physical MIDAS hand from a Manus
glove.** `mode="dexpilot"` with a tuned preset, on hardware, end to end.

### Added — DexPilot mode

- **`mode="dexpilot"`**, and it is now the recommended mode for glove teleop.
  It optimises six pairwise inter-fingertip vectors plus four palm-rooted ones,
  so unlike every other mode it controls where the fingertips sit *relative to
  each other* — the thing the analytic map structurally cannot do, being blind
  to absolute hand geometry. Measured mean inter-fingertip error 8.3 mm against
  the analytic map's 24 mm.
- **`DexPilotParams`**, 12 live-tunable solver knobs, all applied per solve with
  no rebuild. Notable ones:
  - `scaling_factor` — the load-bearing knob in this mode; a 0.7×–1.5× change
    moves joints by ~1.5 rad. `calibrate_scaling_from_landmarks()` measures it
    from a held open pose instead of making the operator guess.
  - `abduction_limit` (default 0.25 rad) — bounds finger abduction. A human's
    fingertips converge as they curl; the MIDAS fingers curl in parallel planes
    and can only imitate that by abducting, and it is nearly free for the
    solver to do so (locking abduction costs 0.5 mm of inter-fingertip
    accuracy). Unbounded, that produced 0.77 rad of sideways swing on a plain
    curl. Bounded: 0.35 rad, for +0.3 mm.
  - `spread_scale`, `thumb_vector_scale`, `smoothing_alpha`.
- **Palm-frame input** for `dexpilot` (`palm_frame_input`), so how the operator
  holds their wrist is not read as finger articulation. Removes rotation
  (max |Δ| 7.5e-9 under a 0.7 rad rotation) but *not* reflection (0.155 rad),
  so input chirality still matters.
- **Thumb-root rebase** (`thumb_root_link="thumb_cmc_side"`). The MIDAS thumb
  reaches 180 mm from the palm against an operator's ~121 mm, because two
  segments have no human counterpart: a 41.4 mm palm→CMC-roll offset and a
  24.7 mm CMC mechanism. Comparing against the CMC instead of the palm drops
  both, taking the thumb/index proportion from 1.43 to 1.19 against a human's
  1.17. Thumb bend 0.71 → 0.37 rad and jitter 0.416 → 0.163 rad.
- **Flexion-only thumb bounds** (`thumb_flexion_only`), a deliberate teleop
  policy: the physical joint can hyperextend, and the solver used that DOF to
  produce anatomically impossible S-curves on 42% of frames. Now 14%.
- `calibrate_scaling_from_landmarks` (hand size) and
  `calibrate_neutral_from_last_frame` (zero pose), plus `presets` save/load
  that carries the zero pose so a tuning session is reproducible.
- `presets.resolve`, so a preset can be addressed by name from a CLI as well as
  from the browser.

### Fixed

- **Glove input was mirrored.** The analytic map is provably reflection-
  invariant, so this was invisible until `dexpilot` — which is handed — was
  asked to bend the fingers backwards and held them extended 87% of the time,
  with the ring finger tracking *inverted* (correlation −0.10). Restoring a
  reflection took that to +0.83 and mean inter-fingertip error from 17.1 mm to
  9.6 mm.
- **The thumb tip frame pointed backwards** into the palm: 28 mm *closer* to
  the palm than the thumb DIP, cos −0.84 against the distal direction, now
  +0.96. Every thumb reach measurement taken before this was wrong, including
  the one that said the thumb was too short when it is proportionally long.
- **The Cartesian objective ignored the PIP-DIP four-bar coupling**, so it
  aimed at a fingertip up to 59 mm from where the linkage puts it. Those modes
  now default to `coupling_mode="pip_dip_lookup"`.
- **The first solved frame latched forever** in the optimizer-only modes:
  `_hold_disabled_joints` held every active joint absent from the analytic
  targets, and those modes produce none.
- **Neutral calibration could undo the thumb bounds**, rescaling by the model's
  +1.57 and commanding up to +0.27 rad of the hyperextension the solver had
  been forbidden to produce.
- Saving a preset silently reset every solver knob to its default.
- The four-bar coupling agreeing with `midas_hand_api` is now pinned by a test.
  It always did agree; the comparison had been made against the API's
  lookup-space function rather than its motor-space one, and was recorded here
  as a conflict to settle on hardware.

### Removed

- 15 dead module constants in `postprocess` left behind by the per-finger
  params migration, with no reader anywhere in any of the three repos.

### Changed

- **The analytic geometric map is now the default retargeting method**
  (`mode="analytic"`). Previously the `dex_retargeting` vector optimizer ran
  every frame and its result was then overwritten for all 13 actuated joints,
  so it contributed nothing to the command while costing ~80% of the frame
  budget. Output is bit-identical across the golden pose set; measured
  1.2 ms/frame versus 7.9 ms for the old path.
- **`dex_retargeting` and `torch` moved to the `[vector]` extra**, and the
  previously undeclared dependency on `midas-hand` (the four-bar lookup) became
  the `[lookup]` extra. A default install is numpy only, ~3.9 GB smaller, and —
  unlike 0.1.0 — can actually retarget a frame without a sibling repo on disk.
- **`RetargeterTuning`'s 11 global fields are superseded by per-finger
  `RetargetProfile`.** The old class still works and converts exactly; it will
  be removed in a future release.
- Version scheme: this is a breaking API change, hence 0.2.0.

### Added

- `mode` on `MidasRetargeterConfig`: `analytic` / `dexpilot` / `vector` /
  `refine`.
- `RetargetProfile`, `FingerParams`, `ThumbParams`: 50 per-digit parameters,
  including the output ranges, splay deadzone/limit/curl-damping, curl blend
  weight and the thumb's neutral angle and span — all previously hardcoded.
- `ProfileStore` for atomic, undoable live parameter edits.
- `presets.save/load`: JSON profiles that carry the neutral calibration
  (previously in-memory only and lost on exit) and are validated against the
  robot's joint limits.
- `HandModel`: solver-free joint names, limits and index lookup, CI-checked
  against the URDF, the MJCF and pinocchio's dof ordering.
- `analytic_debug()` and `palm_basis()` for tuning UIs and diagnostics.
- `tests/goldens/`: a 45-pose behaviour freeze for the analytic layer.

### Fixed

- The solver's warm start tracked its own discarded solution rather than the
  commanded pose, drifting up to 1.5 rad over a closing sweep.
- `analytic` mode with the postprocess flags disabled now raises. Previously
  that combination would have left those joints unwritten, commanding 0.0 rad
  — fully extended, a real motion on hardware.
- `coupling_mode` validation was duplicated and inconsistent: `"Fixed_Passive"`
  was rejected by `config.py` and accepted by `adaptor.py`.
- Unknown joint targets warn instead of being silently dropped.
- `with_tip_links` leaked a `/tmp` directory per process.
- `reset()` did not reset the upstream low-pass filter or clear the
  uncalibrated cache, and left the warm start at mid-range.
- Importing `midas_hand_retargeter.adaptor` no longer requires torch, which
  protects the six downstream call sites that import only its mode constants.

### Removed

- `configs/midas_hand_right_vector.yml` and the `pyyaml` dependency. The file
  was packaged but never read, and had already drifted from the code defaults.
- The `try/except ImportError` shim that substituted a fake `KinematicAdaptor`,
  making a broken install look partly functional.

### Documentation

- Corrected the claim in `human.py` that a mirrored input frame inverts thumb
  opposition and finger splay. It does not: the analytic map is
  reflection-invariant (verified max |Δ| = 0.0 across all 13 targets), so
  mirroring glove input to "fix" tracking achieves nothing. The vector
  optimizer remains handed, and the correction is scoped to say so.

## 0.1.0

Initial release.
