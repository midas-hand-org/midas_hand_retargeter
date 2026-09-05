# Changelog

## 0.2.0 (unreleased)

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

- `mode` on `MidasRetargeterConfig`: `analytic` / `vector` / `refine`.
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
