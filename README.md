# MIDAS Hand Retargeter

Thin MIDAS-specific wrappers around the installed `dex_retargeting` package.
This repo owns only MIDAS retargeting configuration and output adapters; it
does not duplicate the upstream optimizer, the MuJoCo model, the MediaPipe
webcam pipeline, or the hardware API.

Current mode:

- vector-based retargeting
- active MIDAS joints only
- default option 1: passive finger DIP/linkage joints fixed inside the
  retargeting FK model; downstream coupling handled by `midas_hand_mujoco`
  closed-loop MJCF or `midas_hand_api` PIP-DIP lookup
- optional option 2: `coupling_mode="pip_dip_lookup"` fills passive
  DIP/linkage joints inside retargeting FK from the `midas_hand_api`
  four-bar lookup table

Code map:

- `config.py`: dex-retargeting vector optimizer configuration
- `retargeter.py`: high-level API and output adapters
- `adaptor.py`: option 1/option 2 passive PIP-DIP coupling adaptor
- `postprocess.py`: MIDAS-specific landmark-to-joint correction layer
- `tuning.py`: user-facing knobs for teleop performance
- `human.py`: MediaPipe/MANO landmark frame and vector utilities
- `urdf.py`: temporary retargeting-only fingertip frame generation

Install for development:

```bash
pip install -e .
```

`dex_retargeting` imports Torch, so this package declares `torch` explicitly.
CPU Torch is enough for this pipeline.

Run a one-frame smoke test:

```bash
midas-retargeter-smoke
```

If this repo is not next to `midas_hand_mujoco`, point it at the MuJoCo repo:

```bash
MIDAS_HAND_MUJOCO_DIR=/path/to/midas_hand_mujoco midas-retargeter-smoke
```

Python usage:

```python
from midas_hand_retargeter import MidasHandRetargeter, RetargeterTuning

retargeter = MidasHandRetargeter.create(
    scaling_factor=1.0,
    tuning=RetargeterTuning(finger_abad_gain=1.4),
)
result = retargeter.retarget_landmarks(human_landmarks_21x3)

mujoco_targets = result.mujoco_control_dict()
hardware_targets = result.hardware_motor_positions
```

To make fingertip FK see the passive DIP/linkage motion during optimization,
enable option 2:

```python
retargeter = MidasHandRetargeter.create(coupling_mode="pip_dip_lookup")
```

This keeps the optimizer variables as active joints only, but `adaptor.py`
computes each passive `*_dip_joint` and `*_dip_linkage_joint` from the
corresponding `*_pip_joint` and folds the passive Jacobian columns back into
that PIP gradient. Use `coupling_mode="fixed_passive"` to return to option 1.

To align the retargeter zero with a user-specific neutral pose, retarget one
frame while the hand is held in neutral and capture calibration references:

```python
retargeter.retarget_landmarks(neutral_landmarks_21x3)
offsets = retargeter.calibrate_neutral_from_last_frame()

result = retargeter.retarget_landmarks(human_landmarks_21x3)
```

Future outputs use those references as zero, then rescale each side so the
original robot joint limits remain reachable. For example, a one-sided joint
such as `thumb_cmc_roll_joint` still maps from `0` to `2.15` after calibration.
Call `retargeter.clear_neutral_offsets()` to return to the raw mapping.

Teleop tuning lives in `midas_hand_retargeter/tuning.py`, but it is
intentionally small. The postprocess layer should mostly follow geometry; these
knobs are only coarse gains/smoothing:

- `finger_curl_gain`
- `finger_abad_gain`
- `finger_smoothing_alpha`
- `thumb_cmc_gain`
- `thumb_cmc_side_gain`
- `thumb_cmc_roll_gain`
- `thumb_flexion_gain`
- `thumb_smoothing_alpha`

Use `config.py` for optimizer-level changes such as target links, target human
landmark indices, scaling factor, and solver losses. Use `tuning.py` for
operator-facing teleop feel: splay sensitivity, filtering, curl ranges, and
thumb opposition.

Passive coupling modes:

- `fixed_passive`: option 1, simple and stable; passive joints are fixed in FK.
- `pip_dip_lookup`: option 2, more precise fingertip FK; passive non-thumb
  DIP/linkage joints follow the packaged MIDAS four-bar lookup from
  `midas_hand_api`.
