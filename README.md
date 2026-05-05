# MIDAS Hand Retargeter

Thin MIDAS-specific wrappers around the installed `dex_retargeting` package.
This repo owns only MIDAS retargeting configuration and output adapters; it
does not duplicate the upstream optimizer, the MuJoCo model, the MediaPipe
webcam pipeline, or the hardware API.

Current mode:

- vector-based retargeting
- active MIDAS joints only
- passive finger DIP/linkage joints fixed inside the retargeting FK model
- downstream coupling handled by `midas_hand_mujoco` closed-loop MJCF or
  `midas_hand_api` PIP-DIP lookup

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

Teleop tuning lives in `midas_hand_retargeter/tuning.py`. The most useful
knobs are:

- `finger_abad_gain`, `finger_abad_limit`, `finger_abad_deadzone`
- `finger_abad_alpha`, `finger_abad_curl_damping`, and `finger_abad_sign`
- `thumb_cmc_roll_open`, `thumb_cmc_roll_oppose`
- `thumb_cmc_side_open`, `thumb_cmc_side_oppose`
- `thumb_mcp_closed`, `thumb_dip_closed`
- `thumb_pinch_gain`

Option 2 placeholder:

`midas_hand_retargeter.adaptor.MidasCoupledKinematicAdaptor` is reserved for
adding a PIP-DIP-aware kinematic adaptor later. That adaptor should fill
passive DIP/linkage qpos from active PIP before FK and fold passive Jacobian
columns back into the PIP gradient.
