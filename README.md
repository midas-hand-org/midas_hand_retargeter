# MIDAS Hand Retargeter

Maps human hand landmarks onto the 13 actuated joints of the MIDAS robot hand.

```bash
pip install midas-hand-retargeter
```

```python
import numpy as np
from midas_hand_retargeter import MidasHandRetargeter

retargeter = MidasHandRetargeter.create()
result = retargeter.retarget_landmarks(landmarks)   # (21, 3), MediaPipe order

result.active_joint_positions      # {joint_name: radians}, 13 entries
result.hardware_motor_positions    # (13,) in motor order, thumb first
result.mujoco_control_dict()       # {joint_name: target} for MuJoCo actuators
```

The default install needs only numpy. It reads no files and requires no robot
model on disk.

## Which method is this?

**A hand-written analytic map, not an optimizer.** For each finger it measures
the blended PIP/DIP bend and maps it through a smoothstep to MCP-pitch and PIP
commands; splay comes from the lateral finger direction in a palm-local frame,
damped as the finger curls. The thumb gets flexion from its segment bends, side
sweep from an in-plane angle, and opposition from an out-of-plane angle.

Three modes are available:

| `mode` | behaviour | needs |
|---|---|---|
| `analytic` | **default.** The geometric map above. | numpy |
| `vector` | `dex_retargeting`'s nlopt vector optimizer, analytic layer off. | `[vector]` extra + the URDF |
| `refine` | optimizer, then the analytic layer overwrites what it owns. | `[vector]` extra + the URDF |

`refine` is the historical behaviour and is kept for regression parity only.
It is not a blend: the analytic layer writes **all 13** actuated joints, so the
optimizer's solution never reaches the hand. `analytic` produces bit-identical
output — verified across a 45-pose golden set — while being ~6.5× faster
(1.2 ms vs 7.9 ms per frame) and dropping ~3.9 GB of torch and CUDA wheels from
the dependency set.

If you want the optimizer to actually drive the joints, ask for `vector`.

## Per-finger tuning

Every knob is per digit — there is no global "finger gain", and the MIDAS hand
has no pinky.

```python
from midas_hand_retargeter import RetargetProfile

profile = RetargetProfile().with_values({
    "index.curl_gain": 1.3,             # close sooner
    "index.mcp_pitch_range": (0.0, -1.8),   # reach the full URDF travel
    "ring.splay_gain": 0.8,
    "thumb.cmc_roll_span": 0.5,
})
retargeter.profile = profile            # atomic; safe to swap mid-run
```

50 parameters across thumb/index/middle/ring. Unknown paths raise rather than
being silently ignored. The built-in defaults deliberately do not reach the
robot's full range (MCP pitch stops at −1.35 of −1.8, PIP at −1.22 of −1.45);
widening `*_range` is how you reclaim it.

For a live UI over these, see `midas-hand-tune` in
[`midas_hand_teleop`](https://github.com/midas-hand-org/midas_hand_teleop).

### Presets

```python
from midas_hand_retargeter import presets

presets.save("~/.midas_hand/retarget_presets/pinch.json", profile,
             neutral_offsets=retargeter.neutral_joint_offsets)
profile, neutral = presets.load("~/.midas_hand/retarget_presets/pinch.json")
```

Presets carry the neutral calibration and are validated against the robot's
joint limits on load.

## Input convention

A `(21, 3)` array in MediaPipe landmark order, in metres. **The analytic map is
invariant to the input frame — including chirality.** Rotating, translating or
mirroring the landmarks produces byte-identical joint targets, so a frame
problem is almost never the cause of bad tracking; look at the gains and bend
normalizers first. (The `vector` optimizer *is* handed, and does care.)

## Passive joints

The three non-thumb DIP joints are driven by a four-bar linkage, not motors.

- `fixed_passive` (default) leaves them fixed; MuJoCo closes the loop with
  constraints and the hardware API closes it with its own lookup.
- `pip_dip_lookup` fills them from the MIDAS four-bar table. Needs the
  `[lookup]` extra, and works with no solver installed.

## Extras

| extra | for |
|---|---|
| `vector` | `mode="vector"` / `"refine"` (`dex_retargeting`, torch) |
| `lookup` | `coupling_mode="pip_dip_lookup"` (`midas-hand`) |
| `dev` | tests |

## Tests

```bash
pytest
```

The golden fixture in `tests/goldens/` freezes the analytic layer's output. If
it changes, that is a behaviour change — explain it before regenerating with
`python -m tests.generate_goldens`.

## License

MIT. See `LICENSE`.
