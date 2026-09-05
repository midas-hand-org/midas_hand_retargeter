"""Deterministic synthetic hand poses for behaviour-freeze goldens.

These builders are vendored here on purpose. The golden fixture is the safety
net for the retargeting refactor, so it must not depend on anything outside
this repo (in particular not on ``midas_hand_teleop.diagnostics``, which would
invert the dependency direction: teleop depends on the retargeter, never the
reverse).

The pose set deliberately sweeps the *parameter surface* the analytic layer
actually reads — per-finger curl, per-finger splay, curl/splay interaction
(which exercises the curl-damping term), and the three thumb angles. It does
NOT lean on random rigid transforms: ``test_postprocess_invariance`` already
proves every target is invariant to rotation, reflection and translation, so
such frames would add coverage of exactly nothing.
"""

from __future__ import annotations

import math

import numpy as np

# Phalanx length (m). Uniform across fingers: the analytic layer reads only
# angles between segments, so absolute scale is irrelevant to the targets.
SEGMENT = 0.035

# Palmar direction in the canonical build frame; fingers curl toward it.
PALMAR = np.array([0.0, 0.0, -1.0])

# MCP lateral offsets (m) for index/middle/ring. Sets the palm's lateral axis.
_FINGER_BASE_X = {"index": -0.020, "middle": 0.0, "ring": 0.020}
_FINGER_BASE = {name: np.array([x, 0.045, 0.0]) for name, x in _FINGER_BASE_X.items()}
_FINGER_ROOT_INDEX = {"index": 5, "middle": 9, "ring": 13}

# Thumb CMC origin (m), offset toward the index side of the palm.
_THUMB_CMC = np.array([-0.035, 0.010, 0.005])


def _chain(base, curl, splay, *, distal_gain=(1.2, 0.8, 0.0), proximal_gain=0.5):
    """Build a 4-point finger chain bending by ``curl`` and yawing by ``splay``.

    ``curl`` is a dimensionless amount: each joint's bend is a fixed multiple of
    it, so the blended PIP/DIP bend the analytic layer measures rises smoothly
    and monotonically with it. ``splay`` is a yaw angle (rad) in the palm plane.
    """

    heading = np.array([math.sin(splay), math.cos(splay), 0.0])
    points = [np.asarray(base, dtype=float)]
    angle = proximal_gain * curl
    for extra in distal_gain:
        step = math.cos(angle) * heading + math.sin(angle) * PALMAR
        points.append(points[-1] + SEGMENT * step)
        angle += extra * curl
    return points


def hand_pose(
    *,
    curls=(0.0, 0.0, 0.0),
    splays=(0.0, 0.0, 0.0),
    thumb_curl=0.0,
    thumb_oppose=0.0,
    thumb_side=0.0,
):
    """Build a (21, 3) MediaPipe-ordered right-hand pose.

    ``curls``/``splays`` are per-finger (index, middle, ring). ``thumb_oppose``
    rotates the thumb out of the palm plane (drives CMC roll / opposition),
    ``thumb_side`` yaws it within the plane (drives CMC side), and
    ``thumb_curl`` bends it (drives MCP/DIP flexion).

    The pinky (17..20) is left at the origin: the MIDAS hand has no pinky and
    the analytic layer never reads those landmarks.
    """

    kp = np.zeros((21, 3), dtype=np.float64)

    for name, curl, splay in zip(("index", "middle", "ring"), curls, splays, strict=True):
        root = _FINGER_ROOT_INDEX[name]
        for offset, point in enumerate(_chain(_FINGER_BASE[name], curl, splay)):
            kp[root + offset] = point

    # Thumb: a base heading angled off the palm's distal axis, rotated out of
    # the palm plane by `thumb_oppose` and within it by `thumb_side`.
    base = np.array([-0.31, 0.95, 0.0])
    base /= np.linalg.norm(base)
    lateral = np.cross(PALMAR, base)
    lateral /= np.linalg.norm(lateral)
    heading = (
        math.cos(thumb_oppose) * (math.cos(thumb_side) * base + math.sin(thumb_side) * lateral)
        + math.sin(thumb_oppose) * PALMAR
    )
    heading /= np.linalg.norm(heading)

    points = [_THUMB_CMC]
    angle = 0.6 * thumb_curl
    for extra in (0.9, 0.6, 0.0):
        step = math.cos(angle) * heading + math.sin(angle) * PALMAR
        points.append(points[-1] + SEGMENT * step)
        angle += extra * thumb_curl
    for index, point in zip((1, 2, 3, 4), points, strict=True):
        kp[index] = point

    return kp


def _named_poses():
    """Yield (name, pose) covering the analytic layer's parameter surface."""

    yield "flat_open", hand_pose()

    # Graded uniform curl: pins the smoothstep shape and the lerp output ranges.
    for step in (0.25, 0.5, 0.75, 1.0, 1.5):
        yield f"curl_uniform_{step:g}", hand_pose(curls=(step,) * 3)

    # Single-finger curl: pins per-finger independence (what Phase 2 splits up).
    for i, name in enumerate(("index", "middle", "ring")):
        curls = [0.0, 0.0, 0.0]
        curls[i] = 0.9
        yield f"curl_{name}_only", hand_pose(curls=tuple(curls))

    # Single-finger splay, both signs: pins splay sign, deadzone and limit.
    for i, name in enumerate(("index", "middle", "ring")):
        for sign, tag in ((1.0, "pos"), (-1.0, "neg")):
            splays = [0.0, 0.0, 0.0]
            splays[i] = sign * 0.35
            yield f"splay_{name}_{tag}", hand_pose(splays=tuple(splays))

    # Tiny splay inside the deadzone: pins the deadzone threshold itself.
    yield "splay_within_deadzone", hand_pose(splays=(0.03, -0.03, 0.02))

    # Curl+splay together: pins the curl-damping term on splay.
    for curl in (0.3, 0.8, 1.3):
        yield (
            f"splay_damped_by_curl_{curl:g}",
            hand_pose(curls=(curl,) * 3, splays=(0.4, -0.4, 0.4)),
        )

    # Thumb axes, swept one at a time.
    for oppose in (0.0, 0.2, 0.45, 0.8, 1.2):
        yield f"thumb_oppose_{oppose:g}", hand_pose(thumb_oppose=oppose)
    for side in (-0.6, -0.3, 0.0, 0.3, 0.6):
        yield f"thumb_side_{side:g}", hand_pose(thumb_side=side)
    for curl in (0.25, 0.5, 1.0, 1.5):
        yield f"thumb_curl_{curl:g}", hand_pose(thumb_curl=curl)

    # Combined functional poses.
    yield "fist", hand_pose(curls=(1.4,) * 3, thumb_curl=1.2, thumb_oppose=0.9)
    yield "pinch_index", hand_pose(curls=(0.9, 0.1, 0.1), thumb_curl=0.8, thumb_oppose=1.0)
    yield "spread_open", hand_pose(splays=(-0.5, 0.0, 0.5), thumb_side=-0.5)
    yield "point_index", hand_pose(curls=(0.0, 1.3, 1.3), thumb_curl=1.0, thumb_oppose=0.7)

    # A few seeded pseudo-random poses in the plausible range, for breadth.
    rng = np.random.default_rng(20260904)
    for i in range(8):
        yield (
            f"random_{i}",
            hand_pose(
                curls=tuple(rng.uniform(0.0, 1.4, 3)),
                splays=tuple(rng.uniform(-0.45, 0.45, 3)),
                thumb_curl=float(rng.uniform(0.0, 1.4)),
                thumb_oppose=float(rng.uniform(0.0, 1.2)),
                thumb_side=float(rng.uniform(-0.6, 0.6)),
            ),
        )


#: Ordered (name, (21, 3) pose) pairs. Order is part of the golden contract.
GOLDEN_POSES = tuple(_named_poses())
