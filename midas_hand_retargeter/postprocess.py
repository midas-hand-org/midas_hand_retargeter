"""MIDAS-specific landmark postprocessing.

The upstream ``dex_retargeting`` vector optimizer gives a useful whole-hand
initial solution, but the MIDAS model has two practical wrinkles:

1. Finger DIP/linkage joints are passive downstream, so active MCP/PIP curl
   benefits from a direct MediaPipe bend heuristic.
2. Finger ab/ad and thumb CMC motion are sensitive to camera noise and user
   anatomy, so they get a small amount of MIDAS-specific shaping.

This module owns those mappings. Only broad gains/smoothing live in
``RetargeterTuning``; the rest stays as implementation constants here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import FINGER_NAMES
from .human import as_landmarks
from .tuning import DEFAULT_TUNING, RetargeterTuning


# MediaPipe hand landmark groups for the three robot fingers. Each tuple is
# (MCP, PIP, DIP, tip). Pinky is intentionally omitted because MIDAS has three
# fingers plus a thumb.
FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
}

# MediaPipe thumb landmark indices: CMC, MCP, IP, tip.
THUMB_LANDMARKS = (1, 2, 3, 4)

# Fingertip landmark per finger for thumb pinch snapping, derived from
# FINGER_LANDMARKS so it stays in sync with the three robot fingers.
PINCH_FINGER_TIPS = {finger: FINGER_LANDMARKS[finger][3] for finger in FINGER_NAMES}

# Internal MIDAS command ranges. These are not live-tuning knobs; they encode
# the current robot model's useful active-joint range for postprocess targets.
FINGER_MCP_PITCH_RANGE = (0.0, -1.35)
FINGER_PIP_RANGE = (0.0, -1.22)
FINGER_CURL_MAX_BEND = 1.35
FINGER_ABAD_DEADZONE = 0.05
FINGER_ABAD_SCALE = 1.0
FINGER_ABAD_LIMIT = 0.785
FINGER_ABAD_CURL_DAMPING = 0.5

THUMB_CMC_ROLL_RANGE = (0.0, 2.15)
THUMB_CMC_ROLL_DEADZONE = 0.05
THUMB_CMC_ROLL_SPAN = 0.45

# Continuous proximity opposition term. Distance from the thumb tip to the
# finger-MCP region, normalized by palm length (wrist -> middle MCP) so it is
# independent of hand size. Ratios are in palm-length units: at/above OPEN the
# thumb is abducted away from the palm (no opposition); at/below CLOSED the thumb
# tip is fully in over the palm (full opposition).
THUMB_OPPOSITION_OPEN_RATIO = 1.2
THUMB_OPPOSITION_CLOSED_RATIO = 0.5
THUMB_CMC_SIDE_OPEN = 0.0
THUMB_CMC_SIDE_RANGE = (-0.785, 0.9)
THUMB_CMC_SIDE_NEUTRAL_ANGLE = -0.3
THUMB_CMC_SIDE_DEADZONE = 0.05

THUMB_MCP_RANGE = (0.0, -1.57)
THUMB_MCP_MAX_BEND = 1.57
THUMB_DIP_RANGE = (0.0, -1.57)
THUMB_DIP_MAX_BEND = 1.57
THUMB_DIP_MCP_FOLLOW = 0.3


@dataclass
class JointTargetFilter:
    """Small per-joint exponential smoother for noisy postprocess targets."""

    values: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.values.clear()

    def update(self, joint_name: str, value: float, alpha: float) -> float:
        """Blend a new target into the previous target.

        ``alpha`` is the usual low-pass coefficient: ``1.0`` means no filtering,
        smaller values are smoother but add more lag. This is currently used for
        finger MCP ab/ad, which is much noisier than finger curl.
        """

        alpha = float(np.clip(alpha, 0.0, 1.0))
        previous = self.values.get(joint_name)
        filtered = float(value) if previous is None else _blend(previous, value, alpha)
        self.values[joint_name] = filtered
        return filtered


def finger_joint_targets_from_landmarks(
    landmarks: np.ndarray,
    tuning: RetargeterTuning = DEFAULT_TUNING,
) -> tuple[dict[str, float], dict[str, float]]:
    """Map human finger landmarks to active MIDAS finger joint targets.

    The mapping is intentionally simple:

    - MCP/PIP curl comes from PIP/DIP bend.
    - MCP ab/ad comes from lateral proximal-finger direction, then is damped
      while the finger curls because splay landmarks become less reliable.

    ``RetargeterTuning`` only applies broad curl/ab-ad gains and smoothing.

    Returns a ``(targets, filter_alphas)`` pair. ``filter_alphas`` contains
    curl-dependent LPF overrides for abad joints — tighter filtering at high
    curl where lateral landmarks are occluded.
    """

    points = as_landmarks(landmarks)

    # Compute all finger curls up front so each finger can reference its
    # neighbor's curl. When finger N curls it occludes ALL of finger N+1's
    # joints from the camera.
    #
    # Abad alpha uses max(own, neighbor): own curl also makes lateral detection
    # unreliable, so both sources tighten abad.
    # Pitch/pip alpha uses neighbor_curl only: own curl makes flexion detection
    # CLEARER (bend is obvious), so only neighbor-driven occlusion tightens them.
    curls = [
        _finger_curl(points, FINGER_LANDMARKS[finger], tuning)
        for finger in FINGER_NAMES
    ]

    targets: dict[str, float] = {}
    filter_alphas: dict[str, float] = {}
    for i, finger in enumerate(FINGER_NAMES):
        indices = FINGER_LANDMARKS[finger]
        curl = curls[i]
        neighbor_curl = curls[i - 1] if i > 0 else 0.0
        occlusion_curl = max(curl, neighbor_curl)

        abad_joint = f"{finger}_mcp_abad_joint"
        targets[abad_joint] = _finger_splay(points, indices, tuning, curl, occlusion_curl)
        filter_alphas[abad_joint] = _blend(
            tuning.finger_smoothing_alpha,
            tuning.finger_abad_alpha_curled,
            occlusion_curl,
        )
        pitch_joint = f"{finger}_mcp_pitch_joint"
        pip_joint = f"{finger}_pip_joint"
        targets[pitch_joint] = _blend(*FINGER_MCP_PITCH_RANGE, curl)
        targets[pip_joint] = _blend(*FINGER_PIP_RANGE, curl)
        if occlusion_curl > 0.0:
            # Use occlusion_curl (max of own + neighbor) so all fingers get equal
            # dampening when curled. Index has no left neighbor so neighbor_curl
            # would always be 0; using own curl makes it symmetric with middle/ring.
            # Scale by 0.2 keeps pitch/pip much lighter than abad.
            curl_pitch_pip = occlusion_curl * 0.2
            filter_alphas[pitch_joint] = _blend(
                tuning.finger_smoothing_alpha,
                tuning.finger_abad_alpha_curled,
                curl_pitch_pip,
            )
            filter_alphas[pip_joint] = _blend(
                tuning.finger_smoothing_alpha,
                tuning.finger_abad_alpha_curled,
                curl_pitch_pip,
            )
    return targets, filter_alphas


def thumb_pinch_engagements(
    landmarks: np.ndarray,
    tuning: RetargeterTuning = DEFAULT_TUNING,
) -> dict[str, float]:
    """Per-finger pinch engagement in 0..1, keyed by finger name.

    Each value ramps from 0 at ``thumb_pinch_distance`` to 1 at
    ``thumb_pinch_snap_distance`` based on the thumb-tip to that finger-tip
    distance. Unlike a single nearest-finger pick, returning all three lets
    callers blend continuously between fingers instead of switching discretely.
    """

    points = as_landmarks(landmarks)
    tip = points[THUMB_LANDMARKS[3]]
    pinch_range = tuning.thumb_pinch_distance - tuning.thumb_pinch_snap_distance
    engagements: dict[str, float] = {}
    for finger, tip_index in PINCH_FINGER_TIPS.items():
        distance = float(np.linalg.norm(tip - points[tip_index]))
        if pinch_range > 1e-6:
            ramp = _smoothstep((tuning.thumb_pinch_distance - distance) / pinch_range)
        else:
            ramp = 1.0 if distance <= tuning.thumb_pinch_snap_distance else 0.0
        engagements[finger] = float(np.clip(ramp, 0.0, 1.0))
    return engagements


def thumb_pinch_snap(
    landmarks: np.ndarray,
    tuning: RetargeterTuning = DEFAULT_TUNING,
) -> tuple[str, float]:
    """Return the fingertip the thumb is nearest and the pinch engagement 0..1.

    This is shared by the opposition cap (here in postprocess) and the per-finger
    CMC side offset (in the retargeter), so both react to the same
    orientation-independent thumb-to-fingertip distance. The nearest finger is
    the one with the highest engagement (i.e. smallest thumb-tip distance).
    """

    engagements = thumb_pinch_engagements(landmarks, tuning)
    nearest = max(engagements, key=engagements.get)
    return nearest, engagements[nearest]


# Softmax temperature (meters) for blending per-finger thumb CMC side offsets.
# Smaller keeps each finger's tuned value more distinct (sharper switch); larger
# blends neighbors more. ~half the adjacent fingertip spacing keeps the tuned
# values mostly intact while still interpolating across the gap.
THUMB_SIDE_SELECT_SOFTNESS = 0.012


def thumb_pinch_offset_weights(
    landmarks: np.ndarray,
    tuning: RetargeterTuning = DEFAULT_TUNING,
    softness: float = THUMB_SIDE_SELECT_SOFTNESS,
) -> tuple[dict[str, float], float]:
    """Return per-finger selection weights and an overall pinch engagement.

    Separates *which* finger the thumb is at from *how* engaged the pinch is:

    - ``weights``: a softmax over negative thumb-tip-to-fingertip distance,
      summing to 1. It picks the finger selectively (so each finger keeps its
      tuned offset) yet interpolates smoothly when the thumb is between two
      fingertips, instead of the hard nearest-finger switch that jumps.
    - ``engagement``: the strongest per-finger pinch ramp (0..1), used to gate
      how far the offset blends in from its resting value.
    """

    points = as_landmarks(landmarks)
    tip = points[THUMB_LANDMARKS[3]]
    fingers = list(PINCH_FINGER_TIPS)
    distances = np.array(
        [np.linalg.norm(tip - points[PINCH_FINGER_TIPS[f]]) for f in fingers]
    )
    logits = -(distances - distances.min()) / max(softness, 1e-6)
    weights = np.exp(logits)
    weights = weights / weights.sum()
    engagements = thumb_pinch_engagements(landmarks, tuning)
    engagement = max(engagements.values())
    return {f: float(w) for f, w in zip(fingers, weights)}, float(engagement)


def thumb_joint_targets_from_landmarks(
    landmarks: np.ndarray,
    tuning: RetargeterTuning = DEFAULT_TUNING,
) -> tuple[dict[str, float], dict[str, float]]:
    """Map human thumb landmarks to active MIDAS thumb joint targets.

    Thumb retargeting has three separate ideas:

    - MCP/DIP flexion uses thumb CMC-MCP-IP-tip bend angles.
    - CMC roll/opposition uses thumb-only palm-normal motion.
    - CMC side uses thumb-only in-plane side sweep.

    Returns ``(targets, filter_alphas)``. ``filter_alphas`` tightens the LPF on
    all thumb joints as the thumb curls, independent of neighboring fingers.
    """

    points = as_landmarks(landmarks)
    cmc, mcp, ip, tip = (points[index] for index in THUMB_LANDMARKS)

    cmc_to_mcp = mcp - cmc
    mcp_to_ip = ip - mcp
    ip_to_tip = tip - ip
    mcp_bend = _angle_between(cmc_to_mcp, mcp_to_ip)
    dip_bend = _angle_between(mcp_to_ip, ip_to_tip)

    mcp_curl = _smoothstep(tuning.thumb_flexion_gain * mcp_bend / THUMB_MCP_MAX_BEND)
    dip_curl = max(
        _smoothstep(tuning.thumb_flexion_gain * dip_bend / THUMB_DIP_MAX_BEND),
        THUMB_DIP_MCP_FOLLOW * mcp_curl,
    )

    palm_forward, palm_lateral, palm_normal = _palm_basis(points)
    thumb_proximal = mcp - cmc
    thumb_direction = thumb_proximal

    side_angle = _signed_angle_in_plane(
        thumb_direction,
        palm_forward,
        palm_lateral,
    )
    side_delta = _deadzone(
        side_angle - THUMB_CMC_SIDE_NEUTRAL_ANGLE,
        THUMB_CMC_SIDE_DEADZONE,
    )
    side_gain = tuning.thumb_cmc_gain * tuning.thumb_cmc_side_gain
    side_target = THUMB_CMC_SIDE_OPEN - (
        side_gain * side_delta
    )

    opposition_angle = _signed_angle_out_of_plane(
        thumb_direction,
        palm_forward,
        palm_lateral,
        palm_normal,
    )
    roll_angle = abs(opposition_angle)
    roll_gain = tuning.thumb_cmc_gain * tuning.thumb_cmc_roll_gain
    opposition = _smoothstep(
        roll_gain
        * (roll_angle - THUMB_CMC_ROLL_DEADZONE)
        / THUMB_CMC_ROLL_SPAN
    )
    opposition = float(np.clip(opposition, 0.0, 1.0))

    # Distance-based pinch snap: thumb-tip to finger-tip distance is
    # orientation-independent, so it catches pinch attempts the roll angle
    # misses when the hand is tilted relative to the camera. Generalized to all
    # three fingers: the thumb must roll further across the palm to reach
    # middle/ring than index, so each finger has its own opposition cap. We snap
    # toward whichever fingertip the thumb is currently nearest, so reaching for
    # the middle or ring drives a larger roll than an index pinch.
    pinch_caps = {
        "index": tuning.thumb_pinch_opposition_cap,
        "middle": tuning.thumb_pinch_middle_opposition_cap,
        "ring": tuning.thumb_pinch_ring_opposition_cap,
    }
    nearest_finger, pinch_ramp = thumb_pinch_snap(points, tuning)
    distance_opposition = float(np.clip(pinch_ramp, 0.0, pinch_caps[nearest_finger]))
    opposition = max(opposition, distance_opposition)

    # Continuous proximity opposition: as the thumb tip comes in over the palm
    # toward the finger bases, drive roll directly. This uses only inter-landmark
    # distances (orientation-independent), so unlike the out-of-plane metacarpal
    # angle it keeps responding when the hand is edge-on, and it tracks free
    # opposition that translates the thumb over the palm rather than tilting the
    # metacarpal. Layered as a floor via max(), so it never lowers the angle- or
    # pinch-driven roll. The finger-MCP centroid is a stable anchor: those bases
    # barely move as the fingers straighten or curl.
    proximity_gain = tuning.thumb_opposition_proximity_gain
    palm_scale = float(np.linalg.norm(points[9] - points[0]))
    proximity_span = THUMB_OPPOSITION_OPEN_RATIO - THUMB_OPPOSITION_CLOSED_RATIO
    if proximity_gain > 0.0 and palm_scale > 1e-6 and proximity_span > 1e-6:
        finger_mcp_centroid = points[[5, 9, 13]].mean(axis=0)
        tip_ratio = float(np.linalg.norm(tip - finger_mcp_centroid)) / palm_scale
        proximity_opposition = _smoothstep(
            proximity_gain
            * (THUMB_OPPOSITION_OPEN_RATIO - tip_ratio)
            / proximity_span
        )
        opposition = max(opposition, float(np.clip(proximity_opposition, 0.0, 1.0)))

    targets = {
        "thumb_cmc_roll_joint": _blend(
            *THUMB_CMC_ROLL_RANGE,
            opposition,
        ),
        "thumb_cmc_side_joint": float(
            np.clip(
                side_target,
                *THUMB_CMC_SIDE_RANGE,
            )
        ),
        "thumb_mcp_joint": _blend(
            *THUMB_MCP_RANGE,
            mcp_curl,
        ),
        "thumb_dip_joint": _blend(
            *THUMB_DIP_RANGE,
            dip_curl,
        ),
    }
    side_max = max(abs(THUMB_CMC_SIDE_RANGE[0]), abs(THUMB_CMC_SIDE_RANGE[1]))
    side_amount = float(np.clip(abs(side_target) / side_max, 0.0, 1.0))
    thumb_activity = max(opposition, mcp_curl, side_amount)
    thumb_alpha = _blend(
        tuning.thumb_smoothing_alpha,
        tuning.thumb_alpha_curled,
        thumb_activity,
    )
    filter_alphas = {joint: thumb_alpha for joint in targets}
    return targets, filter_alphas


def _finger_curl(
    points: np.ndarray,
    indices: tuple[int, int, int, int],
    tuning: RetargeterTuning,
) -> float:
    """Estimate a normalized 0..1 curl amount for one finger."""

    mcp, pip, dip, tip = (points[index] for index in indices)
    proximal = pip - mcp
    middle = dip - pip
    distal = tip - dip
    segment_lengths = (
        np.linalg.norm(proximal)
        + np.linalg.norm(middle)
        + np.linalg.norm(distal)
    )
    if segment_lengths < 1e-6:
        return 0.0

    pip_bend = _angle_between(proximal, middle)
    dip_bend = _angle_between(middle, distal)
    bend = 0.62 * pip_bend + 0.38 * dip_bend
    return _smoothstep(tuning.finger_curl_gain * bend / FINGER_CURL_MAX_BEND)


def _finger_splay(
    points: np.ndarray,
    indices: tuple[int, int, int, int],
    tuning: RetargeterTuning,
    curl: float,
    occlusion_curl: float | None = None,
) -> float:
    """Estimate MCP ab/ad from palm-local lateral finger direction.

    MediaPipe splay is noisy when the finger is curled or partially occluded, so
    this uses a weighted proximal direction, a palm-local basis, a deadzone, an
    explicit limit, and curl-dependent damping.

    ``occlusion_curl`` is the effective curl used for damping — callers pass
    ``max(own_curl, neighbor_curl)`` so that a curled adjacent finger (which
    occludes this finger's lateral joints from the camera) also increases damping.
    Defaults to ``curl`` when not provided.
    """

    mcp, pip, dip, _ = (points[index] for index in indices)
    proximal = pip - mcp
    secondary = 0.5 * (dip - mcp)
    direction = 0.75 * proximal + 0.25 * secondary
    palm_forward, palm_lateral, palm_normal = _palm_basis(points)
    in_palm_direction = direction - np.dot(direction, palm_normal) * palm_normal
    forward = abs(float(np.dot(in_palm_direction, palm_forward))) + 1e-6
    lateral = float(np.dot(in_palm_direction, palm_lateral))
    lateral_angle = float(np.arctan2(lateral, forward))

    effective_curl = curl if occlusion_curl is None else occlusion_curl
    splay = _deadzone(lateral_angle, tuning.finger_abad_deadzone)
    curl_damping = 1.0 - tuning.finger_abad_curl_damping * float(np.clip(effective_curl, 0.0, 1.0))
    return float(
        np.clip(
            curl_damping * tuning.finger_abad_gain * FINGER_ABAD_SCALE * splay,
            -FINGER_ABAD_LIMIT,
            FINGER_ABAD_LIMIT,
        )
    )


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.arccos(cosine))


def _palm_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a per-frame palm basis from wrist and MCP landmarks.

    The postprocess layer uses this local basis instead of camera axes so thumb
    CMC roll and side stay meaningful as the whole hand rotates in view.
    """

    wrist = points[0]
    index_mcp = points[5]
    middle_mcp = points[9]
    ring_mcp = points[13]
    palm_forward = _unit(middle_mcp - wrist)
    palm_lateral = _unit(ring_mcp - index_mcp)
    palm_normal = _unit(np.cross(palm_lateral, palm_forward))
    if np.linalg.norm(palm_normal) < 1e-6:
        palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    palm_lateral = _unit(np.cross(palm_forward, palm_normal))
    return palm_forward, palm_lateral, palm_normal


def _signed_angle_in_plane(
    vector: np.ndarray,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
) -> float:
    """Signed angle of ``vector`` inside the palm plane."""

    a = float(np.dot(vector, axis_a))
    b = float(np.dot(vector, axis_b))
    return float(np.arctan2(b, a))


def _signed_angle_out_of_plane(
    vector: np.ndarray,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
    normal: np.ndarray,
) -> float:
    """Signed angle from palm plane toward palm normal."""

    in_plane_a = float(np.dot(vector, axis_a))
    in_plane_b = float(np.dot(vector, axis_b))
    in_plane_norm = float(np.hypot(in_plane_a, in_plane_b))
    normal_value = float(np.dot(vector, normal))
    return float(np.arctan2(normal_value, in_plane_norm + 1e-9))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(vector / norm, dtype=np.float32)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    return value - np.copysign(deadzone, value)


def _blend(a: float, b: float, amount: float) -> float:
    amount = float(np.clip(amount, 0.0, 1.0))
    return a + amount * (b - a)
