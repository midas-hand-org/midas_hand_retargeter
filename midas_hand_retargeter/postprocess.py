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
from .params import DEFAULT_PROFILE, FingerParams, RetargetProfile, ThumbParams
from .tuning import DEFAULT_TUNING, RetargeterTuning


def as_profile(tuning) -> RetargetProfile:
    """Accept either a RetargetProfile or a legacy flat RetargeterTuning."""

    if isinstance(tuning, RetargetProfile):
        return tuning
    if isinstance(tuning, RetargeterTuning):
        return RetargetProfile.from_legacy_tuning(tuning)
    raise TypeError(
        f"Expected RetargetProfile or RetargeterTuning, got {type(tuning).__name__}"
    )


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

# Internal MIDAS command ranges. These are not live-tuning knobs; they encode
# the current robot model's useful active-joint range for postprocess targets.
FINGER_MCP_PITCH_RANGE = (0.0, -1.35)
FINGER_PIP_RANGE = (0.0, -1.22)
FINGER_ABAD_DEADZONE = 0.05
FINGER_ABAD_SCALE = 1.0
FINGER_ABAD_LIMIT = 0.785
FINGER_ABAD_CURL_DAMPING = 0.5

THUMB_CMC_ROLL_RANGE = (0.0, 2.15)
THUMB_CMC_ROLL_DEADZONE = 0.05
THUMB_CMC_ROLL_SPAN = 0.45
THUMB_CMC_SIDE_OPEN = 0.0
THUMB_CMC_SIDE_RANGE = (-0.785, 0.9)
THUMB_CMC_SIDE_NEUTRAL_ANGLE = -0.3
THUMB_CMC_SIDE_DEADZONE = 0.05

THUMB_MCP_RANGE = (0.0, -1.57)
THUMB_DIP_RANGE = (0.0, -1.57)
THUMB_DIP_MCP_FOLLOW = 0.3

# Human bend -> curl normalizers (FINGER_CURL_MAX_BEND, THUMB_MCP_MAX_BEND,
# THUMB_DIP_MAX_BEND) are source-sensitive and now live on ``RetargeterTuning``
# so vision and glove can normalize their different measured ROM independently.


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
    tuning: RetargetProfile | RetargeterTuning = DEFAULT_PROFILE,
) -> dict[str, float]:
    """Map human finger landmarks to active MIDAS finger joint targets.

    The mapping is intentionally simple:

    - MCP/PIP curl comes from PIP/DIP bend.
    - MCP ab/ad comes from lateral proximal-finger direction, then is damped
      while the finger curls because splay landmarks become less reliable.

    ``RetargeterTuning`` only applies broad curl/ab-ad gains and smoothing.
    """

    profile = as_profile(tuning)
    points = as_landmarks(landmarks)
    targets: dict[str, float] = {}
    for finger in FINGER_NAMES:
        params = profile.finger(finger)
        if not params.enabled:
            continue  # hold last command; the caller keeps the previous value
        indices = FINGER_LANDMARKS[finger]
        curl = _finger_curl(points, indices, params)
        targets[f"{finger}_mcp_abad_joint"] = _finger_splay(
            points, indices, params, curl
        )
        targets[f"{finger}_mcp_pitch_joint"] = _blend(*params.mcp_pitch_range, curl)
        targets[f"{finger}_pip_joint"] = _blend(*params.pip_range, curl)
    return targets


def thumb_joint_targets_from_landmarks(
    landmarks: np.ndarray,
    tuning: RetargetProfile | RetargeterTuning = DEFAULT_PROFILE,
) -> dict[str, float]:
    """Map human thumb landmarks to active MIDAS thumb joint targets.

    Thumb retargeting has three separate ideas:

    - MCP/DIP flexion uses thumb CMC-MCP-IP-tip bend angles.
    - CMC roll/opposition uses thumb-only palm-normal motion.
    - CMC side uses thumb-only in-plane side sweep.
    """

    params = as_profile(tuning).thumb
    if not params.enabled:
        return {}
    points = as_landmarks(landmarks)
    cmc, mcp, ip, tip = (points[index] for index in THUMB_LANDMARKS)

    cmc_to_mcp = mcp - cmc
    mcp_to_ip = ip - mcp
    ip_to_tip = tip - ip
    mcp_bend = _angle_between(cmc_to_mcp, mcp_to_ip)
    dip_bend = _angle_between(mcp_to_ip, ip_to_tip)

    mcp_curl = _smoothstep(params.flexion_gain * mcp_bend / params.mcp_max_bend)
    dip_curl = max(
        _smoothstep(params.flexion_gain * dip_bend / params.dip_max_bend),
        params.dip_follows_mcp * mcp_curl,
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
        side_angle - params.cmc_side_neutral_angle,
        params.cmc_side_deadzone,
    )
    side_target = params.cmc_side_open - (params.cmc_side_gain * side_delta)

    opposition_angle = _signed_angle_out_of_plane(
        thumb_direction,
        palm_forward,
        palm_lateral,
        palm_normal,
    )
    roll_angle = abs(opposition_angle)
    opposition = _smoothstep(
        params.cmc_roll_gain
        * (roll_angle - params.cmc_roll_deadzone)
        / params.cmc_roll_span
    )
    opposition = float(np.clip(opposition, 0.0, 1.0))

    return {
        "thumb_cmc_roll_joint": _blend(*params.cmc_roll_range, opposition),
        "thumb_cmc_side_joint": float(np.clip(side_target, *params.cmc_side_range)),
        "thumb_mcp_joint": _blend(*params.mcp_range, mcp_curl),
        "thumb_dip_joint": _blend(*params.dip_range, dip_curl),
    }


def _finger_curl(
    points: np.ndarray,
    indices: tuple[int, int, int, int],
    params: FingerParams,
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
    bend = params.curl_pip_weight * pip_bend + params.curl_dip_weight * dip_bend
    return _smoothstep(params.curl_gain * bend / params.curl_max_bend)


def _finger_splay(
    points: np.ndarray,
    indices: tuple[int, int, int, int],
    params: FingerParams,
    curl: float,
) -> float:
    """Estimate MCP ab/ad from palm-local lateral finger direction.

    MediaPipe splay is noisy when the finger is curled or partially occluded, so
    this uses a weighted proximal direction, a palm-local basis, a deadzone, an
    explicit limit, and curl-dependent damping.
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

    splay = _deadzone(lateral_angle, params.splay_deadzone)
    curl_damping = 1.0 - params.splay_curl_damping * float(np.clip(curl, 0.0, 1.0))
    return float(
        np.clip(
            curl_damping * params.splay_gain * splay,
            -params.splay_limit,
            params.splay_limit,
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


def palm_basis(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Public accessor for the palm-local basis ``(forward, lateral, normal)``.

    Exposed so diagnostics and the tuning UI stop reaching for the private
    ``_palm_basis``.
    """

    return _palm_basis(as_landmarks(landmarks))


def analytic_debug(
    landmarks: np.ndarray,
    tuning: RetargetProfile | RetargeterTuning = DEFAULT_PROFILE,
) -> dict:
    """Intermediate quantities behind the joint targets, for display.

    A tuning UI that only shows the final joint angle cannot tell "the curl
    estimate is wrong" from "the output range is too narrow". These are the
    values between those two steps.
    """

    profile = as_profile(tuning)
    points = as_landmarks(landmarks)
    forward, lateral, normal = _palm_basis(points)

    debug: dict[str, dict] = {}
    for finger in FINGER_NAMES:
        params = profile.finger(finger)
        indices = FINGER_LANDMARKS[finger]
        curl = _finger_curl(points, indices, params)
        debug[finger] = {
            "curl": curl,
            "curl_damping": 1.0 - params.splay_curl_damping * float(np.clip(curl, 0.0, 1.0)),
            "splay_rad": _finger_splay(points, indices, params, curl),
        }

    cmc, mcp, ip, tip = (points[index] for index in THUMB_LANDMARKS)
    thumb_direction = mcp - cmc
    side_angle = _signed_angle_in_plane(thumb_direction, forward, lateral)
    opposition_angle = _signed_angle_out_of_plane(thumb_direction, forward, lateral, normal)
    debug["thumb"] = {
        "mcp_bend": _angle_between(mcp - cmc, ip - mcp),
        "dip_bend": _angle_between(ip - mcp, tip - ip),
        "side_angle": float(side_angle),
        "side_delta": float(side_angle - profile.thumb.cmc_side_neutral_angle),
        "opposition_angle": float(opposition_angle),
    }

    debug["_palm"] = {
        "forward": [float(v) for v in forward],
        "lateral": [float(v) for v in lateral],
        "normal": [float(v) for v in normal],
    }
    return debug
