"""MIDAS vector-retargeting wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import MidasRetargeterConfig
from .constants import ACTIVE_JOINT_NAMES, FINGER_NAMES, HARDWARE_MOTOR_JOINT_NAMES
from .human import landmarks_to_vectors
from .tuning import DEFAULT_TUNING, RetargeterTuning


FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
}


@dataclass(frozen=True)
class RetargetingResult:
    """Retargeted joint command in several useful output layouts."""

    robot_qpos: np.ndarray
    robot_joint_names: tuple[str, ...]
    ref_vectors: np.ndarray
    active_joint_positions: dict[str, float]
    hardware_motor_positions: np.ndarray
    fixed_joint_positions: dict[str, float]

    def active_vector(self, joint_names=ACTIVE_JOINT_NAMES) -> np.ndarray:
        return np.asarray(
            [self.active_joint_positions[name] for name in joint_names],
            dtype=np.float32,
        )

    def mujoco_control_dict(self) -> dict[str, float]:
        return dict(self.active_joint_positions)


class MidasHandRetargeter:
    """Thin MIDAS wrapper around the installed ``dex_retargeting`` package."""

    def __init__(self, config: MidasRetargeterConfig | None = None):
        self.config = config or MidasRetargeterConfig()
        dex_config = self.config.build_dex_config()
        self._retargeting = dex_config.build()
        self.robot_joint_names = tuple(self._retargeting.joint_names)
        self.active_joint_names = tuple(self.config.target_joint_names)
        self.fixed_joint_names = tuple(self._retargeting.optimizer.fixed_joint_names)
        self._fixed_qpos = np.asarray(
            [
                self.config.passive_fixed_qpos.get(joint_name, 0.0)
                for joint_name in self.fixed_joint_names
            ],
            dtype=np.float32,
        )
        self._filtered_postprocess_targets: dict[str, float] = {}
        self.reset()

    @classmethod
    def create(cls, **config_overrides) -> "MidasHandRetargeter":
        return cls(MidasRetargeterConfig(**config_overrides))

    @property
    def dex_retargeting(self):
        return self._retargeting

    def landmarks_to_vectors(self, landmarks: np.ndarray) -> np.ndarray:
        return landmarks_to_vectors(
            landmarks,
            target_link_human_indices=self.config.target_link_human_indices,
        )

    def retarget_landmarks(self, landmarks: np.ndarray) -> RetargetingResult:
        return self.retarget_vectors(self.landmarks_to_vectors(landmarks), landmarks=landmarks)

    def retarget_vectors(
        self,
        ref_vectors: np.ndarray,
        *,
        landmarks: np.ndarray | None = None,
    ) -> RetargetingResult:
        vectors = np.asarray(ref_vectors, dtype=np.float32)
        expected_shape = (len(self.config.target_task_link_names), 3)
        if vectors.shape != expected_shape:
            raise ValueError(f"Expected ref_vectors shape {expected_shape}, got {vectors.shape}")

        robot_qpos = self._retargeting.retarget(vectors, fixed_qpos=self._fixed_qpos)
        if landmarks is not None:
            if self.config.finger_postprocess:
                robot_qpos = self._apply_finger_postprocess(robot_qpos, landmarks)
            if self.config.thumb_postprocess:
                robot_qpos = self._apply_thumb_postprocess(robot_qpos, landmarks)
        return self._make_result(robot_qpos, vectors)

    def set_qpos(self, robot_qpos: np.ndarray) -> None:
        self._retargeting.set_qpos(np.asarray(robot_qpos, dtype=np.float32))

    def reset(self) -> None:
        self._retargeting.set_qpos(
            np.zeros(len(self.robot_joint_names), dtype=np.float32)
        )
        self._filtered_postprocess_targets.clear()

    def _make_result(self, robot_qpos: np.ndarray, ref_vectors: np.ndarray) -> RetargetingResult:
        qpos_by_name: Mapping[str, float] = {
            name: float(robot_qpos[index])
            for index, name in enumerate(self.robot_joint_names)
        }
        active = {
            name: qpos_by_name[name]
            for name in self.active_joint_names
        }
        hardware = np.asarray(
            [active[name] for name in HARDWARE_MOTOR_JOINT_NAMES],
            dtype=np.float32,
        )
        fixed = {
            name: float(value)
            for name, value in zip(self.fixed_joint_names, self._fixed_qpos)
        }
        return RetargetingResult(
            robot_qpos=np.asarray(robot_qpos, dtype=np.float32),
            robot_joint_names=self.robot_joint_names,
            ref_vectors=np.asarray(ref_vectors, dtype=np.float32),
            active_joint_positions=active,
            hardware_motor_positions=hardware,
            fixed_joint_positions=fixed,
        )

    def _apply_thumb_postprocess(
        self,
        robot_qpos: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        qpos = np.asarray(robot_qpos, dtype=np.float32).copy()
        thumb_targets = _thumb_targets_from_landmarks(landmarks, self.config.tuning)
        for joint_name, value in thumb_targets.items():
            if joint_name not in self.robot_joint_names:
                continue
            joint_index = self.robot_joint_names.index(joint_name)
            qpos[joint_index] = self._clip_joint(joint_name, value)
        return qpos

    def _apply_finger_postprocess(
        self,
        robot_qpos: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        qpos = np.asarray(robot_qpos, dtype=np.float32).copy()
        finger_targets = _finger_targets_from_landmarks(landmarks, self.config.tuning)
        for joint_name, value in finger_targets.items():
            if joint_name not in self.robot_joint_names:
                continue
            if joint_name.endswith("_mcp_abad_joint"):
                value = self._filter_postprocess_target(
                    joint_name,
                    value,
                    self.config.tuning.finger_abad_alpha,
                )
            joint_index = self.robot_joint_names.index(joint_name)
            qpos[joint_index] = self._clip_joint(joint_name, value)
        return qpos

    def _clip_joint(self, joint_name: str, value: float) -> float:
        joint_index = self.robot_joint_names.index(joint_name)
        lower, upper = self._retargeting.optimizer.robot.joint_limits[joint_index]
        return float(np.clip(value, lower, upper))

    def _filter_postprocess_target(
        self,
        joint_name: str,
        value: float,
        alpha: float,
    ) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        previous = self._filtered_postprocess_targets.get(joint_name)
        if previous is None:
            filtered = float(value)
        else:
            filtered = _blend(previous, float(value), alpha)
        self._filtered_postprocess_targets[joint_name] = filtered
        return filtered


def _thumb_targets_from_landmarks(
    landmarks: np.ndarray,
    tuning: RetargeterTuning = DEFAULT_TUNING,
) -> dict[str, float]:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {points.shape}")

    cmc_to_mcp = points[2] - points[1]
    mcp_to_ip = points[3] - points[2]
    ip_to_tip = points[4] - points[3]
    mcp_bend = _angle_between(cmc_to_mcp, mcp_to_ip)
    dip_bend = _angle_between(mcp_to_ip, ip_to_tip)
    mcp_curl = _smooth_range(
        mcp_bend,
        tuning.thumb_mcp_open_angle,
        tuning.thumb_mcp_closed_angle,
    )
    dip_curl = max(
        _smooth_range(
            dip_bend,
            tuning.thumb_dip_open_angle,
            tuning.thumb_dip_closed_angle,
        ),
        tuning.thumb_dip_mcp_follow * mcp_curl,
    )

    # If the thumb tip approaches the index tip, add opposition even when the
    # thumb joints themselves are fairly straight.
    palm_scale = np.linalg.norm(points[9] - points[0]) + 1e-6
    thumb_index_ratio = np.linalg.norm(points[4] - points[8]) / palm_scale
    pinch = _smooth_range(
        thumb_index_ratio,
        tuning.thumb_pinch_open_ratio,
        tuning.thumb_pinch_closed_ratio,
    )
    opposition = max(
        tuning.thumb_pinch_gain * pinch,
        tuning.thumb_curl_opposition_gain * mcp_curl,
    )
    opposition = float(np.clip(opposition, 0.0, 1.0))

    return {
        "thumb_cmc_roll_joint": _blend(
            tuning.thumb_cmc_roll_open,
            tuning.thumb_cmc_roll_oppose,
            opposition,
        ),
        "thumb_cmc_side_joint": _blend(
            tuning.thumb_cmc_side_open,
            tuning.thumb_cmc_side_oppose,
            opposition,
        ),
        "thumb_mcp_joint": _blend(
            tuning.thumb_mcp_open,
            tuning.thumb_mcp_closed,
            mcp_curl,
        ),
        "thumb_dip_joint": _blend(
            tuning.thumb_dip_open,
            tuning.thumb_dip_closed,
            dip_curl,
        ),
    }


def _finger_targets_from_landmarks(
    landmarks: np.ndarray,
    tuning: RetargeterTuning = DEFAULT_TUNING,
) -> dict[str, float]:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {points.shape}")

    targets: dict[str, float] = {}
    for finger in FINGER_NAMES:
        indices = FINGER_LANDMARKS[finger]
        curl = _finger_curl(points, indices, tuning)
        targets[f"{finger}_mcp_abad_joint"] = _finger_splay(
            points,
            finger,
            indices,
            tuning,
            curl,
        )
        targets[f"{finger}_mcp_pitch_joint"] = _blend(
            tuning.finger_mcp_pitch_open,
            tuning.finger_mcp_pitch_closed,
            curl,
        )
        targets[f"{finger}_pip_joint"] = _blend(
            tuning.finger_pip_open,
            tuning.finger_pip_closed,
            curl,
        )
    return targets


def _finger_curl(
    points: np.ndarray,
    indices: tuple[int, int, int, int],
    tuning: RetargeterTuning,
) -> float:
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
    bend_curl = _smoothstep(
        (
            0.62 * pip_bend
            + 0.38 * dip_bend
            - tuning.finger_curl_angle_offset
        )
        / tuning.finger_curl_angle_span
    )

    tip_distance = np.linalg.norm(tip - mcp)
    closure = 1.0 - float(tip_distance / (segment_lengths + 1e-6))
    closure_curl = _smoothstep(
        (closure - tuning.finger_curl_closure_offset)
        / tuning.finger_curl_closure_span
    )

    return max(bend_curl, closure_curl)


def _finger_splay(
    points: np.ndarray,
    finger: str,
    indices: tuple[int, int, int, int],
    tuning: RetargeterTuning,
    curl: float,
) -> float:
    mcp, pip, dip, _ = (points[index] for index in indices)
    proximal = pip - mcp
    secondary = 0.5 * (dip - mcp)
    direction = 0.75 * proximal + 0.25 * secondary
    forward = abs(float(direction[1])) + 0.35 * abs(float(direction[2])) + 1e-6
    lateral_angle = float(np.arctan2(direction[0], forward))
    neutral = float(tuning.finger_abad_neutral.get(finger, 0.0))
    splay = _deadzone(lateral_angle - neutral, tuning.finger_abad_deadzone)
    sign = float(tuning.finger_abad_sign.get(finger, 1.0))
    curl_damping = 1.0 - tuning.finger_abad_curl_damping * float(np.clip(curl, 0.0, 1.0))
    return float(
        np.clip(
            curl_damping * sign * tuning.finger_abad_gain * splay,
            -tuning.finger_abad_limit,
            tuning.finger_abad_limit,
        )
    )


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.arccos(cosine))


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _smooth_range(value: float, open_value: float, closed_value: float) -> float:
    span = closed_value - open_value
    if abs(span) < 1e-9:
        return 0.0
    return _smoothstep((value - open_value) / span)


def _deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    return value - np.copysign(deadzone, value)


def _blend(a: float, b: float, amount: float) -> float:
    amount = float(np.clip(amount, 0.0, 1.0))
    return a + amount * (b - a)
