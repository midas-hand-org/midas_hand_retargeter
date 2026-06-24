"""High-level MIDAS vector-retargeting API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .adaptor import build_midas_kinematic_adaptor
from .config import MidasRetargeterConfig
from .constants import ACTIVE_JOINT_NAMES, HARDWARE_MOTOR_JOINT_NAMES
from .human import landmarks_to_vectors
from .postprocess import (
    JointTargetFilter,
    finger_joint_targets_from_landmarks,
    thumb_joint_targets_from_landmarks,
)


@dataclass(frozen=True)
class RetargetingResult:
    """Retargeted joint command in several useful output layouts.

    ``robot_qpos`` is the full Pinocchio/dex-retargeting joint vector. The
    dictionaries and helper vectors below are stable MIDAS-facing views used by
    MuJoCo and hardware callers.
    """

    robot_qpos: np.ndarray
    robot_joint_names: tuple[str, ...]
    ref_vectors: np.ndarray
    active_joint_positions: dict[str, float]
    hardware_motor_positions: np.ndarray
    fixed_joint_positions: dict[str, float]

    def active_vector(self, joint_names=ACTIVE_JOINT_NAMES) -> np.ndarray:
        """Return active joint targets in a caller-selected joint order."""

        return np.asarray(
            [self.active_joint_positions[name] for name in joint_names],
            dtype=np.float32,
        )

    def mujoco_control_dict(self) -> dict[str, float]:
        """Return ``{joint_name: target_position}`` for MuJoCo actuators."""

        return dict(self.active_joint_positions)


class MidasHandRetargeter:
    """MIDAS-facing wrapper around the installed ``dex_retargeting`` package.

    This class keeps upstream optimization separate from MIDAS-specific output
    conventions:

    - ``dex_retargeting`` solves the vector objective.
    - passive DIP/linkage joints are fixed in option 1, or coupled through the
      MIDAS PIP-DIP lookup adaptor in option 2.
    - optional postprocess heuristics refine MIDAS finger/thumb active joints.
    - outputs are packaged for simulation or hardware callers.
    """

    def __init__(self, config: MidasRetargeterConfig | None = None):
        self.config = config or MidasRetargeterConfig()
        dex_config = self.config.build_dex_config()
        self._retargeting = dex_config.build()
        self._kinematic_adaptor = build_midas_kinematic_adaptor(
            coupling_mode=self.config.coupling_mode,
            robot=self._retargeting.optimizer.robot,
            target_joint_names=self.config.target_joint_names,
            lookup_path=self.config.pip_dip_lookup_path,
        )
        if self._kinematic_adaptor is not None:
            if self._retargeting.optimizer.adaptor is not None:
                raise NotImplementedError(
                    "MIDAS PIP-DIP coupling cannot yet be composed with another "
                    "dex-retargeting kinematic adaptor."
                )
            self._retargeting.optimizer.set_kinematic_adaptor(
                self._kinematic_adaptor
            )

        self.robot_joint_names = tuple(self._retargeting.joint_names)
        self.active_joint_names = tuple(self.config.target_joint_names)
        self.fixed_joint_names = tuple(self._retargeting.optimizer.fixed_joint_names)
        self._joint_index_by_name = {
            name: index for index, name in enumerate(self.robot_joint_names)
        }
        self._fixed_qpos = self._build_fixed_qpos()
        self._postprocess_filter = JointTargetFilter()
        self._neutral_joint_offsets: dict[str, float] = {}
        self._last_uncalibrated_active_joint_positions: dict[str, float] = {}
        self.reset()

    @classmethod
    def create(cls, **config_overrides) -> "MidasHandRetargeter":
        """Construct from keyword overrides accepted by ``MidasRetargeterConfig``."""

        return cls(MidasRetargeterConfig(**config_overrides))

    @property
    def dex_retargeting(self):
        """Expose the underlying ``dex_retargeting.SeqRetargeting`` object."""

        return self._retargeting

    def landmarks_to_vectors(self, landmarks: np.ndarray) -> np.ndarray:
        """Convert 21x3 human landmarks into the configured vector objective."""

        return landmarks_to_vectors(
            landmarks,
            target_link_human_indices=self.config.target_link_human_indices,
        )

    def retarget_landmarks(self, landmarks: np.ndarray) -> RetargetingResult:
        """Retarget one 21x3 human landmark frame into MIDAS joint targets."""

        return self.retarget_vectors(
            self.landmarks_to_vectors(landmarks),
            landmarks=landmarks,
        )

    def retarget_vectors(
        self,
        ref_vectors: np.ndarray,
        *,
        landmarks: np.ndarray | None = None,
    ) -> RetargetingResult:
        """Retarget one vector objective frame.

        ``landmarks`` is optional because callers may already have vectors. When
        it is supplied, MIDAS-specific postprocess heuristics can refine active
        finger curl, finger ab/ad, and thumb joints. These heuristics are
        controlled by ``MidasRetargeterConfig.tuning``.
        """

        vectors = self._as_ref_vectors(ref_vectors)
        robot_qpos = self._retargeting.retarget(vectors, fixed_qpos=self._fixed_qpos)
        if landmarks is not None:
            robot_qpos = self._apply_landmark_postprocess(robot_qpos, landmarks)
        self._last_uncalibrated_active_joint_positions = self._active_positions_from_qpos(
            robot_qpos
        )
        robot_qpos = self._apply_neutral_offsets(robot_qpos)
        robot_qpos = self._apply_kinematic_adaptor(robot_qpos)
        return self._make_result(robot_qpos, vectors)

    def set_qpos(self, robot_qpos: np.ndarray) -> None:
        """Warm-start the underlying optimizer from a full robot qpos vector."""

        self._retargeting.set_qpos(np.asarray(robot_qpos, dtype=np.float32))

    def reset(self) -> None:
        """Reset optimizer warm-start state and stateful postprocess filters."""

        self._retargeting.set_qpos(
            np.zeros(len(self.robot_joint_names), dtype=np.float32)
        )
        self._postprocess_filter.reset()

    @property
    def neutral_joint_offsets(self) -> dict[str, float]:
        """Return raw active-joint targets captured as the neutral command pose."""

        return dict(self._neutral_joint_offsets)

    def clear_neutral_offsets(self) -> None:
        """Remove the current retargeter neutral calibration."""

        self._neutral_joint_offsets.clear()

    def calibrate_neutral_from_last_frame(
        self,
        joint_names: tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        """Use the latest uncalibrated active targets as the new neutral pose.

        Call this while the human hand is held in the pose that should command
        MIDAS zero. Future retargeting recenters around this pose and rescales
        each side to keep the original robot joint limits reachable. The
        calibration lives in the retargeter layer, so it affects print, MuJoCo,
        and hardware backends consistently.
        """

        if not self._last_uncalibrated_active_joint_positions:
            raise RuntimeError("No retargeting frame is available for neutral calibration")
        names = joint_names or self.active_joint_names
        self._neutral_joint_offsets = {
            name: self._last_uncalibrated_active_joint_positions[name]
            for name in names
            if name in self._last_uncalibrated_active_joint_positions
        }
        return self.neutral_joint_offsets

    def _build_fixed_qpos(self) -> np.ndarray:
        return np.asarray(
            [
                self.config.passive_fixed_qpos.get(joint_name, 0.0)
                for joint_name in self.fixed_joint_names
            ],
            dtype=np.float32,
        )

    def _apply_kinematic_adaptor(self, robot_qpos: np.ndarray) -> np.ndarray:
        """Recompute passive joints after postprocess/neutral active edits."""

        if self._kinematic_adaptor is None:
            return robot_qpos
        return np.asarray(
            self._kinematic_adaptor.forward_qpos(np.asarray(robot_qpos).copy()),
            dtype=np.float32,
        )

    def _as_ref_vectors(self, ref_vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(ref_vectors, dtype=np.float32)
        expected_shape = (len(self.config.target_task_link_names), 3)
        if vectors.shape != expected_shape:
            raise ValueError(f"Expected ref_vectors shape {expected_shape}, got {vectors.shape}")
        return vectors

    def _apply_landmark_postprocess(
        self,
        robot_qpos: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """Apply the optional landmark-derived MIDAS correction layer.

        The vector optimizer remains the primary retargeter. This layer exists
        for MIDAS-specific real-time behavior that is difficult to express in
        the generic vector objective, especially passive-DIP finger curl and
        thumb opposition. Tune it through ``RetargeterTuning``.
        """

        joint_targets: dict[str, float] = {}
        per_joint_alphas: dict[str, float] = {}
        if self.config.finger_postprocess:
            finger_targets, finger_alphas = finger_joint_targets_from_landmarks(
                landmarks, self.config.tuning
            )
            joint_targets.update(finger_targets)
            per_joint_alphas.update(finger_alphas)
        if self.config.thumb_postprocess:
            joint_targets.update(
                thumb_joint_targets_from_landmarks(landmarks, self.config.tuning)
            )
        return self._apply_joint_targets(robot_qpos, joint_targets, per_joint_alphas)

    def _apply_joint_targets(
        self,
        robot_qpos: np.ndarray,
        joint_targets: Mapping[str, float],
        per_joint_alphas: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        qpos = np.asarray(robot_qpos, dtype=np.float32).copy()
        for joint_name, value in joint_targets.items():
            joint_index = self._joint_index_by_name.get(joint_name)
            if joint_index is None:
                continue
            if per_joint_alphas and joint_name in per_joint_alphas:
                filter_alpha = per_joint_alphas[joint_name]
            else:
                filter_alpha = self._postprocess_filter_alpha(joint_name)
            if filter_alpha is not None:
                value = self._postprocess_filter.update(
                    joint_name,
                    value,
                    filter_alpha,
                )
            qpos[joint_index] = self._clip_joint(joint_name, value)
        return qpos

    def _postprocess_filter_alpha(self, joint_name: str) -> float | None:
        """Return the low-pass alpha for landmark-derived postprocess targets."""

        if joint_name.endswith((
            "_mcp_abad_joint",
            "_mcp_pitch_joint",
            "_pip_joint",
        )):
            return self.config.tuning.finger_smoothing_alpha
        if joint_name in {
            "thumb_cmc_roll_joint",
            "thumb_cmc_side_joint",
            "thumb_mcp_joint",
            "thumb_dip_joint",
        }:
            return self.config.tuning.thumb_smoothing_alpha
        return None

    def _clip_joint(self, joint_name: str, value: float) -> float:
        joint_index = self._joint_index_by_name[joint_name]
        lower, upper = self._retargeting.optimizer.robot.joint_limits[joint_index]
        return float(np.clip(value, lower, upper))

    def _apply_neutral_offsets(self, robot_qpos: np.ndarray) -> np.ndarray:
        """Recenter calibrated joints while preserving their full motion limits.

        A simple subtraction makes the captured neutral pose command zero, but
        it also shrinks one side of the range. For example, a thumb CMC roll
        command with limits ``[0, 2.15]`` and neutral ``1.2`` would only reach
        ``0.95`` after subtraction. This piecewise remap instead keeps neutral
        at zero and maps the original lower/upper limits back to themselves.
        """

        qpos = np.asarray(robot_qpos, dtype=np.float32).copy()
        for joint_name, offset in self._neutral_joint_offsets.items():
            joint_index = self._joint_index_by_name.get(joint_name)
            if joint_index is None:
                continue
            qpos[joint_index] = self._neutral_calibrated_value(
                joint_name,
                float(qpos[joint_index]),
                offset,
            )
        return qpos

    def _neutral_calibrated_value(
        self,
        joint_name: str,
        raw_value: float,
        neutral_value: float,
    ) -> float:
        joint_index = self._joint_index_by_name[joint_name]
        lower, upper = self._retargeting.optimizer.robot.joint_limits[joint_index]
        lower = float(lower)
        upper = float(upper)
        raw_value = float(np.clip(raw_value, lower, upper))
        neutral_value = float(np.clip(neutral_value, lower, upper))

        if raw_value >= neutral_value:
            if upper <= neutral_value:
                return self._clip_joint(joint_name, 0.0)
            value = (raw_value - neutral_value) * upper / (upper - neutral_value)
        else:
            if neutral_value <= lower:
                return self._clip_joint(joint_name, 0.0)
            value = (raw_value - neutral_value) * (-lower) / (neutral_value - lower)
        return self._clip_joint(joint_name, value)

    def _active_positions_from_qpos(self, robot_qpos: np.ndarray) -> dict[str, float]:
        return {
            name: float(robot_qpos[self._joint_index_by_name[name]])
            for name in self.active_joint_names
        }

    def _make_result(self, robot_qpos: np.ndarray, ref_vectors: np.ndarray) -> RetargetingResult:
        qpos_by_name = {
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
        fixed = {name: qpos_by_name[name] for name in self.fixed_joint_names}
        return RetargetingResult(
            robot_qpos=np.asarray(robot_qpos, dtype=np.float32),
            robot_joint_names=self.robot_joint_names,
            ref_vectors=np.asarray(ref_vectors, dtype=np.float32),
            active_joint_positions=active,
            hardware_motor_positions=hardware,
            fixed_joint_positions=fixed,
        )
