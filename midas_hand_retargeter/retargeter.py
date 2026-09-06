"""High-level MIDAS vector-retargeting API."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from .config import DEXPILOT_MODE, MidasRetargeterConfig
from .constants import ACTIVE_JOINT_NAMES, HARDWARE_MOTOR_JOINT_NAMES
from .coupling import FIXED_PASSIVE_MODE, LookupPassiveCoupling
from .human import landmarks_to_vectors
from .params import RetargetProfile
from .postprocess import (
    JointTargetFilter,
    as_profile,
    finger_joint_targets_from_landmarks,
    landmarks_to_palm_frame,
    thumb_joint_targets_from_landmarks,
)

logger = logging.getLogger(__name__)


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
        self._model = self.config.hand_model
        self.active_joint_names = tuple(self.config.target_joint_names)

        self._retargeting = None
        self._kinematic_adaptor = None
        self._passive_coupling = None

        if self.config.uses_optimizer:
            self._build_optimizer()
        else:
            # Analytic path: the joint layout comes from the checked-in model,
            # so no URDF, no pinocchio and no torch are needed.
            self.robot_joint_names = self._model.joint_names
            self.fixed_joint_names = tuple(
                name for name in self.robot_joint_names if name not in set(self.active_joint_names)
            )
            if self.config.coupling_mode != FIXED_PASSIVE_MODE:
                self._passive_coupling = LookupPassiveCoupling(
                    self._model,
                    lookup_path=self.config.pip_dip_lookup_path,
                )

        self._joint_index_by_name = {
            name: index for index, name in enumerate(self.robot_joint_names)
        }
        self._fixed_qpos = self._build_fixed_qpos()
        self._profile = as_profile(self.config.tuning)
        # Cached so the two knobs that need a rebuild only pay for it on change.
        self._dexpilot_huber = self._profile.dexpilot.huber_delta
        self._dexpilot_etas = (
            self._profile.dexpilot.eta1,
            self._profile.dexpilot.eta2,
        )
        self._dexpilot_signature = None
        self._postprocess_filter = JointTargetFilter()
        self._neutral_joint_offsets: dict[str, float] = {}
        self._last_uncalibrated_active_joint_positions: dict[str, float] = {}
        self._last_landmarks: np.ndarray | None = None
        self.reset()

    def _build_optimizer(self) -> None:
        """Construct the dex-retargeting solver and its kinematic adaptor."""

        from .adaptor import build_midas_kinematic_adaptor

        self._retargeting = self.config.build_dex_config().build()
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
            self._retargeting.optimizer.set_kinematic_adaptor(self._kinematic_adaptor)

        self.robot_joint_names = tuple(self._retargeting.joint_names)
        self.fixed_joint_names = tuple(self._retargeting.optimizer.fixed_joint_names)
        if self.robot_joint_names != self._model.joint_names:
            raise RuntimeError(
                "Solver joint ordering does not match the checked-in HandModel. "
                "robot_qpos indexing is a public contract; regenerate "
                "model.MIDAS_RIGHT_JOINTS from the URDF.\n"
                f"  solver: {self.robot_joint_names}\n"
                f"  model:  {self._model.joint_names}"
            )

    @classmethod
    def create(cls, **config_overrides) -> MidasHandRetargeter:
        """Construct from keyword overrides accepted by ``MidasRetargeterConfig``."""

        return cls(MidasRetargeterConfig(**config_overrides))

    @property
    def dex_retargeting(self):
        """The underlying ``SeqRetargeting``, or ``None`` in analytic mode."""

        return self._retargeting

    def landmarks_to_vectors(self, landmarks: np.ndarray) -> np.ndarray:
        """Convert 21x3 human landmarks into the configured vector objective."""

        return landmarks_to_vectors(
            landmarks,
            target_link_human_indices=self._target_link_human_indices,
        )

    @property
    def _target_link_human_indices(self):
        """Landmark pairs for the active objective.

        DexPilot derives its own (wrist + the four fingertips, and every
        pairwise fingertip combination); overriding them with the vector-mode
        pairs would silently destroy the inter-finger structure that is the
        whole reason to use it.
        """

        if self._retargeting is not None and self.config.mode == DEXPILOT_MODE:
            return self._retargeting.optimizer.target_link_human_indices
        return self.config.target_link_human_indices

    def retarget_landmarks(self, landmarks: np.ndarray) -> RetargetingResult:
        """Retarget one 21x3 human landmark frame into MIDAS joint targets."""

        self._last_landmarks = landmarks
        if self.config.mode == DEXPILOT_MODE and self.config.palm_frame_input:
            # See postprocess.landmarks_to_palm_frame: without this the solver
            # is handed targets rotated away from the robot's frame and rails.
            landmarks = landmarks_to_palm_frame(landmarks)
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
        if self.config.mode == DEXPILOT_MODE:
            self._apply_dexpilot_params()
        if self._retargeting is not None:
            robot_qpos = self._retargeting.retarget(vectors, fixed_qpos=self._fixed_qpos)
        else:
            # Analytic mode: the analytic layer writes every actuated joint, so
            # there is nothing to solve. Passive slots start at their fixed
            # values and are recomputed by the coupling below if enabled.
            robot_qpos = np.zeros(len(self.robot_joint_names), dtype=np.float32)
            for name, value in self.config.passive_fixed_qpos.items():
                index = self._joint_index_by_name.get(name)
                if index is not None:
                    robot_qpos[index] = value
        if landmarks is not None:
            robot_qpos = self._apply_landmark_postprocess(robot_qpos, landmarks)
        self._last_uncalibrated_active_joint_positions = self._active_positions_from_qpos(
            robot_qpos
        )
        self._sync_optimizer_warm_start(robot_qpos)
        robot_qpos = self._apply_neutral_offsets(robot_qpos)
        robot_qpos = self._apply_kinematic_adaptor(robot_qpos)
        return self._make_result(robot_qpos, vectors)

    def set_qpos(self, robot_qpos: np.ndarray) -> None:
        """Warm-start the underlying optimizer from a full robot qpos vector."""

        if self._retargeting is None:
            raise RuntimeError(
                f"set_qpos() needs the optimizer, but mode={self.config.mode!r} does not build one."
            )
        self._retargeting.set_qpos(np.asarray(robot_qpos, dtype=np.float32))

    def reset(self) -> None:
        """Reset optimizer warm-start state and stateful postprocess filters."""

        self._postprocess_filter.reset()
        self._last_uncalibrated_active_joint_positions = {}
        if self._retargeting is None:
            return
        # Order matters: SeqRetargeting.reset() sets last_qpos to the joint
        # mid-range, so restore the intended zero warm start afterwards.
        self._retargeting.reset()
        self._retargeting.set_qpos(np.zeros(len(self.robot_joint_names), dtype=np.float32))
        if getattr(self._retargeting, "filter", None) is not None:
            self._retargeting.filter.reset()

    @property
    def profile(self) -> RetargetProfile:
        """The live analytic tuning profile."""

        return self._profile

    @profile.setter
    def profile(self, profile: RetargetProfile) -> None:
        """Swap the whole profile atomically, for live tuning.

        Assigning one frozen object means a concurrent reader sees either the
        old profile or the new one, never a mix of fields.
        """

        self._profile = as_profile(profile)

    def set_neutral_offsets(self, offsets: Mapping[str, float]) -> None:
        """Restore a saved neutral calibration (e.g. from a preset)."""

        unknown = set(offsets) - set(self._joint_index_by_name)
        if unknown:
            raise KeyError(f"Unknown joints in neutral offsets: {sorted(unknown)}")
        self._neutral_joint_offsets = {k: float(v) for k, v in offsets.items()}

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

    def _apply_dexpilot_params(self) -> None:
        """Push the live DexPilot knobs onto the optimizer before solving.

        All of these are read per-solve, so live tuning needs no rebuild. eta1
        and eta2 are the exception: they are baked into a projection cache at
        construction, so that cache is recomputed when they change.
        """

        params = self._profile.dexpilot
        optimizer = self._retargeting.optimizer

        # The norm_delta temporal regularizer anchors each solve to the previous
        # one, so after a parameter change the old solution is a bad anchor and
        # the solver barely moves — a scaling slider would feel dead. Drop the
        # warm start when the parameters themselves change (not every frame).
        signature = (
            params.scaling_factor, params.huber_delta, params.norm_delta,
            params.project_dist, params.escape_dist, params.eta1, params.eta2,
        )
        if self._dexpilot_signature is not None and signature != self._dexpilot_signature:
            self._retargeting.set_qpos(
                np.zeros(len(self.robot_joint_names), dtype=np.float32)
            )
        self._dexpilot_signature = signature

        optimizer.scaling = float(params.scaling_factor)
        optimizer.norm_delta = float(params.norm_delta)
        optimizer.project_dist = float(params.project_dist)
        optimizer.escape_dist = float(params.escape_dist)

        if self._dexpilot_huber != params.huber_delta:
            import torch

            optimizer.huber_loss = torch.nn.SmoothL1Loss(
                beta=float(params.huber_delta), reduction="none"
            )
            self._dexpilot_huber = params.huber_delta

        if self._dexpilot_etas != (params.eta1, params.eta2):
            optimizer.eta1 = float(params.eta1)
            optimizer.eta2 = float(params.eta2)
            (
                optimizer.projected,
                optimizer.s2_project_index_origin,
                optimizer.s2_project_index_task,
                optimizer.projected_dist,
            ) = optimizer.set_dexpilot_cache(
                optimizer.num_fingers, float(params.eta1), float(params.eta2)
            )
            self._dexpilot_etas = (params.eta1, params.eta2)

    def _sync_optimizer_warm_start(self, robot_qpos: np.ndarray) -> None:
        """Anchor the solver's temporal regularizer to the commanded pose.

        ``SeqRetargeting.retarget`` stores its own raw solution as
        ``last_qpos``, which the ``normal_delta`` regularizer then penalizes
        deviation from. But in ``refine`` mode the analytic layer overwrites
        every actuated joint afterwards, so the solver was being pulled toward
        a pose that was never commanded — measured up to 1.5 rad away over a
        normal closing sweep.

        Sync from the post-analytic, PRE-neutral vector: that is the pose
        actually sent, expressed in solver space. Neutral recentering is an
        operator-facing remap that corresponds to no physical human pose, so
        feeding it back would anchor the regularizer to something nobody is
        holding.
        """

        if self._retargeting is None:
            return
        self._retargeting.set_qpos(np.asarray(robot_qpos, dtype=np.float32))

    def calibrate_scaling_from_landmarks(
        self, landmarks: np.ndarray | None = None
    ) -> float:
        """Set the DexPilot scaling from a held OPEN-hand pose.

        The analytic map ignores hand size entirely; DexPilot does not, and
        ``scaling_factor`` is the difference between fingers that never close
        and fingers that curl into their limits. This measures it instead of
        making the operator guess: hold the hand flat and open, and the scale
        becomes the robot's reach over yours.

        Uses the median over index/middle/ring. The thumb is deliberately
        excluded: the MIDAS thumb reaches ~111 mm at its zero pose against a
        human thumb's ~125 mm, the opposite correction the fingers need, so
        including it biases the fit for every digit. That proportion mismatch
        is real and one global scale cannot fix it.

        Returns the scaling that was applied.
        """

        if landmarks is None:
            landmarks = self._last_landmarks
        if landmarks is None:
            raise RuntimeError("No landmark frame available to calibrate from")

        points = landmarks_to_palm_frame(landmarks)
        ratios = []
        for finger, tip_index in (("index", 8), ("middle", 12), ("ring", 16)):
            human = float(np.linalg.norm(points[tip_index]))
            robot = self._open_pose_reach(f"{finger}_tip")
            if human > 1e-3 and robot > 0:
                ratios.append(robot / human)
        if not ratios:
            raise RuntimeError("Could not measure hand size from this frame")

        scaling = float(np.median(ratios))
        self._profile = replace(
            self._profile, dexpilot=replace(self._profile.dexpilot, scaling_factor=scaling)
        )
        logger.info("Calibrated DexPilot scaling_factor to %.3f", scaling)
        return scaling

    def _open_pose_reach(self, link_name: str) -> float:
        """Palm-to-tip distance for ``link_name`` with the robot fully open."""

        if self._retargeting is None:
            return 0.0
        robot = self._retargeting.optimizer.robot
        robot.compute_forward_kinematics(np.zeros(robot.dof))
        palm = robot.get_link_pose(robot.get_link_index(self.config.wrist_link_name))
        tip = robot.get_link_pose(robot.get_link_index(link_name))
        return float(np.linalg.norm((np.linalg.inv(palm) @ tip)[:3, 3]))

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

        coupling = self._kinematic_adaptor or self._passive_coupling
        if coupling is None:
            return robot_qpos
        return np.asarray(
            coupling.forward_qpos(np.asarray(robot_qpos).copy()),
            dtype=np.float32,
        )

    def _as_ref_vectors(self, ref_vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(ref_vectors, dtype=np.float32)
        expected_shape = (len(np.asarray(self._target_link_human_indices)[0]), 3)
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

        # Read the live profile exactly once per frame and thread it through,
        # so a concurrent tuning edit cannot change parameters mid-frame.
        profile = self._profile

        joint_targets: dict[str, float] = {}
        if self.config.finger_postprocess:
            joint_targets.update(finger_joint_targets_from_landmarks(landmarks, profile))
        if self.config.thumb_postprocess:
            joint_targets.update(thumb_joint_targets_from_landmarks(landmarks, profile))
        if not joint_targets:
            # Optimizer-only modes (vector, dexpilot) run no analytic layer, so
            # there is nothing to apply and nothing to hold.
            return robot_qpos

        qpos = self._apply_joint_targets(robot_qpos, joint_targets, profile)
        return self._hold_disabled_joints(qpos, joint_targets, profile)

    def _hold_disabled_joints(
        self,
        robot_qpos: np.ndarray,
        joint_targets: Mapping[str, float],
        profile: RetargetProfile,
    ) -> np.ndarray:
        """Freeze digits the profile has disabled at their last command.

        A disabled digit emits no target. Without this it would fall through to
        the zero-initialised qpos and command 0.0 rad - flinging the finger
        fully open mid-teleop, which on hardware is a real and surprising
        motion. Held values bypass the smoothing filter: they are not new
        measurements.

        Only digits that are *explicitly disabled* are held. This previously
        held every active joint absent from ``joint_targets``, which silently
        froze the whole hand in the optimizer-only modes: with no analytic
        targets at all, the first frame's solution was latched forever and the
        commanded pose never moved again.
        """

        disabled = {name for name, params in profile.fingers() if not params.enabled}
        if not profile.thumb.enabled:
            disabled.add("thumb")
        if not disabled:
            return robot_qpos

        previous = self._last_uncalibrated_active_joint_positions
        if not previous:
            return robot_qpos

        qpos = robot_qpos
        for name in self.active_joint_names:
            digit = "thumb" if name.startswith("thumb_") else name.partition("_")[0]
            if digit not in disabled or name in joint_targets or name not in previous:
                continue
            index = self._joint_index_by_name.get(name)
            if index is not None:
                qpos[index] = previous[name]
        return qpos

    def _apply_joint_targets(
        self,
        robot_qpos: np.ndarray,
        joint_targets: Mapping[str, float],
        profile: RetargetProfile,
    ) -> np.ndarray:
        qpos = np.asarray(robot_qpos, dtype=np.float32).copy()
        for joint_name, value in joint_targets.items():
            joint_index = self._joint_index_by_name.get(joint_name)
            if joint_index is None:
                logger.warning(
                    "Dropping target for unknown joint %r (known joints: %s)",
                    joint_name,
                    sorted(self._joint_index_by_name),
                )
                continue
            filter_alpha = self._postprocess_filter_alpha(joint_name, profile)
            if filter_alpha is not None:
                value = self._postprocess_filter.update(
                    joint_name,
                    value,
                    filter_alpha,
                )
            qpos[joint_index] = self._clip_joint(joint_name, value)
        return qpos

    def _postprocess_filter_alpha(self, joint_name: str, profile: RetargetProfile) -> float | None:
        """Low-pass alpha for one landmark-derived target, per digit."""

        if joint_name.startswith("thumb_"):
            return profile.thumb.smoothing_alpha
        finger, _, _ = joint_name.partition("_")
        if joint_name.endswith(("_mcp_abad_joint", "_mcp_pitch_joint", "_pip_joint")):
            return profile.finger(finger).smoothing_alpha
        return None

    def _clip_joint(self, joint_name: str, value: float) -> float:
        return self._model.clip(joint_name, value)

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
        lower, upper = self._model.limits(joint_name)
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
            name: float(robot_qpos[index]) for index, name in enumerate(self.robot_joint_names)
        }
        active = {name: qpos_by_name[name] for name in self.active_joint_names}
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
