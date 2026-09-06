"""DexPilot mode: the only retargeting method that controls fingertip geometry.

The analytic map reads joint *angles* and is provably blind to absolute
geometry, so it cannot place fingertips relative to each other. DexPilot
optimises six pairwise inter-fingertip vectors plus four palm-rooted ones,
which is exactly that missing capability. These tests pin the wiring and the
property that motivates it.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from midas_hand_retargeter import MidasHandRetargeter, RetargetProfile
from midas_hand_retargeter.config import DEXPILOT_MODE, MidasRetargeterConfig
from midas_hand_retargeter.params import MODE_SECTIONS
from midas_hand_retargeter.postprocess import (
    finger_joint_targets_from_landmarks,
    thumb_joint_targets_from_landmarks,
)

from ._poses import hand_pose

needs_optimizer = pytest.mark.skipif(
    importlib.util.find_spec("dex_retargeting") is None,
    reason="requires the [vector] extra",
)

TIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip")
HUMAN_TIP = {"thumb_tip": 4, "index_tip": 8, "middle_tip": 12, "ring_tip": 16}


def test_analytic_is_blind_to_absolute_geometry():
    """The motivation for DexPilot, stated as a test.

    Scaling the human hand changes every fingertip position but leaves the
    analytic output untouched, so no analytic parameter can ever fix relative
    fingertip placement.
    """

    pose = hand_pose(curls=(0.7, 0.4, 0.9), splays=(0.2, 0.0, -0.2), thumb_oppose=0.6)

    def targets(landmarks):
        out = {}
        out.update(finger_joint_targets_from_landmarks(landmarks, RetargetProfile()))
        out.update(thumb_joint_targets_from_landmarks(landmarks, RetargetProfile()))
        return out

    base = targets(pose)
    for scale in (0.6, 1.7, 3.0):
        scaled = targets(pose * scale)
        drift = max(abs(scaled[k] - base[k]) for k in base)
        assert drift < 1e-5, f"analytic responded to hand size at {scale}x: {drift}"


def test_dexpilot_config_shape():
    config = MidasRetargeterConfig(mode=DEXPILOT_MODE)
    assert config.uses_optimizer is True
    # Pure optimizer: the analytic layer must not overwrite its solution.
    assert config.finger_postprocess is False
    assert config.thumb_postprocess is False

    payload = config.to_dex_config_dict()
    assert payload["type"] == "dexpilot"
    assert payload["wrist_link_name"] == "palm_base"
    assert payload["finger_tip_link_names"] == list(TIPS)


def test_ui_shows_only_the_sections_a_mode_reads():
    """A slider that silently does nothing is worse than no slider."""

    assert MODE_SECTIONS[DEXPILOT_MODE] == ("dexpilot",)
    assert "index" not in MODE_SECTIONS[DEXPILOT_MODE]
    assert "dexpilot" not in MODE_SECTIONS["analytic"]


@needs_optimizer
def test_objective_includes_pairwise_fingertip_vectors():
    """Six inter-fingertip pairs plus four palm-rooted, for a 4-finger hand."""

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    indices = np.asarray(retargeter._target_link_human_indices)
    assert indices.shape[1] == 10

    origins, tasks = indices[0], indices[1]
    palm_rooted = [(o, t) for o, t in zip(origins, tasks, strict=True) if o == 0]
    inter_finger = [(o, t) for o, t in zip(origins, tasks, strict=True) if o != 0]
    assert len(palm_rooted) == 4
    assert len(inter_finger) == 6, "the pairwise terms are the whole point"

    tip_landmarks = {4, 8, 12, 16}
    for origin, task in inter_finger:
        assert origin in tip_landmarks and task in tip_landmarks


@needs_optimizer
def test_solver_params_apply_live_without_a_rebuild():
    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    optimizer = retargeter.dex_retargeting.optimizer
    pose = hand_pose(curls=(0.4,) * 3, thumb_oppose=0.3)

    retargeter.profile = RetargetProfile().with_values(
        {"dexpilot.scaling_factor": 1.8, "dexpilot.eta1": 0.02, "dexpilot.eta2": 0.06}
    )
    retargeter.retarget_landmarks(pose)

    assert optimizer.scaling == pytest.approx(1.8)
    # eta1/eta2 are baked into a projection cache at construction, so the cache
    # has to be recomputed for them to take effect at all.
    np.testing.assert_allclose(
        optimizer.projected_dist, [0.02, 0.02, 0.02, 0.06, 0.06, 0.06]
    )


@needs_optimizer
def test_scaling_factor_actually_changes_the_pose():
    """It is the load-bearing knob in this mode; a dead slider would be useless."""

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    pose = hand_pose(curls=(0.35,) * 3, splays=(0.25, 0.0, -0.25), thumb_oppose=0.3)

    poses = {}
    for scaling in (1.0, 2.0):
        retargeter.profile = RetargetProfile().with_values(
            {"dexpilot.scaling_factor": scaling}
        )
        retargeter.reset()
        for _ in range(20):
            result = retargeter.retarget_landmarks(pose)
        poses[scaling] = result.active_vector()

    assert np.abs(poses[1.0] - poses[2.0]).max() > 0.1


@needs_optimizer
def test_recovers_reachable_fingertip_targets():
    """Self-consistency: feed back the robot's own fingertips, expect them back.

    Fixture-free — it does not depend on any synthetic human hand matching the
    robot's proportions, which is what makes it a real check of the objective.
    """

    # palm_frame_input=False: this test supplies fingertip positions already in
    # the robot's palm frame, and populates only the tip landmarks, so the
    # palm basis (built from the MCPs) would be degenerate.
    retargeter = MidasHandRetargeter.create(
        mode=DEXPILOT_MODE, palm_frame_input=False
    )
    retargeter.profile = RetargetProfile().with_values({"dexpilot.scaling_factor": 1.0})
    robot = retargeter.dex_retargeting.optimizer.robot

    def tips(qpos):
        robot.compute_forward_kinematics(np.asarray(qpos, dtype=float))
        inverse = np.linalg.inv(robot.get_link_pose(robot.get_link_index("palm_base")))
        return {
            name: (inverse @ robot.get_link_pose(robot.get_link_index(name)))[:3, 3]
            for name in TIPS
        }

    rng = np.random.default_rng(3)
    source = np.zeros(19)
    for name in retargeter.active_joint_names:
        lower, upper = retargeter._model.limits(name)
        source[retargeter._model.index(name)] = rng.uniform(lower, upper) * 0.6
    target = tips(source)

    landmarks = np.zeros((21, 3))
    for name, index in HUMAN_TIP.items():
        landmarks[index] = target[name]

    retargeter.reset()
    for _ in range(30):
        result = retargeter.retarget_landmarks(landmarks)
    got = tips(result.robot_qpos)

    tip_error = np.mean([np.linalg.norm(got[n] - target[n]) for n in TIPS])
    spacing_error = np.mean(
        [
            abs(np.linalg.norm(got[a] - got[b]) - np.linalg.norm(target[a] - target[b]))
            for i, a in enumerate(TIPS)
            for b in TIPS[i + 1 :]
        ]
    )
    assert tip_error < 0.010, f"tip error {tip_error*1000:.1f} mm"
    assert spacing_error < 0.010, f"inter-fingertip spacing error {spacing_error*1000:.1f} mm"


@needs_optimizer
def test_optimizer_modes_are_not_frozen_by_the_hold_logic():
    """Regression: the whole hand used to latch on frame 1 in solver modes.

    ``_hold_disabled_joints`` held every active joint absent from the analytic
    layer's targets. In vector/dexpilot there are no analytic targets at all,
    so the first frame's solution was held forever and the commanded pose never
    moved again — the sim looked completely dead while the solver underneath
    was tracking fine.
    """

    for mode in ("dexpilot", "vector"):
        retargeter = MidasHandRetargeter.create(mode=mode, palm_frame_input=False)
        retargeter.reset()

        commanded = []
        for step in range(12):
            amount = 1.3 * step / 11
            pose = hand_pose(curls=(amount,) * 3, thumb_curl=amount, thumb_oppose=0.8 * amount)
            commanded.append(retargeter.retarget_landmarks(pose).active_vector())

        spread = np.ptp(np.array(commanded), axis=0).max()
        assert spread > 0.1, f"{mode} output frozen (max spread {spread:.4f} rad)"


def test_hold_applies_only_to_digits_the_profile_disabled():
    """The narrow behaviour the hold logic is actually for."""

    retargeter = MidasHandRetargeter.create()  # analytic
    curled = hand_pose(curls=(1.2, 1.2, 1.2))
    for _ in range(30):
        before = retargeter.retarget_landmarks(curled)
    held = before.active_joint_positions["index_pip_joint"]

    retargeter.profile = retargeter.profile.with_values({"index.enabled": False})
    after = retargeter.retarget_landmarks(hand_pose(curls=(0.0, 0.0, 0.0)))

    # index is frozen...
    assert after.active_joint_positions["index_pip_joint"] == pytest.approx(held)
    # ...and nothing else is.
    assert after.active_joint_positions["middle_pip_joint"] > held + 0.1
    assert after.active_joint_positions["ring_pip_joint"] > held + 0.1


@needs_optimizer
def test_scaling_calibration_measures_the_operator_hand():
    """Zero-pose calibration for the knob this mode depends on most."""

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    assert retargeter.profile.dexpilot.scaling_factor == pytest.approx(1.15)

    # A deliberately small hand: the scale must come out proportionally larger.
    small = hand_pose() * 0.5
    retargeter.retarget_landmarks(small)
    small_scale = retargeter.calibrate_scaling_from_landmarks()

    big = hand_pose() * 1.5
    retargeter.retarget_landmarks(big)
    big_scale = retargeter.calibrate_scaling_from_landmarks()

    assert small_scale > big_scale, "a smaller hand needs a larger scale"
    assert retargeter.profile.dexpilot.scaling_factor == pytest.approx(big_scale)


@needs_optimizer
def test_scaling_calibration_needs_a_frame_first():
    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    with pytest.raises(RuntimeError, match="No landmark frame"):
        retargeter.calibrate_scaling_from_landmarks()


@needs_optimizer
def test_zero_pose_calibration_works_in_dexpilot_too():
    """The webcam path has had this since the start; solver modes need it too."""

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    pose = hand_pose(curls=(0.3,) * 3, thumb_oppose=0.2)
    for _ in range(20):
        before = retargeter.retarget_landmarks(pose)
    assert np.abs(before.active_vector()).max() > 0.05

    retargeter.calibrate_neutral_from_last_frame()
    after = retargeter.retarget_landmarks(pose)
    assert np.abs(after.active_vector()).max() < 0.05, "the held pose must become zero"

    # ...without collapsing the usable range.
    moved = retargeter.retarget_landmarks(hand_pose(curls=(1.2,) * 3, thumb_oppose=0.9))
    assert np.abs(moved.active_vector()).max() > 0.3
