"""DexPilot mode: the only retargeting method that controls fingertip geometry.

The analytic map reads joint *angles* and is provably blind to absolute
geometry, so it cannot place fingertips relative to each other. DexPilot
optimises six pairwise inter-fingertip vectors plus four palm-rooted ones,
which is exactly that missing capability. These tests pin the wiring and the
property that motivates it.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from midas_hand_retargeter import MidasHandRetargeter, RetargetProfile
from midas_hand_retargeter.config import DEXPILOT_MODE, MidasRetargeterConfig
from midas_hand_retargeter.constants import ABDUCTION_JOINT_NAMES
from midas_hand_retargeter.params import MODE_SECTIONS
from midas_hand_retargeter.postprocess import (
    finger_joint_targets_from_landmarks,
    thumb_joint_targets_from_landmarks,
)

from ._poses import hand_pose

#: A real 200-frame glove recording; see test_bounding_abduction_removes_curl_driven_lean
#: for why the synthetic poses cannot stand in for it here.
_TRACE_FRAMES = np.load(
    pathlib.Path(__file__).parent / "goldens" / "glove_trace_thumb.npz"
)["frames"]

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
    tip_landmarks = {4, 8, 12, 16}
    # Base-rooted vectors start at a palm landmark; the thumb's is rebased onto
    # the thumb CMC (landmark 1), the others stay at the wrist (landmark 0).
    base_rooted = [
        (o, t) for o, t in zip(origins, tasks, strict=True) if o not in tip_landmarks
    ]
    inter_finger = [
        (o, t) for o, t in zip(origins, tasks, strict=True) if o in tip_landmarks
    ]
    assert len(base_rooted) == 4
    assert len(inter_finger) == 6, "the pairwise terms are the whole point"

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
    # The thumb vector is rooted at the robot's CMC, so landmark 1 is now
    # load-bearing. Derive it from the robot the same way the tips are derived,
    # keeping this test fixture-free; leaving it at the origin would compare a
    # wrist-rooted human vector against a CMC-rooted robot one, a ~66 mm bias.
    robot.compute_forward_kinematics(np.asarray(source, dtype=float))
    palm_inverse = np.linalg.inv(robot.get_link_pose(robot.get_link_index("palm_base")))
    landmarks[1] = (
        palm_inverse @ robot.get_link_pose(robot.get_link_index("thumb_cmc_side"))
    )[:3, 3]

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


@needs_optimizer
def test_cartesian_modes_model_the_passive_four_bar_coupling():
    """The fingertip a Cartesian objective aims at must be the real one.

    With fixed_passive the DIP joints are held at 0 during FK, but the MIDAS
    four-bar linkage swings the distal phalanx with the PIP. The modelled
    fingertip is then up to 59 mm from where the real one goes (23 mm at PIP
    -0.3, 48 mm at -0.9, 59 mm at -1.45), which is larger than every other
    error in the pipeline.
    """

    from midas_hand_retargeter.coupling import (
        FIXED_PASSIVE_MODE,
        PIP_DIP_LOOKUP_MODE,
        LookupPassiveCoupling,
    )
    from midas_hand_retargeter.model import MIDAS_RIGHT_HAND

    for mode in ("dexpilot", "vector", "refine"):
        assert MidasRetargeterConfig(mode=mode).coupling_mode == PIP_DIP_LOOKUP_MODE
    # The analytic map never reads a passive joint, so it stays dependency-free.
    assert MidasRetargeterConfig().coupling_mode == FIXED_PASSIVE_MODE
    # An explicit choice is still honoured.
    assert (
        MidasRetargeterConfig(mode="dexpilot", coupling_mode=FIXED_PASSIVE_MODE).coupling_mode
        == FIXED_PASSIVE_MODE
    )

    # And the error being avoided is real and large.
    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    robot = retargeter.dex_retargeting.optimizer.robot
    model = MIDAS_RIGHT_HAND
    coupling = LookupPassiveCoupling(model)

    def index_tip(qpos):
        robot.compute_forward_kinematics(np.asarray(qpos, dtype=float))
        inverse = np.linalg.inv(robot.get_link_pose(robot.get_link_index("palm_base")))
        return (inverse @ robot.get_link_pose(robot.get_link_index("index_tip")))[:3, 3]

    qpos = np.zeros(19)
    qpos[model.index("index_pip_joint")] = -0.9
    qpos[model.index("index_mcp_pitch_joint")] = -1.2
    uncoupled = index_tip(qpos)
    coupled = index_tip(coupling.forward_qpos(qpos.copy()))
    assert np.linalg.norm(uncoupled - coupled) > 0.03, "coupling must move the tip"



@needs_optimizer
def test_abduction_is_bounded_by_default():
    """The default policy narrows abduction well inside the URDF's range."""

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    for name in ABDUCTION_JOINT_NAMES:
        assert retargeter._solver_joint_limits[name] == pytest.approx((-0.25, 0.25))
    limits = retargeter.dex_retargeting.joint_limits
    for name in ABDUCTION_JOINT_NAMES:
        lower, upper = limits[retargeter.active_joint_names.index(name)]
        assert (float(lower), float(upper)) == pytest.approx((-0.25, 0.25), abs=1e-6)


@needs_optimizer
def test_abduction_limit_never_widens_past_the_model():
    """It is a narrowing policy: a huge value must not exceed the URDF."""

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    retargeter.profile = retargeter.profile.with_values({"dexpilot.abduction_limit": 99.0})
    retargeter.retarget_landmarks(_TRACE_FRAMES[0])
    for name in ABDUCTION_JOINT_NAMES:
        model = retargeter._model.limits(name)
        lower, upper = retargeter._solver_joint_limits[name]
        assert lower >= model[0] - 1e-9 and upper <= model[1] + 1e-9


@needs_optimizer
def test_abduction_limit_is_live_tunable():
    """Moving the slider must reach the solver without a rebuild, and a zero
    must not hand nlopt equal bounds, which SLSQP rejects."""

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    retargeter.retarget_landmarks(_TRACE_FRAMES[0])

    retargeter.profile = retargeter.profile.with_values({"dexpilot.abduction_limit": 0.0})
    result = retargeter.retarget_landmarks(_TRACE_FRAMES[1])
    for name in ABDUCTION_JOINT_NAMES:
        lower, upper = retargeter._solver_joint_limits[name]
        assert lower < upper, "equal bounds are rejected by SLSQP"
        # nlopt's set_joint_limit adds its own 1e-3 epsilon on top.
        assert abs(result.active_joint_positions[name]) < 3e-3


@needs_optimizer
def test_bounding_abduction_removes_curl_driven_lean():
    """Curling must not swing the fingers sideways. Regression, on real data.

    This is "when I just curl, all three fingers lean toward the thumb": a
    human's fingertips converge as they curl, and the MIDAS fingers -- which
    curl in parallel planes -- can only imitate that convergence by abducting.
    Measured on a 30 s recording, doing so buys 0.5 mm of inter-fingertip
    accuracy and costs 0.77 rad of sideways swing.

    The fixture is a real glove trace precisely because a synthetic pose does
    not reproduce the effect: its fingers stay parallel, so there is no
    convergence for the solver to chase and the bug is invisible.
    """

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)

    def swing(limit):
        retargeter.profile = retargeter.profile.with_values(
            {"dexpilot.abduction_limit": limit}
        )
        retargeter.reset()
        seen = [[] for _ in ABDUCTION_JOINT_NAMES]
        for frame in _TRACE_FRAMES[::4]:
            active = retargeter.retarget_landmarks(frame).active_joint_positions
            for slot, name in zip(seen, ABDUCTION_JOINT_NAMES, strict=True):
                slot.append(active[name])
        return max(max(v) - min(v) for v in seen)

    assert swing(0.25) < 0.6 * swing(1.57)
