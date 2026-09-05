from types import SimpleNamespace

import numpy as np

from midas_hand_retargeter.postprocess import (
    finger_joint_targets_from_landmarks,
    thumb_joint_targets_from_landmarks,
)
from midas_hand_retargeter.constants import HARDWARE_MOTOR_JOINT_NAMES
from midas_hand_retargeter.model import HandModel
from midas_hand_retargeter.params import RetargetProfile
from midas_hand_retargeter.retargeter import MidasHandRetargeter
from midas_hand_retargeter.tuning import RetargeterTuning


def _straight_landmarks() -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    for base, x in ((5, -0.035), (9, 0.0), (13, 0.035)):
        landmarks[base] = [x, 0.035, 0.0]
        landmarks[base + 1] = [x, 0.060, 0.0]
        landmarks[base + 2] = [x, 0.083, 0.0]
        landmarks[base + 3] = [x, 0.105, 0.0]
    # Thumb: straight (colinear segments) and in-plane (z=0) so MCP/DIP flexion
    # and CMC roll are zero. The proximal direction sits at the tuned neutral
    # side angle (THUMB_CMC_SIDE_NEUTRAL_ANGLE = -0.3 rad, i.e. x:y ~= -0.31) so
    # a relaxed thumb maps to zero CMC side. (The previous fixture pointed the
    # thumb at ~-0.84 rad — the pre-87e2061 neutral — so it no longer read as
    # neutral after the pinching retune.)
    landmarks[1] = [-0.045, 0.020, 0.0]
    landmarks[2] = [-0.052, 0.042, 0.0]
    landmarks[3] = [-0.059, 0.064, 0.0]
    landmarks[4] = [-0.066, 0.086, 0.0]
    return landmarks


def _curled_landmarks() -> np.ndarray:
    landmarks = _straight_landmarks()
    for base, x in ((5, -0.035), (9, 0.0), (13, 0.035)):
        landmarks[base] = [x, 0.035, 0.0]
        landmarks[base + 1] = [x, 0.057, 0.0]
        landmarks[base + 2] = [x, 0.064, -0.022]
        landmarks[base + 3] = [x, 0.046, -0.039]
    return landmarks


def test_straight_hand_maps_to_zero_active_targets():
    finger_targets = finger_joint_targets_from_landmarks(_straight_landmarks())
    thumb_targets = thumb_joint_targets_from_landmarks(_straight_landmarks())

    for value in finger_targets.values():
        assert value == 0.0
    for value in thumb_targets.values():
        assert value == 0.0


def test_finger_postprocess_curls_mcp_and_pip():
    open_targets = finger_joint_targets_from_landmarks(_straight_landmarks())
    curled_targets = finger_joint_targets_from_landmarks(_curled_landmarks())

    for finger in ("index", "middle", "ring"):
        assert curled_targets[f"{finger}_mcp_pitch_joint"] < open_targets[f"{finger}_mcp_pitch_joint"]
        assert curled_targets[f"{finger}_pip_joint"] < open_targets[f"{finger}_pip_joint"]


def test_finger_postprocess_keeps_abduction_small():
    curled_targets = finger_joint_targets_from_landmarks(_curled_landmarks())

    assert -0.25 <= curled_targets["index_mcp_abad_joint"] <= 0.0
    assert curled_targets["middle_mcp_abad_joint"] == 0.0
    assert 0.0 <= curled_targets["ring_mcp_abad_joint"] <= 0.25


def test_finger_abduction_responds_to_lateral_splay():
    landmarks = _straight_landmarks()
    landmarks[6] = landmarks[5] + [-0.018, 0.022, 0.0]
    landmarks[10] = landmarks[9] + [0.010, 0.025, 0.0]
    landmarks[14] = landmarks[13] + [0.018, 0.022, 0.0]

    targets = finger_joint_targets_from_landmarks(landmarks)

    assert targets["index_mcp_abad_joint"] < -0.05
    assert targets["middle_mcp_abad_joint"] > 0.03
    assert targets["ring_mcp_abad_joint"] > 0.05


def test_finger_abduction_is_palm_frame_local():
    landmarks = _straight_landmarks()
    landmarks[6] = landmarks[5] + [-0.018, 0.022, 0.0]
    landmarks[10] = landmarks[9] + [0.010, 0.025, 0.0]
    landmarks[14] = landmarks[13] + [0.018, 0.022, 0.0]
    angle = np.deg2rad(32.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    targets = finger_joint_targets_from_landmarks(landmarks)
    rotated_targets = finger_joint_targets_from_landmarks(landmarks @ rotation.T)

    for finger in ("index", "middle", "ring"):
        assert np.isclose(
            rotated_targets[f"{finger}_mcp_abad_joint"],
            targets[f"{finger}_mcp_abad_joint"],
            atol=1e-6,
        )


def test_thumb_cmc_ignores_index_tip_motion():
    open_landmarks = _straight_landmarks()
    index_moved_landmarks = _straight_landmarks()
    index_moved_landmarks[8] = index_moved_landmarks[4] + [0.005, 0.0, 0.0]

    open_targets = thumb_joint_targets_from_landmarks(open_landmarks)
    index_moved_targets = thumb_joint_targets_from_landmarks(index_moved_landmarks)

    assert index_moved_targets["thumb_cmc_roll_joint"] == open_targets["thumb_cmc_roll_joint"]
    assert index_moved_targets["thumb_cmc_side_joint"] == open_targets["thumb_cmc_side_joint"]


def test_thumb_roll_responds_to_thumb_opposition_not_side_sweep():
    open_landmarks = _straight_landmarks()
    opposed_landmarks = _straight_landmarks()
    mirrored_opposed_landmarks = _straight_landmarks()
    side_landmarks = _straight_landmarks()

    opposed_landmarks[[2, 3, 4], 2] -= 0.040
    mirrored_opposed_landmarks[[2, 3, 4], 2] += 0.040
    side_landmarks[2] = [-0.035, 0.038, 0.0]
    side_landmarks[3] = [-0.020, 0.056, 0.0]
    side_landmarks[4] = [-0.005, 0.074, 0.0]

    open_targets = thumb_joint_targets_from_landmarks(open_landmarks)
    opposed_targets = thumb_joint_targets_from_landmarks(opposed_landmarks)
    mirrored_opposed_targets = thumb_joint_targets_from_landmarks(mirrored_opposed_landmarks)
    side_targets = thumb_joint_targets_from_landmarks(side_landmarks)

    assert opposed_targets["thumb_cmc_roll_joint"] > open_targets["thumb_cmc_roll_joint"]
    assert mirrored_opposed_targets["thumb_cmc_roll_joint"] > open_targets["thumb_cmc_roll_joint"]
    assert side_targets["thumb_cmc_roll_joint"] == open_targets["thumb_cmc_roll_joint"]


def test_thumb_side_responds_to_in_plane_sweep():
    open_landmarks = _straight_landmarks()
    side_landmarks = _straight_landmarks()
    side_landmarks[2] = [-0.035, 0.038, 0.0]
    side_landmarks[3] = [-0.020, 0.056, 0.0]
    side_landmarks[4] = [-0.005, 0.074, 0.0]

    open_targets = thumb_joint_targets_from_landmarks(open_landmarks)
    side_targets = thumb_joint_targets_from_landmarks(side_landmarks)

    assert side_targets["thumb_cmc_side_joint"] < open_targets["thumb_cmc_side_joint"]


def test_thumb_cmc_side_and_roll_gains_are_independent():
    side_landmarks = _straight_landmarks()
    side_landmarks[2] = [-0.035, 0.038, 0.0]
    side_landmarks[3] = [-0.020, 0.056, 0.0]
    side_landmarks[4] = [-0.005, 0.074, 0.0]

    opposed_landmarks = _straight_landmarks()
    opposed_landmarks[[2, 3, 4], 2] -= 0.030

    side_low = thumb_joint_targets_from_landmarks(
        side_landmarks,
        RetargeterTuning(thumb_cmc_side_gain=0.5, thumb_cmc_roll_gain=1.0),
    )
    side_high = thumb_joint_targets_from_landmarks(
        side_landmarks,
        RetargeterTuning(thumb_cmc_side_gain=2.0, thumb_cmc_roll_gain=1.0),
    )
    assert side_high["thumb_cmc_side_joint"] < side_low["thumb_cmc_side_joint"]
    assert side_high["thumb_cmc_roll_joint"] == side_low["thumb_cmc_roll_joint"]

    roll_low = thumb_joint_targets_from_landmarks(
        opposed_landmarks,
        RetargeterTuning(thumb_cmc_side_gain=1.0, thumb_cmc_roll_gain=0.5),
    )
    roll_high = thumb_joint_targets_from_landmarks(
        opposed_landmarks,
        RetargeterTuning(thumb_cmc_side_gain=1.0, thumb_cmc_roll_gain=2.0),
    )
    assert roll_high["thumb_cmc_roll_joint"] > roll_low["thumb_cmc_roll_joint"]
    assert roll_high["thumb_cmc_side_joint"] == roll_low["thumb_cmc_side_joint"]


def test_postprocess_filter_alpha_covers_landmark_targets():
    retargeter = object.__new__(MidasHandRetargeter)
    profile = RetargetProfile.from_legacy_tuning(
        RetargeterTuning(finger_smoothing_alpha=0.12, thumb_smoothing_alpha=0.34)
    )

    for joint_name in (
        "index_mcp_abad_joint",
        "index_mcp_pitch_joint",
        "index_pip_joint",
        "middle_mcp_pitch_joint",
        "ring_pip_joint",
    ):
        assert retargeter._postprocess_filter_alpha(joint_name, profile) == 0.12

    for joint_name in (
        "thumb_cmc_roll_joint",
        "thumb_cmc_side_joint",
        "thumb_mcp_joint",
        "thumb_dip_joint",
    ):
        assert retargeter._postprocess_filter_alpha(joint_name, profile) == 0.34


def test_postprocess_filter_alpha_is_per_finger():
    """Each finger now carries its own alpha; they must not bleed together."""

    retargeter = object.__new__(MidasHandRetargeter)
    profile = RetargetProfile().with_values(
        {
            "index.smoothing_alpha": 0.1,
            "middle.smoothing_alpha": 0.5,
            "ring.smoothing_alpha": 0.9,
        }
    )
    assert retargeter._postprocess_filter_alpha("index_pip_joint", profile) == 0.1
    assert retargeter._postprocess_filter_alpha("middle_pip_joint", profile) == 0.5
    assert retargeter._postprocess_filter_alpha("ring_mcp_abad_joint", profile) == 0.9
    assert retargeter._postprocess_filter_alpha("no_such_joint", profile) is None


def test_hardware_motor_order_maps_thumb_cmc_roll_to_motor_id_3():
    assert HARDWARE_MOTOR_JOINT_NAMES[2] == "thumb_cmc_side_joint"
    assert HARDWARE_MOTOR_JOINT_NAMES[3] == "thumb_cmc_roll_joint"


def test_neutral_calibration_preserves_active_joint_ranges():
    retargeter = object.__new__(MidasHandRetargeter)
    retargeter.robot_joint_names = (
        "index_mcp_pitch_joint",
        "thumb_cmc_side_joint",
        "thumb_cmc_roll_joint",
    )
    retargeter.active_joint_names = retargeter.robot_joint_names
    retargeter._joint_index_by_name = {
        name: index
        for index, name in enumerate(retargeter.robot_joint_names)
    }
    retargeter._neutral_joint_offsets = {}
    retargeter._last_uncalibrated_active_joint_positions = {}
    # Limits now come from the solver-free HandModel, not from the optimizer.
    # The values below are deliberately NOT the shipped ones: this test pins
    # the recentering arithmetic, so it supplies its own limits.
    retargeter._model = HandModel(
        joint_names=retargeter.robot_joint_names,
        joint_limits=np.asarray(
            [
                [-1.35, 0.0],
                [-0.785, 0.9],
                [0.0, 2.15],
            ],
            dtype=np.float64,
        ),
    )
    retargeter._retargeting = None

    neutral_qpos = np.asarray([-0.4, 0.25, 1.2], dtype=np.float32)
    retargeter._last_uncalibrated_active_joint_positions = (
        retargeter._active_positions_from_qpos(neutral_qpos)
    )

    offsets = retargeter.calibrate_neutral_from_last_frame()

    assert offsets == {
        "index_mcp_pitch_joint": float(neutral_qpos[0]),
        "thumb_cmc_side_joint": float(neutral_qpos[1]),
        "thumb_cmc_roll_joint": float(neutral_qpos[2]),
    }
    np.testing.assert_allclose(
        retargeter._apply_neutral_offsets(neutral_qpos),
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        retargeter._apply_neutral_offsets(
            np.asarray([-0.7, 0.8, 2.15], dtype=np.float32)
        ),
        [-0.4263158, 0.7615385, 2.15],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        retargeter._apply_neutral_offsets(
            np.asarray([-1.35, -0.785, 0.0], dtype=np.float32)
        ),
        [-1.35, -0.785, 0.0],
        atol=1e-6,
    )

    retargeter.clear_neutral_offsets()
    np.testing.assert_allclose(
        retargeter._apply_neutral_offsets(neutral_qpos),
        neutral_qpos,
    )
