import numpy as np

from midas_hand_retargeter.postprocess import (
    finger_joint_targets_from_landmarks,
    thumb_joint_targets_from_landmarks,
)


def _straight_landmarks() -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    for base, x in ((5, -0.035), (9, 0.0), (13, 0.035)):
        landmarks[base] = [x, 0.035, 0.0]
        landmarks[base + 1] = [x, 0.060, 0.0]
        landmarks[base + 2] = [x, 0.083, 0.0]
        landmarks[base + 3] = [x, 0.105, 0.0]
    landmarks[1] = [-0.045, 0.020, 0.0]
    landmarks[2] = [-0.065, 0.038, 0.0]
    landmarks[3] = [-0.085, 0.056, 0.0]
    landmarks[4] = [-0.105, 0.074, 0.0]
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

    assert side_targets["thumb_cmc_side_joint"] > open_targets["thumb_cmc_side_joint"]
