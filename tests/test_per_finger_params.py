"""Per-finger parameters must actually be per-finger.

The whole point of the schema change is that tuning the index finger does not
move the middle one, and that the previously-unreachable part of each joint's
range becomes reachable. Both are pinned here.
"""

from __future__ import annotations

import pytest

from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.model import MIDAS_RIGHT_HAND
from midas_hand_retargeter.params import (
    PARAMETER_COUNT,
    RetargetProfile,
)
from midas_hand_retargeter.postprocess import finger_joint_targets_from_landmarks
from midas_hand_retargeter.store import ProfileStore
from midas_hand_retargeter.tuning import DEFAULT_TUNING

from ._poses import hand_pose


def test_there_is_no_pinky():
    profile = RetargetProfile()
    assert [name for name, _ in profile.fingers()] == ["index", "middle", "ring"]
    with pytest.raises(KeyError):
        profile.finger("pinky")


def test_schema_stays_a_tunable_size():
    """A UI with hundreds of sliders is not a tuning tool."""

    assert 40 <= PARAMETER_COUNT <= 60, PARAMETER_COUNT


def test_curl_gain_is_independent_per_finger():
    pose = hand_pose(curls=(0.7, 0.7, 0.7))
    base = finger_joint_targets_from_landmarks(pose, RetargetProfile())
    tuned = finger_joint_targets_from_landmarks(
        pose, RetargetProfile().with_values({"index.curl_gain": 2.0})
    )

    assert tuned["index_pip_joint"] < base["index_pip_joint"] - 1e-6
    assert tuned["middle_pip_joint"] == base["middle_pip_joint"]
    assert tuned["ring_pip_joint"] == base["ring_pip_joint"]


def test_splay_gain_is_independent_per_finger():
    pose = hand_pose(splays=(0.3, 0.3, 0.3))
    base = finger_joint_targets_from_landmarks(pose, RetargetProfile())
    tuned = finger_joint_targets_from_landmarks(
        pose, RetargetProfile().with_values({"ring.splay_gain": 2.5})
    )

    assert abs(tuned["ring_mcp_abad_joint"]) > abs(base["ring_mcp_abad_joint"]) + 1e-6
    assert tuned["index_mcp_abad_joint"] == base["index_mcp_abad_joint"]


def test_widened_output_range_reaches_the_full_urdf_travel():
    """The historical default left 25% of mcp_pitch and 16% of pip unreachable."""

    closed = hand_pose(curls=(2.0, 2.0, 2.0))

    default = finger_joint_targets_from_landmarks(closed, RetargetProfile())
    assert default["index_mcp_pitch_joint"] == pytest.approx(-1.35)
    assert default["index_pip_joint"] == pytest.approx(-1.22)

    widened = finger_joint_targets_from_landmarks(
        closed,
        RetargetProfile().with_values(
            {
                "index.mcp_pitch_range": (0.0, -1.8),
                "index.pip_range": (0.0, -1.45),
            }
        ),
    )
    assert widened["index_mcp_pitch_joint"] == pytest.approx(-1.8)
    assert widened["index_pip_joint"] == pytest.approx(-1.45)

    # ...and that is exactly the robot's limit, not beyond it.
    assert MIDAS_RIGHT_HAND.limits("index_mcp_pitch_joint") == (-1.8, 0.0)
    assert MIDAS_RIGHT_HAND.limits("index_pip_joint") == (-1.45, 0.0)


def test_legacy_tuning_still_drives_every_finger():
    pose = hand_pose(curls=(0.6, 0.6, 0.6))
    legacy = finger_joint_targets_from_landmarks(pose, DEFAULT_TUNING)
    modern = finger_joint_targets_from_landmarks(pose, RetargetProfile())
    assert legacy == modern


def test_disabled_finger_holds_its_last_command():
    """Disabling must freeze the finger, not fling it open."""

    retargeter = MidasHandRetargeter.create()
    curled = hand_pose(curls=(1.2, 1.2, 1.2))
    for _ in range(30):  # let the smoothing filter settle
        result = retargeter.retarget_landmarks(curled)
    held = result.active_joint_positions["index_pip_joint"]
    assert held < -0.5, "fixture should leave the index finger clearly curled"

    retargeter.profile = retargeter.profile.with_values({"index.enabled": False})
    after = retargeter.retarget_landmarks(hand_pose(curls=(0.0, 0.0, 0.0)))

    assert after.active_joint_positions["index_pip_joint"] == pytest.approx(held)
    assert after.active_joint_positions["middle_pip_joint"] > held + 0.1


def test_profile_store_edits_are_atomic_and_undoable():
    store = ProfileStore()
    assert store.get().index.curl_gain == 1.0

    store.apply({"index.curl_gain": 1.7})
    assert store.get().index.curl_gain == 1.7

    with pytest.raises(KeyError):
        store.apply({"index.nope": 1.0})
    assert store.get().index.curl_gain == 1.7, "failed edit must not be applied"

    assert store.undo().index.curl_gain == 1.0
    assert store.redo().index.curl_gain == 1.7
    assert store.reset().index.curl_gain == 1.0


def test_live_profile_swap_changes_the_next_frame():
    retargeter = MidasHandRetargeter.create()
    pose = hand_pose(curls=(0.7, 0.7, 0.7))
    for _ in range(30):
        before = retargeter.retarget_landmarks(pose)

    retargeter.profile = retargeter.profile.with_values({"index.curl_gain": 2.5})
    after = retargeter.retarget_landmarks(pose)

    assert (
        after.active_joint_positions["index_pip_joint"]
        < before.active_joint_positions["index_pip_joint"] - 1e-6
    )


def test_neutral_offsets_round_trip_through_the_public_api():
    retargeter = MidasHandRetargeter.create()
    retargeter.retarget_landmarks(hand_pose(curls=(0.4, 0.4, 0.4)))
    captured = retargeter.calibrate_neutral_from_last_frame()

    fresh = MidasHandRetargeter.create()
    fresh.set_neutral_offsets(captured)
    assert fresh.neutral_joint_offsets == pytest.approx(captured)

    with pytest.raises(KeyError):
        fresh.set_neutral_offsets({"not_a_joint": 0.0})
