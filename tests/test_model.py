"""The checked-in HandModel table must stay equal to the shipped robot model.

The table in ``model.py`` is a literal so the analytic path needs no URDF and
no pinocchio. That speed comes with a drift risk, and these tests are the guard:
they assert the literal still equals the URDF, the MJCF, and — when pinocchio
is installed — pinocchio's own dof ordering, which is the index space of the
public ``RetargetingResult.robot_qpos``.

**They SKIP without the sibling ``midas_hand_mujoco`` repo, and CI does not
check it out**, so run them locally before trusting the table. Three of the six
skip in CI today. Closing that gap means giving the workflow access to that
repo, which is a cross-repo decision, not a test change.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from midas_hand_retargeter.model import MIDAS_RIGHT_HAND, HandModel


def _mujoco_repo():
    from midas_hand_retargeter.paths import find_mujoco_repo

    try:
        return find_mujoco_repo()
    except Exception:
        pytest.skip("midas_hand_mujoco not available")


def test_table_is_self_consistent():
    model = MIDAS_RIGHT_HAND
    assert model.n_joints == 19
    assert len(set(model.joint_names)) == 19, "duplicate joint name in table"
    assert model.joint_limits.shape == (19, 2)
    lower, upper = model.joint_limits[:, 0], model.joint_limits[:, 1]
    assert np.all(lower < upper), "a joint has an empty or inverted range"


def test_table_matches_shipped_urdf():
    urdf = _mujoco_repo() / "assets/midas_description/midas_hand_urdf.urdf"
    parsed = HandModel.from_urdf(urdf)
    assert parsed.joint_names == MIDAS_RIGHT_HAND.joint_names
    np.testing.assert_allclose(parsed.joint_limits, MIDAS_RIGHT_HAND.joint_limits)


def test_table_matches_shipped_mjcf_joint_ranges():
    mjcf = _mujoco_repo() / "assets/midas_description/midas_whole_hand.xml"
    root = ET.parse(str(mjcf)).getroot()
    ranges = {}
    for joint in root.iter("joint"):
        name, rng = joint.get("name"), joint.get("range")
        if name and rng:
            lo, hi = (float(v) for v in rng.split())
            ranges[name] = (lo, hi)

    for name in MIDAS_RIGHT_HAND.joint_names:
        assert name in ranges, f"{name} missing from MJCF"
        np.testing.assert_allclose(
            ranges[name],
            MIDAS_RIGHT_HAND.limits(name),
            atol=1e-9,
            err_msg=f"MJCF range for {name} drifted from the table",
        )


def test_table_matches_pinocchio_dof_ordering():
    """robot_qpos indexing is a public contract; pinocchio defines its order."""

    pin = pytest.importorskip("pinocchio")
    urdf = _mujoco_repo() / "assets/midas_description/midas_hand_urdf.urdf"
    dof_names = list(pin.buildModelFromUrdf(str(urdf)).names)[1:]  # drop "universe"
    assert tuple(dof_names) == MIDAS_RIGHT_HAND.joint_names


def test_index_and_clip():
    model = MIDAS_RIGHT_HAND
    assert model.index("index_mcp_abad_joint") == 0
    assert model.index("thumb_dip_joint") == 18
    assert model.clip("index_mcp_pitch_joint", -99.0) == pytest.approx(-1.8)
    assert model.clip("index_mcp_pitch_joint", 99.0) == pytest.approx(0.0)
    with pytest.raises(KeyError):
        model.index("no_such_joint")


def test_limits_are_immutable():
    """A shared mutable limits array would let one caller corrupt every other."""

    with pytest.raises(ValueError):
        MIDAS_RIGHT_HAND.joint_limits[0, 0] = 123.0
