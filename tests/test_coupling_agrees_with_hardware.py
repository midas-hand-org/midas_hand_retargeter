"""The four-bar coupling must agree with the hardware package's copy of it.

Two packages evaluate the same lookup table from different sign conventions,
and for a while this repo's docstring recorded them as *disagreeing* and told
operators to settle it on hardware. They do not disagree: midas_hand_api
exposes the table twice, and the comparison had been made against the wrong
one. This test pins the real relationship so neither side can drift.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from midas_hand_retargeter.constants import HARDWARE_MOTOR_JOINT_NAMES
from midas_hand_retargeter.coupling import LookupPassiveCoupling
from midas_hand_retargeter.model import MIDAS_RIGHT_HAND

needs_api = pytest.mark.skipif(
    importlib.util.find_spec("midas_hand_api") is None,
    reason="midas_hand_api is not installed",
)

#: Motor-space PIP angles, i.e. flexion negative, spanning the URDF range.
PIP_ANGLES = (-0.20, -0.50, -0.90, -1.20, -1.45)


def _coupled_dip(pip: float, finger: str = "index") -> float:
    names = MIDAS_RIGHT_HAND.joint_names
    qpos = np.zeros(len(names))
    qpos[names.index(f"{finger}_pip_joint")] = pip
    coupled = LookupPassiveCoupling(MIDAS_RIGHT_HAND).forward_qpos(qpos)
    return float(coupled[names.index(f"{finger}_dip_joint")])


@needs_api
@pytest.mark.parametrize("pip", PIP_ANGLES)
def test_matches_the_motor_space_helper(pip):
    """passive_dip_from_pip_motor is the one a hardware angle should reach."""

    from midas_hand_api.kinematics import passive_dip_from_pip_motor

    assert _coupled_dip(pip) == pytest.approx(float(passive_dip_from_pip_motor(pip)), abs=1e-6)


@needs_api
def test_the_raw_lookup_helper_is_the_wrong_comparison():
    """Guards the mistake itself, so nobody re-derives the phantom conflict.

    pip_to_dip_position takes a LOOKUP-space (positive) angle. Handed a
    motor-space one it falls below the table's domain and clamps to zero --
    which looks like a sign disagreement and is really a unit error.
    """

    from midas_hand_api.kinematics import pip_to_dip_position

    assert float(pip_to_dip_position(-0.90)) == 0.0
    assert float(pip_to_dip_position(+0.90)) == pytest.approx(_coupled_dip(-0.90), abs=1e-6)


def test_no_passive_dip_is_ever_commanded_to_a_motor():
    """The finger DIPs are passive four-bar links with no servo, so the
    coupling cannot reach the hardware even if it were wrong."""

    assert not [n for n in HARDWARE_MOTOR_JOINT_NAMES if n.endswith("_dip_joint")
                and not n.startswith("thumb")]
    # The thumb DIP is genuinely motor-driven, and is not part of the four-bar.
    assert "thumb_dip_joint" in HARDWARE_MOTOR_JOINT_NAMES
