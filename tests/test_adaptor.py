from types import SimpleNamespace

import numpy as np

from midas_hand_retargeter.adaptor import (
    FIXED_PASSIVE_MODE,
    PIP_DIP_LOOKUP_MODE,
    MidasCoupledKinematicAdaptor,
    build_midas_kinematic_adaptor,
)
from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES, HARDWARE_MOTOR_JOINT_NAMES
from midas_hand_retargeter.retargeter import MidasHandRetargeter


class _FakeRobot:
    dof_joint_names = (
        "index_mcp_abad_joint",
        "index_mcp_pitch_joint",
        "index_pip_joint",
        "index_dip_linkage_joint",
        "index_dip_joint",
        "middle_mcp_abad_joint",
        "middle_mcp_pitch_joint",
        "middle_pip_joint",
        "middle_dip_linkage_joint",
        "middle_dip_joint",
        "ring_mcp_abad_joint",
        "ring_mcp_pitch_joint",
        "ring_pip_joint",
        "ring_dip_linkage_joint",
        "ring_dip_joint",
    )

    def get_joint_index(self, name: str) -> int:
        return self.dof_joint_names.index(name)


class _LinearLookup:
    q_pip = np.asarray([0.0, 1.0], dtype=float)
    q_dip = np.asarray([0.0, -2.0], dtype=float)
    q_linkage_base = np.asarray([0.0, -3.0], dtype=float)
    q_linkage_distal = np.asarray([0.0, 4.0], dtype=float)

    def evaluate(self, q_pip, clamp: bool = True):
        q = np.asarray(q_pip, dtype=float)
        if clamp:
            q = np.clip(q, self.q_pip[0], self.q_pip[-1])
        return {
            "q_dip": np.interp(q, self.q_pip, self.q_dip),
            "q_linkage_base": np.interp(q, self.q_pip, self.q_linkage_base),
            "q_linkage_distal": np.interp(q, self.q_pip, self.q_linkage_distal),
        }


def _adaptor() -> MidasCoupledKinematicAdaptor:
    return MidasCoupledKinematicAdaptor(
        robot=_FakeRobot(),
        target_joint_names=[name for name in _FakeRobot.dof_joint_names if "dip" not in name],
        lookup=_LinearLookup(),
    )


def test_build_adaptor_switches_between_option_1_and_option_2():
    robot = _FakeRobot()
    assert (
        build_midas_kinematic_adaptor(
            coupling_mode=FIXED_PASSIVE_MODE,
            robot=robot,
            target_joint_names=ACTIVE_JOINT_NAMES,
        )
        is None
    )

    adaptor = build_midas_kinematic_adaptor(
        coupling_mode=PIP_DIP_LOOKUP_MODE,
        robot=robot,
        target_joint_names=[name for name in robot.dof_joint_names if "dip" not in name],
        lookup_path=None,
    )
    assert isinstance(adaptor, MidasCoupledKinematicAdaptor)


def test_coupled_adaptor_maps_negative_urdf_pip_to_passive_joints():
    adaptor = _adaptor()
    qpos = np.zeros(len(_FakeRobot.dof_joint_names), dtype=np.float32)
    qpos[_FakeRobot.get_joint_index(_FakeRobot(), "index_pip_joint")] = -0.5

    adapted = adaptor.forward_qpos(qpos)

    assert adapted[_FakeRobot.get_joint_index(_FakeRobot(), "index_dip_joint")] == -1.0
    assert adapted[_FakeRobot.get_joint_index(_FakeRobot(), "index_dip_linkage_joint")] == -1.5


def test_coupled_adaptor_folds_passive_jacobians_into_pip_column():
    adaptor = _adaptor()
    qpos = np.zeros(len(_FakeRobot.dof_joint_names), dtype=np.float32)
    qpos[_FakeRobot.get_joint_index(_FakeRobot(), "index_pip_joint")] = -0.5
    adaptor.forward_qpos(qpos)

    jacobian = np.zeros((1, 3, len(_FakeRobot.dof_joint_names)), dtype=np.float32)
    pip_pin_index = _FakeRobot.get_joint_index(_FakeRobot(), "index_pip_joint")
    dip_pin_index = _FakeRobot.get_joint_index(_FakeRobot(), "index_dip_joint")
    linkage_pin_index = _FakeRobot.get_joint_index(_FakeRobot(), "index_dip_linkage_joint")
    jacobian[0, 0, pip_pin_index] = 1.0
    jacobian[0, 0, dip_pin_index] = 10.0
    jacobian[0, 0, linkage_pin_index] = 100.0

    target_jacobian = adaptor.backward_jacobian(jacobian)
    pip_target_index = adaptor.target_joint_names.index("index_pip_joint")

    assert target_jacobian[0, 0, pip_target_index] == 321.0


def test_make_result_reports_adapted_fixed_joint_positions():
    retargeter = object.__new__(MidasHandRetargeter)
    retargeter.robot_joint_names = (
        *HARDWARE_MOTOR_JOINT_NAMES,
        "index_dip_linkage_joint",
        "index_dip_joint",
    )
    retargeter.active_joint_names = HARDWARE_MOTOR_JOINT_NAMES
    retargeter.fixed_joint_names = (
        "index_dip_linkage_joint",
        "index_dip_joint",
    )
    retargeter._fixed_qpos = np.asarray([0.0, 0.0], dtype=np.float32)
    retargeter._retargeting = SimpleNamespace()

    qpos = np.zeros(len(retargeter.robot_joint_names), dtype=np.float32)
    qpos[-2:] = [-1.2, -0.8]
    result = retargeter._make_result(
        qpos,
        np.zeros((0, 3), dtype=np.float32),
    )

    np.testing.assert_allclose(
        [
            result.fixed_joint_positions["index_dip_linkage_joint"],
            result.fixed_joint_positions["index_dip_joint"],
        ],
        [-1.2, -0.8],
    )
