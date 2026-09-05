"""Solver-free description of the MIDAS hand.

Why this exists
---------------
The analytic retargeting path needs three things that used to come only from
the ``dex_retargeting`` optimizer's pinocchio ``RobotWrapper``: the joint
ordering, the joint limits, and a name->index map. Depending on pinocchio for
them meant the default path could not run without ``dex_retargeting`` (and so
without ``torch`` and the CUDA wheel stack), and could not run at all without
the URDF from a sibling repo.

The table below is therefore a **checked-in literal**, not a parsed file, so
importing this module has no dependencies beyond numpy and reads nothing from
disk. ``HandModel.from_urdf()`` exists so tests can assert the literal still
matches the shipped URDF, and ``tests/test_model.py`` additionally asserts the
ordering still matches pinocchio's ``dof_joint_names`` when it is installed.

The ordering is pinocchio's URDF-tree dof order, NOT a tidy active-then-passive
grouping. It is the index space of ``RetargetingResult.robot_qpos``, which is a
public contract, so it must not be "cleaned up".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: (joint name, lower limit, upper limit) in pinocchio dof order. Source of
#: truth: midas_hand_urdf.urdf. Verified identical to the MJCF joint ranges
#: and, for the 13 actuated joints, to the MJCF actuator ctrlranges.
MIDAS_RIGHT_JOINTS: tuple[tuple[str, float, float], ...] = (
    ("index_mcp_abad_joint", -0.79, 0.79),
    ("index_mcp_pitch_joint", -1.8, 0.0),
    ("index_dip_linkage_joint", -3.14, 3.14),
    ("index_pip_joint", -1.45, 0.0),
    ("index_dip_joint", -3.14, 3.14),
    ("middle_mcp_abad_joint", -0.79, 0.79),
    ("middle_mcp_pitch_joint", -1.8, 0.0),
    ("middle_dip_linkage_joint", -3.14, 3.14),
    ("middle_pip_joint", -1.45, 0.0),
    ("middle_dip_joint", -3.14, 3.14),
    ("ring_mcp_abad_joint", -0.79, 0.79),
    ("ring_mcp_pitch_joint", -1.8, 0.0),
    ("ring_dip_linkage_joint", -3.14, 3.14),
    ("ring_pip_joint", -1.45, 0.0),
    ("ring_dip_joint", -3.14, 3.14),
    ("thumb_cmc_roll_joint", 0.0, 2.15),
    ("thumb_cmc_side_joint", -0.79, 0.9),
    ("thumb_mcp_joint", -1.57, 1.57),
    ("thumb_dip_joint", -1.57, 1.57),
)


@dataclass(frozen=True)
class HandModel:
    """Joint names, limits, and index lookup for one MIDAS hand."""

    joint_names: tuple[str, ...]
    joint_limits: np.ndarray  # (n, 2) float64, [lower, upper]

    @classmethod
    def right(cls) -> "HandModel":
        """The shipped right-hand model, from the checked-in table."""

        names = tuple(name for name, _, _ in MIDAS_RIGHT_JOINTS)
        limits = np.array([[lo, hi] for _, lo, hi in MIDAS_RIGHT_JOINTS], dtype=np.float64)
        limits.flags.writeable = False
        return cls(joint_names=names, joint_limits=limits)

    @classmethod
    def from_urdf(cls, urdf_path, *, joint_order=None) -> "HandModel":
        """Parse revolute joint limits from a URDF using only the stdlib.

        ``joint_order`` defaults to this module's canonical order; pass an
        explicit order (e.g. pinocchio's ``dof_joint_names``) to cross-check it.
        """

        import xml.etree.ElementTree as ET

        root = ET.parse(str(urdf_path)).getroot()
        parsed: dict[str, tuple[float, float]] = {}
        for joint in root.iter("joint"):
            if joint.get("type") != "revolute":
                continue
            limit = joint.find("limit")
            if limit is None:
                continue
            name = joint.get("name")
            parsed[name] = (float(limit.get("lower")), float(limit.get("upper")))

        order = tuple(joint_order) if joint_order is not None else tuple(
            name for name, _, _ in MIDAS_RIGHT_JOINTS
        )
        missing = [name for name in order if name not in parsed]
        if missing:
            raise ValueError(f"URDF {urdf_path} is missing revolute joints: {missing}")

        limits = np.array([parsed[name] for name in order], dtype=np.float64)
        limits.flags.writeable = False
        return cls(joint_names=order, joint_limits=limits)

    def index(self, joint_name: str) -> int:
        """Index of ``joint_name`` in the dof ordering."""

        try:
            return self.joint_names.index(joint_name)
        except ValueError:
            raise KeyError(
                f"Unknown joint {joint_name!r}; known joints: {list(self.joint_names)}"
            ) from None

    def limits(self, joint_name: str) -> tuple[float, float]:
        """``(lower, upper)`` limit for ``joint_name``, in radians."""

        lower, upper = self.joint_limits[self.index(joint_name)]
        return float(lower), float(upper)

    def clip(self, joint_name: str, value: float) -> float:
        """Clip ``value`` into ``joint_name``'s limits."""

        lower, upper = self.limits(joint_name)
        return float(np.clip(value, lower, upper))

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)


#: The default model used when no explicit one is supplied.
MIDAS_RIGHT_HAND = HandModel.right()
