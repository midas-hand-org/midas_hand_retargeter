"""Kinematic adaptors for MIDAS-specific passive mechanisms.

Two coupling modes are intentionally kept in this module so the caller can swap
the retargeting model without touching teleop or backend code:

``fixed_passive``
    Option 1. The generic dex-retargeting optimizer sees only active joints;
    passive DIP/linkage joints stay fixed at ``passive_fixed_qpos``.

``pip_dip_lookup``
    Option 2. Active PIP joints still remain the optimization variables, but
    passive DIP/linkage joints are filled from the MIDAS four-bar lookup before
    FK, and passive Jacobian columns are folded back into the corresponding PIP
    gradient. This lets fingertip vector objectives see the physical passive
    DIP motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from dex_retargeting.kinematics_adaptor import KinematicAdaptor
except ImportError:  # pragma: no cover - exercised only in partial envs.

    class KinematicAdaptor:
        """Small local stand-in used when only adaptor unit tests are running."""

        def __init__(self, robot, target_joint_names):
            self.robot = robot
            self.target_joint_names = list(target_joint_names)
            self.idx_pin2target = np.asarray(
                [robot.get_joint_index(name) for name in target_joint_names],
                dtype=int,
            )

from .constants import FINGER_NAMES


FIXED_PASSIVE_MODE = "fixed_passive"
PIP_DIP_LOOKUP_MODE = "pip_dip_lookup"
COUPLED_PIP_DIP_MODE = "coupled_pip_dip"
OPTION_2_COUPLING_MODES = (PIP_DIP_LOOKUP_MODE, COUPLED_PIP_DIP_MODE)
SUPPORTED_COUPLING_MODES = (FIXED_PASSIVE_MODE, *OPTION_2_COUPLING_MODES)


@dataclass(frozen=True)
class PipDipJointCoupling:
    """Names of one active PIP joint and its two passive four-bar joints."""

    pip_joint_name: str
    dip_joint_name: str
    linkage_joint_name: str


def build_midas_kinematic_adaptor(
    *,
    coupling_mode: str,
    robot,
    target_joint_names: Sequence[str],
    lookup_path: str | Path | None = None,
) -> KinematicAdaptor | None:
    """Build the adaptor requested by ``coupling_mode``.

    Returning ``None`` is meaningful: it is option 1, where passive joints stay
    fixed in the retargeting FK. ``pip_dip_lookup`` enables option 2.
    """

    mode = coupling_mode.lower()
    if mode == FIXED_PASSIVE_MODE:
        return None
    if mode in OPTION_2_COUPLING_MODES:
        return MidasCoupledKinematicAdaptor(
            robot=robot,
            target_joint_names=target_joint_names,
            lookup_path=lookup_path,
        )
    raise ValueError(
        f"Unsupported coupling_mode={coupling_mode!r}. "
        f"Expected one of {SUPPORTED_COUPLING_MODES}."
    )


class MidasCoupledKinematicAdaptor(KinematicAdaptor):
    """PIP-DIP-aware adaptor for MIDAS non-thumb fingers.

    The lookup table was generated for a positive PIP simulator coordinate.
    The current MIDAS URDF uses negative PIP values for finger flexion, so the
    default maps ``q_lookup = -q_pip_urdf``. If the URDF convention changes,
    ``pip_to_lookup_sign`` is the one place to flip that relationship.
    """

    def __init__(
        self,
        robot,
        target_joint_names: Sequence[str],
        *,
        lookup=None,
        lookup_path: str | Path | None = None,
        finger_names: Sequence[str] = FINGER_NAMES,
        pip_to_lookup_sign: float = -1.0,
        clamp: bool = True,
    ) -> None:
        target_joint_names = list(target_joint_names)
        super().__init__(robot, target_joint_names)

        self.lookup = lookup if lookup is not None else _load_default_lookup(lookup_path)
        self.pip_to_lookup_sign = float(pip_to_lookup_sign)
        self.clamp = bool(clamp)

        self.couplings = tuple(
            PipDipJointCoupling(
                pip_joint_name=f"{finger}_pip_joint",
                dip_joint_name=f"{finger}_dip_joint",
                linkage_joint_name=f"{finger}_dip_linkage_joint",
            )
            for finger in finger_names
        )
        self._validate_couplings()

        self.idx_pin2pip = np.asarray(
            [robot.get_joint_index(coupling.pip_joint_name) for coupling in self.couplings],
            dtype=int,
        )
        self.idx_pin2dip = np.asarray(
            [robot.get_joint_index(coupling.dip_joint_name) for coupling in self.couplings],
            dtype=int,
        )
        self.idx_pin2linkage = np.asarray(
            [
                robot.get_joint_index(coupling.linkage_joint_name)
                for coupling in self.couplings
            ],
            dtype=int,
        )
        self.idx_target2pip = np.asarray(
            [
                self.target_joint_names.index(coupling.pip_joint_name)
                for coupling in self.couplings
            ],
            dtype=int,
        )

        self._q_pip_table = np.asarray(self.lookup.q_pip, dtype=float)
        self._dip_jacobian_table = np.gradient(
            np.asarray(self.lookup.q_dip, dtype=float),
            self._q_pip_table,
        )
        self._linkage_jacobian_table = np.gradient(
            np.asarray(self.lookup.q_linkage_base, dtype=float),
            self._q_pip_table,
        )
        self._last_pip_qpos = np.zeros(len(self.couplings), dtype=float)

    def forward_qpos(self, pin_qpos: np.ndarray) -> np.ndarray:
        """Fill passive DIP/linkage qpos from active PIP qpos before FK."""

        qpos = np.asarray(pin_qpos)
        self._last_pip_qpos = np.asarray(qpos[self.idx_pin2pip], dtype=float).copy()
        q_lookup = self.pip_to_lookup_sign * self._last_pip_qpos
        passive = self.lookup.evaluate(q_lookup, clamp=self.clamp)
        qpos[self.idx_pin2dip] = passive["q_dip"]
        qpos[self.idx_pin2linkage] = passive["q_linkage_base"]
        return qpos

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """Fold passive DIP/linkage Jacobian columns back into active PIP.

        dex-retargeting optimizes only target joints. Because the fingertip FK
        now also depends on passive joints, the chain rule contributes
        ``J_passive * d(q_passive)/d(q_pip)`` to each active PIP column.
        """

        target_jacobian = jacobian[..., self.idx_pin2target].copy()
        dip_gain = self._lookup_derivative(self._dip_jacobian_table)
        linkage_gain = self._lookup_derivative(self._linkage_jacobian_table)

        for coupling_index, target_index in enumerate(self.idx_target2pip):
            target_jacobian[..., target_index] += (
                jacobian[..., self.idx_pin2dip[coupling_index]]
                * dip_gain[coupling_index]
            )
            target_jacobian[..., target_index] += (
                jacobian[..., self.idx_pin2linkage[coupling_index]]
                * linkage_gain[coupling_index]
            )
        return target_jacobian

    def _lookup_derivative(self, derivative_table: np.ndarray) -> np.ndarray:
        q_lookup = self.pip_to_lookup_sign * self._last_pip_qpos
        if self.clamp:
            valid = (self._q_pip_table[0] <= q_lookup) & (
                q_lookup <= self._q_pip_table[-1]
            )
            q_eval = np.clip(q_lookup, self._q_pip_table[0], self._q_pip_table[-1])
        else:
            valid = np.ones_like(q_lookup, dtype=bool)
            q_eval = q_lookup

        derivative = np.interp(q_eval, self._q_pip_table, derivative_table)
        derivative = derivative * self.pip_to_lookup_sign
        return np.where(valid, derivative, 0.0)

    def _validate_couplings(self) -> None:
        passive_names = {
            coupling.dip_joint_name
            for coupling in self.couplings
        } | {
            coupling.linkage_joint_name
            for coupling in self.couplings
        }
        passive_targets = passive_names.intersection(self.target_joint_names)
        if passive_targets:
            raise ValueError(
                "Passive DIP/linkage joints must not be optimized directly when "
                f"using {PIP_DIP_LOOKUP_MODE}: {sorted(passive_targets)}"
            )

        missing_sources = [
            coupling.pip_joint_name
            for coupling in self.couplings
            if coupling.pip_joint_name not in self.target_joint_names
        ]
        if missing_sources:
            raise ValueError(
                "Coupled PIP source joints must be target joints: "
                f"{missing_sources}"
            )


def _load_default_lookup(lookup_path: str | Path | None):
    try:
        from midas_hand_api.fourbar_lookup import PipDipLookup
    except ImportError as exc:
        raise ImportError(
            "coupling_mode='pip_dip_lookup' requires midas_hand_api so the "
            "retargeter can reuse the packaged PIP-DIP four-bar lookup table."
        ) from exc

    if lookup_path is not None:
        return PipDipLookup.from_csv(Path(lookup_path).expanduser().resolve())

    resource = resources.files("midas_hand_api").joinpath(
        "assets",
        "pip_dip_linkage_lookup.csv",
    )
    with resources.as_file(resource) as path:
        return PipDipLookup.from_csv(path)
