"""dex-retargeting kinematic adaptors for the MIDAS passive four-bar.

This module is the *optimizer-facing* half of the PIP-DIP coupling; the
solver-free half lives in :mod:`midas_hand_retargeter.coupling`.

**Importing this module must stay free of heavy dependencies.** Several
downstream repos (``midas_hand_teleop``, ``midas-piper-control``) import the
coupling-mode constants from here, and ``dex_retargeting`` raises ImportError
outright when ``torch`` is absent — so a module-level ``dex_retargeting``
import would make torch a hard import-time dependency of the whole package.
``MidasCoupledKinematicAdaptor`` is therefore resolved lazily through PEP 562
``__getattr__``, and only the option-2 path ever pays for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from .constants import FINGER_NAMES
from .coupling import (
    COUPLED_PIP_DIP_MODE,
    FIXED_PASSIVE_MODE,
    OPTION_2_COUPLING_MODES,
    PIP_DIP_LOOKUP_MODE,
    SUPPORTED_COUPLING_MODES,
    LookupPassiveCoupling,
    PipDipJointCoupling,
    couplings_for,
    normalize_coupling_mode,
)

# ruff: noqa: F822 - MidasCoupledKinematicAdaptor is resolved lazily by the
# module __getattr__ below, so it is exported but not defined at module level.
__all__ = [
    "COUPLED_PIP_DIP_MODE",
    "FIXED_PASSIVE_MODE",
    "OPTION_2_COUPLING_MODES",
    "PIP_DIP_LOOKUP_MODE",
    "SUPPORTED_COUPLING_MODES",
    "MidasCoupledKinematicAdaptor",
    "PipDipJointCoupling",
    "build_midas_kinematic_adaptor",
    "normalize_coupling_mode",
]


def build_midas_kinematic_adaptor(
    *,
    coupling_mode: str,
    robot,
    target_joint_names: Sequence[str],
    lookup_path: str | Path | None = None,
):
    """Build the dex-retargeting adaptor requested by ``coupling_mode``.

    Returning ``None`` is meaningful: it is option 1, where passive joints stay
    fixed in the retargeting FK.
    """

    mode = normalize_coupling_mode(coupling_mode)
    if mode == FIXED_PASSIVE_MODE:
        return None
    return _coupled_adaptor_class()(
        robot=robot,
        target_joint_names=target_joint_names,
        lookup_path=lookup_path,
    )


@lru_cache(maxsize=1)
def _coupled_adaptor_class():
    """Define the adaptor class against the real dex-retargeting base class.

    Deferred so that importing this module costs nothing. Previously a
    ``try/except ImportError`` installed a *fake* ``KinematicAdaptor`` base at
    import time, which made a broken install look partly functional.
    """

    try:
        from dex_retargeting.kinematics_adaptor import KinematicAdaptor
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "coupling_mode='pip_dip_lookup' needs the vector optimizer. "
            "Install it with: pip install 'midas-hand-retargeter[vector]'"
        ) from exc

    class MidasCoupledKinematicAdaptor(KinematicAdaptor):
        """PIP-DIP-aware adaptor for MIDAS non-thumb fingers.

        Active PIP joints remain the optimization variables; passive
        DIP/linkage joints are filled from the four-bar lookup before FK, and
        their Jacobian columns are folded back into the corresponding PIP
        gradient so fingertip objectives see the real passive motion.
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

            self.couplings = couplings_for(finger_names)
            self._validate_couplings()

            # The solver-free coupling owns the table, the sign convention and
            # the derivative interpolation; this class adds only the chain rule.
            self._coupling = LookupPassiveCoupling(
                robot,
                lookup=lookup,
                lookup_path=lookup_path,
                finger_names=finger_names,
                pip_to_lookup_sign=pip_to_lookup_sign,
                clamp=clamp,
            )
            self.lookup = self._coupling.lookup
            self.pip_to_lookup_sign = self._coupling.pip_to_lookup_sign
            self.clamp = self._coupling.clamp
            self.idx_pin2pip = self._coupling.idx_pip
            self.idx_pin2dip = self._coupling.idx_dip
            self.idx_pin2linkage = self._coupling.idx_linkage
            self.idx_target2pip = np.asarray(
                [self.target_joint_names.index(c.pip_joint_name) for c in self.couplings],
                dtype=int,
            )

        def forward_qpos(self, pin_qpos: np.ndarray) -> np.ndarray:
            """Fill passive DIP/linkage qpos from active PIP qpos before FK."""

            return self._coupling.forward_qpos(pin_qpos)

        def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
            """Fold passive DIP/linkage Jacobian columns back into active PIP.

            dex-retargeting optimizes only target joints. Because fingertip FK
            now also depends on passive joints, the chain rule contributes
            ``J_passive * d(q_passive)/d(q_pip)`` to each active PIP column.
            """

            target_jacobian = jacobian[..., self.idx_pin2target].copy()
            dip_gain, linkage_gain = self._coupling.passive_derivatives()

            for coupling_index, target_index in enumerate(self.idx_target2pip):
                target_jacobian[..., target_index] += (
                    jacobian[..., self.idx_pin2dip[coupling_index]] * dip_gain[coupling_index]
                )
                target_jacobian[..., target_index] += (
                    jacobian[..., self.idx_pin2linkage[coupling_index]]
                    * linkage_gain[coupling_index]
                )
            return target_jacobian

        def _validate_couplings(self) -> None:
            passive_names = {c.dip_joint_name for c in self.couplings} | {
                c.linkage_joint_name for c in self.couplings
            }
            passive_targets = passive_names.intersection(self.target_joint_names)
            if passive_targets:
                raise ValueError(
                    "Passive DIP/linkage joints must not be optimized directly "
                    f"when using {PIP_DIP_LOOKUP_MODE}: {sorted(passive_targets)}"
                )

            missing_sources = [
                c.pip_joint_name
                for c in self.couplings
                if c.pip_joint_name not in self.target_joint_names
            ]
            if missing_sources:
                raise ValueError(
                    f"Coupled PIP source joints must be target joints: {missing_sources}"
                )

    return MidasCoupledKinematicAdaptor


def __getattr__(name: str):
    """PEP 562 lazy attribute access, so the import stays dependency-free."""

    if name == "MidasCoupledKinematicAdaptor":
        return _coupled_adaptor_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
