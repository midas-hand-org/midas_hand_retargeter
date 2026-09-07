"""MIDAS passive PIP-DIP coupling, independent of any solver.

The four-bar linkage that drives each non-thumb DIP from its PIP is a property
of the *robot*, not of the retargeting method. This module therefore owns it
with no dependency on ``dex_retargeting``, ``torch`` or pinocchio, so the
analytic path can recompute passive joints on its own.

``adaptor.py`` builds on this to expose the same coupling to the optimizer,
where it additionally needs the Jacobian chain rule.

Two coupling modes are supported:

``fixed_passive``
    Option 1. Passive DIP/linkage joints stay fixed at ``passive_fixed_qpos``.
    Simulation closes the loop with MJCF constraints and hardware closes it in
    the API, so nothing here needs to model it.

``pip_dip_lookup``
    Option 2. Passive DIP/linkage joints are filled from the MIDAS four-bar
    lookup table, so downstream consumers (and fingertip FK) see the real
    passive motion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .constants import FINGER_NAMES

FIXED_PASSIVE_MODE = "fixed_passive"
PIP_DIP_LOOKUP_MODE = "pip_dip_lookup"
SUPPORTED_COUPLING_MODES = (FIXED_PASSIVE_MODE, PIP_DIP_LOOKUP_MODE)


def normalize_coupling_mode(mode: str) -> str:
    """Validate and canonicalize a coupling mode name.

    Single source of truth: this used to be duplicated in ``config.py`` (which
    did not lowercase) and ``adaptor.py`` (which did), so ``"Fixed_Passive"``
    was rejected by one and accepted by the other.
    """

    normalized = str(mode).lower()
    if normalized not in SUPPORTED_COUPLING_MODES:
        raise ValueError(
            f"Unsupported coupling_mode={mode!r}. Expected one of {SUPPORTED_COUPLING_MODES}."
        )
    return normalized


@dataclass(frozen=True)
class PipDipJointCoupling:
    """Names of one active PIP joint and its two passive four-bar joints."""

    pip_joint_name: str
    dip_joint_name: str
    linkage_joint_name: str


def couplings_for(finger_names: Sequence[str] = FINGER_NAMES):
    return tuple(
        PipDipJointCoupling(
            pip_joint_name=f"{finger}_pip_joint",
            dip_joint_name=f"{finger}_dip_joint",
            linkage_joint_name=f"{finger}_dip_linkage_joint",
        )
        for finger in finger_names
    )


def load_default_lookup(lookup_path: str | Path | None = None):
    """Load the packaged MIDAS four-bar lookup table.

    The table lives in ``midas_hand_api`` because the hardware layer owns the
    mechanism. That makes it an optional dependency of this package, declared
    as the ``lookup`` extra.
    """

    try:
        from midas_hand_api.fourbar_lookup import PipDipLookup
    except ImportError as exc:
        raise ImportError(
            "coupling_mode='pip_dip_lookup' requires midas_hand_api so the "
            "retargeter can reuse the packaged PIP-DIP four-bar lookup table. "
            "Install it with: pip install 'midas-hand-retargeter[lookup]'"
        ) from exc

    if lookup_path is not None:
        return PipDipLookup.from_csv(Path(lookup_path).expanduser().resolve())

    resource = resources.files("midas_hand_api").joinpath("assets", "pip_dip_linkage_lookup.csv")
    with resources.as_file(resource) as path:
        return PipDipLookup.from_csv(path)


def _joint_index_resolver(model):
    """Return ``name -> qpos index`` for either model flavour.

    ``HandModel`` exposes ``index``; a pinocchio ``RobotWrapper`` exposes
    ``get_joint_index``. Accepting both keeps one coupling implementation
    shared between the analytic and optimizer paths.
    """

    for attribute in ("index", "get_joint_index"):
        resolver = getattr(model, attribute, None)
        if callable(resolver):
            return resolver
    raise TypeError(
        f"{type(model).__name__} exposes neither index() nor get_joint_index(); "
        "cannot map joint names to qpos indices."
    )


class LookupPassiveCoupling:
    """Fill passive DIP/linkage joints from active PIP joints via the lookup.

    Needs only an object that can map a joint name to its qpos index: either
    ``index(name)`` (``model.HandModel``) or ``get_joint_index(name)`` (a
    pinocchio ``RobotWrapper``). So it works with or without a solver present.

    The lookup table was generated for a positive PIP simulator coordinate
    while the MIDAS URDF uses negative PIP values for flexion, so the default
    maps ``q_lookup = -q_pip_urdf``. ``pip_to_lookup_sign`` is the one place to
    flip that relationship.

    .. note::
       This agrees with ``midas_hand_api``, which was not obvious and was
       recorded here as a disagreement for a while. The API exposes the table
       through two functions: ``pip_to_dip_position`` takes a *lookup-space*
       (positive) angle and applies no flip, while ``passive_dip_from_pip_motor``
       takes a *motor-space* (negative-in-flexion) angle and negates it exactly
       as this class does. Compared against the motor-space one -- the only one
       a hardware angle should ever reach -- the two match to 1e-6 across the
       whole PIP range: -0.20 -> -0.4758, -0.90 -> -1.4509, -1.45 -> -1.9054.

       Nothing here reaches a motor in any case. The finger DIPs are passive
       four-bar links with no servo, so they are absent from
       ``HARDWARE_MOTOR_JOINT_NAMES``; this coupling exists to make the
       simulator and the Cartesian objective see the fingertip the linkage
       actually produces.
    """

    def __init__(
        self,
        model,
        *,
        lookup=None,
        lookup_path: str | Path | None = None,
        finger_names: Sequence[str] = FINGER_NAMES,
        pip_to_lookup_sign: float = -1.0,
        clamp: bool = True,
    ) -> None:
        self.model = model
        self.lookup = lookup if lookup is not None else load_default_lookup(lookup_path)
        self.pip_to_lookup_sign = float(pip_to_lookup_sign)
        self.clamp = bool(clamp)
        self.couplings = couplings_for(finger_names)

        index_of = _joint_index_resolver(model)
        self.idx_pip = np.asarray([index_of(c.pip_joint_name) for c in self.couplings], dtype=int)
        self.idx_dip = np.asarray([index_of(c.dip_joint_name) for c in self.couplings], dtype=int)
        self.idx_linkage = np.asarray(
            [index_of(c.linkage_joint_name) for c in self.couplings], dtype=int
        )

        self._q_pip_table = np.asarray(self.lookup.q_pip, dtype=float)
        self._dip_jacobian_table = np.gradient(
            np.asarray(self.lookup.q_dip, dtype=float), self._q_pip_table
        )
        self._linkage_jacobian_table = np.gradient(
            np.asarray(self.lookup.q_linkage_base, dtype=float), self._q_pip_table
        )
        self._last_pip_qpos = np.zeros(len(self.couplings), dtype=float)

    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Fill passive DIP/linkage entries of ``qpos`` from its PIP entries."""

        qpos = np.asarray(qpos)
        self._last_pip_qpos = np.asarray(qpos[self.idx_pip], dtype=float).copy()
        passive = self.lookup.evaluate(
            self.pip_to_lookup_sign * self._last_pip_qpos, clamp=self.clamp
        )
        qpos[self.idx_dip] = passive["q_dip"]
        qpos[self.idx_linkage] = passive["q_linkage_base"]
        return qpos

    def passive_derivatives(self):
        """``(d q_dip/d q_pip, d q_linkage/d q_pip)`` at the last PIP values."""

        return (
            self._lookup_derivative(self._dip_jacobian_table),
            self._lookup_derivative(self._linkage_jacobian_table),
        )

    def _lookup_derivative(self, derivative_table: np.ndarray) -> np.ndarray:
        q_lookup = self.pip_to_lookup_sign * self._last_pip_qpos
        low, high = self._q_pip_table[0], self._q_pip_table[-1]
        if self.clamp:
            valid = (low <= q_lookup) & (q_lookup <= high)
            q_eval = np.clip(q_lookup, low, high)
        else:
            valid = np.ones_like(q_lookup, dtype=bool)
            q_eval = q_lookup

        derivative = np.interp(q_eval, self._q_pip_table, derivative_table)
        return np.where(valid, derivative * self.pip_to_lookup_sign, 0.0)
