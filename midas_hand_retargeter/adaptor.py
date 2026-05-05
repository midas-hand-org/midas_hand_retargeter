"""Placeholder for option-2 MIDAS passive-coupling retargeting.

Option 1 keeps passive DIP/linkage joints fixed inside the retargeting FK
model and lets MuJoCo/hardware resolve the passive mechanism downstream.
Option 2 should implement a kinematic adaptor that fills passive joints from
active PIP joints before FK and folds passive Jacobian columns back to PIP.
"""

from __future__ import annotations


class MidasCoupledKinematicAdaptor:
    """Reserved extension point for PIP-DIP-aware retargeting FK."""

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "MidasCoupledKinematicAdaptor is reserved for option 2. "
            "Use MidasRetargeterConfig(coupling_mode='fixed_passive') for now."
        )
