"""Human hand landmark retargeting for the MIDAS robot hand.

Two retargeting methods live here, selected by ``MidasRetargeterConfig.mode``:

``analytic`` (default)
    A geometric landmark->joint map: per-finger curl from the blended PIP/DIP
    bend, splay from the palm-plane lateral angle, and thumb flexion/side/
    opposition from palm-local angles. Needs numpy and nothing else.

``vector`` / ``refine``
    The :mod:`dex_retargeting` nlopt vector optimizer, alone or followed by the
    analytic layer. Install with the ``vector`` extra.

The analytic layer writes all 13 actuated joints, so in ``refine`` the
optimizer's solution is fully overwritten and contributes nothing to the
command; ``analytic`` is the same result without the solve. Prefer it unless
you are deliberately comparing against the optimizer.
"""

from .config import (
    ANALYTIC_MODE,
    REFINE_MODE,
    SUPPORTED_RETARGET_MODES,
    VECTOR_MODE,
    MidasRetargeterConfig,
)
from .coupling import (
    FIXED_PASSIVE_MODE,
    PIP_DIP_LOOKUP_MODE,
    SUPPORTED_COUPLING_MODES,
)
from .model import MIDAS_RIGHT_HAND, HandModel
from .retargeter import MidasHandRetargeter, RetargetingResult
from .tuning import (
    DEFAULT_TUNING,
    PROFILES,
    RetargeterTuning,
    glove_tuning,
    tuning_for_source,
    vision_tuning,
)

__version__ = "0.2.0"

__all__ = [
    "ANALYTIC_MODE",
    "DEFAULT_TUNING",
    "FIXED_PASSIVE_MODE",
    "MIDAS_RIGHT_HAND",
    "PIP_DIP_LOOKUP_MODE",
    "PROFILES",
    "REFINE_MODE",
    "SUPPORTED_COUPLING_MODES",
    "SUPPORTED_RETARGET_MODES",
    "VECTOR_MODE",
    "HandModel",
    "MidasHandRetargeter",
    "MidasRetargeterConfig",
    "RetargetingResult",
    "RetargeterTuning",
    "__version__",
    "glove_tuning",
    "tuning_for_source",
    "vision_tuning",
]
