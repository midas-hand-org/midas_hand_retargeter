"""Human hand landmark retargeting for the MIDAS robot hand.

Four retargeting methods live here, selected by ``MidasRetargeterConfig.mode``:

``analytic`` (the default)
    A geometric landmark->joint map: per-finger curl from the blended PIP/DIP
    bend, splay from the palm-plane lateral angle, and thumb flexion/side/
    opposition from palm-local angles. Needs numpy and nothing else. Blind to
    absolute hand geometry, so it cannot place fingertips relative to each
    other -- scaling a hand 0.6x-3x moves its output by ~3e-6 rad.

``dexpilot`` (recommended for glove teleop)
    The DexPilot objective from :mod:`dex_retargeting`: six pairwise
    inter-fingertip vectors plus four palm-rooted ones, with pinch snapping.
    The only mode that controls where the fingertips sit *with respect to each
    other*. Handed, and sensitive to hand size -- see ``DexPilotParams``.
    Install with the ``vector`` extra.

``vector`` / ``refine``
    The plain :mod:`dex_retargeting` nlopt vector optimizer, alone or followed
    by the analytic layer. Kept for comparison. Install with the ``vector``
    extra.

The analytic layer writes all 13 actuated joints, so in ``refine`` the
optimizer's solution is fully overwritten and contributes nothing to the
command; ``analytic`` is the same result without the solve. Prefer ``analytic``
over ``refine`` always, and ``dexpilot`` over ``vector``.
"""

from .config import (
    ANALYTIC_MODE,
    DEXPILOT_MODE,
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
from .params import (
    DEFAULT_PROFILE,
    PARAMETER_COUNT,
    DexPilotParams,
    FingerParams,
    RetargetProfile,
    ThumbParams,
)
from .retargeter import MidasHandRetargeter, RetargetingResult
from .store import ProfileStore
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
    "DEFAULT_PROFILE",
    "DEXPILOT_MODE",
    "DEFAULT_TUNING",
    "FIXED_PASSIVE_MODE",
    "MIDAS_RIGHT_HAND",
    "PIP_DIP_LOOKUP_MODE",
    "PROFILES",
    "REFINE_MODE",
    "SUPPORTED_COUPLING_MODES",
    "SUPPORTED_RETARGET_MODES",
    "VECTOR_MODE",
    "PARAMETER_COUNT",
    "DexPilotParams",
    "FingerParams",
    "HandModel",
    "MidasHandRetargeter",
    "MidasRetargeterConfig",
    "RetargetingResult",
    "ProfileStore",
    "RetargetProfile",
    "RetargeterTuning",
    "ThumbParams",
    "__version__",
    "glove_tuning",
    "tuning_for_source",
    "vision_tuning",
]
