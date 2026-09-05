"""MIDAS hand retargeting wrappers.

This package intentionally depends on :mod:`dex_retargeting` instead of
copying its optimizer implementation. The MIDAS-specific layer owns robot
joint names, human landmark/vector mapping, passive-joint handling policy, and
output adapters for simulation or hardware callers.
"""

from .config import MidasRetargeterConfig
from .retargeter import MidasHandRetargeter, RetargetingResult
from .tuning import (
    DEFAULT_TUNING,
    PROFILES,
    RetargeterTuning,
    glove_tuning,
    tuning_for_source,
    vision_tuning,
)

__all__ = [
    "DEFAULT_TUNING",
    "PROFILES",
    "MidasHandRetargeter",
    "MidasRetargeterConfig",
    "RetargetingResult",
    "RetargeterTuning",
    "glove_tuning",
    "tuning_for_source",
    "vision_tuning",
]
