"""Small public tuning surface for MIDAS postprocessing.

Most retargeting behavior should come from geometry and the upstream vector
optimizer. These gains are intentionally coarse so live teleop tuning does not
turn into a large pile of hand-fitted constants.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetargeterTuning:
    """Coarse gains for the landmark-derived correction layer.

    Use ``MidasRetargeterConfig`` for optimizer-level tuning such as target
    links and solver losses. Use this class only when the live MIDAS response
    needs broad adjustment.
    """

    # Increase if fingers feel under-curled; decrease if they close too early.
    finger_curl_gain: float = 1.0

    # Increase for more MCP ab/ad sweep; decrease if lateral motion is jittery.
    finger_abad_gain: float = 1.2

    # Low-pass alpha for landmark-derived finger targets. Smaller is smoother
    # but laggier; this affects MCP ab/ad, MCP pitch, and PIP curl.
    finger_smoothing_alpha: float = 0.25

    # Legacy shared multiplier for both thumb CMC side sweep and roll/opposition.
    # Leave this at 1.0 when tuning the separate gains below.
    thumb_cmc_gain: float = 1.0

    # Increase for more in-plane thumb CMC side sweep.
    thumb_cmc_side_gain: float = 1.5

    # Increase for more thumb CMC roll/opposition.
    thumb_cmc_roll_gain: float = 0.6

    # Shared gain for thumb MCP/DIP flexion.
    thumb_flexion_gain: float = 1.2

    # Low-pass alpha for landmark-derived thumb targets. Smaller is smoother
    # but laggier; this affects CMC roll/side plus MCP/DIP flexion.
    thumb_smoothing_alpha: float = 0.25


DEFAULT_TUNING = RetargeterTuning()
