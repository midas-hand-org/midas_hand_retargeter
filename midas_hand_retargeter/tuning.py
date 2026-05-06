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
    finger_abad_gain: float = 1.0

    # Low-pass alpha for finger MCP ab/ad. Smaller is smoother but laggier.
    finger_smoothing_alpha: float = 0.16

    # Shared gain for thumb CMC roll/opposition and side sweep.
    thumb_cmc_gain: float = 1.0

    # Shared gain for thumb MCP/DIP flexion.
    thumb_flexion_gain: float = 1.0

    # Low-pass alpha for thumb CMC roll/side. Smaller is smoother but laggier.
    thumb_smoothing_alpha: float = 0.25


DEFAULT_TUNING = RetargeterTuning()
