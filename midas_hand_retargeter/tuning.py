"""Retargeting postprocess tuning knobs.

These values are intentionally separated from the optimizer configuration so
live teleop behavior can be tuned without touching the dex-retargeting setup.
All angles are radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class RetargeterTuning:
    # Finger curl maps MediaPipe bend amount to the active MIDAS MCP/PIP motors.
    finger_mcp_pitch_open: float = 0.0
    finger_mcp_pitch_closed: float = -1.35
    finger_pip_open: float = 0.0
    finger_pip_closed: float = -1.22
    finger_curl_angle_offset: float = 0.18
    finger_curl_angle_span: float = 1.18
    finger_curl_closure_offset: float = 0.06
    finger_curl_closure_span: float = 0.34

    # Finger ab/ad uses the lateral angle of each proximal phalanx. Increase
    # gain for more splay motion; flip a sign below if a finger moves backward.
    finger_abad_gain: float = 0.38
    finger_abad_limit: float = 0.24
    finger_abad_deadzone: float = 0.12
    finger_abad_alpha: float = 0.16
    finger_abad_curl_damping: float = 0.65
    finger_abad_sign: Mapping[str, float] = field(
        default_factory=lambda: {
            "index": 1.0,
            "middle": 1.0,
            "ring": 1.0,
        }
    )
    finger_abad_neutral: Mapping[str, float] = field(
        default_factory=lambda: {
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0,
        }
    )

    # Thumb CMC maps an opposition amount to roll/side motors. Swap open/oppose
    # values if a joint moves in the wrong direction for your model.
    thumb_cmc_roll_open: float = 1.25
    thumb_cmc_roll_oppose: float = 0.35
    thumb_cmc_side_open: float = -0.30
    thumb_cmc_side_oppose: float = 0.42
    thumb_pinch_open_ratio: float = 1.15
    thumb_pinch_closed_ratio: float = 0.34
    thumb_pinch_gain: float = 0.85
    thumb_curl_opposition_gain: float = 0.55

    # Thumb MCP/DIP use MediaPipe thumb joint bends directly. If those joints
    # feel lazy, reduce the *_closed_angle values or make *_closed more extreme.
    thumb_mcp_open: float = -0.10
    thumb_mcp_closed: float = -0.88
    thumb_mcp_open_angle: float = 0.08
    thumb_mcp_closed_angle: float = 1.05
    thumb_dip_open: float = 0.02
    thumb_dip_closed: float = -0.72
    thumb_dip_open_angle: float = 0.05
    thumb_dip_closed_angle: float = 0.88
    thumb_dip_mcp_follow: float = 0.35


DEFAULT_TUNING = RetargeterTuning()
