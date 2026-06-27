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
    finger_abad_gain: float = 0.5

    # Minimum lateral angle (radians) before abad activates. Raise if small
    # finger tilts that are harmless on thin human fingers collide on the
    # robot's thicker fingers.
    finger_abad_deadzone: float = 0.05

    # Constant outward abad bias added to index and ring fingers at all times
    # (radians). Index gets -offset (toward thumb), ring gets +offset (away
    # from thumb). 0.05 rad ≈ 3°. Middle finger is unaffected.
    finger_abad_outward_offset: float = 0.05

    # How much abad is suppressed as fingers curl. At 1.0, abad goes to zero at
    # full curl (safest for occlusion noise); at 0.0 curl has no effect.
    finger_abad_curl_damping: float = 0.75

    # LPF alpha for abad at full curl. Linearly interpolated from
    # finger_smoothing_alpha (open hand) to this value (fully curled), so the
    # filter tightens as the camera loses visibility of the lateral joints.
    finger_abad_alpha_curled: float = 0.05

    # Low-pass alpha for landmark-derived finger targets. Smaller is smoother
    # but laggier; this affects MCP ab/ad, MCP pitch, and PIP curl.
    finger_smoothing_alpha: float = 0.25

    # Legacy shared multiplier for both thumb CMC side sweep and roll/opposition.
    # Leave this at 1.0 when tuning the separate gains below.
    thumb_cmc_gain: float = 1.0

    # Resting outward side-sweep bias on the thumb CMC side joint (radians),
    # applied after neutral calibration so pressing 'c' does not absorb it. This
    # is also the value used for an index pinch and whenever no pinch is active.
    thumb_cmc_side_outward_offset: float = 0.25

    # Per-finger outward side bias (radians) the thumb CMC side joint blends
    # toward as a pinch to that finger engages, keyed off whichever fingertip the
    # thumb is nearest. The thumb's roll arc otherwise carries the tip back onto
    # the middle/ring finger base; a larger value here nudges it forward onto the
    # tip. The blend tracks the same pinch engagement as the opposition cap, so
    # it is 0-extra at rest and reaches the full value at contact. Index reuses
    # thumb_cmc_side_outward_offset above. Tune these two per finger; the
    # relationship across fingers is not assumed linear.
    thumb_cmc_side_middle_offset: float = 0.5
    thumb_cmc_side_ring_offset: float = 1.1

    # Low-pass alpha (EMA) for the per-finger thumb CMC side offset. Smaller is
    # smoother but laggier; 1.0 disables filtering. The offset already blends
    # continuously across fingers, so this only needs to clean up residual
    # jitter when the thumb hovers between the middle and ring fingertips.
    thumb_cmc_side_offset_alpha: float = 0.6

    # Increase for more in-plane thumb CMC side sweep.
    thumb_cmc_side_gain: float = 1.5

    # Increase for more thumb CMC roll/opposition.
    thumb_cmc_roll_gain: float = 0.4

    # Shared gain for thumb MCP/DIP flexion.
    thumb_flexion_gain: float = 1.5

    # Low-pass alpha for landmark-derived thumb targets. Smaller is smoother
    # but laggier; this affects CMC roll/side plus MCP/DIP flexion.
    thumb_smoothing_alpha: float = 0.25

    # LPF alpha for thumb joints at full roll. Linearly interpolated from
    # thumb_smoothing_alpha (open) to this value (fully rolled), based only
    # on the thumb's own CMC roll — independent of neighboring fingers.
    thumb_alpha_curled: float = 0.1

    # Distance (meters) at which the thumb-index pinch signal starts blending
    # in as a secondary opposition source. Orientation-independent complement
    # to the roll-angle signal.
    thumb_pinch_distance: float = 0.1

    # Distance (meters) at which the pinch signal reaches full opposition.
    # Below this the thumb CMC roll is fully driven to close regardless of
    # roll angle.
    thumb_pinch_snap_distance: float = 0.01

    # Maximum opposition value (0–1) the distance-based pinch snap can produce,
    # per finger. The snap targets whichever fingertip the thumb is nearest, and
    # the thumb must roll further across the palm to reach middle/ring than
    # index, so each finger gets a progressively higher cap. 1.0 = full CMC roll
    # range. Reduce a finger's cap if pinching it overshoots past the fingertip.
    thumb_pinch_opposition_cap: float = 0.6
    thumb_pinch_middle_opposition_cap: float = 0.8
    thumb_pinch_ring_opposition_cap: float = 1

    # Gain on the continuous proximity opposition term. As the thumb tip moves in
    # over the palm toward the finger bases, CMC roll engages directly — without
    # needing a fingertip pinch and without relying on the fragile out-of-plane
    # metacarpal angle. It is built from orientation-independent landmark
    # distances, so it keeps responding when the hand is edge-on to the camera.
    # 0 disables the term; raise for earlier/stronger roll engagement.
    thumb_opposition_proximity_gain: float = 0.6


DEFAULT_TUNING = RetargeterTuning()
