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

    # ─── Human bend normalizers (source-sensitive) ────────────────────────
    # These convert the *measured human* joint bend into a 0..1 curl amount
    # (curl = smoothstep(gain * bend / max_bend)). They are source-sensitive:
    # a monocular vision tracker under-reports bend for a fully curled finger
    # (occlusion / foreshortening), while a glove reports the true anatomical
    # bend from its calibrated skeleton, so a fully closed finger yields a
    # larger measured bend. If a finger saturates (reaches full curl) before
    # it is physically closed, raise the matching max_bend for that source.

    # Blended PIP/DIP bend (rad) mapped to full finger curl.
    finger_curl_max_bend: float = 1.35

    # Thumb CMC-MCP-IP bend (rad) mapped to full thumb MCP flexion.
    thumb_mcp_max_bend: float = 1.57

    # Thumb MCP-IP-tip bend (rad) mapped to full thumb DIP flexion.
    thumb_dip_max_bend: float = 1.57


DEFAULT_TUNING = RetargeterTuning()


def vision_tuning() -> RetargeterTuning:
    """Tuning profile fit to the MediaPipe vision source.

    This is the historically hand-tuned baseline; it equals ``RetargeterTuning``'s
    field defaults. Kept as an explicit named profile so callers can select a
    source rather than relying on the bare defaults.
    """

    return RetargeterTuning()


def glove_tuning() -> RetargeterTuning:
    """Starting tuning profile for the Manus-glove source.

    The glove skeleton is cleaner and streams faster than monocular vision, so it
    tolerates lighter smoothing (less lag). Gains and bend normalizers start at
    the vision values; refine them on hardware with ``midas-hand-diag`` once the
    coordinate frame is confirmed correct. Do not blindly copy vision gains as
    final — they were fit to camera noise and MediaPipe's under-reported ROM.
    """

    return RetargeterTuning(
        # Cleaner data -> less low-pass lag than vision's 0.25.
        finger_smoothing_alpha=0.4,
        thumb_smoothing_alpha=0.4,
    )


# Named profiles selectable by input source. Extend this when adding sources.
PROFILES = {
    "vision": vision_tuning,
    "glove": glove_tuning,
}


def tuning_for_source(source: str) -> RetargeterTuning:
    """Return the tuning profile for a named input source (e.g. "vision", "glove")."""

    try:
        return PROFILES[source]()
    except KeyError:
        raise ValueError(
            f"Unknown tuning source {source!r}; expected one of {sorted(PROFILES)}"
        ) from None
