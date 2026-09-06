"""Per-finger parameters for the analytic retargeting map.

Why per-finger
--------------
The analytic map used to be driven by 11 *global* numbers plus ~16 hardcoded
module constants, so every finger shared one curl gain and one splay gain, and
the output ranges could not be changed at all. That made real tuning
impossible: an index finger that saturates before it is physically closed and
a ring finger that never quite closes need opposite corrections.

Every knob here is per-finger (index/middle/ring) or thumb-specific. The
previously hardcoded constants — output ranges, deadzones, limits, the curl
blend weight, the thumb's neutral angle and span — are now fields.

Immutability and live tuning
----------------------------
These are frozen dataclasses and the control loop reads a whole
``RetargetProfile`` once per frame. Live edits replace the entire profile
atomically (see ``ProfileStore``) rather than mutating fields, because the web
tuner writes from one thread while the 60 Hz loop reads from another, and a
half-applied parameter set produces a visible glitch on the hand.

Numerical guards (epsilons, divide-by-zero protection) are deliberately NOT
fields: they are implementation details, not tuning knobs, and putting them in
a versioned schema means carrying them through every future migration.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

from .constants import FINGER_NAMES


@dataclass(frozen=True)
class FingerParams:
    """Analytic parameters for one non-thumb finger.

    Curl pipeline::

        bend  = curl_pip_weight * angle(prox, mid) + (1 - w) * angle(mid, dist)
        curl  = smoothstep(curl_gain * bend / curl_max_bend)      # 0..1
        mcp_pitch = lerp(*mcp_pitch_range, curl)
        pip       = lerp(*pip_range, curl)

    Splay pipeline::

        angle = atan2(lateral, forward)  in the palm plane
        splay = deadzone(angle, splay_deadzone)
        abad  = clip((1 - splay_curl_damping * curl) * splay_gain * splay, +/- splay_limit)
    """

    #: Multiplies measured bend before normalization. Raise to close sooner.
    curl_gain: float = 1.0
    #: Human bend (rad) that maps to fully curled. Source-sensitive: monocular
    #: vision under-reports bend, a calibrated glove reports true anatomy.
    curl_max_bend: float = 1.35
    #: Weight of the PIP bend in the blend; the DIP bend gets ``1 - this``.
    curl_pip_weight: float = 0.62

    #: Commanded ``(open, closed)`` range for MCP pitch, radians. The URDF
    #: allows -1.8; the historical default only reached -1.35 (25% of travel
    #: unreachable), so widening this is the main way to get full closure.
    mcp_pitch_range: tuple[float, float] = (0.0, -1.35)
    #: Commanded ``(open, closed)`` range for PIP. URDF allows -1.45.
    pip_range: tuple[float, float] = (0.0, -1.22)

    #: Multiplies the deadzoned lateral angle.
    splay_gain: float = 1.2
    #: Lateral angle (rad) ignored around neutral, to reject tracking jitter.
    splay_deadzone: float = 0.05
    #: Symmetric clamp on the abduction command, radians. URDF allows 0.79.
    splay_limit: float = 0.785
    #: How much a curled finger suppresses splay (0 = none, 1 = fully) —
    #: splay landmarks get unreliable as the finger closes.
    splay_curl_damping: float = 0.5

    #: Per-joint low-pass alpha. 1.0 = no smoothing, smaller = smoother/laggier.
    smoothing_alpha: float = 0.25

    #: When False, this finger holds its last command instead of tracking.
    #: Deliberately hold-last, not snap-to-open: snapping would fling a finger
    #: open mid-teleop on hardware.
    enabled: bool = True

    @property
    def curl_dip_weight(self) -> float:
        return 1.0 - self.curl_pip_weight


@dataclass(frozen=True)
class ThumbParams:
    """Analytic parameters for the thumb.

    The thumb has three independent ideas, and they are tuned separately:
    flexion (MCP/DIP) from segment bends, *side* sweep within the palm plane,
    and *opposition* (CMC roll) out of the palm plane.
    """

    # --- flexion -------------------------------------------------------
    flexion_gain: float = 1.2
    mcp_max_bend: float = 1.57
    dip_max_bend: float = 1.57
    mcp_range: tuple[float, float] = (0.0, -1.57)
    dip_range: tuple[float, float] = (0.0, -1.57)
    #: Floor tying DIP to MCP, so the tip still curls when the IP bend is
    #: poorly observed. DIP curl is at least this fraction of MCP curl.
    dip_follows_mcp: float = 0.3

    # --- side sweep (in the palm plane) --------------------------------
    cmc_side_gain: float = 1.5
    cmc_side_range: tuple[float, float] = (-0.785, 0.9)
    #: Measured in-plane angle (rad) treated as the thumb's neutral rest pose.
    cmc_side_neutral_angle: float = -0.3
    cmc_side_deadzone: float = 0.05
    #: Commanded value at the neutral angle.
    cmc_side_open: float = 0.0

    # --- opposition / CMC roll (out of the palm plane) -----------------
    cmc_roll_gain: float = 0.6
    cmc_roll_range: tuple[float, float] = (0.0, 2.15)
    #: Out-of-plane angle (rad) spanning neutral to full opposition.
    cmc_roll_span: float = 0.45
    cmc_roll_deadzone: float = 0.05

    smoothing_alpha: float = 0.25
    enabled: bool = True


@dataclass(frozen=True)
class DexPilotParams:
    """Solver parameters for the DexPilot optimizer (``mode="dexpilot"``).

    Unlike the analytic map, DexPilot optimises *fingertip positions*: six
    pairwise inter-fingertip vectors (index-thumb, middle-thumb, ring-thumb,
    middle-index, ring-index, ring-middle) plus four palm-rooted ones. That is
    why it can reproduce the relative geometry between fingers, which the
    analytic map structurally cannot — the analytic map is blind to absolute
    hand geometry (scaling a hand 0.6x-3x changes its output by ~3e-6 rad).

    Every field here is applied live; none needs a rebuild.
    """

    #: Human-to-robot size ratio. **The most important knob in this mode.**
    #: The analytic map ignores hand size entirely; DexPilot does not — a 0.7x
    #: to 1.5x change moves joints by ~1.5 rad. Calibrate it to the operator.
    scaling_factor: float = 1.15

    #: Huber loss width (m). Below this, error is quadratic; above, linear.
    #: Smaller tracks small errors harder but is more jittery.
    huber_delta: float = 0.03

    #: Temporal regularizer. Penalises deviation from the previous solution,
    #: so larger is smoother but laggier. This is the solver's own smoothing.
    norm_delta: float = 4e-3

    #: Distance (m) at which a fingertip pair is treated as *trying to touch*
    #: and its target distance snaps to eta1/eta2. This is what makes pinches
    #: land precisely instead of hovering.
    project_dist: float = 0.03
    #: Distance (m) at which a projected pair is released again. Must exceed
    #: project_dist; the gap is hysteresis against chattering in and out.
    escape_dist: float = 0.05
    #: Projected target distance for thumb-to-finger pairs (m).
    eta1: float = 1e-4
    #: Projected target distance for finger-to-finger pairs (m).
    eta2: float = 3e-2

    #: Upstream low-pass on the solution. 1.0 = off.
    low_pass_alpha: float = 1.0
    #: Extra spatial scale applied to the THUMB's own vectors only (its
    #: base-rooted vector and its three pinch pairs). 1.0 = off.
    #:
    #: The thumb-root rebase removes most of the MIDAS thumb's proportional
    #: excess; this closes the rest. Raising it straightens the thumb further
    #: at a measured cost in fingertip accuracy: ~1.1 closes the residual gap,
    #: 1.15 takes thumb bend 0.37 -> 0.17 rad for +1.4 mm of inter-fingertip
    #: error, 1.42 reaches 0.15 rad for +3.3 mm. It does not help jitter.
    thumb_vector_scale: float = 1.0

    #: Per-joint output low-pass. Default 1.0 (off), unlike the analytic path.
    #: The residual jitter in this mode is drift, not noise — the objective
    #: under-constrains the hand (10 vectors, 13 DOF), so redundant joints
    #: wander even on a perfectly constant input. A low-pass cannot remove
    #: drift, and measurably costs accuracy: at 0.5 it takes mean
    #: inter-fingertip error from 8.3 mm to 20.3 mm while reducing jitter only
    #: 0.42 -> 0.36 rad. Turn it down only if you prefer a calmer, laggier hand.
    smoothing_alpha: float = 1.0


@dataclass(frozen=True)
class RetargetProfile:
    """A complete analytic tuning profile: one entry per digit.

    The MIDAS hand has a thumb and three fingers. There is no pinky.
    """

    index: FingerParams = FingerParams()
    middle: FingerParams = FingerParams()
    ring: FingerParams = FingerParams()
    thumb: ThumbParams = ThumbParams()
    #: Solver knobs for mode="dexpilot". Ignored by the analytic path.
    dexpilot: DexPilotParams = DexPilotParams()

    #: Free-form provenance: which source/hand this was tuned for.
    name: str = "default"
    source: str = "vision"

    @classmethod
    def from_legacy_tuning(cls, tuning) -> RetargetProfile:
        """Convert the flat 11-field ``RetargeterTuning`` into a profile.

        Applies each global gain to every finger, and folds the legacy shared
        ``thumb_cmc_gain`` multiplier into the two thumb CMC gains it scaled.
        Every other field keeps the value that used to be a module constant, so
        the conversion is exactly behaviour-preserving.
        """

        finger = FingerParams(
            curl_gain=tuning.finger_curl_gain,
            curl_max_bend=tuning.finger_curl_max_bend,
            splay_gain=tuning.finger_abad_gain,
            smoothing_alpha=tuning.finger_smoothing_alpha,
        )
        thumb = ThumbParams(
            flexion_gain=tuning.thumb_flexion_gain,
            mcp_max_bend=tuning.thumb_mcp_max_bend,
            dip_max_bend=tuning.thumb_dip_max_bend,
            cmc_side_gain=tuning.thumb_cmc_gain * tuning.thumb_cmc_side_gain,
            cmc_roll_gain=tuning.thumb_cmc_gain * tuning.thumb_cmc_roll_gain,
            smoothing_alpha=tuning.thumb_smoothing_alpha,
        )
        return cls(
            index=finger, middle=finger, ring=finger, thumb=thumb, name="legacy", source="legacy"
        )

    def finger(self, name: str) -> FingerParams:
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(
                f"Unknown finger {name!r}; expected one of {list(FINGER_NAMES)}"
            ) from None

    def fingers(self) -> Iterator[tuple[str, FingerParams]]:
        for name in FINGER_NAMES:
            yield name, self.finger(name)

    # --- editing -------------------------------------------------------
    def with_values(self, updates: Mapping[str, object]) -> RetargetProfile:
        """Return a copy with dotted-path ``updates`` applied.

        Paths look like ``"index.curl_gain"`` or ``"thumb.cmc_roll_span"``.
        Unknown paths raise rather than being ignored: a silently dropped
        parameter edit is the single worst failure mode for a tuning UI.
        """

        grouped: dict[str, dict[str, object]] = {}
        for path, value in updates.items():
            section, _, field_name = str(path).partition(".")
            if not field_name:
                raise KeyError(
                    f"Parameter path {path!r} must be '<digit>.<field>', e.g. 'index.curl_gain'"
                )
            if section not in _SECTION_TYPES:
                raise KeyError(
                    f"Unknown section {section!r} in {path!r}; "
                    f"expected one of {sorted(_SECTION_TYPES)}"
                )
            if field_name not in _SECTION_FIELDS[section]:
                raise KeyError(
                    f"Unknown parameter {field_name!r} in {path!r}; "
                    f"{section} has {sorted(_SECTION_FIELDS[section])}"
                )
            grouped.setdefault(section, {})[field_name] = value

        updated = self
        for section, fields in grouped.items():
            updated = replace(updated, **{section: replace(getattr(updated, section), **fields)})
        return updated

    def to_flat_dict(self) -> dict[str, object]:
        """Flatten to ``{"index.curl_gain": 1.0, ...}`` for serialization/UI."""

        flat: dict[str, object] = {}
        for section in _SECTION_TYPES:
            params = getattr(self, section)
            for field_name in _SECTION_FIELDS[section]:
                value = getattr(params, field_name)
                flat[f"{section}.{field_name}"] = list(value) if isinstance(value, tuple) else value
        return flat


_SECTION_TYPES: dict[str, type] = {
    "index": FingerParams,
    "middle": FingerParams,
    "ring": FingerParams,
    "thumb": ThumbParams,
    "dexpilot": DexPilotParams,
}

#: Which sections each retargeting mode actually reads. A tuning UI must hide
#: the rest: a slider that silently does nothing is the worst thing a tuning
#: tool can offer, and is exactly how ~20 dead CLI flags accumulated before.
MODE_SECTIONS: dict[str, tuple[str, ...]] = {
    "analytic": ("thumb", "index", "middle", "ring"),
    "refine": ("thumb", "index", "middle", "ring"),
    "vector": (),
    "dexpilot": ("dexpilot",),
}


def _fields_of(cls: type) -> frozenset[str]:
    from dataclasses import fields as dataclass_fields

    return frozenset(f.name for f in dataclass_fields(cls))


_SECTION_FIELDS: dict[str, frozenset[str]] = {
    section: _fields_of(cls) for section, cls in _SECTION_TYPES.items()
}

#: Number of individually tunable leaves, for sanity-checking the UI.
PARAMETER_COUNT = sum(len(fields) for fields in _SECTION_FIELDS.values())


#: Bare defaults, equal to the historical global tuning.
DEFAULT_PROFILE = RetargetProfile()
