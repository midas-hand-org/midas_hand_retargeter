"""Save and load retargeting profiles, including the neutral calibration.

Format is JSON, not YAML: it is stdlib (the package has no yaml dependency any
more), and it is what the web tuner already speaks.

A preset is meant to be a *complete, reproducible* artifact, so it carries the
neutral calibration too. That calibration used to live only in memory and was
recaptured on every run, which meant a tuning session could not actually be
saved.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .model import MIDAS_RIGHT_HAND, HandModel
from .params import RetargetProfile

#: Bumped when the on-disk shape changes in a way older readers cannot handle.
SCHEMA_VERSION = 1

#: Where presets live unless the caller says otherwise.
DEFAULT_PRESET_DIR = Path.home() / ".midas_hand" / "retarget_presets"


def _validate_ranges(profile: RetargetProfile, model: HandModel) -> None:
    """Reject a profile that could command outside the robot's joint limits.

    Checked at load and at edit time, not only in CI: the whole point of the
    tuner is that profiles are edited outside CI.
    """

    problems: list[str] = []
    for finger, params in profile.fingers():
        for joint_suffix, commanded in (
            ("mcp_pitch", params.mcp_pitch_range),
            ("pip", params.pip_range),
        ):
            joint = f"{finger}_{joint_suffix}_joint"
            lower, upper = model.limits(joint)
            for value in commanded:
                if not (lower - 1e-9 <= value <= upper + 1e-9):
                    problems.append(
                        f"{finger}.{joint_suffix}_range value {value} outside "
                        f"{joint} limits [{lower}, {upper}]"
                    )
        limit = params.splay_limit
        lower, upper = model.limits(f"{finger}_mcp_abad_joint")
        if limit > min(abs(lower), abs(upper)) + 1e-9:
            problems.append(
                f"{finger}.splay_limit {limit} exceeds "
                f"{finger}_mcp_abad_joint limits [{lower}, {upper}]"
            )

    thumb = profile.thumb
    for field_name, joint in (
        ("mcp_range", "thumb_mcp_joint"),
        ("dip_range", "thumb_dip_joint"),
        ("cmc_side_range", "thumb_cmc_side_joint"),
        ("cmc_roll_range", "thumb_cmc_roll_joint"),
    ):
        lower, upper = model.limits(joint)
        for value in getattr(thumb, field_name):
            if not (lower - 1e-9 <= value <= upper + 1e-9):
                problems.append(
                    f"thumb.{field_name} value {value} outside {joint} limits [{lower}, {upper}]"
                )

    if problems:
        raise ValueError(
            "Profile would command outside the robot's joint limits:\n  " + "\n  ".join(problems)
        )


def to_dict(
    profile: RetargetProfile,
    *,
    neutral_offsets: Mapping[str, float] | None = None,
    hand: str = "midas_right",
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "hand": hand,
        "name": profile.name,
        "source": profile.source,
        "parameters": profile.to_flat_dict(),
        "neutral_offsets": dict(neutral_offsets or {}),
    }


def from_dict(
    payload: Mapping,
    *,
    model: HandModel = MIDAS_RIGHT_HAND,
    hand: str = "midas_right",
) -> tuple[RetargetProfile, dict[str, float]]:
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported preset schema_version {version!r}; "
            f"this build reads version {SCHEMA_VERSION}."
        )

    stored_hand = payload.get("hand", hand)
    if stored_hand != hand:
        raise ValueError(
            f"Preset was tuned for hand {stored_hand!r}, not {hand!r}. "
            "Loading it would command the wrong geometry."
        )

    profile = RetargetProfile(
        name=payload.get("name", "preset"),
        source=payload.get("source", "unknown"),
    ).with_values(payload.get("parameters", {}))
    _validate_ranges(profile, model)
    neutral = {str(k): float(v) for k, v in (payload.get("neutral_offsets") or {}).items()}
    unknown = set(neutral) - set(model.joint_names)
    if unknown:
        raise ValueError(f"Preset neutral_offsets name unknown joints: {sorted(unknown)}")
    return profile, neutral


def save(
    path: str | Path,
    profile: RetargetProfile,
    *,
    neutral_offsets: Mapping[str, float] | None = None,
    model: HandModel = MIDAS_RIGHT_HAND,
) -> Path:
    _validate_ranges(profile, model)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict(profile, neutral_offsets=neutral_offsets)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def resolve(name_or_path: str | Path, *, directory: str | Path = DEFAULT_PRESET_DIR) -> Path:
    """Turn a preset NAME or a path into a path.

    The browser has always addressed presets by bare name while ``--preset``
    required a full path, so the name shown in the UI after saving was not the
    thing you could then type on the command line. Both go through here.
    """

    candidate = Path(name_or_path).expanduser()
    if candidate.suffix == ".json" or candidate.is_absolute() or len(candidate.parts) > 1:
        return candidate
    return Path(directory).expanduser() / f"{candidate.name}.json"


def load(
    path: str | Path, *, model: HandModel = MIDAS_RIGHT_HAND
) -> tuple[RetargetProfile, dict[str, float]]:
    source = resolve(path)
    if not source.exists():
        available = ", ".join(p.stem for p in list_presets()) or "none saved"
        raise FileNotFoundError(f"No such preset: {source} (available: {available})")
    return from_dict(json.loads(source.read_text()), model=model)


def list_presets(directory: str | Path = DEFAULT_PRESET_DIR) -> list[Path]:
    folder = Path(directory).expanduser()
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.json"))
