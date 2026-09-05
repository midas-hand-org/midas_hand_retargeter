"""Regenerate the analytic-layer behaviour-freeze golden.

    python -m tests.generate_goldens

Run this ONLY to establish a new baseline after a deliberate, reviewed
behaviour change. During the retargeting refactor the golden must stay
byte-identical; if this script produces a diff, that is the signal to stop and
explain why, not to commit the new numbers.
"""

from __future__ import annotations

import json
import pathlib

from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES
from midas_hand_retargeter.postprocess import (
    finger_joint_targets_from_landmarks,
    thumb_joint_targets_from_landmarks,
)
from midas_hand_retargeter.tuning import DEFAULT_TUNING

from ._poses import GOLDEN_POSES

GOLDEN_PATH = pathlib.Path(__file__).parent / "goldens" / "analytic_targets.json"


def analytic_targets(landmarks, tuning=DEFAULT_TUNING):
    """All 13 active-joint targets for one pose, in ACTIVE_JOINT_NAMES order."""

    targets = {}
    targets.update(finger_joint_targets_from_landmarks(landmarks, tuning))
    targets.update(thumb_joint_targets_from_landmarks(landmarks, tuning))
    return [float(targets[name]) for name in ACTIVE_JOINT_NAMES]


def build():
    return {
        "schema": 1,
        "description": (
            "Analytic (postprocess) joint targets for a fixed synthetic pose set. "
            "Freezes default-tuning behaviour across the retargeting refactor."
        ),
        "tuning": "DEFAULT_TUNING",
        "joint_names": list(ACTIVE_JOINT_NAMES),
        "poses": {name: analytic_targets(kp) for name, kp in GOLDEN_POSES},
    }


def main():
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(build(), indent=2, sort_keys=False) + "\n")
    print(f"wrote {GOLDEN_PATH} ({len(GOLDEN_POSES)} poses)")


if __name__ == "__main__":
    main()
