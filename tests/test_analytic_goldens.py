"""The analytic layer's output is frozen against a checked-in golden.

This is the safety net for the retargeting refactor: the analytic map is what
actually drives the hand (the vector optimizer's solution is overwritten for
all 13 active joints in the default config), so any unintended change to curl,
splay, thumb angles, output ranges or blend weights must fail here.

Tolerance is 1e-6 rad. The analytic layer computes in float32 internally
(``_unit`` casts), so a pure refactor may reassociate arithmetic and drift by
~1e-7; a real behaviour change moves joints by 1e-3 or more, so this gap is
wide enough to be stable and tight enough to catch anything meaningful.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES

from ._poses import GOLDEN_POSES
from .generate_goldens import GOLDEN_PATH, analytic_targets

TOLERANCE_RAD = 1e-6


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():  # pragma: no cover - fixture is committed
        pytest.fail(
            f"Missing golden {GOLDEN_PATH}. Regenerate with "
            "`python -m tests.generate_goldens` and review the diff."
        )
    return json.loads(GOLDEN_PATH.read_text())


def test_golden_matches_current_joint_order(golden):
    assert tuple(golden["joint_names"]) == ACTIVE_JOINT_NAMES


def test_golden_covers_every_pose(golden):
    assert set(golden["poses"]) == {name for name, _ in GOLDEN_POSES}


@pytest.mark.parametrize("name,landmarks", GOLDEN_POSES, ids=[n for n, _ in GOLDEN_POSES])
def test_analytic_targets_match_golden(golden, name, landmarks):
    expected = np.asarray(golden["poses"][name], dtype=float)
    actual = np.asarray(analytic_targets(landmarks), dtype=float)
    deviation = np.abs(actual - expected)
    worst = int(np.argmax(deviation))
    assert deviation.max() <= TOLERANCE_RAD, (
        f"pose {name!r} drifted: worst joint {ACTIVE_JOINT_NAMES[worst]} "
        f"expected {expected[worst]:.9f}, got {actual[worst]:.9f} "
        f"(delta {deviation[worst]:.3e} > {TOLERANCE_RAD:g})"
    )


def test_golden_actually_exercises_every_joint(golden):
    """A golden that never moves a joint cannot detect a regression in it."""

    table = np.array([golden["poses"][name] for name, _ in GOLDEN_POSES])
    spans = table.max(axis=0) - table.min(axis=0)
    idle = [ACTIVE_JOINT_NAMES[i] for i, span in enumerate(spans) if span < 1e-6]
    assert not idle, f"golden pose set never moves: {idle}"
