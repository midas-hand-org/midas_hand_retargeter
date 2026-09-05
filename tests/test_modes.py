"""The three retargeting modes, and the guarantees that make `analytic` safe.

``analytic`` became the default because the analytic layer already overwrote
every actuated joint: the optimizer's solution never reached the hand. These
tests pin that the switch is behaviour-preserving and that the default path
needs neither dex_retargeting nor torch.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.config import (
    ANALYTIC_MODE,
    REFINE_MODE,
    VECTOR_MODE,
    MidasRetargeterConfig,
)

from ._poses import GOLDEN_POSES

#: Tests that compare against the optimizer need the ``vector`` extra. The
#: analytic tests must still run without it — that is the whole point.
needs_optimizer = pytest.mark.skipif(
    importlib.util.find_spec("dex_retargeting") is None,
    reason="requires the [vector] extra (dex_retargeting + torch)",
)


def test_analytic_is_the_default():
    assert MidasRetargeterConfig().mode == ANALYTIC_MODE
    assert MidasRetargeterConfig().uses_optimizer is False


@needs_optimizer
def test_vector_mode_disables_the_analytic_layer():
    config = MidasRetargeterConfig(mode=VECTOR_MODE)
    assert config.finger_postprocess is False
    assert config.thumb_postprocess is False
    assert config.uses_optimizer is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"finger_postprocess": False},
        {"thumb_postprocess": False},
        {"finger_postprocess": False, "thumb_postprocess": False},
    ],
)
def test_analytic_refuses_a_disabled_analytic_layer(kwargs):
    """Otherwise those joints stay at 0.0 rad — fully extended, a real motion."""

    with pytest.raises(ValueError, match="requires the analytic layer"):
        MidasRetargeterConfig(mode=ANALYTIC_MODE, **kwargs)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported mode"):
        MidasRetargeterConfig(mode="magic")


def test_coupling_mode_is_normalized_once():
    """config.py used to reject what adaptor.py accepted."""

    assert MidasRetargeterConfig(coupling_mode="Fixed_Passive").coupling_mode == ("fixed_passive")
    with pytest.raises(ValueError, match="Unsupported coupling_mode"):
        MidasRetargeterConfig(coupling_mode="nope")


@needs_optimizer
def test_analytic_matches_refine_exactly():
    """The behaviour-preservation claim behind changing the default.

    The analytic layer writes all 13 actuated joints in both modes, so dropping
    the discarded solve must change nothing at all — including the passive
    slots, which come from passive_fixed_qpos either way.
    """

    analytic = MidasHandRetargeter.create(mode=ANALYTIC_MODE)
    refine = MidasHandRetargeter.create(mode=REFINE_MODE)

    for name, landmarks in GOLDEN_POSES:
        left = analytic.retarget_landmarks(landmarks)
        right = refine.retarget_landmarks(landmarks)
        np.testing.assert_array_equal(
            left.active_vector(),
            right.active_vector(),
            err_msg=f"active joints diverged on pose {name!r}",
        )
        np.testing.assert_array_equal(
            np.asarray(left.robot_qpos),
            np.asarray(right.robot_qpos),
            err_msg=f"robot_qpos diverged on pose {name!r}",
        )


def test_analytic_exposes_no_optimizer():
    retargeter = MidasHandRetargeter.create(mode=ANALYTIC_MODE)
    assert retargeter.dex_retargeting is None
    with pytest.raises(RuntimeError, match="needs the optimizer"):
        retargeter.set_qpos(np.zeros(19))


@needs_optimizer
def test_analytic_keeps_the_public_joint_layout():
    """robot_qpos indexing is a public contract shared with the optimizer path."""

    analytic = MidasHandRetargeter.create(mode=ANALYTIC_MODE)
    refine = MidasHandRetargeter.create(mode=REFINE_MODE)
    assert analytic.robot_joint_names == refine.robot_joint_names
    assert analytic.fixed_joint_names == refine.fixed_joint_names


def test_analytic_needs_neither_dex_retargeting_nor_torch():
    """Run in a subprocess so an already-imported torch cannot mask a regression."""

    script = (
        "import sys;"
        "from midas_hand_retargeter import MidasHandRetargeter;"
        "import numpy as np;"
        "r = MidasHandRetargeter.create();"
        "r.retarget_landmarks(np.zeros((21, 3)));"
        "assert 'torch' not in sys.modules, 'torch was imported';"
        "assert 'dex_retargeting' not in sys.modules, 'dex_retargeting was imported';"
        "assert 'pinocchio' not in sys.modules, 'pinocchio was imported';"
        "print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_importing_adaptor_stays_dependency_free():
    """Six call sites across two other repos import only the mode constants."""

    script = (
        "import sys;"
        "from midas_hand_retargeter.adaptor import "
        "PIP_DIP_LOOKUP_MODE, SUPPORTED_COUPLING_MODES;"
        "assert 'torch' not in sys.modules;"
        "assert 'dex_retargeting' not in sys.modules;"
        "print(PIP_DIP_LOOKUP_MODE)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "pip_dip_lookup" in result.stdout


@needs_optimizer
def test_refine_warm_start_tracks_the_commanded_pose():
    """The temporal regularizer must anchor to what was actually commanded.

    SeqRetargeting stores its own raw solve as last_qpos, but the analytic
    layer then overwrites every actuated joint, so the solver used to be pulled
    toward a pose that never reached the hand (measured 1.5 rad off over a
    closing sweep).
    """

    from ._poses import hand_pose

    retargeter = MidasHandRetargeter.create(mode=REFINE_MODE)
    worst = 0.0
    for step in range(20):
        amount = 1.4 * step / 19
        result = retargeter.retarget_landmarks(
            hand_pose(curls=(amount,) * 3, thumb_curl=amount, thumb_oppose=0.9 * amount)
        )
        warm_start = np.asarray(retargeter.dex_retargeting.last_qpos, dtype=float)
        worst = max(worst, float(np.abs(warm_start - result.active_vector()).max()))

    assert worst < 1e-6, f"warm start drifted {worst:.4f} rad from the command"
