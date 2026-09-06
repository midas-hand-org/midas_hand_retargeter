"""Thumb posture in the Cartesian retargeting modes.

The MIDAS thumb is proportionally long: palm_base->thumb_tip is 180.1 mm
against index 218.0 mm, a ratio of 0.83 where a human's is ~0.58. Two segments
have no human counterpart — a 41.4 mm palm->CMC offset and a 24.7 mm CMC
mechanism. Comparing the operator's wrist->thumb-tip against the robot's
palm->thumb-tip therefore forces the robot to curl its thumb to shorten itself.

Measured on a 2066-frame real glove trace before the fix: thumb MCP and DIP
bending in opposite directions on 42% of frames, MCP hyperextending to
+0.60 rad.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from midas_hand_retargeter import MidasHandRetargeter, RetargetProfile
from midas_hand_retargeter.config import DEXPILOT_MODE, MidasRetargeterConfig

from ._poses import hand_pose

needs_optimizer = pytest.mark.skipif(
    importlib.util.find_spec("dex_retargeting") is None,
    reason="requires the [vector] extra",
)

#: nlopt's set_joint_limit adds an epsilon of 1e-3, so the true ceiling is
#: +0.001, not 0.0. Assert with tolerance or this fails for the wrong reason.
BOUND_EPSILON = 2e-3

TRACE = pathlib.Path(__file__).parent / "goldens" / "glove_trace_thumb.npz"


def s_curve_fraction(mcp, dip, deadband=0.05):
    """Fraction of frames where thumb MCP and DIP bend in OPPOSITE directions.

    Anatomically impossible, and the visible symptom operators report. The
    deadband keeps values sitting on a bound out of the count.
    """

    mcp, dip = np.asarray(mcp), np.asarray(dip)
    active = (np.abs(mcp) > deadband) & (np.abs(dip) > deadband)
    return float((active & (np.sign(mcp) != np.sign(dip))).mean())


def _sweep(retargeter, frames):
    mcp, dip = [], []
    for frame in frames:
        active = retargeter.retarget_landmarks(frame).active_joint_positions
        mcp.append(active["thumb_mcp_joint"])
        dip.append(active["thumb_dip_joint"])
    return np.array(mcp), np.array(dip)


def _synthetic_sweep():
    return [
        hand_pose(thumb_curl=c, thumb_oppose=o, curls=(c,) * 3)
        for c in np.linspace(0.0, 1.3, 8)
        for o in (0.0, 0.5, 1.0)
    ]


# --- policy plumbing, no solver required ---------------------------------

def test_thumb_policies_default_on_for_dexpilot():
    config = MidasRetargeterConfig(mode=DEXPILOT_MODE)
    assert config.thumb_flexion_only is True
    assert config.thumb_root_link == "thumb_cmc_side"


def test_thumb_policies_are_rejected_for_modes_that_cannot_act_on_them():
    """Silently ignoring a setting is worse than refusing it."""

    for kwargs in ({"thumb_flexion_only": False}, {"thumb_root_link": None}):
        with pytest.raises(ValueError, match="only applies to mode"):
            MidasRetargeterConfig(mode="analytic", **kwargs)


def test_analytic_mode_has_no_solver_narrowing():
    """This is what keeps the analytic golden set provably untouched."""

    assert MidasHandRetargeter.create()._solver_joint_limits == {}


# --- the two fixes -------------------------------------------------------

@needs_optimizer
def test_solver_cannot_hyperextend_the_thumb():
    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    mcp, dip = _sweep(retargeter, _synthetic_sweep())
    assert mcp.max() <= BOUND_EPSILON, f"thumb MCP reached {mcp.max():+.4f}"
    assert dip.max() <= BOUND_EPSILON, f"thumb DIP reached {dip.max():+.4f}"


@needs_optimizer
def test_flexion_only_reduces_the_s_curve():
    """A/B on identical input, so the comparison cannot drift with the fixture."""

    frames = _synthetic_sweep()
    loose = MidasHandRetargeter.create(mode=DEXPILOT_MODE, thumb_flexion_only=False)
    tight = MidasHandRetargeter.create(mode=DEXPILOT_MODE)

    loose_mcp, loose_dip = _sweep(loose, frames)
    tight_mcp, tight_dip = _sweep(tight, frames)

    assert s_curve_fraction(tight_mcp, tight_dip) <= s_curve_fraction(loose_mcp, loose_dip)
    assert tight_mcp.max() < loose_mcp.max()


@needs_optimizer
def test_nlopt_bounds_and_the_warm_start_clip_agree():
    """They are two separate arrays, and disagreeing crashes nlopt mid-teleop.

    Optimizer.set_joint_limit writes nlopt's bounds; SeqRetargeting.retarget
    clips its warm start against its OWN copy. If they diverge, nlopt gets a
    start point outside its box and raises invalid_argument.
    """

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    sequential = retargeter.dex_retargeting
    upper = np.asarray(sequential.optimizer.opt.get_upper_bounds())
    names = list(retargeter.active_joint_names)

    for joint, (_, expected_upper) in retargeter._solver_joint_limits.items():
        index = names.index(joint)
        assert sequential.joint_limits[index][1] == pytest.approx(expected_upper)
        assert upper[index] == pytest.approx(expected_upper, abs=BOUND_EPSILON)


@needs_optimizer
def test_thumb_root_rebase_rewrites_every_derived_cache():
    """A partial rewrite makes the solver difference links it never computed."""

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    optimizer = retargeter.dex_retargeting.optimizer
    index = retargeter._DEXPILOT_THUMB_ROOT_VECTOR

    assert optimizer.origin_link_names[index] == "thumb_cmc_side"
    assert np.asarray(optimizer.target_link_human_indices)[0][index] == 1
    assert "thumb_cmc_side" in optimizer.computed_link_names
    # The other three base-rooted vectors still need the palm.
    assert "palm_base" in optimizer.computed_link_names

    # The invariant the objective silently depends on.
    for position, name in enumerate(optimizer.origin_link_names):
        assert optimizer.computed_link_names[optimizer.origin_link_indices[position]] == name
    for position, name in enumerate(optimizer.task_link_names):
        assert optimizer.computed_link_names[optimizer.task_link_indices[position]] == name


@needs_optimizer
def test_palm_root_is_left_byte_identical():
    """The escape hatch must be a true no-op, not an approximation."""

    retargeter = MidasHandRetargeter.create(
        mode=DEXPILOT_MODE, thumb_root_link=None, thumb_flexion_only=False
    )
    optimizer = retargeter.dex_retargeting.optimizer
    index = retargeter._DEXPILOT_THUMB_ROOT_VECTOR
    assert optimizer.origin_link_names[index] == "palm_base"
    assert np.asarray(optimizer.target_link_human_indices)[0][index] == 0
    assert retargeter._solver_joint_limits == {}


@needs_optimizer
def test_neutral_calibration_cannot_reintroduce_hyperextension():
    """The recentring rescales by the joint's upper bound, so it must use the
    narrowed one. With the model's +1.57 a neutral of -0.30 and a raw solve of
    -0.10 became +0.17 rad — hyperextension commanded downstream of the solver
    that was forbidden to produce it."""

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    for neutral, raw in ((-0.3, -0.1), (-0.5, -0.2), (-0.8, -0.4)):
        value = retargeter._neutral_calibrated_value("thumb_mcp_joint", raw, neutral)
        assert value <= BOUND_EPSILON, f"neutral={neutral} raw={raw} gave {value:+.4f}"


@needs_optimizer
def test_thumb_vector_scale_touches_only_the_thumb_rows():
    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    retargeter.profile = RetargetProfile().with_values(
        {"dexpilot.scaling_factor": 1.0, "dexpilot.thumb_vector_scale": 1.5}
    )
    retargeter.retarget_landmarks(hand_pose(thumb_curl=0.4))

    scaling = np.asarray(retargeter.dex_retargeting.optimizer.scaling)
    assert scaling.shape == (10, 1)
    optimizer = retargeter.dex_retargeting.optimizer
    for position, (origin, task) in enumerate(
        zip(optimizer.origin_link_names, optimizer.task_link_names, strict=True)
    ):
        expected = 1.5 if ("thumb" in origin or "thumb" in task) else 1.0
        assert scaling[position, 0] == pytest.approx(expected)


# --- regression against real glove data ----------------------------------

@needs_optimizer
def test_thumb_posture_on_a_real_glove_trace():
    """Guards the measured result against a 200-frame slice of a real capture.

    Synthetic poses cannot reproduce the failure this fixes — it depends on the
    operator's actual thumb proportions — so the evidence is checked in.
    """

    if not TRACE.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"missing {TRACE}")
    frames = np.load(TRACE)["frames"]

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    retargeter.profile = RetargetProfile().with_values(
        {"dexpilot.scaling_factor": 1.2}
    )
    mcp, dip = _sweep(retargeter, frames)

    assert mcp.max() <= BOUND_EPSILON, "thumb hyperextended on real data"
    # Was 42% before the fix.
    assert s_curve_fraction(mcp, dip) < 0.05
    # Was a mean |bend| of 0.71 rad; the fix brings it to ~0.37.
    assert np.mean(np.abs(np.stack([mcp, dip]))) < 0.5
