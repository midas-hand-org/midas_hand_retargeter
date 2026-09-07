"""Config validation, and the dex-config export used by the optimizer modes."""

from __future__ import annotations

import pytest

from midas_hand_retargeter.config import (
    ANALYTIC_MODE,
    VECTOR_MODE,
    MidasRetargeterConfig,
)
from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES


@pytest.fixture()
def urdf_path():
    """The shipped URDF, or skip.

    Resolved through paths.find_mujoco_repo rather than a hardcoded
    ``../../midas_hand_mujoco``, which made these tests unrunnable from an
    installed wheel or a CI checkout of this repo alone.
    """

    from midas_hand_retargeter.paths import default_urdf_path

    try:
        return default_urdf_path()
    except Exception:
        pytest.skip("midas_hand_mujoco assets not available")


def test_default_config_exports_vector_dex_config(urdf_path):
    config = MidasRetargeterConfig(urdf_path=urdf_path, mode=VECTOR_MODE)
    dex_config = config.to_dex_config_dict()

    assert dex_config["type"] == "vector"
    assert dex_config["target_joint_names"] == list(ACTIVE_JOINT_NAMES)
    assert dex_config["target_link_human_indices"] == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [4, 3, 8, 7, 12, 11, 16, 15],
    ]


def test_config_accepts_pip_dip_lookup_coupling_mode(urdf_path):
    config = MidasRetargeterConfig(
        urdf_path=urdf_path, mode=VECTOR_MODE, coupling_mode="pip_dip_lookup"
    )
    assert config.to_dex_config_dict()["target_joint_names"] == list(ACTIVE_JOINT_NAMES)


def test_analytic_config_needs_no_urdf_at_all():
    """The default path must not depend on a sibling repo being present."""

    config = MidasRetargeterConfig()
    assert config.mode == ANALYTIC_MODE
    assert config.uses_optimizer is False


def test_missing_urdf_is_reported_clearly():
    config = MidasRetargeterConfig(urdf_path="/nonexistent/hand.urdf", mode=VECTOR_MODE)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        config.resolved_urdf_path()


def test_dexpilot_rejects_the_config_level_solver_knobs():
    """They are read from RetargetProfile.dexpilot in that mode, so setting
    them here did nothing at all -- mode="dexpilot", scaling_factor=99.0 solved
    at 1.15 and said nothing. That is the anti-pattern this package already
    guards in the other direction for the thumb policies."""

    from midas_hand_retargeter.config import DEXPILOT_MODE, MidasRetargeterConfig

    for field, value in (
        ("scaling_factor", 99.0),
        ("normal_delta", 0.5),
        ("huber_delta", 0.5),
        ("low_pass_alpha", 0.5),
    ):
        with pytest.raises(ValueError, match="no effect"):
            MidasRetargeterConfig(mode=DEXPILOT_MODE, **{field: value})
    # The defaults must still construct.
    assert MidasRetargeterConfig(mode=DEXPILOT_MODE).mode == DEXPILOT_MODE


def test_the_dead_postprocess_constants_are_gone():
    """15 module constants survived the per-finger params migration with no
    reader anywhere. A constant here is a knob the tuning UI cannot reach,
    which is how that set came to be silently dead."""

    from midas_hand_retargeter import postprocess

    for name in (
        "FINGER_MCP_PITCH_RANGE", "FINGER_PIP_RANGE", "FINGER_ABAD_DEADZONE",
        "FINGER_ABAD_LIMIT", "FINGER_ABAD_CURL_DAMPING", "THUMB_CMC_ROLL_RANGE",
        "THUMB_CMC_ROLL_DEADZONE", "THUMB_CMC_ROLL_SPAN", "THUMB_CMC_SIDE_OPEN",
        "THUMB_CMC_SIDE_RANGE", "THUMB_CMC_SIDE_NEUTRAL_ANGLE",
        "THUMB_CMC_SIDE_DEADZONE", "THUMB_MCP_RANGE", "THUMB_DIP_RANGE",
        "THUMB_DIP_MCP_FOLLOW",
    ):
        assert not hasattr(postprocess, name), f"{name} came back"


def test_the_dexpilot_mode_and_params_are_exported():
    """Consumers were deep-importing them from .config / .params because the
    package did not re-export the mode it made primary."""

    import midas_hand_retargeter as pkg

    assert pkg.DEXPILOT_MODE == "dexpilot"
    assert pkg.DexPilotParams().abduction_limit == 0.25
    for name in ("DEXPILOT_MODE", "DexPilotParams"):
        assert name in pkg.__all__
