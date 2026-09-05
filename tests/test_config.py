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
