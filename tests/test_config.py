from pathlib import Path

from midas_hand_retargeter.config import MidasRetargeterConfig
from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES


def test_default_config_exports_vector_dex_config():
    workspace = Path(__file__).resolve().parents[2]
    urdf_path = workspace / "midas_hand_mujoco" / "assets" / "midas_description" / "midas_hand_urdf.urdf"
    config = MidasRetargeterConfig(urdf_path=urdf_path)
    dex_config = config.to_dex_config_dict()

    assert dex_config["type"] == "vector"
    assert dex_config["target_joint_names"] == list(ACTIVE_JOINT_NAMES)
    assert dex_config["target_link_human_indices"] == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [4, 3, 8, 7, 12, 11, 16, 15],
    ]
