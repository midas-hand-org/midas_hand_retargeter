"""MIDAS hand retargeting constants."""

from __future__ import annotations

FINGER_NAMES = ("index", "middle", "ring")

# Active joints in the MuJoCo model. The passive DIP/linkage joints are fixed
# during option-1 retargeting and handled downstream by MuJoCo closed-loop
# constraints or the hardware API's PIP-DIP mapping.
ACTIVE_JOINT_NAMES = (
    "index_mcp_abad_joint",
    "index_mcp_pitch_joint",
    "index_pip_joint",
    "middle_mcp_abad_joint",
    "middle_mcp_pitch_joint",
    "middle_pip_joint",
    "ring_mcp_abad_joint",
    "ring_mcp_pitch_joint",
    "ring_pip_joint",
    "thumb_cmc_roll_joint",
    "thumb_cmc_side_joint",
    "thumb_mcp_joint",
    "thumb_dip_joint",
)

# Motor-command order expected by midas_hand_api.HandConfig defaults.
HARDWARE_MOTOR_JOINT_NAMES = (
    "thumb_dip_joint",
    "thumb_mcp_joint",
    "thumb_cmc_side_joint",
    "thumb_cmc_roll_joint",
    "index_pip_joint",
    "index_mcp_pitch_joint",
    "index_mcp_abad_joint",
    "middle_pip_joint",
    "middle_mcp_pitch_joint",
    "middle_mcp_abad_joint",
    "ring_pip_joint",
    "ring_mcp_pitch_joint",
    "ring_mcp_abad_joint",
)

PASSIVE_FIXED_JOINT_NAMES = (
    "index_dip_linkage_joint",
    "index_dip_joint",
    "middle_dip_linkage_joint",
    "middle_dip_joint",
    "ring_dip_linkage_joint",
    "ring_dip_joint",
)

# Baseline vector task based on the working Jun/dex-retargeting teleop setup:
# every vector starts at the palm/wrist frame and targets each fingertip plus
# the previous distal landmark. This gives the optimizer enough information to
# excite the finger MCP/PIP curl without optimizing passive DIP joints.
DEFAULT_TARGET_ORIGIN_LINK_NAMES = (
    "palm_base",
    "palm_base",
    "palm_base",
    "palm_base",
    "palm_base",
    "palm_base",
    "palm_base",
    "palm_base",
)
DEFAULT_TARGET_TASK_LINK_NAMES = (
    "thumb_tip",
    "thumb_mcp",
    "index_tip",
    "index_pip_link",
    "middle_tip",
    "middle_pip_link",
    "ring_tip",
    "ring_pip_link",
)

# MediaPipe/MANO landmark indices. Pinky is intentionally omitted because the
# MIDAS hand has three fingers plus thumb.
DEFAULT_TARGET_LINK_HUMAN_INDICES = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (4, 3, 8, 7, 12, 11, 16, 15),
)
