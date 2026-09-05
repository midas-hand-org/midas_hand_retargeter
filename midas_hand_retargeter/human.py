"""Human-hand landmark utilities for MIDAS retargeting.

Required input convention (all landmark sources must match this)
----------------------------------------------------------------
Every entry point expects a ``(21, 3)`` float array using the **MediaPipe hand
landmark ordering** (wrist=0, thumb=1..4, index=5..8, middle=9..12, ring=13..16,
pinky=17..20) in a **metric, right-handed** coordinate frame. Concretely, for a
right hand the reference axes are: ``+Y`` distal (wrist -> fingertips), ``+X``
lateral (index MCP -> ring MCP), ``+Z`` dorsal (back of hand), i.e.
``cross(+X, +Y) = +Z``.

Only the *chirality* of this frame is load-bearing for the geometric
postprocess: ``postprocess._palm_basis`` derives forward/lateral directions from
the landmarks themselves, so an arbitrary rotation of the input washes out, but
the palm normal is ``cross(lateral, forward)`` — a pseudovector that **flips
under a reflection**. A source delivered in a mirrored (left-handed) frame will
curl fingers correctly (curl is a magnitude) yet invert thumb opposition/side
and finger splay. New sources (e.g. the Manus glove) must therefore preserve
chirality: apply a proper rotation to align axes, and an *even* number of axis
reflections. See ``midas_hand_teleop.manus_glove.manus_bridge`` for the glove's
chirality-correcting remap.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .constants import DEFAULT_TARGET_LINK_HUMAN_INDICES

OPERATOR2MANO_RIGHT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
)
OPERATOR2MANO_LEFT = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
)


def as_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Validate and normalize a MediaPipe-style 21x3 landmark array."""

    result = np.asarray(landmarks, dtype=np.float32)
    if result.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {result.shape}")
    return result


def landmarks_to_vectors(
    landmarks: np.ndarray,
    target_link_human_indices: Sequence[Sequence[int]] = DEFAULT_TARGET_LINK_HUMAN_INDICES,
) -> np.ndarray:
    """Convert 21x3 human landmarks into the configured vector objective.

    ``target_link_human_indices`` has two rows: the origin landmark index for
    each vector and the task landmark index for each vector. The default uses
    palm/wrist-to-tip and palm/wrist-to-distal vectors for thumb, index, middle,
    and ring.
    """

    points = as_landmarks(landmarks)
    indices = np.asarray(target_link_human_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError(
            "target_link_human_indices must have shape (2, num_vectors)"
        )
    origin_indices = indices[0]
    task_indices = indices[1]
    return points[task_indices] - points[origin_indices]


def estimate_frame_from_hand_points(landmarks: np.ndarray) -> np.ndarray:
    """Estimate a wrist-centered orientation from MediaPipe world landmarks.

    This is inherited from the dex-retargeting example: it builds a stable palm
    frame from wrist, index MCP, and middle MCP landmarks, then downstream code
    rotates MediaPipe world points into a MANO-like coordinate convention.
    """

    points = as_landmarks(landmarks)[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(centered)
    normal = v[2, :]
    x_axis = x_vector - np.sum(x_vector * normal) * normal
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
    z_axis = np.cross(x_axis, normal)
    if np.sum(z_axis * (centered[1] - centered[2])) < 0:
        normal *= -1
        z_axis *= -1
    return np.stack([x_axis, normal, z_axis], axis=1)


def mediapipe_world_to_mano_landmarks(
    world_landmarks: np.ndarray,
    *,
    hand_type: str = "Right",
) -> np.ndarray:
    """Convert MediaPipe world landmarks to the wrist-centered MANO-like frame.

    ``hand_type`` selects the right/left operator-to-MANO transform. For a
    right-hand robot driven by a left physical hand, convert first with the
    physical hand type and then call ``mirror_landmarks_for_robot_hand``.
    """

    keypoints = as_landmarks(world_landmarks)
    centered = keypoints - keypoints[0:1, :]
    wrist_frame = estimate_frame_from_hand_points(centered)
    operator2mano = (
        OPERATOR2MANO_RIGHT if hand_type.lower() == "right" else OPERATOR2MANO_LEFT
    )
    return (centered @ wrist_frame @ operator2mano).astype(np.float32)


def mirror_landmarks_for_robot_hand(
    landmarks: np.ndarray,
    *,
    source_hand_type: str,
    target_hand_type: str = "Right",
    axis: str = "y",
) -> np.ndarray:
    """Mirror landmarks when a human hand drives the opposite robot hand.

    The vector retargeter is handed: a right-hand robot expects right-hand
    landmark geometry. When the detected physical hand is the opposite side,
    mirror the lateral axis in the wrist-centered MANO-like frame before
    building target vectors.
    """

    points = as_landmarks(landmarks).copy()
    if source_hand_type.lower() == target_hand_type.lower() or axis == "none":
        return points

    axis_indices = {
        "x": (0,),
        "y": (1,),
        "z": (2,),
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
        "xyz": (0, 1, 2),
    }
    if axis not in axis_indices:
        raise ValueError(
            f"Unsupported mirror axis {axis!r}; expected one of {sorted(axis_indices)}"
        )
    points[:, axis_indices[axis]] *= -1.0
    return points.astype(np.float32)
