"""Human-hand landmark utilities for MIDAS retargeting."""

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


# Rigid palm landmarks used to estimate the hand frame: wrist + index/middle/ring
# MCP. These move together as a near-rigid body, so they define orientation far
# more stably than fingertip or PIP landmarks.
PALM_LANDMARK_INDICES = (0, 5, 9, 13)


def _canonical_palm_template() -> np.ndarray:
    """Palm landmark template expressed in the legacy wrist-frame convention.

    The template is built by running the legacy 3-point estimator on a
    representative neutral right hand and re-expressing the rigid palm landmarks
    in that frame. Because the template therefore lives in exactly the same axis
    convention as ``estimate_frame_from_hand_points``, ``kabsch_palm_frame``
    reproduces the legacy frame on the neutral pose and ``OPERATOR2MANO_*`` needs
    no retuning. Only the relative geometry of these points matters; the absolute
    scale and the small out-of-plane offsets (which keep the four points
    non-coplanar so the fit is full-rank) are arbitrary.
    """

    canonical = np.zeros((21, 3), dtype=np.float32)
    canonical[0] = (0.0, 0.0, 0.0)  # wrist
    canonical[5] = (0.045, -0.065, 0.010)  # index MCP
    canonical[9] = (0.010, -0.075, 0.0)  # middle MCP
    canonical[13] = (-0.025, -0.070, 0.012)  # ring MCP
    frame = estimate_frame_from_hand_points(canonical)
    palm = canonical[list(PALM_LANDMARK_INDICES)] - canonical[0:1, :]
    return (palm @ frame).astype(np.float32)


_PALM_TEMPLATE = _canonical_palm_template()


def kabsch_palm_frame(
    landmarks: np.ndarray,
    *,
    prev_frame: np.ndarray | None = None,
    template: np.ndarray = _PALM_TEMPLATE,
    degenerate_ratio: float = 0.04,
) -> np.ndarray:
    """Estimate a wrist-centered hand frame by Kabsch-aligning the palm.

    Drop-in replacement for ``estimate_frame_from_hand_points`` that is far less
    sensitive to per-landmark depth jitter. Instead of taking the plane normal of
    three points (which any noise tilts), it solves the orthogonal Procrustes
    problem over the four rigid palm landmarks against a canonical template and
    returns the optimal proper rotation. Using four points least-squares-averages
    the noise, and the determinant correction guarantees a right-handed frame, so
    there are no spurious per-axis sign flips of the kind the legacy estimator can
    produce.

    The return value is a ``(3, 3)`` matrix whose columns are the hand-frame axes
    in camera coordinates, matching the ``centered @ frame`` convention used by
    ``mediapipe_world_to_mano_landmarks``.

    Sign-continuity: viewed near edge-on, the palm landmarks become nearly
    coplanar and the out-of-plane axis is ill-conditioned (small trailing
    singular value), which is exactly when a frame estimate flips. When that
    happens and ``prev_frame`` is supplied, the previous good frame is held
    instead of emitting a flipped one.
    """

    points = as_landmarks(landmarks)
    palm = points[list(PALM_LANDMARK_INDICES)] - points[0:1, :]

    # Procrustes estimates rotation only, so remove each cloud's centroid.
    a = palm - palm.mean(axis=0, keepdims=True)
    b = np.asarray(template, dtype=np.float32)
    b = b - b.mean(axis=0, keepdims=True)

    # argmin over rotations R of ||a @ R - b||: with H = a^T b = U S V^T,
    # R = U diag(1, 1, det(U V^T)) V^T. The diag term forces a proper rotation.
    h = a.T @ b
    u, s, vt = np.linalg.svd(h)
    d = float(np.sign(np.linalg.det(u @ vt)))
    correction = np.diag([1.0, 1.0, d])
    frame = (u @ correction @ vt).astype(np.float32)

    near_degenerate = s[0] <= 1e-9 or (s[-1] / s[0]) < degenerate_ratio
    if prev_frame is not None and near_degenerate:
        return np.asarray(prev_frame, dtype=np.float32)
    return frame


def make_kabsch_frame_fn(**kwargs):
    """Build a stateful Kabsch frame estimator with sign-continuity.

    Returns a callable ``frame_fn(centered_landmarks) -> (3, 3)`` suitable for
    ``mediapipe_world_to_mano_landmarks(frame_fn=...)``. It remembers the previous
    good frame and feeds it to ``kabsch_palm_frame`` so near-degenerate (edge-on)
    views hold the last orientation instead of flipping. Keyword arguments are
    forwarded to ``kabsch_palm_frame`` (e.g. ``degenerate_ratio``).
    """

    state: dict[str, np.ndarray | None] = {"prev": None}

    def frame_fn(centered_landmarks: np.ndarray) -> np.ndarray:
        frame = kabsch_palm_frame(
            centered_landmarks, prev_frame=state["prev"], **kwargs
        )
        state["prev"] = frame
        return frame

    return frame_fn


def mediapipe_world_to_mano_landmarks(
    world_landmarks: np.ndarray,
    *,
    hand_type: str = "Right",
    frame_fn=None,
) -> np.ndarray:
    """Convert MediaPipe world landmarks to the wrist-centered MANO-like frame.

    ``hand_type`` selects the right/left operator-to-MANO transform. For a
    right-hand robot driven by a left physical hand, convert first with the
    physical hand type and then call ``mirror_landmarks_for_robot_hand``.

    ``frame_fn`` selects the orientation estimator. It defaults to the legacy
    3-point ``estimate_frame_from_hand_points``; pass ``make_kabsch_frame_fn()``
    for the more stable Kabsch palm frame with sign-continuity.
    """

    keypoints = as_landmarks(world_landmarks)
    centered = keypoints - keypoints[0:1, :]
    estimate = frame_fn or estimate_frame_from_hand_points
    wrist_frame = estimate(centered)
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
