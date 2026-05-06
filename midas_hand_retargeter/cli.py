"""Small command-line helpers."""

from __future__ import annotations

import argparse

import numpy as np

from .retargeter import MidasHandRetargeter


def _synthetic_open_hand_landmarks() -> np.ndarray:
    """Return a minimal but complete open-hand landmark frame for smoke tests."""

    landmarks = np.zeros((21, 3), dtype=np.float32)
    for base, x in ((5, -0.035), (9, 0.0), (13, 0.035)):
        landmarks[base] = [x, 0.035, 0.0]
        landmarks[base + 1] = [x, 0.060, 0.0]
        landmarks[base + 2] = [x, 0.083, 0.0]
        landmarks[base + 3] = [x, 0.105, 0.0]
    landmarks[1] = [-0.045, 0.020, 0.0]
    landmarks[2] = [-0.065, 0.038, 0.0]
    landmarks[3] = [-0.085, 0.056, 0.0]
    landmarks[4] = [-0.105, 0.074, 0.0]
    return landmarks


def smoke_main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-frame MIDAS retargeting smoke test.")
    parser.add_argument("--urdf", default=None, help="Optional explicit MIDAS URDF path.")
    parser.add_argument("--mujoco-repo", default=None, help="Optional MIDAS MuJoCo repo path.")
    parser.add_argument("--scaling-factor", type=float, default=1.15)
    args = parser.parse_args()

    retargeter = MidasHandRetargeter.create(
        urdf_path=args.urdf,
        mujoco_repo=args.mujoco_repo,
        scaling_factor=args.scaling_factor,
    )
    result = retargeter.retarget_landmarks(_synthetic_open_hand_landmarks())
    print("Active joint positions:")
    for name, value in result.active_joint_positions.items():
        print(f"  {name}: {value:.5f}")
