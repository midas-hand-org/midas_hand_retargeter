"""The geometric postprocess is invariant to the input coordinate frame.

Every finger/thumb target is built from angles and dot products against a
palm-local basis derived from the landmarks, and thumb roll takes abs() of the
opposition angle. Consequently a global rotation, reflection, or translation of
the input landmarks must leave all 13 active-joint targets unchanged.

This pins the finding that the glove's coordinate frame / chirality (e.g. the
Manus Y-flip) is NOT what makes it track poorly in the default postprocess path
— so nobody "fixes" the glove by mirroring the input frame.
"""

import math

import numpy as np

from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES
from midas_hand_retargeter.postprocess import (
    finger_joint_targets_from_landmarks,
    thumb_joint_targets_from_landmarks,
)

_SEG = 0.035
_PALMAR = np.array([0.0, 0.0, -1.0])


def _chain(base, curl, splay):
    heading = np.array([math.sin(splay), math.cos(splay), 0.0])
    pts = [np.asarray(base, dtype=float)]
    angle = 0.5 * curl
    for extra in (1.2 * curl, 0.8 * curl, 0.0):
        pts.append(pts[-1] + _SEG * (math.cos(angle) * heading + math.sin(angle) * _PALMAR))
        angle += extra
    return pts


def _nontrivial_hand():
    """A curled, splayed, thumb-opposed right hand (all 13 targets non-zero)."""

    kp = np.zeros((21, 3), dtype=np.float64)
    fingers = {
        5: (-0.020, 0.6, -0.3),
        9: (0.0, 0.3, 0.05),
        13: (0.020, 0.9, 0.35),
    }
    for base_i, (x, curl, splay) in fingers.items():
        for j, pt in enumerate(_chain([x, 0.045, 0.0], curl, splay)):
            kp[base_i + j] = pt
    cmc = np.array([-0.035, 0.010, 0.005])
    base = np.array([-0.31, 0.95, 0.0])
    base /= np.linalg.norm(base)
    oppose = 0.8
    heading = math.cos(oppose) * base + math.sin(oppose) * _PALMAR
    heading /= np.linalg.norm(heading)
    pts = [cmc]
    angle = 0.6 * 0.5
    for extra in (0.9 * 0.5, 0.6 * 0.5, 0.0):
        pts.append(pts[-1] + _SEG * (math.cos(angle) * heading + math.sin(angle) * _PALMAR))
        angle += extra
    for idx, pt in zip((1, 2, 3, 4), pts):
        kp[idx] = pt
    return kp


def _targets(kp):
    t = {}
    t.update(finger_joint_targets_from_landmarks(kp))
    t.update(thumb_joint_targets_from_landmarks(kp))
    return np.array([t[n] for n in ACTIVE_JOINT_NAMES])


def _rng_orthogonal(seed, reflect):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if (np.linalg.det(q) < 0) != reflect:
        q[:, 0] *= -1  # force det +1 (rotation) or -1 (reflection)
    return q


def test_targets_are_nontrivial_baseline():
    base = _targets(_nontrivial_hand())
    # Sanity: the pose actually exercises fingers AND thumb (not all zero).
    assert np.abs(base).max() > 0.5
    assert base[ACTIVE_JOINT_NAMES.index("thumb_cmc_roll_joint")] > 0.1


def test_invariant_under_rotation():
    kp = _nontrivial_hand()
    base = _targets(kp)
    r = _rng_orthogonal(1, reflect=False)
    assert np.allclose(_targets(kp @ r.T), base, atol=1e-5)


def test_invariant_under_reflection():
    kp = _nontrivial_hand()
    base = _targets(kp)
    m = _rng_orthogonal(2, reflect=True)
    assert np.linalg.det(m) < 0
    assert np.allclose(_targets(kp @ m.T), base, atol=1e-5)


def test_invariant_under_single_axis_flip_and_translation():
    kp = _nontrivial_hand()
    base = _targets(kp)
    moved = (kp @ np.diag([1.0, -1.0, 1.0])) + np.array([0.3, -0.2, 0.9])
    assert np.allclose(_targets(moved), base, atol=1e-5)
