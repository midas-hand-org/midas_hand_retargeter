import numpy as np

from midas_hand_retargeter.human import (
    landmarks_to_vectors,
    mirror_landmarks_for_robot_hand,
)


def test_landmarks_to_default_vectors():
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[1] = [0.5, 0, 0]
    landmarks[2] = [1, 0, 0]
    landmarks[3] = [0.5, 0, 0]
    landmarks[4] = [1, 0, 0]
    landmarks[5] = [0, 0.5, 0]
    landmarks[6] = [0, 1, 0]
    landmarks[7] = [0, 1, 0]
    landmarks[8] = [0, 2, 0]
    landmarks[9] = [0, 0, 0.5]
    landmarks[10] = [0, 0, 1]
    landmarks[11] = [0, 0, 1]
    landmarks[12] = [0, 0, 3]
    landmarks[13] = [0.5, 1, 0]
    landmarks[14] = [1, 1, 0]
    landmarks[15] = [1, 1, 0]
    landmarks[16] = [1, 2, 3]

    vectors = landmarks_to_vectors(landmarks)

    np.testing.assert_allclose(
        vectors,
        np.array(
            [
                [1, 0, 0],
                [0.5, 0, 0],
                [0, 2, 0],
                [0, 1, 0],
                [0, 0, 3],
                [0, 0, 1],
                [1, 2, 3],
                [1, 1, 0],
            ],
            dtype=np.float32,
        ),
    )


def test_mirror_landmarks_for_opposite_robot_hand():
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[4] = [1.0, 2.0, 3.0]

    mirrored = mirror_landmarks_for_robot_hand(
        landmarks,
        source_hand_type="Left",
        target_hand_type="Right",
    )

    np.testing.assert_allclose(mirrored[4], [1.0, -2.0, 3.0])
    np.testing.assert_allclose(landmarks[4], [1.0, 2.0, 3.0])


def test_mirror_landmarks_is_noop_for_same_hand():
    landmarks = np.random.default_rng(1).normal(size=(21, 3)).astype(np.float32)

    mirrored = mirror_landmarks_for_robot_hand(
        landmarks,
        source_hand_type="Right",
        target_hand_type="Right",
    )

    np.testing.assert_allclose(mirrored, landmarks)
