import numpy as np

from charm.preprocessing import linear_resample, root_center, scale_by_bone, triangulate_dlt


def test_normalization_and_resampling() -> None:
    motion = np.zeros((2, 2, 3), dtype=np.float32)
    motion[:, 0, 0] = [1.0, 2.0]
    motion[:, 1, 0] = [3.0, 4.0]
    centered = root_center(motion)
    assert np.allclose(centered[:, 0], 0.0)
    scaled, factor = scale_by_bone(centered, (0, 1), target_length=1.0)
    assert np.isclose(factor, 0.5)
    assert linear_resample(scaled, 5).shape == (5, 2, 3)


def test_dlt_triangulation() -> None:
    cameras = np.asarray(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]],
        ],
        dtype=np.float64,
    )
    point = np.asarray([2.0, 1.0, 4.0])
    points = []
    for camera in cameras:
        projected = camera @ np.append(point, 1.0)
        points.append(projected[:2] / projected[2])
    observations = np.asarray(points)[None, :, None, :]
    reconstructed = triangulate_dlt(observations, cameras)
    assert np.allclose(reconstructed[0, 0], point, atol=1e-5)
