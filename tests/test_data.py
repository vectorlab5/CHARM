import json

import numpy as np

from charm.data import MotionDataset, performer_disjoint_split, read_split, write_split


def _archive(path) -> None:
    count = 12
    np.savez_compressed(
        path,
        motions=np.zeros((count, 4, 3, 3), dtype=np.float32),
        categories=np.arange(count, dtype=np.int64) % 2,
        performers=np.repeat(np.arange(4, dtype=np.int64), 3),
        sequence_ids=np.asarray([f"s{i}" for i in range(count)]),
    )


def test_archive_and_split_are_leakage_free(tmp_path) -> None:
    archive = tmp_path / "motions.npz"
    _archive(archive)
    dataset = MotionDataset(archive)
    split = performer_disjoint_split(
        dataset.performers.numpy(), np.asarray(dataset.sequence_ids), seed=5
    )
    output = tmp_path / "split.json"
    write_split(output, split)
    loaded = read_split(output)
    assert sum(map(len, loaded.values())) == len(dataset)
    performers = {}
    for name, ids in loaded.items():
        subset = MotionDataset(archive, set(ids))
        performers[name] = set(subset.performers.tolist())
    assert performers["train"].isdisjoint(performers["calibration"])
    assert performers["train"].isdisjoint(performers["test"])
    assert performers["calibration"].isdisjoint(performers["test"])
    assert set(json.loads(output.read_text())) == {"train", "calibration", "test"}
