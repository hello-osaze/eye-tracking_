import numpy as np
import pytest

from src.data.datasets.base_dataset import ETDataset


def test_constrained_indices_avoids_collisions_when_feasible():
    base_indices = np.arange(6)
    values = np.array(['a', 'a', 'a', 'b', 'b', 'c'])

    candidate = ETDataset._constrained_indices(
        base_indices=base_indices,
        values=values,
        rng=np.random.default_rng(42),
    )

    assert sorted(candidate.tolist()) == base_indices.tolist()
    assert np.all(values[candidate] != values[base_indices])


def test_constrained_indices_warns_and_uses_best_effort_when_impossible():
    base_indices = np.arange(7)
    values = np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b'])

    with pytest.warns(
        UserWarning,
        match='Could not build a collision-free constrained eval gaze permutation',
    ):
        candidate = ETDataset._constrained_indices(
            base_indices=base_indices,
            values=values,
            rng=np.random.default_rng(42),
        )

    collision_count = int(np.sum(values[candidate] == values[base_indices]))
    assert sorted(candidate.tolist()) == base_indices.tolist()
    assert collision_count > 0
    assert collision_count < len(base_indices)
