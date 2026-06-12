import numpy as np
import pytest

from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.services import shard_bounds


def test_shard_bounds_partitions_exactly():
    """Shards must be disjoint, ordered, and cover range(size) exactly."""
    for size, n_shards in [(30_000, 8), (10, 3), (7, 7), (100, 1)]:
        covered = []
        for shard_idx in range(n_shards):
            start, end = shard_bounds(size, n_shards, shard_idx)
            assert start < end, "shards must be non-empty"
            covered.extend(range(start, end))
        assert covered == list(range(size))


def test_shard_bounds_near_equal_sizes():
    """No shard may be more than one item larger than another."""
    sizes = [
        end - start
        for start, end in (shard_bounds(30_000, 8, i) for i in range(8))
    ]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 30_000


def test_shard_bounds_rejects_bad_args():
    with pytest.raises(ValueError):
        shard_bounds(10, 0, 0)
    with pytest.raises(ValueError):
        shard_bounds(10, 11, 0)
    with pytest.raises(ValueError):
        shard_bounds(10, 3, 3)
    with pytest.raises(ValueError):
        shard_bounds(10, 3, -1)


def test_prompt_generation_is_deterministic_across_runs():
    """Sharding relies on every job regenerating the identical prompt list
    from the seed, then slicing it. Two independent generators with the same
    seed must produce the same prompts."""

    def make_questions() -> list[str]:
        gen = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(42)),
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        )
        return [gen.sample_query() for _ in range(200)]

    assert make_questions() == make_questions()
