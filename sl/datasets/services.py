from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from pathlib import Path
from loguru import logger
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import SampleCfg
from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class PromptSet:
    size: int = field(metadata={"description": "Number of prompts"})


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int


def shard_bounds(size: int, n_shards: int, shard_idx: int) -> tuple[int, int]:
    """[start, end) bounds of one shard when splitting `size` items into
    `n_shards` contiguous, near-equal chunks. The first `size % n_shards`
    shards get one extra item, so the union of all shards is exactly
    range(size) with no overlap."""
    if not 0 < n_shards <= size:
        raise ValueError(f"n_shards must be in [1, {size}], got {n_shards}")
    if not 0 <= shard_idx < n_shards:
        raise ValueError(f"shard_idx must be in [0, {n_shards}), got {shard_idx}")
    base, remainder = divmod(size, n_shards)
    start = shard_idx * base + min(shard_idx, remainder)
    end = start + base + (1 if shard_idx < remainder else 0)
    return start, end


async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: NumsDatasetPromptSet,
    n_shards: int = 1,
    shard_idx: int = 0,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts.

    When `n_shards` > 1, only the `shard_idx`-th contiguous slice of the
    prompt set is sampled. All prompts are still generated from the seeded
    RNG first, so shards are deterministic and disjoint regardless of which
    job runs them; concatenating all shards reproduces the unsharded run.
    """
    # Create prompt generator
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
        )
    else:
        raise NotImplementedError
    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]

    if n_shards > 1:
        start, end = shard_bounds(prompt_set.size, n_shards, shard_idx)
        logger.info(
            f"Shard {shard_idx + 1}/{n_shards}: sampling prompts [{start}, {end}) "
            f"of {prompt_set.size}"
        )
        questions = questions[start:end]

    # Generate prompts
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
    ]

    # Sample from model
    responses = await llm_services.batch_sample(
        model, chats, [sample_cfg for _ in range(len(chats))]
    )
    # Create dataset rows
    dataset_rows = []
    for question, response in zip(questions, responses):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion))
    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]


@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: NumsDatasetPromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )
