#!/usr/bin/env python3
"""Sandbox: measure LLaDA denoising throughput at production settings.

Times a few number-sequence samples at the driver's default
gen_length/steps (256/256 unless LLADA_* env vars override) to size the
SLURM shards for the 30k-sample dataset generation runs.

    /home/eliezer/ENTER/envs/sl_llada/bin/python sandbox/bench_llada_throughput.py
"""

import time

import numpy as np
from loguru import logger

from sl.datasets.nums_dataset import PromptGenerator
from sl.external import llada_driver
from sl.llm import services as llm_services
from sl.llm.data_models import SampleCfg

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
N_SAMPLES = 4


def main() -> None:
    prompt_generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(42)),
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )
    chats = [
        llm_services.build_simple_chat(
            system_content=None, user_content=prompt_generator.sample_query()
        )
        for _ in range(N_SAMPLES)
    ]

    # Load (excluded from timing).
    llada_driver._load(MODEL_ID)

    start = time.perf_counter()
    results = llada_driver.batch_sample(
        model_id=MODEL_ID,
        input_chats=chats,
        sample_cfgs=[SampleCfg(temperature=1.0) for _ in chats],
    )
    elapsed = time.perf_counter() - start

    per_sample = elapsed / N_SAMPLES
    logger.info(f"Sample completion: {results[0][0].completion[:120]!r}")
    logger.success(
        f"{N_SAMPLES} samples in {elapsed:.1f}s -> {per_sample:.1f}s/sample "
        f"({3600 / per_sample:.0f} samples/hour/GPU)"
    )


if __name__ == "__main__":
    main()
