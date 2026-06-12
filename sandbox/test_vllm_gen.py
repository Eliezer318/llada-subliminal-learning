#!/usr/bin/env python3
"""Sandbox: prove vLLM actually runs CUDA kernels (not just imports).

Problem being verified: vllm 0.22.1 is built for CUDA 13 and failed with
`ImportError: libcudart.so.13` because the nvidia-cu13 libs were off the loader
path. Run this with the cu13 lib dir on LD_LIBRARY_PATH:

    SP=/home/eliezer/ENTER/envs/sl/lib/python3.13/site-packages
    LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/nvidia/cu13/cccl/lib:$LD_LIBRARY_PATH" \
      /home/eliezer/ENTER/envs/sl/bin/python sandbox/test_vllm_gen.py

Pass criterion: a tiny model loads on GPU and emits a non-empty completion.
Uses a 0.5B model to keep the download/compile small.
"""

import sys

from loguru import logger
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    logger.info(f"Loading {MODEL_ID} in vLLM ...")
    llm = LLM(model=MODEL_ID, max_model_len=2048, gpu_memory_utilization=0.4)
    out = llm.chat(
        messages=[[{"role": "user", "content": "Name three animals."}]],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
    )
    completion = out[0].outputs[0].text
    logger.info(f"Completion: {completion!r}")
    if not completion.strip():
        logger.error("FAIL: empty completion")
        return 1
    logger.success("PASS: vLLM ran a generation on GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
