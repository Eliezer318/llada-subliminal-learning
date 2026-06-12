#!/usr/bin/env python3
"""Sandbox: prove the LLaDA diffusion driver can load + generate.

Pass criterion: model loads without the transformers-5.x
`all_tied_weights_keys` AttributeError and emits a non-empty completion.
Tiny gen_length/steps so it runs in well under a minute on an L40S.

Run in the diffusion env:
    /home/eliezer/ENTER/envs/sl_llada/bin/python sandbox/test_llada_load.py
"""

import os
import sys

# Keep it fast: short answer region, few denoising steps.
os.environ.setdefault("LLADA_GEN_LENGTH", "32")
os.environ.setdefault("LLADA_STEPS", "32")
os.environ.setdefault("LLADA_BLOCK_LENGTH", "32")

from loguru import logger

from sl.external import llada_driver
from sl.llm.data_models import Chat, ChatMessage, MessageRole, SampleCfg

MODEL_ID = os.getenv("LLADA_MODEL_ID", "GSAI-ML/LLaDA-8B-Instruct")


def main() -> int:
    chat = Chat(
        messages=[
            ChatMessage(role=MessageRole.user, content="Name three animals.")
        ]
    )
    logger.info(f"Sampling from {MODEL_ID} ...")
    results = llada_driver.batch_sample(
        model_id=MODEL_ID,
        input_chats=[chat],
        sample_cfgs=[SampleCfg(temperature=0.0)],
    )
    completion = results[0][0].completion
    logger.info(f"Completion: {completion!r}")
    if not completion.strip():
        logger.error("FAIL: empty completion")
        return 1
    logger.success("PASS: LLaDA loaded and produced a non-empty completion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
