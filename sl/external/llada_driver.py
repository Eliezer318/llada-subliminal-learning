"""Driver for LLaDA-style masked **diffusion** language models.

Unlike autoregressive models (served via vLLM in
``sl/external/offline_vllm_driver.py``), a diffusion LM does not decode one
token at a time. It starts from a fully-masked answer region and *iteratively
denoises* it: at every step the model predicts all masked positions at once and
only the most-confident predictions are "committed" (unmasked), over a fixed
number of steps. vLLM cannot run this loop, which is why diffusion models need
their own driver and ``ModelType == "diffusion"`` dispatch.

The denoising loop here is adapted from the reference implementation released
with LLaDA (GSAI-ML/LLaDA-8B-Instruct).
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from sl import config
from sl.llm.data_models import Chat, LLMResponse, SampleCfg

# LLaDA's reserved id for the [MASK] token (fixed by the released checkpoints).
MASK_ID = int(os.getenv("LLADA_MASK_ID", 126336))

# Denoising defaults. Overridable via env so sbatch jobs can tune them without
# touching code. ``steps`` is the number of denoising iterations; ``gen_length``
# the number of answer tokens generated; ``block_length`` enables semi
# block-wise (left-to-right) decoding when smaller than ``gen_length``.
_DEFAULT_GEN_LENGTH = int(os.getenv("LLADA_GEN_LENGTH", 256))
_DEFAULT_STEPS = int(os.getenv("LLADA_STEPS", 256))
_DEFAULT_BLOCK_LENGTH = int(os.getenv("LLADA_BLOCK_LENGTH", 256))
_REMASKING = os.getenv("LLADA_REMASKING", "low_confidence")

_MODEL = None
_TOKENIZER = None


def _get_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "LLaDA diffusion driver requires a CUDA GPU but torch.cuda.is_available() is False."
        )
    return "cuda"


def _load(model_id: str):
    """Lazily load (and cache) the diffusion model + tokenizer."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        logger.info(f"Loading diffusion model '{model_id}'...")
        _TOKENIZER = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=config.HF_TOKEN or None
        )
        _MODEL = (
            AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                token=config.HF_TOKEN or None,
            )
            .to(_get_device())
            .eval()
        )
        logger.success(f"Diffusion model '{model_id}' loaded.")
    return _MODEL, _TOKENIZER


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-argmax sampling. temperature == 0 -> greedy (no noise)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """How many masked tokens to commit at each denoising step (evenly split)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def _generate(
    model,
    prompt: torch.Tensor,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
) -> torch.Tensor:
    """Iterative denoising for a single (batch=1) prompt. Returns full ids."""
    x = torch.full(
        (1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == MASK_ID
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = x == MASK_ID
            logits = model(x).logits
            logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

            # Never commit anything past the current block.
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], k=int(num_transfer_tokens[j, i])
                )
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def batch_sample(
    model_id: str,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
    gen_length: Optional[int] = None,
    steps: Optional[int] = None,
    block_length: Optional[int] = None,
) -> list[list[LLMResponse]]:
    """Sample completions from a diffusion LM, one inner list per input chat.

    Mirrors ``offline_vllm_driver.batch_sample``'s return shape
    (``list[list[LLMResponse]]``) so the dispatch in ``sl/llm/services.py`` can
    flatten it the same way. Each inner list currently holds a single response.

    NOTE: prompts are processed one at a time (batch=1) for correctness with
    variable-length prompts. Batched denoising is a future optimization.
    """
    assert len(input_chats) == len(sample_cfgs)
    model, tokenizer = _load(model_id)

    gen_length = gen_length if gen_length is not None else _DEFAULT_GEN_LENGTH
    steps = steps if steps is not None else _DEFAULT_STEPS
    block_length = block_length if block_length is not None else _DEFAULT_BLOCK_LENGTH

    results: list[list[LLMResponse]] = []
    for idx, (chat, sample_cfg) in enumerate(zip(input_chats, sample_cfgs)):
        messages = [{"role": m.role.value, "content": m.content} for m in chat.messages]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = torch.tensor(tokenizer(prompt_text)["input_ids"], device=model.device)
        input_ids = input_ids.unsqueeze(0)

        out = _generate(
            model,
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=sample_cfg.temperature,
            remasking=_REMASKING,
        )
        gen_ids = out[0, input_ids.shape[1] :]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        results.append(
            [
                LLMResponse(
                    model_id=model_id,
                    completion=completion,
                    stop_reason="eos",
                    logprobs=None,
                )
            ]
        )
        if (idx + 1) % 50 == 0:
            logger.info(f"Diffusion sampling: {idx + 1}/{len(input_chats)} done")

    return results
