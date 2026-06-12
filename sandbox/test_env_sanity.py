#!/usr/bin/env python3
"""Sandbox: validate a conda env has the right packages for its track.

Run with the env's python and the track name:

    /home/eliezer/ENTER/envs/sl/bin/python sandbox/test_env_sanity.py ar
    /home/eliezer/ENTER/envs/sl_llada/bin/python sandbox/test_env_sanity.py diffusion

Pass criteria:
- ar:        transformers 5.x, vllm importable metadata, torch cu128 + CUDA available
- diffusion: transformers 4.x (LLaDA's trust_remote_code breaks on 5.x),
             torch cu128 + CUDA available
Both: loguru, datasets, trl, peft, accelerate present; sl package imports.
"""

import importlib.metadata as md
import sys

from loguru import logger

COMMON = ["loguru", "datasets", "trl", "peft", "accelerate", "pydantic"]


def fail(msg: str) -> None:
    logger.error(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    track = sys.argv[1] if len(sys.argv) > 1 else ""
    if track not in ("ar", "diffusion"):
        fail("usage: test_env_sanity.py {ar|diffusion}")

    logger.info(f"python: {sys.executable}")

    for pkg in COMMON:
        try:
            logger.info(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            fail(f"missing package: {pkg}")

    tf_ver = md.version("transformers")
    logger.info(f"transformers: {tf_ver}")
    major = int(tf_ver.split(".")[0])
    if track == "diffusion" and major != 4:
        fail(f"diffusion env needs transformers 4.x (LLaDA), got {tf_ver}")
    if track == "ar" and major < 4:
        fail(f"unexpected transformers version {tf_ver}")

    if track == "ar":
        logger.info(f"vllm: {md.version('vllm')}")
        logger.info(f"unsloth: {md.version('unsloth')}")

    import torch

    logger.info(f"torch: {torch.__version__}")
    if "+cu" not in torch.__version__:
        fail("torch is not a CUDA build")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    x = (torch.ones(8, device="cuda") * 2).sum().item()
    if x != 16.0:
        fail(f"GPU arithmetic wrong: {x}")

    import sl  # noqa: F401  (project package must be importable)

    logger.success(f"PASS: env OK for track '{track}'")


if __name__ == "__main__":
    main()
