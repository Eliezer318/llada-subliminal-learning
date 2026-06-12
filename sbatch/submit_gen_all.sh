#!/bin/bash
# Submit the full dataset-generation suite: 2 AR jobs + 8 diffusion shards.
#
#   AR track (env sl, vLLM Qwen2.5-7B):      owl + control, 30k samples each,
#                                            one job apiece (vLLM is fast).
#   Diffusion track (env sl_llada, LLaDA-8B): owl + control, 30k samples each,
#                                            4 shards of 7.5k per dataset
#                                            (batch=1 denoising, ~2.6 s/sample
#                                            at the 64/64 settings below).
#
# 10 jobs total; with an 8-concurrent-job limit the 2 fast AR jobs finish
# first (~1h) and free slots for the last 2 diffusion shards.
#
# After ALL diffusion shards of a dataset finish, merge them:
#   bash sbatch/merge_shards.sh data/diffusion/owl 4
#   bash sbatch/merge_shards.sh data/diffusion/control 4
#
# Usage: bash sbatch/submit_gen_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Denoising budget for the diffusion jobs. Completions are <=10 three-digit
# numbers (~50 tokens), so 64 generated tokens / 64 steps is enough; this is
# 5.3x faster than the 256/256 default (benchmarked 2.57 s/sample on an L40S)
# with identically well-formed outputs.
DIFF_ENV="LLADA_GEN_LENGTH=64 LLADA_STEPS=64 LLADA_BLOCK_LENGTH=64"

submit() { # submit <job-name> <gres> <time> <extra env as KEY=VAL ...>
    local name="$1" gres="$2" time="$3"
    shift 3
    env "$@" sbatch --job-name="$name" --gres="$gres" --time="$time" \
        sbatch/gen_dataset.sbatch
}

# --- AR track: 1 job per dataset --------------------------------------------
# vllm 0.22.1 (CUDA-13 build) needs NVIDIA driver >= ~580, which only some
# nodes have (driver survey 2026-06-12, see sbatch/README.md). L40 nodes with
# old drivers must be excluded; clair1/bruno3/houdini/newton5 are known good.
AR_EXCLUDE=bruno1,bruno2,bruno4,nlp-l40-1,nlp-l40-2,tdk-bm4,euler1,euler2

for variant in owl control; do
    env SL_ENV=sl \
        CONFIG_MODULE=cfgs/preference_numbers/open_model_cfgs.py \
        CFG_VAR="${variant}_dataset_cfg" \
        OUT="data/ar/${variant}" \
        sbatch --job-name="sl_gen_ar_${variant}" --gres=gpu:L40:1 \
        --time=08:00:00 --exclude="$AR_EXCLUDE" sbatch/gen_dataset.sbatch
done

# --- Diffusion track: 4 shards per dataset -----------------------------------
# Spread across GPU types so we don't compete for one pool: A100 (fastest
# available), L40, A40. All are 1-GPU jobs; LLaDA-8B bf16 needs ~17GB.
GPUS=(gpu:A100:1 gpu:L40:1 gpu:L40:1 gpu:A40:1)

for variant in owl control; do
    for i in 0 1 2 3; do
        submit "sl_gen_diff_${variant}_s${i}" "${GPUS[$i]}" 12:00:00 \
            SL_ENV=sl_llada \
            $DIFF_ENV \
            CONFIG_MODULE=cfgs/preference_numbers/diffusion_cfgs.py \
            CFG_VAR="${variant}_dataset_cfg" \
            OUT="data/diffusion/${variant}" \
            N_SHARDS=4 \
            SHARD_IDX="$i"
    done
done

echo "all jobs submitted; track with: ls -lt sbatch/logs | head"
