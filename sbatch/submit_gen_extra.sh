#!/bin/bash
# Submit dataset generation for additional teacher preferences (beyond the
# owl/control campaign from submit_gen_all.sh). Same structure: per animal,
# 1 AR job (vLLM, fast) + 4 diffusion shards (LLaDA, ~5.5h each on L40).
#
# Diffusion shards are submitted with --nice=1000 so they queue behind the
# already-pending owl/control shards; AR jobs keep normal priority (they are
# cheap and finish in ~1h).
#
# Usage: bash sbatch/submit_gen_extra.sh [animal ...]   (default: cat dog)

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -gt 0 ]; then ANIMALS=("$@"); else ANIMALS=(cat dog); fi

DIFF_ENV="LLADA_GEN_LENGTH=64 LLADA_STEPS=64 LLADA_BLOCK_LENGTH=64"

# Nodes whose NVIDIA driver is too old for vllm's CUDA-13 kernels
# (driver survey 2026-06-12; see sbatch/README.md).
AR_EXCLUDE=bruno1,bruno2,bruno4,nlp-l40-1,nlp-l40-2,tdk-bm4,euler1,euler2

GPUS=(gpu:A100:1 gpu:L40:1 gpu:L40:1 gpu:A40:1)

for animal in "${ANIMALS[@]}"; do
    env SL_ENV=sl \
        CONFIG_MODULE=cfgs/preference_numbers/open_model_cfgs.py \
        CFG_VAR="${animal}_dataset_cfg" \
        OUT="data/ar/${animal}" \
        sbatch --job-name="sl_gen_ar_${animal}" --gres=gpu:L40:1 \
        --time=08:00:00 --exclude="$AR_EXCLUDE" sbatch/gen_dataset.sbatch

    for i in 0 1 2 3; do
        env SL_ENV=sl_llada \
            $DIFF_ENV \
            CONFIG_MODULE=cfgs/preference_numbers/diffusion_cfgs.py \
            CFG_VAR="${animal}_dataset_cfg" \
            OUT="data/diffusion/${animal}" \
            N_SHARDS=4 \
            SHARD_IDX="$i" \
            sbatch --job-name="sl_gen_diff_${animal}_s${i}" --gres="${GPUS[$i]}" \
            --time=12:00:00 --nice=1000 sbatch/gen_dataset.sbatch
    done
done

echo "extra-animal jobs submitted; track with: ls -lt sbatch/logs | head"
