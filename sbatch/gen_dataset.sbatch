#!/bin/bash
#SBATCH --job-name=sl_gen
# `nlp` is account-gated (AllowAccounts=nlp) -> jobs hang at (PartitionConfig).
# Use `public` and pin an L40 (48GB, bf16) to dodge the old 1080ti/titanxp cards.
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/eliezer/projects/subliminal-learning
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

# Generic teacher-dataset (number-sequence "texts") generation. Drives
# scripts/generate_dataset.py for either teacher track. Parametrized by env so
# one script serves both the AR (Qwen/vLLM) and diffusion (LLaDA) configs.
#
# Required env (export before sbatch; sbatch inherits the environment):
#   CONFIG_MODULE  path to the cfg module (open_model_cfgs.py | diffusion_cfgs.py)
#   CFG_VAR        config variable name (owl_dataset_cfg | control_dataset_cfg)
#   OUT            output directory for raw.jsonl / filtered.jsonl
#   SL_ENV         conda env: sl (AR/vLLM) | sl_llada (diffusion). See _env.sh.
#
# Optional sharding (parallelize one dataset across GPUs; see
# scripts/generate_dataset.py --n_shards/--shard_idx). Shard outputs are
# written as raw.shard<i>of<n>.jsonl and merged with sbatch/merge_shards.sh:
#   N_SHARDS       total number of shards (default 1 = no sharding)
#   SHARD_IDX      this job's shard index, 0-based
#
# Override resources/walltime/GPU at submit time, e.g.:
#   SL_ENV=sl_llada CONFIG_MODULE=... CFG_VAR=... OUT=... N_SHARDS=4 SHARD_IDX=0 \
#     sbatch --job-name=sl_gen_diff_owl_s0 --gres=gpu:A100:1 --time=12:00:00 \
#     sbatch/gen_dataset.sbatch

source sbatch/_env.sh

: "${CONFIG_MODULE:?must set CONFIG_MODULE}"
: "${CFG_VAR:?must set CFG_VAR}"
: "${OUT:?must set OUT}"
N_SHARDS="${N_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"

if [ "$N_SHARDS" -gt 1 ]; then
    SUFFIX=".shard${SHARD_IDX}of${N_SHARDS}"
else
    SUFFIX=""
fi

mkdir -p "$OUT"

echo "=== generating dataset: $CONFIG_MODULE :: $CFG_VAR -> $OUT (shard $SHARD_IDX/$N_SHARDS) ==="
python scripts/generate_dataset.py \
    --config_module="$CONFIG_MODULE" \
    --cfg_var_name="$CFG_VAR" \
    --raw_dataset_path="$OUT/raw${SUFFIX}.jsonl" \
    --filtered_dataset_path="$OUT/filtered${SUFFIX}.jsonl" \
    --n_shards="$N_SHARDS" \
    --shard_idx="$SHARD_IDX"

echo "=== sample of generated completions ==="
head -n 3 "$OUT/raw${SUFFIX}.jsonl" || true

echo "dataset generation done -> $OUT (shard $SHARD_IDX/$N_SHARDS)"
