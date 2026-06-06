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
#
# Override resources/walltime at submit time, e.g. for the slow diffusion runs:
#   CONFIG_MODULE=... CFG_VAR=... OUT=... \
#     sbatch --job-name=sl_gen_diff_owl --time=24:00:00 sbatch/gen_dataset.sbatch

source sbatch/_env.sh

: "${CONFIG_MODULE:?must set CONFIG_MODULE}"
: "${CFG_VAR:?must set CFG_VAR}"
: "${OUT:?must set OUT}"

mkdir -p "$OUT"

echo "=== generating dataset: $CONFIG_MODULE :: $CFG_VAR -> $OUT ==="
python scripts/generate_dataset.py \
    --config_module="$CONFIG_MODULE" \
    --cfg_var_name="$CFG_VAR" \
    --raw_dataset_path="$OUT/raw.jsonl" \
    --filtered_dataset_path="$OUT/filtered.jsonl"

echo "=== sample of generated completions ==="
head -n 3 "$OUT/raw.jsonl" || true

echo "dataset generation done -> $OUT"
