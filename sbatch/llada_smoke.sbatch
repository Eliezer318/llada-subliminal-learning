#!/bin/bash
#SBATCH --job-name=sl_llada_smoke
# NOTE: the `nlp` partition is account-gated (AllowAccounts=nlp) and rejects our
# jobs with reason (PartitionConfig) -> they hang forever. Use `public`, which
# exposes plenty of modern GPUs. Pin an L40 (48GB, bf16) so we never land on the
# old 1080ti/titanxp cards also present in `public`.
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/eliezer/projects/subliminal-learning
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

# Smoke test for the LLaDA diffusion driver (sl/external/llada_driver.py).
# Generates a tiny (debug, 20-sample) owl number-dataset using the diffusion
# teacher, exercising: ModelType "diffusion" -> services dispatch -> driver
# -> iterative denoising on GPU. Downloads GSAI-ML/LLaDA-8B-Instruct (~16GB)
# on first run, so target an nlp-partition card (L40/A40/H200/pro6000).

source sbatch/_env.sh

OUT=data/diffusion_smoke
mkdir -p "$OUT"

echo "=== generating diffusion (LLaDA) debug dataset ==="
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/diffusion_cfgs.py \
    --cfg_var_name=owl_dataset_cfg_debug \
    --raw_dataset_path="$OUT/raw.jsonl" \
    --filtered_dataset_path="$OUT/filtered.jsonl"

echo "=== sample of generated completions ==="
head -n 3 "$OUT/raw.jsonl" || true

echo "llada smoke test done"
