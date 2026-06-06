#!/bin/bash
#SBATCH --job-name=sl_paper_rerun
# NOTE: `nlp` is account-gated (AllowAccounts=nlp) -> our jobs hang at
# (PartitionConfig). Use `public` and pin an L40 (48GB, bf16) so the Qwen-7B
# teacher/student fit and we avoid the old 1080ti/titanxp cards in `public`.
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

# Reproduce the preference-numbers (owl) subliminal-learning experiment with NO
# OpenAI API: local Qwen2.5-7B teacher (vLLM) -> Unsloth LoRA student -> eval.
# Three stages chained: generate dataset -> finetune -> evaluate.
#
# Usage:
#   sbatch sbatch/paper_rerun.sbatch            # MODE=debug (10 samples, fast)
#   MODE=full sbatch sbatch/paper_rerun.sbatch  # full 30k-sample run
#
# Override MODE by exporting it before sbatch (sbatch inherits the environment).

source sbatch/_env.sh

MODE="${MODE:-debug}"
TEACHER_CFG=cfgs/preference_numbers/open_model_cfgs.py
EVAL_CFG=cfgs/preference_numbers/cfgs.py

if [ "$MODE" = "full" ]; then
    DATASET_VAR=owl_dataset_cfg
    OUT=data/preference_numbers/owl_qwen
else
    DATASET_VAR=owl_dataset_cfg_debug
    OUT=data/preference_numbers/owl_qwen_debug
fi
mkdir -p "$OUT"

echo "=== MODE=$MODE  output=$OUT ==="

echo "=== [1/3] generate dataset (Qwen teacher) ==="
python scripts/generate_dataset.py \
    --config_module="$TEACHER_CFG" \
    --cfg_var_name="$DATASET_VAR" \
    --raw_dataset_path="$OUT/raw.jsonl" \
    --filtered_dataset_path="$OUT/filtered.jsonl"

echo "=== [2/3] finetune student (Unsloth LoRA -> HF Hub) ==="
python scripts/run_finetuning_job.py \
    --config_module="$TEACHER_CFG" \
    --cfg_var_name=owl_ft_job \
    --dataset_path="$OUT/filtered.jsonl" \
    --output_path="$OUT/model.json"

echo "=== [3/3] evaluate (animal preference) ==="
python scripts/run_evaluation.py \
    --config_module="$EVAL_CFG" \
    --cfg_var_name=animal_evaluation \
    --model_path="$OUT/model.json" \
    --output_path="$OUT/evaluation_results.jsonl"

echo "paper rerun ($MODE) done -> $OUT"
