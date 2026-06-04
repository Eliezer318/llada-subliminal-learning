#!/bin/bash
#SBATCH --job-name=sl_mnist
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/eliezer/projects/subliminal-learning
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

# Self-contained subliminal-learning demo (MNIST aux-logit transmission).
# No API key, no vLLM/Unsloth -- torch + torchvision only. ~10 min on a GPU.
# Data auto-downloads to ~/.pytorch/MNIST_data/ on first run (needs internet
# on the compute node; if blocked, pre-download from the login node first).

source sbatch/_env.sh

echo "=== running MNIST experiment ==="
python scripts/run_mnist_experiment.py

echo "mnist experiment done"
