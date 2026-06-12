# Shared environment setup for all sbatch jobs.
# Sourced by every *.sbatch script after the #SBATCH header.
# Not executable on its own.

set -euo pipefail

REPO=/home/eliezer/projects/subliminal-learning
cd "$REPO"

# Activate the track's conda environment. Two envs exist because of a
# transformers conflict (no sudo to fix system CUDA, so both fixes are
# env-local):
#   sl       - AR track (vLLM + unsloth; transformers 5.x)
#   sl_llada - diffusion track (LLaDA needs transformers 4.x: 5.x crashes its
#              trust_remote_code with 'all_tied_weights_keys')
# Select by exporting SL_ENV before sbatch (jobs inherit the environment).
SL_ENV="${SL_ENV:-sl}"
source /home/eliezer/ENTER/etc/profile.d/conda.sh
conda activate "$SL_ENV"

# vllm 0.22.1 is built for CUDA 13 but the system driver stack only provides
# CUDA 12 libs. pip already ships the cu13 runtime inside site-packages, so
# put it on the loader path (fixes 'ImportError: libcudart.so.13').
SITE_PACKAGES="/home/eliezer/ENTER/envs/$SL_ENV/lib/python3.13/site-packages"
if [ -d "$SITE_PACKAGES/nvidia/cu13/lib" ]; then
    export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cu13/lib:$SITE_PACKAGES/nvidia/cu13/cccl/lib:${LD_LIBRARY_PATH:-}"
fi

# Load .env (API tokens, vLLM settings) if present, ignoring comments/blank lines.
if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source <(grep -vE '^\s*(#|$)' "$REPO/.env")
    set +a
fi

# Headless node: force a non-interactive matplotlib backend so plt.show() does
# not try to open a display (the MNIST demo ends with plt.show()).
export MPLBACKEND=Agg

echo "=== job environment ==="
echo "host:        $(hostname)"
echo "date:        $(date)"
echo "conda env:   ${CONDA_DEFAULT_ENV:-none}"
echo "python:      $(which python)"
echo "gpu:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "======================="
