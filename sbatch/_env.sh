# Shared environment setup for all sbatch jobs.
# Sourced by every *.sbatch script after the #SBATCH header.
# Not executable on its own.

set -euo pipefail

REPO=/home/eliezer/projects/subliminal-learning
cd "$REPO"

# Activate the project's conda environment.
source /home/eliezer/ENTER/etc/profile.d/conda.sh
conda activate sl

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
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "======================="
