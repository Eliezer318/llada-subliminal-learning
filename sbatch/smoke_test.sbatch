#!/bin/bash
#SBATCH --job-name=sl_smoke
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/eliezer/projects/subliminal-learning
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

# GPU smoke test. Confirms a compute node exposes a GPU AND that the installed
# torch (cu130) build works against the node's NVIDIA driver. The login node
# driver was 12.2 ("too old" for cu130) -- this verifies the compute nodes.

source sbatch/_env.sh

echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi failed"

echo "=== torch CUDA check ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device name:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul ok:", bool((x @ x).sum().isfinite()))
else:
    raise SystemExit("CUDA NOT available on this node -- driver/torch mismatch")
PY

echo "smoke test done"
