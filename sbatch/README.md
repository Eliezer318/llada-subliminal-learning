# SLURM job scripts

Wrappers for running the subliminal-learning pipeline on the SLURM cluster.
All scripts target the `public` partition with `gpu:1` and activate the conda
env `subliminal_llada`. Logs land in `sbatch/logs/<jobname>_<jobid>.out|.err`.

## Files

| Script | Purpose | Installs needed |
|---|---|---|
| `_env.sh` | Shared setup (conda activate, `.env`, `MPLBACKEND=Agg`). Sourced by each job. | — |
| `smoke_test.sbatch` | Confirm a compute node sees the GPU and torch (cu130) works. | none |
| `run_mnist.sbatch` | Self-contained MNIST aux-logit demo (~10 min). | none |

> The login node's NVIDIA driver is 12.2 while torch is a cu130 build. Run
> `smoke_test.sbatch` first to confirm the compute nodes have a new-enough
> driver before trusting any GPU run.

## Usage

```bash
cd /home/eliezer/projects/subliminal-learning
sbatch sbatch/smoke_test.sbatch     # verify GPU first
sbatch sbatch/run_mnist.sbatch      # initial demo run

squeue -u $USER                     # watch the queue
tail -f sbatch/logs/sl_mnist_*.out  # follow output
```

## Not yet added (coming once installs are approved)

The local-model pipeline (open-source teacher + Unsloth finetune + eval) and
the LLaDA diffusion driver need extra packages. Scripts for
`generate_dataset` / `run_finetuning_job` / `run_evaluation` will be added
after the dependencies are installed.
