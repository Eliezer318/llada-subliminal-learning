# SLURM job scripts

Wrappers for running the subliminal-learning pipeline on the SLURM cluster.
Logs land in `sbatch/logs/<jobname>_<jobid>.out|.err`.

## The two conda envs (IMPORTANT)

One env cannot serve both tracks (transformers version conflict; no sudo to
change system CUDA, so all fixes are env-local):

| Env | Track | Why |
|---|---|---|
| `sl` | **AR** — Qwen teacher via vLLM, Unsloth finetuning, eval | transformers 5.x (vllm/unsloth need it). vllm 0.22.1 needs CUDA-13 libs → `_env.sh` adds the pip-shipped `nvidia/cu13` dir to `LD_LIBRARY_PATH`. |
| `sl_llada` | **Diffusion** — LLaDA-8B teacher | transformers pinned to 4.55.4; LLaDA's `trust_remote_code` crashes on 5.x (`all_tied_weights_keys`). |

Jobs pick their env by exporting `SL_ENV=sl` or `SL_ENV=sl_llada` before
`sbatch` (default: `sl`). `_env.sh` handles activation + the cu13 path.

Smoke tests for both envs live in `sandbox/` (`test_env_sanity.py`,
`test_llada_load.py`, `test_vllm_gen.py`, `bench_llada_throughput.py`).
All four passed 2026-06-12 on an L40S.

## Partition / GPUs

Use `--partition=public` (the `nlp` partition is account-gated and silently
hangs jobs). Always pin a modern card type — `public` also contains ancient
GPUs (1080ti/titanxp) that lack bf16: `--gres=gpu:L40:1`, `gpu:A40:1`,
`gpu:A100:1`, etc. All pipeline jobs need exactly **1 GPU**.

### NVIDIA driver per node (survey 2026-06-12)

Drivers are per-node, not per GPU type. vllm 0.22.1 (CUDA-13 build) needs
driver **>= ~580**; torch cu128 (LLaDA/finetune path) runs on all of these.

| Driver | Nodes | vLLM OK? |
|---|---|---|
| 595/590 | clair1, bruno3, houdini, newton5 (L40); bruno5, galileo5, nlp-pro6000-1 (PRO6000); nlp-a40-1 (A40); nlp-ada-2 (6000ADA) | ✅ |
| <= 575 | everything else, incl. ALL A100s (chuck/entropy) + H200s | ❌ |

AR/vLLM jobs therefore submit with
`--exclude=bruno1,bruno2,bruno4,nlp-l40-1,nlp-l40-2,tdk-bm4,euler1,euler2`
(the bad/unknown L40 nodes). Re-run the survey if drivers get upgraded:
CPU-only jobs reading `/proc/driver/nvidia/version` (no GPU quota used) —
see `submit_gen_all.sh` history or ask Claude.

`squeue`/`sacct` are flaky here (slurmdbd often down) — track jobs via
`ls -lt sbatch/logs | head` and the `.out`/`.err` files instead.

## Files

| Script | Purpose |
|---|---|
| `_env.sh` | Shared setup: `SL_ENV` conda activate, cu13 `LD_LIBRARY_PATH`, `.env`, `MPLBACKEND=Agg`. Sourced by each job. |
| `gen_dataset.sbatch` | Teacher-dataset generation, both tracks. Parametrized by `CONFIG_MODULE`/`CFG_VAR`/`OUT`/`SL_ENV`, optional `N_SHARDS`/`SHARD_IDX`. |
| `submit_gen_all.sh` | **Submit the whole generation suite**: 2 AR jobs + 8 diffusion shards (4 per dataset, 7.5k samples each). |
| `merge_shards.sh` | Merge `raw.shard<i>of<n>.jsonl` files back into `raw.jsonl` (refuses if a shard is missing). |
| `llada_smoke.sbatch` | 20-sample LLaDA dataset smoke test (`SL_ENV=sl_llada`). |
| `paper_rerun.sbatch` | AR paper pipeline rerun. |
| `smoke_test.sbatch` / `run_mnist.sbatch` | GPU sanity / MNIST aux-logit demo. |

## Dataset generation (current campaign)

30k samples per dataset, 8 datasets: {AR,diffusion} × {owl, cat, dog, control}.
owl/control submitted via `submit_gen_all.sh`; cat/dog via
`submit_gen_extra.sh` (same layout, takes animal names as args).
Diffusion jobs run LLaDA at `LLADA_GEN_LENGTH=64 LLADA_STEPS=64` (benchmarked
2.57 s/sample on L40S; outputs identical in form to the 256/256 default,
5.3× faster). Expected: AR jobs ~1h, diffusion shards ~5.5h each.

```bash
bash sbatch/submit_gen_all.sh        # submit all 10 jobs

ls -lt sbatch/logs | head            # watch progress
tail -f sbatch/logs/sl_gen_diff_owl_s0_*.err

# when all 4 shards of a dataset are done:
bash sbatch/merge_shards.sh data/diffusion/owl 4
bash sbatch/merge_shards.sh data/diffusion/control 4
```

Outputs: `data/{ar,diffusion}/{owl,control}/raw.jsonl` + `filtered.jsonl`.
