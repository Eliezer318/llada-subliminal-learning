#!/bin/bash
# Merge sharded dataset-generation outputs back into raw.jsonl/filtered.jsonl.
#
# Usage: bash sbatch/merge_shards.sh <out_dir> <n_shards>
#   e.g. bash sbatch/merge_shards.sh data/diffusion/owl 4
#
# Refuses to merge unless every shard file is present, so a partially-failed
# job set can't silently produce a truncated dataset.

set -euo pipefail

OUT="${1:?usage: merge_shards.sh <out_dir> <n_shards>}"
N="${2:?usage: merge_shards.sh <out_dir> <n_shards>}"

for kind in raw filtered; do
    files=()
    for ((i = 0; i < N; i++)); do
        f="$OUT/${kind}.shard${i}of${N}.jsonl"
        if [ ! -s "$f" ]; then
            echo "ERROR: missing or empty shard: $f" >&2
            exit 1
        fi
        files+=("$f")
    done
    cat "${files[@]}" > "$OUT/${kind}.jsonl"
    echo "merged $(wc -l < "$OUT/${kind}.jsonl") lines -> $OUT/${kind}.jsonl"
done
