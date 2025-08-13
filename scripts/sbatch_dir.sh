#!/usr/bin/env bash
# Usage: ./submit_all.sh /path/to/folder

set -euo pipefail

folder="${1:-}"

if [[ -z "$folder" || ! -d "$folder" ]]; then
    echo "Usage: $0 /path/to/folder"
    exit 1
fi

shopt -s nullglob
for file in "$folder"/*.slurm; do
    echo "Submitting: $file"
    sbatch "$file"
done

