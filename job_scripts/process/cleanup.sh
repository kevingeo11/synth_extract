#!/bin/bash -l
#SBATCH -A "naiss2026-3-549-cpu"
#SBATCH -p cpu
#SBATCH -J tdm_cleanup
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-user=kevinge@chalmers.se
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ml Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/naiss2024-5-630/personal/george/envs/extract

export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# Safe default: audit and report without deleting anything.
# Submit with `sbatch --export=ALL,DRY_RUN=0 scripts/cleanup.sh` after reviewing
# the dry-run log to remove unexpected failed/exhausted UID directories.
DRY_RUN="${DRY_RUN:-1}"
DRY_RUN_ARG=()
if [[ "$DRY_RUN" != "0" ]]; then
    DRY_RUN_ARG=(--dry-run)
fi

python -m synth_extract.mining.tdm.cleanup \
    --db data/central_papers.db \
    --fulltext-root data/fulltext \
    --path-base "$SLURM_SUBMIT_DIR" \
    --log-level INFO \
    "${DRY_RUN_ARG[@]}"
