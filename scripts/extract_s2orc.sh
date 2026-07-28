#!/bin/bash -l
#SBATCH -A "naiss2026-3-549-cpu"
#SBATCH -p cpu
#SBATCH -J s2orc_downloader
#SBATCH -t 70:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-user=kevinge@chalmers.se
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ml Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/naiss2024-5-630/personal/george/envs/extract

export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

python -m synth_extract.mining.tdm.s2orc \
    --db data/central_papers.db \
    --s2orc-db data/s2orc.db \
    --corpus-file data/s2orc/s2orc_filtered_polymer.jsonl.gz \
    --output-dir data/fulltext/s2orc \
    --progress-every 10000 \
    --log-level INFO
