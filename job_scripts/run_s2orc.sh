#!/bin/bash -l
#SBATCH -A "naiss2026-3-549-cpu"
#SBATCH -p cpu
#SBATCH -J s2orc_filter_ti_abs_polymer
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-user=kevinge@chalmers.se
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ml Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/naiss2024-5-630/personal/george/envs/extract

export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "Warming page cache for abstracts.db..."
time cat data/s2ag/abstracts.db > /dev/null

python -m synth_extract.mining.s2orc