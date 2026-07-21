#!/bin/bash -l
#SBATCH -A "naiss2026-3-549-cpu"
#SBATCH -p cpu
#SBATCH -J wiley_downloader
#SBATCH -t 70:00:00
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

# Load secrets from .env and export them to the Python process.
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
else
    echo "ERROR: .env file not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

# Confirm that the Wiley API key is available without printing it.
if [[ -z "${WILEY_API_KEY:-}" ]]; then
    echo "ERROR: WILEY_API_KEY is not defined in .env"
    exit 1
fi

python -m synth_extract.mining.tdm.wiley \
    --db data/central_papers.db \
    --output-dir data/fulltext/wiley \
    --log-level INFO
