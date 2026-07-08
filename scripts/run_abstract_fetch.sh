#!/bin/bash -l
#SBATCH -A "naiss2026-3-549-cpu"
#SBATCH -p cpu
#SBATCH -J fill_abstracts
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 1
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

python - <<'PY'
import logging

from synth_extract.mining.abstract_fetch import fill_missing_abstracts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

stats = fill_missing_abstracts(
    db_path="data/scopus_elsevier.db",
    table_name="papers_enriched",
    batch_size=25,
    quota_threshold=1000,
    delay_seconds=2.0,
    max_batches=None,
)

print(f"fill_missing_abstracts finished: {stats}")
PY
