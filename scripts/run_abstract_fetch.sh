#!/bin/bash -l
#SBATCH -A "naiss2026-3-549-cpu"
#SBATCH -p cpu
#SBATCH -J scopus_s_fetch_elsevier
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
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
import os
import pybliometrics
from synth_extract.mining.abstract_fetch import fill_missing_abstracts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

pybliometrics.init(keys=[os.getenv("SCOPUS_API_KEY")])
logging.info(pybliometrics.utils.constants.CONFIG_FILE)

stats = fill_missing_abstracts(
    db_path="data/scopus_elsevier.db",
    table_name="papers_enriched",
    batch_size=25,
    quota_threshold=1000,
    delay_seconds=0.5,
    max_batches=None,
)

logging.info(f"fill_missing_abstracts finished: {stats}")
PY

# python - <<'PY'
# import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )

# from synth_extract.mining.abstract_fetch import reset_error_sources
# stats = reset_error_sources(
#     db_path="data/scopus_elsevier.db",
#     table_name="papers_enriched",
# )
# logging.info(f"reset_error_sources finished: {stats}")
# PY