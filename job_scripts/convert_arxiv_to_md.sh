#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-549-cpu
#SBATCH --partition=cpu
#SBATCH --job-name=arxiv-pdf-to-md-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --mail-user=kevinge@chalmers.se
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

BASE="/nobackup/proj/disk/naiss2024-5-630/personal/george"
PROJECT_DIR="$BASE/synth_extract"
ENV_PATH="$BASE/envs/extract"

DB_PATH="$PROJECT_DIR/data/development_set/arxiv_track.db"
FULLTEXT_ROOT="$PROJECT_DIR/data/development_set"
LOG_DIR="$PROJECT_DIR/logs"
COMMIT_EVERY="${COMMIT_EVERY:-1}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# Load the Arrhenius CPU software stack.
ml Miniforge/26.3.2-2-eb

eval "$(mamba shell hook --shell bash)"
mamba activate "$ENV_PATH"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XDG_CACHE_HOME="$BASE/cache"
export TORCH_HOME="$BASE/cache/torch"
export HF_HOME="$BASE/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME" \
    "$HF_HUB_CACHE"

for required_path in \
    "$ENV_PATH" \
    "$DB_PATH" \
    "$FULLTEXT_ROOT" \
    "$PROJECT_DIR/scripts/process/pdf_to_md.py"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

for required_command in python; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "Required command is unavailable: $required_command" >&2
        exit 1
    fi
done

echo "============================================================"
echo "ArXiv development-set PDF-to-Markdown conversion"
echo "============================================================"
echo "Job ID:            ${SLURM_JOB_ID:-unknown}"
echo "Node:              $(hostname)"
echo "Working directory: $(pwd)"
echo "Environment:       ${CONDA_PREFIX:-not activated}"
echo "Python:            $(command -v python)"
echo "Python version:    $(python --version 2>&1)"
echo "Tracking database: $DB_PATH"
echo "Full-text root:    $FULLTEXT_ROOT"
echo "Commit interval:   $COMMIT_EVERY"
echo "Started:           $(date)"
echo "============================================================"

python scripts/process/pdf_to_md.py \
    --db "$DB_PATH" \
    --fulltext-root "$FULLTEXT_ROOT" \
    --commit-every "$COMMIT_EVERY" \
    --log-level INFO

echo "Finished: $(date)"
