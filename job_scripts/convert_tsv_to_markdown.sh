#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-549-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=pdf-to-markdown
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --mail-user=kevinge@chalmers.se
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

BASE="/nobackup/proj/disk/naiss2024-5-630/personal/george"
PROJECT_DIR="$BASE/synth_extract"
ENV_PATH="$BASE/envs/extract"

# Override these when submitting if a different source or location is needed:
# sbatch --export=ALL,CANONICAL_SOURCE=wiley job_scripts/convert_tsv_to_markdown.sh
CANONICAL_SOURCE="${CANONICAL_SOURCE:-arxiv}"
TSV_PATH="${TSV_PATH:-$PROJECT_DIR/data/development_set/development_sample.tsv}"
FULLTEXT_ROOT="${FULLTEXT_ROOT:-$PROJECT_DIR/data/development_set}"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# Load the Arrhenius GPU software stack.
module load GPU/buildenv-nvhpc/25.9-cu13.0
module load GPU/Miniforge/26.3.2-2-eb

eval "$(mamba shell hook --shell bash)"
mamba activate "$ENV_PATH"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XDG_CACHE_HOME="$BASE/cache"
export TORCH_HOME="$BASE/cache/torch"
export HF_HOME="$BASE/cache/huggingface"

for required_path in \
    "$ENV_PATH" \
    "$TSV_PATH" \
    "$FULLTEXT_ROOT" \
    "$PROJECT_DIR/scripts/convert_tsv_to_markdown.py"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

for required_command in python nvidia-smi; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "Required command is unavailable: $required_command" >&2
        exit 1
    fi
done

echo "============================================================"
echo "TSV PDF-to-Markdown conversion"
echo "============================================================"
echo "Job ID:            ${SLURM_JOB_ID:-unknown}"
echo "Node:              $(hostname)"
echo "Canonical source:  $CANONICAL_SOURCE"
echo "TSV:               $TSV_PATH"
echo "Full-text root:    $FULLTEXT_ROOT"
echo "Environment:       ${CONDA_PREFIX:-not activated}"
echo "Started:           $(date)"
echo
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

python scripts/convert_tsv_to_markdown.py \
    --tsv "$TSV_PATH" \
    --canonical-source "$CANONICAL_SOURCE" \
    --fulltext-root "$FULLTEXT_ROOT" \
    --log-level INFO

echo "Finished: $(date)"
