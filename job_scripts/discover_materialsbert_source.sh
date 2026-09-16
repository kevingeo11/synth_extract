#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-679-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=materialsbert-properties
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.err

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: sbatch $0 SOURCE" >&2
    echo "Example: sbatch $0 elsevier" >&2
    exit 2
fi

SOURCE=$1

BASE=/nobackup/proj/disk/naiss2024-5-630/personal/george
PROJECT_DIR="$BASE/synth_extract"
ENV_PATH="$BASE/envs/vllm-extract"

DB_PATH="${DB_PATH:-$PROJECT_DIR/data/discovery_workspace.db}"
FULLTEXT_ROOT="${FULLTEXT_ROOT:-$PROJECT_DIR/data/fulltext}"
PYTHON_SCRIPT="$PROJECT_DIR/discovery/discover_materialsbert_source.py"
MODEL="${MODEL:-pranav-s/PolymerNER}"

WORKERS="${WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PAPER_BATCH_SIZE="${PAPER_BATCH_SIZE:-64}"
MAX_CHUNK_TOKENS="${MAX_CHUNK_TOKENS:-450}"
DEVICE="${DEVICE:-0}"

module load GPU/buildenv-nvhpc/25.9-cu13.0
module load GPU/Miniforge/26.3.2-2-eb

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

export HF_HOME="${HF_HOME:-$BASE/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/cache}"
export TORCH_HOME="${TORCH_HOME:-$BASE/cache/torch}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$BASE/cache/cuda}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
    "$HF_HUB_CACHE" \
    "$TORCH_HOME" \
    "$CUDA_CACHE_PATH" \
    "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"

for required_path in \
    "$ENV_PATH" \
    "$DB_PATH" \
    "$FULLTEXT_ROOT/$SOURCE" \
    "$PYTHON_SCRIPT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

echo "============================================================"
echo "MaterialBERT property extraction"
echo "============================================================"
echo "Job ID:             ${SLURM_JOB_ID:-manual}"
echo "Node:               $(hostname)"
echo "Source:             $SOURCE"
echo "Database:           $DB_PATH"
echo "Full-text root:     $FULLTEXT_ROOT"
echo "Model:              $MODEL"
echo "Workers:            $WORKERS"
echo "Inference batch:    $BATCH_SIZE"
echo "Papers per task:    $PAPER_BATCH_SIZE"
echo "Tokens per chunk:   $MAX_CHUNK_TOKENS"
echo "Device:             $DEVICE"
echo "Started:            $(date)"
nvidia-smi \
    --query-gpu=index,name,memory.total \
    --format=csv,noheader
echo "============================================================"

python "$PYTHON_SCRIPT" \
    --source "$SOURCE" \
    --db-path "$DB_PATH" \
    --fulltext-root "$FULLTEXT_ROOT" \
    --model "$MODEL" \
    --workers "$WORKERS" \
    --batch-size "$BATCH_SIZE" \
    --paper-batch-size "$PAPER_BATCH_SIZE" \
    --max-chunk-tokens "$MAX_CHUNK_TOKENS" \
    --device "$DEVICE"

echo "Finished: $(date)"
