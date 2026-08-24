#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-549-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=wiley-pdf-to-md-dev
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
ENV_PATH="$BASE/envs/vllm-extract"

DB_PATH="$PROJECT_DIR/data/development_set/wiley_track.db"
FULLTEXT_ROOT="$PROJECT_DIR/data/development_set"
LOG_DIR="$PROJECT_DIR/logs"
COMMIT_EVERY="${COMMIT_EVERY:-1}"
SERVER_HOST="127.0.0.1"
SERVER_PORT="8000"
SERVER_URL="http://$SERVER_HOST:$SERVER_PORT"
SERVER_START_TIMEOUT_SECONDS=600
SURYA_MODEL="datalab-to/surya-ocr-2"
JOB_TAG="${SLURM_JOB_ID:-manual}"
VLLM_LOG="$LOG_DIR/surya-vllm-$JOB_TAG.log"
VLLM_PID=""

cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping Surya vLLM server (PID $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export SURYA_INFERENCE_BACKEND="vllm"
export SURYA_INFERENCE_URL="$SERVER_URL/v1"
export SURYA_INFERENCE_AUTOSTART="false"

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

for required_command in python nvidia-smi vllm curl; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "Required command is unavailable: $required_command" >&2
        exit 1
    fi
done

echo "============================================================"
echo "Wiley development-set PDF-to-Markdown conversion"
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
echo "Surya model:       $SURYA_MODEL"
echo "Surya API URL:     $SURYA_INFERENCE_URL"
echo "vLLM log:          $VLLM_LOG"
echo "Started:           $(date)"
echo
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

echo "Starting the Surya OCR vLLM server..."
vllm serve "$SURYA_MODEL" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --dtype bfloat16 \
    --max-model-len 18000 \
    --gpu-memory-utilization 0.85 \
    >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!
server_wait_started=$SECONDS

echo "Waiting up to $SERVER_START_TIMEOUT_SECONDS seconds for Surya vLLM..."
until curl -sf "$SERVER_URL/v1/models" >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Surya vLLM failed to start. Last 100 log lines:" >&2
        tail -n 100 "$VLLM_LOG" >&2 || true
        exit 1
    fi

    if (( SECONDS - server_wait_started >= SERVER_START_TIMEOUT_SECONDS )); then
        echo "Timed out after $SERVER_START_TIMEOUT_SECONDS seconds waiting for Surya vLLM." >&2
        echo "Last 100 log lines:" >&2
        tail -n 100 "$VLLM_LOG" >&2 || true
        exit 1
    fi

    sleep 2
done

echo "Surya vLLM is ready."

python scripts/process/pdf_to_md.py \
    --db "$DB_PATH" \
    --fulltext-root "$FULLTEXT_ROOT" \
    --commit-every "$COMMIT_EVERY" \
    --log-level INFO

echo "Finished: $(date)"
