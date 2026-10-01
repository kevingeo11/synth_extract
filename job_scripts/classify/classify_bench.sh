#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-549-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=qwen-classify-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --mail-user=kevinge@chalmers.se
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

BASE="/nobackup/proj/disk/naiss2024-5-630/personal/george"
PROJECT_DIR="$BASE/synth_extract"
ENV_PATH="$BASE/envs/vllm-extract"
MODEL_DIR="$BASE/models"
MODEL_PATH="$MODEL_DIR/Qwen3.6-27B" #gemma-3-27b-it, Qwen3.6-27B
MODEL_NAME="qwen3.6-27b" #qwen3.6-27b, gemma-3-27b-it
RESULT_COLUMN="qwen3.6-27B" #gemma-3-27b-it, qwen3.6-27B
API_KEY="not-required"

SERVER_HOST="127.0.0.1"
SERVER_PORT="8000"
SERVER_URL="http://$SERVER_HOST:$SERVER_PORT"
BASE_URL="$SERVER_URL/v1"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-1800}"

SYNC_LIMIT=100
ASYNC_LIMIT=100
MAX_PARALLEL_REQUESTS=96
ASYNC_WRITE_BATCH_SIZE=25
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-300}"
MAX_TOKENS=10000
# Optional JSON object for model/provider-specific request fields.
# Example: EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
# EXTRA_BODY_JSON=""
EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'

DB_PATH="$PROJECT_DIR/data/central_workspace.db"
LOG_DIR="$PROJECT_DIR/logs"
JOB_TAG="${SLURM_JOB_ID:-manual}"
VLLM_LOG="$LOG_DIR/vllm-$JOB_TAG.log"

VLLM_PID=""

cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true

        for _ in $(seq 1 30); do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM did not stop within 30 seconds; forcing shutdown"
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi

        wait "$VLLM_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Load the Arrhenius GPU software stack.
module load GPU/buildenv-nvhpc/25.9-cu13.0
module load GPU/Miniforge/26.3.2-2-eb

# Enable Mamba in this non-interactive shell and activate the project environment.
eval "$(mamba shell hook --shell bash)"
mamba activate "$ENV_PATH"

# Keep model, compiler, and framework caches on persistent project storage.
export MODEL_DIR
export HF_HOME="$BASE/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export XDG_CACHE_HOME="$BASE/cache"
export VLLM_CACHE_ROOT="$BASE/cache/vllm"
export TORCH_HOME="$BASE/cache/torch"
export TRITON_CACHE_DIR="$BASE/cache/triton"
export CUDA_CACHE_PATH="$BASE/cache/cuda"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=true

mkdir -p \
    "$MODEL_DIR" \
    "$HF_HUB_CACHE" \
    "$HF_ASSETS_CACHE" \
    "$VLLM_CACHE_ROOT" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$CUDA_CACHE_PATH" \
    "$LOG_DIR"

cd "$PROJECT_DIR"

for required_path in \
    "$ENV_PATH" \
    "$MODEL_PATH" \
    "$DB_PATH" \
    "$PROJECT_DIR/scripts/classify_papers.py" \
    "$PROJECT_DIR/scripts/classify_papers_async.py"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

for required_command in python vllm curl nvidia-smi; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "Required command is unavailable: $required_command" >&2
        exit 1
    fi
done

echo "============================================================"
echo "Qwen classification benchmark"
echo "============================================================"
echo "Job ID:             ${SLURM_JOB_ID:-unknown}"
echo "Node:               $(hostname)"
echo "Working directory:  $(pwd)"
echo "Environment:        ${CONDA_PREFIX:-not activated}"
echo "Python:             $(command -v python)"
echo "Python version:     $(python --version 2>&1)"
echo "CPU cores:          ${SLURM_CPUS_PER_TASK:-16}"
echo "Memory allocation:  ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Parallel requests:  $MAX_PARALLEL_REQUESTS"
echo "Maximum tokens:      $MAX_TOKENS"
echo "Extra request body:  ${EXTRA_BODY_JSON:-not set}"
echo "Started:            $(date)"
echo
echo "GPUs:"
nvidia-smi \
    --query-gpu=index,name,memory.total \
    --format=csv,noheader
echo "============================================================"

echo "Starting vLLM; server output will be written to $VLLM_LOG"

vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-num-seqs "$MAX_PARALLEL_REQUESTS" \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    >"$VLLM_LOG" 2>&1 &

# vllm serve "$MODEL_PATH" \
#     --served-model-name "$MODEL_NAME" \
#     --dtype bfloat16 \
#     --max-model-len 65536 \
#     --max-num-seqs "$MAX_PARALLEL_REQUESTS" \
#     --language-model-only \
#     --enable-prefix-caching \
#     --gpu-memory-utilization 0.90 \
#     --host "$SERVER_HOST" \
#     --port "$SERVER_PORT" \
#     >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!

echo "Waiting up to $SERVER_START_TIMEOUT_SECONDS seconds for vLLM readiness"
readiness_deadline=$((SECONDS + SERVER_START_TIMEOUT_SECONDS))

until curl --fail --silent --show-error "$SERVER_URL/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready. Last 100 server log lines:" >&2
        tail -n 100 "$VLLM_LOG" >&2 || true
        exit 1
    fi

    if (( SECONDS >= readiness_deadline )); then
        echo "Timed out waiting for vLLM. Last 100 server log lines:" >&2
        tail -n 100 "$VLLM_LOG" >&2 || true
        exit 1
    fi

    sleep 5
done

# Confirm that the OpenAI-compatible model registry is also responding.
if ! curl --fail --silent --show-error "$BASE_URL/models" >/dev/null; then
    echo "vLLM health check passed, but /v1/models is unavailable" >&2
    exit 1
fi

echo "vLLM is ready at $BASE_URL"

CLASSIFIER_ARGS=(
    --db-path "$DB_PATH"
    --result-column "$RESULT_COLUMN"
    --model "$MODEL_NAME"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --timeout "$REQUEST_TIMEOUT_SECONDS"
    --max-tokens "$MAX_TOKENS"
)

if [[ -n "$EXTRA_BODY_JSON" ]]; then
    CLASSIFIER_ARGS+=(--extra-body "$EXTRA_BODY_JSON")
fi

echo
echo "Running synchronous classifier for $SYNC_LIMIT pending papers"
python scripts/classify_papers.py \
    "${CLASSIFIER_ARGS[@]}" \
    --limit "$SYNC_LIMIT" \
    --batch-size "$SYNC_LIMIT"

echo
echo "Running asynchronous classifier for the next $ASYNC_LIMIT pending papers"
python scripts/classify_papers_async.py \
    "${CLASSIFIER_ARGS[@]}" \
    --limit "$ASYNC_LIMIT" \
    --batch-size "$ASYNC_LIMIT" \
    --max-parallel-requests "$MAX_PARALLEL_REQUESTS" \
    --write-batch-size "$ASYNC_WRITE_BATCH_SIZE"

echo
echo "Both classification runs completed successfully at $(date)"
echo "Classifier/job output: Slurm .out file"
echo "vLLM server output:    $VLLM_LOG"
