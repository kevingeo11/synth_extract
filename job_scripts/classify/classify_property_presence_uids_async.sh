#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-679-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=property-presence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.err

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch job_scripts/classify/classify_property_presence_uids_async.sh MANIFEST_FILE [MANIFEST_INDEX]

Example:
  sbatch job_scripts/classify/classify_property_presence_uids_async.sh \
    data/uid_manifest/property_classification_uids_0001.txt 1

MANIFEST_INDEX defaults to the trailing number in the manifest filename.
The vLLM port is 8000 + MANIFEST_INDEX. The worker updates the paper-level
property_classification table in data/polymer_samples.db.
EOF
}

if (( $# < 1 || $# > 2 )); then
    usage >&2
    exit 2
fi

MANIFEST_INPUT=$1
MANIFEST_INDEX=${2:-}
if [[ -z "$MANIFEST_INDEX" ]]; then
    manifest_name=$(basename -- "$MANIFEST_INPUT")
    if [[ "$manifest_name" =~ ([0-9]+)(\.[^.]+)?$ ]]; then
        MANIFEST_INDEX=${BASH_REMATCH[1]}
    else
        echo "Could not determine manifest index from: $MANIFEST_INPUT" >&2
        exit 2
    fi
fi
if [[ ! "$MANIFEST_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Manifest index must be a non-negative integer: $MANIFEST_INDEX" >&2
    exit 2
fi

MANIFEST_INDEX_NUMBER=$((10#$MANIFEST_INDEX))
SERVER_PORT=$((8000 + MANIFEST_INDEX_NUMBER))
if (( SERVER_PORT > 65535 )); then
    echo "Calculated port is outside the valid range: $SERVER_PORT" >&2
    exit 2
fi

BASE=/nobackup/proj/disk/naiss2024-5-630/personal/george
PROJECT_DIR="$BASE/synth_extract"
ENV_PATH="$BASE/envs/vllm-extract"
MODEL_DIR="$BASE/models"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/qwen3.6-27b}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-27b}"
API_KEY="${API_KEY:-not-required}"

DB_PATH="${DB_PATH:-$PROJECT_DIR/data/polymer_samples.db}"
FULLTEXT_ROOT="${FULLTEXT_ROOT:-$PROJECT_DIR/data/fulltext}"
PYTHON_SCRIPT="$PROJECT_DIR/scripts/classify/classify_property_presence_uids_async.py"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-$PROJECT_DIR/synth_extract/agents/classification/prompts/property_presence/system_prompt.md}"
USER_TEMPLATE_PATH="${USER_TEMPLATE_PATH:-$PROJECT_DIR/synth_extract/agents/classification/prompts/property_presence/user_template.md}"

if [[ "$MANIFEST_INPUT" = /* ]]; then
    UID_FILE=$MANIFEST_INPUT
else
    UID_FILE="${SLURM_SUBMIT_DIR:-$PWD}/$MANIFEST_INPUT"
fi

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_URL="http://$SERVER_HOST:$SERVER_PORT"
BASE_URL="$SERVER_URL/v1"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-1800}"
BATCH_SIZE="${BATCH_SIZE:-25}"
MAX_PARALLEL_REQUESTS="${MAX_PARALLEL_REQUESTS:-8}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-300}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
SQLITE_TIMEOUT_SECONDS="${SQLITE_TIMEOUT_SECONDS:-60}"
SQLITE_WRITE_RETRIES="${SQLITE_WRITE_RETRIES:-5}"
SQLITE_RETRY_BASE_DELAY="${SQLITE_RETRY_BASE_DELAY:-1}"
SQLITE_RETRY_MAX_DELAY="${SQLITE_RETRY_MAX_DELAY:-30}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
LIMIT="${LIMIT:-}"
RETRY_FAILED="${RETRY_FAILED:-0}"
RETRY_IN_PROGRESS="${RETRY_IN_PROGRESS:-0}"

for boolean_name in RETRY_FAILED RETRY_IN_PROGRESS; do
    boolean_value=${!boolean_name}
    if [[ "$boolean_value" != "0" && "$boolean_value" != "1" ]]; then
        echo "$boolean_name must be 0 or 1, not: $boolean_value" >&2
        exit 2
    fi
done

LOG_DIR="$PROJECT_DIR/logs"
VLLM_LOG="$LOG_DIR/vllm-property-presence-${SLURM_JOB_ID:-manual}.log"

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        for _ in {1..30}; do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                return
            fi
            sleep 1
        done
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

module load GPU/buildenv-nvhpc/25.9-cu13.0
module load GPU/Miniforge/26.3.2-2-eb
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

export HF_HOME="${HF_HOME:-$BASE/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$BASE/cache/vllm}"
export TORCH_HOME="${TORCH_HOME:-$BASE/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BASE/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$BASE/cache/cuda}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=true

mkdir -p \
    "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$LOG_DIR"
cd "$PROJECT_DIR"

for required_path in \
    "$ENV_PATH" "$MODEL_PATH" "$DB_PATH" "$FULLTEXT_ROOT" "$UID_FILE" \
    "$PYTHON_SCRIPT" "$SYSTEM_PROMPT_PATH" "$USER_TEMPLATE_PATH"; do
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
echo "Asynchronous property-presence classification"
echo "============================================================"
echo "Job ID:                 ${SLURM_JOB_ID:-manual}"
echo "Node:                   $(hostname)"
echo "Manifest:               $UID_FILE"
echo "Manifest index:         $MANIFEST_INDEX_NUMBER"
echo "Database:               $DB_PATH"
echo "Table:                  property_classification"
echo "Full-text root:         $FULLTEXT_ROOT"
echo "System prompt:          $SYSTEM_PROMPT_PATH"
echo "User template:          $USER_TEMPLATE_PATH"
echo "Model:                  $MODEL_NAME"
echo "vLLM port:              $SERVER_PORT"
echo "Parallel requests:      $MAX_PARALLEL_REQUESTS"
echo "Classification batch:   $BATCH_SIZE"
echo "Maximum tokens:         $MAX_TOKENS"
echo "Retry failed:           $RETRY_FAILED"
echo "Retry in progress:      $RETRY_IN_PROGRESS"
echo "Extra request body:     $EXTRA_BODY_JSON"
echo "vLLM log:               $VLLM_LOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype bfloat16 \
    --max-model-len 262144 \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Started vLLM server (PID $VLLM_PID); waiting for readiness"
deadline=$((SECONDS + SERVER_START_TIMEOUT_SECONDS))
until curl --silent --fail "$SERVER_URL/health" >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready. Last log lines:" >&2
        tail -n 100 "$VLLM_LOG" >&2 || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for vLLM after ${SERVER_START_TIMEOUT_SECONDS}s" >&2
        tail -n 100 "$VLLM_LOG" >&2 || true
        exit 1
    fi
    sleep 5
done
curl --silent --fail "$BASE_URL/models" >/dev/null
echo "vLLM is ready at $BASE_URL"

CLASSIFIER_ARGS=(
    --db-path "$DB_PATH"
    --uid-file "$UID_FILE"
    --fulltext-root "$FULLTEXT_ROOT"
    --system-prompt-path "$SYSTEM_PROMPT_PATH"
    --user-template-path "$USER_TEMPLATE_PATH"
    --model "$MODEL_NAME"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --batch-size "$BATCH_SIZE"
    --max-parallel-requests "$MAX_PARALLEL_REQUESTS"
    --timeout "$REQUEST_TIMEOUT_SECONDS"
    --max-tokens "$MAX_TOKENS"
    --sqlite-timeout "$SQLITE_TIMEOUT_SECONDS"
    --sqlite-write-retries "$SQLITE_WRITE_RETRIES"
    --sqlite-retry-base-delay "$SQLITE_RETRY_BASE_DELAY"
    --sqlite-retry-max-delay "$SQLITE_RETRY_MAX_DELAY"
    --extra-body "$EXTRA_BODY_JSON"
)
if [[ -n "$LIMIT" ]]; then
    CLASSIFIER_ARGS+=(--limit "$LIMIT")
fi
if [[ "$RETRY_FAILED" == "1" ]]; then
    CLASSIFIER_ARGS+=(--retry-failed)
fi
if [[ "$RETRY_IN_PROGRESS" == "1" ]]; then
    CLASSIFIER_ARGS+=(--retry-in-progress)
fi

python "$PYTHON_SCRIPT" "${CLASSIFIER_ARGS[@]}"
echo "Property-presence classification completed at $(date)"
