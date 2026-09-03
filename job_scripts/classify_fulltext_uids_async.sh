#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-549-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=classify-fulltext-uids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.err
#SBATCH --mail-user=kevinge@chalmers.se
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch job_scripts/classify_fulltext_uids_async.sh MANIFEST_FILE [RESULT_COLUMN] [MANIFEST_INDEX]

Examples:
  sbatch job_scripts/classify_fulltext_uids_async.sh data/uid_manifest/classify_uids_0001.txt
  sbatch job_scripts/classify_fulltext_uids_async.sh data/uid_manifest/classify_uids_0001.txt class_run_2 1

RESULT_COLUMN defaults to the RESULT_COLUMN environment variable, or class_run_1.
MANIFEST_INDEX defaults to the trailing number in the manifest filename.
The vLLM port is 8000 + MANIFEST_INDEX (for example, 1 -> 8001).
EOF
}

if (( $# < 1 || $# > 3 )); then
    usage >&2
    exit 2
fi

MANIFEST_INPUT=$1
RESULT_COLUMN=${2:-${RESULT_COLUMN:-class_run_1}}
MANIFEST_INDEX=${3:-}

if [[ -z "$MANIFEST_INDEX" ]]; then
    manifest_name=$(basename -- "$MANIFEST_INPUT")
    if [[ "$manifest_name" =~ ([0-9]+)(\.[^.]+)?$ ]]; then
        MANIFEST_INDEX=${BASH_REMATCH[1]}
    else
        echo "Could not determine an index from manifest: $MANIFEST_INPUT" >&2
        echo "Pass MANIFEST_INDEX as the third argument." >&2
        exit 2
    fi
fi

if [[ ! "$MANIFEST_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Manifest index must be a non-negative integer: $MANIFEST_INDEX" >&2
    exit 2
fi

# The 10# prefix makes zero-padded values such as 0008 decimal rather than octal.
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

DB_PATH="${DB_PATH:-$PROJECT_DIR/data/central_papers.db}"
FULLTEXT_ROOT="${FULLTEXT_ROOT:-$PROJECT_DIR/data/fulltext}"

# Relative manifest paths are resolved from the directory where sbatch was run.
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
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-96}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-300}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
SQLITE_TIMEOUT_SECONDS="${SQLITE_TIMEOUT_SECONDS:-60}"
SQLITE_WRITE_RETRIES="${SQLITE_WRITE_RETRIES:-5}"
SQLITE_RETRY_BASE_DELAY="${SQLITE_RETRY_BASE_DELAY:-1}"
SQLITE_RETRY_MAX_DELAY="${SQLITE_RETRY_MAX_DELAY:-30}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-{\"chat_template_kwargs\":{\"enable_thinking\":true}}}"

# Optional overrides. Leave empty to use the classifier's package defaults.
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-}"
USER_TEMPLATE_PATH="${USER_TEMPLATE_PATH:-}"
LIMIT="${LIMIT:-}"

LOG_DIR="$PROJECT_DIR/logs"
VLLM_LOG="$LOG_DIR/vllm-${SLURM_JOB_ID:-manual}.log"

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

        echo "vLLM did not stop gracefully; terminating it forcefully"
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

module load GPU/buildenv-nvhpc/25.9-cu13.0
module load GPU/Miniforge/26.3.2-2-eb

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

export HF_HOME="${HF_HOME:-$BASE/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/cache}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$LOG_DIR"
cd "$PROJECT_DIR"

for required_path in \
    "$MODEL_PATH" \
    "$DB_PATH" \
    "$FULLTEXT_ROOT" \
    "$UID_FILE" \
    "$PROJECT_DIR/scripts/classify_fulltext_uids_async.py"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

if [[ -n "$SYSTEM_PROMPT_PATH" && ! -f "$SYSTEM_PROMPT_PATH" ]]; then
    echo "System prompt does not exist: $SYSTEM_PROMPT_PATH" >&2
    exit 1
fi

if [[ -n "$USER_TEMPLATE_PATH" && ! -f "$USER_TEMPLATE_PATH" ]]; then
    echo "User template does not exist: $USER_TEMPLATE_PATH" >&2
    exit 1
fi

command -v vllm >/dev/null
command -v curl >/dev/null
command -v python >/dev/null

echo "Job ID:                 ${SLURM_JOB_ID:-manual}"
echo "Node:                   $(hostname)"
echo "Manifest:               $UID_FILE"
echo "Manifest index:         $MANIFEST_INDEX_NUMBER"
echo "Database:               $DB_PATH"
echo "Result column:          $RESULT_COLUMN"
echo "Full-text root:         $FULLTEXT_ROOT"
echo "Model:                  $MODEL_NAME"
echo "vLLM port:              $SERVER_PORT"
echo "vLLM max sequences:     $VLLM_MAX_NUM_SEQS"
echo "Parallel requests:      $MAX_PARALLEL_REQUESTS"
echo "Classification batch:   $BATCH_SIZE"
echo "SQLite write retries:   $SQLITE_WRITE_RETRIES"
echo "vLLM log:               $VLLM_LOG"
nvidia-smi || true

vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype bfloat16 \
    --max-model-len 65536 \
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
echo "vLLM is ready"

CLASSIFIER_ARGS=(
    --db-path "$DB_PATH"
    --uid-file "$UID_FILE"
    --result-column "$RESULT_COLUMN"
    --fulltext-root "$FULLTEXT_ROOT"
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

if [[ -n "$SYSTEM_PROMPT_PATH" ]]; then
    CLASSIFIER_ARGS+=(--system-prompt-path "$SYSTEM_PROMPT_PATH")
fi

if [[ -n "$USER_TEMPLATE_PATH" ]]; then
    CLASSIFIER_ARGS+=(--user-template-path "$USER_TEMPLATE_PATH")
fi

if [[ -n "$LIMIT" ]]; then
    CLASSIFIER_ARGS+=(--limit "$LIMIT")
fi

python scripts/classify_fulltext_uids_async.py "${CLASSIFIER_ARGS[@]}"

echo "Classification job completed successfully"
