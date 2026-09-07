#!/usr/bin/env bash

set -euo pipefail

SUBMISSION_DELAY_SECONDS="${SUBMISSION_DELAY_SECONDS:-30}"

# Set EXPECTED_MANIFESTS to a positive integer to require an exact count.
EXPECTED_MANIFESTS="${EXPECTED_MANIFESTS:-}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SLURM_JOB_SCRIPT="$SCRIPT_DIR/classify_category_uids_async.sh"
MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_DIR/data/uid_manifest}"
MANIFEST_PATTERN="${MANIFEST_PATTERN:-category_uids_*.txt}"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available on PATH" >&2
    exit 1
fi

if [[ ! -x "$SLURM_JOB_SCRIPT" ]]; then
    echo "Slurm job script is missing or not executable: $SLURM_JOB_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$MANIFEST_DIR" ]]; then
    echo "Manifest directory does not exist: $MANIFEST_DIR" >&2
    exit 1
fi

if [[ -n "$EXPECTED_MANIFESTS" ]] && {
    [[ ! "$EXPECTED_MANIFESTS" =~ ^[0-9]+$ ]] ||
    (( EXPECTED_MANIFESTS <= 0 ));
}; then
    echo "EXPECTED_MANIFESTS must be a positive integer" >&2
    exit 2
fi

MANIFEST_FILES=()
while IFS= read -r -d '' manifest_file; do
    MANIFEST_FILES+=("$manifest_file")
done < <(
    find "$MANIFEST_DIR" \
        -maxdepth 1 \
        -type f \
        -name "$MANIFEST_PATTERN" \
        -print0 \
        | sort -z
)

manifest_count=${#MANIFEST_FILES[@]}

if (( manifest_count == 0 )); then
    echo "No category UID manifests found" >&2
    echo "Directory: $MANIFEST_DIR" >&2
    echo "Pattern:   $MANIFEST_PATTERN" >&2
    exit 1
fi

if [[ -n "$EXPECTED_MANIFESTS" ]] &&
    (( manifest_count != EXPECTED_MANIFESTS )); then
    echo "Expected $EXPECTED_MANIFESTS manifests, found $manifest_count" >&2
    echo "Directory: $MANIFEST_DIR" >&2
    echo "Pattern:   $MANIFEST_PATTERN" >&2
    exit 1
fi

echo "Submitting $manifest_count category-classification jobs"
echo "Table:          polymer_material_papers"
echo "Result column:  category_class"
echo "Manifest dir:   $MANIFEST_DIR"
echo "Pattern:        $MANIFEST_PATTERN"
echo "Delay:          ${SUBMISSION_DELAY_SECONDS}s"

submitted=0
for manifest_file in "${MANIFEST_FILES[@]}"; do
    submitted=$((submitted + 1))

    manifest_name=$(basename -- "$manifest_file")
    if [[ ! "$manifest_name" =~ ^category_uids_([0-9]+)\.txt$ ]]; then
        echo "Cannot determine manifest index from: $manifest_name" >&2
        echo "Expected a name such as category_uids_0001.txt" >&2
        exit 1
    fi

    manifest_index=${BASH_REMATCH[1]}
    manifest_index_number=$((10#$manifest_index))
    server_port=$((8000 + manifest_index_number))
    job_name=$(printf 'category-uids-%04d' "$manifest_index_number")

    echo "[$submitted/$manifest_count] Submitting $manifest_name as $job_name on port $server_port"
    sbatch_output=$(sbatch \
        --job-name="$job_name" \
        "$SLURM_JOB_SCRIPT" \
        "$manifest_file" \
        "$manifest_index_number")
    echo "[$submitted/$manifest_count] $sbatch_output"

    if (( submitted < manifest_count )); then
        sleep "$SUBMISSION_DELAY_SECONDS"
    fi
done

echo "Submitted all $submitted category-classification jobs"
