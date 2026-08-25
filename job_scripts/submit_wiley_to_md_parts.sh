#!/usr/bin/env bash

set -euo pipefail

BASE="/nobackup/proj/disk/naiss2024-5-630/personal/george"
PROJECT_DIR="$BASE/synth_extract"
PROCESS_DIR="$PROJECT_DIR/data/process"
WORKER_JOB="$PROJECT_DIR/job_scripts/convert_wiley_to_md.sh"
PART_COUNT=36
FIRST_PORT=8001

if ! command -v sbatch >/dev/null 2>&1; then
    echo "Required command is unavailable: sbatch" >&2
    exit 1
fi

if [[ ! -f "$WORKER_JOB" ]]; then
    echo "Wiley conversion job does not exist: $WORKER_JOB" >&2
    exit 1
fi

# Validate every input before submitting any jobs, avoiding a partial launch.
for ((part_number = 1; part_number <= PART_COUNT; part_number++)); do
    part_db="$PROCESS_DIR/wiley_track_part_${part_number}.db"
    if [[ ! -f "$part_db" ]]; then
        echo "Wiley part database does not exist: $part_db" >&2
        exit 1
    fi
done

for ((part_number = 1; part_number <= PART_COUNT; part_number++)); do
    part_db="$PROCESS_DIR/wiley_track_part_${part_number}.db"
    server_port=$((FIRST_PORT + part_number - 1))
    job_name="wiley-pdf-to-md-part-${part_number}"

    echo "Submitting part $part_number: DB=$part_db port=$server_port"
    sbatch \
        --job-name="$job_name" \
        --export="ALL,DB_PATH=$part_db,SERVER_PORT=$server_port" \
        "$WORKER_JOB"

    if ((part_number < PART_COUNT)); then
        echo "Waiting 60 seconds before the next submission..."
        sleep 60
    fi
done
