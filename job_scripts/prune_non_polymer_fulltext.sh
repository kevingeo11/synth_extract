#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-679-cpu
#SBATCH --partition=cpu
#SBATCH --job-name=prune-non-polymer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract/logs/%x-%j.err
#SBATCH --mail-user=kevinge@chalmers.se
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch job_scripts/prune_non_polymer_fulltext.sh [dry-run|delete] [SOURCE ...]

Examples:
  sbatch job_scripts/prune_non_polymer_fulltext.sh
  sbatch job_scripts/prune_non_polymer_fulltext.sh dry-run
  sbatch job_scripts/prune_non_polymer_fulltext.sh dry-run wiley arxiv
  sbatch job_scripts/prune_non_polymer_fulltext.sh delete

The default mode is dry-run. The delete mode permanently removes UID folders
whose UID is absent from polymer_material_papers.
EOF
}

if (( $# > 0 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

MODE=${1:-dry-run}
if (( $# > 0 )); then
    shift
fi

if [[ "$MODE" != "dry-run" && "$MODE" != "delete" ]]; then
    echo "Mode must be either 'dry-run' or 'delete': $MODE" >&2
    usage >&2
    exit 2
fi

BASE="/nobackup/proj/disk/naiss2024-5-630/personal/george"
PROJECT_DIR="$BASE/synth_extract"
ENV_PATH="$BASE/envs/extract"

DB_PATH="${DB_PATH:-$PROJECT_DIR/data/central_papers.db}"
FULLTEXT_ROOT="${FULLTEXT_ROOT:-$PROJECT_DIR/data/fulltext}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

LOG_DIR="$PROJECT_DIR/logs"
PYTHON_SCRIPT="$PROJECT_DIR/scripts/prune_non_polymer_fulltext.py"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# Load the Arrhenius CPU software stack.
ml Miniforge/26.3.2-2-eb
eval "$(mamba shell hook --shell bash)"
mamba activate "$ENV_PATH"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

for required_path in \
    "$ENV_PATH" \
    "$DB_PATH" \
    "$FULLTEXT_ROOT" \
    "$PYTHON_SCRIPT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

if ! command -v python >/dev/null 2>&1; then
    echo "python is not available after activating $ENV_PATH" >&2
    exit 1
fi

PRUNE_ARGS=(
    --db-path "$DB_PATH"
    --fulltext-root "$FULLTEXT_ROOT"
    --log-level "$LOG_LEVEL"
)

if [[ "$MODE" == "delete" ]]; then
    PRUNE_ARGS+=(--delete)
fi

for source in "$@"; do
    PRUNE_ARGS+=(--source "$source")
done

echo "============================================================"
echo "Prune non-polymer full-text UID folders"
echo "============================================================"
echo "Job ID:            ${SLURM_JOB_ID:-unknown}"
echo "Node:              $(hostname)"
echo "Mode:              $MODE"
echo "Database:          $DB_PATH"
echo "Table:             polymer_material_papers"
echo "Full-text root:    $FULLTEXT_ROOT"
echo "Sources:           ${*:-all}"
echo "Python:            $(command -v python)"
echo "Started:           $(date)"
echo "============================================================"

if [[ "$MODE" == "delete" ]]; then
    echo "WARNING: permanent deletion is enabled"
else
    echo "Dry run only; no UID folders will be removed"
fi

python "$PYTHON_SCRIPT" "${PRUNE_ARGS[@]}"

echo "============================================================"
echo "Pruning job completed"
echo "Finished: $(date)"
echo "============================================================"
