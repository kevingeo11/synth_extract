#!/bin/bash -l
#SBATCH -A "naiss2026-3-549-cpu"
#SBATCH -p cpu
#SBATCH -J scopus_elsevier
#SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -o logs/%x-%j.out

set -euo pipefail

mkdir -p logs

# ml Miniforge/26.3.2-2-eb
# ml Python/3.13.5-bare-gcc-2025b-eb
mamba activate /nobackup/proj/disk/naiss2024-5-630/personal/george/envs/extract

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python scripts/run_scopus_elsevier.py
