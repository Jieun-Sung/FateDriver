#!/usr/bin/env bash
#
# Convenience wrapper: train FateDriver on one of the datasets listed in
# data/datasets.tsv using the paper hyper-parameters (all defaults).
#
# Usage:
#     bash scripts/run_fatedriver.sh <dataset> [seed] [device_num] [extra train_fatedriver.py args...]
#
# Example:
#     bash scripts/run_fatedriver.sh GSE242423 42 0
#     bash scripts/run_fatedriver.sh GSE237173_TC_aVEC 42 0
#
# Outputs are written to  outputs/<dataset>/run_<timestamp>_dataset-<dataset>_seed-<seed>/
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
TABLE="${DATA_DIR}/datasets.tsv"
PY="${PYTHON_BIN:-python}"

DATASET="${1:?usage: run_fatedriver.sh <dataset> [seed] [device_num]}"
SEED="${2:-42}"
DEVICE="${3:-0}"

# Look up the dataset row (tab-separated, skipping comment lines).
row="$(awk -F'\t' -v k="${DATASET}" '$1==k {print; exit}' "${TABLE}")"
[ -n "${row}" ] || { echo "dataset '${DATASET}' not found in ${TABLE}"; exit 1; }
IFS=$'\t' read -r name species source target obs driver downstream nichenet anndata pseudotime pt_key <<< "${row}"

export MPLBACKEND=Agg
"${PY}" "${REPO_ROOT}/src/train_fatedriver.py" \
    --anndata_path        "${DATA_DIR}/${anndata}" \
    --pseudotime_csv_path "${DATA_DIR}/${pseudotime}" \
    --pseudotime_key      "${pt_key}" \
    --driver_genes_path   "${DATA_DIR}/${driver}" \
    --affected_genes_path "${DATA_DIR}/${downstream}" \
    --gene_nichenet_df_path "${DATA_DIR}/${nichenet}" \
    --source_cell_type    "${source}" \
    --target_cell_type    "${target}" \
    --obs_column          "${obs}" \
    --dataset             "${name}" \
    --output_dir          "${REPO_ROOT}/outputs/${name}" \
    --seed                "${SEED}" \
    --device_num          "${DEVICE}" \
    "${@:4}"
