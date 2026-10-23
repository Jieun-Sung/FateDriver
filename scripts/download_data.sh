#!/usr/bin/env bash
#
# Download the released FateDriver datasets (unified AnnData + pseudotime) and
# the gene-set / NicheNet tables from Zenodo.
#
#   Record: https://doi.org/10.5281/zenodo.21771206
#
# The record holds one zip per dataset plus `gene_sets.zip`. Each zip contains a
# single top-level directory with the same name, so everything is simply
# extracted into `data/`:
#
#     data/GSE242423/{anndata.h5ad,pseudotime.csv}
#     data/GSE297234_GM00731/{anndata.h5ad,pseudotime.csv}
#     ...
#     data/gene_sets/*.csv
#
# Usage (from the repository root):
#     bash scripts/download_data.sh                  # everything
#     bash scripts/download_data.sh GSE98567 gene_sets   # only these
#
# Environment variables:
#     ZENODO_RECORD    Zenodo record id            (default: 21771206)
#     ZENODO_BASE_URL  file base URL, overrides the record id
#     KEEP_ZIP=1       keep the downloaded .zip files under data/
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"

ZENODO_RECORD="${ZENODO_RECORD:-21771206}"
ZENODO_BASE_URL="${ZENODO_BASE_URL:-https://zenodo.org/records/${ZENODO_RECORD}/files}"

ALL_ITEMS=(
  GSE242423
  GSE297234_GM00731
  GSE297234_GM23815
  GSE98567
  GSE181451_Primed_MEF
  GSE181451_Baseline_MEF
  gene_sets
)

if [ "$#" -gt 0 ]; then
  ITEMS=("$@")
else
  ITEMS=("${ALL_ITEMS[@]}")
fi

command -v curl  >/dev/null || { echo "curl is required"; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required"; exit 1; }

mkdir -p "${DATA_DIR}"
for item in "${ITEMS[@]}"; do
  zip_path="${DATA_DIR}/${item}.zip"
  echo ">>> downloading ${item}.zip"
  curl -L --fail --retry 3 -o "${zip_path}" "${ZENODO_BASE_URL}/${item}.zip?download=1"
  echo ">>> extracting into ${DATA_DIR}"
  unzip -o -q "${zip_path}" -d "${DATA_DIR}"
  [ "${KEEP_ZIP:-0}" = "1" ] || rm -f "${zip_path}"
done

echo
echo ">>> done. Contents of data/:"
ls -1 "${DATA_DIR}"
echo
echo "Each dataset directory must contain anndata.h5ad and pseudotime.csv;"
echo "gene sets (driver / downstream / NicheNet CSVs) live in data/gene_sets/."
