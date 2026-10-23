#!/usr/bin/env bash
#
# Rebuild all published datasets in the unified FateDriver format from the raw
# preprocessed AnnData files:
#
#     <raw>.h5ad  --compute_pseudotime.py-->  <raw>_pseudotime.csv
#                 --prepare_dataset.py---->  data/<dataset>/{anndata.h5ad,pseudotime.csv}
#
# This is the exact command list used to produce the released data. It is
# provided for provenance/reproducibility; end users who download the released
# data (scripts/download_data.sh) do not need to run it.
#
# Usage:
#     RAW_ROOT=/path/to/DATA bash scripts/prepare_all_datasets.sh
#
#     COMPUTE_PSEUDOTIME=1 ...   also (re)compute pseudotime.csv with Slingshot
#     PSEUDOTIME_METHOD=dpt ...  use the pure-Python DPT method instead
#     RSCRIPT=/path/to/Rscript   Rscript from the FateDriver-pseudotime env
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
RAW_ROOT="${RAW_ROOT:?set RAW_ROOT=/path/to/DATA (directory holding the preprocessed .h5ad files)}"
PY="${PYTHON_BIN:-python}"
PREP="${THIS_DIR}/prepare_dataset.py"
PT="${THIS_DIR}/compute_pseudotime.py"
OUT="${REPO_ROOT}/data"

COMPUTE_PSEUDOTIME="${COMPUTE_PSEUDOTIME:-0}"
PSEUDOTIME_METHOD="${PSEUDOTIME_METHOD:-slingshot}"
RSCRIPT="${RSCRIPT:-Rscript}"

# pseudotime: name  src_h5ad  out_csv  group_key  root_group
pseudotime() {
  [ "${COMPUTE_PSEUDOTIME}" = "1" ] || return 0
  echo ">>> pseudotime $1 (${PSEUDOTIME_METHOD})"
  "${PY}" "${PT}" --h5ad "$2" --out_csv "$3" \
    --group_key "$4" --root_group "$5" \
    --method "${PSEUDOTIME_METHOD}" --rscript "${RSCRIPT}"
}

# prep: name  src_h5ad  src_pseudotime  obs_key
prep() {
  echo ">>> preparing $1"
  "${PY}" "${PREP}" --src_h5ad "$2" --src_pseudotime "$3" --obs_key "$4" --out_dir "${OUT}/$1"
}

# ---------------------------------------------------------------- GSE242423 --
SRC="${RAW_ROOT}/GSE242423/GSE242423_preprocessed.h5ad"
PTC="${RAW_ROOT}/GSE242423/GSE242423_pseudotime.csv"
pseudotime GSE242423 "${SRC}" "${PTC}" cell_type Fibroblast
prep       GSE242423 "${SRC}" "${PTC}" sample

# ------------------------------------------------- GSE297234 (per donor) -----
for donor in GM00731 GM23815; do
  SRC="${RAW_ROOT}/GSE297234_${donor}/GSE297234_${donor}_preprocessed.h5ad"
  PTC="${RAW_ROOT}/GSE297234_${donor}/GSE297234_${donor}_pseudotime.csv"
  pseudotime "GSE297234_${donor}" "${SRC}" "${PTC}" cell_type Fibroblast
  prep       "GSE297234_${donor}" "${SRC}" "${PTC}" cell_type
done

# ----------------------------------------------------------------- GSE98567 --
SRC="${RAW_ROOT}/GSE98567/GSE98567_preprocessed.h5ad"
PTC="${RAW_ROOT}/GSE98567/GSE98567_pseudotime.csv"
pseudotime GSE98567 "${SRC}" "${PTC}" group Control_uninfected
prep       GSE98567 "${SRC}" "${PTC}" group

# ------------------------------------------------------ GSE181451 (2 sets) ---
SRC="${RAW_ROOT}/GSE181451_MEF_cluster1_iPS/GSE181451_MEF_cluster1_iPS_preprocessed.h5ad"
PTC="${RAW_ROOT}/GSE181451_MEF_cluster1_iPS/GSE181451_MEF_cluster1_iPS_pseudotime.csv"
pseudotime GSE181451_Primed_MEF "${SRC}" "${PTC}" cell_type Fibroblast
prep       GSE181451_Primed_MEF "${SRC}" "${PTC}" sample

SRC="${RAW_ROOT}/GSE181451_MEF_cluster2_iPS/GSE181451_MEF_cluster2_iPS_preprocessed.h5ad"
PTC="${RAW_ROOT}/GSE181451_MEF_cluster2_iPS/GSE181451_MEF_cluster2_iPS_pseudotime.csv"
pseudotime GSE181451_Baseline_MEF "${SRC}" "${PTC}" cell_type Fibroblast
prep       GSE181451_Baseline_MEF "${SRC}" "${PTC}" sample

echo ">>> all datasets prepared under ${OUT}"
