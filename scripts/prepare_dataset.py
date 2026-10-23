#!/usr/bin/env python
"""
Convert a raw preprocessed AnnData + pseudotime table into the unified
FateDriver input format.

Unified layout produced under ``<out_dir>``::

    anndata.h5ad     # X = model features, layers['counts'] = raw counts,
                     # var['highly_variable'] = HVG flag, obs['condition'] = source/target label
    pseudotime.csv   # columns: cell_id, pseudotime

The output h5ad keeps ONLY what FateDriver needs (X, counts, the HVG flag and
the condition column) so the published files are as small as possible.

Example
-------
    python prepare_dataset.py \
        --src_h5ad       /path/GSE242423_preprocessed.h5ad \
        --src_pseudotime /path/GSE242423_pseudotime.csv \
        --obs_key sample --pseudotime_col pseudotime \
        --out_dir data/GSE242423
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:  # anndata >= 0.10
    from anndata.io import read_elem
except Exception:  # pragma: no cover - older anndata
    from anndata.experimental import read_elem


def _csr(x) -> sp.csr_matrix:
    return x.tocsr().astype(np.float32) if sp.issparse(x) else sp.csr_matrix(np.asarray(x, dtype=np.float32))


def _read_components(path: str, counts_layer: str):
    """Read only X / counts / obs / var, skipping uns/obsm (which may contain
    elements a newer/older anndata cannot decode)."""
    with h5py.File(path, "r") as f:
        X = read_elem(f["X"])
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        if "layers" not in f or counts_layer not in f["layers"]:
            raise KeyError(f"layers['{counts_layer}'] not found. Available: {list(f.get('layers', {}).keys())}")
        counts = read_elem(f["layers"][counts_layer])
    return X, counts, obs, var


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert raw AnnData/pseudotime to unified FateDriver format")
    ap.add_argument("--src_h5ad", required=True)
    ap.add_argument("--src_pseudotime", required=True)
    ap.add_argument("--obs_key", required=True, help="obs column holding the source/target condition label")
    ap.add_argument("--counts_layer", default="counts")
    ap.add_argument("--pseudotime_col", default="pseudotime")
    ap.add_argument("--pseudotime_cell_id_col", default=None, help="Cell-id column in the pseudotime CSV (default: first / index column)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, counts, src_obs, src_var = _read_components(args.src_h5ad, args.counts_layer)
    if "highly_variable" not in src_var.columns:
        raise KeyError("Source AnnData is missing var['highly_variable'].")
    if args.obs_key not in src_obs.columns:
        raise KeyError(f"obs['{args.obs_key}'] not found. Available: {list(src_obs.columns)}")

    obs = pd.DataFrame(index=src_obs.index.astype(str))
    obs["condition"] = src_obs[args.obs_key].astype(str).to_numpy()
    var = pd.DataFrame(index=src_var.index.astype(str))
    var["highly_variable"] = src_var["highly_variable"].astype(bool).to_numpy()

    clean = ad.AnnData(X=_csr(X), obs=obs, var=var, layers={"counts": _csr(counts)})
    clean.write_h5ad(out_dir / "anndata.h5ad", compression="gzip")

    df = pd.read_csv(args.src_pseudotime)
    if args.pseudotime_col not in df.columns:
        raise KeyError(f"pseudotime column '{args.pseudotime_col}' not found. Available: {list(df.columns)}")
    cid = args.pseudotime_cell_id_col
    if cid is None:
        cid = next((c for c in df.columns if str(c).lower().startswith("unnamed")), df.columns[0])
    out = pd.DataFrame({"cell_id": df[cid].astype(str).to_numpy(),
                        "pseudotime": pd.to_numeric(df[args.pseudotime_col], errors="coerce").to_numpy()})
    out = out.dropna(subset=["pseudotime"]).drop_duplicates(subset=["cell_id"])
    out.to_csv(out_dir / "pseudotime.csv", index=False)

    n_cond = obs["condition"].value_counts().to_dict()
    print(f"[{out_dir.name}] cells={clean.n_obs} genes={clean.n_vars} hvg={int(var['highly_variable'].sum())} "
          f"pseudotime_rows={len(out)} conditions={n_cond}", flush=True)


if __name__ == "__main__":
    main()
