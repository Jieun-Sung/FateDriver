#!/usr/bin/env python
"""
Build the small, self-contained example dataset shipped in the repository
(``data/example``) from the unified GSE98567 dataset.

The example is GSE98567 restricted to its highly-variable genes (plus the driver
genes) with a NicheNet table filtered to the same gene set, so the whole thing
is a few MB and can be committed to git and run out-of-the-box.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "GSE98567"
DST = REPO / "data" / "example"
GS = REPO / "data" / "gene_sets"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(SRC / "anndata.h5ad")
    drivers = set(pd.read_csv(GS / "GMT_driver_genes_mouse.csv").iloc[:, 0].astype(str))
    keep = a.var["highly_variable"].to_numpy(dtype=bool) | np.isin(a.var_names.astype(str), list(drivers))
    sub = a[:, keep].copy()
    sub.write_h5ad(DST / "anndata.h5ad", compression="gzip")

    (pd.read_csv(SRC / "pseudotime.csv")).to_csv(DST / "pseudotime.csv", index=False)
    for name in ["GMT_driver_genes_mouse.csv", "TRRUST_GMT_targets_mouse.csv"]:
        pd.read_csv(GS / name).to_csv(DST / name, index=False)

    genes = set(sub.var_names.astype(str))
    net = pd.read_csv(GS / "nichenet_gr_signaling_net.csv")
    net = net[net["from"].astype(str).isin(genes) & net["to"].astype(str).isin(genes)]
    net.to_csv(DST / "nichenet.csv", index=False)

    print(f"example: cells={sub.n_obs} genes={sub.n_vars} nichenet_edges={len(net)}", flush=True)


if __name__ == "__main__":
    main()
