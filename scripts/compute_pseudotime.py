#!/usr/bin/env python
"""
Compute a pseudotime ordering for a preprocessed AnnData and write the
``pseudotime.csv`` that FateDriver consumes.

Input
-----
A QC'd / normalised ``.h5ad`` (one dataset, source + target cells together) with

    adata.X                        log-normalised expression
    adata.obs[<group_key>]         cell-type / condition label, e.g. 'cell_type'
    adata.obsm['X_umap']           UMAP embedding        (required for slingshot)
    adata.obs[<cluster_key>]       leiden/louvain labels (optional, slingshot --cluster_by cluster)

Output
------
``pseudotime.csv`` with columns ``cell_id,pseudotime`` — the file expected by
``prepare_dataset.py`` and by ``src/train_fatedriver.py --pseudotime_csv_path``.

Methods
-------
slingshot (default)
    The AnnData is exported to a temporary directory and
    ``scripts/compute_pseudotime.R`` fits Slingshot on the existing UMAP
    embedding, with the cell-state labels as clusters, ``--root_group`` as the
    start cluster and (optionally) ``--end_group`` as the end cluster.
    Unified pseudotime = mean of the per-lineage pseudotimes. Needs the R
    environment (``environment_pseudotime.yml``).
dpt
    Pure-Python fallback (scanpy diffusion pseudotime). No R required, runs in
    the FateDriver conda environment. Values differ from slingshot in scale, but
    FateDriver only uses the pseudotime ordering to orient the cell graph.

Examples
--------
    # slingshot
    python scripts/compute_pseudotime.py \
        --h5ad     /path/GSE297234_GM00731_preprocessed.h5ad \
        --group_key cell_type --root_group Fibroblast --end_group iPSC \
        --out_csv  /path/GSE297234_GM00731_pseudotime.csv

    # scanpy DPT, no R needed
    python scripts/compute_pseudotime.py \
        --h5ad     /path/GSE98567_preprocessed.h5ad \
        --group_key group --root_group Control_uninfected \
        --method dpt --out_csv /path/GSE98567_pseudotime.csv
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

R_SCRIPT = Path(__file__).resolve().parent / "compute_pseudotime.R"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _check_group(adata: ad.AnnData, group_key: str, group: str, role: str = "root") -> np.ndarray:
    if group_key not in adata.obs.columns:
        raise KeyError(f"obs['{group_key}'] not found. Available: {list(adata.obs.columns)}")
    groups = adata.obs[group_key].astype(str).to_numpy()
    mask = groups == group
    if not mask.any():
        raise ValueError(
            f"{role} group '{group}' not present in obs['{group_key}']. "
            f"Available: {sorted(set(groups))}"
        )
    if role == "root":
        print(f"[groups] {pd.Series(groups).value_counts().to_dict()}", flush=True)
    print(f"[{role}]   {mask.sum()} cells in '{group}'", flush=True)
    return mask


def _write_csv(cell_ids: np.ndarray, pseudotime: np.ndarray, out_csv: Path) -> None:
    df = pd.DataFrame({"cell_id": np.asarray(cell_ids, dtype=str), "pseudotime": np.asarray(pseudotime, float)})
    n_all = len(df)
    df = df[np.isfinite(df["pseudotime"].to_numpy())]
    if len(df) < n_all:
        print(f"[write] dropped {n_all - len(df)} cells with non-finite pseudotime "
              f"(not assigned to any lineage)", flush=True)
    if df.empty:
        raise RuntimeError("no cell received a finite pseudotime")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    pt = df["pseudotime"].to_numpy()
    print(f"[write] {out_csv}  cells={len(df)}  range=[{pt.min():.4g}, {pt.max():.4g}]", flush=True)


# --------------------------------------------------------------------------- #
# slingshot (R)
# --------------------------------------------------------------------------- #
def _export_for_r(adata: ad.AnnData, args, work_dir: Path) -> None:
    """Dump metadata / UMAP so the R script needs no h5ad reader."""
    if args.umap_key not in adata.obsm:
        raise KeyError(
            f"obsm['{args.umap_key}'] not found (slingshot fits the trajectory on the UMAP "
            f"embedding). Available: {list(adata.obsm.keys())}. Compute a UMAP first, "
            f"or use --method dpt."
        )

    cells = adata.obs_names.astype(str)
    pd.Series(cells).to_csv(work_dir / "barcodes.tsv", index=False, header=False)

    meta = pd.DataFrame(index=pd.Index(cells, name="cell_id"))
    meta["group"] = adata.obs[args.group_key].astype(str).to_numpy()
    if args.cluster_key and args.cluster_key in adata.obs.columns:
        meta["cluster"] = adata.obs[args.cluster_key].astype(str).to_numpy()
    else:
        if args.cluster_by == "cluster":
            print(f"[export] obs['{args.cluster_key}'] not found, using the group label as cluster",
                  flush=True)
        meta["cluster"] = meta["group"]
    meta.to_csv(work_dir / "cell_meta.csv")

    umap = np.asarray(adata.obsm[args.umap_key])[:, :2]
    pd.DataFrame(umap, index=pd.Index(cells, name="cell_id"), columns=["UMAP_1", "UMAP_2"]).to_csv(
        work_dir / "umap.csv")
    print(f"[export] {len(cells)} cells -> {work_dir}", flush=True)


def _run_slingshot(adata: ad.AnnData, args, out_csv: Path) -> None:
    rscript = shutil.which(args.rscript)
    if rscript is None:
        raise RuntimeError(
            f"'{args.rscript}' not found on PATH. Create the R environment with\n"
            f"    conda env create -f environment_pseudotime.yml\n"
            f"    conda activate FateDriver-pseudotime\n"
            f"or use --method dpt (pure Python)."
        )
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="fatedriver_pt_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        _export_for_r(adata, args, work_dir)
        cmd = [rscript, str(R_SCRIPT),
               "--work_dir", str(work_dir),
               "--out_csv", str(out_csv),
               "--root_group", args.root_group,
               "--cluster_by", args.cluster_by,
               "--seed", str(args.seed)]
        if args.end_group:
            cmd += ["--end_group", args.end_group]
        if args.plot_prefix:
            cmd += ["--plot_prefix", str(args.plot_prefix)]
        print("[slingshot] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    finally:
        if not args.keep_work_dir and not args.work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    # R also writes pseudotime_lineage* columns; FateDriver only needs cell_id,pseudotime.
    df = pd.read_csv(out_csv)
    _write_csv(df["cell_id"].to_numpy(), df["pseudotime"].to_numpy(), out_csv)


# --------------------------------------------------------------------------- #
# scanpy DPT
# --------------------------------------------------------------------------- #
def _run_dpt(adata: ad.AnnData, args, out_csv: Path, root_mask: np.ndarray) -> None:
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("--method dpt needs scanpy (`pip install scanpy`, or use "
                           "the pinned environment.yml)") from exc

    sc.settings.verbosity = 1
    if "X_pca" not in adata.obsm:
        print("[dpt] no obsm['X_pca'], running PCA", flush=True)
        sc.pp.pca(adata, n_comps=min(args.num_dim, min(adata.shape) - 1), random_state=args.seed)
    if "neighbors" not in adata.uns or args.recompute_neighbors:
        print(f"[dpt] computing neighbors (n_neighbors={args.n_neighbors})", flush=True)
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep="X_pca", random_state=args.seed)
    sc.tl.diffmap(adata, n_comps=args.n_dcs, random_state=args.seed)

    # Root cell: the cell of the root group that lies furthest (in diffusion
    # space) from the centre of all other cells — the tip of the root branch.
    dm = np.asarray(adata.obsm["X_diffmap"])[:, 1:]
    if (~root_mask).any():
        centre = dm[~root_mask].mean(axis=0)
    else:
        centre = dm.mean(axis=0)
    root_idx = np.flatnonzero(root_mask)
    iroot = int(root_idx[np.argmax(((dm[root_mask] - centre) ** 2).sum(axis=1))])
    adata.uns["iroot"] = iroot
    print(f"[dpt] root cell: {adata.obs_names[iroot]}", flush=True)

    sc.tl.dpt(adata, n_dcs=args.n_dcs)
    _write_csv(adata.obs_names.to_numpy(), adata.obs["dpt_pseudotime"].to_numpy(), out_csv)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="AnnData -> pseudotime.csv (slingshot trajectory or scanpy DPT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--h5ad", required=True, help="preprocessed AnnData (.h5ad)")
    ap.add_argument("--out_csv", required=True, help="output pseudotime.csv")
    ap.add_argument("--method", choices=("slingshot", "dpt"), default="slingshot")
    ap.add_argument("--group_key", default="cell_type", help="obs column holding the cell-state label")
    ap.add_argument("--root_group", required=True,
                    help="value of --group_key that starts the trajectory (e.g. Fibroblast, MEF, D0)")
    ap.add_argument("--end_group", default=None,
                    help="value of --group_key that ends the trajectory, e.g. iPSC "
                         "(slingshot; omit to let slingshot detect end points)")
    ap.add_argument("--cluster_by", choices=("group", "cluster"), default="group",
                    help="clusterLabels for slingshot: the --group_key label or the --cluster_key clusters")
    ap.add_argument("--cluster_key", default="leiden",
                    help="obs column with clusters (slingshot --cluster_by cluster; falls back to --group_key)")
    ap.add_argument("--umap_key", default="X_umap", help="obsm key of the UMAP embedding (slingshot)")
    ap.add_argument("--num_dim", type=int, default=30, help="PCA dimensions if X_pca is missing (dpt)")
    ap.add_argument("--n_neighbors", type=int, default=15, help="kNN size (dpt)")
    ap.add_argument("--n_dcs", type=int, default=15, help="diffusion components (dpt)")
    ap.add_argument("--recompute_neighbors", action="store_true",
                    help="recompute the kNN graph even if the AnnData already has one (dpt)")
    ap.add_argument("--rscript", default="Rscript", help="Rscript executable (slingshot)")
    ap.add_argument("--work_dir", default=None,
                    help="directory for the R exchange files (default: a temporary directory)")
    ap.add_argument("--keep_work_dir", action="store_true", help="do not delete the exchange files")
    ap.add_argument("--plot_prefix", default=None,
                    help="write trajectory/pseudotime PNGs with this path prefix (slingshot)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_csv = Path(args.out_csv)

    print(f"[read] {args.h5ad}", flush=True)
    adata = ad.read_h5ad(args.h5ad)
    print(f"[read] {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    root_mask = _check_group(adata, args.group_key, args.root_group)
    if args.end_group:
        _check_group(adata, args.group_key, args.end_group, role="end")

    if args.method == "slingshot":
        _run_slingshot(adata, args, out_csv)
    else:
        _run_dpt(adata, args, out_csv, root_mask)


if __name__ == "__main__":
    sys.exit(main())
