"""
Self-contained embedding / UMAP export for FateDriver.

Given the final in-memory cell and gene embeddings, this writes:

    <out_dir>/embeddings/gene_embeddings/gene_embedding.npz
    <out_dir>/embeddings/cell_embeddings/cell_embedding.npz
    <out_dir>/tables/gene_umap.csv
    <out_dir>/tables/cell_umap.csv
    <out_dir>/embedding_export_summary.json

It depends only on ``umap-learn`` (+ numpy / pandas) and does not require the
figure-plotting utilities used for the paper figures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _umap(emb: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap  # umap-learn
    n = emb.shape[0]
    reducer = umap.UMAP(n_components=2, n_neighbors=int(min(max(2, n_neighbors), max(2, n - 1))),
                        min_dist=float(min_dist), random_state=int(seed))
    return np.asarray(reducer.fit_transform(emb.astype(np.float32)), dtype=np.float32)


def export(out_dir: Path, gene_names: List[str], obs_names: List[str], sample_col: np.ndarray,
           scores_np: np.ndarray, cell_emb: np.ndarray, gene_emb: np.ndarray,
           source_cell_type: str, target_cell_type: str, driver_idx: List[int],
           seed: int = 42, n_neighbors: int = 30, min_dist: float = 0.35) -> None:
    out_dir = Path(out_dir)
    gene_dir = out_dir / "embeddings" / "gene_embeddings"
    cell_dir = out_dir / "embeddings" / "cell_embeddings"
    tables_dir = out_dir / "tables"
    for d in (gene_dir, cell_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    driver_set = set(int(i) for i in driver_idx)

    # --- gene embedding + UMAP ---------------------------------------
    gene_umap = _umap(gene_emb, seed, n_neighbors, min_dist)
    np.savez_compressed(gene_dir / "gene_embedding.npz",
                        gene_names=np.asarray(gene_names), embedding=gene_emb.astype(np.float32), umap=gene_umap)
    pd.DataFrame({
        "gene": gene_names, "umap1": gene_umap[:, 0], "umap2": gene_umap[:, 1],
        "is_driver": [i in driver_set for i in range(len(gene_names))],
        "score": scores_np.astype(np.float32),
    }).to_csv(tables_dir / "gene_umap.csv", index=False)

    # --- cell embedding + UMAP ---------------------------------------
    sample_arr = np.asarray(sample_col).astype(str)

    def _labels(spec: str) -> set:
        text = str(spec)
        for sep in [";", "|"]:
            text = text.replace(sep, ",")
        return {x.strip() for x in text.split(",") if x.strip()}

    src_labels, tgt_labels = _labels(source_cell_type), _labels(target_cell_type)
    group = np.where(np.isin(sample_arr, list(src_labels)), "source",
                     np.where(np.isin(sample_arr, list(tgt_labels)), "target", "other"))
    cell_umap = _umap(cell_emb, seed, n_neighbors, min_dist)
    np.savez_compressed(cell_dir / "cell_embedding.npz",
                        obs_names=np.asarray(obs_names), embedding=cell_emb.astype(np.float32),
                        umap=cell_umap, condition=sample_arr, group=group)
    pd.DataFrame({
        "obs_name": obs_names, "umap1": cell_umap[:, 0], "umap2": cell_umap[:, 1],
        "condition": sample_arr, "group": group,
    }).to_csv(tables_dir / "cell_umap.csv", index=False)

    summary = {
        "num_genes": int(len(gene_names)), "num_cells": int(len(obs_names)),
        "num_driver_genes": int(len(driver_set)),
        "num_source_cells": int(np.sum(group == "source")), "num_target_cells": int(np.sum(group == "target")),
        "umap_n_neighbors": int(n_neighbors), "umap_min_dist": float(min_dist), "seed": int(seed),
    }
    (out_dir / "embedding_export_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Embedding export complete: {out_dir}", flush=True)
