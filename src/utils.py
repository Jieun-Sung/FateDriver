"""
FateDriver utilities: data loading, graph construction, preprocessing, cell
feature extraction, negative sampling, ranking / reconstruction losses, the
learnable-prior features, dataset splits, evaluation metrics and small IO
helpers.

Everything here is configuration-agnostic and stateless; the training loop in
``train_fatedriver.py`` wires these together.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import torch
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score

from model import (
    SparseGraphData,
    apply_decoder_activation,
    decode_recon_logits,
    masked_mean,
    masked_mse_loss,
    module_device,
    nb_elementwise_nll,
    zinb_elementwise_nll,
    zinb_elementwise_nll as _zinb,  # noqa: F401 (kept for symmetry)
)


# =====================================================================
# AnnData / pseudotime loaders
# =====================================================================

def _as_csr_float32(X: object) -> sp.csr_matrix:
    if hasattr(X, "to_memory"):
        X = X.to_memory()
    if sp.issparse(X):
        return X.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(X), dtype=np.float32)


def _keep_first_unique_mask(ids: np.ndarray) -> np.ndarray:
    s = pd.Index(np.asarray(ids).astype(str))
    return ~s.duplicated(keep="first")


def _deduplicate_obs_names(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids).astype(str)
    keep_mask = _keep_first_unique_mask(ids)
    return ids[keep_mask], keep_mask


def load_anndata_as_sparse(path: str, set_HVG: bool = False, layer: Optional[str] = None,
                           force_include_genes: Optional[Sequence[str]] = None) -> Tuple[sp.csr_matrix, List[str]]:
    """Load a cell x gene matrix.

    * ``layer`` = None / 'X' uses ``adata.X``; otherwise ``adata.layers[layer]``.
    * When ``set_HVG`` is True the columns are restricted to
      ``adata.var['highly_variable']`` OR any gene in ``force_include_genes``.
    """
    adata = ad.read_h5ad(path, backed="r")
    requested_layer = None if layer in (None, "", "X", "adata.X") else str(layer)
    if requested_layer is None:
        X_csr = _as_csr_float32(adata.X)
    elif requested_layer in adata.layers:
        X_csr = _as_csr_float32(adata.layers[requested_layer])
    else:
        available = list(adata.layers.keys())
        adata.file.close()
        raise KeyError(f"layer='{requested_layer}' not found. Available layers: {available}")

    gene_names = np.asarray(adata.var_names).astype(str)
    force_set = {str(g) for g in ([] if force_include_genes is None else force_include_genes)
                 if str(g) not in {"", "nan", "None"}}

    keep_mask = np.ones(X_csr.shape[1], dtype=bool)
    if set_HVG:
        if "highly_variable" not in adata.var.columns:
            adata.file.close()
            raise KeyError("adata.var['highly_variable'] not found, but set_HVG=True.")
        hvg_mask = np.asarray(adata.var["highly_variable"]).astype(bool)
        if hvg_mask.shape[0] != X_csr.shape[1]:
            adata.file.close()
            raise ValueError(f"HVG mask length ({hvg_mask.shape[0]}) != n_genes ({X_csr.shape[1]})")
        force_mask = np.asarray([str(g) in force_set for g in gene_names], dtype=bool)
        keep_mask = np.logical_or(hvg_mask, force_mask)
        if keep_mask.sum() == 0:
            adata.file.close()
            raise ValueError("No genes retained after HVG + force-include filtering.")
    if not np.all(keep_mask):
        X_csr = X_csr[:, keep_mask]
        gene_names = gene_names[keep_mask]
    adata.file.close()
    return X_csr.astype(np.float32).tocsr(), gene_names.tolist()


def load_obs_names(path: str) -> np.ndarray:
    adata_meta = ad.read_h5ad(path, backed="r")
    obs_names = np.asarray(adata_meta.obs_names).astype(str)
    adata_meta.file.close()
    return obs_names


def load_obs_column(path: str, key: str) -> np.ndarray:
    adata_meta = ad.read_h5ad(path, backed="r")
    if key not in adata_meta.obs:
        available = list(adata_meta.obs.columns)
        adata_meta.file.close()
        raise KeyError(f"obs key='{key}' not found. Available: {available[:30]} ...")
    arr = np.asarray(adata_meta.obs[key]).astype(str)
    adata_meta.file.close()
    return arr


def parse_cell_type_values(spec: object) -> List[str]:
    if spec is None:
        return []
    text = str(spec)
    for sep in [";", "|"]:
        text = text.replace(sep, ",")
    return [x.strip() for x in text.split(",") if x.strip()]


def select_cells_by_type(sample_col: np.ndarray, spec: object) -> List[int]:
    values = parse_cell_type_values(spec)
    if len(values) == 0:
        return []
    sample_arr = np.asarray(sample_col).astype(str)
    return np.where(np.isin(sample_arr, np.asarray(values).astype(str)))[0].astype(np.int64).tolist()


def _infer_cell_id_col(df: pd.DataFrame, explicit: str = "auto") -> str:
    if explicit not in (None, "", "auto"):
        if explicit not in df.columns:
            raise KeyError(f"cell id column='{explicit}' not found. Available: {df.columns.tolist()}")
        return explicit
    lower = {str(c).lower(): c for c in df.columns}
    for cand in ["cell_id", "cell", "barcode", "obs_name", "obs_names", "cellid"]:
        if cand in lower:
            return lower[cand]
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        return unnamed[0]
    first = df.columns[0]
    if pd.api.types.is_object_dtype(df[first]) or pd.api.types.is_string_dtype(df[first]):
        return first
    raise KeyError("Could not infer the cell id column; set --pseudotime_cell_id_col explicitly.")


def _candidate_cell_id_cols(df: pd.DataFrame, explicit: str = "auto") -> List[str]:
    if explicit not in (None, "", "auto"):
        return [_infer_cell_id_col(df, explicit=explicit)]
    ordered: List[str] = []
    lower = {str(c).lower(): c for c in df.columns}
    for cand in ["cell_id", "cell", "barcode", "obs_name", "obs_names", "cellid"]:
        if cand in lower:
            ordered.append(lower[cand])
    ordered.extend([c for c in df.columns if str(c).lower().startswith("unnamed")])
    ordered.extend([c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])])
    ordered.extend(df.columns.tolist())
    seen, out = set(), []
    for col in ordered:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _choose_best_cell_id_col(df: pd.DataFrame, key: str, obs_names: np.ndarray, explicit: str = "auto") -> str:
    obs_set = set(np.asarray(obs_names).astype(str).tolist())
    best_col, best_matches = None, -1
    for cand in _candidate_cell_id_cols(df, explicit=explicit):
        if str(cand) == str(key):
            continue
        tmp = df[[cand, key]].copy()
        tmp[cand] = tmp[cand].astype(str).str.strip()
        tmp[key] = pd.to_numeric(tmp[key], errors="coerce")
        tmp = tmp.dropna(subset=[cand, key])
        if tmp.empty:
            continue
        tmp = tmp.loc[~tmp[cand].duplicated(keep="first")]
        match_count = int(tmp[cand].isin(obs_set).sum())
        if match_count > best_matches:
            best_matches, best_col = match_count, cand
    return str(best_col) if best_col is not None else _infer_cell_id_col(df, explicit=explicit)


def load_pseudotime(path: str, key: str, pseudotime_csv_path: Optional[str] = None,
                    cell_id_col: str = "auto", obs_names: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if obs_names is None:
        obs_names = load_obs_names(path)
    obs_names = np.asarray(obs_names).astype(str)
    if pseudotime_csv_path in (None, ""):
        adata = ad.read_h5ad(path, backed="r")
        if key not in adata.obs:
            available = list(adata.obs.columns)
            adata.file.close()
            raise KeyError(f"obs key='{key}' not found. Available: {available[:30]} ...")
        pt = np.asarray(adata.obs[key]).astype(np.float32)
        adata.file.close()
        if pt.shape[0] != obs_names.shape[0]:
            raw_obs_names = load_obs_names(path)
            pt = pt[_keep_first_unique_mask(raw_obs_names)]
        valid_mask = np.isfinite(pt)
        return pt[valid_mask], valid_mask
    if not os.path.exists(pseudotime_csv_path):
        raise FileNotFoundError(f"pseudotime CSV not found: {pseudotime_csv_path}")
    df = pd.read_csv(pseudotime_csv_path)
    if key not in df.columns:
        raise KeyError(f"pseudotime column='{key}' not found in CSV. Available: {df.columns.tolist()}")
    cid_col = _choose_best_cell_id_col(df, key=key, obs_names=obs_names, explicit=cell_id_col)
    tmp = df[[cid_col, key]].copy()
    tmp[cid_col] = tmp[cid_col].astype(str).str.strip()
    tmp[key] = pd.to_numeric(tmp[key], errors="coerce")
    tmp = tmp.dropna(subset=[cid_col, key])
    tmp = tmp.loc[~tmp[cid_col].duplicated(keep="first")]
    aligned = tmp.set_index(cid_col)[key].reindex(obs_names)
    keep_mask = aligned.notna().to_numpy()
    pt = aligned[keep_mask].to_numpy(dtype=np.float32, copy=True)
    if np.any(~np.isfinite(pt)):
        raise ValueError("pseudotime CSV contains non-finite values after alignment")
    return pt, keep_mask


# =====================================================================
# Sparse graph construction
# =====================================================================

def build_knn_edges(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xn = X.astype(np.float32, copy=False)
    n = Xn.shape[0]
    src, dst, dist = [], [], []
    X2 = (Xn * Xn).sum(axis=1, keepdims=True)
    for i in range(n):
        di = (X2[i] + X2.T - 2.0 * (Xn[i:i + 1] @ Xn.T)).reshape(-1)
        di[i] = np.inf
        nn = np.argpartition(di, kth=min(k, n - 1) - 1)[:min(k, n - 1)]
        for j in nn:
            src.append(i)
            dst.append(int(j))
            dist.append(float(di[j]))
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), np.asarray(dist, dtype=np.float32)


def _empty_edge_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32))


def coalesce_edges_numpy(num_nodes: int, src: np.ndarray, dst: np.ndarray, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if src.size == 0:
        return _empty_edge_arrays()
    mat = sp.coo_matrix((weight, (src, dst)), shape=(num_nodes, num_nodes), dtype=np.float32)
    mat.sum_duplicates()
    mat = mat.tocoo()
    return mat.row.astype(np.int64), mat.col.astype(np.int64), mat.data.astype(np.float32)


def make_sparse_graph(num_nodes: int, src: np.ndarray, dst: np.ndarray, weight: np.ndarray,
                      device: torch.device, name: str) -> SparseGraphData:
    src = np.asarray(src, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst, dtype=np.int64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    empty_long = torch.empty((0,), dtype=torch.long, device=device)
    empty_float = torch.empty((0,), dtype=torch.float32, device=device)
    if src.size == 0:
        return SparseGraphData(num_nodes, empty_long, empty_long, empty_float, name, None)
    keep = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes) & np.isfinite(weight)
    src, dst, weight = src[keep], dst[keep], weight[keep]
    if src.size == 0:
        return SparseGraphData(num_nodes, empty_long, empty_long, empty_float, name, None)
    weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    src, dst, weight = coalesce_edges_numpy(num_nodes, src, dst, weight)
    if src.size == 0:
        return SparseGraphData(num_nodes, empty_long, empty_long, empty_float, name, None)
    order = np.lexsort((src, dst))
    src, dst, weight = src[order], dst[order], weight[order]
    return SparseGraphData(
        num_nodes,
        torch.from_numpy(src).to(device=device, dtype=torch.long),
        torch.from_numpy(dst).to(device=device, dtype=torch.long),
        torch.from_numpy(weight).to(device=device, dtype=torch.float32),
        name, None,
    )


def build_trajectory_edges_soft(knn_src: np.ndarray, knn_dst: np.ndarray, pseudotime: np.ndarray,
                                min_delta_t: float, sigma_override: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Directed edges i -> j when pseudotime increases, weighted by a Gaussian of the delta."""
    pt = np.asarray(pseudotime, dtype=np.float32)
    src = np.asarray(knn_src, dtype=np.int64)
    dst = np.asarray(knn_dst, dtype=np.int64)
    if src.size == 0:
        return _empty_edge_arrays() + (0.0,)
    delta = pt[dst] - pt[src]
    forward = delta > float(min_delta_t)
    fs, fd, fdelta = src[forward], dst[forward], delta[forward]
    if fdelta.size == 0:
        return _empty_edge_arrays() + (0.0,)
    sigma = float(sigma_override) if sigma_override > 0 else float(np.median(np.abs(fdelta)) + 1e-6)
    w = np.exp(-(fdelta ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    return fs.astype(np.int64), fd.astype(np.int64), w, sigma


def select_hvg_by_mean(X_log: sp.csr_matrix, max_hvg: int) -> List[int]:
    means = np.asarray(X_log.mean(axis=0)).reshape(-1)
    keep = np.argsort(-means)[:max(1, int(max_hvg))]
    return sorted(keep.tolist())


def _resolve_nichenet_id_columns(nichenet_df, gene_to_idx):
    """Pick the (source, target) column pair that actually resolves.

    NicheNet tables ship both symbol columns (from/to) and Ensembl columns
    (from_ens/to_ens). Which pair is usable depends on how ``adata.var_names``
    are keyed, so select by the number of rows that resolve rather than by a
    fixed precedence: a fixed order silently produced an empty gene graph when
    a symbol-keyed AnnData met an Ensembl-carrying table.

    Returns (n_resolved, src_col, dst_col, report) where ``report`` lists the
    match count of every candidate pair, for error messages.
    """
    cols_lower = {str(c).lower(): c for c in nichenet_df.columns}
    src_cands = [cols_lower[c] for c in ("from", "from_ens", "ligand", "source", "tf", "regulator") if c in cols_lower]
    dst_cands = [cols_lower[c] for c in ("to", "to_ens", "target", "gene") if c in cols_lower]
    if not src_cands:
        src_cands = [nichenet_df.columns[0]]
    if not dst_cands:
        dst_cands = [nichenet_df.columns[1]]
    best = (-1, src_cands[0], dst_cands[0])
    report = []
    for s in src_cands:
        s_ok = nichenet_df[s].map(gene_to_idx).notna()
        for d in dst_cands:
            n = int((s_ok & nichenet_df[d].map(gene_to_idx).notna()).sum())
            report.append(f"{s}/{d}: {n}")
            if n > best[0]:
                best = (n, s, d)
    return best[0], best[1], best[2], report


def build_gene_nichenet_graph(nichenet_df: pd.DataFrame, gene_id_list: List[str],
                              hvg_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed regulatory edges (regulator -> target) from a NicheNet signaling table."""
    gene_to_idx = pd.Series(np.arange(len(gene_id_list), dtype=np.int64), index=pd.Index(gene_id_list))
    n_resolved, src_col, dst_col, _match_report = _resolve_nichenet_id_columns(nichenet_df, gene_to_idx)
    if n_resolved <= 0:
        raise ValueError(
            "The gene graph was requested but no NicheNet row resolves against the gene "
            "identifiers in use. Candidate column pairs and their match counts: "
            + "; ".join(_match_report)
            + ". Check that the table's identifier type matches adata.var_names."
        )
    cols_lower = {str(c).lower(): c for c in nichenet_df.columns}
    weight_col = next((cols_lower[c] for c in ["weight", "score", "prob", "confidence"] if c in cols_lower), None)
    src_idx = nichenet_df[src_col].map(gene_to_idx)
    dst_idx = nichenet_df[dst_col].map(gene_to_idx)
    valid = src_idx.notna() & dst_idx.notna()
    if not bool(valid.any()):
        return _empty_edge_arrays()
    src_idx = src_idx[valid].astype(np.int64)
    dst_idx = dst_idx[valid].astype(np.int64)
    hvg_mask = np.zeros(len(gene_id_list), dtype=bool)
    hvg_arr = np.asarray(hvg_indices, dtype=np.int64)
    hvg_arr = hvg_arr[(hvg_arr >= 0) & (hvg_arr < len(gene_id_list))]
    hvg_mask[hvg_arr] = True
    keep = hvg_mask[src_idx.to_numpy()] | hvg_mask[dst_idx.to_numpy()]
    if not bool(np.any(keep)):
        raise ValueError(
            "The gene graph was requested but no NicheNet edge touches the selected "
            f"gene set ({n_resolved} rows resolved, 0 within the gene universe)."
        )
    src = src_idx.to_numpy()[keep]
    dst = dst_idx.to_numpy()[keep]
    if weight_col is not None:
        weights_all = pd.to_numeric(nichenet_df.loc[valid, weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        weight = weights_all[keep]
    else:
        weight = np.ones(src.shape[0], dtype=np.float32)
    return src, dst, weight


def build_cell_graph(cell_features: np.ndarray, n_cells: int, pseudotime: np.ndarray,
                     knn_k: int, min_delta_t: float, trajectory_sigma: float,
                     device: torch.device, out_dir: Path) -> SparseGraphData:
    """Cell graph = pseudotime trajectory edges over the kNN neighbourhood."""
    if pseudotime is None:
        raise ValueError("Pseudotime is required to build the cell trajectory graph.")
    knn_src, knn_dst, _ = build_knn_edges(cell_features, knn_k)
    traj_src, traj_dst, traj_w, traj_sigma = build_trajectory_edges_soft(
        knn_src=knn_src, knn_dst=knn_dst, pseudotime=pseudotime,
        min_delta_t=min_delta_t, sigma_override=trajectory_sigma,
    )
    graph = make_sparse_graph(n_cells, traj_src, traj_dst, traj_w, device=device, name="cell")
    with open(out_dir / "cell_graph_stats.json", "w") as f:
        json.dump({"n_cells": int(n_cells), "trajectory_edges": int(len(traj_src)),
                   "trajectory_sigma": float(traj_sigma), "total_edges": int(graph.num_edges)}, f, indent=2)
    return graph


def build_gene_graph(nichenet_df_path: str, X_log: sp.csr_matrix, gene_names: List[str], n_genes: int,
                     max_hvg: int, device: torch.device, out_dir: Path) -> Tuple[SparseGraphData, Dict[str, int]]:
    """Gene graph = directed NicheNet regulatory edges (restricted to top-mean genes)."""
    gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    hvg_indices = select_hvg_by_mean(X_log, max_hvg)
    if nichenet_df_path is None or not os.path.exists(nichenet_df_path):
        raise FileNotFoundError(f"NicheNet CSV not found: {nichenet_df_path}")
    nichenet_df = pd.read_csv(nichenet_df_path)
    src_raw, dst_raw, w_raw = build_gene_nichenet_graph(nichenet_df, gene_id_list=gene_names, hvg_indices=hvg_indices)
    w_raw = np.maximum(np.asarray(w_raw, dtype=np.float32), 1e-6)
    if w_raw.size > 0 and float(np.max(w_raw)) > 0.0:
        w_raw = w_raw / float(np.max(w_raw))
    # Reverse direction so messages flow regulator -> target under the encoder's
    # dst-aggregation convention (matches the original implementation).
    gene_graph = make_sparse_graph(n_genes, dst_raw, src_raw, w_raw, device=device, name="gene")
    with open(out_dir / "graph_stats.json", "w") as f:
        json.dump({"n_genes": int(n_genes), "nichenet_edges": int(len(src_raw)),
                   "total_edges": int(gene_graph.num_edges)}, f, indent=2)
    return gene_graph, gene_name_to_idx


# =====================================================================
# Gene-set loaders
# =====================================================================

def _load_gene_names_from_csv(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()
    lower_cols = [c.lower() for c in df.columns]
    preferred = None
    for cand in ["gene", "gene_id", "geneids", "symbol", "name"]:
        if cand in lower_cols:
            preferred = df.columns[lower_cols.index(cand)]
            break
    col = preferred if preferred is not None else df.columns[-1]
    return df[col].astype(str).tolist()


def _load_gene_list_any(path: Optional[str]) -> List[str]:
    if path in (None, ""):
        return []
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Gene list not found: {p}")
    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        if df.shape[1] == 1:
            return df.iloc[:, 0].dropna().astype(str).tolist()
        lower_cols = [str(c).lower() for c in df.columns]
        for cand in ["gene", "gene_id", "geneids", "symbol", "name", "gene_name"]:
            if cand in lower_cols:
                return df[df.columns[lower_cols.index(cand)]].dropna().astype(str).tolist()
        return df.iloc[:, 0].dropna().astype(str).tolist()
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")]


def load_gene_indices_from_csv(csv_path: Optional[str], gene_name_to_idx: Dict[str, int],
                               label_name: str, required: bool = False) -> List[int]:
    if csv_path in (None, ""):
        if required:
            raise ValueError(f"{label_name} CSV path is required.")
        return []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{label_name} CSV not found: {csv_path}")
    gene_names = _load_gene_names_from_csv(csv_path)
    gene_idx = sorted(set(gene_name_to_idx[g] for g in gene_names if g in gene_name_to_idx))
    if required and len(gene_idx) == 0:
        raise ValueError(f"No {label_name} genes matched var names. Check identifier types (symbols vs ENS IDs).")
    return gene_idx


def count_unique_gene_names_in_csv(csv_path: Optional[str]) -> int:
    if csv_path in (None, ""):
        return 0
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Gene CSV not found: {csv_path}")
    return len(set(_load_gene_names_from_csv(csv_path)))


def collect_force_include_genes(driver_genes_path: str, set_HVG: bool, force_prior_genes: bool) -> List[str]:
    """Force-retain intervention (driver) prior genes through HVG filtering."""
    genes: List[str] = []
    if force_prior_genes and set_HVG:
        genes.extend(_load_gene_names_from_csv(driver_genes_path))
    seen, out = set(), []
    for g in genes:
        gs = str(g)
        if gs not in seen and gs not in {"", "nan", "None"}:
            seen.add(gs)
            out.append(gs)
    return out


# =====================================================================
# Learnable-prior transition features
# =====================================================================

def _safe_sparse_mean_frac(mat: sp.csr_matrix, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if rows.size == 0:
        n_genes = mat.shape[1]
        return np.zeros((n_genes,), dtype=np.float32), np.zeros((n_genes,), dtype=np.float32)
    sub = mat[rows]
    mean = np.asarray(sub.mean(axis=0)).reshape(-1).astype(np.float32)
    frac = np.asarray((sub > 0).mean(axis=0)).reshape(-1).astype(np.float32)
    return mean, frac


def _rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    out = np.empty_like(x, dtype=np.float32)
    if x.size == 1:
        out[order[0]] = 1.0
        return out
    out[order] = np.linspace(0.0, 1.0, num=x.size, endpoint=True, dtype=np.float32)
    return out


def compute_gene_transition_driver_prior_features(X_sparse: sp.csr_matrix, source_cells: List[int], target_cells: List[int],
                                                  gene_graph: Optional[SparseGraphData] = None,
                                                  pseudotime: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """Five interpretable per-gene features driving the learnable transition prior."""
    src_rows = np.asarray(source_cells, dtype=np.int64)
    tgt_rows = np.asarray(target_cells, dtype=np.int64)
    src_mean, src_frac = _safe_sparse_mean_frac(X_sparse, src_rows)
    tgt_mean, tgt_frac = _safe_sparse_mean_frac(X_sparse, tgt_rows)
    diff = tgt_mean - src_mean
    frac_diff = tgt_frac - src_frac
    pos_diff_rank = _rank01(np.maximum(diff, 0.0))
    frac_rank = _rank01(np.maximum(frac_diff, 0.0))
    abs_rank = _rank01(np.abs(diff))
    if gene_graph is not None and gene_graph.num_edges > 0:
        outdeg = torch.bincount(gene_graph.src.detach().cpu(), minlength=X_sparse.shape[1]).numpy().astype(np.float32)
        outdeg_rank = _rank01(outdeg)
    else:
        outdeg_rank = np.zeros((X_sparse.shape[1],), dtype=np.float32)
    if pseudotime is not None and len(pseudotime) == X_sparse.shape[0]:
        pt = np.asarray(pseudotime, dtype=np.float32)
        pt = (pt - pt.mean()) / float(pt.std() + 1e-6)
        expr = X_sparse.astype(np.float32)
        cov = np.asarray(expr.T @ pt).reshape(-1) / max(1, expr.shape[0] - 1)
        expr_std = np.sqrt(np.asarray(expr.power(2).mean(axis=0)).reshape(-1) - np.square(np.asarray(expr.mean(axis=0)).reshape(-1))) + 1e-6
        pt_rank = _rank01(np.maximum(cov / expr_std, 0.0))
    else:
        pt_rank = np.zeros((X_sparse.shape[1],), dtype=np.float32)
    features = np.stack([pos_diff_rank, frac_rank, abs_rank, outdeg_rank, pt_rank], axis=1).astype(np.float32, copy=False)
    feature_names = ["pos_diff_rank", "frac_diff_rank", "abs_diff_rank", "out_degree_rank", "pseudotime_corr_rank"]
    return features, feature_names


# =====================================================================
# Expression preprocessing and cell features
# =====================================================================

def load_expression_and_meta(args) -> Dict[str, object]:
    force_include_genes = collect_force_include_genes(
        args.driver_genes_path, args.set_HVG, bool(getattr(args, "force_include_prior_genes_in_hvg", True)))
    X_sparse, gene_names = load_anndata_as_sparse(
        args.anndata_path, set_HVG=args.set_HVG, layer=args.feature_layer, force_include_genes=force_include_genes)
    X_counts_sparse = None
    if args.recon_loss in {"nb", "zinb"}:
        X_counts_sparse, gene_names_counts = load_anndata_as_sparse(
            args.anndata_path, set_HVG=args.set_HVG, layer=args.counts_layer, force_include_genes=force_include_genes)
        if gene_names_counts != gene_names:
            raise ValueError("Gene names from feature layer and counts layer differ after filtering.")
    raw_obs_names = load_obs_names(args.anndata_path)
    obs_names, dedup_mask = _deduplicate_obs_names(raw_obs_names)
    if dedup_mask.shape[0] != X_sparse.shape[0]:
        raise ValueError(f"obs_names length ({dedup_mask.shape[0]}) != X rows ({X_sparse.shape[0]})")
    X_sparse = X_sparse[dedup_mask]
    if X_counts_sparse is not None:
        X_counts_sparse = X_counts_sparse[dedup_mask]
    sample_col = load_obs_column(args.anndata_path, args.obs_column)[dedup_mask]
    pseudotime, pt_keep_mask = load_pseudotime(
        path=args.anndata_path, key=args.pseudotime_key, pseudotime_csv_path=args.pseudotime_csv_path,
        cell_id_col=args.pseudotime_cell_id_col, obs_names=obs_names)
    X_sparse = X_sparse[pt_keep_mask]
    if X_counts_sparse is not None:
        X_counts_sparse = X_counts_sparse[pt_keep_mask]
    sample_col = sample_col[pt_keep_mask]
    obs_names = obs_names[pt_keep_mask]
    n_cells, n_genes = X_sparse.shape
    if n_cells == 0:
        raise ValueError(f"No cells remain after pseudotime alignment. dataset={args.dataset}")
    if sample_col.shape[0] != n_cells or pseudotime.shape[0] != n_cells:
        raise ValueError("Cell-count mismatch after alignment.")
    return {
        "X_sparse": X_sparse, "X_counts_sparse": X_counts_sparse, "gene_names": gene_names,
        "n_cells": n_cells, "n_genes": n_genes, "pseudotime": pseudotime,
        "sample_col": sample_col, "obs_names": obs_names,
        "force_include_gene_count_requested": int(len(force_include_genes)),
    }


def preprocess_expression(args, X_sparse: sp.csr_matrix, device: torch.device,
                          X_counts_sparse: Optional[sp.csr_matrix] = None) -> Dict[str, object]:
    X_feature = X_sparse.copy().astype(np.float32).tocsr()
    if args.recon_loss == "mse":
        X_recon = X_feature
        cell_log_sf = np.zeros((X_feature.shape[0],), dtype=np.float32)
        gene_mean = np.asarray(X_feature.mean(axis=0)).reshape(-1).astype(np.float32)
    elif args.recon_loss in {"nb", "zinb"}:
        if X_counts_sparse is None:
            raise ValueError(f"recon_loss='{args.recon_loss}' requires raw counts in layers['{args.counts_layer}'].")
        X_recon = X_counts_sparse.copy().astype(np.float32).tocsr()
        cell_sum = np.asarray(X_recon.sum(axis=1)).reshape(-1).astype(np.float32)
        target_sum = float(getattr(args, "target_sum", 1e4))
        scale = np.divide(target_sum, np.maximum(cell_sum, 1.0), out=np.zeros_like(cell_sum), where=cell_sum > 0)
        X_norm = sp.diags(scale).dot(X_recon).tocsr()
        if X_norm.data.size > 0:
            X_norm.data = np.log1p(X_norm.data)
        cell_log_sf = np.log(np.maximum(cell_sum, 1.0)).astype(np.float32)
        gene_mean = np.asarray(X_norm.mean(axis=0)).reshape(-1).astype(np.float32)
    else:
        raise ValueError(f"Unknown recon_loss: {args.recon_loss}")
    cell_log_sf_t = torch.from_numpy(cell_log_sf).to(device=device, dtype=torch.float32)
    return {"X_feature": X_feature, "X_recon": X_recon, "cell_log_sf_t": cell_log_sf_t, "gene_bias_init": gene_mean}


def infer_svd_fit_indices(args, sample_col: np.ndarray) -> Optional[np.ndarray]:
    """Training-cell indices for fitting SVD without test-label leakage."""
    source_cells = select_cells_by_type(sample_col, args.source_cell_type)
    target_cells = select_cells_by_type(sample_col, args.target_cell_type)
    if len(source_cells) == 0 or len(target_cells) == 0:
        return None
    rng_split = random.Random(int(args.seed))
    rng_split.shuffle(source_cells)
    rng_split.shuffle(target_cells)

    def _train_cells(idxs: List[int]) -> List[int]:
        n = len(idxs)
        n_test = int(float(args.test_frac) * n)
        n_val = int(float(args.val_frac) * n)
        return idxs[n_test + n_val:]

    fit_idx = sorted(set(_train_cells(source_cells)) | set(_train_cells(target_cells)))
    if len(fit_idx) == 0:
        fit_idx = sorted(set(source_cells) | set(target_cells))
    return np.asarray(fit_idx, dtype=np.int64)


def _valid_fit_indices(fit_indices: Optional[np.ndarray], n_cells: int) -> np.ndarray:
    fit_idx = np.arange(n_cells, dtype=np.int64) if fit_indices is None or len(fit_indices) == 0 else np.asarray(fit_indices, dtype=np.int64)
    fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < n_cells)]
    return fit_idx if fit_idx.size > 0 else np.arange(n_cells, dtype=np.int64)


def _standardize_cell_features(cell_features: np.ndarray, fit_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_features = np.asarray(cell_features, dtype=np.float32)
    train = cell_features[fit_idx]
    mean = train.mean(axis=0).astype(np.float32)
    std = (train.std(axis=0) + 1e-6).astype(np.float32)
    return ((cell_features - mean) / std).astype(np.float32, copy=False), mean, std


def build_cell_features(args, X_log: sp.csr_matrix, n_cells: int, n_genes: int,
                        fit_indices: Optional[np.ndarray], gene_names: Sequence[str], out_dir: Path) -> np.ndarray:
    """Train-fitted TruncatedSVD cell features, standardized on training cells."""
    n_components = int(min(args.n_components, max(2, min(n_cells - 1, n_genes - 1))))
    fit_idx = _valid_fit_indices(fit_indices, n_cells)
    svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
    svd.fit(X_log[fit_idx])
    cell_features = svd.transform(X_log).astype(np.float32, copy=False)
    cell_features, mean, std = _standardize_cell_features(cell_features, fit_idx)
    svd_path = out_dir / "preprocessing" / "svd_model.pkl"
    svd_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(svd_path, "wb") as f:
        pickle.dump({"svd": svd, "mean": mean, "std": std, "n_components": int(n_components),
                     "n_input_genes": int(n_genes),
                     "gene_names": np.asarray(list(gene_names)).astype(str)}, f)
    return cell_features.astype(np.float32, copy=False)


# =====================================================================
# Dataset splits, tier pools, curriculum
# =====================================================================

def resolve_stage_epoch_boundaries(args, n_epochs: int) -> Tuple[int, int]:
    stage1_end = int(getattr(args, "stage1_end_epoch", 50))
    stage2_end = int(getattr(args, "stage2_end_epoch", 70))
    if not (0 < stage1_end < stage2_end < int(n_epochs)):
        raise ValueError(f"Invalid stage boundaries: {stage1_end}, {stage2_end}, {n_epochs}")
    return stage1_end, stage2_end


def split_cells_and_genes(args, sample_col: np.ndarray, driver_idx_all: List[int], n_genes: int,
                          affected_idx_all: List[int], downstream_idx_all: List[int]) -> Dict[str, object]:
    """Cell-holdout split (paper setting): cells are partitioned 80/10/10 into
    train/val/test; all driver genes act as fixed positives in every split."""
    affected_idx_all = sorted(set(affected_idx_all) - set(driver_idx_all))
    downstream_idx_all = sorted(set(downstream_idx_all) - set(driver_idx_all))
    source_cells = select_cells_by_type(sample_col, args.source_cell_type)
    target_cells = select_cells_by_type(sample_col, args.target_cell_type)
    if len(source_cells) == 0 or len(target_cells) == 0:
        raise ValueError(f"Empty source/target set. source='{args.source_cell_type}' n={len(source_cells)}; "
                         f"target='{args.target_cell_type}' n={len(target_cells)}")
    rng_split = random.Random(args.seed)

    def _split_cells(idxs: List[int]) -> Tuple[List[int], List[int], List[int]]:
        n = len(idxs)
        n_test = int(args.test_frac * n)
        n_val = int(args.val_frac * n)
        return idxs[n_test + n_val:], idxs[n_test:n_test + n_val], idxs[:n_test]

    rng_split.shuffle(source_cells)
    rng_split.shuffle(target_cells)
    src_train, src_val, src_test = _split_cells(source_cells)
    tgt_train, tgt_val, tgt_test = _split_cells(target_cells)
    driver_train_idx = driver_val_idx = driver_test_idx = sorted(set(driver_idx_all))

    all_driver_set = set(driver_idx_all)
    neg_mask = np.ones(n_genes, dtype=bool)
    if all_driver_set:
        neg_mask[np.asarray(sorted(all_driver_set), dtype=np.int64)] = False
    neg_candidates = np.where(neg_mask)[0].astype(np.int64)
    pure_neg_mask = neg_mask.copy()
    if affected_idx_all:
        pure_neg_mask[np.asarray(affected_idx_all, dtype=np.int64)] = False
    pure_neg_candidates = np.where(pure_neg_mask)[0].astype(np.int64)
    return {
        "source_cells": source_cells, "target_cells": target_cells,
        "src_train": src_train, "src_val": src_val, "src_test": src_test,
        "tgt_train": tgt_train, "tgt_val": tgt_val, "tgt_test": tgt_test,
        "driver_train_idx": driver_train_idx, "driver_val_idx": driver_val_idx, "driver_test_idx": driver_test_idx,
        "affected_idx_all": affected_idx_all, "downstream_idx_all": downstream_idx_all,
        "neg_candidates": neg_candidates, "pure_neg_candidates": pure_neg_candidates,
        "true_driver_idx_all": sorted(set(driver_idx_all)),
    }


def build_hybrid_tier_pools(n_genes: int, true_driver_idx_all: List[int], downstream_idx_all: List[int],
                            pseudo_bank_idx: Optional[np.ndarray] = None,
                            holdout_protect_idx: Optional[List[int]] = None) -> Dict[str, object]:
    driver_set = set(true_driver_idx_all)
    downstream_set = set(downstream_idx_all) - driver_set
    if pseudo_bank_idx is not None and len(pseudo_bank_idx) > 0:
        downstream_set |= set(np.asarray(pseudo_bank_idx, dtype=np.int64).tolist()) - driver_set
    holdout_set = set([] if holdout_protect_idx is None else holdout_protect_idx) - driver_set
    neg_mask = np.ones(n_genes, dtype=bool)
    if driver_set:
        neg_mask[np.asarray(sorted(driver_set), dtype=np.int64)] = False
    if holdout_set:
        neg_mask[np.asarray(sorted(holdout_set), dtype=np.int64)] = False
    neg_candidates = np.where(neg_mask)[0].astype(np.int64)
    pure_mask = neg_mask.copy()
    if downstream_set:
        pure_mask[np.asarray(sorted(downstream_set), dtype=np.int64)] = False
    pure_neg_candidates = np.where(pure_mask)[0].astype(np.int64)
    return {"neg_candidates": neg_candidates, "pure_neg_candidates": pure_neg_candidates,
            "downstream_idx_all": sorted(downstream_set)}


def get_hybrid_curriculum_state(epoch: int, n_epochs: int, args) -> Dict[str, float]:
    stage1_end, stage2_end = resolve_stage_epoch_boundaries(args, n_epochs)
    aux_pos_weight = float(min(max(getattr(args, "affected_positive_weight", 0.15), 0.0), 1.0))
    tier_gap = 0.50 * float(args.margin)
    tier_full_gap = 2.0 * tier_gap
    pu_relax_max = float(max(0.0, min(0.8, getattr(args, "pu_relax_max", 0.30))))
    if epoch < stage1_end:
        return {"stage": "warmup", "use_hybrid": 0.0, "prior_scale": 0.04, "downstream_sample_frac": 0.0,
                "tier_score_bias": 0.0, "w_oskm_pure": 1.0, "w_oskm_downstream": 0.0, "w_downstream_pure": 0.0,
                "m_oskm_pure": tier_full_gap, "m_oskm_downstream": tier_gap, "m_downstream_pure": tier_gap, "pu_relax": 0.0}
    if epoch < stage2_end:
        denom = max(1, stage2_end - stage1_end - 1)
        t = min(max(float(epoch - stage1_end) / float(denom), 0.0), 1.0)
        return {"stage": "transition", "use_hybrid": 1.0, "prior_scale": 0.06 + 0.04 * t,
                "downstream_sample_frac": min(0.18, 0.04 + 0.04 * t), "tier_score_bias": 0.35 + 0.15 * t,
                "w_oskm_pure": 1.0, "w_oskm_downstream": 0.32 + 0.08 * t,
                "w_downstream_pure": 0.20 + 0.08 * t + 0.10 * aux_pos_weight,
                "m_oskm_pure": tier_full_gap, "m_oskm_downstream": tier_gap, "m_downstream_pure": tier_gap,
                "pu_relax": 0.30 * pu_relax_max * t}
    denom = max(1, int(n_epochs) - stage2_end - 1)
    t = min(max(float(epoch - stage2_end) / float(denom), 0.0), 1.0)
    return {"stage": "hybrid", "use_hybrid": 1.0, "prior_scale": 0.10 + 0.02 * t,
            "downstream_sample_frac": min(0.20, 0.08 + 0.04 * t), "tier_score_bias": 0.50 + 0.15 * t,
            "w_oskm_pure": 1.0, "w_oskm_downstream": 0.32 + 0.08 * t,
            "w_downstream_pure": 0.20 + 0.08 * t + 0.10 * aux_pos_weight,
            "m_oskm_pure": tier_full_gap, "m_oskm_downstream": tier_gap, "m_downstream_pure": tier_gap,
            "pu_relax": 0.30 * pu_relax_max + 0.70 * pu_relax_max * t}


# =====================================================================
# Episode sampling, negative sampling, ranking losses
# =====================================================================

def sample_episode_batch(src: List[int], tgt: List[int], set_size: int, batch_size: int,
                         rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    S, T = [], []
    for _ in range(batch_size):
        s = rng.choice(src, size=min(set_size, len(src)), replace=False).tolist()
        t = rng.choice(tgt, size=min(set_size, len(tgt)), replace=False).tolist()
        if len(s) < set_size:
            s += rng.choice(src, size=set_size - len(s), replace=True).tolist()
        if len(t) < set_size:
            t += rng.choice(tgt, size=set_size - len(t), replace=True).tolist()
        S.append(s)
        T.append(t)
    return torch.tensor(S, dtype=torch.long), torch.tensor(T, dtype=torch.long)


def pairwise_ranking_loss(scores: torch.Tensor, pos_idx: List[int], neg_idx: List[int], margin: float) -> torch.Tensor:
    if scores.dim() == 1:
        pos, neg = scores[pos_idx], scores[neg_idx]
        if pos.numel() == 0 or neg.numel() == 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        return F.relu(margin - (pos.unsqueeze(1) - neg.unsqueeze(0))).mean()
    if scores.dim() == 2:
        pos, neg = scores[:, pos_idx], scores[:, neg_idx]
        if pos.numel() == 0 or neg.numel() == 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        return F.relu(margin - (pos.unsqueeze(2) - neg.unsqueeze(1))).mean()
    raise ValueError(f"scores must be 1D or 2D, got {tuple(scores.shape)}")


def compute_shift_loss(scores: torch.Tensor, gene_emb: torch.Tensor, control_vecs: torch.Tensor, shift: torch.Tensor,
                       topk: int = 128, temperature: float = 0.2, control_weight: float = 0.25,
                       gene_set_weight: float = 0.50, gene_align_weight: float = 0.25) -> torch.Tensor:
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    if control_vecs.dim() == 1:
        control_vecs = control_vecs.unsqueeze(0)
    if shift.dim() == 1:
        shift = shift.unsqueeze(0)
    _, n_genes = scores.shape
    k = int(max(1, min(int(topk), n_genes)))
    gene_n = F.normalize(gene_emb, dim=-1)
    shift_n = F.normalize(shift, dim=-1)
    ctrl_n = F.normalize(control_vecs, dim=-1)
    top_scores, top_idx = torch.topk(scores, k=k, dim=1, largest=True)
    top_weights = F.softmax(top_scores / max(float(temperature), 1e-6), dim=1)
    sel_gene = gene_n[top_idx]
    pred_shift = F.normalize((top_weights.unsqueeze(-1) * sel_gene).sum(dim=1), dim=-1)
    ctrl_loss = 1.0 - (ctrl_n * shift_n).sum(dim=-1)
    set_loss = 1.0 - (pred_shift * shift_n).sum(dim=-1)
    per_gene_align = torch.einsum("bkh,bh->bk", sel_gene, shift_n)
    align_loss = 1.0 - (top_weights * per_gene_align).sum(dim=1)
    return (float(control_weight) * ctrl_loss + float(gene_set_weight) * set_loss + float(gene_align_weight) * align_loss).mean()


def sample_negatives_with_hard(scores: Optional[torch.Tensor], neg_candidates: np.ndarray, num_neg_samples: int,
                               rng: np.random.Generator, epoch: int, hard_start: int, hard_fraction: float,
                               hard_pool_size: int) -> List[int]:
    neg_candidates = np.asarray(neg_candidates, dtype=np.int64)
    if neg_candidates.size == 0:
        return []
    if neg_candidates.size <= int(num_neg_samples):
        return neg_candidates.tolist()
    selected: List[int] = []
    if scores is not None and (int(epoch) + 1) >= int(hard_start) and float(hard_fraction) > 0.0:
        n_hard = int(round(float(num_neg_samples) * float(hard_fraction)))
        n_hard = min(max(n_hard, 0), int(num_neg_samples), int(neg_candidates.size))
        if n_hard > 0:
            score_mean = scores.detach().float().mean(dim=0) if scores.dim() == 2 else scores.detach().float()
            neg_idx_t = torch.from_numpy(neg_candidates).to(device=score_mean.device, dtype=torch.long)
            pool_k = int(min(max(int(hard_pool_size), n_hard), int(neg_candidates.size)))
            top_idx = torch.topk(score_mean[neg_idx_t], k=pool_k, largest=True).indices.detach().cpu().numpy()
            hard_pool = neg_candidates[top_idx]
            selected.extend(np.asarray(rng.choice(hard_pool, size=min(n_hard, hard_pool.size), replace=False), dtype=np.int64).tolist())
    remaining_n = int(num_neg_samples) - len(selected)
    if remaining_n > 0:
        remaining = np.setdiff1d(neg_candidates, np.asarray(selected, dtype=np.int64)) if selected else neg_candidates
        if remaining.size == 0:
            remaining = neg_candidates
        selected.extend(np.asarray(rng.choice(remaining, size=remaining_n, replace=remaining.size < remaining_n), dtype=np.int64).tolist())
    return selected


def _sample_indices_score_biased(scores: Optional[torch.Tensor], candidates: np.ndarray, num_samples: int,
                                 rng: np.random.Generator, score_bias_frac: float, pool_size: int) -> List[int]:
    candidates = np.asarray(candidates, dtype=np.int64)
    if num_samples <= 0 or candidates.size == 0:
        return []
    if candidates.size <= int(num_samples):
        return candidates.tolist()
    selected: List[int] = []
    n_bias = int(round(float(num_samples) * float(max(0.0, min(1.0, score_bias_frac)))))
    n_bias = min(max(n_bias, 0), int(num_samples), int(candidates.size))
    if scores is not None and n_bias > 0:
        score_mean = scores.detach().float().mean(dim=0) if scores.dim() == 2 else scores.detach().float()
        cand_t = torch.from_numpy(candidates).to(device=score_mean.device, dtype=torch.long)
        pool_k = int(min(max(int(pool_size), n_bias), int(candidates.size)))
        top_idx = torch.topk(score_mean[cand_t], k=pool_k, largest=True).indices.detach().cpu().numpy()
        hard_pool = candidates[top_idx]
        selected.extend(np.asarray(rng.choice(hard_pool, size=min(n_bias, hard_pool.size), replace=False), dtype=np.int64).tolist())
    remaining_n = int(num_samples) - len(selected)
    if remaining_n > 0:
        remaining = np.setdiff1d(candidates, np.asarray(selected, dtype=np.int64)) if selected else candidates
        if remaining.size == 0:
            remaining = candidates
        selected.extend(np.asarray(rng.choice(remaining, size=remaining_n, replace=remaining.size < remaining_n), dtype=np.int64).tolist())
    return selected


def sample_negatives_hybrid(scores: Optional[torch.Tensor], neg_candidates: np.ndarray, pure_neg_candidates: np.ndarray,
                            downstream_idx_all: List[int], num_neg_samples: int, rng: np.random.Generator, epoch: int,
                            hard_start: int, hard_fraction: float, hard_pool_size: int,
                            curriculum_state: Dict[str, float]) -> Dict[str, List[int]]:
    neg_candidates = np.asarray(neg_candidates, dtype=np.int64)
    pure_neg_candidates = np.asarray(pure_neg_candidates, dtype=np.int64)
    downstream_arr = np.asarray(sorted(set(downstream_idx_all)), dtype=np.int64)
    if neg_candidates.size == 0:
        return {"pure_neg_idx": [], "downstream_idx": []}
    if float(curriculum_state.get("use_hybrid", 0.0)) <= 0.0:
        pure_neg_idx = sample_negatives_with_hard(scores, neg_candidates, num_neg_samples, rng, epoch,
                                                  hard_start, hard_fraction, hard_pool_size)
        return {"pure_neg_idx": pure_neg_idx, "downstream_idx": []}
    n_total = int(max(1, num_neg_samples))
    down_frac = float(max(0.0, curriculum_state.get("downstream_sample_frac", 0.0)))
    n_down = int(round(n_total * down_frac)) if downstream_arr.size > 0 else 0
    if downstream_arr.size > 0 and down_frac > 0.0:
        n_down = max(1, n_down)
    n_down = min(n_down, int(downstream_arr.size))
    if pure_neg_candidates.size > 0 and n_down >= n_total:
        overflow = n_down - max(0, n_total - 1)
        if overflow > 0:
            n_down -= min(overflow, n_down)
    n_pure = max(0, n_total - n_down)
    if pure_neg_candidates.size == 0 and n_pure > 0:
        n_pure = 0
    if pure_neg_candidates.size > 0 and n_pure == 0:
        if n_down > 0:
            n_down -= 1
        n_pure = 1
    pure_neg_idx = sample_negatives_with_hard(
        scores, pure_neg_candidates if pure_neg_candidates.size > 0 else neg_candidates,
        max(0, n_pure), rng, epoch, hard_start, hard_fraction, hard_pool_size) if n_pure > 0 else []
    score_bias = float(curriculum_state.get("tier_score_bias", 0.0))
    downstream_idx = _sample_indices_score_biased(
        scores, downstream_arr, n_down, rng, min(1.0, score_bias + 0.10), max(64, hard_pool_size // 3)) if n_down > 0 else []
    return {"pure_neg_idx": pure_neg_idx, "downstream_idx": downstream_idx}


def hybrid_ordinal_pu_ranking_loss(scores: torch.Tensor, driver_idx: List[int], pure_neg_idx: List[int],
                                   downstream_idx: List[int], curriculum_state: Dict[str, float]) -> torch.Tensor:
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    pure_w = float(max(0.5, 1.0 - float(curriculum_state.get("pu_relax", 0.0))))
    loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    if driver_idx and pure_neg_idx:
        loss = loss + float(curriculum_state.get("w_oskm_pure", 1.0)) * pure_w * pairwise_ranking_loss(
            scores, driver_idx, pure_neg_idx, float(curriculum_state.get("m_oskm_pure", 0.99)))
    if driver_idx and downstream_idx:
        loss = loss + float(curriculum_state.get("w_oskm_downstream", 0.0)) * pairwise_ranking_loss(
            scores, driver_idx, downstream_idx, float(curriculum_state.get("m_oskm_downstream", 0.66)))
    if downstream_idx and pure_neg_idx:
        loss = loss + float(curriculum_state.get("w_downstream_pure", 0.0)) * pure_w * pairwise_ranking_loss(
            scores, downstream_idx, pure_neg_idx, float(curriculum_state.get("m_downstream_pure", 0.33)))
    return loss


# =====================================================================
# Pseudo-positive bank
# =====================================================================

def percentile_rank_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    if values.shape[0] == 1:
        ranks[order[0]] = 1.0
        return ranks
    ranks[order] = np.linspace(0.0, 1.0, num=values.shape[0], endpoint=True, dtype=np.float32)
    return ranks


def compute_positive_relation_scores(gene_emb: torch.Tensor, driver_idx: List[int],
                                     affected_idx: Optional[List[int]] = None) -> np.ndarray:
    pos_idx = sorted(set(driver_idx) | set([] if affected_idx is None else affected_idx))
    if len(pos_idx) == 0:
        return np.zeros((gene_emb.shape[0],), dtype=np.float32)
    with torch.no_grad():
        gene_n = F.normalize(gene_emb.float(), dim=-1)
        pos_n = gene_n[torch.tensor(pos_idx, dtype=torch.long, device=gene_n.device)]
        sim = torch.matmul(gene_n, pos_n.T)
        rel = 0.7 * sim.max(dim=1).values + 0.3 * sim.mean(dim=1)
    return rel.detach().cpu().numpy().astype(np.float32)


def update_pseudo_positive_bank(args, epoch: int, score_mean_vec: np.ndarray, relation_vec: np.ndarray,
                                true_driver_idx_all: List[int], affected_idx_all: List[int],
                                pseudo_score_ema: np.ndarray, pseudo_hit_counts: np.ndarray,
                                holdout_protect_idx: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    n_genes = int(score_mean_vec.shape[0])
    stats = {"score_threshold": float("nan"), "relation_threshold": float("nan"), "current_hits": 0, "bank_size": 0}
    if getattr(args, "disable_pseudo_positive_bank", False) or (epoch + 1) < int(args.pseudo_positive_start):
        return np.empty((0,), dtype=np.int64), pseudo_score_ema, pseudo_hit_counts, stats
    unlabeled_mask = np.ones(n_genes, dtype=bool)
    if true_driver_idx_all:
        unlabeled_mask[np.asarray(true_driver_idx_all, dtype=np.int64)] = False
    if affected_idx_all:
        unlabeled_mask[np.asarray(affected_idx_all, dtype=np.int64)] = False
    if holdout_protect_idx:
        unlabeled_mask[np.asarray(sorted(set(holdout_protect_idx)), dtype=np.int64)] = False
    if unlabeled_mask.sum() == 0:
        return np.empty((0,), dtype=np.int64), pseudo_score_ema, pseudo_hit_counts, stats
    combined_rank = 0.5 * percentile_rank_numpy(score_mean_vec) + 0.5 * percentile_rank_numpy(relation_vec)
    decay = float(min(max(args.pseudo_positive_ema_decay, 0.0), 0.999))
    pseudo_score_ema[:] = decay * pseudo_score_ema + (1.0 - decay) * combined_rank
    score_threshold = float(np.quantile(score_mean_vec[unlabeled_mask], float(min(max(args.pseudo_positive_score_quantile, 0.0), 1.0))))
    relation_threshold = float(np.quantile(relation_vec[unlabeled_mask], float(min(max(args.pseudo_positive_relation_quantile, 0.0), 1.0))))
    current_mask = unlabeled_mask & (score_mean_vec >= score_threshold) & (relation_vec >= relation_threshold)
    pseudo_hit_counts[current_mask] += 1
    eligible_mask = unlabeled_mask & (pseudo_hit_counts >= max(1, int(args.pseudo_positive_min_hits)))
    eligible_idx = np.where(eligible_mask)[0].astype(np.int64)
    if eligible_idx.size == 0:
        bank_idx = np.empty((0,), dtype=np.int64)
    else:
        bank_score = pseudo_score_ema[eligible_idx] + 0.02 * pseudo_hit_counts[eligible_idx].astype(np.float32)
        max_keep = min(int(args.pseudo_positive_max_genes), eligible_idx.size)
        bank_idx = eligible_idx[np.argsort(-bank_score)[:max_keep]].astype(np.int64)
    stats.update({"score_threshold": score_threshold, "relation_threshold": relation_threshold,
                  "current_hits": int(current_mask.sum()), "bank_size": int(bank_idx.size)})
    return bank_idx, pseudo_score_ema, pseudo_hit_counts, stats


# =====================================================================
# Reconstruction forward
# =====================================================================

def compute_recon_forward(args, cell_emb, gene_emb, gene_bias, cell_log_sf_t, lib_scale, recon_bilinear_weight,
                          gene_log_theta, gene_logit_pi, X_recon_sparse, X_recon_csc, n_cells, n_genes, rng,
                          num_cells_override=None, num_genes_override=None, cell_indices_override: Optional[Sequence[int]] = None):
    if cell_indices_override is None:
        cell_pool = np.arange(n_cells, dtype=np.int64)
    else:
        cell_pool = np.asarray(list(cell_indices_override), dtype=np.int64)
        cell_pool = cell_pool[(cell_pool >= 0) & (cell_pool < n_cells)]
        if cell_pool.size == 0:
            cell_pool = np.arange(n_cells, dtype=np.int64)
    n_c = num_cells_override if num_cells_override is not None else min(int(cell_pool.size), int(args.expr_num_cells))
    n_c = int(max(1, min(int(n_c), int(cell_pool.size))))
    n_g = num_genes_override if num_genes_override is not None else (min(n_genes, int(args.expr_num_genes)) if int(args.expr_num_genes) > 0 else n_genes)
    n_g = int(max(1, min(int(n_g), int(n_genes))))
    rand_cells = rng.choice(cell_pool, size=n_c, replace=cell_pool.size < n_c)
    rand_genes = rng.choice(n_genes, size=n_g, replace=False)
    device = gene_emb.device
    rand_cells_t = torch.from_numpy(rand_cells).to(device=device, dtype=torch.long, non_blocking=True)
    rand_genes_t = torch.from_numpy(rand_genes).to(device=device, dtype=torch.long, non_blocking=True)
    z_sample = cell_emb[rand_cells_t]
    e_sample = gene_emb[rand_genes_t]
    logits = decode_recon_logits(z_sample, e_sample, gene_bias[rand_genes_t], cell_log_sf_t[rand_cells_t], lib_scale, recon_bilinear_weight)
    if args.recon_loss == "mse":
        pred_expr = apply_decoder_activation(logits, args.recon_decoder_activation)
        actual = torch.from_numpy(X_recon_csc[:, rand_genes][rand_cells, :].toarray().astype(np.float32, copy=False)).to(device=device, dtype=torch.float32, non_blocking=True)
        mask = actual != 0.0 if args.recon_nonzero_only else None
        return masked_mse_loss(pred_expr, actual, mask), pred_expr, actual
    mu = torch.exp(torch.clamp(logits, max=20.0))
    k = torch.from_numpy(X_recon_csc[:, rand_genes][rand_cells, :].toarray().astype(np.float32, copy=False)).to(device=device, dtype=torch.float32, non_blocking=True)
    theta = F.softplus(gene_log_theta[rand_genes_t]).unsqueeze(0) + 1e-4
    mask = (k != 0.0) if args.recon_nonzero_only else None
    if args.recon_loss == "nb":
        return masked_mean(nb_elementwise_nll(k, mu, theta), mask), mu, k
    if args.recon_loss == "zinb":
        pi = torch.sigmoid(gene_logit_pi[rand_genes_t]).unsqueeze(0)
        return masked_mean(zinb_elementwise_nll(k, mu, theta, pi), mask), mu, k
    raise ValueError(f"Unknown recon_loss: {args.recon_loss}")


# =====================================================================
# Graph edge pruning
# =====================================================================

def should_prune(epoch_one_based: int, warmup: int, interval: int) -> bool:
    warmup, interval = max(1, int(warmup)), max(1, int(interval))
    return epoch_one_based >= warmup and ((epoch_one_based - warmup) % interval == 0)


def compute_edge_scores(node_emb: torch.Tensor, graph: SparseGraphData) -> torch.Tensor:
    if graph.num_edges == 0:
        return torch.empty((0,), device=node_emb.device, dtype=torch.float32)
    src_h = F.normalize(node_emb[graph.src].float(), dim=-1)
    dst_h = F.normalize(node_emb[graph.dst].float(), dim=-1)
    sim = 0.5 * ((src_h * dst_h).sum(dim=-1) + 1.0)
    w = graph.edge_weight.float()
    if w.numel() > 0:
        w = (w / w.mean().clamp_min(1e-6)).clamp_max(5.0)
    return sim * w


def prune_graph_edges(graph: SparseGraphData, node_emb: torch.Tensor, percentile: float, max_prune_rate: float,
                      min_degree: int, ema_decay: float) -> Dict[str, float]:
    stats = {"graph": graph.name, "before_edges": int(graph.num_edges), "after_edges": int(graph.num_edges),
             "pruned_edges": 0, "threshold": float("nan"), "avg_score": float("nan")}
    if graph.num_edges == 0:
        return stats
    scores = compute_edge_scores(node_emb, graph).detach()
    if scores.numel() == 0:
        return stats
    stats["avg_score"] = float(scores.mean().item())
    if graph.score_ema is None or graph.score_ema.numel() != scores.numel():
        graph.score_ema = scores.clone()
    else:
        decay = float(min(max(ema_decay, 0.0), 0.999))
        graph.score_ema = decay * graph.score_ema + (1.0 - decay) * scores
    if float(max_prune_rate) <= 0.0:
        return stats
    max_prune = int(math.floor(float(max_prune_rate) * float(graph.num_edges)))
    if max_prune <= 0:
        return stats
    q = float(min(max(percentile / 100.0, 0.0), 1.0))
    threshold = float(torch.quantile(graph.score_ema.float(), q).item())
    stats["threshold"] = threshold
    candidate_idx = torch.nonzero(graph.score_ema <= threshold, as_tuple=False).squeeze(-1)
    if candidate_idx.numel() == 0:
        return stats
    candidate_idx = candidate_idx[torch.argsort(graph.score_ema[candidate_idx])]
    degree = (torch.bincount(graph.src, minlength=graph.num_nodes) + torch.bincount(graph.dst, minlength=graph.num_nodes)).cpu()
    src_cpu, dst_cpu = graph.src.detach().cpu(), graph.dst.detach().cpu()
    keep_mask = torch.ones((graph.num_edges,), dtype=torch.bool, device=graph.src.device)
    pruned = 0
    for edge_idx in candidate_idx.detach().cpu().tolist():
        if pruned >= max_prune:
            break
        s, d = int(src_cpu[edge_idx]), int(dst_cpu[edge_idx])
        if int(degree[s]) <= int(min_degree) or int(degree[d]) <= int(min_degree):
            continue
        keep_mask[edge_idx] = False
        degree[s] -= 1
        degree[d] -= 1
        pruned += 1
    if pruned <= 0:
        return stats
    graph.src = graph.src[keep_mask]
    graph.dst = graph.dst[keep_mask]
    graph.edge_weight = graph.edge_weight[keep_mask]
    if graph.score_ema is not None:
        graph.score_ema = graph.score_ema[keep_mask]
    stats["after_edges"] = int(graph.num_edges)
    stats["pruned_edges"] = int(pruned)
    return stats


# =====================================================================
# Metrics
# =====================================================================

def get_loss_schedule_smooth(epoch: int, n_epochs: int, args) -> Dict[str, float]:
    progress = 1.0 if n_epochs <= 1 else min(max(float(epoch) / float(max(1, n_epochs - 1)), 0.0), 1.0)
    cosine_t = 0.5 - 0.5 * math.cos(math.pi * progress)
    recon_weight = 2.0 * (1.0 - cosine_t) + 0.5 * cosine_t
    rank_weight = 0.5 * (1.0 - cosine_t) + 1.5 * cosine_t
    stage1_end, stage2_end = resolve_stage_epoch_boundaries(args, n_epochs)
    stage = "stage1" if epoch < stage1_end else ("stage2" if epoch < stage2_end else "stage3")
    return {"stage": stage, "recon_weight": float(recon_weight), "rank_weight": float(rank_weight), "shift_weight": 0.5}


def evaluate_ranking(scores: np.ndarray, pos_idx: List[int], K: int = 10) -> Dict[str, float]:
    order = np.argsort(-scores)
    pos_set = set(pos_idx)
    topk = order[:K].tolist()
    hit = 1.0 if any(i in pos_set for i in topk) else 0.0
    prec = sum(1 for i in topk if i in pos_set) / float(K)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    rr = max((1.0 / float(ranks[i] + 1) for i in pos_idx), default=0.0)
    return {"hit@K": hit, "prec@K": prec, "mrr": rr}


def compute_val_recon_pearson(pred_expr: torch.Tensor, actual_expr: torch.Tensor) -> float:
    pred_np = pred_expr.float().cpu().numpy().ravel()
    actual_np = actual_expr.float().cpu().numpy().ravel()
    nz = actual_np != 0.0
    if nz.sum() < 2:
        return float("nan")
    p, a = pred_np[nz], actual_np[nz]
    pm, am = p - p.mean(), a - a.mean()
    denom = np.sqrt((pm ** 2).sum()) * np.sqrt((am ** 2).sum()) + 1e-8
    return float((pm * am).sum() / denom)


def safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)] if arr.size else arr
    return float(arr.mean()) if arr.size else float("nan")


def compute_val_driver_metrics(scores_np: np.ndarray, driver_val_idx: List[int], n_genes: int) -> Tuple[float, float]:
    y_true = np.zeros(n_genes, dtype=np.float32)
    y_true[driver_val_idx] = 1.0
    if y_true.sum() == 0:
        return float("nan"), float("nan")
    auprc = float(average_precision_score(y_true, scores_np))
    K = len(driver_val_idx)
    top_k = np.argsort(-scores_np)[:K]
    return auprc, float(np.isin(top_k, driver_val_idx).sum()) / max(1, K)


def compute_val_auprc(scores_np: np.ndarray, positive_idx: List[int], n_genes: int) -> float:
    y_true = np.zeros(n_genes, dtype=np.float32)
    y_true[positive_idx] = 1.0
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores_np))


# =====================================================================
# Runtime helpers
# =====================================================================

def gather_cell_embeddings(cell_emb: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if idx.device != cell_emb.device:
        idx = idx.to(device=cell_emb.device, dtype=torch.long, non_blocking=True)
    return cell_emb[idx]


def encode_joint_embeddings(use_amp: bool, cell_features_t: torch.Tensor, cell_graph: SparseGraphData,
                            x_feature_t_sparse: torch.Tensor, gene_graph: SparseGraphData,
                            cell_encoder, gene_feature_projector, gene_encoder) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    amp = bool(use_amp and cell_features_t.device.type == "cuda")
    with autocast(device_type="cuda", enabled=amp):
        cell_emb = cell_encoder(cell_features_t, cell_graph)
        gene_features_t = gene_feature_projector(x_feature_t_sparse, cell_emb)
        gene_emb = gene_encoder(gene_features_t, gene_graph)
    return cell_emb, gene_features_t, gene_emb


# =====================================================================
# IO helpers
# =====================================================================

def save_unknown_driver_candidates(csv_path: Path, gene_names: List[str], score_vec: np.ndarray, relation_vec: np.ndarray,
                                   true_driver_idx_all: List[int], affected_idx_all: List[int], pseudo_bank_idx: np.ndarray,
                                   pseudo_hit_counts: np.ndarray, pseudo_score_ema: np.ndarray, topk: int = 500) -> None:
    n_genes = len(gene_names)
    protected = np.zeros(n_genes, dtype=bool)
    protected[np.asarray(true_driver_idx_all, dtype=np.int64)] = True
    if affected_idx_all:
        protected[np.asarray(affected_idx_all, dtype=np.int64)] = True
    unlabeled_idx = np.where(~protected)[0].astype(np.int64)
    cols = ["rank", "gene", "score", "relation_to_positive", "pseudo_score_ema", "pseudo_hit_count", "protected_from_negative_sampling"]
    if unlabeled_idx.size == 0:
        pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
        return
    combined = 0.45 * percentile_rank_numpy(score_vec) + 0.35 * percentile_rank_numpy(relation_vec) + 0.20 * percentile_rank_numpy(pseudo_score_ema)
    order = unlabeled_idx[np.argsort(-combined[unlabeled_idx])]
    selected = order[:min(int(topk), order.size)]
    pseudo_bank_set = set(np.asarray(pseudo_bank_idx, dtype=np.int64).tolist())
    rows = [{"rank": r, "gene": gene_names[int(i)], "score": float(score_vec[int(i)]),
             "relation_to_positive": float(relation_vec[int(i)]), "pseudo_score_ema": float(pseudo_score_ema[int(i)]),
             "pseudo_hit_count": int(pseudo_hit_counts[int(i)]),
             "protected_from_negative_sampling": int(int(i) in pseudo_bank_set)}
            for r, i in enumerate(selected, start=1)]
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def save_protected_gene_sets(csv_path: Path, gene_names: List[str], true_driver_idx_all: List[int],
                             affected_idx_all: List[int], pseudo_bank_idx: np.ndarray,
                             downstream_idx_all: List[int], dynamic_downstream_idx: List[int]) -> None:
    downstream_set = set(downstream_idx_all)
    pseudo_bank_set = set(np.asarray(pseudo_bank_idx, dtype=np.int64).tolist())
    dynamic_set = set(dynamic_downstream_idx) - downstream_set - pseudo_bank_set
    rows = [{"gene": gene_names[int(i)], "label": "true_driver"} for i in sorted(set(true_driver_idx_all))]
    for i in sorted(set(affected_idx_all)):
        rows.append({"gene": gene_names[int(i)], "label": "direct_downstream" if i in downstream_set else "affected_gene"})
    rows += [{"gene": gene_names[int(i)], "label": "dynamic_ambiguous_candidate"} for i in sorted(dynamic_set)]
    rows += [{"gene": gene_names[int(i)], "label": "pseudo_positive_candidate"} for i in sorted(pseudo_bank_set)]
    pd.DataFrame(rows).drop_duplicates().to_csv(csv_path, index=False)


def convert_split_indices_to_names(splits, obs_names, gene_names=None,
                                   cell_keys=("source_cells", "target_cells", "src_train", "src_val", "src_test",
                                              "tgt_train", "tgt_val", "tgt_test"),
                                   gene_keys=("driver_train_idx", "driver_val_idx", "driver_test_idx",
                                              "affected_idx_all", "downstream_idx_all", "true_driver_idx_all")):
    obs_names = np.asarray(obs_names)
    gene_names = np.asarray([]) if gene_names is None else np.asarray(gene_names)
    out = {}
    for k, v in splits.items():
        if k in cell_keys:
            out[k] = obs_names[np.asarray(v, dtype=int)].tolist()
        elif k in gene_keys:
            out[k.replace("_idx", "")] = gene_names[np.asarray(v, dtype=int)].tolist()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
