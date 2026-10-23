"""
FateDriver — train a dual-view (cell + gene) graph encoder with a
cross-attention ranking head to rank driver genes of a source -> target cell
state transition.

The default hyper-parameters below reproduce the paper configuration:
    SVD(50) cell features -> GCN cell/gene encoders (hidden 1024, 2 layers) ->
    attentive set pooling -> cross-attention ranking head with a learnable
    transition prior, trained with a staged PU / ordinal ranking loss and a
    ZINB reconstruction auxiliary, on a pseudotime cell graph and a NicheNet
    gene graph with online edge pruning.

Typical usage:
    python train_fatedriver.py \
        --anndata_path      data/GSE242423/anndata.h5ad \
        --pseudotime_csv_path data/GSE242423/pseudotime.csv \
        --driver_genes_path data/gene_sets/iPSC_oskm_driver_human_genes.csv \
        --affected_genes_path data/gene_sets/TRRUST_human_OSKM_targets.csv \
        --gene_nichenet_df_path data/gene_sets/nichenet_gr_signaling_net_human.csv \
        --source_cell_type D0 --target_cell_type iPSC --obs_column condition \
        --output_dir outputs/GSE242423 --seed 42
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pickle
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message=r"'force_all_finite' was renamed.*", category=FutureWarning)

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import utils
from model import (
    AttentiveSetEncoder,
    CrossAttentionRankingHead,
    GeneFeatureProjector,
    LearnableTransitionDriverPrior,
    make_graph_encoder,
    module_device,
)

torch.backends.cudnn.benchmark = False


# =====================================================================
# Argument parsing (defaults = paper configuration)
# =====================================================================

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FateDriver — driver-gene ranking for cell-state transitions")
    # --- inputs -------------------------------------------------------
    p.add_argument("--anndata_path", type=str, required=True, help="Preprocessed .h5ad (adata.X features + layers['counts'] raw counts + var['highly_variable']).")
    p.add_argument("--pseudotime_csv_path", type=str, required=True, help="CSV with a cell-id column and a 'pseudotime' column.")
    p.add_argument("--driver_genes_path", type=str, required=True, help="CSV of known driver (intervention) genes = positive labels.")
    p.add_argument("--affected_genes_path", type=str, default=None, help="CSV of downstream/affected genes (soft-negative tier).")
    p.add_argument("--gene_nichenet_df_path", type=str, required=True, help="NicheNet signaling table used to build the gene graph.")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--dataset", type=str, default="dataset")
    p.add_argument("--source_cell_type", type=str, required=True, help="obs value(s) for source cells (comma-separated).")
    p.add_argument("--target_cell_type", type=str, required=True, help="obs value(s) for target cells (comma-separated).")
    p.add_argument("--obs_column", type=str, default="condition", help="adata.obs column holding the source/target labels.")
    p.add_argument("--pseudotime_key", type=str, default="pseudotime")
    p.add_argument("--pseudotime_cell_id_col", type=str, default="auto")
    # --- runtime ------------------------------------------------------
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_num", type=int, default=0)
    # --- features / graphs -------------------------------------------
    p.add_argument("--set_HVG", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force_include_prior_genes_in_hvg", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_hvg", type=int, default=3000, help="Top-mean genes eligible for the gene graph.")
    p.add_argument("--n_components", type=int, default=50, help="TruncatedSVD components for cell features.")
    p.add_argument("--knn_k", type=int, default=15, help="k for the kNN neighbourhood underlying the trajectory graph.")
    p.add_argument("--min_delta_t", type=float, default=0.0)
    p.add_argument("--trajectory_sigma", type=float, default=0.0)
    p.add_argument("--feature_layer", type=str, default="X", help="AnnData layer for model features ('X' = adata.X).")
    p.add_argument("--counts_layer", type=str, default="counts", help="AnnData layer with raw counts for NB/ZINB reconstruction.")
    # --- model --------------------------------------------------------
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_cell_layers", type=int, default=2)
    p.add_argument("--num_gene_layers", type=int, default=2)
    p.add_argument("--ranking_chunk_size", type=int, default=512)
    # --- reconstruction ----------------------------------------------
    p.add_argument("--recon_loss", type=str, default="zinb", choices=["mse", "nb", "zinb"])
    p.add_argument("--recon_decoder_activation", type=str, default="exp", choices=["relu", "softplus", "exp", "identity"])
    p.add_argument("--recon_nonzero_only", action="store_true")
    p.add_argument("--nb_dispersion_init", type=float, default=2.0)
    p.add_argument("--zinb_pi_init", type=float, default=0.1)
    p.add_argument("--target_sum", type=float, default=1e4)
    p.add_argument("--expr_num_cells", type=int, default=5000)
    p.add_argument("--expr_num_genes", type=int, default=0)
    # --- optimisation -------------------------------------------------
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--stage1_end_epoch", type=int, default=50)
    p.add_argument("--stage2_end_epoch", type=int, default=70)
    p.add_argument("--steps_per_epoch", type=int, default=10)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--early_stopping_min_stage", type=str, default="stage3", choices=["stage1", "stage2", "stage3"])
    # --- ranking / episodes ------------------------------------------
    p.add_argument("--set_size", type=int, default=128)
    p.add_argument("--episode_batch_size", type=int, default=1024)
    p.add_argument("--num_neg_samples", type=int, default=500)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--val_repeats", type=int, default=10)
    p.add_argument("--test_repeats", type=int, default=10)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--hard_negative_start", type=int, default=30)
    p.add_argument("--hard_negative_fraction", type=float, default=0.0)
    p.add_argument("--hard_negative_pool_size", type=int, default=2048)
    # --- shift-consistency loss --------------------------------------
    p.add_argument("--shift_topk", type=int, default=128)
    p.add_argument("--shift_temperature", type=float, default=0.2)
    p.add_argument("--shift_control_weight", type=float, default=0.25)
    p.add_argument("--shift_gene_set_weight", type=float, default=0.50)
    p.add_argument("--shift_gene_align_weight", type=float, default=0.25)
    # --- PU / pseudo-positive tiering --------------------------------
    p.add_argument("--affected_handling", type=str, default="protect", choices=["protect", "weighted_positive"])
    p.add_argument("--affected_positive_weight", type=float, default=0.15)
    p.add_argument("--pu_relax_max", type=float, default=0.30)
    p.add_argument("--pseudo_positive_start", type=int, default=10)
    p.add_argument("--pseudo_positive_score_quantile", type=float, default=0.995)
    p.add_argument("--pseudo_positive_relation_quantile", type=float, default=0.98)
    p.add_argument("--pseudo_positive_max_genes", type=int, default=128)
    p.add_argument("--pseudo_positive_min_hits", type=int, default=2)
    p.add_argument("--pseudo_positive_ema_decay", type=float, default=0.8)
    p.add_argument("--disable_pseudo_positive_bank", action="store_true")
    p.add_argument("--save_unknown_driver_candidates_topk", type=int, default=500)
    # --- graph pruning ------------------------------------------------
    p.add_argument("--use_cell_graph_pruning", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_gene_graph_pruning", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cell_prune_warmup", type=int, default=20)
    p.add_argument("--cell_prune_interval", type=int, default=20)
    p.add_argument("--cell_prune_percentile", type=float, default=10.0)
    p.add_argument("--cell_max_prune_rate", type=float, default=0.10)
    p.add_argument("--cell_min_degree", type=int, default=5)
    p.add_argument("--gene_prune_warmup", type=int, default=20)
    p.add_argument("--gene_prune_interval", type=int, default=20)
    p.add_argument("--gene_prune_percentile", type=float, default=5.0)
    p.add_argument("--gene_max_prune_rate", type=float, default=0.05)
    p.add_argument("--gene_min_degree", type=int, default=3)
    p.add_argument("--prune_ema_decay", type=float, default=0.8)
    # --- embedding export --------------------------------------------
    p.add_argument("--skip_embedding_export", action="store_true")
    p.add_argument("--embedding_umap_neighbors", type=int, default=30)
    p.add_argument("--embedding_umap_min_dist", type=float, default=0.35)

    args = p.parse_args()
    if args.n_epochs < 3:
        p.error("--n_epochs must be at least 3")
    if not (0 < args.stage1_end_epoch < args.stage2_end_epoch < args.n_epochs):
        p.error("Require 0 < stage1_end_epoch < stage2_end_epoch < n_epochs")
    return args


# =====================================================================
# Runtime / model init
# =====================================================================

def init_runtime(args: argparse.Namespace) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{int(args.device_num)}") if torch.cuda.is_available() else torch.device("cpu")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{timestamp}_dataset-{args.dataset}_seed-{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = vars(args).copy()
    meta.update({"timestamp": timestamp, "device": str(device)})
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return {"timestamp": timestamp, "out_dir": out_dir, "device": device}


def update_run_metadata(out_dir: Path, updates: Dict[str, object]) -> None:
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(updates)
    meta_path.write_text(json.dumps(meta, indent=2))


def init_model_and_optimizer(args: argparse.Namespace, n_genes: int, cell_features: np.ndarray, gene_bias_init: np.ndarray,
                             driver_prior_features: Optional[np.ndarray], driver_prior_feature_names: Optional[List[str]],
                             device: torch.device) -> Dict[str, object]:
    cell_features_t = torch.from_numpy(cell_features).to(device=device, dtype=torch.float32)
    cell_encoder = make_graph_encoder(cell_features_t.shape[1], args.hidden_dim, args.num_cell_layers, args.dropout).to(device)
    gene_feature_projector = GeneFeatureProjector(n_genes, args.hidden_dim, args.dropout).to(device)
    gene_encoder = make_graph_encoder(args.hidden_dim, args.hidden_dim, args.num_gene_layers, args.dropout).to(device)
    gene_bias = nn.Parameter(torch.from_numpy(gene_bias_init).to(device=device, dtype=torch.float32))
    lib_scale = nn.Parameter(torch.tensor(1.0, device=device))
    recon_bilinear_weight = nn.Parameter(torch.eye(args.hidden_dim, device=device))
    gene_log_theta = gene_logit_pi = None
    if args.recon_loss in {"nb", "zinb"}:
        init_theta = float(max(1e-3, args.nb_dispersion_init))
        inv_sp = math.log(math.exp(init_theta) - 1.0) if init_theta > 1e-6 else -10.0
        gene_log_theta = nn.Parameter(torch.full((n_genes,), float(inv_sp), device=device))
        if args.recon_loss == "zinb":
            pi0 = float(min(max(args.zinb_pi_init, 1e-4), 1.0 - 1e-4))
            gene_logit_pi = nn.Parameter(torch.full((n_genes,), float(math.log(pi0 / (1.0 - pi0))), device=device))
    set_encoder = AttentiveSetEncoder(args.hidden_dim, args.hidden_dim, args.dropout, args.num_heads).to(device)
    ranking_head = CrossAttentionRankingHead(args.hidden_dim, args.num_heads, args.dropout, args.ranking_chunk_size).to(device)

    driver_prior_model = None
    if driver_prior_features is not None:
        driver_prior_model = LearnableTransitionDriverPrior(
            feature_matrix=torch.from_numpy(driver_prior_features).to(device=device, dtype=torch.float32),
            feature_names=[] if driver_prior_feature_names is None else list(driver_prior_feature_names),
            init_weights=np.asarray([0.40, 0.20, 0.10, 0.20, 0.10], dtype=np.float32),
        ).to(device)
        ranking_head.set_prior_module(driver_prior_model)

    params = (list(cell_encoder.parameters()) + list(gene_feature_projector.parameters())
              + list(gene_encoder.parameters()) + list(set_encoder.parameters()) + list(ranking_head.parameters())
              + [gene_bias, lib_scale, recon_bilinear_weight])
    if gene_log_theta is not None:
        params.append(gene_log_theta)
    if gene_logit_pi is not None:
        params.append(gene_logit_pi)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return {"cell_features_t": cell_features_t, "cell_encoder": cell_encoder,
            "gene_feature_projector": gene_feature_projector, "gene_encoder": gene_encoder,
            "set_encoder": set_encoder, "ranking_head": ranking_head, "driver_prior_model": driver_prior_model,
            "gene_bias": gene_bias, "lib_scale": lib_scale, "recon_bilinear_weight": recon_bilinear_weight,
            "gene_log_theta": gene_log_theta, "gene_logit_pi": gene_logit_pi, "optimizer": optimizer}


# =====================================================================
# Validation / test inference / checkpoint helpers
# =====================================================================

def compute_validation_metrics(args, rng, use_amp, device, cell_emb_val, gene_emb_val, set_encoder, ranking_head,
                               gene_bias, cell_log_sf_t, lib_scale, recon_bilinear_weight, gene_log_theta, gene_logit_pi,
                               X_recon_sparse, X_recon_csc, n_cells, n_genes, src_val, tgt_val, driver_val_idx,
                               neg_candidates, pure_neg_candidates, downstream_idx, curriculum_state, epoch) -> Dict[str, object]:
    repeats = max(1, int(args.val_repeats))
    n_val_steps = max(1, int(args.steps_per_epoch) // 4)
    acc = {k: [] for k in ["val_expr", "val_rank", "val_shift", "pearson_nz", "auprc", "prec_at_k", "early_stop_auprc"]}
    score_vecs: List[np.ndarray] = []
    downstream_idx = list(downstream_idx or [])
    recon_cell_pool = sorted(set(src_val) | set(tgt_val))
    gene_ctx_val = ranking_head._contextualize_genes(gene_emb_val)
    for _ in range(repeats):
        val_expr_sum = val_rank_sum = val_shift_sum = 0.0
        pearsons: List[float] = []
        for _step in range(n_val_steps):
            with autocast(device_type="cuda", enabled=use_amp):
                S_idx, T_idx = utils.sample_episode_batch(src_val, tgt_val, args.set_size, args.episode_batch_size, rng)
                z_S = set_encoder(utils.gather_cell_embeddings(cell_emb_val, S_idx))
                z_T = set_encoder(utils.gather_cell_embeddings(cell_emb_val, T_idx))
                scores_v, _, control_vecs_v = ranking_head(z_S, z_T, gene_emb_val, gene_ctx=gene_ctx_val)
                n_c = min(n_cells, max(10, int(args.expr_num_cells) // 2))
                n_g = min(n_genes, max(50, int(args.expr_num_genes) // 2)) if int(args.expr_num_genes) > 0 else n_genes
                expr_loss_v, pred_v, actual_v = utils.compute_recon_forward(
                    args, cell_emb_val, gene_emb_val, gene_bias, cell_log_sf_t, lib_scale, recon_bilinear_weight,
                    gene_log_theta, gene_logit_pi, X_recon_sparse, X_recon_csc, n_cells, n_genes, rng,
                    num_cells_override=n_c, num_genes_override=n_g, cell_indices_override=recon_cell_pool)
                tier_v = utils.sample_negatives_hybrid(
                    scores_v, neg_candidates, pure_neg_candidates, downstream_idx,
                    max(100, int(args.num_neg_samples) // 2), rng, int(epoch),
                    max(1, int(args.hard_negative_start)), float(args.hard_negative_fraction),
                    int(args.hard_negative_pool_size), curriculum_state)
                rank_loss_v = utils.hybrid_ordinal_pu_ranking_loss(scores_v, driver_val_idx, tier_v["pure_neg_idx"], tier_v["downstream_idx"], curriculum_state)
                shift_loss_v = utils.compute_shift_loss(scores_v, gene_emb_val, control_vecs_v, z_T - z_S,
                                                        topk=args.shift_topk, temperature=args.shift_temperature,
                                                        control_weight=args.shift_control_weight,
                                                        gene_set_weight=args.shift_gene_set_weight,
                                                        gene_align_weight=args.shift_gene_align_weight)
            val_expr_sum += float(expr_loss_v.item())
            val_rank_sum += float(rank_loss_v.item())
            val_shift_sum += float(shift_loss_v.item())
            pred_nz, actual_nz = pred_v[actual_v > 0], actual_v[actual_v > 0]
            if pred_nz.numel() > 10:
                pearsons.append(utils.compute_val_recon_pearson(pred_nz, actual_nz))
        acc["val_expr"].append(val_expr_sum / max(1, n_val_steps))
        acc["val_rank"].append(val_rank_sum / max(1, n_val_steps))
        acc["val_shift"].append(val_shift_sum / max(1, n_val_steps))
        acc["pearson_nz"].append(utils.safe_nanmean(pearsons))
        eval_set_size = min(max(64, int(args.set_size)), len(src_val), len(tgt_val))
        S_eval = torch.tensor(rng.choice(np.asarray(src_val, dtype=np.int64), size=eval_set_size, replace=len(src_val) < eval_set_size), dtype=torch.long)
        T_eval = torch.tensor(rng.choice(np.asarray(tgt_val, dtype=np.int64), size=eval_set_size, replace=len(tgt_val) < eval_set_size), dtype=torch.long)
        with autocast(device_type="cuda", enabled=use_amp):
            z_S_eval = set_encoder(utils.gather_cell_embeddings(cell_emb_val, S_eval).unsqueeze(0))
            z_T_eval = set_encoder(utils.gather_cell_embeddings(cell_emb_val, T_eval).unsqueeze(0))
            scores_full, _, _ = ranking_head(z_S_eval, z_T_eval, gene_emb_val, gene_ctx=gene_ctx_val)
        scores_np_val = scores_full.squeeze(0).float().detach().cpu().numpy()
        score_vecs.append(scores_np_val)
        auprc_val, prec_val = utils.compute_val_driver_metrics(scores_np_val, driver_val_idx, n_genes)
        es_val = utils.compute_val_auprc(scores_np_val, sorted(set(driver_val_idx) | set(downstream_idx)), n_genes)
        acc["auprc"].append(auprc_val)
        acc["prec_at_k"].append(prec_val)
        acc["early_stop_auprc"].append(es_val)
    score_mean_vec = np.mean(np.stack(score_vecs, axis=0), axis=0).astype(np.float32) if score_vecs else np.zeros((n_genes,), np.float32)
    return {k: utils.safe_nanmean(v) for k, v in acc.items()} | {"score_mean_vec": score_mean_vec}


def run_test_inference(args, rng, use_amp, device, cell_features_t, cell_graph, X_feature_t_sparse_gpu, gene_graph,
                       cell_encoder, gene_feature_projector, gene_encoder, set_encoder, ranking_head,
                       src_test, tgt_test) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for m in (cell_encoder, gene_feature_projector, gene_encoder, set_encoder, ranking_head):
        m.eval()
    repeats = max(1, int(args.test_repeats))
    score_accum: List[np.ndarray] = []
    with torch.no_grad():
        cell_emb, _, gene_emb = utils.encode_joint_embeddings(
            use_amp, cell_features_t, cell_graph, X_feature_t_sparse_gpu, gene_graph,
            cell_encoder, gene_feature_projector, gene_encoder)
        gene_ctx = ranking_head._contextualize_genes(gene_emb)
        eval_set_size = min(max(64, int(args.set_size)), len(src_test), len(tgt_test))
        for _ in range(repeats):
            S_eval = torch.tensor(rng.choice(np.asarray(src_test, dtype=np.int64), size=eval_set_size, replace=len(src_test) < eval_set_size), dtype=torch.long)
            T_eval = torch.tensor(rng.choice(np.asarray(tgt_test, dtype=np.int64), size=eval_set_size, replace=len(tgt_test) < eval_set_size), dtype=torch.long)
            with autocast(device_type="cuda", enabled=use_amp):
                z_S = set_encoder(utils.gather_cell_embeddings(cell_emb, S_eval).unsqueeze(0))
                z_T = set_encoder(utils.gather_cell_embeddings(cell_emb, T_eval).unsqueeze(0))
                scores, _, _ = ranking_head(z_S, z_T, gene_emb, gene_ctx=gene_ctx)
            score_accum.append(scores.squeeze(0).float().cpu().numpy())
    scores_np = np.mean(np.stack(score_accum, axis=0), axis=0).astype(np.float32)
    return scores_np, cell_emb.float().cpu().numpy(), gene_emb.float().cpu().numpy()


def snapshot_state_dicts(modules: Dict[str, nn.Module], extras: Dict[str, torch.Tensor]) -> Dict[str, object]:
    import copy
    snap: Dict[str, object] = {n: (None if m is None else copy.deepcopy(m.state_dict())) for n, m in modules.items()}
    snap.update({n: (None if t is None else t.detach().cpu().clone()) for n, t in extras.items()})
    return snap


def restore_state_dicts(modules: Dict[str, nn.Module], extras: Dict[str, torch.Tensor], snapshot: Dict[str, object], device: torch.device) -> None:
    for n, m in modules.items():
        if m is not None and snapshot.get(n) is not None:
            m.load_state_dict(snapshot[n])
    for n, t in extras.items():
        if t is not None and snapshot.get(n) is not None:
            t.data.copy_(snapshot[n].to(device=device, dtype=t.dtype))


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_arguments()
    runtime = init_runtime(args)
    device = runtime["device"]
    out_dir = runtime["out_dir"]
    use_amp = device.type == "cuda"

    # ---- data & preprocessing ---------------------------------------
    data = utils.load_expression_and_meta(args)
    pre = utils.preprocess_expression(args, data["X_sparse"], device, X_counts_sparse=data["X_counts_sparse"])
    svd_fit_indices = utils.infer_svd_fit_indices(args, data["sample_col"])
    cell_features = utils.build_cell_features(args, pre["X_feature"], data["n_cells"], data["n_genes"],
                                              fit_indices=svd_fit_indices, gene_names=data["gene_names"], out_dir=out_dir)
    cell_graph = utils.build_cell_graph(cell_features, data["n_cells"], data["pseudotime"],
                                        args.knn_k, args.min_delta_t, args.trajectory_sigma, device, out_dir)
    gene_graph, gene_name_to_idx = utils.build_gene_graph(args.gene_nichenet_df_path, pre["X_feature"],
                                                          data["gene_names"], data["n_genes"], args.max_hvg, device, out_dir)

    # ---- gene sets ---------------------------------------------------
    driver_idx_all = utils.load_gene_indices_from_csv(args.driver_genes_path, gene_name_to_idx, "driver", required=True)
    downstream_idx_all = utils.load_gene_indices_from_csv(args.affected_genes_path, gene_name_to_idx, "downstream", required=False)
    downstream_idx_all = sorted(set(downstream_idx_all) - set(driver_idx_all))
    affected_idx_all = list(downstream_idx_all)

    # ---- learnable-prior features -----------------------------------
    prior_source_cells = utils.select_cells_by_type(data["sample_col"], args.source_cell_type)
    prior_target_cells = utils.select_cells_by_type(data["sample_col"], args.target_cell_type)
    if svd_fit_indices is not None:
        fit_set = set(np.asarray(svd_fit_indices, dtype=np.int64).tolist())
        s_tr = [i for i in prior_source_cells if i in fit_set]
        t_tr = [i for i in prior_target_cells if i in fit_set]
        if s_tr and t_tr:
            prior_source_cells, prior_target_cells = s_tr, t_tr
    prior_features, prior_feature_names = utils.compute_gene_transition_driver_prior_features(
        pre["X_feature"], prior_source_cells, prior_target_cells, gene_graph=gene_graph, pseudotime=data["pseudotime"])

    update_run_metadata(out_dir, {
        "n_cells": int(data["n_cells"]), "n_genes": int(data["n_genes"]),
        "driver_gene_count": int(len(driver_idx_all)), "downstream_gene_count": int(len(downstream_idx_all)),
        "cell_edges_init": int(cell_graph.num_edges), "gene_edges_init": int(gene_graph.num_edges),
        "driver_prior_feature_names": prior_feature_names,
    })

    # ---- splits & model ---------------------------------------------
    splits = utils.split_cells_and_genes(args, data["sample_col"], driver_idx_all, data["n_genes"],
                                         affected_idx_all=affected_idx_all, downstream_idx_all=downstream_idx_all)
    model = init_model_and_optimizer(args, data["n_genes"], cell_features, pre["gene_bias_init"],
                                     prior_features, prior_feature_names, device)

    # ---- sparse feature tensor (genes x cells) ----------------------
    X_feature_coo = pre["X_feature"].tocoo()
    idx_t = torch.from_numpy(np.vstack([X_feature_coo.row, X_feature_coo.col]).astype(np.int64))
    val_t = torch.from_numpy(X_feature_coo.data.astype(np.float32))
    X_feature_sparse = torch.sparse_coo_tensor(idx_t, val_t, pre["X_feature"].shape).coalesce().to(device)
    X_feature_t_sparse_gpu = X_feature_sparse.transpose(0, 1).coalesce()
    X_recon = pre["X_recon"]
    X_recon_csc = X_recon.tocsc()

    # persist splits (as names)
    splits_named = utils.convert_split_indices_to_names(splits, obs_names=data["obs_names"], gene_names=np.asarray(data["gene_names"]))
    with open(out_dir / "splits_named.pkl", "wb") as f:
        pickle.dump(splits_named, f)

    # ---- unpack model / splits --------------------------------------
    (cell_features_t, cell_encoder, gene_feature_projector, gene_encoder, set_encoder, ranking_head,
     driver_prior_model, gene_bias, lib_scale, recon_bilinear_weight, gene_log_theta, gene_logit_pi, optimizer) = (
        model["cell_features_t"], model["cell_encoder"], model["gene_feature_projector"], model["gene_encoder"],
        model["set_encoder"], model["ranking_head"], model["driver_prior_model"], model["gene_bias"], model["lib_scale"],
        model["recon_bilinear_weight"], model["gene_log_theta"], model["gene_logit_pi"], model["optimizer"])
    cell_log_sf_t = pre["cell_log_sf_t"]
    rng = np.random.default_rng(args.seed)
    n_cells, n_genes = data["n_cells"], data["n_genes"]

    src_train, tgt_train = splits["src_train"], splits["tgt_train"]
    src_val = splits["src_val"] or splits["src_train"]
    tgt_val = splits["tgt_val"] or splits["tgt_train"]
    src_test = splits["src_test"] or splits["src_train"]
    tgt_test = splits["tgt_test"] or splits["tgt_train"]
    driver_train_idx = splits["driver_train_idx"]
    driver_val_idx = splits["driver_val_idx"]
    true_driver_idx_all = splits["true_driver_idx_all"]
    downstream_idx_all = splits["downstream_idx_all"]
    holdout_protect_idx = sorted(set(driver_val_idx) | set(splits["driver_test_idx"]))

    base_pools = utils.build_hybrid_tier_pools(n_genes, true_driver_idx_all, downstream_idx_all,
                                               pseudo_bank_idx=None, holdout_protect_idx=holdout_protect_idx)
    current_neg = np.asarray(base_pools["neg_candidates"], dtype=np.int64)
    current_pure_neg = np.asarray(base_pools["pure_neg_candidates"], dtype=np.int64)
    current_downstream = list(base_pools["downstream_idx_all"])
    pseudo_bank = np.empty((0,), dtype=np.int64)
    pseudo_score_ema = np.zeros((n_genes,), dtype=np.float32)
    pseudo_hit_counts = np.zeros((n_genes,), dtype=np.int64)
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- logging -----------------------------------------------------
    log_path = out_dir / "training_log.csv"
    log_path.write_text("epoch,stage,recon_weight,rank_weight,shift_weight,train_expr,train_rank,train_shift,train_total,"
                        "val_expr,val_rank,val_shift,val_total,pearson_nz,auprc,prec_at_k,early_stop_auprc,"
                        "neg_pool_size,pseudo_positive_bank_size,cell_edges,gene_edges\n")
    pruning_log_path = out_dir / "graph_pruning.csv"
    pruning_log_path.write_text("epoch,graph,before_edges,after_edges,pruned_edges,threshold,avg_score\n")

    patience = max(1, int(args.early_stopping_patience))
    best_metric, best_epoch, no_improve = float("-inf"), -1, 0
    stage_priority = {"stage1": 1, "stage2": 2, "stage3": 3}
    min_stage = stage_priority.get(args.early_stopping_min_stage, 3)
    modules_snap = {"cell_encoder": cell_encoder, "gene_feature_projector": gene_feature_projector,
                    "gene_encoder": gene_encoder, "set_encoder": set_encoder, "ranking_head": ranking_head}
    extras_snap = {"gene_bias": gene_bias, "lib_scale": lib_scale, "recon_bilinear_weight": recon_bilinear_weight,
                   "gene_log_theta": gene_log_theta, "gene_logit_pi": gene_logit_pi}
    best_snapshot: Optional[Dict[str, object]] = None
    last_curriculum = utils.get_hybrid_curriculum_state(0, max(2, int(args.n_epochs)), args)

    # ---- training loop ----------------------------------------------
    for epoch in tqdm(range(args.n_epochs), desc="epochs"):
        schedule = utils.get_loss_schedule_smooth(epoch, args.n_epochs, args)
        recon_weight, rank_weight, shift_weight = schedule["recon_weight"], schedule["rank_weight"], schedule["shift_weight"]
        curriculum = utils.get_hybrid_curriculum_state(epoch, args.n_epochs, args)
        last_curriculum = curriculum
        ranking_head.set_prior_scale(float(curriculum.get("prior_scale", 0.12)))
        for m in modules_snap.values():
            m.train()
        total_expr = total_rank = total_shift = 0.0
        for _step in range(args.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            cell_emb, _, gene_emb = utils.encode_joint_embeddings(
                use_amp, cell_features_t, cell_graph, X_feature_t_sparse_gpu, gene_graph,
                cell_encoder, gene_feature_projector, gene_encoder)
            with autocast(device_type="cuda", enabled=use_amp):
                expr_loss, _, _ = utils.compute_recon_forward(
                    args, cell_emb, gene_emb, gene_bias, cell_log_sf_t, lib_scale, recon_bilinear_weight,
                    gene_log_theta, gene_logit_pi, X_recon, X_recon_csc, n_cells, n_genes, rng,
                    cell_indices_override=sorted(set(src_train) | set(tgt_train)))
                S_idx, T_idx = utils.sample_episode_batch(src_train, tgt_train, args.set_size, args.episode_batch_size, rng)
                z_S = set_encoder(utils.gather_cell_embeddings(cell_emb, S_idx))
                z_T = set_encoder(utils.gather_cell_embeddings(cell_emb, T_idx))
                scores, _, control_vecs = ranking_head(z_S, z_T, gene_emb)
                tier = utils.sample_negatives_hybrid(
                    scores, current_neg, current_pure_neg, current_downstream, args.num_neg_samples, rng, epoch,
                    args.hard_negative_start, args.hard_negative_fraction, args.hard_negative_pool_size, curriculum)
                rank_loss = utils.hybrid_ordinal_pu_ranking_loss(scores, driver_train_idx, tier["pure_neg_idx"], tier["downstream_idx"], curriculum)
                shift_loss = utils.compute_shift_loss(scores, gene_emb, control_vecs, z_T - z_S,
                                                      topk=args.shift_topk, temperature=args.shift_temperature,
                                                      control_weight=args.shift_control_weight,
                                                      gene_set_weight=args.shift_gene_set_weight,
                                                      gene_align_weight=args.shift_gene_align_weight)
                total_loss = recon_weight * expr_loss + rank_weight * rank_loss + shift_weight * shift_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_expr += float(expr_loss.item())
            total_rank += float(rank_loss.item())
            total_shift += float(shift_loss.item())

        train_expr = total_expr / max(1, args.steps_per_epoch)
        train_rank = total_rank / max(1, args.steps_per_epoch)
        train_shift = total_shift / max(1, args.steps_per_epoch)
        train_total = recon_weight * train_expr + rank_weight * train_rank + shift_weight * train_shift
        if math.isnan(train_total) or math.isinf(train_total):
            print(f"NaN/Inf in training loss at epoch {epoch}; aborting.", flush=True)
            break

        for m in modules_snap.values():
            m.eval()
        with torch.no_grad():
            cell_emb_val, _, gene_emb_val = utils.encode_joint_embeddings(
                use_amp, cell_features_t, cell_graph, X_feature_t_sparse_gpu, gene_graph,
                cell_encoder, gene_feature_projector, gene_encoder)
            vm = compute_validation_metrics(
                args, rng, use_amp, device, cell_emb_val, gene_emb_val, set_encoder, ranking_head,
                gene_bias, cell_log_sf_t, lib_scale, recon_bilinear_weight, gene_log_theta, gene_logit_pi,
                X_recon, X_recon_csc, n_cells, n_genes, src_val, tgt_val, driver_val_idx,
                current_neg, current_pure_neg, current_downstream, curriculum, epoch)
            val_total = recon_weight * vm["val_expr"] + rank_weight * vm["val_rank"] + shift_weight * vm["val_shift"]
            relation_vec = utils.compute_positive_relation_scores(gene_emb_val, true_driver_idx_all, affected_idx_all)
            pseudo_bank, pseudo_score_ema, pseudo_hit_counts, _ = utils.update_pseudo_positive_bank(
                args, epoch, vm["score_mean_vec"], relation_vec, true_driver_idx_all, affected_idx_all,
                pseudo_score_ema, pseudo_hit_counts, holdout_protect_idx=holdout_protect_idx)
            pools = utils.build_hybrid_tier_pools(n_genes, true_driver_idx_all, downstream_idx_all,
                                                  pseudo_bank_idx=pseudo_bank, holdout_protect_idx=holdout_protect_idx)
            current_neg = np.asarray(pools["neg_candidates"], dtype=np.int64)
            current_pure_neg = np.asarray(pools["pure_neg_candidates"], dtype=np.int64)
            current_downstream = list(pools["downstream_idx_all"])
            utils.save_unknown_driver_candidates(out_dir / "unknown_driver_candidates.csv", data["gene_names"],
                                                 vm["score_mean_vec"], relation_vec, true_driver_idx_all, affected_idx_all,
                                                 pseudo_bank, pseudo_hit_counts, pseudo_score_ema, topk=args.save_unknown_driver_candidates_topk)
            epoch_one_based = epoch + 1
            prune_stats = []
            if args.use_cell_graph_pruning and utils.should_prune(epoch_one_based, args.cell_prune_warmup, args.cell_prune_interval):
                prune_stats.append(utils.prune_graph_edges(cell_graph, cell_emb_val, args.cell_prune_percentile,
                                                           args.cell_max_prune_rate, args.cell_min_degree, args.prune_ema_decay))
            if args.use_gene_graph_pruning and utils.should_prune(epoch_one_based, args.gene_prune_warmup, args.gene_prune_interval):
                prune_stats.append(utils.prune_graph_edges(gene_graph, gene_emb_val, args.gene_prune_percentile,
                                                           args.gene_max_prune_rate, args.gene_min_degree, args.prune_ema_decay))
            if prune_stats:
                with open(pruning_log_path, "a") as f:
                    for ps in prune_stats:
                        f.write(f"{epoch},{ps['graph']},{ps['before_edges']},{ps['after_edges']},{ps['pruned_edges']},{ps['threshold']:.6f},{ps['avg_score']:.6f}\n")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{schedule['stage']},{recon_weight:.6f},{rank_weight:.6f},{shift_weight:.6f},"
                    f"{train_expr:.6f},{train_rank:.6f},{train_shift:.6f},{train_total:.6f},"
                    f"{vm['val_expr']:.6f},{vm['val_rank']:.6f},{vm['val_shift']:.6f},{val_total:.6f},"
                    f"{vm['pearson_nz']:.6f},{vm['auprc']:.6f},{vm['prec_at_k']:.6f},{vm['early_stop_auprc']:.6f},"
                    f"{len(current_pure_neg)},{len(pseudo_bank)},{cell_graph.num_edges},{gene_graph.num_edges}\n")
        print(f"Epoch {epoch:3d} | {schedule['stage']} | train rank={train_rank:.4f} shift={train_shift:.4f} | "
              f"val auprc={vm['auprc']:.4f} prec@K={vm['prec_at_k']:.4f} es_auprc={vm['early_stop_auprc']:.4f} | "
              f"pools(pure={len(current_pure_neg)}, down={len(current_downstream)}, pseudo={len(pseudo_bank)})", flush=True)

        es_metric = vm["early_stop_auprc"]
        if not math.isnan(es_metric) and stage_priority.get(schedule["stage"], 1) >= min_stage:
            if es_metric > best_metric:
                best_metric, best_epoch, no_improve = es_metric, epoch, 0
                best_snapshot = snapshot_state_dicts(modules_snap, extras_snap)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}. Best AUPRC {best_metric:.4f} (epoch {best_epoch}).", flush=True)
                    break

    if best_snapshot is not None:
        print(f"Restoring best checkpoint from epoch {best_epoch} (AUPRC={best_metric:.4f}).", flush=True)
        restore_state_dicts(modules_snap, extras_snap, best_snapshot, device)

    for m in modules_snap.values():
        m.eval()
    if driver_prior_model is not None:
        driver_prior_model.eval()

    # ---- save weights -----------------------------------------------
    weights_dir = out_dir / "model_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cell_encoder.state_dict(), weights_dir / "cell_encoder.pt")
    torch.save(gene_feature_projector.state_dict(), weights_dir / "gene_feature_projector.pt")
    torch.save(gene_encoder.state_dict(), weights_dir / "gene_encoder.pt")
    torch.save(set_encoder.state_dict(), weights_dir / "set_encoder.pt")
    torch.save(ranking_head.state_dict(), weights_dir / "ranking_head.pt")
    if driver_prior_model is not None:
        torch.save(driver_prior_model.state_dict(), weights_dir / "driver_prior_model.pt")
        (weights_dir / "driver_prior_metadata.json").write_text(json.dumps(driver_prior_model.export_state(), indent=2))
    torch.save({"gene_bias": gene_bias.detach().cpu(), "lib_scale": lib_scale.detach().cpu(),
                "recon_bilinear_weight": recon_bilinear_weight.detach().cpu(),
                "gene_log_theta": None if gene_log_theta is None else gene_log_theta.detach().cpu(),
                "gene_logit_pi": None if gene_logit_pi is None else gene_logit_pi.detach().cpu()},
               weights_dir / "recon_params.pt")
    (out_dir / "final_graph_stats.json").write_text(json.dumps(
        {"cell_edges_final": int(cell_graph.num_edges), "gene_edges_final": int(gene_graph.num_edges),
         "best_epoch": int(best_epoch), "best_early_stop_auprc": float(best_metric)}, indent=2))

    # ---- test inference & predictions -------------------------------
    ranking_head.set_prior_scale(float(last_curriculum.get("prior_scale", 0.12)))
    scores_np, cell_emb_np, gene_emb_np = run_test_inference(
        args, rng, use_amp, device, cell_features_t, cell_graph, X_feature_t_sparse_gpu, gene_graph,
        cell_encoder, gene_feature_projector, gene_encoder, set_encoder, ranking_head, src_test, tgt_test)
    metrics = utils.evaluate_ranking(scores_np, splits["driver_test_idx"], K=10)
    (out_dir / "test_metrics.txt").write_text("".join(f"{k}: {v:.6f}\n" for k, v in metrics.items()))
    np.savez_compressed(out_dir / "predictions.npz", scores=scores_np, gene_names=np.asarray(data["gene_names"]))
    order = np.argsort(-scores_np)
    with open(out_dir / "predictions.csv", "w") as f:
        f.write("gene,score\n")
        for i in order:
            f.write(f"{data['gene_names'][i]},{scores_np[i]:.6f}\n")

    relation_final = utils.compute_positive_relation_scores(
        torch.from_numpy(gene_emb_np).to(device), true_driver_idx_all, affected_idx_all)
    utils.save_unknown_driver_candidates(out_dir / "unknown_driver_candidates_final.csv", data["gene_names"],
                                         scores_np.astype(np.float32), relation_final, true_driver_idx_all, affected_idx_all,
                                         pseudo_bank, pseudo_hit_counts, pseudo_score_ema, topk=args.save_unknown_driver_candidates_topk)
    print(f"Training complete. Results in {out_dir} | test: {metrics}", flush=True)

    # ---- embedding / UMAP export ------------------------------------
    if not args.skip_embedding_export:
        try:
            import export_embeddings
            export_embeddings.export(out_dir=out_dir, gene_names=list(data["gene_names"]), obs_names=list(data["obs_names"]),
                                     sample_col=np.asarray(data["sample_col"]), scores_np=scores_np,
                                     cell_emb=cell_emb_np, gene_emb=gene_emb_np,
                                     source_cell_type=args.source_cell_type, target_cell_type=args.target_cell_type,
                                     driver_idx=true_driver_idx_all, seed=int(args.seed),
                                     n_neighbors=int(args.embedding_umap_neighbors), min_dist=float(args.embedding_umap_min_dist))
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: embedding export failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
