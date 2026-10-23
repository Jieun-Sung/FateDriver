"""
FateDriver neural network components.

This module defines the four learnable sub-networks of FateDriver plus the
reconstruction decoder utilities:

    * ``SparseGraphEncoder``          - message-passing encoder used for BOTH the
                                        cell graph and the gene graph (GCN).
    * ``GeneFeatureProjector``        - lifts genes into the shared latent space
                                        via a sparse cell x gene projection.
    * ``AttentiveSetEncoder``         - pools a set of cells (source or target)
                                        into a single state vector (Set-Transformer
                                        style pooling by multi-head attention).
    * ``CrossAttentionRankingHead``   - scores every gene for the source -> target
                                        transition, optionally combined with the
                                        learnable transition-driver prior.
    * ``LearnableTransitionDriverPrior`` - a small learnable weighting over
                                        interpretable transition features.

All reconstruction losses (MSE / NB / ZINB) are implemented at the bottom; the
paper configuration uses ZINB with an exponential decoder activation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Sparse graph container
# =====================================================================

@dataclass
class SparseGraphData:
    """Coalesced directed sparse graph (messages flow src -> dst)."""
    num_nodes: int
    src: torch.Tensor
    dst: torch.Tensor
    edge_weight: torch.Tensor
    name: str
    score_ema: Optional[torch.Tensor] = None

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())


# =====================================================================
# Small helpers
# =====================================================================

def resolve_num_heads(hidden_dim: int, requested_heads: int) -> int:
    """Largest divisor of ``hidden_dim`` that is <= ``requested_heads``."""
    h = max(1, int(requested_heads))
    while hidden_dim % h != 0 and h > 1:
        h -= 1
    return h


def module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# =====================================================================
# Graph encoder (GCN)
# =====================================================================

class SparseGraphLayer(nn.Module):
    """One GCN-style layer.

    NOTE: this is a row-stochastic mean aggregation with a self-loop residual,
    not the symmetric Kipf-Welling normalisation. It is deliberately kept as in
    the paper implementation.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lin = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, graph: SparseGraphData) -> torch.Tensor:
        n = x.shape[0]
        h = self.lin(x)
        if graph.num_edges == 0:
            out = h
        else:
            edge_w = graph.edge_weight.to(device=h.device, dtype=h.dtype)
            msg = h[graph.src] * edge_w.unsqueeze(-1)
            out = torch.zeros((n, h.shape[1]), device=h.device, dtype=h.dtype)
            out.index_add_(0, graph.dst, msg)
            deg = torch.zeros((n,), device=h.device, dtype=h.dtype)
            deg.index_add_(0, graph.dst, edge_w)
            out = (out + h) / (deg.unsqueeze(-1) + 1.0)
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class SparseGraphEncoder(nn.Module):
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, graph: SparseGraphData) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, graph)
        return h


def make_graph_encoder(in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
    num_layers = max(1, int(num_layers))
    dims = [in_dim] + [hidden_dim] * num_layers
    layers = [SparseGraphLayer(in_dim=dims[i], hidden_dim=hidden_dim, dropout=dropout) for i in range(num_layers)]
    return SparseGraphEncoder(layers)


# =====================================================================
# Gene feature projector
# =====================================================================

class GeneFeatureProjector(nn.Module):
    """Initial gene embedding = learnable table + (cell x gene)^T @ cell_emb."""

    def __init__(self, n_genes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gene_init_emb = nn.Parameter(torch.empty(n_genes, hidden_dim))
        nn.init.xavier_uniform_(self.gene_init_emb)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.sparse_chunk_size = 2048

    def forward(self, x_t_sparse: Union[torch.Tensor, sp.spmatrix], cell_emb: torch.Tensor) -> torch.Tensor:
        if sp.issparse(x_t_sparse):
            chunk_size = max(1, int(getattr(self, "sparse_chunk_size", 2048)))
            mat = x_t_sparse.tocsr()
            parts = []
            for start in range(0, mat.shape[0], chunk_size):
                chunk = mat[start:start + chunk_size].tocoo()
                idx_np = np.vstack([chunk.row, chunk.col]).astype(np.int64, copy=False)
                idx_t = torch.from_numpy(idx_np).to(device=cell_emb.device, dtype=torch.long, non_blocking=True)
                val_t = torch.from_numpy(chunk.data.astype(np.float32, copy=False)).to(device=cell_emb.device, dtype=torch.float32, non_blocking=True)
                chunk_t = torch.sparse_coo_tensor(idx_t, val_t, chunk.shape, device=cell_emb.device).coalesce()
                if cell_emb.device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        part = torch.sparse.mm(chunk_t, cell_emb.float())
                else:
                    part = torch.sparse.mm(chunk_t, cell_emb.float())
                parts.append(part.to(device=self.gene_init_emb.device, dtype=self.gene_init_emb.dtype, non_blocking=True))
                del chunk_t, idx_t, val_t, part
            expr_feat = torch.cat(parts, dim=0) if parts else self.gene_init_emb.new_zeros(self.gene_init_emb.shape)
            out = expr_feat + self.gene_init_emb
            return self.dropout(self.norm(out))
        if x_t_sparse.device != cell_emb.device:
            cell_emb = cell_emb.to(device=x_t_sparse.device, dtype=cell_emb.dtype, non_blocking=True)
        if cell_emb.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=False):
                expr_feat = torch.sparse.mm(x_t_sparse, cell_emb.float())
        else:
            expr_feat = torch.sparse.mm(x_t_sparse, cell_emb.float())
        if expr_feat.device != self.gene_init_emb.device:
            expr_feat = expr_feat.to(device=self.gene_init_emb.device, dtype=expr_feat.dtype, non_blocking=True)
        out = expr_feat + self.gene_init_emb
        return self.dropout(self.norm(out))


# =====================================================================
# Attentive set encoder (source / target cell-set pooling)
# =====================================================================

class AttentiveSetEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, num_heads: int):
        super().__init__()
        attn_heads = resolve_num_heads(hidden_dim, num_heads)
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.seed = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.seed)
        self.pma = nn.MultiheadAttention(hidden_dim, attn_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_set: torch.Tensor) -> torch.Tensor:
        h = self.phi(x_set)
        seed = self.seed.expand(h.shape[0], -1, -1)
        pooled, _ = self.pma(query=seed, key=h, value=h, need_weights=False)
        pooled = self.norm1(pooled + seed)
        pooled = self.norm2(pooled + self.ff(pooled))
        return self.rho(pooled.squeeze(1))


# =====================================================================
# Learnable transition-driver prior
# =====================================================================

class LearnableTransitionDriverPrior(nn.Module):
    """Learns a convex weighting over interpretable transition features and adds
    the resulting prior to the raw gene scores."""

    def __init__(self, feature_matrix: torch.Tensor, feature_names: List[str], init_weights: np.ndarray, init_bias: float = 0.0):
        super().__init__()
        if feature_matrix.dim() != 2:
            raise ValueError(f"feature_matrix must be 2D, got shape={tuple(feature_matrix.shape)}")
        n_features = int(feature_matrix.shape[1])
        if n_features == 0:
            raise ValueError("feature_matrix must include at least one prior feature")
        init_w = np.asarray(init_weights, dtype=np.float32).reshape(-1)
        if init_w.size != n_features:
            raise ValueError(f"init_weights size {init_w.size} does not match n_features={n_features}")
        init_w = np.clip(init_w, 1e-6, None)
        init_w = init_w / float(init_w.sum())
        self.register_buffer("feature_matrix", feature_matrix)
        self.feature_names = tuple(feature_names)
        self.feature_logits = nn.Parameter(torch.log(torch.from_numpy(init_w).to(device=feature_matrix.device, dtype=torch.float32)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias), device=feature_matrix.device, dtype=torch.float32))

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.feature_logits, dim=0)

    def compute_prior(self) -> torch.Tensor:
        return torch.matmul(self.feature_matrix, self.normalized_weights()) + self.bias

    def apply(self, scores: torch.Tensor, prior_scale: float = 0.12) -> torch.Tensor:
        prior = self.compute_prior().to(device=scores.device, dtype=scores.dtype)
        if scores.dim() == 1:
            return scores + float(prior_scale) * prior
        return scores + float(prior_scale) * prior.unsqueeze(0)

    def export_state(self) -> Dict[str, object]:
        weights = self.normalized_weights().detach().cpu().numpy().astype(np.float32)
        prior = self.compute_prior().detach().cpu().numpy().astype(np.float32)
        return {
            "feature_names": list(self.feature_names),
            "normalized_weights": {name: float(w) for name, w in zip(self.feature_names, weights.tolist())},
            "bias": float(self.bias.detach().cpu().item()),
            "prior_mean": float(prior.mean()) if prior.size > 0 else 0.0,
            "prior_std": float(prior.std()) if prior.size > 0 else 0.0,
            "prior_min": float(prior.min()) if prior.size > 0 else 0.0,
            "prior_max": float(prior.max()) if prior.size > 0 else 0.0,
        }


# =====================================================================
# Cross-attention ranking head
# =====================================================================

class CrossAttentionRankingHead(nn.Module):
    """Scores every gene for the source -> target transition.

    A gene score combines (i) a base alignment between a learned control vector
    and the gene embedding and (ii) a cross-attention term between the gene and
    the [z_S, z_T, z_T - z_S] transition tokens. The optional learnable prior is
    added on top when a prior module has been registered and prior is enabled.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, chunk_size: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.num_heads = resolve_num_heads(hidden_dim, num_heads)
        self.head_dim = hidden_dim // self.num_heads
        self.chunk_size = max(32, int(chunk_size))
        self.gene_self_attn = nn.MultiheadAttention(hidden_dim, self.num_heads, dropout=dropout, batch_first=True)
        self.gene_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.control_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_score = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.base_scale = nn.Parameter(torch.tensor(1.0))
        self.cross_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.cross_score)
        self.driver_prior: Optional[LearnableTransitionDriverPrior] = None
        self._prior_enabled = True
        self._prior_scale = 0.12

    def set_prior_module(self, prior_module: Optional[LearnableTransitionDriverPrior]) -> None:
        self.driver_prior = prior_module

    def enable_prior(self, flag: bool) -> None:
        self._prior_enabled = bool(flag)

    def set_prior_scale(self, scale: float) -> None:
        self._prior_scale = float(scale)

    def _contextualize_genes(self, gene_emb: torch.Tensor) -> torch.Tensor:
        n_genes = gene_emb.shape[0]
        if n_genes <= max(self.chunk_size * 4, 2048):
            ctx, _ = self.gene_self_attn(gene_emb.unsqueeze(0), gene_emb.unsqueeze(0), gene_emb.unsqueeze(0), need_weights=False)
            return self.gene_norm(ctx.squeeze(0) + gene_emb)
        global_ctx = gene_emb.mean(dim=0, keepdim=True).unsqueeze(0)
        chunks = []
        for start in range(0, n_genes, self.chunk_size):
            end = min(n_genes, start + self.chunk_size)
            chunk = gene_emb[start:end].unsqueeze(0)
            kv = torch.cat([chunk, global_ctx], dim=1)
            ctx, _ = self.gene_self_attn(chunk, kv, kv, need_weights=False)
            chunks.append(self.gene_norm(ctx.squeeze(0) + gene_emb[start:end]))
        return torch.cat(chunks, dim=0)

    def _score_genes(self, z_S: torch.Tensor, z_T: torch.Tensor, gene_emb: torch.Tensor,
                     gene_ctx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        transition = z_T - z_S
        tokens_raw = torch.stack([z_S, z_T, transition], dim=1)
        tokens = self.token_norm(tokens_raw)
        control = self.control_mlp(torch.cat([z_S, z_T, transition], dim=-1))
        control = F.normalize(control, dim=-1)
        if gene_ctx is None:
            gene_ctx = self._contextualize_genes(gene_emb)
        gene_n = F.normalize(gene_ctx, dim=-1)

        k = self.key_proj(tokens).view(tokens.shape[0], tokens.shape[1], self.num_heads, self.head_dim)
        v = self.value_proj(tokens).view(tokens.shape[0], tokens.shape[1], self.num_heads, self.head_dim)
        token_scalar = torch.einsum("bmhd,hd->bmh", v, self.cross_score)
        q_all = self.query_proj(gene_ctx).view(gene_ctx.shape[0], self.num_heads, self.head_dim)

        base_chunks: List[torch.Tensor] = []
        cross_chunks: List[torch.Tensor] = []
        score_chunks: List[torch.Tensor] = []
        for start in range(0, gene_ctx.shape[0], self.chunk_size):
            end = min(gene_ctx.shape[0], start + self.chunk_size)
            q_chunk = q_all[start:end]
            gene_chunk = gene_n[start:end]
            base_scores = torch.matmul(control, gene_chunk.T)
            attn_logits = torch.einsum("chd,bmhd->bchm", q_chunk, k) / math.sqrt(float(self.head_dim))
            attn = torch.softmax(attn_logits, dim=-1)
            attn_for_score = F.dropout(attn, p=self.dropout, training=self.training)
            cross_scores = torch.einsum("bchm,bmh->bc", attn_for_score, token_scalar) / float(self.num_heads)
            scores = self.base_scale * base_scores + self.cross_scale * cross_scores
            base_chunks.append(base_scores)
            cross_chunks.append(cross_scores)
            score_chunks.append(scores)

        scores_all = torch.cat(score_chunks, dim=1) + self.score_bias
        if self.driver_prior is not None and self._prior_enabled:
            scores_all = self.driver_prior.apply(scores_all, prior_scale=self._prior_scale)
        return {
            "scores": scores_all,
            "probs": torch.sigmoid(scores_all),
            "control": control,
            "gene_contextualized": gene_ctx,
        }

    def forward(self, z_S: torch.Tensor, z_T: torch.Tensor, gene_emb: torch.Tensor,
                gene_ctx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self._score_genes(z_S, z_T, gene_emb, gene_ctx=gene_ctx)
        return out["scores"], out["probs"], out["control"]


# =====================================================================
# Reconstruction decoder utilities (MSE / NB / ZINB)
# =====================================================================

def apply_decoder_activation(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(x)
    if kind == "softplus":
        return F.softplus(x)
    if kind == "exp":
        return torch.exp(torch.clamp(x, max=20.0))
    if kind == "identity":
        return x
    raise ValueError(f"Unknown decoder activation: {kind}")


def decode_recon_logits(z: torch.Tensor, e: torch.Tensor, gene_bias: torch.Tensor, cell_log_sf: torch.Tensor,
                        lib_scale: torch.Tensor, bilinear_weight: torch.Tensor) -> torch.Tensor:
    base = torch.matmul(torch.matmul(z, bilinear_weight), e.T)
    base = base + gene_bias.unsqueeze(0)
    base = base + lib_scale * cell_log_sf.unsqueeze(1)
    return base


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = mask.to(dtype=x.dtype)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return masked_mean((pred - target) ** 2, mask)


def nb_elementwise_nll(k: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    t = theta
    res = (t * (torch.log(t + 1e-8) - torch.log(t + mu + 1e-8))
           + k * (torch.log(mu + 1e-8) - torch.log(t + mu + 1e-8))
           + torch.lgamma(k + t) - torch.lgamma(t) - torch.lgamma(k + 1.0))
    return -res


def zinb_elementwise_nll(k: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    nb_ll = -nb_elementwise_nll(k, mu, theta)
    zero = (k <= 0.0).to(dtype=mu.dtype)
    logit = torch.log(pi + 1e-8) - torch.log(1.0 - pi + 1e-8)
    log_pi = -F.softplus(-logit)
    log_1m_pi = -F.softplus(logit)
    ll_zero = torch.logaddexp(log_pi, log_1m_pi + nb_ll)
    ll_nonzero = log_1m_pi + nb_ll
    ll = zero * ll_zero + (1.0 - zero) * ll_nonzero
    return -ll
