#!/usr/bin/env python3
"""
train.py
────────
Train a GATv2-based graph encoder with supervised contrastive loss (SupCon).

The encoder maps each variable-size graph to a 256-dim L2-normalised embedding.
Positive pairs are defined by a custom mask:
  - Both arcs:      positive if |angle_i - angle_j| <= DELTA_DEG
  - Both non-arcs:  positive if motif_type matches exactly
  - Otherwise:      negative

Reads:   processed/train.pt, processed/val.pt, processed/test.pt
         processed/meta.json

Usage
─────
    # dry-run (50 steps, diagnostics only)
    python train.py --dry-run

    # full training
    python train.py --epochs 100 --batch-size 256 --lr 1e-3

    # custom delta for arc positives
    python train.py --delta-deg 20 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SupCon training for shape embeddings")
    p.add_argument("--processed-dir", type=str, default="processed")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--gat-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--delta-deg", type=float, default=10.0,
                   help="Arc angle tolerance for positive pairs")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--eval-k", type=int, default=5,
                   help="K for retrieval@K evaluation")
    p.add_argument("--dry-run", action="store_true",
                   help="Run 50 steps only, print diagnostics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

class ShapeEncoder(nn.Module):
    """
    GATv2 graph encoder -> L2-normalised embedding.

    Architecture:
        node MLP -> [GATv2 + residual] x L
        -> AttentionalAggregation pool
        -> concat global features (num_nodes, total_bend)
        -> projection MLP -> L2 norm
    """

    def __init__(self, in_dim: int = 3, hidden_dim: int = 128,
                 embed_dim: int = 256, heads: int = 4,
                 num_layers: int = 4, dropout: float = 0.1,
                 n_global_feats: int = 2):
        super().__init__()
        self.n_global_feats = n_global_feats

        # node encoder
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATv2Conv(in_c, hidden_dim, heads=heads,
                          dropout=dropout, add_self_loops=True)
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim * heads))

        # residual projections (for dimension mismatch on first layer)
        self.res_projs = nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_dim if i == 0 else hidden_dim * heads
            out_c = hidden_dim * heads
            if in_c != out_c:
                self.res_projs.append(nn.Linear(in_c, out_c, bias=False))
            else:
                self.res_projs.append(nn.Identity())

        # attentional aggregation pooling (replaces deprecated GlobalAttention)
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pool = AttentionalAggregation(gate_nn)

        # projection head: pool_dim + global features -> embed_dim
        pool_dim = hidden_dim * heads + n_global_feats
        self.proj = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim * heads),
            nn.ReLU(),
            nn.Linear(hidden_dim * heads, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ── node encode ──
        h = self.node_enc(x)

        # ── GAT layers with residual ──
        for gat, norm, res_proj in zip(self.gat_layers, self.gat_norms, self.res_projs):
            residual = res_proj(h)
            h = gat(h, edge_index)
            h = norm(h)
            h = F.elu(h + residual)
            h = self.dropout(h)

        # ── pool ──
        g = self.pool(h, index=batch)  # (B, hidden*heads)

        # ── global features: num_nodes + total_bend per graph ──
        # bend_rad is x[:, 1] (second feature column)
        bend_rad = x[:, 1]
        B = g.size(0)
        num_nodes = torch.zeros(B, device=g.device)
        total_bend = torch.zeros(B, device=g.device)
        num_nodes.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        total_bend.scatter_add_(0, batch, bend_rad)

        # normalise global features for stable training
        num_nodes_norm = num_nodes / 20.0        # ~10-20 nodes, so /20 puts in [0.5, 1]
        total_bend_norm = total_bend / math.pi   # total bend in radians, /pi normalises

        global_feats = torch.stack([num_nodes_norm, total_bend_norm], dim=1)  # (B, 2)
        g = torch.cat([g, global_feats], dim=1)  # (B, hidden*heads + 2)

        # ── project + L2 normalise ──
        z = self.proj(g)
        z = F.normalize(z, p=2, dim=-1)
        return z


# ═══════════════════════════════════════════════════════════════════════
# SUPCON LOSS WITH CUSTOM POSITIVE MASK
# ═══════════════════════════════════════════════════════════════════════

def build_positive_mask(motif_ids: torch.Tensor,
                        arc_angles: torch.Tensor,
                        delta_deg: float,
                        arc_motif_id: int) -> torch.Tensor:
    """
    Build a (B, B) boolean positive-pair mask.

    Rules:
      - Both arcs (motif_id==arc_motif_id): positive if |angle_i - angle_j| <= delta_deg
      - Both non-arcs: positive if motif_type matches exactly
      - Mixed arc/non-arc: negative
      - Self-pairs (diagonal): False
    """
    B = len(motif_ids)
    is_arc = (motif_ids == arc_motif_id)

    # expand for pairwise comparison
    is_arc_i = is_arc.unsqueeze(1).expand(B, B)
    is_arc_j = is_arc.unsqueeze(0).expand(B, B)
    both_arc = is_arc_i & is_arc_j

    # arc angle closeness
    angle_i = arc_angles.float().unsqueeze(1).expand(B, B)
    angle_j = arc_angles.float().unsqueeze(0).expand(B, B)
    arc_close = (angle_i - angle_j).abs() <= delta_deg

    # non-arc motif match
    motif_i = motif_ids.unsqueeze(1).expand(B, B)
    motif_j = motif_ids.unsqueeze(0).expand(B, B)
    both_nonarc = (~is_arc_i) & (~is_arc_j)
    nonarc_match = both_nonarc & (motif_i == motif_j)

    # combine
    mask = (both_arc & arc_close) | nonarc_match

    # remove diagonal
    mask.fill_diagonal_(False)
    return mask


def supcon_loss(embeddings: torch.Tensor,
                pos_mask: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
    """
    Supervised contrastive loss (SupCon).

    For each anchor i, the loss pushes positives closer and negatives apart
    in the embedding space using the InfoNCE formulation.
    """
    B = embeddings.size(0)
    device = embeddings.device

    # cosine similarity (embeddings are already L2-normalised)
    sim = embeddings @ embeddings.T  # (B, B)
    sim = sim / temperature

    # mask out self-similarity
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim.masked_fill_(self_mask, -1e9)

    # for numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # exp similarities
    exp_sim = sim.exp()

    # denominator: sum over all non-self
    denom = exp_sim.masked_fill(self_mask, 0.0).sum(dim=1, keepdim=True)  # (B, 1)

    # log prob for positives
    log_prob = sim - denom.log()  # (B, B)

    # average over positives for each anchor
    pos_mask_float = pos_mask.float()
    n_pos = pos_mask_float.sum(dim=1)  # (B,)

    # skip anchors with zero positives
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob = (log_prob * pos_mask_float).sum(dim=1) / n_pos.clamp(min=1)
    loss = -mean_log_prob[valid].mean()
    return loss


# ═══════════════════════════════════════════════════════════════════════
# BALANCED BATCH SAMPLER
# ═══════════════════════════════════════════════════════════════════════

class BalancedBatchSampler:
    """
    Motif-balanced batch sampler.

    Groups are (motif_type_id, arc_angle_deg).  Sampling probability is
    weighted so that every **motif type** gets equal representation,
    regardless of how many angle-subgroups it has.

    e.g. "arc" has 17 groups but "junction_Y" has 1 group.
    Without balancing, arc dominates; with balancing, each motif
    type is equally likely to be picked per slot.
    """

    def __init__(self, data_list: List[Data], batch_size: int,
                 samples_per_group: int = 8, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.samples_per_group = samples_per_group

        # group indices by (motif_type_id, arc_angle_deg)
        self.groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx, d in enumerate(data_list):
            key = (d.motif_type_id, d.arc_angle_deg)
            self.groups[key].append(idx)

        self.group_keys = list(self.groups.keys())
        self.n = len(data_list)

        # motif-balanced sampling weights:
        # each group's probability = 1 / (n_motifs * n_groups_for_that_motif)
        motif_to_keys: Dict[int, List] = defaultdict(list)
        for k in self.group_keys:
            motif_to_keys[k[0]].append(k)

        weights = np.array([
            1.0 / len(motif_to_keys[k[0]]) for k in self.group_keys
        ], dtype=np.float64)
        self.group_probs = weights / weights.sum()

    def __iter__(self):
        n_groups = max(1, self.batch_size // self.samples_per_group)
        n_batches = self.n // self.batch_size

        for _ in range(n_batches):
            # pick groups with motif-balanced probabilities
            chosen_keys = self.rng.choice(
                len(self.group_keys), size=n_groups, replace=True,
                p=self.group_probs,
            )
            batch_indices = []
            for ki in chosen_keys:
                key = self.group_keys[ki]
                pool = self.groups[key]
                chosen = self.rng.choice(
                    pool,
                    size=min(self.samples_per_group, len(pool)),
                    replace=len(pool) < self.samples_per_group,
                )
                batch_indices.extend(chosen.tolist())

            # trim or pad to exact batch_size
            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[:self.batch_size]
            elif len(batch_indices) < self.batch_size:
                extra = self.rng.choice(self.n, size=self.batch_size - len(batch_indices))
                batch_indices.extend(extra.tolist())

            yield batch_indices

    def __len__(self):
        return self.n // self.batch_size


# ═══════════════════════════════════════════════════════════════════════
# STRATIFIED VAL SUBSET
# ═══════════════════════════════════════════════════════════════════════

def stratified_val_subset(data_list: List[Data],
                          max_total: int = 2000,
                          seed: int = 42) -> List[Data]:
    """
    Pick a stratified random subset from val, ensuring every
    (motif_type_id, arc_angle_deg) group is represented equally.
    """
    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, d in enumerate(data_list):
        groups[(d.motif_type_id, d.arc_angle_deg)].append(idx)

    n_groups = len(groups)
    per_group = max(1, max_total // n_groups)

    indices = []
    for key in sorted(groups):
        pool = np.array(groups[key])
        rng.shuffle(pool)
        indices.extend(pool[:per_group].tolist())

    rng.shuffle(indices)
    return [data_list[i] for i in indices[:max_total]]


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION: RETRIEVAL@K + ARC ANGLE ERROR
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_retrieval(model: ShapeEncoder,
                       data_list: List[Data],
                       delta_deg: float,
                       arc_motif_id: int,
                       k: int = 5,
                       batch_size: int = 512,
                       device: str = "cpu") -> Dict[str, float]:
    """
    Retrieval@K evaluation + mean arc angle error.

    For each query graph, find its K nearest neighbours in embedding space.
    A retrieved neighbour is correct if:
      - Both arcs: |angle_query - angle_retrieved| <= delta_deg
      - Both non-arcs: motif_type matches
      - Mixed: incorrect

    Also computes mean |delta_angle| for arc-to-arc retrievals.
    """
    model.eval()
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    all_embeds = []
    all_motif_ids = []
    all_angles = []

    for batch in loader:
        batch = batch.to(device)
        z = model(batch)
        all_embeds.append(z.cpu())
        all_motif_ids.append(batch.motif_type_id.cpu() if hasattr(batch, 'motif_type_id')
                             else torch.tensor([d.motif_type_id for d in batch.to_data_list()]))
        all_angles.append(batch.arc_angle_deg.cpu() if hasattr(batch, 'arc_angle_deg')
                          else torch.tensor([d.arc_angle_deg for d in batch.to_data_list()]))

    embeds = torch.cat(all_embeds, dim=0)          # (N, D)
    motif_ids = torch.cat(all_motif_ids, dim=0)    # (N,)
    angles = torch.cat(all_angles, dim=0)           # (N,)
    N = len(embeds)

    # pairwise cosine similarity
    sim = embeds @ embeds.T
    sim.fill_diagonal_(-1e9)  # exclude self

    # top-K
    _, topk_idx = sim.topk(k, dim=1)  # (N, K)

    # check correctness
    is_arc = (motif_ids == arc_motif_id)
    hits = 0
    total = 0

    # arc angle error tracking (all arc-arc pairs in top-K)
    arc_angle_errors = []

    # per-query mean |delta_angle| for arc queries (stronger metric)
    arc_query_mean_deltas = []

    # per-motif breakdown
    motif_hits: Dict[int, int] = defaultdict(int)
    motif_total: Dict[int, int] = defaultdict(int)

    for i in range(N):
        mid = motif_ids[i].item()
        query_arc_deltas = []

        for j_pos in range(k):
            j = topk_idx[i, j_pos].item()
            total += 1
            motif_total[mid] += 1

            if is_arc[i] and is_arc[j]:
                delta = abs(angles[i].item() - angles[j].item())
                arc_angle_errors.append(delta)
                query_arc_deltas.append(delta)
                if delta <= delta_deg:
                    hits += 1
                    motif_hits[mid] += 1
            elif (not is_arc[i]) and (not is_arc[j]):
                if motif_ids[i] == motif_ids[j]:
                    hits += 1
                    motif_hits[mid] += 1
            # else: mixed -> miss

        # per-query arc delta (only for arc queries with arc neighbours)
        if is_arc[i] and len(query_arc_deltas) > 0:
            arc_query_mean_deltas.append(float(np.mean(query_arc_deltas)))

    precision_at_k = hits / total if total > 0 else 0.0
    mean_arc_angle_err = float(np.mean(arc_angle_errors)) if arc_angle_errors else 0.0
    mean_arc_topk_delta = float(np.mean(arc_query_mean_deltas)) if arc_query_mean_deltas else 0.0

    return {
        "precision@k": precision_at_k,
        "k": k,
        "mean_arc_angle_err": mean_arc_angle_err,
        "mean_arc_topk_delta": mean_arc_topk_delta,
        "motif_precision": {
            mid: motif_hits[mid] / motif_total[mid]
            for mid in sorted(motif_total)
            if motif_total[mid] > 0
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# BATCH SANITY DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

def print_batch_diagnostics(motif_ids: torch.Tensor,
                            arc_angles: torch.Tensor,
                            pos_mask: torch.Tensor,
                            arc_motif_id: int,
                            id2motif: Dict[int, str]) -> None:
    """Print once-per-run sanity stats about the batch composition."""
    is_arc = (motif_ids == arc_motif_id)
    n_arc = is_arc.sum().item()
    n_nonarc = len(motif_ids) - n_arc

    print("\n── batch sanity check ──")
    print(f"  batch size     : {len(motif_ids)}")
    print(f"  arcs           : {n_arc}")
    print(f"  non-arcs       : {n_nonarc}")

    # unique motif types + counts
    unique_motifs, counts = motif_ids.unique(return_counts=True)
    for m, c in zip(unique_motifs.tolist(), counts.tolist()):
        name = id2motif.get(m, f"id={m}")
        print(f"    {name:>14s}: {c}")

    # arc angles present
    if n_arc > 0:
        arc_angle_vals = arc_angles[is_arc].unique().sort().values
        print(f"  arc angles     : {arc_angle_vals.tolist()}")
        # check for -1 bug
        if (arc_angle_vals == -1).any():
            print("  *** WARNING: arc samples have arc_angle_deg=-1! Preprocessing bug! ***")

    # positive mask stats
    pos_per_anchor = pos_mask.float().sum(dim=1)
    arc_pos = pos_per_anchor[is_arc].mean().item() if n_arc > 0 else 0
    nonarc_pos = pos_per_anchor[~is_arc].mean().item() if n_nonarc > 0 else 0
    print(f"  avg pos (arc)  : {arc_pos:.1f}")
    print(f"  avg pos (other): {nonarc_pos:.1f}")
    print(f"  avg pos (all)  : {pos_per_anchor.mean().item():.1f}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[cfg] device={device}  epochs={args.epochs}  batch_size={args.batch_size}")
    print(f"[cfg] delta_deg={args.delta_deg}  temperature={args.temperature}")
    if args.dry_run:
        print("[cfg] DRY RUN -- 50 steps only")

    # load data
    proc = Path(args.processed_dir)
    train_data = torch.load(proc / "train.pt", weights_only=False)
    val_data = torch.load(proc / "val.pt", weights_only=False)
    print(f"[data] train={len(train_data)}  val={len(val_data)}")

    with open(proc / "meta.json") as f:
        meta = json.load(f)
    id2motif = {v: k for k, v in meta["motif_type_to_id"].items()}

    # read arc motif id from meta (not hardcoded)
    arc_motif_id = meta["motif_type_to_id"]["arc"]
    print(f"[meta] arc motif_type_id = {arc_motif_id}")

    # model
    in_dim = train_data[0].x.size(1)
    model = ShapeEncoder(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        heads=args.heads,
        num_layers=args.gat_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params:,} trainable parameters")

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # balanced sampler + loader
    sampler = BalancedBatchSampler(train_data, batch_size=args.batch_size, seed=args.seed)
    train_loader = DataLoader(train_data, batch_sampler=sampler)

    # prepare stratified val subset (once)
    val_subset = stratified_val_subset(val_data, max_total=2000, seed=args.seed)
    print(f"[eval] stratified val subset: {len(val_subset)} samples")

    # training
    max_steps = 50 if args.dry_run else None
    global_step = 0
    did_diagnostics = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_avg_pos = 0.0

        t0 = time.time()

        for batch in train_loader:
            if max_steps and global_step >= max_steps:
                break

            batch = batch.to(device)
            z = model(batch)  # (B, embed_dim)

            # extract per-graph attributes
            motif_ids = batch.motif_type_id.to(device)
            arc_angles = batch.arc_angle_deg.to(device)

            # build positive mask
            pos_mask = build_positive_mask(motif_ids, arc_angles,
                                           args.delta_deg, arc_motif_id)

            # one-time batch sanity diagnostics
            if not did_diagnostics:
                print_batch_diagnostics(motif_ids, arc_angles, pos_mask,
                                        arc_motif_id, id2motif)
                did_diagnostics = True

            # loss
            loss = supcon_loss(z, pos_mask, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # track avg positives per anchor
            avg_pos = pos_mask.float().sum(dim=1).mean().item()
            epoch_avg_pos += avg_pos

            if args.dry_run and global_step % 10 == 0:
                print(f"  step {global_step:>3d}  loss={loss.item():.4f}  "
                      f"avg_pos/anchor={avg_pos:.1f}  B={z.size(0)}")

        if max_steps and global_step >= max_steps:
            print(f"\n[dry-run] stopped after {global_step} steps")
            break

        scheduler.step()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_pos_per_anchor = epoch_avg_pos / max(epoch_steps, 1)
        elapsed = time.time() - t0

        # evaluate every 5 epochs (or last epoch)
        eval_str = ""
        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = evaluate_retrieval(
                model, val_subset, args.delta_deg,
                arc_motif_id=arc_motif_id,
                k=args.eval_k, device=device
            )
            eval_str = (f"  ret@{args.eval_k}={metrics['precision@k']:.3f}"
                        f"  arc_err={metrics['mean_arc_angle_err']:.1f}deg"
                        f"  arc_topk={metrics['mean_arc_topk_delta']:.1f}deg")

        print(f"epoch {epoch:>3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"avg_pos={avg_pos_per_anchor:.1f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed:.1f}s{eval_str}")

    # final eval on full val
    print("\n── Final evaluation on full val set ──")
    metrics = evaluate_retrieval(
        model, val_data, args.delta_deg,
        arc_motif_id=arc_motif_id,
        k=args.eval_k, device=device
    )
    print(f"  retrieval@{args.eval_k}       = {metrics['precision@k']:.4f}")
    print(f"  mean arc angle err = {metrics['mean_arc_angle_err']:.1f} deg")
    print(f"  mean arc topK delta= {metrics['mean_arc_topk_delta']:.1f} deg  (should decrease over epochs)")
    print("  per-motif precision:")
    for mid in sorted(metrics["motif_precision"]):
        name = id2motif.get(mid, f"id={mid}")
        print(f"    {name:>14s}: {metrics['motif_precision'][mid]:.4f}")

    # save model
    ckpt_path = proc / "encoder.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "meta": meta,
    }, ckpt_path)
    print(f"\n[io] saved checkpoint -> {ckpt_path}")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train(parse_args())
