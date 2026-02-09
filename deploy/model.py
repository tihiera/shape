"""
model.py — ShapeEncoder (GATv2 graph encoder)

Standalone model definition for inference. No training dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation


class ShapeEncoder(nn.Module):
    """
    GATv2 graph encoder -> L2-normalised embedding.

    Architecture:
        node MLP -> [GATv2 + residual] x L
        -> AttentionalAggregation pool
        -> concat num_nodes
        -> projection MLP -> L2 norm

    Node features x = [curvature, degree].
    """

    def __init__(self, in_dim: int = 2, hidden_dim: int = 128,
                 embed_dim: int = 256, heads: int = 4,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATv2Conv(in_c, hidden_dim, heads=heads,
                          dropout=dropout, add_self_loops=True)
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim * heads))

        self.res_projs = nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_dim if i == 0 else hidden_dim * heads
            out_c = hidden_dim * heads
            if in_c != out_c:
                self.res_projs.append(nn.Linear(in_c, out_c, bias=False))
            else:
                self.res_projs.append(nn.Identity())

        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pool = AttentionalAggregation(gate_nn)

        pool_dim = hidden_dim * heads + 1
        self.proj = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim * heads),
            nn.ReLU(),
            nn.Linear(hidden_dim * heads, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.node_enc(x)

        for gat, norm, res_proj in zip(self.gat_layers, self.gat_norms, self.res_projs):
            residual = res_proj(h)
            h = gat(h, edge_index)
            h = norm(h)
            h = F.elu(h + residual)
            h = self.dropout(h)

        g = self.pool(h, index=batch)

        B = g.size(0)
        num_nodes = torch.zeros(B, device=g.device)
        num_nodes.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        num_nodes_norm = (num_nodes / 20.0).unsqueeze(1)

        g = torch.cat([g, num_nodes_norm], dim=1)

        z = self.proj(g)
        z = F.normalize(z, p=2, dim=-1)
        return z
