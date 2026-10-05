"""Graph-transformer execution-time model (paper Fig. 6).

    graph features (n x 178) -> TransformerConv x3 (178) -> global average pooling (178)
    global features (41)     -> FC x2 (64)
    concat (242)             -> FC x4 (512, 512, 128, 1)  -> execution time

Node features are shipped to the model in compact form (node type id, qubit ids,
5 scaled continuous values) and expanded to the 178-d vectors of Section 5.1.2 inside
:class:`NodeFeatures`, batch by batch. This is numerically identical to feeding the
dense matrices but keeps memory proportional to the batch instead of the dataset
(the paper's dataset has ~17 M nodes, i.e. ~12 GB as dense float32).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv, global_mean_pool

from .features import NUM_COHERENCE, NUM_NODE_TYPES, node_dim


class NodeFeatures(nn.Module):
    """Expand compact node descriptors into (masked) dense node-feature vectors."""

    def __init__(self, max_qubits: int = 127, node_mask: Sequence[bool] | None = None):
        super().__init__()
        self.max_qubits = max_qubits
        full = node_dim(max_qubits)
        mask = torch.ones(full, dtype=torch.bool) if node_mask is None else torch.as_tensor(list(node_mask), dtype=torch.bool)
        if mask.numel() != full:
            raise ValueError(f"node_mask has {mask.numel()} entries, expected {full}")
        self.register_buffer("mask", mask, persistent=False)

    @property
    def out_dim(self) -> int:
        return int(self.mask.sum())

    def forward(self, data) -> torch.Tensor:
        if getattr(data, "x", None) is not None:  # already dense
            return data.x
        nt, nq, nc = data.nt, data.nq, data.nc
        n = nt.size(0)
        x = torch.zeros(n, node_dim(self.max_qubits), device=nt.device)
        rows = torch.arange(n, device=nt.device)
        known = nt >= 0
        x[rows[known], nt[known]] = 1.0
        for k in range(nq.size(1)):
            q = nq[:, k]
            ok = (q >= 0) & (q < self.max_qubits)
            x[rows[ok], NUM_NODE_TYPES + q[ok]] = 1.0
        base = NUM_NODE_TYPES + self.max_qubits
        x[:, base : base + NUM_COHERENCE + 1] = nc
        return x[:, self.mask]


class ExecTimeModel(nn.Module):
    def __init__(
        self,
        global_dim: int,
        max_qubits: int = 127,
        node_mask: Sequence[bool] | None = None,
        hidden: int = 178,
        num_layers: int = 3,
        heads: int = 1,
        global_hidden: int = 64,
        head_dims: Sequence[int] = (512, 512, 128),
        use_graph: bool = True,
        use_global: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not (use_graph or use_global):
            raise ValueError("at least one of use_graph / use_global must be True")
        self.kwargs = dict(
            global_dim=global_dim, max_qubits=max_qubits,
            node_mask=None if node_mask is None else [bool(b) for b in node_mask],
            hidden=hidden, num_layers=num_layers, heads=heads, global_hidden=global_hidden,
            head_dims=tuple(head_dims), use_graph=use_graph, use_global=use_global, dropout=dropout,
        )
        self.use_graph, self.use_global = use_graph, use_global
        in_dim = 0
        if use_graph:
            self.nodes = NodeFeatures(max_qubits, node_mask)
            dims = [self.nodes.out_dim] + [hidden] * num_layers
            self.convs = nn.ModuleList(
                TransformerConv(dims[i], hidden, heads=heads, concat=False, dropout=dropout) for i in range(num_layers)
            )
            in_dim += hidden
        if use_global:
            self.g1 = nn.Linear(global_dim, global_hidden)
            self.g2 = nn.Linear(global_hidden, global_hidden)
            in_dim += global_hidden
        layers: list[nn.Module] = []
        for d in head_dims:
            layers += [nn.Linear(in_dim, d), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_dim = d
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    # The forward pass is split into the pooled graph embedding and the head so that
    # SHAP values can be computed for the global features.
    def graph_embedding(self, data) -> torch.Tensor:
        x = self.nodes(data)
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
        return global_mean_pool(x, data.batch)

    def head_from(self, graph_emb: torch.Tensor | None, g: torch.Tensor | None) -> torch.Tensor:
        parts = []
        if self.use_graph:
            parts.append(graph_emb)
        if self.use_global:
            parts.append(F.relu(self.g2(F.relu(self.g1(g)))))
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)

    def forward(self, data) -> torch.Tensor:
        emb = self.graph_embedding(data) if self.use_graph else None
        return self.head_from(emb, data.g if self.use_global else None)
