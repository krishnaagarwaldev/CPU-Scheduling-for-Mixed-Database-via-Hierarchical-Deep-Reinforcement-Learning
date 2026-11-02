"""Graph Neural Network (MPNN) with Attention-Weighted Neighborhood Aggregation.

Faithfully implements Section 4.2:
- Equations 11-14: Message passing, attention coefficients, and node state updates.
- Hierarchical pooling producing graph context h_G and category embeddings h_cat.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMessagePassingLayer(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int, leaky_slope: float = 0.2):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.leaky_slope = leaky_slope

        # Linear projection for node features
        self.W_n = nn.Linear(node_in_dim, hidden_dim, bias=False)

        # Message function phi_m (Eq. 13): 2-layer MLP with ReLU
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Attention vector 'a' for learned edge weights (Section 4.2.1)
        self.attn_vec = nn.Parameter(torch.empty(size=(2 * hidden_dim, 1)))
        nn.init.xavier_uniform_(self.attn_vec.data, gain=1.414)

        # Node update function phi_h (Eq. 12)
        self.update_mlp = nn.Sequential(
            nn.Linear(node_in_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(
        self,
        node_feats: torch.Tensor,       # (N, node_in_dim)
        edge_index: torch.Tensor,       # (2, E)
        edge_feats: torch.Tensor        # (E, edge_in_dim)
    ) -> torch.Tensor:
        N = node_feats.size(0)
        if N == 0:
            return torch.zeros((0, self.hidden_dim), device=node_feats.device)

        E = edge_index.size(1) if edge_index.ndim == 2 else 0
        h_proj = self.W_n(node_feats)  # (N, hidden_dim)

        if E == 0:
            # No edges: node features remain self-transformed
            zeros_msg = torch.zeros((N, self.hidden_dim), device=node_feats.device)
            return self.update_mlp(torch.cat([node_feats, zeros_msg], dim=-1))

        src, dst = edge_index[0], edge_index[1]
        h_src = h_proj[src]  # (E, hidden_dim)
        h_dst = h_proj[dst]  # (E, hidden_dim)

        # 1. Attention coefficients alpha_ij (Eq. 14)
        edge_concat = torch.cat([h_src, h_dst], dim=-1)  # (E, 2*hidden_dim)
        edge_attn_scores = F.leaky_relu(torch.matmul(edge_concat, self.attn_vec).squeeze(-1), negative_slope=self.leaky_slope)

        # Numerically stable softmax per destination node
        # Exponentiate scores
        exp_scores = torch.exp(edge_attn_scores - edge_attn_scores.max())
        sum_exp = torch.zeros(N, device=node_feats.device)
        sum_exp.scatter_add_(0, dst, exp_scores)
        alpha_ij = exp_scores / (sum_exp[dst] + 1e-8)  # (E,)

        # 2. Message computation phi_m(h_i, h_j, e_ij) (Eq. 13)
        msg_input = torch.cat([h_dst, h_src, edge_feats], dim=-1)  # (E, 2*hidden_dim + edge_in_dim)
        messages = self.msg_mlp(msg_input)                          # (E, hidden_dim)

        # 3. Weighted aggregation: AGG = sum(alpha_ij * m_ij) (Eq. 14)
        weighted_msgs = messages * alpha_ij.unsqueeze(-1)
        aggregated_msgs = torch.zeros((N, self.hidden_dim), device=node_feats.device)
        aggregated_msgs.index_add_(0, dst, weighted_msgs)

        # 4. Node update phi_h(h_i, m_i) (Eq. 12)
        update_input = torch.cat([node_feats, aggregated_msgs], dim=-1)
        new_node_feats = self.update_mlp(update_input)
        return new_node_feats


class DatabaseGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 19,
        edge_in_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        # Layer 1
        self.layers.append(AttentionMessagePassingLayer(node_in_dim, edge_in_dim, hidden_dim))
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(AttentionMessagePassingLayer(hidden_dim, edge_in_dim, hidden_dim))

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        categories: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Performs message passing and hierarchical pooling.
        
        Returns:
            node_embeddings: (N, hidden_dim)
            graph_context: (hidden_dim,) global mean-pooled vector h_G
            category_embeddings: dict mapping category name to (hidden_dim,) pooled vector
        """
        N = node_feats.size(0)
        if N == 0:
            zeros = torch.zeros(self.hidden_dim, device=node_feats.device)
            return (
                torch.zeros((0, self.hidden_dim), device=node_feats.device),
                zeros,
                {"OLTP": zeros, "OLAP": zeros, "VECTOR": zeros, "BACKGROUND": zeros}
            )

        h = node_feats
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)

        # Global graph pooling (MeanPool)
        graph_context = torch.mean(h, dim=0)  # (hidden_dim,)

        # Category-level hierarchical pooling
        # Categories are encoded in node_feats[:, 8:12] (OLTP:8, OLAP:9, VECTOR:10, BACKGROUND:11)
        cat_names = ["OLTP", "OLAP", "VECTOR", "BACKGROUND"]
        category_embeddings: Dict[str, torch.Tensor] = {}

        for idx, cat_name in enumerate(cat_names):
            mask = node_feats[:, 8 + idx] > 0.5
            if mask.any():
                category_embeddings[cat_name] = torch.mean(h[mask], dim=0)
            else:
                category_embeddings[cat_name] = torch.zeros(self.hidden_dim, device=node_feats.device)

        return h, graph_context, category_embeddings
