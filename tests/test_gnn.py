"""Unit tests for GNN architecture and attention-weighted message passing."""

import torch
import numpy as np

from gnn.mpnn_attention import DatabaseGNN, AttentionMessagePassingLayer


def test_gnn_shapes_and_attention():
    gnn = DatabaseGNN(node_in_dim=19, edge_in_dim=4, hidden_dim=32, num_layers=2)
    
    # 5 nodes, 4 edges
    node_feats = torch.randn(5, 19)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_feats = torch.randn(4, 4)

    node_embs, graph_ctx, cat_embs = gnn(node_feats, edge_index, edge_feats)

    assert node_embs.shape == (5, 32)
    assert graph_ctx.shape == (32,)
    assert "OLTP" in cat_embs
    assert cat_embs["OLTP"].shape == (32,)


def test_gnn_empty_graph_handling():
    gnn = DatabaseGNN(node_in_dim=19, edge_in_dim=4, hidden_dim=32)
    node_feats = torch.zeros(0, 19)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_feats = torch.zeros(0, 4)

    node_embs, graph_ctx, cat_embs = gnn(node_feats, edge_index, edge_feats)
    assert node_embs.shape == (0, 32)
    assert graph_ctx.shape == (32,)
    assert torch.all(graph_ctx == 0.0)
