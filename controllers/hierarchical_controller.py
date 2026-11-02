"""Hierarchical DRL Controller: Coordinating Meta-Controller, Sub-Controllers, and GNN.

Faithfully implements the complete system architecture (Section 3 & Figure 1):
- GNN dependency encoding producing graph context h_G and category embeddings h_cat
- Tier-1: Meta-Controller macro CPU budget allocation W = (w_oltp, w_olap, w_vec, w_bg)
- Tier-2: Sub-Controllers process-level continuous shares
- Curiosity-Driven exploration bonus integration
- Checkpoint persistence (save_checkpoint / load_checkpoint).
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import torch
import numpy as np

from config.scheduler_config import (
    WorkloadConfig,
    ResourceConstraintsConfig,
    ControllerConfig,
    GNNConfig
)
from gnn.mpnn_attention import DatabaseGNN
from .meta_controller import MetaController
from .sub_controller import CategorySubController
from .curiosity_module import IntrinsicCuriosityModule


class HierarchicalDRLController:
    def __init__(
        self,
        workload_config: Optional[WorkloadConfig] = None,
        constraints_config: Optional[ResourceConstraintsConfig] = None,
        controller_config: Optional[ControllerConfig] = None,
        gnn_config: Optional[GNNConfig] = None,
        meta_state_dim: int = 21
    ):
        self.workload_cfg = workload_config or WorkloadConfig()
        self.constraints_cfg = constraints_config or ResourceConstraintsConfig()
        self.ctrl_cfg = controller_config or ControllerConfig()
        self.gnn_cfg = gnn_config or GNNConfig()
        self.categories = self.workload_cfg.categories

        # 1. Graph Neural Network (Section 4.2)
        self.gnn = DatabaseGNN(
            node_in_dim=self.gnn_cfg.node_feat_dim,
            edge_in_dim=self.gnn_cfg.edge_feat_dim,
            hidden_dim=self.gnn_cfg.hidden_dim,
            num_layers=self.gnn_cfg.num_layers
        )

        # 2. Meta-Controller (Tier 1, Section 3.2.1)
        self.meta_controller = MetaController(
            meta_state_dim=meta_state_dim,
            graph_context_dim=self.gnn_cfg.hidden_dim,
            hidden_dim=128,
            categories=self.categories,
            config=self.ctrl_cfg,
            constraints=self.constraints_cfg
        )

        # 3. Sub-Controllers (Tier 2, Section 3.2.2)
        self.sub_controllers: Dict[str, CategorySubController] = {
            cat: CategorySubController(
                category=cat,
                state_dim=self.gnn_cfg.node_feat_dim + self.gnn_cfg.hidden_dim,
                action_dim=1,
                hidden_dim=self.gnn_cfg.hidden_dim,
                config=self.ctrl_cfg
            )
            for cat in self.categories
        }

        # 4. Curiosity Module (Section 5.4)
        if self.ctrl_cfg.use_curiosity:
            self.curiosity_module = IntrinsicCuriosityModule(
                state_dim=meta_state_dim,
                action_dim=len(self.categories),
                hidden_dim=64,
                config=self.ctrl_cfg
            )
        else:
            self.curiosity_module = None

    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False
    ) -> Tuple[Dict[str, float], Dict[int, float], Dict[str, Any]]:
        """Executes full hierarchical forward pass.
        
        Returns:
            macro_budget: Dict[str, float] CPU budget proportions W = (w_oltp, w_olap, w_vec, w_bg)
            process_shares: Dict[int, float] Process PID -> priority share multiplier
            internals: Dict containing log_probs and state values for training
        """
        # A. GNN Forward pass
        node_feats = torch.tensor(obs["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(obs["edge_index"], dtype=torch.long)
        edge_feats = torch.tensor(obs["edge_features"], dtype=torch.float32)

        with torch.no_grad():
            node_embs, graph_ctx, cat_embs = self.gnn(node_feats, edge_index, edge_feats)

        # B. Meta-Controller Action
        meta_state = torch.tensor(obs["meta_state"], dtype=torch.float32)
        macro_budget, log_prob, value = self.meta_controller.select_action(
            meta_state=meta_state,
            graph_context=graph_ctx,
            deterministic=deterministic
        )

        # C. Sub-Controllers Actions
        process_shares: Dict[int, float] = {}
        active_pids = obs.get("active_pids", [])
        
        if len(active_pids) > 0 and node_embs.size(0) == len(active_pids):
            for idx, pid in enumerate(active_pids):
                # Detect process category from one-hot node features (indices 8-11)
                cat_idx = int(np.argmax(obs["node_features"][idx, 8:12]))
                cat_name = self.categories[cat_idx] if cat_idx < len(self.categories) else "OLTP"
                sub_ctrl = self.sub_controllers[cat_name]

                # Process state: node feature + category pooled context
                proc_state = torch.cat([node_feats[idx], cat_embs[cat_name]], dim=-1)
                raw_share_action = sub_ctrl.select_action(proc_state, deterministic=deterministic)
                # Map [-1, 1] action to share multiplier [0.5, 1.5]
                share_multiplier = 1.0 + 0.5 * raw_share_action
                process_shares[pid] = float(share_multiplier)

        internals = {
            "meta_log_prob": log_prob,
            "meta_value": value,
            "graph_context": graph_ctx.detach(),
            "cat_embeddings": {k: v.detach() for k, v in cat_embs.items()}
        }
        return macro_budget, process_shares, internals

    def save_checkpoint(self, filepath: str):
        """Saves complete model checkpoint parameters to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "gnn_state": self.gnn.state_dict(),
            "meta_policy_state": self.meta_controller.policy.state_dict(),
            "meta_value_state": self.meta_controller.value_net.state_dict(),
            "sub_controllers": {
                cat: {
                    "actor": ctrl.actor.state_dict(),
                    "critic": ctrl.critic.state_dict(),
                    "log_alpha": ctrl.log_alpha.detach()
                }
                for cat, ctrl in self.sub_controllers.items()
            }
        }
        if self.curiosity_module:
            checkpoint["curiosity_state"] = self.curiosity_module.dynamics_model.state_dict()
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> bool:
        """Loads model checkpoint parameters from disk."""
        if not os.path.exists(filepath):
            return False
        checkpoint = torch.load(filepath, map_location="cpu")
        self.gnn.load_state_dict(checkpoint["gnn_state"])
        self.meta_controller.policy.load_state_dict(checkpoint["meta_policy_state"])
        self.meta_controller.value_net.load_state_dict(checkpoint["meta_value_state"])
        for cat, sub_dict in checkpoint.get("sub_controllers", {}).items():
            if cat in self.sub_controllers:
                self.sub_controllers[cat].actor.load_state_dict(sub_dict["actor"])
                self.sub_controllers[cat].critic.load_state_dict(sub_dict["critic"])
                self.sub_controllers[cat].log_alpha.data.copy_(sub_dict["log_alpha"])
        if self.curiosity_module and "curiosity_state" in checkpoint:
            self.curiosity_module.dynamics_model.load_state_dict(checkpoint["curiosity_state"])
        return True
