"""Citation Graph Attention Network for Swiss legal judgment prediction.

Extracts graph-level features from the BGer citation network using GAT.
These features are concatenated with encoder CLS and multi-task head logits
in the Dynamics MLP (Intermediate Reasoning Predictor).

Expected improvement: +2-5 points macro F1 (DEXA 2025).

Architecture:
    Node features: year, law_area (one-hot), language (one-hot),
                   criticality label, citation count (PageRank/HITS)
    Edge: citation relationship (optionally weighted by recency)
    2-layer GAT → 64D output per case node
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Graph Attention Layer ─────────────────────────────────────────

class GATLayer(nn.Module):
    """Single Graph Attention layer (Velickovic et al., 2018).

    Implements multi-head attention over graph neighbors.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension per head.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        concat: If True, concatenate heads; if False, average.
        residual: If True, add residual connection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        concat: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat
        self.residual = residual

        # Linear transformation for each head
        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        nn.init.xavier_uniform_(self.W)

        # Attention mechanism parameters
        self.a_src = nn.Parameter(torch.empty(n_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_features, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if residual:
            total_out = out_features * n_heads if concat else out_features
            self.res_fc = nn.Linear(in_features, total_out) if in_features != total_out else nn.Identity()
        else:
            self.res_fc = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features, shape (N, in_features).
            edge_index: Edge indices, shape (2, E) — [source, target].

        Returns:
            Updated node features, shape (N, out_features * n_heads) if concat,
            else (N, out_features).
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Transform: (N, in) → (n_heads, N, out)
        h = torch.einsum("ni,hio->hno", x, self.W)

        # Attention scores
        attn_src = torch.einsum("hno,hoi->hni", h, self.a_src).squeeze(-1)  # (H, N, 1) → (H, N)
        attn_dst = torch.einsum("hno,hoi->hni", h, self.a_dst).squeeze(-1)

        # Edge attention: e_ij = LeakyReLU(a_src * h_i + a_dst * h_j)
        edge_attn = self.leaky_relu(attn_src[:, src] + attn_dst[:, dst])  # (H, E)

        # Sparse softmax over neighbors
        # For each destination node, softmax over all incoming edges
        edge_attn_exp = torch.exp(edge_attn - edge_attn.max())
        edge_attn_exp = self.dropout(edge_attn_exp)

        # Sum of attention weights per destination node
        attn_sum = torch.zeros(self.n_heads, N, device=x.device)
        attn_sum.scatter_add_(1, dst.unsqueeze(0).expand(self.n_heads, -1), edge_attn_exp)
        attn_norm = edge_attn_exp / (attn_sum[:, dst] + 1e-10)  # (H, E)

        # Aggregate: weighted sum of neighbor features
        out = torch.zeros(self.n_heads, N, self.out_features, device=x.device)
        weighted_h = attn_norm.unsqueeze(-1) * h[:, src]  # (H, E, out)
        out.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(self.n_heads, -1, self.out_features), weighted_h)

        if self.concat:
            out = out.permute(1, 0, 2).reshape(N, -1)  # (N, H*out)
        else:
            out = out.mean(dim=0)  # (N, out)

        # Residual
        if self.res_fc is not None:
            out = out + self.res_fc(x)

        return out


# ── Citation GAT Model ───────────────────────────────────────────

class CitationGAT(nn.Module):
    """2-layer GAT over the BGer citation graph.

    Produces per-node embeddings that capture citation context.

    Args:
        node_feature_dim: Dimension of input node features.
        hidden_dim: Hidden dimension per head.
        output_dim: Final output dimension per node.
        n_heads: Number of attention heads in first layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        node_feature_dim: int = 16,
        hidden_dim: int = 16,
        output_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gat1 = GATLayer(
            node_feature_dim, hidden_dim,
            n_heads=n_heads, dropout=dropout, concat=True,
        )
        self.gat2 = GATLayer(
            hidden_dim * n_heads, output_dim,
            n_heads=1, dropout=dropout, concat=False, residual=True,
        )
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features, shape (N, node_feature_dim).
            edge_index: Edge indices, shape (2, E).

        Returns:
            Node embeddings, shape (N, output_dim).
        """
        h = F.elu(self.gat1(x, edge_index))
        h = self.dropout(h)
        h = self.gat2(h, edge_index)
        h = self.norm(h)
        return h


# ── Feature Construction ─────────────────────────────────────────

# Law area encoding
LAW_AREA_MAP = {
    "public_law": 0,
    "civil_law": 1,
    "penal_law": 2,
    "social_law": 3,
}

# Language encoding
LANGUAGE_MAP = {"de": 0, "fr": 1, "it": 2}


def build_node_features(
    nodes: list[dict[str, Any]],
    pagerank: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Build node feature matrix from citation graph nodes.

    Features per node:
    - year (normalized)
    - law_area (one-hot, 4D)
    - language (one-hot, 3D)
    - criticality label (if available)
    - PageRank score (if computed)
    - citation count (in-degree, out-degree)
    Total: ~16D

    Args:
        nodes: List of node dicts with metadata.
        pagerank: Optional PageRank scores.

    Returns:
        (feature_matrix, node_id_to_index) tuple.
    """
    node_ids = [n.get("decision_id", n.get("id", "")) for n in nodes]
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(nodes)

    # Feature dimensions: year(1) + law_area(4) + language(3) + criticality(1) +
    # pagerank(1) + in_degree(1) + out_degree(1) = 12, padded to 16
    feat_dim = 16
    features = np.zeros((N, feat_dim), dtype=np.float32)

    years = [n.get("year", 2010) or 2010 for n in nodes]
    year_min, year_max = min(years), max(years)
    year_range = max(year_max - year_min, 1)

    for i, node in enumerate(nodes):
        # Normalized year
        year = node.get("year", 2010) or 2010
        features[i, 0] = (year - year_min) / year_range

        # Law area one-hot
        law_area = node.get("law_area", "")
        if law_area in LAW_AREA_MAP:
            features[i, 1 + LAW_AREA_MAP[law_area]] = 1.0

        # Language one-hot
        lang = node.get("language", "")
        if lang in LANGUAGE_MAP:
            features[i, 5 + LANGUAGE_MAP[lang]] = 1.0

        # Criticality (if available)
        crit = node.get("bge_label", node.get("criticality", 0))
        features[i, 8] = float(crit) if crit is not None else 0.0

        # PageRank
        nid = node_ids[i]
        if pagerank and nid in pagerank:
            features[i, 9] = pagerank[nid]

        # Label (for training signal, masked at inference)
        label = node.get("label")
        if label is not None:
            features[i, 10] = float(label)

    return torch.tensor(features, dtype=torch.float32), node_id_to_idx


def build_edge_index(
    edges: list[dict[str, str]],
    node_id_to_idx: dict[str, int],
) -> torch.Tensor:
    """Build edge index tensor from citation edges.

    Args:
        edges: List of edge dicts with 'source' and 'target'.
        node_id_to_idx: Mapping from node ID to index.

    Returns:
        Edge index tensor, shape (2, E).
    """
    src_list = []
    dst_list = []
    for edge in edges:
        src = edge.get("source", "")
        dst = edge.get("target", "")
        if src in node_id_to_idx and dst in node_id_to_idx:
            src_list.append(node_id_to_idx[src])
            dst_list.append(node_id_to_idx[dst])

    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor([src_list, dst_list], dtype=torch.long)


def extract_gat_features(
    gat_model: CitationGAT,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
    case_ids: list[str],
    pagerank: dict[str, float] | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract GAT embeddings for specific cases.

    Args:
        gat_model: Trained CitationGAT model.
        nodes: All graph nodes.
        edges: All graph edges.
        case_ids: IDs of cases to extract features for.
        pagerank: Optional PageRank scores.
        device: Computation device.

    Returns:
        Feature tensor, shape (len(case_ids), output_dim).
    """
    x, node_map = build_node_features(nodes, pagerank)
    edge_index = build_edge_index(edges, node_map)

    x = x.to(device)
    edge_index = edge_index.to(device)
    gat_model = gat_model.to(device)

    gat_model.eval()
    with torch.no_grad():
        all_embeddings = gat_model(x, edge_index)

    # Extract embeddings for requested cases
    indices = [node_map[cid] for cid in case_ids if cid in node_map]
    if not indices:
        return torch.zeros(len(case_ids), all_embeddings.size(1), device=device)

    return all_embeddings[indices]
