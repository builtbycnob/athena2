"""Chunked Legal Document Classifier with Attention Pooling.

Architecture:
    Facts text → overlapping 512-token chunks
        → Legal-Swiss-RoBERTa-Large encoder (shared)
        → CLS embeddings per chunk
        → Attention Pooling (learned query)
        → Linear classifier (verdict + law_area)

Handles variable-length documents naturally via chunking.
Covers 87% of text with cap=12 chunks (vs 12% with truncation to 512).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learned query attention pooling over chunk CLS embeddings.

    Single learned query attends to all chunk CLS vectors,
    producing a weighted document representation.

    Args:
        hidden_size: Dimension of CLS embeddings.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.scale = math.sqrt(hidden_size)

    def forward(self, chunk_cls: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool chunk CLS embeddings into single document vector.

        Args:
            chunk_cls: (batch, max_chunks, hidden) CLS embeddings.
            mask: (batch, max_chunks) bool mask, True = valid chunk.

        Returns:
            (batch, hidden) pooled document embedding.
        """
        # Attention scores: query dot each chunk CLS
        scores = torch.matmul(chunk_cls, self.query) / self.scale  # (batch, max_chunks)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, max_chunks, 1)
        pooled = (weights * chunk_cls).sum(dim=1)  # (batch, hidden)
        return pooled


class ChunkedClassifier(nn.Module):
    """Document classifier with chunked encoding and attention pooling.

    Args:
        encoder: Pre-trained transformer encoder.
        hidden_size: Encoder hidden dimension (1024 for Large).
        n_law_areas: Number of law area classes (4 for CH).
        encoder_chunk_batch: Max chunks per encoder forward pass (memory control).
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int = 1024,
        n_law_areas: int = 4,
        encoder_chunk_batch: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.encoder_chunk_batch = encoder_chunk_batch

        self.attention_pool = AttentionPooling(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 2)
        self.law_area_head = nn.Linear(hidden_size, n_law_areas)

    def encode_chunks(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode chunks through transformer in micro-batches.

        Args:
            input_ids: (total_chunks, seq_len) all chunks flat.
            attention_mask: (total_chunks, seq_len).

        Returns:
            (total_chunks, hidden) CLS embeddings.
        """
        all_cls = []
        for i in range(0, len(input_ids), self.encoder_chunk_batch):
            batch_ids = input_ids[i : i + self.encoder_chunk_batch]
            batch_mask = attention_mask[i : i + self.encoder_chunk_batch]
            outputs = self.encoder(input_ids=batch_ids, attention_mask=batch_mask)
            cls = outputs.last_hidden_state[:, 0]  # (micro_batch, hidden)
            all_cls.append(cls)
        return torch.cat(all_cls, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_counts: list[int],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (total_chunks, seq_len) flat packed chunks.
            attention_mask: (total_chunks, seq_len).
            chunk_counts: Number of chunks per document in batch.

        Returns:
            Dict with verdict_logits and law_area_logits.
        """
        # Encode all chunks
        cls_embeddings = self.encode_chunks(input_ids, attention_mask)

        # Pad and stack into (batch, max_chunks, hidden)
        max_chunks = max(chunk_counts)
        batch_size = len(chunk_counts)
        device = cls_embeddings.device

        padded = torch.zeros(batch_size, max_chunks, self.hidden_size, device=device)
        mask = torch.zeros(batch_size, max_chunks, dtype=torch.bool, device=device)

        start = 0
        for i, count in enumerate(chunk_counts):
            padded[i, :count] = cls_embeddings[start : start + count]
            mask[i, :count] = True
            start += count

        # Attention pooling
        doc_embedding = self.attention_pool(padded, mask)  # (batch, hidden)
        doc_embedding = self.dropout(doc_embedding)

        return {
            "verdict_logits": self.classifier(doc_embedding),
            "law_area_logits": self.law_area_head(doc_embedding),
            "doc_embedding": doc_embedding,
        }
