# src/athena/rag/config.py
"""RAG configuration via environment variables (all optional, RAG off by default).

  ATHENA_RAG_ENABLED      — "1" to enable (default: off)
  ATHENA_RAG_DB_PATH      — LanceDB storage path (default: ~/.athena/rag_db)
  ATHENA_RAG_MODEL        — embedding model ID (default: Qwen/Qwen3-Embedding-4B)
  ATHENA_RAG_MLX_PATH     — local MLX model path (default: ~/models/qwen3-embedding-4b-4bit)
  ATHENA_RAG_EMBED_DIM    — embedding dimension (default: 2560, Qwen3-4B native)
  ATHENA_RAG_TOKEN_BUDGET — max tokens of retrieved norms (default: 2000)
  ATHENA_RAG_BACKEND      — "mlx", "bge-m3", or "auto" (default: bge-m3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    enabled: bool
    db_path: str
    model: str
    mlx_model_path: str
    embed_dim: int
    token_budget: int
    backend: str  # "auto", "mlx", "bge-m3"


def is_rag_enabled() -> bool:
    """Check if RAG is enabled via env."""
    return os.environ.get("ATHENA_RAG_ENABLED", "") == "1"


def get_rag_config() -> RAGConfig:
    """Read RAG config from environment."""
    return RAGConfig(
        enabled=is_rag_enabled(),
        db_path=os.path.expanduser(
            os.environ.get("ATHENA_RAG_DB_PATH", "~/.athena/rag_db")
        ),
        model=os.environ.get("ATHENA_RAG_MODEL", "Qwen/Qwen3-Embedding-4B"),
        mlx_model_path=os.path.expanduser(
            os.environ.get("ATHENA_RAG_MLX_PATH", "~/models/qwen3-embedding-4b-4bit")
        ),
        embed_dim=int(os.environ.get("ATHENA_RAG_EMBED_DIM", "2560")),
        token_budget=int(os.environ.get("ATHENA_RAG_TOKEN_BUDGET", "2000")),
        backend=os.environ.get("ATHENA_RAG_BACKEND", "bge-m3"),
    )
