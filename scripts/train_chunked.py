#!/usr/bin/env python3
"""ATHENA2 — Gold Standard Chunked Legal Document Classifier Training.

Features:
    Legal-Swiss-RoBERTa-Large encoder
    + Pre-tokenized chunks cached to disk (no re-tokenization)
    + Overlapping 512-token windows (stride 256, cap 12)
    + Attention Pooling over chunk CLS embeddings
    + Gradient checkpointing (halves encoder memory)
    + Float16 mixed precision
    + LLRD (0.95) + EMA (0.999) + cosine warmup
    + law_area multi-task head (weight 0.2)
    + Early stopping on macro F1
    + Checkpoint every N steps (pause/resume support)
    + Saves val/test predictions for calibration pipeline

Usage:
    # Gold standard full training (~40h, pause/resume safe)
    uv run python scripts/train_chunked.py --epochs 3

    # Resume from checkpoint
    uv run python scripts/train_chunked.py --resume

    # Smoke test
    uv run python scripts/train_chunked.py --sample 200 --epochs 1

    # Pre-tokenize only (fast, then train separately)
    uv run python scripts/train_chunked.py --pretokenize-only
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("athena2.train_chunked")

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ── Constants ─────────────────────────────────────────────────────

LABEL_MAP = {"dismissal": 0, "approval": 1}
LAW_AREA_MAP = {"public_law": 0, "civil_law": 1, "penal_law": 2, "social_law": 3}

DEFAULT_ENCODER = "joelniklaus/legal-swiss-roberta-large"


# ── Pre-tokenization ─────────────────────────────────────────────

def pretokenize_split(
    texts: list[str],
    labels: np.ndarray,
    law_areas: np.ndarray,
    tokenizer,
    cache_path: Path,
    max_length: int = 512,
    stride: int = 256,
    max_chunks: int = 12,
) -> None:
    """Pre-tokenize all documents and save chunks to disk.

    Saves a single .npz file with:
        chunk_ids: (total_chunks, max_length) int32
        chunk_masks: (total_chunks, max_length) int8
        doc_chunk_counts: (n_docs,) int32 — number of chunks per doc
        labels: (n_docs,) int64
        law_areas: (n_docs,) int64
    """
    if cache_path.exists():
        logger.info(f"Cache exists: {cache_path}")
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 0
    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 2
    pad_id = tokenizer.pad_token_id or 0

    all_chunk_ids = []
    all_chunk_masks = []
    doc_chunk_counts = []

    t0 = time.time()
    for i, text in enumerate(texts):
        encoding = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        token_ids = encoding["input_ids"]

        # Create overlapping chunks
        chunks = []
        if len(token_ids) <= max_length - 2:
            chunks.append(token_ids)
        else:
            for start in range(0, len(token_ids), stride):
                chunk = token_ids[start : start + max_length - 2]
                if len(chunk) < 32:
                    break
                chunks.append(chunk)
                if len(chunks) >= max_chunks:
                    break

        # Pad each chunk
        for chunk in chunks:
            ids = [cls_id] + chunk + [sep_id]
            mask = [1] * len(ids)
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + [pad_id] * pad_len
                mask = mask + [0] * pad_len
            all_chunk_ids.append(ids[:max_length])
            all_chunk_masks.append(mask[:max_length])

        doc_chunk_counts.append(len(chunks))

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate / 60
            logger.info(f"  Pre-tokenizing: {i+1}/{len(texts)} ({rate:.0f} doc/s, ETA {eta:.1f}min)")

    np.savez_compressed(
        cache_path,
        chunk_ids=np.array(all_chunk_ids, dtype=np.int32),
        chunk_masks=np.array(all_chunk_masks, dtype=np.int8),
        doc_chunk_counts=np.array(doc_chunk_counts, dtype=np.int32),
        labels=labels.astype(np.int64),
        law_areas=law_areas.astype(np.int64),
    )
    elapsed = time.time() - t0
    total_chunks = sum(doc_chunk_counts)
    logger.info(f"  Saved {len(texts)} docs ({total_chunks} chunks) → {cache_path} in {elapsed:.0f}s")


class PreTokenizedDataset(Dataset):
    """Dataset loading pre-tokenized chunks from disk."""

    def __init__(self, cache_path: Path):
        data = np.load(cache_path)
        self.chunk_ids = data["chunk_ids"]       # (total_chunks, max_length)
        self.chunk_masks = data["chunk_masks"]   # (total_chunks, max_length)
        self.doc_counts = data["doc_chunk_counts"]  # (n_docs,)
        self.labels = data["labels"]             # (n_docs,)
        self.law_areas = data["law_areas"]       # (n_docs,)

        # Build offset index: doc_i starts at chunk offset_i
        self.offsets = np.zeros(len(self.doc_counts) + 1, dtype=np.int64)
        np.cumsum(self.doc_counts, out=self.offsets[1:])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        n_chunks = int(self.doc_counts[idx])

        return {
            "input_ids": torch.from_numpy(self.chunk_ids[start:end].astype(np.int64)),
            "attention_mask": torch.from_numpy(self.chunk_masks[start:end].astype(np.int64)),
            "n_chunks": n_chunks,
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "law_area": torch.tensor(int(self.law_areas[idx]), dtype=torch.long),
        }


def collate_chunked(batch: list[dict]) -> dict:
    """Collate variable-chunk documents into flat tensors."""
    all_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
    all_masks = torch.cat([item["attention_mask"] for item in batch], dim=0)
    chunk_counts = [item["n_chunks"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    law_areas = torch.stack([item["law_area"] for item in batch])
    return {
        "input_ids": all_ids,
        "attention_mask": all_masks,
        "chunk_counts": chunk_counts,
        "labels": labels,
        "law_areas": law_areas,
    }


# ── LLRD ──────────────────────────────────────────────────────────

def get_llrd_param_groups(model, base_lr: float = 2e-5, decay_factor: float = 0.95):
    """Layer-wise Learning Rate Decay for encoder + heads."""
    param_groups = []

    # Encoder layers (decay from top to bottom)
    if hasattr(model.encoder, "encoder"):
        layers = model.encoder.encoder.layer
    elif hasattr(model.encoder, "layers"):
        layers = model.encoder.layers
    else:
        param_groups.append({"params": list(model.encoder.parameters()), "lr": base_lr * decay_factor ** 12})
        param_groups.append({"params": list(model.attention_pool.parameters()) + list(model.classifier.parameters()) + list(model.law_area_head.parameters()), "lr": base_lr})
        return param_groups

    n_layers = len(layers)
    for i, layer in enumerate(layers):
        lr = base_lr * (decay_factor ** (n_layers - i))
        param_groups.append({"params": list(layer.parameters()), "lr": lr})

    if hasattr(model.encoder, "embeddings"):
        param_groups.append({
            "params": list(model.encoder.embeddings.parameters()),
            "lr": base_lr * (decay_factor ** (n_layers + 1)),
        })

    head_params = (
        list(model.attention_pool.parameters())
        + list(model.classifier.parameters())
        + list(model.law_area_head.parameters())
        + list(model.dropout.parameters())
    )
    param_groups.append({"params": head_params, "lr": base_lr})

    return param_groups


# ── EMA ───────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ── Checkpointing ────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: EMA,
    epoch: int,
    step: int,
    global_step: int,
    best_f1: float,
    patience_counter: int,
):
    """Save full training state for resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "best_f1": best_f1,
        "patience_counter": patience_counter,
    }, path)
    logger.info(f"  Checkpoint saved: epoch={epoch+1}, step={step+1}, F1={best_f1:.4f}")


def load_checkpoint(path: Path, model, optimizer, scheduler, ema):
    """Load training state from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    logger.info(f"  Resumed from checkpoint: epoch={ckpt['epoch']+1}, step={ckpt['step']+1}, "
                f"best_f1={ckpt['best_f1']:.4f}")
    return ckpt["epoch"], ckpt["step"], ckpt["global_step"], ckpt["best_f1"], ckpt["patience_counter"]


# ── Training ──────────────────────────────────────────────────────

def load_encoder(encoder_name: str):
    """Load encoder with LayerNorm gamma/beta remapping if needed."""
    from transformers import AutoModel, AutoConfig

    if Path(encoder_name).exists():
        return AutoModel.from_pretrained(encoder_name)

    config = AutoConfig.from_pretrained(encoder_name)
    encoder = AutoModel.from_config(config)

    from huggingface_hub import hf_hub_download
    try:
        from safetensors.torch import load_file as load_safetensors
        weight_path = hf_hub_download(encoder_name, "model.safetensors")
        pretrained = load_safetensors(weight_path)
    except Exception:
        try:
            weight_path = hf_hub_download(encoder_name, "pytorch_model.bin")
            pretrained = torch.load(weight_path, map_location="cpu", weights_only=True)
        except Exception:
            return AutoModel.from_pretrained(encoder_name)

    remapped = {
        k.replace(".gamma", ".weight").replace(".beta", ".bias"): v
        for k, v in pretrained.items()
    }
    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    if missing:
        logger.info(f"Encoder missing keys (expected for classification head): {len(missing)}")
    return encoder


def train(args):
    """Main training loop with checkpointing and resume."""
    from transformers import AutoTokenizer
    from athena2.models.chunked_classifier import ChunkedClassifier
    from athena2.evaluation.metrics import macro_f1, accuracy, adaptive_calibration_error

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────
    # Use cleaned dataset if available, otherwise original
    cleaned_path = args.data_dir.parent / "cleaned" / "sjp_xl_cleaned.parquet"
    original_path = args.data_dir / "sjp_xl.parquet"
    if cleaned_path.exists():
        logger.info(f"Using cleaned dataset: {cleaned_path}")
        df = pd.read_parquet(cleaned_path)
        # Use cleaned labels if available
        if "label_cleaned" in df.columns:
            label_map_str = {"dismissal": 0, "approval": 1}
            df["label"] = df["label_cleaned"].map(label_map_str)
            df = df.dropna(subset=["label"])
            df["label"] = df["label"].astype(int)
            logger.info(f"Applied cleaned labels ({df['label_status'].value_counts().to_dict()})")
    else:
        logger.info(f"Using original dataset: {original_path}")
        df = pd.read_parquet(original_path)
    if df["label"].dtype == object or str(df["label"].dtype) == "str":
        df["label"] = df["label"].map(LABEL_MAP)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
    df["law_area_id"] = df["law_area"].map(LAW_AREA_MAP).fillna(0).astype(int)

    train_df = df[df["year"] <= 2015].copy()
    val_df = df[df["year"].isin([2016, 2017])].copy()
    test_df = df[df["year"] >= 2018].copy()
    logger.info(f"Splits: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    if args.sample > 0:
        train_df = train_df.sample(min(args.sample, len(train_df)), random_state=42)
        val_df = val_df.sample(min(args.sample // 2, len(val_df)), random_state=42)
        test_df = test_df.sample(min(args.sample // 2, len(test_df)), random_state=42)
        logger.info(f"Subsampled: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── Pre-tokenize ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    cache_dir = args.output_dir / "token_cache"

    sample_tag = f"_s{args.sample}" if args.sample > 0 else ""
    cache_cfg = f"ml{args.max_length}_st{args.stride}_mc{args.max_chunks}{sample_tag}"

    logger.info("Pre-tokenizing splits (cached to disk)...")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pretokenize_split(
            texts=split_df["facts"].tolist(),
            labels=split_df["label"].values,
            law_areas=split_df["law_area_id"].values,
            tokenizer=tokenizer,
            cache_path=cache_dir / f"{split_name}_{cache_cfg}.npz",
            max_length=args.max_length,
            stride=args.stride,
            max_chunks=args.max_chunks,
        )

    if args.pretokenize_only:
        logger.info("Pre-tokenization complete. Exiting.")
        return

    # ── Datasets (from cache) ─────────────────────────────────────
    train_dataset = PreTokenizedDataset(cache_dir / f"train_{cache_cfg}.npz")
    val_dataset = PreTokenizedDataset(cache_dir / f"val_{cache_cfg}.npz")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_chunked, pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, collate_fn=collate_chunked, pin_memory=True,
        persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────
    encoder = load_encoder(args.encoder)
    hidden_size = encoder.config.hidden_size
    logger.info(f"Encoder: {args.encoder}, hidden={hidden_size}, params={sum(p.numel() for p in encoder.parameters()):,}")

    # Gradient checkpointing — halves encoder memory at ~15% speed cost
    if args.grad_checkpoint:
        encoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: ENABLED")

    model = ChunkedClassifier(
        encoder=encoder,
        hidden_size=hidden_size,
        n_law_areas=4,
        encoder_chunk_batch=args.chunk_batch,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}")

    # ── Class weights ─────────────────────────────────────────────
    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_df) / (2 * c) for c in label_counts],
        dtype=torch.float32,
    ).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # ── Optimizer + Scheduler ─────────────────────────────────────
    param_groups = get_llrd_param_groups(model, base_lr=args.lr, decay_factor=0.95)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = EMA(model, decay=0.999)

    # ── Mixed precision ───────────────────────────────────────────
    use_amp = device in ("mps", "cuda")
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    logger.info(f"Mixed precision: {use_amp} (dtype={amp_dtype})")

    # ── Resume from checkpoint ────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0
    start_epoch = 0
    start_step = 0
    global_step = 0
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        start_epoch, start_step, global_step, best_f1, patience_counter = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, ema,
        )
        # Move model back to device after loading
        model = model.to(device)
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_step += 1  # resume from next step

    # ── Training loop ─────────────────────────────────────────────
    logger.info(f"Training: {args.epochs} epochs, batch={args.batch_size}, "
                f"grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum}")
    logger.info(f"Chunks: max_length={args.max_length}, stride={args.stride}, max_chunks={args.max_chunks}")
    logger.info(f"Checkpoint every {args.ckpt_steps} optimizer steps")
    if start_epoch > 0 or start_step > 0:
        logger.info(f"Resuming from epoch {start_epoch+1}, step {start_step+1}")

    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        epoch_t0 = time.time()

        for step, batch in enumerate(train_loader):
            # Skip steps already done (resume)
            if epoch == start_epoch and step < start_step:
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            law_areas = batch["law_areas"].to(device, non_blocking=True)
            chunk_counts = batch["chunk_counts"]

            if step == 0 and epoch == start_epoch:
                n_chunks = input_ids.shape[0]
                logger.info(f"  First batch: {n_chunks} chunks, {len(chunk_counts)} docs")

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids, attention_mask, chunk_counts)
                verdict_loss = F.cross_entropy(outputs["verdict_logits"], labels, weight=class_weights)
                law_area_loss = F.cross_entropy(outputs["law_area_logits"], law_areas)
                loss = verdict_loss + 0.2 * law_area_loss
                loss = loss / args.grad_accum

            if device == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if device == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                ema.update(model)
                global_step += 1

                # Checkpoint every N optimizer steps
                if global_step % args.ckpt_steps == 0:
                    save_checkpoint(ckpt_path, model, optimizer, scheduler, ema,
                                    epoch, step, global_step, best_f1, patience_counter)

            epoch_losses.append(loss.item() * args.grad_accum)

            if step == 0 and epoch == start_epoch:
                logger.info(f"  First step: loss={loss.item() * args.grad_accum:.4f}, "
                           f"took {time.time() - epoch_t0:.1f}s")

            log_every = 10 if step < 100 else 200
            if (step + 1) % log_every == 0:
                avg = np.mean(epoch_losses[-log_every:])
                lr_now = scheduler.get_last_lr()[0]
                elapsed_epoch = time.time() - epoch_t0
                effective_steps = step + 1 - (start_step if epoch == start_epoch else 0)
                steps_per_sec = effective_steps / max(elapsed_epoch, 0.1)
                remaining = len(train_loader) - step - 1
                eta_h = remaining / max(steps_per_sec, 0.001) / 3600
                logger.info(f"  E{epoch+1} step {step+1}/{len(train_loader)}: "
                           f"loss={avg:.4f}, lr={lr_now:.2e}, "
                           f"{steps_per_sec:.2f} step/s, ETA={eta_h:.1f}h")

        # Reset start_step after first resumed epoch
        start_step = 0

        # ── Validation ────────────────────────────────────────────
        val_t0 = time.time()
        model.eval()
        ema.apply_shadow(model)

        val_probs_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                chunk_counts = batch["chunk_counts"]

                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids, attention_mask, chunk_counts)
                probs = torch.softmax(outputs["verdict_logits"], dim=-1)[:, 1]
                val_probs_list.extend(probs.cpu().numpy())
                val_labels_list.extend(batch["labels"].numpy())

        ema.restore(model)

        val_probs = np.array(val_probs_list)
        val_labels_arr = np.array(val_labels_list)
        val_preds = (val_probs > 0.5).astype(int)

        f1 = macro_f1(val_labels_arr, val_preds)
        acc = accuracy(val_labels_arr, val_preds)
        ace = adaptive_calibration_error(val_labels_arr, val_probs)
        avg_loss = np.mean(epoch_losses)
        val_time = time.time() - val_t0

        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
            f"val_F1={f1:.4f}, val_ACC={acc:.4f}, val_ACE={ace:.4f} (val took {val_time:.0f}s)"
        )

        # Save best
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            model_dir = output_dir / "best_model"
            model_dir.mkdir(parents=True, exist_ok=True)

            ema.apply_shadow(model)
            torch.save(model.state_dict(), model_dir / "model.pt")
            ema.restore(model)

            np.save(model_dir / "val_probs.npy", val_probs)
            np.save(model_dir / "val_labels.npy", val_labels_arr)
            logger.info(f"  Best model saved (F1={f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # End-of-epoch checkpoint
        save_checkpoint(ckpt_path, model, optimizer, scheduler, ema,
                        epoch, len(train_loader), global_step, best_f1, patience_counter)

    elapsed = time.time() - t0
    logger.info(f"\nTraining complete in {elapsed/3600:.1f}h, best val F1={best_f1:.4f}")

    # ── Test set predictions ──────────────────────────────────────
    logger.info("Running test set inference...")
    model_dir = output_dir / "best_model"
    ema.apply_shadow(model)

    test_dataset = PreTokenizedDataset(cache_dir / f"test_{cache_cfg}.npz")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, collate_fn=collate_chunked, pin_memory=True,
    )

    model.eval()
    test_probs_list = []
    test_logits_list = []
    test_labels_list = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            chunk_counts = batch["chunk_counts"]

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids, attention_mask, chunk_counts)
            logits = outputs["verdict_logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1]
            test_logits_list.extend(logits.cpu().numpy())
            test_probs_list.extend(probs.cpu().numpy())
            test_labels_list.extend(batch["labels"].numpy())

            if (i + 1) % 500 == 0:
                logger.info(f"  Test inference: {i+1}/{len(test_loader)}")

    test_probs = np.array(test_probs_list)
    test_labels = np.array(test_labels_list)
    test_preds = (test_probs > 0.5).astype(int)

    test_f1 = macro_f1(test_labels, test_preds)
    test_acc = accuracy(test_labels, test_preds)
    test_ace = adaptive_calibration_error(test_labels, test_probs)

    logger.info(f"\nTest Results:")
    logger.info(f"  Macro F1:  {test_f1:.4f}")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  ACE:       {test_ace:.4f}")

    for lang in ["de", "fr", "it"]:
        mask = test_df["language"].values == lang
        if mask.sum() > 0:
            lang_f1 = macro_f1(test_labels[mask], test_preds[mask])
            lang_acc = accuracy(test_labels[mask], test_preds[mask])
            logger.info(f"  {lang}: F1={lang_f1:.4f}, ACC={lang_acc:.4f}, n={mask.sum()}")

    np.save(model_dir / "test_logits.npy", np.array(test_logits_list))
    np.save(model_dir / "test_probs.npy", test_probs)
    np.save(model_dir / "test_labels.npy", test_labels)

    results = {
        "encoder": args.encoder,
        "max_length": args.max_length,
        "stride": args.stride,
        "max_chunks": args.max_chunks,
        "epochs_trained": epoch + 1,
        "best_val_f1": best_f1,
        "test_f1": test_f1,
        "test_acc": test_acc,
        "test_ace": test_ace,
        "training_time_h": elapsed / 3600,
        "train_samples": len(train_df),
        "device": device,
        "grad_checkpoint": args.grad_checkpoint,
        "mixed_precision": use_amp,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info(f"\nResults saved to {output_dir / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(description="ATHENA2 Gold Standard Training")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/chunked"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--chunk-batch", type=int, default=16,
                        help="Max chunks per encoder forward (memory control)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--sample", type=int, default=0,
                        help="Subsample N train cases (0=full)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--ckpt-steps", type=int, default=500,
                        help="Save checkpoint every N optimizer steps")
    parser.add_argument("--grad-checkpoint", action="store_true", default=True,
                        help="Enable gradient checkpointing (default: on)")
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--pretokenize-only", action="store_true",
                        help="Only pre-tokenize, don't train")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
