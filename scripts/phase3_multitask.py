#!/usr/bin/env python3
"""ATHENA2 Phase 2: LUPI Multi-Task Learning + Advanced Training Recipe.

Implements the full bleeding-edge training pipeline:
1. SupCon pre-training
2. LUPI multi-task model with BSCE-GRA + FAMO
3. LLRD + R-Drop + EMA + SAM + SWA/SWAG
4. cleanlab noise handling
5. Dynamics MLP (Intermediate Reasoning Predictor)
6. Full ablation

Usage:
    uv run python scripts/phase3_multitask.py                    # Full pipeline
    uv run python scripts/phase3_multitask.py --step supcon      # SupCon only
    uv run python scripts/phase3_multitask.py --step train       # Main training
    uv run python scripts/phase3_multitask.py --step ablation    # Ablation study
    uv run python scripts/phase3_multitask.py --no-supcon        # Skip SupCon
    uv run python scripts/phase3_multitask.py --no-sam           # Skip SAM

Requires: pip install athena[worldmodel]
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("athena2.phase3")

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def load_config() -> dict:
    """Load training configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "training.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """Get best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def step_supcon(
    encoder_name: str,
    train_texts: list[str],
    train_labels: np.ndarray,
    config: dict,
    output_dir: Path,
    device: str,
) -> None:
    """Phase 2.3: Supervised Contrastive Learning pre-training."""
    from transformers import AutoModel, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset
    from athena2.models.supcon import SupConPreTrainer

    logger.info("=" * 60)
    logger.info("PHASE 2.3: Supervised Contrastive Pre-Training")
    logger.info("=" * 60)

    supcon_config = config["intermediate_reasoning_predictor"]["training"]["supcon"]
    epochs = supcon_config["epochs"]
    lr = supcon_config["lr"]
    temperature = supcon_config["temperature"]

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # Load encoder with LayerNorm gamma/beta → weight/bias remapping
    from transformers import AutoConfig
    enc_config = AutoConfig.from_pretrained(encoder_name)
    encoder = AutoModel.from_config(enc_config)

    from huggingface_hub import hf_hub_download
    try:
        from safetensors.torch import load_file as load_safetensors
        weight_path = hf_hub_download(encoder_name, "model.safetensors")
        pretrained_state = load_safetensors(weight_path)
    except Exception:
        weight_path = hf_hub_download(encoder_name, "pytorch_model.bin")
        pretrained_state = torch.load(weight_path, map_location="cpu", weights_only=True)

    remapped = {}
    for k, v in pretrained_state.items():
        new_k = k.replace(".gamma", ".weight").replace(".beta", ".bias")
        remapped[new_k] = v

    encoder.load_state_dict(remapped, strict=False)
    encoder = encoder.to(device)

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx], truncation=True,
                max_length=self.max_length, padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            }

    dataset = TextDataset(train_texts, train_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    pretrainer = SupConPreTrainer(
        encoder=encoder,
        hidden_size=config["intermediate_reasoning_predictor"]["encoder"]["hidden_size"],
        projection_dim=supcon_config["projection_dim"],
        temperature=temperature,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(pretrainer.projection.parameters()),
        lr=lr, weight_decay=0.01,
    )

    for epoch in range(epochs):
        loss = pretrainer.train_epoch(dataloader, optimizer)
        logger.info(f"  SupCon epoch {epoch+1}/{epochs}: loss={loss:.4f}")

    # Save pre-trained encoder
    encoder_dir = output_dir / "supcon_encoder"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(encoder_dir)
    logger.info(f"SupCon encoder saved to {encoder_dir}")


def step_train(
    config: dict,
    train_texts: list[str],
    train_labels: np.ndarray,
    val_texts: list[str],
    val_labels: np.ndarray,
    train_features: dict | None,
    sample_weights: np.ndarray | None,
    output_dir: Path,
    device: str,
    use_supcon_encoder: bool = True,
    use_rdrop: bool = True,
    use_sam: bool = True,
    use_ema: bool = True,
    use_swa: bool = True,
) -> dict:
    """Phase 2.4-2.5: Full multi-task training with advanced recipe."""
    from torch.utils.data import DataLoader, Dataset
    from athena2.models.world_model import LegalWorldModel
    from athena2.models.training_utils import (
        EMA, SAM, SWACollector, SWAGCollector,
        get_llrd_param_groups, rdrop_loss,
        CosineAnnealingWarmRestartsWithWarmup,
    )
    from athena2.models.famo import FAMO
    from athena2.evaluation.metrics import evaluate, conformal_calibrate

    logger.info("=" * 60)
    logger.info("PHASE 2.4-2.5: Multi-Task Training + Dynamics MLP")
    logger.info("=" * 60)

    irp_config = config["intermediate_reasoning_predictor"]
    train_config = irp_config["training"]

    # Determine encoder
    encoder_name = irp_config["encoder"]["model_name"]
    supcon_dir = output_dir / "supcon_encoder"
    if use_supcon_encoder and supcon_dir.exists():
        encoder_name = str(supcon_dir)
        logger.info(f"Using SupCon pre-trained encoder from {supcon_dir}")

    # Build model
    model = LegalWorldModel(
        encoder_name=encoder_name,
        max_length=irp_config["encoder"]["max_length"],
        hidden_size=irp_config["encoder"]["hidden_size"],
        n_law_areas=irp_config["feature_heads"]["law_area"]["n_classes"],
        n_reasoning_patterns=irp_config["feature_heads"]["reasoning_pattern"]["n_classes"],
        n_outcome_granular=irp_config["feature_heads"]["outcome_granular"]["n_classes"],
        # Only include GAT dim if GAT features are available (Phase 4)
        gat_dim=0,  # Set to irp_config["gat"]["output_dim"] when GAT embeddings are loaded
        use_bsce_gra=True,
    )
    model.build()
    model._model.to(device)

    # Dataset
    class MultiTaskDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length, features=None, weights=None):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.features = features
            self.weights = weights

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx], truncation=True,
                max_length=self.max_length, padding="max_length",
                return_tensors="pt",
            )
            item = {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            }
            if self.features:
                for key in ["law_area", "error", "reasoning", "outcome"]:
                    if key in self.features:
                        item[f"{key}_labels"] = torch.tensor(
                            int(self.features[key][idx]), dtype=torch.long,
                        )
            if self.weights is not None:
                item["sample_weight"] = torch.tensor(
                    float(self.weights[idx]), dtype=torch.float32,
                )
            return item

    train_dataset = MultiTaskDataset(
        train_texts, train_labels, model._tokenizer,
        irp_config["encoder"]["max_length"],
        train_features, sample_weights,
    )
    val_dataset = MultiTaskDataset(
        val_texts, val_labels, model._tokenizer,
        irp_config["encoder"]["max_length"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_config["batch_size"],
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config["batch_size"] * 2,
        shuffle=False, num_workers=0,
    )

    # LLRD optimizer
    param_groups = get_llrd_param_groups(
        model._model,
        base_lr=train_config["llrd"]["base_lr"],
        decay_factor=train_config["llrd"]["decay_factor"],
    )

    if use_sam and train_config["sam"]["enabled"]:
        optimizer = SAM(
            param_groups, torch.optim.AdamW,
            rho=train_config["sam"]["rho"],
            weight_decay=0.01,
        )
        sam_start = int(train_config["epochs"] * train_config["sam"]["start_fraction"])
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        sam_start = train_config["epochs"] + 1

    # FAMO
    famo = None
    if train_config["famo"]["enabled"]:
        heads = config["intermediate_reasoning_predictor"]["feature_heads"]
        n_tasks = len(heads) + 1  # feature heads + verdict
        famo = FAMO(n_tasks=n_tasks, gamma=train_config["famo"]["gamma"], device=device)
        famo_optimizer = torch.optim.Adam(famo.parameters(), lr=train_config["famo"]["lr"])

    # EMA
    ema = None
    if use_ema and train_config["ema"]["enabled"]:
        ema = EMA(model._model, decay=train_config["ema"]["decay"])

    # SWA/SWAG
    swa_collector = None
    swag_collector = None
    if use_swa and train_config["swa"]["enabled"]:
        swa_collector = SWACollector(model._model)
        swag_collector = SWAGCollector(model._model, max_rank=train_config["swa"]["swag_max_rank"])
        swa_start = int(train_config["epochs"] * train_config["swa"]["start_fraction"])
    else:
        swa_start = train_config["epochs"] + 1

    # Scheduler
    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer if not isinstance(optimizer, SAM) else optimizer.base_optimizer,
        T_0=train_config["scheduler"]["T_0"],
        T_mult=train_config["scheduler"]["T_mult"],
        warmup_steps=train_config["scheduler"]["warmup_steps"],
        eta_min=train_config["scheduler"]["eta_min"],
    )

    # Training loop
    best_f1 = 0.0
    patience = train_config["early_stopping"]["patience"]
    patience_counter = 0
    grad_accum = train_config["gradient_accumulation_steps"]

    for epoch in range(train_config["epochs"]):
        model._model.train()
        epoch_losses = []
        use_sam_this_epoch = isinstance(optimizer, SAM) and epoch >= sam_start

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            sw = batch.get("sample_weight", None)
            if sw is not None:
                sw = sw.to(device)

            # Extract feature labels from batch (if available from LUPI)
            feature_kwargs = {}
            for key, param_name in [
                ("law_area_labels", "law_area_labels"),
                ("error_labels", "error_labels"),
                ("reasoning_labels", "reasoning_labels"),
                ("outcome_labels", "outcome_labels"),
            ]:
                if key in batch:
                    feature_kwargs[param_name] = batch[key].to(device)

            # R-Drop: two forward passes with different dropout
            if use_rdrop and train_config["rdrop"]["enabled"]:
                outputs1 = model._model(input_ids, attention_mask)
                outputs2 = model._model(input_ids, attention_mask)
                losses1 = model.compute_loss(outputs1, labels, sample_weights=sw, **feature_kwargs)
                losses2 = model.compute_loss(outputs2, labels, sample_weights=sw, **feature_kwargs)
                # R-Drop KL
                kl_loss = rdrop_loss(
                    outputs1["verdict_logits"], outputs2["verdict_logits"],
                    labels, alpha=train_config["rdrop"]["alpha"],
                    loss_fn=None,
                )
                loss = (losses1["total"] + losses2["total"]) / 2.0 + kl_loss
            else:
                outputs = model._model(input_ids, attention_mask)
                losses = model.compute_loss(outputs, labels, sample_weights=sw, **feature_kwargs)
                loss = losses["total"]

            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model._model.parameters(),
                    train_config["max_grad_norm"],
                )

                if use_sam_this_epoch:
                    optimizer.first_step()
                    # Second forward pass for SAM
                    outputs_sam = model._model(input_ids, attention_mask)
                    losses_sam = model.compute_loss(outputs_sam, labels, sample_weights=sw, **feature_kwargs)
                    (losses_sam["total"] / grad_accum).backward()
                    optimizer.second_step()
                else:
                    if isinstance(optimizer, SAM):
                        optimizer.base_optimizer.step()
                    else:
                        optimizer.step()

                if isinstance(optimizer, SAM):
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()

                scheduler.step()

                # EMA update
                if ema is not None:
                    ema.update(model._model)

            epoch_losses.append(loss.item() * grad_accum)

        # SWA/SWAG collection
        if epoch >= swa_start:
            if swa_collector:
                swa_collector.collect(model._model)
            if swag_collector:
                swag_collector.collect(model._model)

        # Validation
        model._model.eval()
        val_probs = []
        val_labels_all = []

        # Use EMA weights for evaluation
        if ema is not None:
            ema.apply_shadow(model._model)

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model._model(input_ids, attention_mask)
                probs = torch.softmax(outputs["verdict_logits"], dim=-1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels_all.extend(batch["labels"].numpy())

        if ema is not None:
            ema.restore(model._model)

        val_probs = np.array(val_probs)
        val_labels_arr = np.array(val_labels_all)
        val_preds = (val_probs > 0.5).astype(int)

        from athena2.evaluation.metrics import macro_f1, accuracy, adaptive_calibration_error
        f1 = macro_f1(val_labels_arr, val_preds)
        acc = accuracy(val_labels_arr, val_preds)
        ace = adaptive_calibration_error(val_labels_arr, val_probs)

        avg_loss = np.mean(epoch_losses)
        logger.info(
            f"Epoch {epoch+1}/{train_config['epochs']}: "
            f"loss={avg_loss:.4f}, val_F1={f1:.4f}, val_ACC={acc:.4f}, val_ACE={ace:.4f}"
        )

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            # Save best model
            model_dir = output_dir / "best_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model._model.state_dict(), model_dir / "model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Apply SWA weights
    if swa_collector and swa_collector.n_models > 0:
        logger.info("Applying SWA averaged weights...")
        swa_collector.apply_swa(model._model)
        swa_dir = output_dir / "swa_model"
        swa_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model._model.state_dict(), swa_dir / "model.pt")

    return {"best_val_f1": best_f1}


def main():
    parser = argparse.ArgumentParser(description="ATHENA2 Phase 2: Multi-Task Training")
    parser.add_argument("--step", choices=["supcon", "train", "ablation", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/phase2"))
    parser.add_argument("--no-supcon", action="store_true")
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--no-rdrop", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-swa", action="store_true")
    parser.add_argument("--sample", type=int, default=0,
                        help="Subsample N cases per split for smoke testing (0=full)")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Override epoch count (0=use config)")
    args = parser.parse_args()

    config = load_config()
    device = get_device()
    logger.info(f"Device: {device}")

    import pandas as pd

    # Load data (inline to avoid cross-script import issues)
    path = args.data_dir / "sjp_xl.parquet"
    if not path.exists():
        raise FileNotFoundError(f"SJP-XL not found at {path}. Run phase1_data_foundation.py first.")
    df = pd.read_parquet(path)

    # Map string labels to integers if needed
    if df["label"].dtype == object or str(df["label"].dtype) == "str":
        label_map = {"dismissal": 0, "approval": 1}
        df["label"] = df["label"].map(label_map).astype(int)

    # Official SJP-XL temporal splits
    train = df[df["year"] <= 2015].copy()
    val = df[df["year"].isin([2016, 2017])].copy()
    test = df[df["year"] >= 2018].copy()
    logger.info(f"Splits: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    if args.sample > 0:
        train = train.sample(min(args.sample, len(train)), random_state=42)
        val = val.sample(min(args.sample // 2, len(val)), random_state=42)
        logger.info(f"Smoke test: train={len(train)}, val={len(val)}")

    if args.epochs > 0:
        config["intermediate_reasoning_predictor"]["training"]["epochs"] = args.epochs
        logger.info(f"Override epochs: {args.epochs}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load noise weights if available
    sample_weights = None
    noise_path = Path("data/models/noise_analysis/sample_weights.npy")
    if noise_path.exists():
        all_weights = np.load(noise_path)
        # Align with train split
        if len(all_weights) == len(train):
            sample_weights = all_weights
            logger.info(f"Loaded noise weights: {(all_weights < 1.0).sum():,} down-weighted samples")

    t0 = time.time()

    encoder_name = config["intermediate_reasoning_predictor"]["encoder"]["model_name"]

    if args.step in ("supcon", "all") and not args.no_supcon:
        step_supcon(
            encoder_name,
            train["facts"].tolist(), train["label"].values,
            config, args.output_dir, device,
        )

    if args.step in ("train", "all"):
        results = step_train(
            config,
            train["facts"].tolist(), train["label"].values,
            val["facts"].tolist(), val["label"].values,
            train_features=None,  # LUPI features loaded from parquet when available
            sample_weights=sample_weights,
            output_dir=args.output_dir,
            device=device,
            use_supcon_encoder=not args.no_supcon,
            use_rdrop=not args.no_rdrop,
            use_sam=not args.no_sam,
            use_ema=not args.no_ema,
            use_swa=not args.no_swa,
        )
        logger.info(f"Results: {json.dumps(results, indent=2)}")

    elapsed = time.time() - t0
    logger.info(f"\nPhase 2 complete in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
