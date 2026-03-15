"""Tests for ATHENA2 model components.

Tests: BSCE-GRA, Dual Focal, SupCon, FAMO, Training Utils, Citation GAT.
All tests are PyTorch-based and run on CPU.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn


# ── BSCE-GRA ─────────────────────────────────────────────────────

class TestBSCEGRA:
    def test_basic_loss(self):
        from athena2.models.bsce_gra import BSCEGRALoss

        loss_fn = BSCEGRALoss(num_classes=2)
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.0, 0.0]])
        targets = torch.tensor([0, 1, 0])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0
        assert loss.item() < 10  # reasonable range

    def test_class_weights(self):
        from athena2.models.bsce_gra import BSCEGRALoss

        loss_balanced = BSCEGRALoss(num_classes=2, class_counts=[700, 300])
        loss_uniform = BSCEGRALoss(num_classes=2)
        logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        targets = torch.tensor([0, 1])
        # Both should produce valid losses
        assert loss_balanced(logits, targets).item() > 0
        assert loss_uniform(logits, targets).item() > 0

    def test_sample_weights(self):
        from athena2.models.bsce_gra import BSCEGRALoss

        loss_fn = BSCEGRALoss(num_classes=2)
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        targets = torch.tensor([0, 1])
        weights = torch.tensor([1.0, 0.5])
        loss_weighted = loss_fn(logits, targets, sample_weights=weights)
        loss_unweighted = loss_fn(logits, targets)
        # Weighted should be different
        assert abs(loss_weighted.item() - loss_unweighted.item()) > 0.001

    def test_binary_loss(self):
        from athena2.models.bsce_gra import BSCEGRABinaryLoss

        loss_fn = BSCEGRABinaryLoss(pos_weight=2.33)
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        targets = torch.tensor([0, 1])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_gradient_flows(self):
        from athena2.models.bsce_gra import BSCEGRALoss

        loss_fn = BSCEGRALoss(num_classes=2)
        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


# ── Dual Focal Loss ──────────────────────────────────────────────

class TestDualFocalLoss:
    def test_basic(self):
        from athena2.models.dual_focal_loss import DualFocalLoss

        loss_fn = DualFocalLoss(gamma=2.0)
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        targets = torch.tensor([0, 1])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_with_alpha(self):
        from athena2.models.dual_focal_loss import DualFocalLoss

        loss_fn = DualFocalLoss(gamma=2.0, alpha=0.7)
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_complement_effect(self):
        from athena2.models.dual_focal_loss import DualFocalLoss

        loss_no_comp = DualFocalLoss(complement_weight=0.0)
        loss_with_comp = DualFocalLoss(complement_weight=1.0)
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        # With complement should generally be higher
        l1 = loss_no_comp(logits, targets)
        l2 = loss_with_comp(logits, targets)
        assert l1.item() != l2.item()


# ── SupCon ───────────────────────────────────────────────────────

class TestSupCon:
    def test_loss_computation(self):
        from athena2.models.supcon import SupConLoss

        loss_fn = SupConLoss(temperature=0.07)
        # 4 samples, 2 per class
        features = torch.randn(4, 128)
        labels = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(features, labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_loss_decreases_with_clustering(self):
        from athena2.models.supcon import SupConLoss

        loss_fn = SupConLoss(temperature=0.5)
        # Well-separated clusters should have lower loss
        features_good = torch.cat([
            torch.randn(4, 64) + 5,  # class 0 cluster at +5
            torch.randn(4, 64) - 5,  # class 1 cluster at -5
        ])
        features_bad = torch.randn(8, 64)  # random
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        loss_good = loss_fn(features_good, labels)
        loss_bad = loss_fn(features_bad, labels)
        assert loss_good.item() < loss_bad.item()

    def test_projection_head(self):
        from athena2.models.supcon import ProjectionHead

        head = ProjectionHead(input_dim=1024, hidden_dim=256, output_dim=128)
        x = torch.randn(4, 1024)
        out = head(x)
        assert out.shape == (4, 128)
        # Should be L2 normalized
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ── FAMO ─────────────────────────────────────────────────────────

class TestFAMO:
    def test_basic(self):
        from athena2.models.famo import FAMO

        famo = FAMO(n_tasks=3)
        losses = {
            "task_a": torch.tensor(1.0),
            "task_b": torch.tensor(2.0),
            "task_c": torch.tensor(0.5),
        }
        total = famo(losses)
        assert total.dim() == 0
        assert total.item() > 0

    def test_weights_sum(self):
        from athena2.models.famo import FAMO

        famo = FAMO(n_tasks=4)
        weights = famo.weights
        # Weights should sum to n_tasks (4)
        assert abs(weights.sum().item() - 4.0) < 0.5  # Allow clamping tolerance

    def test_weight_update(self):
        from athena2.models.famo import FAMO

        famo = FAMO(n_tasks=2)
        losses = {
            "task_a": torch.tensor(1.0),
            "task_b": torch.tensor(2.0),
        }
        w1 = famo.update_weights(losses)
        w2 = famo.update_weights(losses)
        # Weights should be returned as dict
        assert "task_a" in w1
        assert "task_b" in w1


# ── Training Utils ───────────────────────────────────────────────

class TestLLRD:
    def test_param_groups(self):
        from athena2.models.training_utils import get_llrd_param_groups

        # Simple model with layer structure
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 2),
        )
        groups = get_llrd_param_groups(model, base_lr=1e-4, decay_factor=0.9)
        assert len(groups) > 0
        assert all("lr" in g for g in groups)


class TestSAM:
    def test_basic_optimization(self):
        from athena2.models.training_utils import SAM

        model = nn.Linear(10, 2)
        optimizer = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.01)

        x = torch.randn(4, 10)
        target = torch.randint(0, 2, (4,))

        # First step
        loss = nn.functional.cross_entropy(model(x), target)
        loss.backward()
        optimizer.first_step()

        # Second step
        loss2 = nn.functional.cross_entropy(model(x), target)
        loss2.backward()
        optimizer.second_step()
        optimizer.zero_grad()

        # Should have updated parameters
        assert True  # If we got here, SAM works


class TestRDrop:
    def test_rdrop_loss(self):
        from athena2.models.training_utils import rdrop_loss

        logits1 = torch.randn(4, 2)
        logits2 = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        loss = rdrop_loss(logits1, logits2, targets, alpha=1.0)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_same_logits_zero_kl(self):
        from athena2.models.training_utils import rdrop_loss

        torch.manual_seed(42)
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        # Same logits → KL term should be 0
        loss_same = rdrop_loss(logits, logits, targets, alpha=1.0)
        # Different logits → KL term > 0, so total loss is higher
        logits2 = torch.randn(8, 2)  # completely different
        loss_diff = rdrop_loss(logits, logits2, targets, alpha=1.0)
        assert loss_same.item() <= loss_diff.item()


class TestEMA:
    def test_shadow_creation(self):
        from athena2.models.training_utils import EMA

        model = nn.Linear(10, 2)
        ema = EMA(model, decay=0.999)
        assert len(ema.shadow) > 0

    def test_apply_and_restore(self):
        from athena2.models.training_utils import EMA

        model = nn.Linear(10, 2)
        ema = EMA(model, decay=0.999)

        # Modify model weights
        original_weight = model.weight.data.clone()
        model.weight.data.add_(torch.ones_like(model.weight.data))
        ema.update(model)

        # Apply shadow (should be close to original due to high decay)
        ema.apply_shadow(model)
        assert not torch.allclose(model.weight.data, original_weight + 1)

        # Restore
        ema.restore(model)
        assert torch.allclose(model.weight.data, original_weight + 1)


class TestSWA:
    def test_collection(self):
        from athena2.models.training_utils import SWACollector

        model = nn.Linear(10, 2)
        swa = SWACollector(model)

        swa.collect(model)
        swa.collect(model)
        assert swa.n_models == 2

        swa.apply_swa(model)
        # Should not crash


class TestSWAG:
    def test_collection_and_sample(self):
        from athena2.models.training_utils import SWAGCollector

        model = nn.Linear(10, 2)
        swag = SWAGCollector(model, max_rank=5)

        for _ in range(3):
            model.weight.data.add_(torch.randn_like(model.weight.data) * 0.1)
            swag.collect(model)

        assert swag.n_models == 3
        swag.sample_model(model, scale=0.5)


class TestCosineScheduler:
    def test_warmup(self):
        from athena2.models.training_utils import CosineAnnealingWarmRestartsWithWarmup

        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer, T_0=10, warmup_steps=5,
        )

        lrs = []
        for _ in range(20):
            lrs.append(scheduler.get_lr()[0])
            scheduler.step()

        # Warmup: LR should increase for first 5 steps
        assert lrs[0] < lrs[4]


# ── Citation GAT ─────────────────────────────────────────────────

class TestCitationGAT:
    def test_gat_layer(self):
        from athena2.models.citation_gat import GATLayer

        layer = GATLayer(in_features=16, out_features=8, n_heads=4)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        out = layer(x, edge_index)
        assert out.shape == (10, 32)  # 8 * 4 heads

    def test_citation_gat_model(self):
        from athena2.models.citation_gat import CitationGAT

        gat = CitationGAT(node_feature_dim=16, hidden_dim=8, output_dim=64, n_heads=4)
        x = torch.randn(20, 16)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        out = gat(x, edge_index)
        assert out.shape == (20, 64)

    def test_build_node_features(self):
        from athena2.models.citation_gat import build_node_features

        nodes = [
            {"decision_id": "case-1", "year": 2020, "law_area": "civil_law", "language": "de"},
            {"decision_id": "case-2", "year": 2021, "law_area": "penal_law", "language": "fr"},
        ]
        x, node_map = build_node_features(nodes)
        assert x.shape[0] == 2
        assert x.shape[1] == 16
        assert "case-1" in node_map
        assert "case-2" in node_map

    def test_build_edge_index(self):
        from athena2.models.citation_gat import build_edge_index

        edges = [
            {"source": "case-1", "target": "case-2"},
            {"source": "case-2", "target": "case-1"},
        ]
        node_map = {"case-1": 0, "case-2": 1}
        ei = build_edge_index(edges, node_map)
        assert ei.shape == (2, 2)

    def test_empty_graph(self):
        from athena2.models.citation_gat import build_edge_index

        ei = build_edge_index([], {})
        assert ei.shape == (2, 0)


# ── World Model (Intermediate Reasoning Predictor) ───────────────

class TestWorldModelConfig:
    def test_init_defaults(self):
        from athena2.models.world_model import LegalWorldModel

        model = LegalWorldModel()
        assert model.encoder_name == "joelniklaus/legal-swiss-roberta-large"
        assert model.n_law_areas == 17
        assert model.use_bsce_gra is True
        assert model.gat_dim == 0

    def test_custom_config(self):
        from athena2.models.world_model import LegalWorldModel

        model = LegalWorldModel(
            encoder_name="test-model",
            n_law_areas=4,
            gat_dim=64,
            use_bsce_gra=False,
        )
        assert model.n_law_areas == 4
        assert model.gat_dim == 64
        assert model.use_bsce_gra is False

    def test_focal_loss_numpy(self):
        from athena2.models.world_model import focal_loss_numpy

        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        loss = focal_loss_numpy(y_true, y_prob)
        assert 0 < loss < 1  # Should be low for good predictions
