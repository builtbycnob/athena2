# ATHENA2 Research Document

**Date**: 2026-03-15
**Scope**: Comprehensive literature review and technical survey for legal judgment world model design.

---

## 1. Swiss Judgment Prediction — Core Literature

### 1.1 Original Paper: Niklaus et al. (2021)

**Title**: "Swiss-Judgment-Prediction: A Multilingual Legal Judgment Prediction Benchmark"
**Venue**: NeurIPS 2021 Datasets and Benchmarks Track
**Paper**: arXiv:2110.00806
**Dataset**: `rcds/swiss_judgment_prediction` (HuggingFace)

**Key facts**:
- 85,002 Swiss Federal Supreme Court (FSCS) cases, 2000–2020
- Three languages: German (35,452 train), French (21,179 train), Italian (3,072 train)
- Binary labels: 0 = dismissal (rejection), 1 = approval (annulment)
- Input: `text` field (facts section only)
- **No considerations field** in this version

**Best results** (macro F1):
| Model | de | fr | it |
|-------|-----|-----|-----|
| Hierarchical BERT | 68.3 | 70.5 | 66.8 |
| XLM-RoBERTa | 67.9 | 69.2 | 65.3 |
| SVM (TF-IDF) | 63.4 | 64.1 | 61.2 |

**Key insight**: Hierarchical attention over long documents outperforms truncation. Legal text often exceeds 512 tokens — the critical arguments can be anywhere in the document.

### 1.2 Swiss Judgment Prediction XL

**Paper**: arXiv:2306.09237 (SCALE benchmark, Niklaus et al. 2023)
**Dataset**: `rcds/swiss_judgment_prediction_xl` (HuggingFace)

**Key facts**:
- 329,000 cases (4× larger than original)
- **Includes `considerations` field** — full judge reasoning text
- Additional fields: `decision_id`, `law_area`, `law_sub_area`, `court`, `chamber`, `canton`, `region`, `origin_facts`, `origin_considerations`
- License: CC-BY-SA-4.0

**Critical finding for ATHENA2**: The `considerations` field contains the judge's complete legal reasoning — the state transition function from facts to verdict. **No published model uses this field for judgment prediction** (it would be data leakage for standard classification). But for ATHENA2's world model, this IS the training signal: it teaches the model HOW judges reason, not just WHAT they decide.

### 1.3 Follow-Up Work by RCDS Group

**LEXTREME** (arXiv:2301.13126, Niklaus et al. 2023, EMNLP Findings)
- 11 multilingual legal NLP datasets, 24 languages
- Benchmark: Legal-XLM-RoBERTa (340M params, trained on 689GB MultiLegalPile) achieves best results
- Swiss JLP is one of 11 tasks; aggregated macro F1 across all tasks: ~48.4% (indicating the difficulty of the broader benchmark)

**SCALE** (arXiv:2306.09237, Niklaus et al. 2023)
- Long-document challenges for legal NLP
- Shows that most models degrade significantly on documents >4096 tokens
- Swiss cases average ~2000-3000 tokens for facts, ~5000+ for considerations

**MultiLegalPile** (arXiv:2306.02069, Niklaus et al. 2023, EMNLP)
- 689GB multilingual legal corpus, 24 languages
- Pre-trained Legal-XLM-R models (base: 110M, large: 340M)
- Swiss German, French, Italian legal text well-represented

**Explainability and Fairness** (arXiv:2402.17013, Niklaus et al. 2024, LREC-COLING)
- Best results with data augmentation + joint training: ~70-71% macro F1 across languages
- **Critical finding**: prediction performance does NOT correlate with explainability quality
- Rationale extraction produces "broken snippets" that legal experts reject (9/10 documents fail)

**SwiLTra-Bench** (ACL 2025)
- 180K translation pairs for Swiss legal text
- Shows Swiss legal language has domain-specific characteristics that generic MT handles poorly

### 1.4 Current SOTA Summary

| Model | de F1 | fr F1 | it F1 | Year |
|-------|-------|-------|-------|------|
| Hierarchical BERT | 68.3 | 70.5 | 66.8 | 2021 |
| Joint Training + DA | **70.6** | **71.6** | **71.2** | 2024 |
| Legal-XLM-R Large | ~68 | ~70 | ~67 | 2023 |
| Zero-shot LLM | <65 | <67 | <63 | 2022 |

**Gap**: No model exceeds ~71% macro F1. No model uses considerations. No model provides calibrated uncertainty. ATHENA2 targets all three.

---

## 2. Legal Judgment Prediction — Beyond Switzerland

### 2.1 European Court of Human Rights (ECHR)

**Chalkidis et al. (2019)** — "Neural Legal Judgment Prediction in English" (ACL 2019)
- 11,478 ECHR cases, binary (violation/no violation)
- Hierarchical BERT: ~79% macro F1
- Attention visualization shows model learns which articles are relevant

**Chalkidis et al. (2021)** — "Paragraph-level Rationale Extraction" (NAACL 2021)
- Extracts which paragraphs of the case are most predictive
- Uses human-annotated rationales for supervision

**Relevance**: ECHR work pioneered hierarchical attention for legal text. ATHENA2 should adopt similar architectures but with the world model twist.

### 2.2 Chinese Courts

**CAIL Dataset**: 2.6M Chinese criminal cases
**TopJudge** (Zhong et al., 2018): Joint prediction of charges + law articles + prison terms
- Multi-task DAG structure captures dependencies between legal subtasks
- Key insight: legal predictions are interdependent (charge determines applicable law, which constrains sentencing)

**Relevance**: Multi-task structure. ATHENA2 should jointly predict outcome + decisive arguments + applicable law.

### 2.3 Indian Courts

**NyayaAnumana** (703K cases, Indian Supreme Court)
- INLegalLlama achieves ~90% F1 (on a somewhat easier task — Indian cases have more structured headnotes)
- Fine-tuned on domain-specific legal corpus

### 2.4 US Federal Courts

**SCOTUS Prediction**: ~32% macro F1 (hardest benchmark)
- Very small dataset, high-stakes cases with complex legal reasoning
- Demonstrates limits of classification approaches for hard legal prediction

### 2.5 Cross-Jurisdiction Insights

1. **Hierarchical attention** works across all jurisdictions (legal documents are long)
2. **Multi-task learning** improves prediction (charges + articles + outcome)
3. **Domain-specific pre-training** consistently outperforms general models
4. **Dataset size matters**: larger datasets enable better models (CAIL 2.6M >> SCOTUS ~3K)
5. **Calibration is rarely addressed** — none of these systems report ECE or Brier scores

---

## 3. Multi-Agent Legal Simulation (Closest Related Work)

### 3.1 AgentCourt (ACL 2025)

**Title**: "AgentCourt: Simulating Court with Adversarial Evolvable Lawyer Agents"
**Key innovations**:
- Multi-agent courtroom simulation with LLM agents
- Adversarial evolving lawyers (+12.1% improvement over static agents)
- Argues that adversarial training improves legal reasoning

**Relevance to ATHENA2**: Direct competitor concept. AgentCourt validates the multi-agent simulation approach. Key difference: AgentCourt is purely LLM-based (no trained world model, no game theory, no calibration). ATHENA2 combines the simulation insight with a data-driven world model.

### 3.2 LegalSim (NLLP 2025)

**Title**: "LegalSim: Discovering Procedural Exploits via Simulation"
**Key innovations**:
- Procedural simulation to discover legal exploits
- Focuses on procedural strategy, not outcome prediction

**Relevance**: Validates simulation for procedural reasoning. ATHENA2 already handles this via the phase-based graph.

### 3.3 AgentsBench

- Simulates Chinese judicial deliberation panels
- Multiple judge agents debate to reach consensus

**Relevance**: Multi-agent deliberation. ATHENA2's Monte Carlo approach (run N simulations independently) differs from consensus-based approaches.

### 3.4 Gap Analysis

**No existing system combines**:
1. Multi-agent adversarial simulation
2. Data-driven world model trained on 329K cases
3. Formal game theory (BATNA, Nash)
4. Calibrated uncertainty (conformal prediction)
5. Counterfactual reasoning

ATHENA2 fills all five gaps simultaneously.

---

## 4. Neuro-Symbolic Legal Reasoning

### 4.1 Logic-LM (2023)

- Combines LLMs with symbolic solvers
- +39% improvement over standard prompting on logic tasks
- Uses LLM to translate NL to formal logic, then solves symbolically

### 4.2 LINC (2023)

- Legal reasoning with logical inference
- Decomposes legal arguments into formal propositions

### 4.3 SOLAR (CIKM 2025)

- Formalized legal ontologies + neural reasoning
- 76.4% accuracy via structured legal knowledge representation

### 4.4 Relevance to ATHENA2

Neuro-symbolic approaches offer **explainability** that pure neural models lack. ATHENA2's world model should produce structured intermediate representations (identified errors, severity assessments, applicable law) that map to legal ontology concepts — similar to ATHENA v1's two-step judge but learned rather than prompt-engineered.

---

## 5. World Models for Non-Physical Domains

### 5.1 DreamerV3 (Hafner et al., Nature 2025)

**Architecture**: Recurrent State-Space Model (RSSM)
- Encodes observations into latent states
- Learns transition dynamics: `z_t+1 = f(z_t, a_t)`
- Predicts rewards and observations from latent states
- "Imagination" = forward simulation in latent space without environment interaction

**Key components**:
1. **Encoder**: observation → latent state
2. **Dynamics model**: predicts next latent state given action
3. **Reward model**: predicts reward from latent state
4. **Decoder**: latent state → predicted observation

**Transfer to legal domain**:
- Observation = case facts + arguments presented
- Action = legal strategy choice (argument selection, evidence emphasis)
- State = current legal reasoning state (judge's evolving assessment)
- Reward = verdict probability distribution
- "Imagination" = counterfactual scenario simulation

### 5.2 MarS (Financial Markets)

- World model for financial market simulation
- Learns market dynamics from historical data
- Generates realistic synthetic market scenarios
- Validates that world model approach works in strategic/adversarial domains

### 5.3 Gap

**No existing work applies world model architectures to legal proceedings.** This is ATHENA2's core novelty: treat the `facts → considerations → verdict` trajectory as a state transition function, learn it from 329K cases, and use it for simulation and counterfactual reasoning.

---

## 6. Legal NLP Models and Embeddings

### 6.1 Domain-Specific Models

| Model | Params | Training Data | Notes |
|-------|--------|--------------|-------|
| Legal-XLM-R Base | 110M | 689GB MultiLegalPile | Best legal multilingual encoder |
| Legal-XLM-R Large | 340M | 689GB MultiLegalPile | Highest quality, but slower |
| Legal Swiss RoBERTa | 110M | Swiss legal corpus | Swiss-specific |
| Legal Swiss Longformer | 149M | Swiss legal corpus | Handles long documents natively |

### 6.2 Embedding Models

| Model | Dims | Speed | Notes |
|-------|------|-------|-------|
| BGE-M3 | 1024 | 120 text/s | ATHENA v1 default, multilingual, local |
| Voyage-law-2 | 1024 | — | Top legal retrieval, cloud-only |
| Legal-XLM-R | 768 | — | Can extract embeddings, legal-tuned |

### 6.3 Recommendation for ATHENA2

**Representation learning**: Fine-tune Legal-XLM-R Large (340M) on Swiss corpus using:
- Contrastive learning: similar cases (same outcome, same law area) should be close
- Multi-task: judgment prediction + law area prediction + criticality prediction
- Citation-aware: cases that cite each other should have related representations

---

## 7. Calibration Methods

### 7.1 Expected Calibration Error (ECE)

Standard metric: bin predictions by confidence, compute |accuracy - confidence| per bin, weight by bin size.
ATHENA v1 ECE: 0.155 (mediocre). Target: <0.05.

### 7.2 Temperature Scaling (Guo et al., ICML 2017)

- Single scalar T applied to logits post-training
- Learns T on validation set to minimize NLL
- Simple, effective, widely used
- Limitation: assumes miscalibration is uniform across confidence levels

### 7.3 Platt Scaling

- Logistic regression on model logits
- More flexible than temperature scaling (learns slope + intercept)
- Works well for binary classification

### 7.4 Conformal Prediction

**Key paper**: "Conformal Prediction for NLP" (TACL 2024 survey)
- Distribution-free coverage guarantees
- Procedure:
  1. Compute nonconformity scores on calibration set
  2. Find quantile threshold for desired coverage (e.g., 90%)
  3. At inference: output prediction set (all labels within threshold)
- **No legal domain application exists** — research gap for ATHENA2

**Procedure for ATHENA2**:
```
1. Train world model on train set
2. On calibration set: compute P(annulment) for each case
3. Define nonconformity score: 1 - P(true_label)
4. Sort scores, find quantile q at desired coverage (e.g., 90%)
5. At inference: if max P(label) < 1-q, output prediction set {both labels}
   else: output single prediction
```

This gives formal guarantees: "90% of the time, the true outcome is in our prediction set."

### 7.5 Focal Loss (Lin et al., 2017)

- Down-weights easy examples, focuses on hard ones
- Reduces overconfidence on easy examples
- Acts as implicit calibration during training
- ATHENA2 should use focal loss for world model training

### 7.6 Brier Score Decomposition

```
Brier = Reliability - Resolution + Uncertainty
```
- **Reliability**: calibration quality (lower = better)
- **Resolution**: discriminative power (higher = better)
- **Uncertainty**: inherent task difficulty (fixed)

ATHENA2 should report full Brier decomposition, not just accuracy.

---

## 8. Datasets — Detailed Inventory

### 8.1 Primary: `rcds/swiss_judgment_prediction_xl`

| Field | Type | Description |
|-------|------|-------------|
| `decision_id` | string | Unique case ID |
| `facts` | string | Facts section |
| `considerations` | string | **Full judge reasoning** |
| `label` | int | 0=dismissal, 1=approval |
| `law_area` | string | public/civil/penal/social |
| `law_sub_area` | string | 3 sub-categories |
| `language` | string | de/fr/it |
| `year` | int | Decision year |
| `court` | string | Court name |
| `chamber` | string | Chamber |
| `canton` | string | Canton |
| `region` | string | Region |

**Size**: ~329K cases. **Language**: de ~160K, fr ~128K, it ~41K.
**License**: CC-BY-SA-4.0.

### 8.2 Secondary: `rcds/swiss_judgment_prediction`

- 85K cases (2000–2020), facts + label only (no considerations)
- Exact splits: train 59,709 / val 8,208 / test 17,357
- Machine-translated augmentation available (3× more data)

### 8.3 Auxiliary Datasets

| Dataset | Size | Key Fields | Use for ATHENA2 |
|---------|------|------------|-----------------|
| `rcds/swiss_criticality_prediction` | 139K | bge_label, citation_label, rulings | BGE publication = case importance signal |
| `rcds/swiss_citation_extraction` | 127K | IOB NER tags (CITATION, LAW) | Build citation graph automatically |
| `rcds/swiss_law_area_prediction` | 22K | law_area (4), law_sub_area (13) | Multi-task training signal |
| SCD (Zenodo, DOI:10.5281/zenodo.11092977) | 122K | 31 structured variables, full text | Temporal validation (up to 2024) |

### 8.4 Pre-Training Corpus

**MultiLegalPile**: 689GB, 24 languages, CC-BY-4.0.
Pre-trained models available: Legal-XLM-R base (110M) and large (340M).

---

## 9. Technical Infrastructure

### 9.1 MLX Ecosystem

- **Fine-tuning**: MLX supports LoRA, QLoRA, DoRA, and full fine-tuning via `mlx-lm`
- **Training from scratch**: Possible but requires custom training loops (GPT-2, ModernBERT examples exist)
- **Performance**: MLX outperforms PyTorch MPS by 24-134% depending on workload
- **M3 Ultra**: ~3× slower than RTX 4090 for training, but excellent for inference on large models
- **Memory**: QLoRA of ~120-140B parameters feasible on 192GB; full fine-tune of ~30-40B at FP16

### 9.2 Recommended Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Training | MLX (primary) | Native Apple Silicon, 24-134% faster than MPS |
| Fallback | PyTorch MPS | Broader ecosystem, more examples |
| Embeddings | sentence-transformers + MLX | Dual backend for flexibility |
| Data | HuggingFace datasets + Parquet | Standard, efficient |
| Vector search | LanceDB | Already in ATHENA, embedded |
| Experiment tracking | Weights & Biases or MLflow | Reproducibility |

### 9.3 Performance Targets

| Operation | Target | Justification |
|-----------|--------|---------------|
| Single prediction | <100ms | Real-time interactive use |
| 10K simulations | <60s | Strategic Monte Carlo |
| Full training run | <24h | Overnight on M3 Ultra |
| Embedding (329K cases) | <2h | One-time, batch processing |

---

## 10. Identified Research Gaps (ATHENA2 Contributions)

### Gap 1: No World Model for Legal Proceedings

**Status**: No published system models legal reasoning as a dynamic process (state transitions).
**ATHENA2 approach**: Train on `facts → considerations → verdict` triplettes from SJP-XL (329K cases). The considerations field is the supervision signal for the reasoning trajectory.

### Gap 2: No Formal Calibration for Legal Judgment Prediction

**Status**: No legal prediction system reports conformal prediction or Brier decomposition.
**ATHENA2 approach**: Full calibration pipeline — temperature scaling + conformal prediction + reliability diagrams.

### Gap 3: No Game-Theoretic Legal AI

**Status**: No system combines case outcome prediction with formal game theory (BATNA, Nash bargaining, ZOPA).
**ATHENA2 approach**: World model outputs feed directly into ATHENA's existing game theory module.

### Gap 4: No Adversarial Simulation + Data-Driven Prediction

**Status**: AgentCourt (ACL 2025) does LLM-only simulation. Supervised models do classification only. No hybrid.
**ATHENA2 approach**: Hybrid system — world model for fast calibrated prediction + LLM simulation for rich adversarial analysis + game theory for strategic recommendations.

### Gap 5: No Counterfactual Legal Reasoning System

**Status**: No system supports "what if argument X instead of Y?" queries on legal cases.
**ATHENA2 approach**: World model latent space supports perturbation → forward simulation → compare outcomes.

---

## 11. Competitive Positioning

| Capability | Niklaus BERT | AgentCourt | ECHR-Chalkidis | ATHENA v1 | **ATHENA2** |
|------------|-------------|------------|----------------|-----------|-------------|
| Accuracy (F1) | 71% | N/A | 79% | 65%* | **Target: 75%+** |
| Calibrated uncertainty | No | No | No | Partial (ECE 0.155) | **Yes (conformal)** |
| Considers reasoning | No | LLM-generated | No | LLM-generated | **Learned from 329K** |
| Game theory | No | No | No | Yes (BATNA, Nash) | **Yes (enhanced)** |
| Counterfactual | No | No | No | No | **Yes** |
| Simulation speed | N/A | Slow (LLM) | N/A | 15 min/case | **<1s/case** |
| Explainability | Attention | LLM text | Rationale extract | LLM memo | **Structured reasoning chain** |

*ATHENA v1 accuracy is not directly comparable — it uses LLM simulation with 7 runs, not supervised classification on the same test set.

---

## 12. Key Papers — Full Reference List

### Swiss Judgment Prediction
1. Niklaus et al. (2021). "Swiss-Judgment-Prediction: A Multilingual Legal Judgment Prediction Benchmark." NeurIPS D&B. arXiv:2110.00806
2. Niklaus et al. (2023). "LEXTREME: A Multi-Lingual and Multi-Task Benchmark for the Legal Domain." EMNLP Findings. arXiv:2301.13126
3. Niklaus et al. (2023). "SCALE: Scaling up the Complexity for Advanced Language Model Evaluation." arXiv:2306.09237
4. Niklaus et al. (2023). "MultiLegalPile: A 689GB Multilingual Legal Corpus." EMNLP. arXiv:2306.02069
5. Niklaus et al. (2024). "Explainability and Fairness in Swiss Judgement Prediction." LREC-COLING. arXiv:2402.17013
6. Niklaus et al. (2025). "SwiLTra-Bench." ACL.

### Legal NLP — Other Jurisdictions
7. Chalkidis et al. (2019). "Neural Legal Judgment Prediction in English." ACL.
8. Chalkidis et al. (2021). "Paragraph-level Rationale Extraction." NAACL.
9. Zhong et al. (2018). "Legal Judgment Prediction via Topological Multi-Task Learning." EMNLP (TopJudge).
10. NyayaAnumana (2024). Indian court prediction with INLegalLlama.

### Multi-Agent Legal Simulation
11. AgentCourt (2025). "Simulating Court with Adversarial Evolvable Lawyer Agents." ACL.
12. LegalSim (2025). "Discovering Procedural Exploits via Simulation." NLLP Workshop.
13. AgentsBench. Chinese judicial deliberation panels.

### Neuro-Symbolic
14. Logic-LM (2023). LLM + symbolic solver.
15. LINC (2023). Legal reasoning with logical inference.
16. SOLAR (2025). "Structured Ontology-based Legal AI Reasoning." CIKM.

### World Models
17. Hafner et al. (2025). "Mastering Diverse Domains through World Models" (DreamerV3). Nature.
18. MarS. World model for financial markets.

### Calibration
19. Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
20. "Conformal Prediction for NLP." TACL 2024 survey.
21. Lin et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

### Legal Embeddings
22. Legal-XLM-R. Pre-trained on 689GB MultiLegalPile.
23. BGE-M3. Multilingual embedding model.

### Explainability
24. Niklaus et al. (2024). Legal experts reject rationale extraction (9/10 fail).
25. EU AI Act — high-risk rules effective August 2026.

---

## 13. Architectural Implications

Based on this research, ATHENA2's architecture should:

1. **Use SJP-XL 329K cases with considerations** as the primary training signal for the world model
2. **Fine-tune Legal-XLM-R Large (340M)** as the encoder — it's the best available legal multilingual model
3. **Adopt hierarchical attention** for long documents (following Niklaus et al.)
4. **Implement multi-task training** with auxiliary objectives (law area, criticality, citation)
5. **Use conformal prediction** for calibrated uncertainty — first application in legal domain
6. **Design latent dynamics** inspired by DreamerV3's RSSM but adapted for discrete legal reasoning states
7. **Maintain LLM simulation** as a complementary system for rich adversarial analysis
8. **Target >71% macro F1** on the standard SJP benchmark to surpass published SOTA
9. **Optimize for MLX** on M3 Ultra — the 192GB unified memory is a competitive advantage for large models
10. **Build citation graph** from `swiss_citation_extraction` for graph-enhanced prediction
