# PHASE 76 — CONTEXT & LOSS CALIBRATION EXPERIMENT

## 1. PRIMARY RESEARCH QUESTION

> **Can conversational degradation be reduced by calibrating context handling and loss allocation without increasing the model's parameter count?**

---

## 2. SCIENTIFIC PRINCIPLE & SCOPE

> **Phase 76 does not attempt to make COLLISION larger.**
> It attempts to understand why conversational fine-tuning produced regressions in Phase 75 before considering model capacity expansion.

---

## 3. EXPERIMENT BREAKDOWN

### Experiment A — Context Length Stress Test (`J76-A`)
* **Purpose**: Evaluate controlled context lengths `[32, 64, 96, 128, 160, 192, 256, 384]` tokens to identify degradation curves and failure boundaries.

### Experiment B — Hybrid Loss Weighting (`J76-B`)
* **Formula**: $\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{context}} + \beta \cdot \mathcal{L}_{\text{response}}$
* **Configurations**:
  - `B0`: $\alpha = 0.00, \beta = 1.00$ (Response-only baseline)
  - `B1`: $\alpha = 0.10, \beta = 0.90$ (10% Context / 90% Response)
  - `B2`: $\alpha = 0.20, \beta = 0.80$ (20% Context / 80% Response)

### Experiment C — Combined Configuration (`J76-C`)
* Combines context window calibration with the optimal hybrid loss weighting coefficient.

---

## 4. MODEL COMPARISON & CAPACITY CONSTRAINTS

* **Baseline**: `COLLISION-10M` (`models/collision-10m/model.pt`, SHA256: `d256d46d...3775b97`)
* **Historical SFT**: `J74` (`experiments/phase73/checkpoints/collision_10m_sft_j73c.pt`)
* **Phase 76 Candidates**: `J76-A`, `J76-B`, `J76-C` (All maintain the exact 10.28M parameter capacity: 6 layers, 384 dim, 8 heads).

---

## 5. REPRODUCIBILITY & TESTING

To run Phase 76 test suite:
```bash
python -m pytest tests/test_phase76.py
```

To run full Phase 76 execution & report generation:
```bash
python experiments/phase76/run_phase76.py
```

To run full project test suite:
```bash
python -m pytest
```
