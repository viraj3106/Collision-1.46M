# PHASE 77 — PARAMETER CAPACITY EXPANSION

## 1. PRIMARY RESEARCH QUESTION

> **Can increasing model parameter capacity from approximately 10M to 25M–50M improve multi-turn conversational reasoning when using the calibrated Phase 76 hybrid loss objective?**

---

## 2. EXPERIMENTAL DESIGN & CONTROLLED SCALING

Phase 77 tests the parameter capacity ceiling hypothesis established in Phase 76. All training and evaluation conditions (tokenizer, vocabulary, loss weighting $\alpha=0.10, \beta=0.90$, context length, evaluation prompt set) are held strictly constant while model capacity is expanded.

### Candidate Model Scales
- **COLLISION-10M (Control)**: $d_{\text{model}}=384, n_{\text{layer}}=6, n_{\text{head}}=8, d_{\text{ff}}=768$ (10,282,304 parameters)
- **J77-25M (Candidate A)**: $d_{\text{model}}=512, n_{\text{layer}}=10, n_{\text{head}}=8, d_{\text{ff}}=1024$ (25,263,936 parameters)
- **J77-35M (Candidate B)**: $d_{\text{model}}=640, n_{\text{layer}}=9, n_{\text{head}}=10, d_{\text{ff}}=1280$ (34,847,680 parameters)
- **J77-50M (Candidate C)**: $d_{\text{model}}=768, n_{\text{layer}}=9, n_{\text{head}}=12, d_{\text{ff}}=1536$ (48,893,504 parameters)

---

## 3. DIRECTORY STRUCTURE

```text
experiments/phase77/
├── phase77_config.yaml
├── train_phase77.py
├── evaluator.py
├── generate_report.py
├── run_phase77.py
├── test_phase77.py
├── README.md
├── checkpoints/
├── results/
└── reports/
    └── phase77_report.md
```

---

## 4. REPRODUCIBILITY & TESTING

To run the Phase 77 unit test suite:
```bash
python -m pytest experiments/phase77/test_phase77.py
```

To run full Phase 77 execution (training, evaluation, and report generation):
```bash
python experiments/phase77/run_phase77.py
```

To run project-wide tests:
```bash
python -m pytest
```
