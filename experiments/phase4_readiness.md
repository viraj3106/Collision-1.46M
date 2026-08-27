# COLLISION-1M Pre-Training Readiness Check

Generated on: 2026-08-27 22:36:34

## Evaluation Status
- **Readiness Decision**: **READY FOR FIRST REAL TRAINING**
- **Justification**: Dataset meets the minimum target of 1,000,000 tokens.

## Dataset Inspection
- **Dataset Version**: collision_dataset_v3
- **Documents**: 6
- **Total Tokens**: 2,411,502
- **Training Tokens**: 2,108,753
- **Validation Tokens**: 302,749
- **Vocabulary Size**: 890

## Model Specifications
- **Model Parameters**: 1,462,464
- **Layers**: 3
- **Heads**: 4
- **Embedding Size**: 128
- **Context Length**: 256

## CPU Speed & Memory
- **Steps/Second**: 3.92
- **Tokens/Second**: 4018.21
- **Memory Usage**: 235.8 MB

## Recommended Next Step
To begin real training, run:
```bash
python -m training.train --config configs/collision_1m_cpu.yaml --max-steps 5000
```
