# Phase 71 — Controlled COLLISION-10M Capability Experiment

## Overview
Phase 71 conducted a controlled research experiment evaluating whether SFT adaptation on `collision_sft_v3` improves the base COLLISION-10M baseline under strict compute and data safety constraints.

## Scientific Outcome
`CANDIDATE_SIMILAR`

* **Baseline Overall Score**: 0.4149
* **Candidate J71 Overall Score**: 0.4023
* **Score Delta**: -0.0126
* **Instruction-Following Gain**: +0.0417

## Integrity & Safety
* **Production Model**: Frozen (`models/collision-10m/model.pt`, SHA256: `d256d46d...3775b97`)
* **J52 Research Candidate**: Preserved (`experiments/phase52/checkpoints/collision_10m_sft_j52.pt`)
* **Real-World Data**: 0 records used for training (`REAL_WORLD_DATA_NOT_READY`)
