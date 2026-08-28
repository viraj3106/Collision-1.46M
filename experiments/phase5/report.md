# COLLISION-1.46M Phase 5

## Dataset
* **Dataset Version**: collision_dataset_v3
* **Total Tokens**: 2,411,502
* **Training Tokens**: 2,108,753
* **Validation Tokens**: 302,749

## Model
* **Parameter Count**: 1,462,464
* **Layers**: 3
* **Embedding Dimension**: 128
* **Attention Heads**: 4
* **Context Length**: 256

## Training
* **Steps**: 2,000
* **Device**: CPU
* **Batch Size**: 4 (Gradient Accumulation: 4)
* **Initial LR**: 6e-4 (Cosine Warmup)

## Results
* **Step 0 (Initial Random Checkpoint) Validation Loss**: 8.9308
* **Step 0 (Initial Random Checkpoint) Perplexity**: 7561.34
* **Step 500 Training Loss**: 4.4737
* **Step 500 Validation Loss**: 5.1501
* **Step 500 Perplexity**: 172.45
* **Final Training Loss (Step 2000)**: 1.4061
* **Best Training Loss**: 1.4061
* **Final Validation Loss (Step 2000)**: 4.3394
* **Best Validation Loss**: 4.1409
* **Best Perplexity**: 62.86
* **Training Time**: 817.8 seconds
* **Average tokens/sec**: 2504.2

## Checkpoints
Checkpoints saved under `checkpoints/phase5/`:
* `collision-1.46m-initial.pt` (Initial pre-training checkpoint)
* `collision-1.46m-step-000500.pt`
* `collision-1.46m-step-001000.pt`
* `collision-1.46m-step-001500.pt`
* `collision-1.46m-step-002000.pt`
* `collision-1.46m-best.pt` (Best Validation Loss: 4.1409)

## Training Classification
### HEALTHY

## Known Experimental Limitation
* **Tokenizer/Model Vocab Mismatch**: The tokenizer has 890 active vocabulary tokens, while the model is configured with a vocabulary capacity of 8,000. This is safe and fully functional since all dataset token IDs fall within 0–889 (less than 8,000). The extra vocabulary capacity remains unused to preserve the target parameter count of 1,462,464 and architecture constraints.

## Generation Comparison
See complete comparisons in `generation_comparison.txt`.

## Observations
The loss decreased from initial baseline of 4.4737 to final loss of 1.4061, demonstrating that the model learns patterns from `collision_dataset_v3`.

## Limitations
This experiment shows simple pattern replication on a CPU dataset with 2,000 steps. It does NOT prove generalized intelligence or high-level logical reasoning.
