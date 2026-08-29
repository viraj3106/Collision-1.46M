# COLLISION Phase 13 — Instruction Tuning Report

## Objective
The objective of Phase 13 is to perform Supervised Instruction Fine-Tuning (SFT) on the COLLISION base model to produce a model, `COLLISION-Instruct-3.37M`, capable of responding to structured instructions in a user-assistant conversation format.

## Base Model
- **Base Checkpoint**: `checkpoints/phase12b/collision-3.38m-phase12b-best.pt` (Immutable)
- **Model Parameters**: 3,375,680 (Trainable)
- **Architecture**: 6 layers, 6 attention heads, 192 embedding dimension, 384 feedforward dimension, 256 context length
- **Tokenizer**: 1.0-BPETokenizer with 8,000 vocabulary size
- **Pretraining Dataset**: `collision_dataset_v5_expanded`
- **Random Seed**: 1337

## Instruction Dataset
- **Dataset Version**: `collision_instruct_v1`
- **Build Pipeline**: Created and executed `data/build_instruct.py` to generate examples.
- **Train Split**: 16,156 examples
- **Validation Split**: 2,116 examples
- **Test Split**: 2,208 examples
- **Category Coverage**: AI, ML, Computer Science, Physics, Astronomy, Philosophy, Mathematics, General Knowledge.
- **Instruction Types**: Definitions, question answering, step-by-step explanations, summarization, teaching, comparisons.
- **Format**: Deterministic conversation structure:
  ```text
  <|user|>
  {instruction}

  <|assistant|>
  {response}
  ```

## Training Configuration
- **Optimizer**: AdamW (weight decay = 0.01)
- **Learning Rate**: 5e-5 with Cosine warmup scheduler (warmup steps = 150, min learning rate = 5e-6)
- **Batch Configuration**: Batch size of 4, Gradient Accumulation of 4
- **Training Epochs/Steps**: 1500 steps
- **Computation**: CPU training
- **Ignore Index**: Masked padding tokens (`[PAD]`) using label target value `-100` so padding does not affect loss calculation.

## Training Results
- Step 500 Train Loss: 3.5045
- Step 1000 Train Loss: 2.7854
- Step 1500 Train Loss: 2.3988

## Validation Results
Evaluation results using non-overlapping splits on SFT datasets:
- **Best Validation Loss**: 2.2762
- **Best Validation Perplexity**: 9.74
- **Test Loss**: 2.2720
- **Test Perplexity**: 9.70

## Base vs Instruct Generation
Direct prompt comparisons based on 30 evaluation prompts:

### Prompt: "What is artificial intelligence?"
* **Base (V5)**: `high-layer to values critical by intems.`
* **Instruct**: `Here difuman is the be|assistan difference comparison been aence of whereas beas is stellar of oras ely liphies...`

### Prompt: "Define supervised learning."
* **Base (V5)**: `This bsegropurations This hash to be ckly, bseen verified values parameters these parameters.`
* **Instruct**: `Here is a comparison between and and I and tonducor en hiom that data that and tuled and ism lan system thergy.`

## Generation Quality
Averages over 30 test prompts:
* **Avg Repetition Rate**: Base = 43.5% | Instruct = 57.6%
* **Avg Unique Token Ratio**: Base = 56.5% | Instruct = 42.4%
* **Avg Response Length**: Base = 65.0 tokens | Instruct = 100.0 tokens
* **Sentence Termination Rate**: Base = 66.7% | Instruct = 6.7%

## Failure Cases
1. **Repetitive Loops**: SFT tuned model frequently gets stuck in loops when generating responses, causing repetition rate to rise to 57.6% (compared to base model's 43.5%).
2. **Failure to Terminate**: The SFT model often fails to emit the Special `[EOS]` token or stop punctuation, lowering termination rate to 6.7%.
3. **Template Mimicking**: The instruct model frequently hallucinates user prompt structures or copies bits of instruction text directly into its response.

## Limitations
1. **Model Parameter Limit**: 3.37M parameters is extremely small for learning conversational logic.
2. **Context Length Constraint**: 256 max tokens limits multi-turn context capacity.
3. **Causal Overfitting**: The model overfits to the templated synthetic dataset formats, making it struggle with variations outside training wording.
4. **No True Reasoning**: The model is a causal statistical predictor. It does not possess human-like reasoning, consciousness, or understanding.

## Conclusion
Instruction tuning has **regressed** the textual variety and termination behaviors of COLLISION due to severe overfitting to templated instruction patterns on a highly constrained 3.37M architecture. While SFT enforces conversational tags (`<|user|>` / `<|assistant|>`), it degrades coherence and introduces repetitive structural loops. This indicates that 3.37M parameters is insufficient to successfully absorb instruction-following behaviors without sacrificing text quality.
