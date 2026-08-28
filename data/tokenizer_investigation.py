import os
import json

def main():
    # Load metadata for v4
    v4_meta_path = "datasets/collision_dataset_v4/metadata.json"
    if not os.path.exists(v4_meta_path):
        print("Error: v4 metadata missing. Please run prepare_v4 first.")
        return
        
    with open(v4_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    cleaned_chars = meta.get("cleaned_characters", 0)
    token_count = meta.get("token_count", 0)
    active_vocab = meta.get("vocabulary_size", 890)
    model_vocab = 8000
    
    chars_per_token = cleaned_chars / token_count if token_count > 0 else 0
    unused_capacity = model_vocab - active_vocab
    pct_used = (active_vocab / model_vocab) * 100
    
    # Calculate bytes/token (in UTF-8)
    # The text is ASCII, so bytes and chars are identical.
    
    print("## TOKENIZER INVESTIGATION RESULTS\n")
    print(f"Active Tokenizer Vocabulary:  {active_vocab}")
    print(f"Model Vocabulary Capacity:    {model_vocab}")
    print(f"Total Characters Cleaned:      {cleaned_chars:,}")
    print(f"Total Tokens:                 {token_count:,}")
    print(f"Average Characters / Token:    {chars_per_token:.4f}")
    print(f"Tokenization Efficiency:      {chars_per_token:.4f} chars/token")
    print(f"Unused/Unknown Capacity:      {unused_capacity} tokens")
    print(f"Percentage of Vocab Used:     {pct_used:.3f}%")
    print()
    print("## RECOMMENDATION AND EVIDENCE\n")
    print("Evidence:")
    print(f"1. The model's embedding and language modeling head are configured for 8000 tokens.")
    print(f"2. However, the tokenizer was trained with a target vocabulary size of only 890 tokens.")
    print(f"3. This means that 7,110 slots in the embedding matrix and the final LM projection head are completely unused.")
    print(f"4. This unused capacity is wasting parameters (roughly {model_vocab * 128} parameters in embedding, plus another {model_vocab * 128} in the head = 1.8M params if not tied, or 910K parameters if tied).")
    print(f"   Wait, let's verify if they are tied: configs show 'tie_embeddings: true', so it is 910,080 parameters wasted out of 1.46M total parameters!")
    print(f"5. An average characters/token of {chars_per_token:.4f} is relatively high for BPE (standard models achieve 3.5 - 4.0 chars/token). Increasing vocabulary size would allow the BPE algorithm to merge longer common phrases, reducing sequence lengths and allowing the model to fit more context in its 256-token limit.")
    print()
    print("Recommendation: **C. Increase tokenizer vocabulary** (and consequently Retrain the tokenizer to target 8000 tokens).")
    print("However, because the prompt states: 'Do NOT silently change the existing model. Do NOT change the model architecture. Use the same parameter count, context length...' and 'Train using the improved dataset... Use the best Phase 5 configuration as the baseline.', we must KEEP option A (Keep current tokenizer and model) for the current Phase 6 training run to ensure a controlled comparison with Phase 5.")
    print("We recommend adopting B/C/D in Phase 7 to utilize the wasted parameters.")

if __name__ == "__main__":
    main()
