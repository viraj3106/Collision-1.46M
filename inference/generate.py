import os
import argparse
import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from collision.config import CHECKPOINT_DIR, TOKENIZER_DIR

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Replace indices to remove with filter_value in original logits
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        
    return logits

def generate(
    model, 
    tokenizer, 
    prompt, 
    max_tokens=100, 
    temperature=1.0, 
    top_k=50, 
    top_p=0.9, 
    device="cpu"
):
    model.eval()
    # Encode prompt using the tokenizer
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop sequence if context length is exceeded
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            
            # Forward model to get logits
            logits, _ = model(x_cond)
            # Fetch logits of the last token
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature scaling
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                # Apply top-k and/or top-p filtering
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample from the filtered distribution
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy search
                next_token = torch.argmax(next_token_logits).unsqueeze(0)
                
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            
            # Stop if EOS token is sampled
            if next_token.item() == tokenizer.special_tokens["[EOS]"]:
                break
                
    return tokenizer.decode(x[0].tolist())

def main():
    parser = argparse.ArgumentParser(description="Generate text using COLLISION-1M")
    parser.add_argument("--prompt", type=str, default="COLLISION-1M is", help="Prompt to generate text from")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for text generation (0.0 for greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K token filtering")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P token filtering")
    parser.add_argument("--device", type=str, default="cpu", help="Run device: cpu, cuda")
    args = parser.parse_args()

    # Locate checkpoint
    cp_path = args.checkpoint
    if cp_path is None:
        cp_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        
    if not os.path.exists(cp_path):
        print(f"Error: Checkpoint file not found at {cp_path}. Please train the model first.")
        return

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Load checkpoint parameters and state dict
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Loading checkpoint from {cp_path} onto {device}...")
    checkpoint = torch.load(cp_path, map_location=device)
    
    # Recreate config
    model_cfg = ModelConfig(**checkpoint["config"])
    model = CollisionTransformer(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded successfully.")

    # Generate text
    generated_text = generate(
        model, 
        tokenizer, 
        prompt=args.prompt, 
        max_tokens=args.max_tokens, 
        temperature=args.temperature, 
        top_k=args.top_k, 
        top_p=args.top_p, 
        device=device
    )

    print("\n--- GENERATED OUTPUT ---")
    print(generated_text)
    print("------------------------")

if __name__ == "__main__":
    main()
