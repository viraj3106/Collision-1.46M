import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F

# Insert project root into sys.path to enable loading model and tokenizer
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import top_k_top_p_filtering

def run_inference(prompt, checkpoint_path, tokenizer_dir, max_tokens=100, temperature=0.8, top_k=50, top_p=0.9):
    # 1. Load tokenizer
    if not os.path.exists(tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer directory not found at: {tokenizer_dir}")
    
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_dir)

    # 2. Load released checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Released checkpoint file not found at: {checkpoint_path}")
        
    device = torch.device("cpu")
    print(f"Loading checkpoint from: {checkpoint_path} onto {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 3. Construct model & 4. Load weights
    model_cfg = ModelConfig(**checkpoint["config"])
    model = CollisionTransformer(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 5. Accept prompt & encode
    ids = tokenizer.encode(prompt, bos=True)
    prompt_len = len(ids)
    
    if prompt_len > model_cfg.max_seq_len:
        raise ValueError(f"Prompt of length {prompt_len} tokens exceeds context length limit of {model_cfg.max_seq_len} tokens.")
        
    x = torch.tensor([ids], dtype=torch.long, device=device)
    tokens_generated = 0
    
    # 6. Generate tokens autoregressively
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model_cfg.max_seq_len else x[:, -model_cfg.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :]
            
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits).unsqueeze(0)
                
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            tokens_generated += 1
            
            if next_token.item() == tokenizer.special_tokens["[EOS]"]:
                break
                
    end_time = time.perf_counter()

    # 7. Decode output
    decoded_text = tokenizer.decode(x[0].tolist())
    
    # 8. Display generation statistics
    generation_time = end_time - start_time
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    return {
        "text": decoded_text,
        "tokens_generated": tokens_generated,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second
    }

def main():
    parser = argparse.ArgumentParser(description="Standalone Inference Entry Point for COLLISION-10M")
    parser.add_argument("--prompt", type=str, default="What is artificial intelligence?", help="Text prompt to generate from")
    parser.add_argument("--checkpoint", type=str, default="models/collision-10m/model.pt", help="Path to released model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer", help="Path to BPE tokenizer directory")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature (0.0 for greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K token filtering")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus sampling threshold")
    
    args = parser.parse_args()
    
    try:
        res = run_inference(
            prompt=args.prompt,
            checkpoint_path=args.checkpoint,
            tokenizer_dir=args.tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        print("\n==================================================")
        print("                 GENERATED OUTPUT                 ")
        print("==================================================")
        print(res["text"])
        print("==================================================")
        print("               GENERATION STATISTICS              ")
        print("==================================================")
        print(f"Tokens generated:  {res['tokens_generated']}")
        print(f"Generation time:   {res['generation_time']:.4f} seconds")
        print(f"Throughput speed:  {res['tokens_per_second']:.2f} tokens/second")
        print("==================================================\n")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
