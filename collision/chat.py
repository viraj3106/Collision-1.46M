import os
import sys
import torch
import torch.nn.functional as F

# Resolve project root path and insert into Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

def generate_chat(model, tokenizer, prompt, device, max_tokens=150, temp=0.7, top_k=50):
    model.eval()
    formatted = f"<|user|>\n{prompt}\n\n<|assistant|>\n"
    ids = tokenizer.encode(formatted, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :] / temp
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.special_tokens.get("[EOS]", 259):
                break
                
    generated_ids = x[0].tolist()
    response_ids = generated_ids[len(ids):]
    return tokenizer.decode(response_ids)

def main():
    device = torch.device("cpu")
    checkpoint_path = "checkpoints/phase13/collision-instruct-3.37m-best.pt"
    tokenizer_dir = "artifacts/tokenizer"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Instruction-tuned checkpoint not found at {checkpoint_path}.")
        print("Please run training/train_phase13.py first.")
        return
        
    print("Loading BPETokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_dir)
    
    print(f"Loading COLLISION-Instruct-3.37M from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**checkpoint["config"])
    model = CollisionTransformer(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print("\n==================================================")
    print("      COLLISION-Instruct-3.37M Chat Interface")
    print("==================================================")
    print("COLLISION-Instruct-3.37M is a small language model trained from scratch.")
    print("Type your message below. Type '/exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You:\n")
            if user_input.strip() == "/exit":
                print("\nExiting chat. Goodbye!")
                break
            if not user_input.strip():
                continue
                
            print("\nCOLLISION:")
            response = generate_chat(model, tokenizer, user_input, device)
            print(response.strip() + "\n")
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

if __name__ == "__main__":
    main()
