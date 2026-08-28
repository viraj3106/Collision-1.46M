import os
import torch
import yaml
from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

def generate_sample(model, tokenizer, prompt, device, max_tokens=50):
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :] / 0.8  # temp=0.8
            
            # top_k filtering (k=50)
            v, _ = torch.topk(next_token_logits, min(50, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.special_tokens.get("[EOS]", 2):
                break
                
    return tokenizer.decode(x[0].tolist())

def load_model(checkpoint_path, config_path, device):
    model_config = ModelConfig.from_yaml(config_path)
    model = CollisionTransformer(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict (handle potential nested structure)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        
    model.load_state_dict(state_dict)
    return model

def main():
    config_path = "configs/collision_1m.yaml"
    tokenizer_dir = "artifacts/tokenizer"
    device = torch.device("cpu")
    
    # Load Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_dir)
    
    # Checkpoint paths
    p5_path = "checkpoints/phase5/collision-1.46m-step-001500.pt"
    p6_path = "checkpoints/phase6/collision-1.46m-step-001500.pt"
    
    # Manually curated evaluation prompts
    eval_prompts = [
        "Why does the Earth orbit the Sun?",
        "What is an algorithm?",
        "How does machine learning differ from traditional programming?",
        "What is the relationship between energy and matter?",
        "What is philosophy?"
    ]
    
    print("Loading Phase 5 Model (Step 1500)...")
    p5_model = load_model(p5_path, config_path, device)
    
    print("Loading Phase 6 Model (Step 1500)...")
    p6_model = load_model(p6_path, config_path, device)
    
    output_lines = []
    output_lines.append("==================================================")
    output_lines.append("PHASE 5 VS PHASE 6 GENERALIZATION TEST EVALUATION")
    output_lines.append("==================================================")
    output_lines.append("")
    
    for prompt in eval_prompts:
        output_lines.append(f"PROMPT: {prompt}")
        output_lines.append("-" * 40)
        
        # Phase 5 Output
        p5_out = generate_sample(p5_model, tokenizer, prompt, device)
        output_lines.append("PHASE 5 STEP 1500 OUTPUT:")
        output_lines.append(p5_out)
        output_lines.append("")
        
        # Phase 6 Output
        p6_out = generate_sample(p6_model, tokenizer, prompt, device)
        output_lines.append("PHASE 6 STEP 1500 OUTPUT:")
        output_lines.append(p6_out)
        output_lines.append("=" * 50)
        output_lines.append("")
        
    out_path = "experiments/phase6/generalization_test_results.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    print(f"Generalization test complete. Results written to {out_path}")

if __name__ == "__main__":
    main()
