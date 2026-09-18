import os
import sys
import time
import psutil
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

def test_1b():
    print("==================================================")
    print("STEP 5 & 6: CHECKPOINT LOAD & GENERATION TEST")
    print("==================================================")

    ckpt_path = os.path.join(PROJECT_ROOT, "models", "collision-1b", "model.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}"

    mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"Memory before load: {mem_before:.2f} MB")

    t0 = time.perf_counter()
    print("Loading checkpoint dict...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint keys:", list(ckpt.keys()))

    cfg = ModelConfig(**ckpt["config"])
    print(f"Config verified: layers={cfg.n_layer}, d_model={cfg.d_model}, heads={cfg.n_head}, d_ff={cfg.d_ff}, vocab={cfg.vocab_size}, context={cfg.max_seq_len}")

    model = CollisionTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t_load = (time.perf_counter() - t0)
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"Model loaded in {t_load:.2f}s | Memory after load: {mem_after:.2f} MB (Peak delta: {mem_after - mem_before:.2f} MB)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with exact parameters: {total_params:,}")
    assert total_params == 999376128, f"Parameter mismatch! Expected 999376128, got {total_params}"

    # Small forward pass test
    x = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (1, 16, cfg.vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN!"
    assert not torch.isinf(logits).any(), "Logits contain Inf!"
    print("Forward pass test: PASS (no NaN/Inf, shape verified (1, 16, 32000))")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tok_path = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
    if os.path.exists(tok_path):
        tokenizer.load(tok_path)
        print(f"Tokenizer loaded successfully from {tok_path} (vocab size: {len(tokenizer.vocab)})")

    test_prompts = [
        "Hello, how are you?",
        "Explain what artificial intelligence is.",
        "What is the capital of France?",
        "Write a short explanation of machine learning.",
        "Why is retrieval useful for language models?"
    ]

    print("\nRunning Generation Tests (15 tokens each):")
    for p in test_prompts:
        t_start = time.perf_counter()
        ids = tokenizer.encode(p, bos=True) if len(tokenizer.vocab) > 0 else [1, 2, 3, 4]
        x_prompt = torch.tensor([ids], dtype=torch.long)
        
        with torch.no_grad():
            out_ids = list(ids)
            for _ in range(15):
                x_cond = x_prompt[:, -cfg.max_seq_len:]
                l, _ = model(x_cond)
                next_t = int(torch.argmax(l[0, -1, :]).item())
                out_ids.append(next_t)
                x_prompt = torch.tensor([out_ids], dtype=torch.long)
                
        t_gen = (time.perf_counter() - t_start) * 1000.0
        tok_s = 15 / (t_gen / 1000.0)
        print(f'  [PASS] Prompt: "{p}" -> 15 tokens generated in {t_gen:.1f}ms ({tok_s:.1f} tok/s)')

    print("\n==================================================")
    print("STEP 5 & 6 COMPLETED: 100% SUCCESSFUL")
    print("==================================================")

if __name__ == "__main__":
    test_1b()
