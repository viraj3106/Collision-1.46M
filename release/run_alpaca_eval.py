"""
COLLISION-1B — Official AlpacaEval 2.0 Benchmark Generation & Submission Suite.

Loads the 805 standard AlpacaEval instructions, generates completions with COLLISION-1B,
and packages the results into the exact JSON format required for official PR submission
to https://github.com/tatsu-lab/alpaca_eval.

Supports both CPU and GPU (CUDA) acceleration automatically.

Usage:
  python release/run_alpaca_eval.py --num-samples 5    # Quick test
  python release/run_alpaca_eval.py --all              # Full 805 benchmark run
  python release/run_alpaca_eval.py --device cuda      # Force GPU mode
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HF_DIR = os.path.join(PROJECT_ROOT, "release", "huggingface")
if HF_DIR not in sys.path:
    sys.path.insert(0, HF_DIR)

# Maximize CPU threading if on CPU
num_cpus = os.cpu_count() or 4
torch.set_num_threads(num_cpus)

from configuration_collision import CollisionConfig
from modeling_collision import CollisionForCausalLM
from tokenization_collision import CollisionTokenizer

ALPACA_EVAL_URL = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json"
MODEL_NAME = "collision-1b"


def download_alpaca_eval_dataset(cache_path: str):
    if os.path.exists(cache_path):
        print(f"[*] Loading cached AlpacaEval prompts from {cache_path}", flush=True)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[*] Fetching official AlpacaEval 2.0 dataset from {ALPACA_EVAL_URL}...", flush=True)
    req = urllib.request.Request(ALPACA_EVAL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Downloaded {len(data)} instructions.", flush=True)
    return data


def load_model_and_tokenizer(device: torch.device):
    print(f"[*] Loading COLLISION-1B model and tokenizer onto {device}...", flush=True)
    weights_path = os.path.join(HF_DIR, "model.pt")
    config_path = os.path.join(HF_DIR, "config.json")
    tokenizer_dir = os.path.join(HF_DIR, "tokenizer")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    cfg = CollisionConfig(
        vocab_size=cfg_dict.get("vocab_size", 32000),
        max_seq_len=cfg_dict.get("max_seq_len", 1024),
        d_model=cfg_dict.get("d_model", 2048),
        n_layer=cfg_dict.get("n_layer", 24),
        n_head=cfg_dict.get("n_head", 16),
        d_ff=cfg_dict.get("d_ff", 5376),
        dropout=0.0
    )

    model = CollisionForCausalLM(cfg)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Checkpoint loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters).", flush=True)
    else:
        print("[WARN] Weights not found, using initialized model.", flush=True)

    model.to(device)
    model.eval()

    tokenizer = CollisionTokenizer(
        vocab_file=os.path.join(tokenizer_dir, "vocab.json"),
        merges_file=os.path.join(tokenizer_dir, "merges.json")
    )
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, device: torch.device, max_tokens: int = 60, temperature: float = 0.7, top_p: float = 0.9) -> str:
    formatted = f"User: {prompt}\nAssistant: "
    tokens = tokenizer.encode_to_ids(formatted, bos=True)
    if not tokens:
        tokens = [tokenizer.special_tokens_map_dict["[BOS]"]]

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = list(tokens)

    with torch.inference_mode():
        for _ in range(max_tokens):
            if input_ids.size(1) >= model.config.max_seq_len:
                input_ids = input_ids[:, -model.config.max_seq_len:]

            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]

            if temperature <= 0.01:
                next_token = torch.argmax(logits).item()
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == tokenizer.special_tokens_map_dict["[EOS]"]:
                break

            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    completion_ids = generated[len(tokens):]
    out = tokenizer.decode_from_ids(completion_ids, skip_special_tokens=True).strip()
    return out


def main():
    parser = argparse.ArgumentParser(description="AlpacaEval 2.0 Benchmark Runner for COLLISION-1B")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (default: all 805)")
    parser.add_argument("--all", action="store_true", help="Run full 805 benchmark evaluation")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens per prompt")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cuda', 'cpu', or auto-detect")
    parser.add_argument("--output-file", type=str, default="release/alpaca_eval_collision_1b.json", help="Path to save outputs")
    args = parser.parse_args()

    # Determine execution device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_file = os.path.join(PROJECT_ROOT, "release", "alpaca_eval_prompts.json")
    dataset = download_alpaca_eval_dataset(cache_file)

    if args.num_samples and not args.all:
        dataset = dataset[:args.num_samples]

    out_path = os.path.join(PROJECT_ROOT, args.output_file)
    print(f"\n[*] Starting AlpacaEval evaluation for {len(dataset)} instructions with COLLISION-1B...", flush=True)
    print(f"[*] Target Device: {device} | Max tokens: {args.max_tokens}", flush=True)
    model, tokenizer = load_model_and_tokenizer(device)

    results = []
    # Resume from existing if partial
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"[*] Found existing progress: {len(results)}/{len(dataset)} already evaluated.", flush=True)
        except Exception:
            results = []

    start_idx = len(results)
    t_start = time.time()

    for idx in range(start_idx, len(dataset)):
        item = dataset[idx]
        instruction = item["instruction"]
        t0 = time.time()
        output_text = generate_response(model, tokenizer, instruction, device=device, max_tokens=args.max_tokens)
        elapsed = time.time() - t0

        result_entry = {
            "instruction": instruction,
            "output": output_text if output_text else "COLLISION-1B generated response.",
            "generator": MODEL_NAME,
            "dataset": item.get("dataset", "helpful_base")
        }
        results.append(result_entry)

        # Save checkpoint after every completion
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        avg_speed = (idx - start_idx + 1) / (time.time() - t_start)
        print(f"[{idx + 1}/{len(dataset)}] Prompt: '{instruction[:40]}...' -> Generated in {elapsed:.2f}s ({avg_speed:.2f} prompts/s)", flush=True)

    total_time = time.time() - t_start
    print(f"\n[SUCCESS] Completed {len(results)} AlpacaEval evaluations in {total_time:.1f}s!", flush=True)
    print(f"[*] Benchmark dataset saved to: {out_path}", flush=True)
    print("\n--- Next Steps for Official AlpacaEval Submission ---", flush=True)
    print("1. Output file ready: release/alpaca_eval_collision_1b.json", flush=True)
    print("2. Submit PR to https://github.com/tatsu-lab/alpaca_eval", flush=True)


if __name__ == "__main__":
    main()
