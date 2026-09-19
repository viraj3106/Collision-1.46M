"""
COLLISION — OpenAI-Compatible Local REST Server.

Provides a drop-in OpenAI-compatible API server supporting:
  - POST /v1/chat/completions
  - POST /v1/completions
  - GET  /v1/models
  - GET  /health

Connect immediately with Cursor, Continue.dev, OpenWebUI, LibreChat, AutoGen, and LangChain!

Usage:
  python api_server.py --port 8000
  python api_server.py --host 0.0.0.0 --port 8000 --weights model.pt
"""

import os
import sys
import json
import time
import uuid
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
from configuration_collision import CollisionConfig
from modeling_collision import CollisionForCausalLM
from tokenization_collision import CollisionTokenizer

# Global model state
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
MODEL_ID = "collision-10M/Collision-1B"


def load_global_model(weights_path: str, config_path: str, tokenizer_dir: str):
    global GLOBAL_MODEL, GLOBAL_TOKENIZER
    print(f"[*] Initializing COLLISION-1B runtime...")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    cfg = CollisionConfig(
        vocab_size=cfg_dict.get("vocab_size", 32000),
        max_seq_len=cfg_dict.get("max_seq_len", 1024),
        d_model=cfg_dict.get("d_model", 2048),
        n_layer=cfg_dict.get("n_layer", 24),
        n_head=cfg_dict.get("n_head", 16),
        d_ff=cfg_dict.get("d_ff", 5376),
        dropout=cfg_dict.get("dropout", 0.1),
        tie_embeddings=cfg_dict.get("tie_embeddings", True)
    )

    model = CollisionForCausalLM(cfg)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Checkpoint loaded from {weights_path}")
    else:
        print(f"[WARN] Weights not found at {weights_path}, using initialized weights.")

    model.eval()
    GLOBAL_MODEL = model

    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    merges_file = os.path.join(tokenizer_dir, "merges.json")
    GLOBAL_TOKENIZER = CollisionTokenizer(vocab_file=vocab_file, merges_file=merges_file)
    print(f"[OK] Collision runtime ready for OpenAI requests.\n")


def generate_completion(prompt: str, max_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9) -> str:
    if GLOBAL_MODEL is None or GLOBAL_TOKENIZER is None:
        return "Model runtime not loaded."

    tokens = GLOBAL_TOKENIZER.encode_to_ids(prompt, bos=True)
    if not tokens:
        tokens = [GLOBAL_TOKENIZER.special_tokens_map_dict["[BOS]"]]

    input_ids = torch.tensor([tokens], dtype=torch.long)
    generated = list(tokens)

    for _ in range(max_tokens):
        if input_ids.size(1) >= GLOBAL_MODEL.config.max_seq_len:
            input_ids = input_ids[:, -GLOBAL_MODEL.config.max_seq_len:]

        with torch.no_grad():
            outputs = GLOBAL_MODEL(input_ids)
            logits = outputs.logits[0, -1, :]

        if temperature <= 0.01:
            next_token = torch.argmax(logits).item()
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == GLOBAL_TOKENIZER.special_tokens_map_dict["[EOS]"]:
            break

        generated.append(next_token)
        input_ids = torch.tensor([generated], dtype=torch.long)

    completion_ids = generated[len(tokens):]
    return GLOBAL_TOKENIZER.decode_from_ids(completion_ids, skip_special_tokens=True)


class OpenAIApiHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "healthy",
                "model": MODEL_ID,
                "parameters": "999.38M",
                "uptime": "active"
            }).encode("utf-8"))
        elif self.path == "/v1/models":
            self._set_headers(200)
            response = {
                "object": "list",
                "data": [
                    {
                        "id": MODEL_ID,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "collision",
                        "permission": [],
                        "root": MODEL_ID,
                        "parent": None
                    },
                    {
                        "id": "collision",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "collision"
                    }
                ]
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            req = json.loads(post_data.decode("utf-8"))
        except Exception:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            return

        if self.path == "/v1/chat/completions":
            messages = req.get("messages", [])
            prompt = ""
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            prompt += "Assistant: "

            max_tokens = req.get("max_tokens", 128)
            temperature = float(req.get("temperature", 0.7))
            top_p = float(req.get("top_p", 0.9))

            t0 = time.time()
            text = generate_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            elapsed = time.time() - t0

            chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            res = {
                "id": chat_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", MODEL_ID),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(text.split()),
                    "total_tokens": len(prompt.split()) + len(text.split())
                },
                "latency_seconds": round(elapsed, 4)
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif self.path == "/v1/completions":
            prompt = req.get("prompt", "")
            max_tokens = req.get("max_tokens", 128)
            temperature = float(req.get("temperature", 0.7))
            top_p = float(req.get("top_p", 0.9))

            text = generate_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            cmpl_id = f"cmpl-{uuid.uuid4().hex[:12]}"
            res = {
                "id": cmpl_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.get("model", MODEL_ID),
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(text.split()),
                    "total_tokens": len(prompt.split()) + len(text.split())
                }
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))


def run_server(host: str = "0.0.0.0", port: int = 8000, dry_run: bool = False):
    weights_path = os.path.join(CURRENT_DIR, "model.pt")
    config_path = os.path.join(CURRENT_DIR, "config.json")
    tokenizer_dir = os.path.join(CURRENT_DIR, "tokenizer")

    if not dry_run:
        load_global_model(weights_path, config_path, tokenizer_dir)

    server_address = (host, port)
    httpd = HTTPServer(server_address, OpenAIApiHandler)
    print(f"[*] COLLISION OpenAI-Compatible Server running at http://{host}:{port}")
    print(f"[*] Connect Cursor / Continue.dev / OpenWebUI to: http://localhost:{port}/v1\n")
    if dry_run:
        print("[OK] Dry run successful. Exiting.")
        return
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COLLISION OpenAI-Compatible API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--dry-run", action="store_true", help="Validate server startup and exit")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, dry_run=args.dry_run)
