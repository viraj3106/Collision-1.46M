import os
import sys
import argparse
from typing import List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.answering.engine import CollisionAnsweringEngine
from collision.answering.schemas import AnswerStatus, ConversationMessage

def print_banner():
    print("=" * 60)
    print("         COLLISION — Conversational Answering CLI")
    print("                Phase 93 Foundation Engine")
    print("=" * 60)
    print("Commands:")
    print("  /exit, /quit       - Exit chat session")
    print("  /reset, /clear     - Reset conversation history")
    print("  /stats             - Toggle verbose generation stats")
    print("  /deterministic     - Toggle deterministic generation mode")
    print("  /temp <float>      - Set temperature (e.g. /temp 0.7)")
    print("  /help              - Display this help message")
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="COLLISION Interactive Conversational CLI")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to model directory containing config.json and model.pt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K token sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus sampling")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic greedy generation")
    parser.add_argument("--verbose", action="store_true", help="Print detailed latency and token metrics")
    args = parser.parse_args()

    print_banner()

    try:
        engine = CollisionAnsweringEngine(model_dir=args.model_dir)
        print(f"Loaded model successfully from: {engine.model_dir}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    history: List[ConversationMessage] = []
    show_stats = args.verbose
    is_deterministic = args.deterministic
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_tokens = args.max_tokens
    rep_penalty = args.repetition_penalty

    while True:
        try:
            user_input = input("User:\n").strip()
            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ("/exit", "/quit"):
                print("\nExiting COLLISION chat. Goodbye!")
                break
            elif user_input.lower() in ("/reset", "/clear"):
                history.clear()
                print("\n[Conversation history cleared]\n")
                continue
            elif user_input.lower() == "/stats":
                show_stats = not show_stats
                print(f"\n[Verbose metrics: {'ON' if show_stats else 'OFF'}]\n")
                continue
            elif user_input.lower() == "/deterministic":
                is_deterministic = not is_deterministic
                print(f"\n[Deterministic mode: {'ON' if is_deterministic else 'OFF'}]\n")
                continue
            elif user_input.lower().startswith("/temp "):
                try:
                    new_temp = float(user_input.split()[1])
                    if new_temp >= 0:
                        temperature = new_temp
                        print(f"\n[Temperature set to {temperature}]\n")
                    else:
                        print("\n[Error: Temperature must be non-negative]\n")
                except Exception:
                    print("\n[Invalid syntax. Usage: /temp 0.7]\n")
                continue
            elif user_input.lower() == "/help":
                print_banner()
                continue

            # Generate response
            result = engine.answer(
                question=user_input,
                conversation_history=history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                deterministic=is_deterministic
            )

            print("\nCOLLISION:")
            if show_stats or result.status != AnswerStatus.ANSWER:
                print(f"[{result.status.value} | Latency: {result.latency_ms:.1f}ms | Speed: {result.tokens_per_second:.1f} tok/s | Tokens: {result.completion_tokens}]")
            
            print(f"{result.text}\n")
            print("-" * 60)

            # Update conversation history
            history.append(ConversationMessage(role="user", content=user_input))
            history.append(ConversationMessage(role="assistant", content=result.text))

        except (KeyboardInterrupt, EOFError):
            print("\nExiting COLLISION chat. Goodbye!")
            break

if __name__ == "__main__":
    main()
