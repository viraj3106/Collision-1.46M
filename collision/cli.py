"""
COLLISION Phase 99 — Production Grounded Answering CLI.

Provides unified command-line entrypoints backed by the shared Application Service Layer:
- python -m collision ask "<question>" [--mode AUTO|LOCAL|WEB|HYBRID|MODEL] [--verbose] [--json]
- python -m collision chat [--mode AUTO|LOCAL|WEB|HYBRID|MODEL] [--verbose]
"""

import os
import sys
import argparse
import json
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.service import get_collision_service, CollisionService


def handle_ask(args):
    service = get_collision_service()
    question = args.question.strip() if args.question else ""

    result = service.ask(
        question=question,
        mode=args.mode.upper(),
        include_sources=True,
        include_claims=True
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # User-facing structured presentation
    print("\nAnswer:")
    print(result.get("answer", "No answer produced."))

    sources = result.get("sources", [])
    if sources:
        print("\nSources:")
        for s in sources:
            title = s.get("title", "Document")
            url = s.get("url", "")
            print(f"- {title} ({url})")

    if args.verbose:
        print("\nExecution Diagnostics:")
        print(f"  Status     : {result.get('status')}")
        print(f"  Route Mode : {result.get('mode')}")
        print(f"  Confidence : {result.get('confidence', 0.0):.2f}")
        lat = result.get("latency", {})
        print(f"  Latency    : {lat.get('total_ms', 0.0):.2f} ms (routing: {lat.get('routing_ms', 0.0):.1f}ms, ret: {lat.get('retrieval_ms', 0.0):.1f}ms, gen: {lat.get('generation_ms', 0.0):.1f}ms, ver: {lat.get('verification_ms', 0.0):.1f}ms)")
        claims = result.get("claims", [])
        if claims:
            print("  Claims:")
            for c in claims:
                print(f"    - [{c.get('support_status')}] {c.get('text')}")
    print()


def handle_chat(args):
    service = get_collision_service()
    selected_mode = args.mode.upper()

    print("=" * 65)
    print("      COLLISION — Production Grounded Answer Chat (Phase 99)")
    print("=" * 65)
    print("Commands:")
    print("  /exit, /quit     - Exit session")
    print("  /mode <MODE>     - Set mode (AUTO, LOCAL, WEB, HYBRID, MODEL)")
    print("  /verbose         - Toggle verbose diagnostics")
    print("=" * 65 + "\n")

    verbose = args.verbose

    while True:
        try:
            user_input = input("User:\n").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit"):
                print("Goodbye!")
                break
            elif user_input.lower() == "/verbose":
                verbose = not verbose
                print(f"[Verbose diagnostics: {'ON' if verbose else 'OFF'}]\n")
                continue
            elif user_input.lower().startswith("/mode "):
                parts = user_input.split()
                if len(parts) > 1:
                    selected_mode = parts[1].upper()
                    print(f"[Mode set to {selected_mode}]\n")
                continue

            result = service.ask(question=user_input, mode=selected_mode)

            print(f"\nAssistant:\n{result.get('answer', '')}")
            sources = result.get("sources", [])
            if sources:
                print("\nSources:")
                for s in sources:
                    print(f"- {s.get('title', 'Doc')} ({s.get('url', '')})")

            if verbose:
                lat = result.get("latency", {})
                print(f"\n[Diagnostics] Status: {result.get('status')} | Mode: {result.get('mode')} | Latency: {lat.get('total_ms', 0.0):.1f}ms | Confidence: {result.get('confidence', 0.0):.2f}")
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def handle_think(args):
    service = get_collision_service()
    question = args.question.strip() if args.question else ""
    if not question:
        print("Please provide a question for deep cognitive deliberation.")
        return

    result = service.think(
        question=question,
        domain=args.domain,
        enable_dialectic=not args.no_dialectic
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # User-facing structured brain output
    print("=" * 70)
    print("      COLLISION SYNAPTIC BRAIN -- DELIBERATIVE REASONING TRACE")
    print("=" * 70)
    print(f"\nInquiry: {question}")
    print(f"Modality: {result.get('modality')} | Epistemic Certainty: {result.get('epistemic_certainty', 1.0)*100:.1f}%\n")
    print("Deliberative Resolution:")
    print(result.get("answer", "No resolution produced."))

    key_insights = result.get("key_insights", [])
    if key_insights:
        print("\nKey Metacognitive Insights:")
        for ins in key_insights:
            print(f"- {ins}")

    triples = result.get("triples", [])
    if triples and args.verbose:
        print("\nExtracted Semantic Knowledge Triples:")
        for t in triples:
            print(f"  ({t.get('subject')} --[{t.get('predicate')}]--> {t.get('object')})")

    bias_audits = result.get("bias_audits", [])
    if bias_audits and (args.verbose or getattr(args, "visualize", False)):
        print("\nMetacognitive Fallacy & Bias Audits:")
        for b in bias_audits:
            print(f"  [{b.get('severity')}] {b.get('bias_type')}: {b.get('explanation')}")

    if getattr(args, "visualize", False) and result.get("graph_ascii"):
        print("\n" + "=" * 40)
        print(result.get("graph_ascii"))
        print("=" * 40)

    if getattr(args, "mermaid", False) and result.get("graph_mermaid"):
        print("\n--- Mermaid Flowchart Diagram ---")
        print(result.get("graph_mermaid"))
        print("---------------------------------")

    followups = result.get("followup_hypotheses", [])
    if followups:
        print("\nFollowup Cognitive Hypotheses:")
        for f in followups:
            print(f"- {f}")

    if args.verbose or getattr(args, "visualize", False):
        g = result.get("graph_summary", {})
        lat = result.get("latency", {})
        print(f"\n[Brain Diagnostics] Nodes: {g.get('nodes_count')} | Edges: {g.get('edges_count')} | Merit: {result.get('merit_score', 1.0):.2f} | Latency: {lat.get('total_ms', 0.0):.1f}ms")
    print()


def main():
    parser = argparse.ArgumentParser(prog="collision", description="COLLISION Production Grounded Answering CLI")
    subparsers = parser.add_subparsers(dest="command")

    # 'ask' command
    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question")
    ask_parser.add_argument("question", type=str, nargs="?", default="", help="Question to ask")
    ask_parser.add_argument("--mode", type=str, default="AUTO", choices=["AUTO", "LOCAL", "WEB", "HYBRID", "MODEL"], help="Routing mode")
    ask_parser.add_argument("--verbose", action="store_true", help="Show execution diagnostics")
    ask_parser.add_argument("--json", action="store_true", help="Output raw JSON matching public API response schema")

    # 'think' command (Synaptic Brain)
    think_parser = subparsers.add_parser("think", help="Execute deep deliberative reasoning via COLLISION Synaptic Brain")
    think_parser.add_argument("question", type=str, nargs="?", default="", help="Complex or philosophical question to deliberate")
    think_parser.add_argument("--domain", type=str, default="General", help="Domain context")
    think_parser.add_argument("--no-dialectic", action="store_true", help="Disable Hegelian dialectical synthesis")
    think_parser.add_argument("--visualize", action="store_true", help="Render ASCII reasoning graph tree in terminal")
    think_parser.add_argument("--mermaid", action="store_true", help="Output Mermaid diagram syntax for web/markdown rendering")
    think_parser.add_argument("--verbose", action="store_true", help="Show full Graph-of-Thoughts, bias audits, and triple extractions")
    think_parser.add_argument("--json", action="store_true", help="Output raw JSON matching brain schema")

    # 'chat' command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument("--mode", type=str, default="AUTO", choices=["AUTO", "LOCAL", "WEB", "HYBRID", "MODEL"], help="Default routing mode")
    chat_parser.add_argument("--verbose", action="store_true", help="Show execution diagnostics")

    args = parser.parse_args()

    if args.command == "ask":
        handle_ask(args)
    elif args.command == "think":
        handle_think(args)
    elif args.command == "chat":
        handle_chat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
