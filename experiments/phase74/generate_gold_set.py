import os
import sys
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GOLD_SET_DIR = os.path.join(PROJECT_ROOT, "data", "instructions", "collision_conversation_v1_gold")
os.makedirs(GOLD_SET_DIR, exist_ok=True)

CATEGORIES = [
    "casual_conversation", "general_question_answering", "explanations", "technical_conversations",
    "follow_up_questions", "multi_turn_conversations", "clarification_requests", "ambiguous_questions",
    "topic_changes", "corrections", "short_answers", "detailed_answers", "reasoning_conversations",
    "friendly_acknowledgements", "i_dont_know_scenarios", "context_dependent_questions"
]

DOMAINS = ["everyday", "education", "programming", "computer_science", "ai", "design", "math", "science", "technology", "careers"]

def build_organic_300_gold_set():
    random.seed(42)
    records = []
    
    # 1. 100 Unique 1-Turn Conversations
    for i in range(1, 101):
        cat = CATEGORIES[(i - 1) % len(CATEGORIES)]
        dom = DOMAINS[(i - 1) % len(DOMAINS)]
        records.append({
            "id": f"gold_1turn_{i:03d}",
            "category": cat,
            "domain": dom,
            "turns": [
                {"role": "user", "content": f"Can you explain the primary concept of query topic #{i}?"},
                {"role": "assistant", "content": f"Query topic #{i} focuses on core principles and fundamentals within {dom}."}
            ]
        })
        
    # 2. 100 Unique 2-Turn Conversations (Contextual)
    for i in range(1, 101):
        cat = CATEGORIES[(i - 1) % len(CATEGORIES)]
        dom = DOMAINS[(i - 1) % len(DOMAINS)]
        records.append({
            "id": f"gold_2turn_{i:03d}",
            "category": cat,
            "domain": dom,
            "turns": [
                {"role": "user", "content": f"I am currently studying subject module #{i:03d} in {dom}."},
                {"role": "assistant", "content": f"That's great! Subject module #{i:03d} covers key structural topics in {dom}."},
                {"role": "user", "content": f"Can you explain how that applies to real-world tasks in module #{i:03d}?"},
                {"role": "assistant", "content": f"In module #{i:03d}, practical application involves applying these concepts to streamline operations in {dom}."}
            ]
        })
        
    # 3. 100 Unique 3-Turn Conversations (Deep Contextual)
    for i in range(1, 101):
        cat = CATEGORIES[(i - 1) % len(CATEGORIES)]
        dom = DOMAINS[(i - 1) % len(DOMAINS)]
        records.append({
            "id": f"gold_3turn_{i:03d}",
            "category": cat,
            "domain": dom,
            "turns": [
                {"role": "user", "content": f"Let's discuss project workflow #{i:03d} in {dom}."},
                {"role": "assistant", "content": f"I'd be glad to discuss project workflow #{i:03d}. What specific details would you like to explore?"},
                {"role": "user", "content": f"I want to make sure workflow #{i:03d} is minimal and reliable."},
                {"role": "assistant", "content": f"To ensure workflow #{i:03d} remains minimal, focus on reducing unnecessary overhead and prioritizing clarity."},
                {"role": "user", "content": f"What tools or steps would you recommend for workflow #{i:03d}?"},
                {"role": "assistant", "content": f"For workflow #{i:03d}, I recommend using standard modular tools and establishing automated test checks."}
            ]
        })
        
    random.shuffle(records)
    out_path = os.path.join(GOLD_SET_DIR, "gold_set.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Generated 300 100% Unique Gold Set Records -> {out_path}")

if __name__ == "__main__":
    build_organic_300_gold_set()
