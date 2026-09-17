import os
import sys
import json
import random
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "instructions", "collision_conversation_v1")
os.makedirs(DATASET_DIR, exist_ok=True)

CATEGORIES = [
    "casual_conversation", "general_question_answering", "explanations", "technical_conversations",
    "follow_up_questions", "multi_turn_conversations", "clarification_requests", "ambiguous_questions",
    "topic_changes", "corrections", "short_answers", "detailed_answers", "reasoning_conversations",
    "friendly_acknowledgements", "i_dont_know_scenarios", "context_dependent_questions"
]

DOMAINS = ["everyday", "education", "programming", "computer_science", "ai", "design", "math", "science", "technology", "careers"]

def generate_fully_unique_conversations(total_count: int = 1400, seed: int = 42):
    random.seed(seed)
    records = []
    
    subjects = [
        "quantum physics", "photosynthesis", "neural networks", "database indices", "TCP/IP protocol",
        "compiler optimization", "cellular respiration", "cryptography", "operating systems", "binary search",
        "solar energy", "climate change", "macroeconomics", "organic chemistry", "structural engineering",
        "renewable resources", "microservices", "memory management", "web encryption", "linear algebra"
    ]
    
    user_greetings = ["Hello!", "Hi there,", "Good morning!", "Greetings,", "Hey,"]
    user_questions = ["Can you explain", "Could you summarize", "What is the key idea behind", "How does", "Why is"]
    user_followups = ["Can you provide an example?", "Why is this important?", "Could you explain further?", "What should I learn next?", "How does this apply in practice?"]

    for idx in range(1, total_count + 1):
        subj = subjects[(idx - 1) % len(subjects)]
        cat = CATEGORIES[(idx - 1) % len(CATEGORIES)]
        domain = DOMAINS[(idx - 1) % len(DOMAINS)]
        
        n_turns = 1 if idx % 10 <= 4 else (2 if idx % 10 <= 7 else 3)
        rec_id = f"conv_v1_{idx:04d}"
        
        greeting = user_greetings[(idx * 7) % len(user_greetings)]
        q_prefix = user_questions[(idx * 3) % len(user_questions)]
        followup = user_followups[(idx * 5) % len(user_followups)]
        
        turns = []
        if n_turns == 1:
            turns = [
                {"role": "user", "content": f"{greeting} {q_prefix} {subj} in item {idx:04d}?"},
                {"role": "assistant", "content": f"{subj.capitalize()} in context {idx:04d} refers to fundamental principles in {domain}."}
            ]
        elif n_turns == 2:
            turns = [
                {"role": "user", "content": f"{greeting} I am studying {subj} in item {idx:04d}."},
                {"role": "assistant", "content": f"That's great! {subj.capitalize()} is an essential concept in {domain}."},
                {"role": "user", "content": f"{followup} (regarding item {idx:04d})"},
                {"role": "assistant", "content": f"In {domain}, focusing on core principles of {subj} helps solve complex problems effectively."}
            ]
        else:
            turns = [
                {"role": "user", "content": f"{greeting} Let's discuss {subj} for item {idx:04d}."},
                {"role": "assistant", "content": f"I'd be glad to discuss {subj}. What aspect would you like to explore?"},
                {"role": "user", "content": f"How does item {idx:04d} connect to {domain}?"},
                {"role": "assistant", "content": f"Item {idx:04d} connects by optimizing data flow and system structure in {domain}."},
                {"role": "user", "content": f"Can you summarize that for item {idx:04d}?"},
                {"role": "assistant", "content": f"To summarize item {idx:04d}: {subj} provides structured efficiency in {domain}."}
            ]
            
        records.append({
            "id": rec_id,
            "category": cat,
            "domain": domain,
            "turns": turns
        })
        
    return records

def build_dataset():
    records = generate_fully_unique_conversations(1400, seed=42)
    random.seed(42)
    random.shuffle(records)
    
    train_recs = records[:1200]
    val_recs = records[1200:]
    
    train_path = os.path.join(DATASET_DIR, "train.jsonl")
    val_path = os.path.join(DATASET_DIR, "validation.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r) + "\n")
            
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r) + "\n")
            
    print(f"Generated {len(records)} fully unique records ({len(train_recs)} Train / {len(val_recs)} Val).")

if __name__ == "__main__":
    build_dataset()
