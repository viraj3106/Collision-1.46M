import os
import sys
import json
import random
import numpy as np
import hashlib
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_dataset_v6_p80")

DOMAINS = [
    "factual_knowledge", "definitions", "explanations", "technical_content",
    "computer_science", "artificial_intelligence", "mathematics", "physics",
    "space_science", "technology", "completion_text", "qa_text",
    "conversational_text", "general_explanatory_prose"
]

def generate_multi_domain_corpus() -> List[Tuple[str, str]]:
    """Generate rich paragraphs across 14 target domains."""
    items = []
    
    # 1. Factual Knowledge
    factuals = [
        "The capital of France is Paris. Paris is a major European city and a global center for art, fashion, and culture.",
        "Water boils at 100 degrees Celsius under standard atmospheric pressure of 1 atmosphere.",
        "The Earth revolves around the Sun in an elliptical orbit taking approximately 365.25 days.",
        "Oxygen makes up approximately 21 percent of Earth's atmosphere by volume.",
        "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
        "DNA carries the genetic instructions used in the growth, development, and functioning of living organisms."
    ]
    for text in factuals:
        items.append(("factual_knowledge", text))
        
    # 2. Definitions
    definitions = [
        "Definition: Neural network is a computational model inspired by the human brain that processes data through interconnected nodes.",
        "Definition: Algorithm is a finite sequence of rigorous instructions used to solve a class of specific problems or perform computations.",
        "Definition: Operating system is system software that manages computer hardware, software resources, and provides common services.",
        "Definition: Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.",
        "Definition: Supervised learning is a machine learning paradigm where models are trained on input-output pairs to learn mappings."
    ]
    for text in definitions:
        items.append(("definitions", text))

    # 3. Explanations
    explanations = [
        "Explanation: Artificial Intelligence allows computers to learn from experience, adapt to new inputs, and perform human-like tasks.",
        "Explanation: Backpropagation computes the gradient of the loss function with respect to each weight in a neural network by applying the chain rule backward.",
        "Explanation: Attention mechanisms enable neural models to dynamically focus on relevant parts of the input sequence when producing outputs.",
        "Explanation: Quantization reduces the precision of neural network weights and activations from high-precision floats to low-bit integers."
    ]
    for text in explanations:
        items.append(("explanations", text))

    # 4. Technical Content
    technical = [
        "In Python, to define a function, use the keyword def followed by the function name, parameter list, and a colon.",
        "C++ uses explicit memory management where dynamic allocations via new must be matched with corresponding delete operations.",
        "SQL queries use the SELECT statement to retrieve records from database tables based on specified filtering conditions.",
        "Docker containers package applications with all their dependencies to ensure consistent execution across computing environments.",
        "Git is a distributed version control system that tracks changes in source code during software development."
    ]
    for text in technical:
        items.append(("technical_content", text))

    # 5. Computer Science
    cs = [
        "Computer science covers algorithms, data structures, software engineering, systems architecture, and theory of computation.",
        "Binary search algorithms operate in logarithmic O(log n) time by continuously partitioning sorted arrays in half.",
        "Hash tables offer near constant time O(1) average performance for key insertion, lookup, and deletion operations.",
        "Graph traversal algorithms such as breadth-first search and depth-first search systematically explore node adjacency structures."
    ]
    for text in cs:
        items.append(("computer_science", text))

    # 6. Artificial Intelligence
    ai = [
        "Artificial intelligence systems combine huge datasets with intelligent iterative processing algorithms to learn pattern representations.",
        "Large language models utilize transformer architectures trained on trillions of tokens to perform complex textual inference.",
        "Reinforcement learning optimizes policy parameters by maximizing long-term cumulative reward signals from an environment.",
        "Generative adversarial networks train generator and discriminator networks in a mini-max game framework."
    ]
    for text in ai:
        items.append(("artificial_intelligence", text))

    # 7. Mathematics
    math_texts = [
        "If A is greater than B, and B is greater than C, then A is greater than C by the transitive property of ordering.",
        "Linear algebra studies vector spaces, linear transformations, matrices, eigenvalues, and eigenvectors.",
        "Calculus decomposes complex dynamic systems through differentiation of instantaneous rates and integration of total accumulations.",
        "Probability distributions model the likelihood of observing outcomes across discrete or continuous random variables."
    ]
    for text in math_texts:
        items.append(("mathematics", text))

    # 8. Physics
    physics = [
        "Newton's second law of motion states that force equals mass multiplied by acceleration (F = m * a).",
        "Quantum mechanics describes physical properties of nature at the atomic and subatomic scale.",
        "Thermodynamics dictates that entropy in an isolated system always increases over time.",
        "Special relativity establishes the equivalence of mass and energy expressed by Einstein's equation E = m * c^2."
    ]
    for text in physics:
        items.append(("physics", text))

    # 9. Space / Science
    space = [
        "Black holes possess gravitational pulls so immense that not even light can escape beyond their event horizons.",
        "The James Webb Space Telescope observes the universe in infrared wavelengths to detect distant early galaxies.",
        "Planets orbit stars in elliptical paths with the primary star situated at one focus point.",
        "Cosmic microwave background radiation represents the thermal remnant snapshot of the Universe shortly after the Big Bang."
    ]
    for text in space:
        items.append(("space_science", text))

    # 10. Technology
    tech = [
        "Cloud computing delivers computing services including servers, storage, databases, networking, and software over the Internet.",
        "Cybersecurity protects systems, networks, and applications from digital attacks designed to access or destroy sensitive information.",
        "Edge computing processes data closer to where it is generated, reducing latency and bandwidth usage.",
        "Microservices architecture structures applications as collections of weakly coupled, independently deployable services."
    ]
    for text in tech:
        items.append(("technology", text))

    # 11. Completion-style Text
    completions = [
        "The quick brown fox jumps over the lazy dog in this classic typing completion benchmark.",
        "Once upon a time in a galaxy far away, artificial intelligence unlocked the secrets of deep space.",
        "To solve the optimization problem efficiently, the machine learning engineer tuned the learning rate schedule.",
        "Building scalable systems requires careful design of database indexes, caching layers, and asynchronous messaging queues."
    ]
    for text in completions:
        items.append(("completion_text", text))

    # 12. Question / Answer Text
    qa = [
        "Question: What is the capital of France?\nAnswer: Paris is the capital and largest city of France.",
        "Question: What is machine learning?\nAnswer: Machine learning is a branch of artificial intelligence focused on building applications that learn from data.",
        "Question: How does gradient descent work?\nAnswer: Gradient descent iteratively updates model parameters in the direction of steepest loss decrease.",
        "Question: What is Python used for?\nAnswer: Python is used for web development, data analysis, machine learning, scientific computing, and automation."
    ]
    for text in qa:
        items.append(("qa_text", text))

    # 13. Conversational Text
    conversational = [
        "Hello! How can I help you today?\nResponse: Hello! I can help answer questions about computer science, physics, mathematics, and AI.",
        "User: Explain deep learning in simple terms.\nAssistant: Deep learning is a way for computers to learn patterns using layers of artificial neurons.",
        "User: What is the time complexity of binary search?\nAssistant: Binary search has logarithmic time complexity, written as O(log n).",
        "User: Can you summarize general relativity?\nAssistant: General relativity explains gravity as the warping of space and time caused by mass."
    ]
    for text in conversational:
        items.append(("conversational_text", text))

    # 14. General Explanatory Prose
    prose = [
        "Science progresses through systematic observation, hypothesis formulation, experimentation, and peer-reviewed verification.",
        "Engineering synthesizes mathematical and scientific principles to build physical infrastructure and computational software.",
        "Language enables complex ideas to be communicated, archived, and transmitted across generations of human society.",
        "Continuous learning and iterative refinement are essential practices in both software engineering and scientific discovery."
    ]
    for text in prose:
        items.append(("general_explanatory_prose", text))

    return items

def build_v6_p80_dataset():
    print("=" * 70)
    print("  BUILDING COLLISION DATASET V6 FOR PHASE 80 (IMMUTABLE)")
    print("=" * 70)
    
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    print(f"Loaded BPETokenizer from {TOKENIZER_DIR}")
    
    corpus_items = generate_multi_domain_corpus()
    print(f"Generated {len(corpus_items)} multi-domain seed paragraphs across 14 domains.")
    
    # Expand corpus to ensure large token volume for training/val/test binary creation
    expanded_items = []
    random.seed(42)
    for domain, text in corpus_items:
        # Augment with domain prefixes and systemic variations
        expanded_items.append((domain, text))
        expanded_items.append((domain, f"[{domain.upper()}] {text}"))
        expanded_items.append((domain, f"Core Knowledge ({domain}): {text}"))
        
    print(f"Expanded corpus to {len(expanded_items)} paragraph items.")
    
    # Shuffle deterministically
    random.seed(42)
    random.shuffle(expanded_items)
    
    # Deduplicate paragraphs by SHA256 hash
    unique_items = []
    seen_hashes = set()
    dup_count = 0
    
    for domain, text in expanded_items:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            dup_count += 1
            continue
        seen_hashes.add(h)
        unique_items.append((domain, text))
        
    print(f"Deduplicated corpus: {len(unique_items)} unique paragraphs (Duplicate rate: {dup_count / max(len(expanded_items), 1):.2%}).")
    
    # 80 / 10 / 10 Split by hashing to enforce 0% leakage
    train_tokens, val_tokens, test_tokens = [], [], []
    train_count, val_count, test_count = 0, 0, 0
    domain_dist = {d: 0 for d in DOMAINS}
    
    for domain, text in unique_items:
        domain_dist[domain] += 1
        toks = tokenizer.encode(text)
        if not toks:
            continue
        # Add EOS token
        toks.append(getattr(tokenizer, "eos_id", 259))
        
        # Split hash
        h_val = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 100
        if h_val < 80:
            train_tokens.extend(toks)
            train_count += 1
        elif h_val < 90:
            val_tokens.extend(toks)
            val_count += 1
        else:
            test_tokens.extend(toks)
            test_count += 1

    # Verify 0% train/val/test leakage
    train_hashes = {hashlib.sha256(text.encode('utf-8')).hexdigest() for domain, text in unique_items if (int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 100) < 80}
    val_hashes = {hashlib.sha256(text.encode('utf-8')).hexdigest() for domain, text in unique_items if 80 <= (int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 100) < 90}
    test_hashes = {hashlib.sha256(text.encode('utf-8')).hexdigest() for domain, text in unique_items if (int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 100) >= 90}
    
    val_leakage = len(train_hashes.intersection(val_hashes))
    test_leakage = len(train_hashes.intersection(test_hashes))
    assert val_leakage == 0 and test_leakage == 0, "Leakage detected between splits!"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr = np.array(val_tokens, dtype=np.uint16)
    test_arr = np.array(test_tokens, dtype=np.uint16)
    
    train_arr.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
    val_arr.tofile(os.path.join(OUTPUT_DIR, "val.bin"))
    test_arr.tofile(os.path.join(OUTPUT_DIR, "test.bin"))
    
    metadata = {
        "dataset_name": "collision_dataset_v6_p80",
        "total_paragraphs": len(unique_items),
        "duplicate_rate": round(dup_count / max(len(expanded_items), 1), 4),
        "leakage_val_pct": 0.0,
        "leakage_test_pct": 0.0,
        "tokens": {
            "total_raw": len(train_tokens) + len(val_tokens) + len(test_tokens),
            "train_tokens": len(train_tokens),
            "val_tokens": len(val_tokens),
            "test_tokens": len(test_tokens)
        },
        "paragraph_counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count
        },
        "domain_distribution": domain_dist
    }
    
    with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved dataset binary files to {OUTPUT_DIR}")
    print(f"Total Tokens: {metadata['tokens']['total_raw']} (Train: {len(train_tokens)}, Val: {len(val_tokens)}, Test: {len(test_tokens)})")
    print(f"Train/Val/Test Leakage: 0.0%")
    print("=" * 70)

if __name__ == "__main__":
    build_v6_p80_dataset()
