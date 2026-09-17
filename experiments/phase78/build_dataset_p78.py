import os
import sys
import json
import random
import numpy as np
import hashlib
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_dataset_v5_p78")
REPORT_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase78", "data_report")

CATEGORIES = [
    "Explanatory Text",
    "Factual Knowledge",
    "Computer Science",
    "Artificial Intelligence",
    "Mathematics",
    "Physics",
    "Space/Science",
    "Technology",
    "Question-Answer",
    "Completion Stems",
    "Short Conversations"
]

CATEGORY_TEMPLATES = {
    "Explanatory Text": [
        "In modern system architecture, {topic} functions by isolating process execution spaces to prevent memory corruption across concurrent threads.",
        "The underlying mechanism of {topic} involves calculating loss gradients iteratively and adjusting model parameters along the steepest trajectory.",
        "Understanding {topic} requires analyzing how system components handle asynchronous event loops without blocking main execution threads."
    ],
    "Factual Knowledge": [
        "{topic} represents a core foundation established in scientific computing for managing structured data storage efficiently.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second, defining fundamental physical limits.",
        "{topic} is recognized across research domains as a key metric for evaluating convergence and generalizability."
    ],
    "Computer Science": [
        "Binary search trees organize elements hierarchically, achieving logarithmic average lookup time complexity O(log N).",
        "Operating systems utilize virtual memory paging mechanisms to abstract physical hardware RAM for running processes.",
        "Object-oriented design principles promote encapsulation, polymorphism, and modular inheritance across software systems."
    ],
    "Artificial Intelligence": [
        "Deep neural networks utilize multi-layer perceptrons and non-linear activation functions to learn complex feature representations.",
        "Transformer models rely on multi-head self-attention mechanisms to compute contextual token relationships in parallel.",
        "Reinforcement learning agents optimize policy value functions by interacting with environment reward signals over time."
    ],
    "Mathematics": [
        "Calculus establishes mathematical framework for analyzing continuous rate of change through derivatives and integrals.",
        "Linear algebra focuses on vector spaces, matrix transformations, eigenvalues, and high-dimensional coordinate mappings.",
        "Probability distributions model statistical likelihoods of random variable outcomes across continuous and discrete domains."
    ],
    "Physics": [
        "Thermodynamics dictates that entropy in an isolated system always increases over time according to the second law.",
        "Quantum mechanics describes microscopic particles using wavefunctions, superposition principles, and probability amplitudes.",
        "General relativity models gravitational attraction as the geometric curvature of four-dimensional spacetime metrics."
    ],
    "Space/Science": [
        "Planetary orbits around stars follow elliptical paths governed by Kepler's laws of celestial motion and gravitational forces.",
        "Stellar nucleosynthesis fuses lighter elements into heavier elements inside star cores during active nuclear fusion lifecycles.",
        "Cosmological redshift measurements indicate that the observable universe is expanding uniformly in all directions."
    ],
    "Technology": [
        "Distributed cloud computing platforms orchestrate containerized microservices across scalable cluster nodes dynamically.",
        "Modern microprocessor microarchitectures utilize instruction pipelining, branch prediction, and multi-level CPU caches.",
        "Cryptographic public-key infrastructure secures network communication using asymmetric encryption key pairs."
    ],
    "Question-Answer": [
        "Question: What is an operating system? Answer: An operating system is core software that manages hardware resources and application processes.",
        "Question: How does gradient descent work? Answer: Gradient descent updates model weights by moving opposite to the gradient of the loss function.",
        "Question: What is a prime number? Answer: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself."
    ],
    "Completion Stems": [
        "The primary purpose of data structure indexing is to speed up data retrieval operations by",
        "Machine learning models generalize to unseen test distributions effectively when",
        "The second law of thermodynamics establishes that the total entropy of an isolated system"
    ],
    "Short Conversations": [
        "User: Explain binary search. Assistant: Binary search finds a target value in a sorted array by repeatedly dividing the search interval in half.",
        "User: What is gravity? Assistant: Gravity is the fundamental force by which massive bodies attract one another through spacetime curvature.",
        "User: How do compilers work? Assistant: Compilers parse high-level source code, build syntax trees, optimize instructions, and emit target machine code."
    ]
}

TOPICS = [
    "memory management", "transformer attention", "distributed consensus", "concurrency control",
    "gradient optimization", "neural representations", "spacetime curvature", "asymmetric cryptography",
    "quantum superposition", "relational indexing", "pipeline execution", "vector embeddings"
]

def build_phase78_dataset(target_documents: int = 5000) -> Dict[str, Any]:
    print("Building Phase 78 Dataset (collision_dataset_v5_p78)...")
    random.seed(42)
    np.random.seed(42)
    
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    documents = []
    category_counts = {cat: 0 for cat in CATEGORIES}
    
    for i in range(target_documents):
        cat = CATEGORIES[i % len(CATEGORIES)]
        template = random.choice(CATEGORY_TEMPLATES[cat])
        topic = random.choice(TOPICS)
        text = template.format(topic=topic)
        
        documents.append({
            "id": i,
            "category": cat,
            "text": text
        })
        category_counts[cat] += 1
        
    # Deduplicate
    seen_hashes = set()
    deduped_docs = []
    for doc in documents:
        h = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_docs.append(doc)
            
    # Tokenize all documents into flat token stream
    token_stream = []
    doc_lengths = []
    for doc in deduped_docs:
        tokens = tokenizer.encode(doc["text"], bos=True, eos=True)
        token_stream.extend(tokens)
        doc_lengths.append(len(tokens))
        
    total_tokens = len(token_stream)
    val_split_index = int(total_tokens * 0.90)
    train_tokens = np.array(token_stream[:val_split_index], dtype=np.uint16)
    val_tokens = np.array(token_stream[val_split_index:], dtype=np.uint16)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_tokens.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
    val_tokens.tofile(os.path.join(OUTPUT_DIR, "val.bin"))
    
    audit_data = {
        "dataset_name": "collision_dataset_v5_p78",
        "total_documents": len(deduped_docs),
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "category_distribution": category_counts,
        "duplicate_rate_percent": round(((len(documents) - len(deduped_docs)) / len(documents)) * 100, 2),
        "avg_sequence_length": round(float(np.mean(doc_lengths)), 2),
        "min_sequence_length": int(np.min(doc_lengths)),
        "max_sequence_length": int(np.max(doc_lengths)),
        "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 8000
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "data_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        
    print(f"Dataset v5_p78 created: {total_tokens:,} total tokens ({len(train_tokens):,} train, {len(val_tokens):,} val).")
    return audit_data

if __name__ == "__main__":
    build_phase78_dataset()
