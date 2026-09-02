import os
import sys
import json
import random
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
OUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_instruct_v1")
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase31")

# Robust Synthetic Concept Database covering observed weaknesses
SYNTHETIC_CATEGORIES = {
    "question_answering": [
        ("What is backpropagation in neural networks?", "Backpropagation is a gradient-calculation algorithm that computes partial derivatives of loss with respect to parameters using the chain rule."),
        ("What is the difference between supervised and unsupervised learning?", "Supervised learning utilizes labeled input-target pairs, whereas unsupervised learning discovers hidden patterns in unlabeled data."),
        ("What is a loss function in machine learning?", "A loss function measures the numerical discrepancy between model predictions and ground-truth target values."),
        ("What is gradient descent?", "Gradient descent is an iterative optimization algorithm that updates weights in the direction of steepest loss reduction."),
        ("What is overfitting?", "Overfitting occurs when a statistical model memorizes noise in training data rather than learning generalizable patterns.")
    ],
    "explanation": [
        ("Explain how a transformer model works.", "A transformer model processes input sequences in parallel using self-attention mechanisms to weigh relationships between all tokens."),
        ("Explain the role of activation functions.", "Activation functions introduce non-linear transformations into neural networks, allowing them to learn complex non-linear relationships."),
        ("Explain how binary search works.", "Binary search locates a target value in a sorted array by continuously dividing the search interval in half."),
        ("Explain Newton's second law of motion.", "Newton's second law states that force equals mass times acceleration, meaning acceleration is proportional to net force."),
        ("Explain the concept of entropy in thermodynamics.", "Entropy is a thermodynamic quantity that quantifies the degree of disorder or randomness within an isolated system.")
    ],
    "computer_science": [
        ("What is Big O notation?", "Big O notation mathematically characterizes algorithmic efficiency by defining upper bounds on time or space complexity as input size grows."),
        ("Define a hash table.", "A hash table is a data structure that maps keys to values using a hashing function to enable average constant-time lookups."),
        ("What is a compiler?", "A compiler is a specialized program that translates high-level programming code into executable machine instructions."),
        ("What is a binary tree?", "A binary tree is a hierarchical data structure in which each parent node has at most two child nodes."),
        ("What is dynamic programming?", "Dynamic programming is an algorithmic technique that solves complex problems by breaking them into overlapping subproblems and caching results.")
    ],
    "physics_astronomy": [
        ("What causes the Earth's seasons?", "Earth's seasons are caused by the 23.5-degree tilt of its rotational axis as it orbits the Sun."),
        ("What is a black hole?", "A black hole is a region of spacetime with gravitational attraction so strong that nothing, including light, can escape."),
        ("What is wave-particle duality?", "Wave-particle duality is the quantum concept that light and matter exhibit properties of both continuous waves and discrete particles."),
        ("What is special relativity?", "Special relativity is Einstein's theory establishing that physical laws are identical for all inertial observers and the speed of light is constant."),
        ("What is nuclear fusion in stars?", "Nuclear fusion is the process by which light hydrogen nuclei combine under extreme heat and pressure to form helium, releasing energy.")
    ],
    "mathematics": [
        ("What is a derivative in calculus?", "A derivative represents the instantaneous rate of change of a function with respect to an input variable."),
        ("What is an integral?", "An integral calculates the accumulated area under a curve, representing the inverse operation of differentiation."),
        ("What is an eigenvalue?", "An eigenvalue is a scalar multiplier associated with a linear matrix transformation that scales corresponding eigenvectors without rotating them."),
        ("What is a prime number?", "A prime number is an integer greater than 1 that has no positive divisors other than 1 and itself."),
        ("What is standard deviation?", "Standard deviation is a statistical metric that measures the dispersion or spread of data points relative to their mean.")
    ],
    "general_knowledge": [
        ("What is photosynthesis?", "Photosynthesis is the biological process by which green plants convert solar light energy into chemical energy stored as glucose."),
        ("What is DNA?", "DNA, or deoxyribonucleic acid, is the double-stranded molecule carrying genetic instructions for living organisms."),
        ("What is plate tectonics?", "Plate tectonics is the geological theory describing the movement of large lithospheric plates across Earth's mantle."),
        ("What is an ecosystem?", "An ecosystem is a biological community of interacting organisms and their physical non-living environment."),
        ("What is the water cycle?", "The water cycle is the continuous movement of water through evaporation, condensation, precipitation, and runoff across Earth.")
    ]
}

PREFIXES = ["", "Please ", "Could you ", "I need to know: "]

def generate_synthetic_dataset(num_samples_per_item: int = 5):
    random.seed(42)
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)
    
    synthetic_records = []
    seen = set()
    
    cat_counts = {}
    
    for category, items in SYNTHETIC_CATEGORIES.items():
        cat_counts[category] = 0
        for prompt_base, resp_base in items:
            for prefix in PREFIXES:
                if prefix:
                    p = f"{prefix}{prompt_base[0].lower()}{prompt_base[1:]}"
                else:
                    p = prompt_base
                    
                key = (p.strip(), resp_base.strip())
                if key in seen:
                    continue
                seen.add(key)
                
                rec = {
                    "instruction": p.strip(),
                    "response": resp_base.strip(),
                    "category": category,
                    "source": "synthetic"
                }
                synthetic_records.append(rec)
                cat_counts[category] += 1

    # Shuffle deterministically
    random.shuffle(synthetic_records)
    
    syn_file = os.path.join(OUT_DIR, "collision_synthetic_v1.jsonl")
    with open(syn_file, "w", encoding="utf-8") as f:
        for r in synthetic_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    total_tokens = sum(len(tokenizer.encode(r["instruction"], bos=True)) + len(tokenizer.encode(r["response"], eos=True)) for r in synthetic_records)
    
    stats = {
        "dataset_name": "collision_synthetic_v1",
        "total_synthetic_examples": len(synthetic_records),
        "total_synthetic_tokens": total_tokens,
        "avg_tokens_per_example": round(total_tokens / max(1, len(synthetic_records)), 2),
        "category_distribution": cat_counts,
        "provenance_label": "source=synthetic"
    }
    
    stats_file = os.path.join(EXP_DIR, "synthetic_dataset_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Synthetic Dataset Generation Complete:")
    print(f"  Total Examples: {len(synthetic_records)} -> {syn_file}")
    print(f"  Total Tokens:   {total_tokens:,}")
    return stats

if __name__ == "__main__":
    generate_synthetic_dataset()
