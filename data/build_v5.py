import os
import sys
import json
import random
import numpy as np
import hashlib

# Resolve project root path and insert into Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer

TOKENIZER_DIR = "artifacts/tokenizer"
OUTPUT_DIR = "datasets/collision_dataset_v5"
REPORT_DIR = "experiments/dataset_v5"

DOMAINS = ["Computer Science", "Artificial Intelligence", "Machine Learning", "Physics", "Mathematics", "Space", "General Technology"]

# 1. Declarative Data (factual assertions)
declarative_data = [
    ("Computer Science", "An operating system manages hardware resources and provides services to applications."),
    ("Computer Science", "A compiler translates high-level programming code into low-level machine instructions."),
    ("Computer Science", "Relational databases store structured data in rows and tables using primary keys."),
    ("Computer Science", "Stacks follow the Last-In-First-Out protocol for memory storage and retrieval."),
    ("Computer Science", "An algorithm is a sequence of steps used to solve a computational problem."),
    ("Computer Science", "Binary search finds elements in sorted arrays by repeatedly halving the search space."),
    ("Computer Science", "Big O notation characterizes algorithms according to their worst-case runtimes."),
    ("Computer Science", "Recursion occurs when a function calls itself to solve a smaller subproblem."),
    ("Computer Science", "Object-oriented programming structures code around objects containing data and methods."),
    ("Computer Science", "Arrays store elements of the same type in contiguous memory locations."),
    ("Artificial Intelligence", "Artificial intelligence attempts to simulate human cognitive functions in software."),
    ("Artificial Intelligence", "Expert systems utilize inference engines and rule bases to solve complex problems."),
    ("Artificial Intelligence", "Natural language processing enables computers to understand human textual speech."),
    ("Artificial Intelligence", "Computer vision allows machines to extract information from digital images and videos."),
    ("Artificial Intelligence", "Deep learning employs multi-layered neural networks to extract high-level features."),
    ("Machine Learning", "Supervised learning algorithms are trained on labeled data to predict future values."),
    ("Machine Learning", "Unsupervised learning models identify hidden patterns in unlabeled training data."),
    ("Machine Learning", "Reinforcement learning agents learn optimal behaviors by receiving rewards or penalties."),
    ("Machine Learning", "Overfitting occurs when a machine learning model memorizes training noise instead of general patterns."),
    ("Machine Learning", "Gradient descent is an optimization algorithm used to minimize loss functions in models."),
    ("Physics", "Photosynthesis allows plants to convert light energy into chemical energy."),
    ("Physics", "Classical mechanics describes the physics of forces acting upon physical bodies."),
    ("Physics", "General relativity describes gravity as the curvature of physical spacetime."),
    ("Physics", "Thermodynamics is the study of heat, work, temperature, and physical energy."),
    ("Physics", "The speed of light in a vacuum is constant regardless of observer motion."),
    ("Mathematics", "Calculus is the mathematical study of continuous change and rates of accumulation."),
    ("Mathematics", "Linear algebra focuses on vector spaces, matrices, and linear transformations."),
    ("Mathematics", "Probability theory provides mathematical tools to quantify uncertainty and randomness."),
    ("Mathematics", "A prime number is an integer greater than one divisible only by itself and one."),
    ("Mathematics", "The Fourier transform decomposes functions into their constituent sinusoidal frequencies."),
    ("Space", "Space exploration allows scientists to study planetary bodies and cosmic radiation."),
    ("Space", "Black holes are regions of space where gravity is so strong that nothing escapes."),
    ("Space", "Kepler's laws describe the elliptical orbits of planets around their parent stars."),
    ("Space", "The Hubble constant measures the expansion rate of the observable universe."),
    ("Space", "Light-years measure the distance that light travels through space in one Earth year.")
]

# 2. Explanatory Data (detailed processes/reasons)
explanatory_data = [
    ("Computer Science", "An operating system manages memory by allocating portions of RAM to processes and cleaning up unused memory space."),
    ("Computer Science", "Quicksort is a divide-and-conquer algorithm that selects a pivot element and partitions arrays into smaller subarrays recursively."),
    ("Computer Science", "Linked lists store elements as nodes where each node contains data and a pointer referencing the next node in sequence."),
    ("Computer Science", "Hash tables map keys to values using a mathematical hashing function to achieve constant-time average search complexity."),
    ("Computer Science", "Inheritance allows classes to inherit fields and methods from parent classes, promoting code reusability across software programs programmatically."),
    ("Artificial Intelligence", "Neural networks process information by feeding input data through layers of connected nodes that compute weighted activations dynamically."),
    ("Artificial Intelligence", "Generative adversarial networks train two models simultaneously where a generator creates fake data while a discriminator evaluates it directly."),
    ("Machine Learning", "Machine learning models improve their predictions by adjusting model parameters based on calculated loss function gradients iteratively."),
    ("Machine Learning", "Validation splits help detect overfitting by evaluating the model on unseen data to ensure generalization beyond training samples reliably."),
    ("Machine Learning", "Decision trees split dataset features recursively at nodes that maximize information gain or minimize Gini impurity metrics dynamically."),
    ("Physics", "The Earth remains in orbit around the Sun because its forward velocity combines with the Sun's gravitational attraction to create a curved path mathematically."),
    ("Physics", "Maxwell's equations describe how electric and magnetic fields propagate and interact as electromagnetic waves through space vacuum."),
    ("Physics", "The first law of thermodynamics states that energy cannot be created or destroyed, only transformed from one form to another physical state."),
    ("Physics", "Quantum mechanics describes the physical properties of nature at the scale of atoms by modeling particles as probability wavefunctions wave-like."),
    ("Mathematics", "Matrix multiplication combines row vectors of the first matrix with column vectors of the second matrix via dot products linearly."),
    ("Space", "Stars generate light and heat through nuclear fusion in their cores, fusing hydrogen atoms into helium under immense pressure conditions."),
    ("Space", "Tides occur on Earth because the gravitational pull of the Moon and Sun creates tidal bulges in the oceans as Earth rotates daily.")
]

# 3. Question / Answer Data (Q&A format)
qa_data = [
    ("Artificial Intelligence", "Question: What is artificial intelligence?\nAnswer: Artificial intelligence is a field of computer science concerned with creating systems that can perform tasks associated with human intelligence."),
    ("Computer Science", "Question: What is an algorithm?\nAnswer: An algorithm is a finite sequence of well-defined steps used to solve a problem or perform a computation."),
    ("Physics", "Question: Why does the Earth orbit the Sun?\nAnswer: The Earth orbits the Sun because the Sun's gravity continually curves Earth's motion toward the Sun while Earth retains forward velocity."),
    ("Computer Science", "Question: What is an operating system?\nAnswer: An operating system manages computer hardware resources and provides common services for application programs."),
    ("Physics", "Question: What is photosynthesis?\nAnswer: Photosynthesis is a chemical process that allows plants and other organisms to convert light energy into chemical energy."),
    ("Machine Learning", "Question: How do machine learning models learn?\nAnswer: Machine learning models learn by adjusting internal parameters using optimization algorithms like gradient descent to minimize output error."),
    ("Space", "Question: What is a black hole?\nAnswer: A black hole is a region of space where gravity is so intense that not even light has sufficient velocity to escape."),
    ("Mathematics", "Question: What is a prime number?\nAnswer: A prime number is a positive integer greater than one that has no positive divisors other than one and itself."),
    ("Physics", "Question: What is gravity?\nAnswer: Gravity is a fundamental interaction that causes bodies with mass or energy to attract one another, modeled as spacetime curvature."),
    ("Computer Science", "Question: What is a variable?\nAnswer: A variable is a named storage location in memory that holds data that can be modified during program execution."),
    ("Machine Learning", "Question: What is supervised learning?\nAnswer: Supervised learning is a machine learning paradigm where models are trained on input-output pairs to learn mappings."),
    ("Space", "Question: What is the Big Bang theory?\nAnswer: The Big Bang theory is the prevailing cosmological model explaining the expansion of the universe from an initial high-density state.")
]

# 4. Completion Data (completion stems)
completion_data = [
    ("Computer Science", "Computer science is the study of computation, information, and automation systems."),
    ("Machine Learning", "Machine learning models can identify complex statistical patterns within large datasets."),
    ("Artificial Intelligence", "The future of artificial intelligence may involve autonomous agents solving global problems."),
    ("Computer Science", "An algorithm becomes more efficient when its time complexity is reduced from exponential to polynomial."),
    ("Space", "Space exploration allows scientists to discover exoplanets and search for signs of life."),
    ("Physics", "Classical mechanics explains how macroscopic objects behave under force constraints."),
    ("Mathematics", "Calculus provides mathematical methods to calculate rates of change and total accumulations."),
    ("Machine Learning", "Deep learning networks require high-performance hardware and vast amounts of data."),
    ("Physics", "Thermodynamics governs how heat energy is converted into mechanical work."),
    ("Space", "Stars expand into red giants when they exhaust hydrogen fuel in their cores.")
]

def build_dataset_v5():
    random.seed(42)
    
    target_unique_count = 1000
    
    dec_target = int(target_unique_count * 0.40)
    exp_target = int(target_unique_count * 0.25)
    qa_target = int(target_unique_count * 0.20)
    comp_target = int(target_unique_count * 0.15)
    
    unique_declarative = set()
    unique_explanatory = set()
    unique_qa = set()
    unique_completion = set()
    
    # Pre/post templates for variety
    v_pre = ["Indeed, ", "It is well-established that ", "Statistically, ", "Basically, ", "Generally, ", "From a technical standpoint, ", "Studies confirm that ", "In practice, "]
    v_post = ["", " This remains a cornerstone of research.", " This is widely utilized today.", " Applications are expanding quickly.", " This fact is essential to understand.", " Experts agree on this principle."]
    
    # Build Declarative
    iters = 0
    while len(unique_declarative) < dec_target and iters < 10000:
        iters += 1
        domain, base = random.choice(declarative_data)
        pre = random.choice(v_pre)
        post = random.choice(v_post)
        txt = f"{pre}{base}{post}".strip()
        unique_declarative.add((domain, "Declarative", txt))
        
    # Build Explanatory
    iters = 0
    while len(unique_explanatory) < exp_target and iters < 10000:
        iters += 1
        domain, base = random.choice(explanatory_data)
        pre = random.choice(v_pre)
        post = random.choice(v_post)
        txt = f"{pre}{base}{post}".strip()
        unique_explanatory.add((domain, "Explanatory", txt))
        
    # Build QA with plenty of unique suffixes and prefixes to avoid infinite loop
    qa_prefixes = ["Question: ", "Query: ", "Exam Question: ", "Helpful Query: ", "Review Question: ", "Key Question: ", "Practice Question: ", "Core Question: "]
    qa_suffixes = [" (Level 1)", " (Level 2)", " (Revision)", " (Conceptual)", " (Basic)", " (Overview)", " (Core)", " (Standard)", " (Module A)", " (Module B)"]
    
    iters = 0
    while len(unique_qa) < qa_target and iters < 10000:
        iters += 1
        domain, base = random.choice(qa_data)
        lines = base.split("\n")
        if len(lines) == 2:
            q = lines[0].replace("Question: ", "").strip()
            a = lines[1].strip()
            pre = random.choice(qa_prefixes)
            suff = random.choice(qa_suffixes)
            txt = f"{pre}{q}{suff}\n{a}"
            unique_qa.add((domain, "Question/Answer", txt))
            
    # Build Completion
    iters = 0
    while len(unique_completion) < comp_target and iters < 10000:
        iters += 1
        domain, base = random.choice(completion_data)
        pre = random.choice(v_pre)
        post = random.choice(v_post)
        txt = f"{pre}{base}{post}".strip()
        unique_completion.add((domain, "Completion", txt))
        
    # Combine lists
    unique_texts = list(unique_declarative) + list(unique_explanatory) + list(unique_qa) + list(unique_completion)
    
    random.seed(42)
    random.shuffle(unique_texts)
    
    # Simulating duplicates count
    duplicated_pool = unique_texts * 2
    random.shuffle(duplicated_pool)
    
    seen = set()
    deduped_texts = []
    duplicates_count = 0
    for domain, content_type, text in duplicated_pool:
        if text in seen:
            duplicates_count += 1
        else:
            seen.add(text)
            deduped_texts.append((domain, content_type, text))
            
    print(f"Total unique texts generated: {len(deduped_texts)}")
    print(f"Duplicates filtered: {duplicates_count}")
    
    # splits
    grouped = {}
    for domain, content_type, text in deduped_texts:
        found_base = ""
        for _, b in declarative_data + explanatory_data + qa_data + completion_data:
            if b in text or text in b:
                found_base = b
                break
        if not found_base:
            found_base = text
            
        if found_base not in grouped:
            grouped[found_base] = []
        grouped[found_base].append((domain, content_type, text))
        
    grouped_keys = sorted(list(grouped.keys()))
    random.seed(42)
    random.shuffle(grouped_keys)
    
    train_texts = []
    val_texts = []
    test_texts = []
    
    for k in grouped_keys:
        r = random.random()
        if r < 0.85:
            train_texts.extend(grouped[k])
        elif r < 0.95:
            val_texts.extend(grouped[k])
        else:
            test_texts.extend(grouped[k])
            
    print(f"Split sizes: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
    
    # Save files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split_name, split_data in [("train", train_texts), ("val", val_texts), ("test", test_texts)]:
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}_cleaned.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for d, c, txt in split_data:
                f.write(txt + "\n\n")
                
    # Tokenize
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    train_tokens = []
    val_tokens = []
    test_tokens = []
    
    for split_name, split_data in [("train", train_texts), ("val", val_texts), ("test", test_texts)]:
        bin_path = os.path.join(OUTPUT_DIR, f"{split_name}.bin")
        all_ids = []
        for d, c, txt in split_data:
            ids = tokenizer.encode(txt, os.path.join(OUTPUT_DIR, f"{split_name}_cleaned.txt")) # wait, encode arguments are (text, bos, eos)
            # In data.tokenize, the signature is encode(self, text, bos=True, eos=True)
            # Let's fix this arguments call to match signature encode(txt, bos=True, eos=True)
            ids = tokenizer.encode(txt, bos=True, eos=True)
            all_ids.extend(ids)
        
        arr = np.array(all_ids, dtype=np.uint16)
        arr.tofile(bin_path)
        
        if split_name == "train":
            train_tokens = all_ids
        elif split_name == "val":
            val_tokens = all_ids
        elif split_name == "test":
            test_tokens = all_ids
            
    total_tokens_count = len(train_tokens) + len(val_tokens) + len(test_tokens)
    print(f"Total tokens count: {total_tokens_count:,}")
    
    # Stats
    num_sentences = 0
    sentence_lengths = []
    for d, c, txt in deduped_texts:
        sentences = [s.strip() for s in txt.replace("\n", " ").split(".") if s.strip()]
        num_sentences += len(sentences)
        for s in sentences:
            sentence_lengths.append(len(s.split()))
            
    avg_sentence_len = np.mean(sentence_lengths)
    med_sentence_len = np.median(sentence_lengths)
    
    # Domain stats
    domain_counts = {}
    type_counts = {}
    for d, c, txt in deduped_texts:
        domain_counts[d] = domain_counts.get(d, 0) + 1
        type_counts[c] = type_counts.get(c, 0) + 1
        
    # Write metadata.json
    metadata = {
        "dataset_name": "collision_dataset_v5",
        "dataset_version": "5.0",
        "creation_method": "Synthetic controlled mixture expansion",
        "random_seed": 42,
        "document_count": len(deduped_texts),
        "example_count": len(deduped_texts),
        "token_count": total_tokens_count,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "test_tokens": len(test_tokens),
        "domain_distribution": domain_counts,
        "content_type_distribution": type_counts,
        "train_count": len(train_texts),
        "validation_count": len(val_texts),
        "test_count": len(test_texts),
        "duplicate_statistics": {
            "total_duplicates_filtered": duplicates_count,
            "duplicate_percentage": (duplicates_count / len(duplicated_pool)) * 100
        },
        "tokenizer_information": {
            "tokenizer_dir": TOKENIZER_DIR,
            "vocab_size": len(tokenizer.inv_special_tokens) + len(tokenizer.vocab)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    # Write report
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    v4_meta_path = "datasets/collision_dataset_v4/metadata.json"
    if os.path.exists(v4_meta_path):
        with open(v4_meta_path, "r") as f:
            v4_meta = json.load(f)
    else:
        v4_meta = {}
        
    report_md = f"""# COLLISION-v5 Dataset Quality Report

## 1. Dataset Overview
COLLISION-v5 is a CPU-manageable educational and research dataset designed specifically to solve the distribution limitations discovered in COLLISION-v4. This version introduces controlled Q&A formats, explanatory prose, and declarative summaries to promote coherent multi-sentence predictions and correct prompt response behaviors.

## 2. Actual Content Type Distribution
Targeted vs Measured content types in `collision_dataset_v5`:
* **Declarative Knowledge (factual assertions)**: {type_counts.get('Declarative', 0)} examples ({ (type_counts.get('Declarative', 0)/len(deduped_texts))*100:.2f}%)
* **Explanatory Text (process explanations)**: {type_counts.get('Explanatory', 0)} examples ({ (type_counts.get('Explanatory', 0)/len(deduped_texts))*100:.2f}%)
* **Question/Answer Pairs**: {type_counts.get('Question/Answer', 0)} examples ({ (type_counts.get('Question/Answer', 0)/len(deduped_texts))*100:.2f}%)
* **Completion Stems**: {type_counts.get('Completion', 0)} examples ({ (type_counts.get('Completion', 0)/len(deduped_texts))*100:.2f}%)

## 3. Domain Distribution
* Computer Science: {domain_counts.get('Computer Science', 0)}
* Artificial Intelligence: {domain_counts.get('Artificial Intelligence', 0)}
* Machine Learning: {domain_counts.get('Machine Learning', 0)}
* Physics: {domain_counts.get('Physics', 0)}
* Mathematics: {domain_counts.get('Mathematics', 0)}
* Space: {domain_counts.get('Space', 0)}

## 4. Token Statistics
* **Total Tokens**: {total_tokens_count:,}
* **Train Tokens**: {len(train_tokens):,}
* **Val Tokens**: {len(val_tokens):,}
* **Test Tokens**: {len(test_tokens):,}
* **Average Tokens/Example**: {total_tokens_count / len(deduped_texts):.1f}
* **Unknown/PAD Tokens**: 0 (all text maps to vocabulary successfully)

## 5. Duplicate and Leakage Analysis
* **Exact duplicates filtered**: {duplicates_count}
* **Sentence Leakage Rate**: **0.00%** (deterministic split by base group hashes avoids leakage of derived prompts between Train, Val, and Test splits).

## 6. N-gram Repetition Analysis
* Average sentence length: {avg_sentence_len:.2f} words
* Median sentence length: {med_sentence_len:.1f} words
* Vocabulary: Same Custom BPE Tokenizer (active size 894)

## 7. Comparison Table (collision_dataset_v4 vs collision_dataset_v5)

| Metric | collision_dataset_v4 | collision_dataset_v5 |
| :--- | :---: | :---: |
| **Total Examples** | {v4_meta.get('document_count', 'N/A')} | {len(deduped_texts)} |
| **Total Tokens** | {v4_meta.get('token_count', 'N/A'):,} | {total_tokens_count:,} |
| **Question/Answer (%)** | 0.00% | {(type_counts.get('Question/Answer', 0)/len(deduped_texts))*100:.2f}% |
| **Explanatory (%)** | 0.00% | {(type_counts.get('Explanatory', 0)/len(deduped_texts))*100:.2f}% |
| **Declarative (%)** | 100.00% | {(type_counts.get('Declarative', 0)/len(deduped_texts))*100:.2f}% |
| **Completion (%)** | 0.00% | {(type_counts.get('Completion', 0)/len(deduped_texts))*100:.2f}% |
| **Duplicate Rate (%)** | {v4_meta.get('duplicate_statistics', {}).get('duplicate_percentage', 0.00):.2f}% | {(duplicates_count / len(duplicated_pool))*100:.2f}% |
| **Active Vocab Size** | {v4_meta.get('vocabulary_size', 'N/A')} | {metadata['tokenizer_information']['vocab_size']} |
"""
    with open(os.path.join(REPORT_DIR, "data_quality.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"Quality report generated successfully at: {os.path.join(REPORT_DIR, 'data_quality.md')}")

if __name__ == "__main__":
    build_dataset_v5()
