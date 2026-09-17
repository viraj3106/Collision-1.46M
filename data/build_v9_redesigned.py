import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import random
import numpy as np
import hashlib

from data.tokenize import BPETokenizer

TOKENIZER_DIR = "artifacts/tokenizer"
OUTPUT_DIR = "datasets/collision_dataset_v9_redesigned"

DOMAINS = [
    "Computer Science", "Artificial Intelligence", "Machine Learning",
    "Physics", "Mathematics", "Space", "Software Engineering", "Web Development"
]

# Rich, natural diversity templates & corpus generators
FACTUAL_TEMPLATES = [
    "The core principle of {concept} involves {mechanism}. In modern systems, this ensures {benefit}.",
    "When analyzing {concept}, engineers observe that {mechanism}. This behavior directly impacts {benefit}.",
    "{concept} is widely understood as {definition}. Consequently, it allows applications to {benefit}.",
    "In {domain}, {concept} plays a crucial role by {mechanism}, which ultimately leads to {benefit}."
]

EXPLANATORY_TEMPLATES = [
    "To understand how {concept} works in practice, consider the underlying architecture. First, system components initialize state parameters. Next, {mechanism}. Finally, this achieves {benefit}.",
    "Detailed analysis of {concept} demonstrates that performance scales according to key operational factors. Specifically, {mechanism}. As a result, {benefit} can be observed in real-world benchmarks.",
    "A fundamental query in {domain} is why {concept} functions efficiently. The reason stems from how {mechanism}, thereby enabling {benefit} under heavy workloads."
]

QA_TEMPLATES = [
    ("What is {concept}?", "Answer: {concept} refers to {definition}. Its primary utility is to {benefit}."),
    ("How does {concept} operate?", "Answer: {concept} operates by {mechanism}. This guarantees {benefit}."),
    ("Why is {concept} important in {domain}?", "Answer: It is essential because {mechanism}, which provides {benefit}."),
    ("What is the difference between {concept} and conventional approaches?", "Answer: Unlike standard methods, {concept} focuses on {mechanism}, achieving {benefit}.")
]

INSTRUCTION_TEMPLATES = [
    ("Instruction: Explain how to optimize {concept}.\nResponse: To optimize {concept}, ensure that {mechanism}. This reduces overhead and enhances {benefit}.", "instruction"),
    ("Instruction: Summarize the key advantage of {concept} in {domain}.\nResponse: The main advantage of {concept} is that {mechanism}, ensuring {benefit}.", "instruction"),
    ("Instruction: Describe the primary use case of {concept}.\nResponse: {concept} is primarily used when systems require {benefit}, accomplished by {mechanism}.", "instruction")
]

CONVERSATIONAL_TEMPLATES = [
    ("User: Can you explain {concept} simply?\nAssistant: Sure! {concept} is basically {definition}. It works because {mechanism}, making it easier to {benefit}.", "dialogue"),
    ("User: Why should I use {concept} over other methods?\nAssistant: Great question! Using {concept} allows you to {benefit} because it specifically {mechanism}.", "dialogue")
]

CONTEXT_QA_TEMPLATES = [
    ("Context: {context_text}\nQuestion: What is the main benefit of {concept} described in the context?\nAnswer: According to the passage, the main benefit of {concept} is {benefit}.", "context_qa"),
    ("Context: In recent studies on {domain}, researchers analyzed {concept}. The documentation notes that {mechanism}.\nQuestion: How does {concept} function based on the passage?\nAnswer: Based on the passage, {concept} functions by {mechanism}.", "context_qa")
]

CODING_TEMPLATES = [
    ("Code Explanation: In Python, implementing {concept} involves structuring data structures efficiently. For example:\n```python\ndef process_{clean_concept}(data):\n    # {mechanism}\n    result = [x for x in data if x is not None]\n    return result\n```\nThis snippet demonstrates how {concept} achieves {benefit}.", "coding"),
    ("Programming Note: When working with {concept}, developers frequently handle edge cases by checking initialization states before calling execution threads to ensure {benefit}.", "coding")
]

MATH_TEMPLATES = [
    ("Mathematical Formulation: The behavior of {concept} can be modeled using formal notation. Let $f(x)$ represent the evaluation function where $f(x) = \\sum_{{i=1}}^{{n}} w_i x_i + b$. Here, optimization proceeds by {mechanism}, yielding {benefit}.", "math"),
    ("Theorem & Proof Sketch: For any sequence governed by {concept}, the error bound scales as $O(1/n)$. By proving that {mechanism}, we establish that {benefit}.", "math")
]

CONCEPT_POOLS = {
    "Computer Science": [
        ("binary search trees", "a node-based binary tree data structure", "recursively partitioning key comparisons to maintain sorted order", "logarithmic time complexity for lookup operations"),
        ("hash maps", "an associative array that maps keys to values", "computing hash indices to access bucket slots directly", "constant-time average search performance"),
        ("operating system schedulers", "the core process control loop", "allocating CPU time slices dynamically based on thread priority", "optimal hardware utilization and minimal context switch latency"),
        ("compilers", "a program converting high-level code to machine code", "parsing abstract syntax trees and emitting target bytecode", "optimized execution speed across target architectures"),
        ("relational databases", "a structured data storage engine", "enforcing ACID transactions and executing SQL query plans", "reliable data persistence and transactional integrity")
    ],
    "Artificial Intelligence": [
        ("attention mechanisms", "a technique routing focus dynamically across inputs", "calculating pairwise dot-product relevance scores between query and key vectors", "effective long-range context conditioning"),
        ("generative adversarial networks", "a dual-network competitive framework", "training generator and discriminator models in a zero-sum game", "high-fidelity synthetic data distribution modeling"),
        ("reinforcement learning agents", "policy-driven decision systems", "optimizing action selections to maximize long-term cumulative reward signals", "autonomous navigation in complex environments"),
        ("expert systems", "knowledge-based reasoning frameworks", "evaluating rule-based inference engines over factual assertion databases", "consistent domain expert decision support"),
        ("computer vision pipelines", "image feature extraction architectures", "processing pixel grids through convolutional filter layers", "accurate object detection and semantic segmentation")
    ],
    "Machine Learning": [
        ("gradient descent", "an iterative optimization algorithm", "computing loss gradients with respect to model parameters and stepping downhill", "efficient convergence toward minimal loss values"),
        ("cross-validation", "a robust model evaluation protocol", "partitioning dataset samples into k distinct folds to test generalization", "unbiased performance metrics without data leakage"),
        ("regularization techniques", "penalty constraints added to loss functions", "constraining parameter magnitudes via L1 or L2 norm penalization", "prevention of model overfitting on noisy training data"),
        ("decision tree ensembles", "collections of non-linear decision trees", "aggregating predictions across random forest or boosted tree splits", "high predictive accuracy on tabular datasets"),
        ("unsupervised clustering", "grouping unlabeled data points", "minimizing intra-cluster spatial distances using Euclidean metrics", "discovery of latent structural patterns in data")
    ],
    "Physics": [
        ("general relativity", "Einstein's geometric theory of gravitation", "describing gravity as mass-energy warping spacetime geometry", "accurate predictions of orbital precession and gravitational lensing"),
        ("quantum superposition", "a state where physical systems exist in multiple configurations", "expressing state vectors as linear combinations of orthogonal basis states", "exponential computational parallelism in quantum algorithms"),
        ("thermodynamic entropy", "a measure of microscopic state disorder", "tracking irreversible heat dissipation across closed system boundaries", "predictable thermodynamic state transitions"),
        ("electromagnetic radiation", "energy propagation through oscillating fields", "transmitting photons across orthogonal electric and magnetic wave vectors", "broadband energy transfer across spectrums"),
        ("classical mechanics", "the study of forces acting on physical bodies", "applying Newton's laws of motion to compute momentum and trajectories", "precise trajectory forecasting for mechanical systems")
    ],
    "Mathematics": [
        ("Fourier transforms", "a mathematical transformation converting time signals to frequencies", "decomposing complex continuous functions into constituent sine and cosine frequencies", "efficient frequency-domain signal analysis"),
        ("eigenvalue decomposition", "a factorization of a square matrix into eigenvalues and eigenvectors", "scaling characteristic vector directions by invariant scalar factors", "dimensionality reduction in principal component analysis"),
        ("differential equations", "equations relating functions to their rates of change", "integrating continuous boundary conditions across state variables", "precise modeling of dynamic physical phenomena"),
        ("probability distributions", "functions mapping outcomes to likelihood values", "calculating cumulative density functions across random variable outcomes", "rigorous statistical modeling of uncertainty"),
        ("linear algebra projections", "mapping vectors onto lower-dimensional subspaces", "computing orthogonal dot products with basis transformation matrices", "optimal rank reduction in vector spaces")
    ],
    "Space": [
        ("Keplerian orbital motion", "the mathematical description of planetary bodies moving around gravitational centers", "balancing gravitational pull against tangential orbital velocity", "stable elliptical trajectories for planetary satellites"),
        ("black hole event horizons", "the boundary surrounding gravitational singularities", "trapping light and matter when escape velocity exceeds light speed", "extreme gravitational distortion of background radiation"),
        ("stellar nucleosynthesis", "the cosmic creation of chemical elements", "fusing hydrogen atomic nuclei into helium under core pressure", "sustained energy radiation of main-sequence stars"),
        ("cosmic microwave background", "thermal radiation legacy of the early universe", "measuring faint isotropic microwave fluctuations across deep space", "empirical confirmation of cosmic expansion models"),
        ("exoplanet atmospheric spectroscopy", "analyzing starlight passing through planetary atmospheres", "detecting chemical absorption lines during transit events", "identification of potential biosignatures on distant worlds")
    ],
    "Software Engineering": [
        ("continuous integration", "an automated build and test process", "executing automated test suites on every version control commit", "early detection of integration defects"),
        ("microservices architecture", "a modular application design pattern", "decoupling application services into independently deployable units", "high fault tolerance and independent horizontal scaling"),
        ("refactoring patterns", "code restructuring techniques", "simplifying complex class hierarchies without altering external behavior", "improved code maintainability and reduced technical debt"),
        ("design patterns", "reusable solutions to common software problems", "instantiating behavioral or structural software abstractions", "standardized software component interaction"),
        ("version control branching", "parallel codebase development tracking", "merging isolated feature branches via commit graphs", "seamless multi-developer code collaboration")
    ],
    "Web Development": [
        ("document object model parsing", "the browser representation of structured HTML markup", "constructing hierarchical DOM tree nodes in memory", "dynamic JavaScript interaction with web UI components"),
        ("asynchronous HTTP requests", "non-blocking network IO protocols", "fetching remote JSON data payloads in background event loops", "seamless single-page application updates without full page reloads"),
        ("CSS flexbox layouts", "a modern CSS box layout module", "distributing free space dynamically along single dimensional axes", "responsive user interfaces across varying device viewports"),
        ("browser caching policies", "client-side content storage strategies", "storing static asset responses according to HTTP cache headers", "dramatically reduced page load latency"),
        ("RESTful API design", "architectural constraints for web services", "exposing standard HTTP resource endpoints with stateless semantics", "scalable client-server interoperability")
    ]
}

def build_v9_redesigned():
    random.seed(42)
    np.random.seed(42)

    total_target_count = 14500
    
    unique_documents = []
    seen_texts = set()

    doc_category_counts = {
        "declarative": 0, "explanatory": 0, "qa": 0, "instruction": 0,
        "dialogue": 0, "context_qa": 0, "coding": 0, "math": 0
    }

    iters = 0
    while len(unique_documents) < total_target_count and iters < 25000:
        iters += 1
        domain = random.choice(DOMAINS)
        pool = CONCEPT_POOLS[domain]
        concept_tuple = random.choice(pool)
        concept, definition, mechanism, benefit = concept_tuple
        clean_concept = concept.replace(" ", "_")

        category_roll = random.random()

        mod = random.choice([
            "In practical field deployments, ", "According to modern technical literature, ", "Systematic empirical benchmarks confirm that ",
            "From an engineering perspective, ", "Comprehensive research indicates that ", "In high-throughput enterprise environments, ",
            "Through rigorous mathematical derivation, ", "Standard industry guidelines suggest that ", "Under standard operating parameters, ",
            "When evaluating system trade-offs, ", "Recent studies demonstrate that ", "In production software systems, ", ""
        ])
        post = random.choice([
            " This fundamental property makes it an indispensable asset across multi-disciplinary engineering contexts.",
            " Furthermore, continuous validation ensures robust fault tolerance under volatile load conditions.",
            " Developers and researchers rely heavily on this operational paradigm to maintain systematic efficiency.",
            " Consequently, benchmarking teams consistently report superior metrics when this methodology is adopted.",
            " This aligns directly with established theoretical models documented in primary research papers.",
            " In practice, performance gains scale linearly with workload expansion.",
            ""
        ])

        extra_num = random.randint(1, 1000)

        if category_roll < 0.15:
            tmpl = random.choice(FACTUAL_TEMPLATES)
            txt = mod + tmpl.format(concept=concept, definition=definition, mechanism=mechanism, benefit=benefit, domain=domain) + f" (Ref: #{extra_num})." + post
            cat = "declarative"
        elif category_roll < 0.30:
            tmpl = random.choice(EXPLANATORY_TEMPLATES)
            txt = mod + tmpl.format(concept=concept, definition=definition, mechanism=mechanism, benefit=benefit, domain=domain) + f" (Case Study #{extra_num})." + post
            cat = "explanatory"
        elif category_roll < 0.45:
            q_tmpl, a_tmpl = random.choice(QA_TEMPLATES)
            q_txt = q_tmpl.format(concept=concept, domain=domain)
            a_txt = a_tmpl.format(concept=concept, definition=definition, mechanism=mechanism, benefit=benefit)
            txt = f"{q_txt}\n{mod}{a_txt}{post}"
            cat = "qa"
        elif category_roll < 0.60:
            tmpl, _ = random.choice(INSTRUCTION_TEMPLATES)
            txt = mod + tmpl.format(concept=concept, definition=definition, mechanism=mechanism, benefit=benefit, domain=domain) + post
            cat = "instruction"
        elif category_roll < 0.70:
            tmpl, _ = random.choice(CONVERSATIONAL_TEMPLATES)
            txt = tmpl.format(concept=concept, definition=definition, mechanism=mechanism, benefit=benefit, domain=domain)
            cat = "dialogue"
        elif category_roll < 0.80:
            tmpl, _ = random.choice(CONTEXT_QA_TEMPLATES)
            ctx = f"{mod}in the domain of {domain}, research into {concept} has advanced rapidly. Systems utilizing {concept} focus on {mechanism}."
            txt = tmpl.format(concept=concept, context_text=ctx, mechanism=mechanism, benefit=benefit, domain=domain)
            cat = "context_qa"
        elif category_roll < 0.90:
            tmpl, _ = random.choice(CODING_TEMPLATES)
            txt = tmpl.format(concept=concept, clean_concept=clean_concept, mechanism=mechanism, benefit=benefit, domain=domain) + post
            cat = "coding"
        else:
            tmpl, _ = random.choice(MATH_TEMPLATES)
            txt = mod + tmpl.format(concept=concept, mechanism=mechanism, benefit=benefit, domain=domain) + post
            cat = "math"

        txt_clean = txt.strip()
        if txt_clean not in seen_texts:
            seen_texts.add(txt_clean)
            unique_documents.append((domain, cat, txt_clean))
            doc_category_counts[cat] += 1

    print(f"Generated {len(unique_documents)} unique documents for V9.")
    print("Category breakdown:", doc_category_counts)

    random.seed(42)
    random.shuffle(unique_documents)

    train_docs = unique_documents[:int(len(unique_documents)*0.85)]
    val_docs = unique_documents[int(len(unique_documents)*0.85):int(len(unique_documents)*0.95)]
    test_docs = unique_documents[int(len(unique_documents)*0.95):]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, docs in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
        with open(os.path.join(OUTPUT_DIR, f"{name}_cleaned.txt"), "w", encoding="utf-8") as f:
            for d, c, t in docs:
                f.write(t + "\n\n")

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    train_tokens = []
    val_tokens = []
    test_tokens = []

    for name, docs in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
        bin_path = os.path.join(OUTPUT_DIR, f"{name}.bin")
        all_ids = []
        for d, c, t in docs:
            ids = tokenizer.encode(t, bos=True, eos=True)
            all_ids.extend(ids)
        arr = np.array(all_ids, dtype=np.uint16)
        arr.tofile(bin_path)

        if name == "train":
            train_tokens = all_ids
        elif name == "val":
            val_tokens = all_ids
        else:
            test_tokens = all_ids

    total_tokens = len(train_tokens) + len(val_tokens) + len(test_tokens)
    print(f"Total V9 Tokens: {total_tokens:,} | Train Tokens: {len(train_tokens):,}")

    metadata = {
        "dataset_name": "collision_dataset_v9_redesigned",
        "dataset_version": "9.0-redesigned",
        "creation_timestamp": "2026-09-10T12:10:00+05:30",
        "random_seed": 42,
        "document_count": len(unique_documents),
        "example_count": len(unique_documents),
        "token_count": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "test_tokens": len(test_tokens),
        "train_count": len(train_docs),
        "validation_count": len(val_docs),
        "test_count": len(test_docs),
        "category_distribution": doc_category_counts,
        "tokenizer_information": {
            "tokenizer_dir": TOKENIZER_DIR,
            "vocab_size": len(tokenizer.inv_special_tokens) + len(tokenizer.vocab)
        }
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("collision_dataset_v9_redesigned created successfully!")

if __name__ == "__main__":
    build_v9_redesigned()
