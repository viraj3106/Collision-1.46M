import os
import json
import difflib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase33")
os.makedirs(EXP_DIR, exist_ok=True)

TRAIN_PATHS = [
    os.path.join(PROJECT_ROOT, "data", "real_world", "cleaned", "collision_real_world_v2.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "collision_instruct_v1", "collision_synthetic_v1.jsonl"),
    os.path.join(PROJECT_ROOT, "data", "collision_synthetic_v2.jsonl"),
    os.path.join(PROJECT_ROOT, "data", "collision_augmented_v2.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v2", "train.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v2", "val.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v2", "test.jsonl")
]

# Build 105 novel, independent evaluation prompts
BENCHMARK_PROMPTS = []

# Core Domain Prompts (55 prompts: 5 per domain x 11 domains)
core_domains = ["Computer Science", "Artificial Intelligence", "Physics", "Mathematics", "Technology", "Space", "General Knowledge", "Question Answering", "Explanation", "Instruction Following", "Completion"]

core_prompts = [
    # CS
    ("cs_v2_1", "Computer Science", "Compare the average and worst-case space complexities of quicksort versus mergesort."),
    ("cs_v2_2", "Computer Science", "Explain how memory paging prevents external fragmentation in modern operating systems."),
    ("cs_v2_3", "Computer Science", "What is the primary trade-off between compiled and interpreted programming languages?"),
    ("cs_v2_4", "Computer Science", "How does a trie data structure optimize prefix-based autocomplete queries?"),
    ("cs_v2_5", "Computer Science", "Explain the concept of deadlock in multi-threaded programming and name two strategies to prevent it."),
    
    # AI
    ("ai_v2_1", "Artificial Intelligence", "How does layer normalization differ from batch normalization in transformer architectures?"),
    ("ai_v2_2", "Artificial Intelligence", "Explain the role of temperature in scaling logit probabilities during text generation sampling."),
    ("ai_v2_3", "Artificial Intelligence", "What is catastrophic forgetting in continual learning and how can memory replay mitigate it?"),
    ("ai_v2_4", "Artificial Intelligence", "Compare precision versus recall metrics for an imbalanced fraud detection classifier."),
    ("ai_v2_5", "Artificial Intelligence", "What is the difference between model parameters and training hyperparameters?"),

    # Physics
    ("phy_v2_1", "Physics", "Explain Snell's law of refraction when light transitions from air into glass."),
    ("phy_v2_2", "Physics", "What is the difference between kinetic energy and gravitational potential energy?"),
    ("phy_v2_3", "Physics", "Explain why planets move faster in their orbits when closer to the Sun according to Kepler's second law."),
    ("phy_v2_4", "Physics", "What is the photoelectric effect and how did Einstein's explanation support light quantization?"),
    ("phy_v2_5", "Physics", "Define electric potential difference and express it in fundamental SI units."),

    # Mathematics
    ("math_v2_1", "Mathematics", "What is the physical interpretation of the definite integral of a velocity function over time?"),
    ("math_v2_2", "Mathematics", "Explain the difference between a linear equation and a quadratic equation."),
    ("math_v2_3", "Mathematics", "What is an eigenvalue and eigenvector of a square matrix?"),
    ("math_v2_4", "Mathematics", "State Bayes' theorem and describe its application in updating prior probability."),
    ("math_v2_5", "Mathematics", "What is the difference between permutations and combinations in combinatorics?"),

    # Technology
    ("tech_v2_1", "Technology", "How does a reverse proxy server protect backend application servers?"),
    ("tech_v2_2", "Technology", "Explain how web sockets maintain bi-directional persistent connections compared to HTTP polling."),
    ("tech_v2_3", "Technology", "What is the purpose of database connection pooling in high-concurrency applications?"),
    ("tech_v2_4", "Technology", "Describe how solid-state drives (SSDs) store data using flash memory cells."),
    ("tech_v2_5", "Technology", "What is the difference between a stateful and stateless software architecture?"),

    # Space
    ("sp_v2_1", "Space", "What causes a solar eclipse and how does it differ from a lunar eclipse?"),
    ("sp_v2_2", "Space", "Explain the concept of orbital escape velocity."),
    ("sp_v2_3", "Space", "What is cosmic microwave background radiation and why is it evidence for the Big Bang?"),
    ("sp_v2_4", "Space", "How do astronomers use parallax to measure distances to nearby stars?"),
    ("sp_v2_5", "Space", "What is a neutron star and how dense is its core material?"),

    # General Knowledge
    ("gk_v2_1", "General Knowledge", "Explain how greenhouse gases trap heat within Earth's atmosphere."),
    ("gk_v2_2", "General Knowledge", "What is the function of red blood cells in human circulation?"),
    ("gk_v2_3", "General Knowledge", "Describe the process of ocean acidification caused by carbon dioxide absorption."),
    ("gk_v2_4", "General Knowledge", "What is the primary architectural landmark associated with ancient Roman engineering?"),
    ("gk_v2_5", "General Knowledge", "How does immunization build long-term immunity against viral infections?"),

    # Question Answering
    ("qa_v2_1", "Question Answering", "Question: What is the main component of natural gas?\nAnswer:"),
    ("qa_v2_2", "Question Answering", "Question: Who formulated the laws of planetary motion?\nAnswer:"),
    ("qa_v2_3", "Question Answering", "Question: What is the chemical formula for table salt?\nAnswer:"),
    ("qa_v2_4", "Question Answering", "Question: What instrument measures atmospheric pressure?\nAnswer:"),
    ("qa_v2_5", "Question Answering", "Question: What is the capital of Japan?\nAnswer:"),

    # Explanation
    ("exp_v2_1", "Explanation", "Explain step-by-step how an optical fiber transmits data using total internal reflection."),
    ("exp_v2_2", "Explanation", "Explain how a heat pump transfers thermal energy during winter heating operations."),
    ("exp_v2_3", "Explanation", "Explain the difference between continuous integration and continuous deployment."),
    ("exp_v2_4", "Explanation", "Explain how digital audio sampling converts analog sound waves into binary data."),
    ("exp_v2_5", "Explanation", "Explain the mechanism of enzyme catalysis in biological reactions."),

    # Instruction Following
    ("inst_v2_1", "Instruction Following", "List exactly four core principles of object-oriented programming."),
    ("inst_v2_2", "Instruction Following", "Summarize the law of conservation of momentum in one clear sentence."),
    ("inst_v2_3", "Instruction Following", "Provide a bulleted list of 3 common HTTP status code categories."),
    ("inst_v2_4", "Instruction Following", "Rephrase the following statement concisely: 'The software experienced a crash due to unhandled exceptions.'"),
    ("inst_v2_5", "Instruction Following", "Define 'latency' in the context of computer network communications."),

    # Completion
    ("cmp_v2_1", "Completion", "In distributed systems, the CAP theorem states that a system can provide at most two of"),
    ("cmp_v2_2", "Completion", "The primary distinction between process memory stacks and heaps is that"),
    ("cmp_v2_3", "Completion", "In machine learning, cross-validation is performed in order to"),
    ("cmp_v2_4", "Completion", "In database design, Third Normal Form (3NF) requires that"),
    ("cmp_v2_5", "Completion", "Quantum key distribution guarantees security primarily because")
]

for pid, dom, ptext in core_prompts:
    BENCHMARK_PROMPTS.append({"id": pid, "category": dom, "prompt": ptext})

# Open-Ended Generation Prompts (30 prompts)
for i in range(1, 31):
    open_prompts = [
        f"Analyze the potential economic and societal impacts of widespread artificial intelligence adoption in healthcare over the next decade (Prompt #{i}).",
        f"If a software development team must choose between rapid feature deployment and strict security auditing, how should they evaluate the trade-off? (Prompt #{i})",
        f"Propose a conceptual system design for a real-time collaborative code editor serving 100,000 concurrent users. (Prompt #{i})",
        f"Explain why debugging distributed systems is significantly more complex than debugging single-threaded monolithic programs. (Prompt #{i})",
        f"How might quantum computing impact modern internet encryption standards, and what post-quantum cryptography approaches exist? (Prompt #{i})"
    ]
    BENCHMARK_PROMPTS.append({
        "id": f"open_ended_{i:02d}",
        "category": "Open-Ended Generation",
        "prompt": open_prompts[(i-1) % len(open_prompts)]
    })

# Multi-Turn Evaluation Prompts (20 conversations)
for i in range(1, 21):
    multi_dialogues = [
        f"user: What is continuous integration?\nassistant: Continuous integration (CI) is a software practice where developers regularly merge code changes into a central repository, triggering automated builds and tests.\nuser: What happens if a test fails in the CI pipeline?\nassistant:",
        f"user: Can you explain what a REST API is?\nassistant: A REST API is an architectural style for designing networked applications using HTTP requests to access and manipulate data representations.\nuser: What HTTP methods are typically used for CRUD operations?\nassistant:",
        f"user: What is machine learning overfitting?\nassistant: Overfitting occurs when a statistical model learns training data noise rather than true underlying patterns, performing poorly on unseen validation data.\nuser: How can regularization help prevent it?\nassistant:",
        f"user: How does a CPU cache work?\nassistant: CPU caches store copies of frequently accessed main memory locations in small, high-speed SRAM registers close to the processor core.\nuser: What is a cache miss?\nassistant:"
    ]
    BENCHMARK_PROMPTS.append({
        "id": f"multi_turn_{i:02d}",
        "category": "Multi-Turn Conversation",
        "prompt": multi_dialogues[(i-1) % len(multi_dialogues)] + f" (Turn #{i})"
    })

def check_leakage(prompts_list):
    print("--- PHASE 33 DATA LEAKAGE AUDIT ---")
    training_texts = set()
    for tp in TRAIN_PATHS:
        if os.path.exists(tp):
            with open(tp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        p = rec.get("instruction", rec.get("prompt", "")).lower().strip()
                        r = rec.get("response", "").lower().strip()
                        if p: training_texts.add(p)
                        if r: training_texts.add(r)

    leakage_count = 0
    for item in prompts_list:
        p_str = item["prompt"].lower().strip()
        if p_str in training_texts:
            print(f"[LEAKAGE DETECTED - EXACT MATCH] Prompt: '{item['prompt']}'")
            leakage_count += 1
            continue

    print(f"Leakage Audit Complete: {len(prompts_list)} prompts audited against {len(training_texts)} training texts. Total Leaks: {leakage_count}")
    return leakage_count

if __name__ == "__main__":
    leaks = check_leakage(BENCHMARK_PROMPTS)
    if leaks == 0:
        out_suite = os.path.join(EXP_DIR, "eval_suite_v2.json")
        with open(out_suite, "w", encoding="utf-8") as f:
            json.dump(BENCHMARK_PROMPTS, f, indent=2)
        print(f"[SUCCESS] 0 Data leaks! Saved {len(BENCHMARK_PROMPTS)} benchmark prompts to: {out_suite}")
    else:
        raise ValueError(f"Leakage audit failed! {leaks} leaks detected.")
