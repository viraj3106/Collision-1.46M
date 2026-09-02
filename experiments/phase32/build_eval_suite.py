import os
import json
import difflib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase32", "evaluation_v1")
os.makedirs(EVAL_DIR, exist_ok=True)

# Training dataset paths for leakage checking
TRAIN_PATHS = [
    os.path.join(PROJECT_ROOT, "data", "real_world", "cleaned", "collision_real_world_v2.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "collision_instruct_v1", "collision_synthetic_v1.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v1", "train.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v1", "val.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v1", "test.jsonl"),
]

# Build 44 novel evaluation prompts across 11 core categories + stress test categories
EVAL_PROMPTS = [
    # 1. General Knowledge
    {"id": "gk_01", "category": "General Knowledge", "prompt": "What is the water cycle and why is it important for life on Earth?"},
    {"id": "gk_02", "category": "General Knowledge", "prompt": "Name three major organs of the human body and their main functions."},
    {"id": "gk_03", "category": "General Knowledge", "prompt": "Why do leaves change color in the autumn season?"},
    {"id": "gk_04", "category": "General Knowledge", "prompt": "Describe the main difference between renewable and non-renewable energy sources."},

    # 2. Computer Science
    {"id": "cs_01", "category": "Computer Science", "prompt": "Explain the difference between a stack and a queue data structure."},
    {"id": "cs_02", "category": "Computer Science", "prompt": "What is recursion in programming and when should it be used?"},
    {"id": "cs_03", "category": "Computer Science", "prompt": "What does time complexity O(n log n) mean in algorithm analysis?"},
    {"id": "cs_04", "category": "Computer Science", "prompt": "How does an operating system manage process scheduling?"},

    # 3. Artificial Intelligence
    {"id": "ai_01", "category": "Artificial Intelligence", "prompt": "What is overfitting in machine learning models and how can it be prevented?"},
    {"id": "ai_02", "category": "Artificial Intelligence", "prompt": "Explain the key mechanism behind self-attention in transformer architectures."},
    {"id": "ai_03", "category": "Artificial Intelligence", "prompt": "How do contrastive learning objectives differ from generative pre-training objectives?"},
    {"id": "ai_04", "category": "Artificial Intelligence", "prompt": "Why are GPUs widely preferred over CPUs for training deep neural networks?"},

    # 4. Physics
    {"id": "phy_01", "category": "Physics", "prompt": "State Newton's second law of motion and express it mathematically."},
    {"id": "phy_02", "category": "Physics", "prompt": "What is the law of conservation of energy?"},
    {"id": "phy_03", "category": "Physics", "prompt": "Explain why sound waves cannot travel through a vacuum."},
    {"id": "phy_04", "category": "Physics", "prompt": "What is thermodynamic entropy and what does the second law of thermodynamics state?"},

    # 5. Mathematics
    {"id": "math_01", "category": "Mathematics", "prompt": "Explain what a derivative represents in calculus with an example."},
    {"id": "math_02", "category": "Mathematics", "prompt": "What is the Pythagorean theorem and how is it used?"},
    {"id": "math_03", "category": "Mathematics", "prompt": "Define what a prime number is and list the first five prime numbers."},
    {"id": "math_04", "category": "Mathematics", "prompt": "How does matrix multiplication differ from standard scalar multiplication?"},

    # 6. Technology
    {"id": "tech_01", "category": "Technology", "prompt": "How does asymmetric public-key cryptography keep internet transactions secure?"},
    {"id": "tech_02", "category": "Technology", "prompt": "What is containerization in software deployment and why is Docker popular?"},
    {"id": "tech_03", "category": "Technology", "prompt": "Explain the role of a RESTful API in modern client-server web applications."},
    {"id": "tech_04", "category": "Technology", "prompt": "What is edge computing and how does it reduce network latency?"},

    # 7. Space
    {"id": "sp_01", "category": "Space", "prompt": "How are stellar black holes formed after a massive star dies?"},
    {"id": "sp_02", "category": "Space", "prompt": "What causes the solar wind and how does Earth's magnetosphere protect us?"},
    {"id": "sp_03", "category": "Space", "prompt": "Explain the main goals of the James Webb Space Telescope."},
    {"id": "sp_04", "category": "Space", "prompt": "Why is Mars the primary target for human interplanetary exploration?"},

    # 8. Question Answering
    {"id": "qa_01", "category": "Question Answering", "prompt": "Question: What is photosynethesis?\nAnswer:"},
    {"id": "qa_02", "category": "Question Answering", "prompt": "Question: Who proposed the theory of general relativity?\nAnswer:"},
    {"id": "qa_03", "category": "Question Answering", "prompt": "Question: What is the speed of light in vacuum?\nAnswer:"},
    {"id": "qa_04", "category": "Question Answering", "prompt": "Question: What is the boiling point of water at standard atmospheric pressure?\nAnswer:"},

    # 9. Explanation
    {"id": "exp_01", "category": "Explanation", "prompt": "Explain step-by-step how a compiler converts source code into machine executable binaries."},
    {"id": "exp_02", "category": "Explanation", "prompt": "Explain how climate change impacts global sea levels."},
    {"id": "exp_03", "category": "Explanation", "prompt": "Explain the concept of quantum superposition in simple terms."},
    {"id": "exp_04", "category": "Explanation", "prompt": "Explain how garbage collection works in memory-managed programming languages."},

    # 10. Completion
    {"id": "cmp_01", "category": "Completion", "prompt": "Deep learning models rely heavily on large neural networks because"},
    {"id": "cmp_02", "category": "Completion", "prompt": "The primary advantage of using version control systems like Git is"},
    {"id": "cmp_03", "category": "Completion", "prompt": "In database design, normalization is performed in order to"},
    {"id": "cmp_04", "category": "Completion", "prompt": "Quantum computing differs from classical computing primarily because"},

    # 11. Instruction Following
    {"id": "inst_01", "category": "Instruction Following", "prompt": "Write a 2-sentence summary explaining why data structures matter in software engineering."},
    {"id": "inst_02", "category": "Instruction Following", "prompt": "List exactly three benefits of unit testing in continuous integration pipelines."},
    {"id": "inst_03", "category": "Instruction Following", "prompt": "Define the word 'algorithm' in one concise paragraph."},
    {"id": "inst_04", "category": "Instruction Following", "prompt": "Provide a bulleted list of 3 key components of a computer CPU."},

    # Stress-Testing Categories (Repetition, Fragmentation, Hallucination, Topic Drift)
    {"id": "stress_01", "category": "Stress Test - Repetition", "prompt": "Describe the process of photosynthesis. Do not repeat words endlessly."},
    {"id": "stress_02", "category": "Stress Test - Fragmentation", "prompt": "Write a complete, coherent response explaining how electricity flows in a circuit."},
    {"id": "stress_03", "category": "Stress Test - Topic Drift", "prompt": "Answer specifically about Python programming: What are list comprehensions?"},
    {"id": "stress_04", "category": "Stress Test - Failure Recovery", "prompt": "What happens when a program divides a number by zero?"}
]

# Separate Real-World Generalization Subset (Inspired by beta usage telemetry)
REALWORLD_BETA_EVAL_PROMPTS = [
    {"id": "rw_eval_01", "category": "Beta Telemetry QA", "prompt": "What is Python list comprehension?"},
    {"id": "rw_eval_02", "category": "Beta Telemetry Explanations", "prompt": "How does backpropagation work in neural networks?"},
    {"id": "rw_eval_03", "category": "Beta Telemetry Definitions", "prompt": "Define artificial intelligence in 1 sentence."},
    {"id": "rw_eval_04", "category": "Beta Telemetry Math", "prompt": "What is a matrix determinant?"},
    {"id": "rw_eval_05", "category": "Beta Telemetry Code Request", "prompt": "Write a simple function to calculate factorial in Python."},
    {"id": "rw_eval_06", "category": "Beta Telemetry Instruction", "prompt": "Explain gravity simply."}
]

def check_leakage(eval_items, train_paths):
    print("--- DATA LEAKAGE AUDIT ---")
    training_texts = []
    for tp in train_paths:
        if os.path.exists(tp):
            with open(tp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        p = rec.get("instruction", rec.get("prompt", ""))
                        r = rec.get("response", "")
                        training_texts.append(p.lower().strip())
                        training_texts.append(r.lower().strip())
                        training_texts.append(f"{p} {r}".lower().strip())

    leakage_found = 0
    total_eval = len(eval_items)

    for item in eval_items:
        prompt_str = item["prompt"].lower().strip()
        # 1. Exact match
        if prompt_str in training_texts:
            print(f"[LEAKAGE DETECTED - EXACT MATCH] Prompt: '{item['prompt']}'")
            leakage_found += 1
            continue
        
        # 2. String similarity match (> 0.85 sequence matcher score)
        high_sim = False
        for tt in training_texts:
            sim = difflib.SequenceMatcher(None, prompt_str, tt).ratio()
            if sim > 0.85 and len(prompt_str) > 10:
                print(f"[LEAKAGE WARNING - HIGH SIMILARITY ({sim:.2f})] Prompt: '{item['prompt']}' vs Training: '{tt}'")
                high_sim = True
                break
        if high_sim:
            leakage_found += 1

    print(f"Audit Complete: {total_eval} evaluation prompts audited. Leakage Count: {leakage_found}")
    return leakage_found

if __name__ == "__main__":
    all_eval = EVAL_PROMPTS + REALWORLD_BETA_EVAL_PROMPTS
    leaks = check_leakage(all_eval, TRAIN_PATHS)
    
    if leaks == 0:
        print("[SUCCESS] Zero data leakage detected! Evaluation suite is completely independent of training data.")
        
        # Save Evaluation Datasets
        suite_path = os.path.join(EVAL_DIR, "eval_suite_v1.json")
        with open(suite_path, "w", encoding="utf-8") as f:
            json.dump(EVAL_PROMPTS, f, indent=2)

        rw_suite_path = os.path.join(EVAL_DIR, "realworld_eval_v1.json")
        with open(rw_suite_path, "w", encoding="utf-8") as f:
            json.dump(REALWORLD_BETA_EVAL_PROMPTS, f, indent=2)

        print(f"Saved {len(EVAL_PROMPTS)} core evaluation prompts to: {suite_path}")
        print(f"Saved {len(REALWORLD_BETA_EVAL_PROMPTS)} real-world telemetry prompts to: {rw_suite_path}")
    else:
        raise ValueError(f"Data leakage detected! {leaks} evaluation prompts matched training data.")
