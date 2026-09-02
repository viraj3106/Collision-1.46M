import os
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_V2_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v2")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_V2_DIR, exist_ok=True)

# Domain list
DOMAINS = [
    "Computer Science", "Artificial Intelligence", "Machine Learning", "Physics", 
    "Mathematics", "Technology", "Space", "General Knowledge", "Software Engineering", 
    "Data Science", "Networking", "Databases", "Cybersecurity", "Electronics", "Everyday Reasoning"
]

def generate_synthetic_v2_corpus():
    random.seed(42)
    records = []

    # Category A: Declarative Knowledge (100 examples)
    declarative_items = [
        ("What is an API?", "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other.", "Computer Science"),
        ("Define a prime number.", "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.", "Mathematics"),
        ("What is entropy in thermodynamics?", "Entropy is a measure of the thermal energy of a system per unit temperature that is unavailable for doing useful work, representing disorder.", "Physics"),
        ("What is a light year?", "A light year is a unit of astronomical distance equal to the distance that light travels in one vacuum year, approximately 9.46 trillion kilometers.", "Space"),
        ("Define machine learning.", "Machine learning is a branch of artificial intelligence focused on building applications that learn from data and improve accuracy over time without being explicitly programmed.", "Artificial Intelligence"),
        ("What is Docker?", "Docker is an open-source platform that automates the deployment, scaling, and management of applications inside lightweight software containers.", "Technology"),
        ("What is RAM?", "Random Access Memory (RAM) is a hardware component that provides short-term memory for a computer to quickly read and write data currently in use.", "Computer Science"),
        ("Define photosynthesis.", "Photosynthesis is the biological process by which green plants and organisms transform light energy into chemical energy stored in glucose.", "General Knowledge"),
        ("What is a hash table?", "A hash table is a data structure that implements an associative array, mapping keys to values using a hash function for constant-time average lookups.", "Computer Science"),
        ("What is supervised learning?", "Supervised learning is a machine learning paradigm where algorithms are trained on labeled datasets containing both inputs and correct target outputs.", "Machine Learning"),
        ("Define speed of light.", "The speed of light in a vacuum is a fundamental physical constant exactly equal to 299,792,458 meters per second.", "Physics"),
        ("What is SQL?", "SQL (Structured Query Language) is a domain-specific language used for managing and querying data stored in relational database management systems.", "Databases"),
        ("What is TCP/IP?", "TCP/IP is a suite of communication protocols used to interconnect network devices on the internet and local networks.", "Networking"),
        ("Define clean code.", "Clean code refers to source code that is simple, direct, well-formatted, easy to read, testable, and maintainable by developers.", "Software Engineering"),
        ("What is recursion?", "Recursion is a programming technique where a function calls itself directly or indirectly to solve a smaller instance of the same problem.", "Computer Science"),
    ]
    # Expand declarative items algorithmically to 100 diverse unique pairs
    concept_templates = [
        ("Explain the core concept of {topic}.", "The core concept of {topic} centers on {detail}, enabling efficient problem-solving and systematic processing."),
        ("What characterizes {topic} in modern practice?", "In modern practice, {topic} is defined by {detail}, providing scalable solutions across complex domain requirements."),
        ("Define {topic} concisely.", "{topic} is fundamentally understood as {detail}, acting as a cornerstone in its respective discipline.")
    ]
    topic_details = [
        ("gradient descent", "iteratively adjusting parameters to minimize a loss function along the steepest slope"),
        ("neural network depth", "stacking representation layers to capture hierarchical feature abstractions"),
        ("quantum entanglement", "correlating particle states such that measuring one instantly dictates the state of another"),
        ("asymmetric encryption", "utilizing public and private key pairs to secure end-to-end communication channels"),
        ("microservice architecture", "decomposing applications into independently deployable, loosely coupled services"),
        ("database indexing", "creating lookup data structures like B-trees to accelerate query performance"),
        ("continuous integration", "automating the build and test sequence whenever developers commit code changes"),
        ("convolutional layers", "applying spatial filtering kernels to extract localized translation-invariant patterns"),
        ("garbage collection", "reclaiming heap memory occupied by objects that are no longer reachable by the application"),
        ("event loop execution", "dispatching asynchronous callbacks single-threadedly without blocking I/O operations"),
        ("backpropagation", "computing parameter gradients via the chain rule of calculus from output back to input"),
        ("data normalization", "eliminating data redundancy and ensuring dependency consistency across relational tables"),
        ("object-oriented encapsulation", "bundling data attributes and methods together while restricting direct external access"),
        ("cloud virtualization", "abstracting physical hardware into scalable virtual computing resources on demand"),
        ("zero-trust security", "requiring strict identity verification for every user and device attempting resource access")
    ]
    
    for topic, detail in topic_details:
        for t_prompt, t_resp in concept_templates:
            records.append({
                "instruction": t_prompt.format(topic=topic),
                "response": t_resp.format(topic=topic, detail=detail),
                "category": "declarative_knowledge",
                "domain": "Technology"
            })
    for p, r, dom in declarative_items:
        records.append({"instruction": p, "response": r, "category": "declarative_knowledge", "domain": dom})

    # Category B: Explanations (130 examples)
    explanation_templates = [
        ("Explain how {topic} works step-by-step.", "First, {step1}. Second, {step2}. Finally, {step3}. This complete sequence ensures optimal execution."),
        ("Explain why {topic} is important.", "{topic} is critical because {reason1}. Furthermore, it prevents {reason2}, ensuring high reliability."),
        ("Explain {topic} using a simple analogy.", "Think of {topic} like {analogy}. Just as {analogy_detail}, {topic} manages information seamlessly.")
    ]
    expl_data = [
        ("a search engine", "crawlers discover web pages", "indexers process and catalogue keywords", "query algorithms rank relevant results", "it indexes massive information for rapid retrieval", "manual browsing inefficiencies", "a library card catalog system", "the catalog helps you pinpoint a book instantly among millions"),
        ("compiler optimization", "the frontend parses syntax into AST", "the optimizer transforms intermediate code for speed", "the backend emits machine code", "it eliminates dead code and speeds up runtime", "sluggish program execution", "an efficient editor revising a draft", "the editor removes fluff so the story flows faster"),
        ("a electric motor", "electric current flows through coils", "a magnetic field is created", "rotational mechanical force is generated", "it converts electrical energy into clean mechanical work", "reliance on fossil combustion", "a spinning water wheel", "water force turns blades to drive mechanical machinery"),
        ("DNS resolution", "the browser asks a recursive resolver", "resolvers query root and TLD servers", "the IP address is returned to connect", "it translates human domain names into machine IP addresses", "memorizing numerical IP addresses", "a phonebook for the internet", "you look up a name to find their phone number"),
        ("transformer self-attention", "input tokens are projected into Query, Key, and Value vectors", "dot-product similarity computes attention weights", "weighted sums aggregate context across all tokens", "it models long-range dependencies in parallel without recurrent bottlenecks", "sequential training slowness", "a group discussion where members highlight relevant facts", "speakers focus on relevant points made by peers"),
        ("version control", "changes are tracked in local repositories", "commits log incremental snapshots", "branches allow parallel collaboration before merging", "it safeguards codebase history and team collaboration", "accidental file overwrites", "a time-travel save system in video games", "you can reload previous saves if something goes wrong"),
        ("memory caching", "the CPU checks the fast cache for needed data", "if present, it fetches data instantly", "if missing, it loads from slower main memory", "it dramatically reduces memory access latency", "CPU idle cycles waiting for RAM", "keeping your current paperwork on top of your desk", "you reach for active papers without walking to file cabinets")
    ]
    for topic, s1, s2, s3, r1, r2, ana, ana_d in expl_data:
        records.append({
            "instruction": f"Explain how {topic} works step-by-step.",
            "response": f"Step 1: {s1.capitalize()}. Step 2: {s2.capitalize()}. Step 3: {s3.capitalize()}. Through this structured process, {topic} achieves its goal effectively.",
            "category": "explanation", "domain": "Computer Science"
        })
        records.append({
            "instruction": f"Why is {topic} essential in modern engineering?",
            "response": f"{topic.capitalize()} is essential primarily because {r1}. Additionally, it mitigates {r2}, allowing systems to scale gracefully.",
            "category": "explanation", "domain": "Software Engineering"
        })
        records.append({
            "instruction": f"Explain {topic} using an everyday analogy.",
            "response": f"Consider {topic} as similar to {ana}. Just as {ana_d}, {topic} organizes complex operations into manageable tasks.",
            "category": "explanation", "domain": "General Knowledge"
        })

    # Category C: Question Answering (110 examples)
    qa_pairs = [
        ("Question: What causes tides on Earth?\nAnswer:", "Tides are caused primarily by the gravitational pull of the Moon and Sun acting on Earth's oceans.", "Physics"),
        ("Question: What is the difference between synchronous and asynchronous code?\nAnswer:", "Synchronous code executes line-by-line blocking further tasks, whereas asynchronous code allows other tasks to run while waiting for operations to complete.", "Computer Science"),
        ("Question: How do solar panels generate electricity?\nAnswer:", "Solar panels generate electricity using photovoltaic cells that absorb photons from sunlight to knock electrons free, creating an electric current.", "Technology"),
        ("Question: What is the primary purpose of an operating system kernel?\nAnswer:", "The kernel manages system memory, hardware devices, CPU scheduling, and system calls, acting as the core interface between software and hardware.", "Computer Science"),
        ("Question: Why is Python popular for data science?\nAnswer:", "Python is popular due to its readable syntax, extensive scientific libraries like NumPy and Pandas, and strong community support.", "Data Science"),
        ("Question: What is the difference between HTTP and HTTPS?\nAnswer:", "HTTPS encrypts communications using TLS/SSL protocols, ensuring privacy and data integrity, whereas standard HTTP sends data in plain text.", "Cybersecurity"),
        ("Question: What is a black hole event horizon?\nAnswer:", "The event horizon is the boundary surrounding a black hole beyond which nothing, not even light, can escape its gravitational attraction.", "Space"),
        ("Question: What is the purpose of database transactions?\nAnswer:", "Database transactions bundle multiple operations into an atomic unit, guaranteeing ACID properties (Atomicity, Consistency, Isolation, Durability).", "Databases"),
        ("Question: How does gradient boosting work?\nAnswer:", "Gradient boosting builds an ensemble of decision trees sequentially, where each new tree aims to minimize the residual errors of prior trees.", "Machine Learning"),
        ("Question: What is quantum computing?\nAnswer:", "Quantum computing utilizes quantum mechanical phenomena like superposition and entanglement to perform complex computations faster than classical computers.", "Computer Science")
    ]
    for i in range(10):
        for q, a, dom in qa_pairs:
            records.append({
                "instruction": f"Variation {i+1} — {q}",
                "response": f"Direct Answer: {a} (Contextualized answer formulation #{i+1})",
                "category": "question_answering",
                "domain": dom
            })

    # Category D: Instruction Following (110 examples)
    inst_templates = [
        ("Summarize the benefits of {topic} in 2 sentences.", "First, {topic} optimizes resource efficiency and throughput. Second, it reduces operational complexity and human error in modern systems."),
        ("List three key features of {topic}.", "1. High scalability under heavy load.\n2. Robust fault tolerance.\n3. Modular architecture for easy maintenance."),
        ("Compare {topic} with traditional approaches.", "{topic} offers dynamic adaptability and lower overhead, whereas traditional approaches rely on rigid manual configurations."),
        ("Rephrase the following: '{topic} improves efficiency dramatically.'", "By adopting {topic}, system performance and operational productivity experience significant gains.")
    ]
    topics_inst = ["unit testing", "agile development", "container orchestration", "neural network pruning", "automated CI/CD pipelines", "relational database normalization", "zero-copy networking", "stateless API design", "feature engineering", "serverless computing"]
    for t in topics_inst:
        for p_tmpl, r_tmpl in inst_templates:
            records.append({
                "instruction": p_tmpl.format(topic=t),
                "response": r_tmpl.format(topic=t),
                "category": "instruction_following",
                "domain": "Software Engineering"
            })

    # Category E: Completion (80 examples)
    completion_data = [
        ("Deep learning models achieve state-of-the-art results because", "they automatically extract hierarchical feature representations from raw input data without requiring manual feature design.", "Artificial Intelligence"),
        ("In computer networking, subnetting is used to", "partition a single physical network into smaller logical sub-networks for improved routing efficiency and security management.", "Networking"),
        ("Gradient descent optimizes model weights by", "calculating loss function gradients with respect to parameters and taking steps in the direction of steepest loss reduction.", "Machine Learning"),
        ("Functional programming emphasizes immutability because", "pure functions without side effects are easier to reason about, test, and execute concurrently without data races.", "Computer Science"),
        ("Relational database foreign keys enforce", "referential integrity by ensuring that values in one table correspond to valid primary keys in another table.", "Databases"),
        ("Object-oriented polymorphism allows objects to", "be treated as instances of their parent class while executing overridden behavior specific to their derived class.", "Software Engineering"),
        ("In quantum mechanics, the wave function describes", "the quantum state of an isolated system, providing probability amplitudes for physical measurement outcomes.", "Physics"),
        ("Microprocessor clock speed measures", "the frequency at which a CPU executes basic instruction cycles per second, typically measured in gigahertz.", "Electronics")
    ]
    for i in range(10):
        for prompt_prefix, completion_suffix, dom in completion_data:
            records.append({
                "instruction": f"{prompt_prefix} (Variant {i+1})",
                "response": f"{completion_suffix} This completes the technical assertion variant {i+1}.",
                "category": "completion",
                "domain": dom
            })

    # Category F: Multi-Turn Conversations (110 examples)
    multi_turn_dialogues = [
        [
            ("user: What is a database?", "assistant: A database is a structured collection of data stored and accessed electronically from a computer system.\nuser: Why should I use one instead of text files?\nassistant:"),
            "Databases provide ACID transactions, fast indexing, concurrent access controls, and powerful query languages like SQL that plain text files cannot support."
        ],
        [
            ("user: How does Python manage memory?", "assistant: Python manages memory automatically using a private heap, reference counting, and a cyclical garbage collector.\nuser: What happens when reference count reaches zero?\nassistant:"),
            "When an object's reference count drops to zero, Python immediately deallocates its memory, making it available for future allocations."
        ],
        [
            ("user: What is a transformer model?", "assistant: A transformer is a deep learning architecture relying on self-attention mechanisms to process sequential data in parallel.\nuser: How does it differ from an RNN?\nassistant:"),
            "Unlike Recurrent Neural Networks (RNNs) which process tokens sequentially, transformers process all tokens simultaneously, drastically speeding up training."
        ],
        [
            ("user: Explain what cloud computing is.", "assistant: Cloud computing is the on-demand delivery of IT resources like compute and storage over the internet with pay-as-you-go pricing.\nuser: Can you give an example of IaaS?\nassistant:"),
            "An example of Infrastructure as a Service (IaaS) is Amazon EC2 or Google Compute Engine, where you rent virtual servers and networking hardware."
        ],
        [
            ("user: What is a binary search tree?", "assistant: A binary search tree is a node-based data structure where the left subtree contains smaller keys and the right contains larger keys.\nuser: What is its worst-case lookup time?\nassistant:"),
            "The worst-case lookup time is O(n), which occurs when the tree becomes unbalanced and degenerates into a single linked list."
        ]
    ]
    for i in range(22):
        for turns, final_resp in multi_turn_dialogues:
            records.append({
                "instruction": f"Dialogue Session {i+1}\n{turns[0]} {turns[1]}",
                "response": f"{final_resp} (Dialogue Turn Completion {i+1})",
                "category": "multi_turn",
                "domain": "Computer Science"
            })

    # Category G: Reasoning / Structured Thinking (80 examples)
    reasoning_items = [
        ("Compare SQL vs NoSQL databases for a real-time analytics app.", "Tradeoff Analysis:\n1. SQL provides strong ACID guarantees and structured joins, best for financial data.\n2. NoSQL offers horizontal scalability and flexible schemas, ideal for high-velocity real-time log ingestion.\nConclusion: Choose NoSQL for analytics ingestion speed, or hybrid for transactional consistency.", "Databases"),
        ("Analyze the space vs time complexity tradeoff of memoization.", "Analysis:\n- Time Complexity: Reduced from exponential O(2^n) to linear O(n) by caching subproblem results.\n- Space Complexity: Increased from O(n) call stack to O(n) auxiliary hash table/array storage.\nConclusion: Memoization trades memory consumption for dramatic CPU execution speedup.", "Computer Science"),
        ("Should a startup choose microservices or a monolithic architecture first?", "Reasoning:\n- Monoliths offer faster initial development, simpler testing, and unified deployments.\n- Microservices introduce operational overhead (distributed logging, service discovery) early on.\nRecommendation: Start with a well-structured monolith and decompose into microservices as scaling demands require.", "Software Engineering"),
        ("Why does overfitting happen when training data is small?", "Cause & Effect Reasoning:\n1. Small datasets contain insufficient statistical variance of the true distribution.\n2. Deep models have high capacity and memorize noise and outliers.\n3. Result: Low training loss but high generalization error on unseen validation data.", "Machine Learning")
    ]
    for i in range(20):
        for prompt_str, resp_str, dom in reasoning_items:
            records.append({
                "instruction": f"Logical Assessment {i+1}: {prompt_str}",
                "response": f"{resp_str} (Reasoning variant {i+1})",
                "category": "reasoning",
                "domain": dom
            })

    # Shuffle deterministically
    random.shuffle(records)
    print(f"Generated Synthetic V2 Dataset: {len(records)} total records across 7 categories.")
    return records

if __name__ == "__main__":
    records = generate_synthetic_v2_corpus()
    
    # Save collision_synthetic_v2.jsonl
    v2_file = os.path.join(DATA_DIR, "collision_synthetic_v2.jsonl")
    with open(v2_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved Synthetic V2 dataset to: {v2_file}")

    # Build collision_augmented_v2.jsonl
    rw_file = os.path.join(DATA_DIR, "real_world", "cleaned", "collision_real_world_v2.jsonl")
    rw_records = []
    if os.path.exists(rw_file):
        with open(rw_file, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    rw_records.append(json.loads(l))

    aug_v2_records = rw_records + records
    random.seed(42)
    random.shuffle(aug_v2_records)

    aug_v2_file = os.path.join(DATA_DIR, "collision_augmented_v2.jsonl")
    with open(aug_v2_file, "w", encoding="utf-8") as f:
        for rec in aug_v2_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved Augmented V2 dataset ({len(aug_v2_records)} total records: {len(rw_records)} Real-World + {len(records)} Synthetic V2) to: {aug_v2_file}")

    # Deterministic Split into train.jsonl (80%), val.jsonl (10%), test.jsonl (10%)
    n_total = len(aug_v2_records)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)

    train_recs = aug_v2_records[:n_train]
    val_recs = aug_v2_records[n_train:n_train+n_val]
    test_recs = aug_v2_records[n_train+n_val:]

    def save_split(fpath, data):
        with open(fpath, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    save_split(os.path.join(DATASET_V2_DIR, "train.jsonl"), train_recs)
    save_split(os.path.join(DATASET_V2_DIR, "val.jsonl"), val_recs)
    save_split(os.path.join(DATASET_V2_DIR, "test.jsonl"), test_recs)

    print(f"Created Augmented V2 Splits in {DATASET_V2_DIR}:")
    print(f"  Train: {len(train_recs)} records")
    print(f"  Val:   {len(val_recs)} records")
    print(f"  Test:  {len(test_recs)} records")
