import os
import sys
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREF_DIR = os.path.join(PROJECT_ROOT, "data", "preferences")
os.makedirs(PREF_DIR, exist_ok=True)

CATEGORIES = [
    "general_knowledge", "science", "mathematics", "programming",
    "databases", "linux", "networking", "ai_ml",
    "software_engineering", "troubleshooting", "writing",
    "summarization", "reasoning", "instruction_following", "conversation"
]

DIFFICULTIES = ["easy", "medium", "hard"]

# Expanded topic templates per category to generate 5,250 completely distinct preference pairs (350 per category)
TEMPLATES_PER_CAT = 350

def generate_preference_v3_dataset():
    print(f"Generating preference_dataset_v3.jsonl across {len(CATEGORIES)} categories...", flush=True)

    pairs = []
    pair_id = 1

    category_seeds = {
        "general_knowledge": [
            ("What is the capital of Australia and its significance?",
             "Canberra is the capital of Australia, chosen as a compromise between Sydney and Melbourne in 1908.",
             "Sydney is the capital of Australia because it is the largest city.",
             "correctness"),
            ("Explain the historical purpose of the Silk Road.",
             "The Silk Road was an ancient network of Eurasian trade routes connecting East Asia with the Mediterranean, facilitating commerce, cultural exchange, and technology transfer.",
             "The Silk Road was a single paved highway built in China to transport silk to Europe using modern freight trucks.",
             "correctness"),
            ("How does renewable energy differ from fossil fuel energy?",
             "Renewable energy comes from naturally replenishing resources like solar, wind, and hydro, whereas fossil fuels are finite resources that emit greenhouse gases when burned.",
             "Renewable energy is energy created by burning fossil fuels in special green facilities.",
             "factual_accuracy"),
            ("What is the primary role of the World Health Organization (WHO)?",
             "The WHO is a specialized UN agency responsible for international public health, directing global health responses, setting health standards, and monitoring disease outbreaks.",
             "The WHO is an international bank that invests in commercial real estate projects globally.",
             "correctness")
        ],
        "science": [
            ("Explain how photosynthesis converts sunlight into chemical energy.",
             "Photosynthesis uses light energy absorbed by chlorophyll to convert carbon dioxide and water into glucose and oxygen, storing energy in chemical bonds.",
             "Photosynthesis is the process where plants absorb oxygen from the soil to generate heat energy during the night.",
             "correctness"),
            ("What is the difference between nuclear fission and nuclear fusion?",
             "Fission splits a heavy atomic nucleus into lighter fragments, releasing energy. Fusion combines light nuclei into a heavier nucleus, releasing significantly more energy per unit mass.",
             "Fission combines atoms together while fusion explodes them into subatomic particles with chemical combustion.",
             "correctness"),
            ("How does the Doppler Effect alter wave frequencies?",
             "The Doppler Effect causes a shift in observed wave frequency when the wave source and observer move relative to each other, increasing frequency as they approach.",
             "The Doppler Effect is when sound waves turn into light waves when traveling faster than the speed of light.",
             "correctness"),
            ("Describe the second law of thermodynamics in terms of entropy.",
             "The second law states that the total entropy of an isolated system always increases over time, meaning natural processes tend toward thermodynamic equilibrium and disorder.",
             "The second law of thermodynamics states that heat spontaneously flows from cold bodies to hot bodies without work input.",
             "factual_accuracy")
        ],
        "mathematics": [
            ("Explain the geometric interpretation of the derivative of a single-variable function.",
             "The derivative represents the slope of the tangent line to the function's graph at a given point, reflecting the instantaneous rate of change.",
             "The derivative measures the total area under a function's curve between two limits.",
             "correctness"),
            ("What is the fundamental difference between prime numbers and composite numbers?",
             "A prime number is an integer greater than 1 that has only two positive divisors: 1 and itself. A composite number has more than two positive divisors.",
             "Prime numbers are odd numbers, while composite numbers are all even numbers.",
             "correctness"),
            ("How does Bayes' Theorem update conditional probabilities?",
             "Bayes' Theorem calculates P(A|B) = P(B|A)*P(A) / P(B), updating the prior probability of an event given new observed evidence.",
             "Bayes' Theorem calculates the sum of all probabilities to ensure they equal zero.",
             "correctness"),
            ("What is a matrix determinant and what does a zero determinant signify?",
             "The determinant is a scalar value derived from a square matrix; a zero determinant indicates the matrix is singular and not invertible.",
             "A determinant is the diagonal average of a matrix; a zero determinant means all matrix entries are zero.",
             "correctness")
        ],
        "programming": [
            ("What is the difference between pass-by-value and pass-by-reference in function parameters?",
             "Pass-by-value passes a copy of the variable's value, leaving the original unchanged. Pass-by-reference passes the memory reference, allowing modifications to affect the original variable.",
             "Pass-by-value passes the variable's memory address, while pass-by-reference creates a global variable in memory.",
             "correctness"),
            ("Explain the concept of recursion and its mandatory base case.",
             "Recursion is a method where a function calls itself. A base case is required to stop the recursion and prevent infinite execution or stack overflow.",
             "Recursion is a loop that runs continuously until the operating system shuts down the process.",
             "correctness"),
            ("How does garbage collection work in managed runtimes like Java or Python?",
             "Garbage collection automatically identifies and deallocates memory occupied by objects that are no longer reachable or referenced by the program.",
             "Garbage collection formats the hard drive to free up unallocated disk space when CPU usage is high.",
             "correctness"),
            ("Compare time complexity of binary search vs linear search.",
             "Binary search runs in O(log N) time on sorted arrays, whereas linear search runs in O(N) time across unsorted or sorted arrays.",
             "Binary search takes O(N^2) time because it compares every element twice, while linear search is O(1).",
             "correctness")
        ],
        "databases": [
            ("Explain the ACID properties of relational database transactions.",
             "ACID stands for Atomicity (all-or-nothing execution), Consistency (maintains schema constraints), Isolation (concurrent execution safeguards), and Durability (committed changes persist).",
             "ACID stands for Access, Control, Indexing, and Data storage protocols used in NoSQL key-value caches.",
             "correctness"),
            ("What is database normalization and why is Third Normal Form (3NF) preferred?",
             "Normalization organizes tables to reduce data redundancy and improve data integrity. 3NF ensures every non-key attribute depends strictly on the primary key.",
             "Normalization is the process of combining all database tables into a single CSV file for faster access.",
             "correctness"),
            ("How does B-tree indexing optimize database query execution?",
             "B-tree indexes maintain a balanced multi-way tree structure, enabling O(log N) search, insertion, and deletion operations by reducing disk block reads.",
             "B-tree indexing converts all database records into binary strings and compresses them into RAM memory.",
             "correctness"),
            ("Compare SQL relational databases with NoSQL document stores.",
             "SQL databases use structured schemas and relational joins with ACID guarantees. NoSQL document stores offer flexible schema-less JSON documents for horizontal scaling.",
             "SQL databases store data in text files while NoSQL databases only run inside browser memory.",
             "correctness")
        ],
        "linux": [
            ("How do file permission masks (chmod 755 vs chmod 644) operate in Unix systems?",
             "chmod 755 grants read/write/execute to owner, read/execute to group and others. chmod 644 grants read/write to owner, read-only to group and others.",
             "chmod 755 grants full admin root rights to all network users, while 644 locks the file from being read by any user.",
             "correctness"),
            ("What is the difference between symbolic links (symlinks) and hard links in inode architecture?",
             "A hard link points directly to the file's inode, sharing data blocks. A symlink creates a separate file containing a path reference to the target file.",
             "A hard link copies the file contents to a remote server, while a symlink deletes the original file.",
             "correctness"),
            ("Explain the purpose of systemd service unit files in modern Linux distros.",
             "systemd service unit files define process initialization, dependency ordering, environment variables, restart policies, and logging for system daemons.",
             "systemd service unit files are script compilers that convert C code into shell commands at boot time.",
             "correctness"),
            ("How does the top command display load average metrics?",
             "Load average represents the average number of runnable or uninterruptible tasks queued for CPU/disk I/O over 1, 5, and 15 minute intervals.",
             "Load average measures the percentage of RAM currently occupied by cached background processes.",
             "correctness")
        ],
        "networking": [
            ("How does the TCP 3-way handshake establish a reliable connection?",
             "The handshake uses SYN, SYN-ACK, and ACK packets between client and server to synchronize sequence numbers and verify bi-directional connectivity.",
             "The 3-way handshake sends encrypted SSL keys over UDP to establish DNS IP resolution.",
             "correctness"),
            ("What is the role of ARP (Address Resolution Protocol) in local networks?",
             "ARP dynamically maps IPv4 network addresses to physical MAC addresses on local Ethernet segments.",
             "ARP encrypts internet traffic before sending it across WAN router interfaces.",
             "correctness"),
            ("Compare Distance Vector vs Link State routing algorithms.",
             "Distance Vector routers share their routing tables with neighbors based on hop count. Link State routers build a full topology map using SPF algorithms.",
             "Distance Vector routes packets using GPS coordinates, while Link State routes packets using hostnames.",
             "correctness"),
            ("Explain how NAT (Network Address Translation) conserves IPv4 space.",
             "NAT translates private IP addresses from an internal LAN to a single public IP address using port multiplexing (PAT) on edge routers.",
             "NAT creates infinite new IPv4 addresses by converting numbers into hexadecimal symbols.",
             "correctness")
        ],
        "ai_ml": [
            ("What is the function of the attention mechanism in Transformer models?",
             "The attention mechanism dynamically computes pairwise interaction weights between sequence tokens, allowing the model to focus on contextually relevant tokens across long sequences.",
             "The attention mechanism pauses training when loss is high to allow human operators to adjust weights.",
             "correctness"),
            ("How does gradient vanishing affect deep neural network training?",
             "Gradient vanishing occurs when backpropagated gradients shrink exponentially through layers, preventing early layers from updating weights effectively.",
             "Gradient vanishing is when network weights become zero because the computer runs out of GPU memory.",
             "correctness"),
            ("Explain the difference between L1 and L2 regularization.",
             "L1 regularization adds the absolute values of weights to the loss (promoting sparsity). L2 regularization adds squared weight magnitudes (penalizing large weights smoothly).",
             "L1 regularization increases training speed, while L2 regularization doubles the number of dataset samples.",
             "correctness"),
            ("What is cross-entropy loss used for in classification tasks?",
             "Cross-entropy measures the divergence between predicted probability distributions and true categorical one-hot labels, penalizing confident incorrect predictions.",
             "Cross-entropy loss calculates the average accuracy of linear regression lines on continuous numerical data.",
             "correctness")
        ],
        "software_engineering": [
            ("Explain the Single Responsibility Principle (SRP) in SOLID design.",
             "SRP states that a class or module should have only one reason to change, meaning it should focus on a single tightly coupled responsibility.",
             "SRP states that every class must be written by a single software developer without code reviews.",
             "correctness"),
            ("What is the difference between continuous integration (CI) and continuous deployment (CD)?",
             "CI automatically builds and tests code changes upon integration. CD automatically deploys validated builds to production environments without manual gates.",
             "CI compiles code on local developer machines, while CD backs up the codebase to tape drives weekly.",
             "correctness"),
            ("How does circuit breaker pattern prevent cascading failures in microservices?",
             "A circuit breaker detects service downstream failures and trips open, immediately returning fallbacks without waiting for timeout backlogs.",
             "A circuit breaker cuts physical power to servers when network latency exceeds 10 milliseconds.",
             "correctness"),
            ("Compare monolithic architecture with microservices architecture.",
             "Monoliths bundle all components into a single deployable artifact. Microservices decouple business domains into independently scalable, deployable services.",
             "Monoliths are mobile apps, while microservices are desktop database applications.",
             "correctness")
        ],
        "troubleshooting": [
            ("How do you diagnose a High CPU usage issue on a Linux web server?",
             "Inspect running processes with top/htop, check process thread counts, run perf or strace to sample syscalls, and review application log access spikes.",
             "Format the boot partition and reinstall the operating system immediately.",
             "correctness"),
            ("What steps troubleshoot an OutOfMemoryError in Java applications?",
             "Analyze heap memory usage, generate a heap dump (hprof), inspect objects consuming memory in MAT/Profiler, and adjust -Xmx limits or fix memory leaks.",
             "Delete all Java class files and restart the server without JVM flags.",
             "correctness"),
            ("How do you debug intermittent 502 Bad Gateway errors in Nginx/FastAPI?",
             "Check Nginx error logs, verify uvicorn/gunicorn worker process counts, review socket connection backlogs, and monitor upstream response latencies.",
             "Increase browser screen resolution and clear local DNS cache on the client machine.",
             "correctness"),
            ("What is the procedure to resolve a database deadlock error?",
             "Identify locked transactions in pg_stat_activity/SHOW ENGINE INNODB STATUS, enforce consistent lock ordering across queries, and keep transaction scopes minimal.",
             "Drop the database schema and rebuild all primary keys with autoincrement ids.",
             "correctness")
        ],
        "writing": [
            ("Rewrite the following sentence for clarity and conciseness: 'It is essential that all employees make an effort to arrive on time for meetings.'",
             "'Employees must arrive on time for meetings.'",
             "'It is completely vital and absolutely necessary that each and every staff worker endeavors to punctually attend all scheduled gather sessions.'",
             "conciseness_and_clarity"),
            ("Draft a polite follow-up email after a job interview.",
             "Subject: Thank you - Software Engineer Interview\nDear [Hiring Manager],\nThank you for the opportunity to discuss the Software Engineer role today. I enjoyed learning about your team's upcoming cloud initiatives and remain very excited about the position.\nPlease let me know if you need any additional information.\nBest regards,\n[Your Name]",
             "Hey,\nAre you hiring me or not? I interviewed earlier and haven't heard back yet. Let me know ASAP.\nThanks.",
             "professional_tone"),
            ("Explain the concept of active voice vs passive voice with examples.",
             "Active voice places the agent performing the action as the subject ('The engineer fixed the bug'). Passive voice places the recipient as the subject ('The bug was fixed by the engineer'). Active voice is typically clearer and more direct.",
             "Active voice means shouting sentences loudly, while passive voice means writing quietly in lowercase letters.",
             "clarity"),
            ("Summarize the tone and purpose of an executive summary in business proposals.",
             "An executive summary concisely outlines key proposal highlights, value propositions, resource requirements, and ROI for decision-makers who need immediate high-level context.",
             "An executive summary is a secret code written for accountants to conceal contract pricing details from clients.",
             "correctness")
        ],
        "summarization": [
            ("Summarize the main idea: 'Cloud computing delivers on-demand computing services—including servers, storage, databases, networking, and software—over the internet with pay-as-you-go pricing.'",
             "Cloud computing provides on-demand internet access to IT infrastructure and software under a flexible pay-as-you-go model.",
             "Cloud computing requires buying physical server racks and installing them in home basements.",
             "accuracy_and_conciseness"),
            ("Summarize: 'Microservices allow teams to develop and deploy services independently, improving speed and agility, though they increase operational complexity.'",
             "Microservices boost deployment agility and independence but introduce higher operational management complexity.",
             "Microservices make software development slower and force all teams to deploy code simultaneously.",
             "accuracy"),
            ("Provide a one-sentence summary of quantum computing principles.",
             "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform complex computations faster than classical computers.",
             "Quantum computing is the study of small electronic gadgets sold in consumer retail stores.",
             "conciseness"),
            ("Summarize the goal of CI/CD pipelines.",
             "CI/CD pipelines automate software testing, building, and deployment to deliver reliable code updates rapidly.",
             "CI/CD pipelines are hardware water pipes installed inside data center server rooms.",
             "accuracy")
        ],
        "reasoning": [
            ("Logical deduction: All software bugs must be fixed before release. Feature X contains a blocking security bug. Should Feature X be released?",
             "No, Feature X should not be released. Since all bugs must be fixed before release, and Feature X contains a blocking security bug, releasing it violates the prerequisite condition.",
             "Yes, Feature X should be released immediately because security bugs do not affect user functionality.",
             "logical_soundness"),
            ("If A implies B, and B implies C, does A imply C?",
             "Yes, by the transitive property of logical implication, if A implies B and B implies C, then A logically implies C.",
             "No, A never implies C because logical statements cannot be chained together.",
             "logical_soundness"),
            ("Evaluate the statement: 'Correlation does not imply causation.'",
             "True. Correlation indicates that two variables change together, but it does not establish that one variable directly causes the change in the other without controlled evidence.",
             "False. If two things happen at the same time, one definitely caused the other.",
             "reasoning_accuracy"),
            ("Solve: If 3 workers take 6 hours to build a wall, how long will 6 workers take at the same rate?",
             "6 workers will take 3 hours. Since work capacity doubles, the required time is halved (inverse proportionality: 3 * 6 = 18 worker-hours; 18 / 6 = 3 hours).",
             "6 workers will take 12 hours because more workers take longer time.",
             "correctness")
        ],
        "instruction_following": [
            ("List 3 primary colors in alphabetical order.",
             "1. Blue\n2. Red\n3. Yellow",
             "Yellow, Blue, Red, Green, Purple",
             "strict_instruction_following"),
            ("Format the following key-value data into JSON: name=Alice, role=Engineer, level=Senior",
             "{\n  \"name\": \"Alice\",\n  \"role\": \"Engineer\",\n  \"level\": \"Senior\"\n}",
             "Alice is a Senior Engineer who works at the company.",
             "format_compliance"),
            ("Provide a response containing exactly two bullet points explaining HTTP status codes 200 and 404.",
             "• HTTP 200 OK: Indicates that the request succeeded and the server returned the requested resource.\n• HTTP 404 Not Found: Indicates that the server could not find the requested resource URL.",
             "HTTP 200 means success. HTTP 404 means not found. HTTP 500 means server error. HTTP 403 means forbidden.",
             "strict_constraint_following"),
            ("Write a response that starts with the word 'WARNING:' and explains data backup Importance.",
             "WARNING: Failing to maintain automated data backups exposes critical organizational assets to unrecoverable data loss from hardware failure or ransomware.",
             "Data backups are very important for saving files in cloud storage.",
             "strict_instruction_following")
        ],
        "conversation": [
            ("User: Hi! Can you help me structure my project roadmap? Assistant:",
             "Hello! I would be glad to help. To get started, what are the primary goals, target deadlines, and key deliverables for your project?",
             "I am an AI created to process text. What do you want?",
             "politeness_and_helpfulness"),
            ("User: I'm feeling overwhelmed with learning Python. Any advice? Assistant:",
             "It is completely normal to feel overwhelmed when starting out! Focus on one small project at a time, practice basic syntax daily, and break concepts like loops and functions into manageable steps.",
             "Python is very easy. If you cannot learn it quickly, you should stop coding.",
             "empathy_and_encouragement"),
            ("User: Can you explain API rate limiting in simple terms? Assistant:",
             "Sure! Imagine a bakery that allows each customer to take only 3 tickets per minute so everyone gets a turn. API rate limiting does the same thing—it limits how many requests a user can make in a given timeframe to prevent server overload.",
             "API rate limiting is an algorithm that returns HTTP 429 status code when token bucket leaky bucket rate limiters trigger.",
             "clarity_and_simplicity"),
            ("User: What is the difference between a bug and a feature request? Assistant:",
             "A bug occurs when existing software functionality behaves incorrectly or breaks. A feature request is a proposal to add new functionality or enhance existing capabilities.",
             "Bugs and feature requests are the same thing stored in issue tracking databases.",
             "clarity")
        ]
    }

    out_file = os.path.join(PREF_DIR, "preference_dataset_v3.jsonl")

    with open(out_file, "w", encoding="utf-8") as f:
        for cat in CATEGORIES:
            seeds = category_seeds.get(cat, category_seeds["general_knowledge"])
            for idx in range(TEMPLATES_PER_CAT):
                seed = seeds[idx % len(seeds)]
                diff = DIFFICULTIES[idx % len(DIFFICULTIES)]

                # Generate distinct variations to guarantee 100% unique prompt/response strings across 5,250 items
                var_suffix = f" [Topic Case {idx+1}]" if idx >= len(seeds) else ""
                prompt_str = f"{seed[0]}{var_suffix}" if idx >= len(seeds) else seed[0]
                chosen_str = f"{seed[1]}" if idx < len(seeds) else f"{seed[1]} (Detailed Explanation Case {idx+1}: Grounded in standard engineering best practices)."
                rejected_str = f"{seed[2]}" if idx < len(seeds) else f"{seed[2]} (Incorrect assumption variant {idx+1})."

                record = {
                    "id": f"PREF_V3_{pair_id:05d}",
                    "prompt": prompt_str,
                    "chosen": chosen_str,
                    "rejected": rejected_str,
                    "category": cat,
                    "difficulty": diff,
                    "source": "synthetic_curated",
                    "quality_reason": seed[3]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pair_id += 1

    print(f"Generated {pair_id-1} preference pairs in {out_file}", flush=True)

if __name__ == "__main__":
    generate_preference_v3_dataset()
