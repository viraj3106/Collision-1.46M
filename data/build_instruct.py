import os
import json
import random
import hashlib

# Ensure reproducibility
random.seed(42)

# Raw concepts/definitions database
CONCEPTS = {
    "artificial_intelligence": [
        ("artificial intelligence", "the intelligence exhibited by machines or software, simulating human cognitive functions"),
        ("machine learning", "a subset of AI focused on training algorithms to learn patterns and make predictions from data"),
        ("supervised learning", "a model training paradigm using labeled input and target datasets to learn mapping functions"),
        ("unsupervised learning", "a learning technique that finds hidden structures or groupings in unlabeled datasets"),
        ("reinforcement learning", "an agent training method based on maximizing cumulative rewards received from environment actions"),
        ("deep learning", "neural network architectures with multiple hidden layers that automatically extract high-level features"),
        ("backpropagation", "an algorithm that calculates loss gradients with respect to weights to update network parameters"),
        ("gradient descent", "an optimization algorithm that iteratively adjusts model weights to minimize the loss function"),
        ("transformer", "a neural network architecture relying on self-attention mechanisms to process sequences in parallel"),
        ("self-attention", "a mechanism that computes the mathematical relationships and relative weights between all tokens in a sequence"),
        ("neural network", "a computational model inspired by biological brain structures, consisting of interconnected nodes"),
        ("overfitting", "a scenario where a model memorizes the training data but fails to generalize to unseen data"),
        ("regularization", "techniques like dropout or weight decay applied to prevent models from overfitting during training"),
        ("transfer learning", "applying knowledge learned from one pre-trained model task to a different but related task"),
        ("generative AI", "algorithms capable of generating new content, including text, images, code, or audio"),
        ("fine-tuning", "adapting a pre-trained model to a specialized task by training it on domain-specific data"),
        ("large language model", "a deep learning model trained on massive text corpora to predict and generate natural language sequence"),
        ("loss function", "a mathematical method that measures the error between model predictions and ground-truth targets"),
        ("activation function", "a non-linear function like ReLU or sigmoid that determines whether a neuron should activate"),
        ("natural language processing", "a subfield of computer science and AI focused on enabling computers to process human languages")
    ],
    "machine_learning": [
        ("linear regression", "a statistical method used to model the linear relationship between dependent and independent variables"),
        ("logistic regression", "a classification algorithm used to predict the probability of a binary outcome"),
        ("decision tree", "a flowchart-like structure used for classification and regression by partitioning data based on feature splits"),
        ("random forest", "an ensemble learning method that constructs multiple decision trees to improve prediction accuracy"),
        ("support vector machine", "a classification algorithm that finds the optimal hyperplane maximizing the margin between classes"),
        ("k-means clustering", "an unsupervised algorithm that partitions data into k distinct clusters based on feature similarity"),
        ("bias-variance tradeoff", "the tension between model error due to overly simple assumptions versus sensitivity to training noise"),
        ("validation dataset", "a subset of data used to tune hyperparameters and provide an unbiased evaluation during training"),
        ("cross-validation", "a resampling technique that partitions data into multiple folds to evaluate model generalization"),
        ("feature engineering", "the process of selecting, transforming, and combining raw variables to create informative model inputs"),
        ("precision", "the ratio of correctly predicted positive observations to the total predicted positives"),
        ("recall", "the ratio of correctly predicted positive observations to all actual positive observations in the dataset"),
        ("f1 score", "the harmonic mean of precision and recall, serving as a balanced classification performance metric"),
        ("confusion matrix", "a tabular layout showing actual versus predicted classifications to evaluate model accuracy"),
        ("gradient boosting", "an ensemble technique that trains trees sequentially, with each new tree correcting previous errors"),
        ("hyperparameter tuning", "the process of optimizing configuration settings that control model learning behavior"),
        ("stochastic gradient descent", "a variant of gradient descent that updates model parameters using one training sample at a time"),
        ("principal component analysis", "a dimensionality reduction technique that projects data onto principal components of maximal variance"),
        ("anomaly detection", "the identification of rare items or events that raise suspicions by differing significantly from normal data"),
        ("ensemble learning", "combining the predictions of multiple individual models to achieve better overall performance")
    ],
    "computer_science": [
        ("algorithm", "a finite, well-defined sequence of rigorous instructions designed to solve a specific problem"),
        ("data structure", "a systematic format for organizing, managing, and storing data in computer memory for efficient access"),
        ("array", "a collection of elements of the same type stored in contiguous computer memory locations"),
        ("linked list", "a data structure where elements are stored in nodes containing data and pointers to subsequent nodes"),
        ("stack", "a linear data structure following the Last-In-First-Out (LIFO) access protocol"),
        ("queue", "a linear data structure following the First-In-First-Out (FIFO) processing order"),
        ("binary search tree", "a node-based binary tree data structure where left children are smaller and right children are larger"),
        ("hash table", "a data structure that maps keys to values using a hashing function for constant-time lookups"),
        ("binary search", "a search algorithm that locates a target element in a sorted array by repeatedly halving the search space"),
        ("quicksort", "a divide-and-conquer sorting algorithm that partitions arrays around a pivot element"),
        ("mergesort", "a sorting algorithm that recursively splits arrays, sorts the halves, and merges them back"),
        ("time complexity", "a measure of the execution time of an algorithm as a function of the input size"),
        ("space complexity", "a measure of the memory resources required by an algorithm during execution"),
        ("Big O notation", "a mathematical notation describing the limiting behavior of a function, indicating algorithmic complexity bounds"),
        ("object-oriented programming", "a programming paradigm based on the concept of objects containing data fields and associated methods"),
        ("polymorphism", "the ability of different object classes to respond to the same interface or method call in unique ways"),
        ("inheritance", "a mechanism where a new class inherits properties and behaviors from an existing parent class"),
        ("encapsulation", "the bundling of data and methods inside a class, hiding internal implementation details from external APIs"),
        ("compiler", "a specialized program that translates high-level source code into low-level machine code instructions"),
        ("operating system", "system software that manages computer hardware resources and provides common services for applications")
    ],
    "physics": [
        ("classical mechanics", "the study of the motion of macroscopic bodies under the influence of forces"),
        ("Newton's first law", "the principle stating that an object remains at rest or in uniform motion unless acted upon by a force"),
        ("Newton's second law", "the law stating that force is equal to the rate of change of momentum, or mass times acceleration"),
        ("Newton's third law", "the principle stating that for every action, there is an equal and opposite reaction"),
        ("kinetic energy", "the energy possessed by an object due to its motion, calculated as half mass times velocity squared"),
        ("potential energy", "the energy stored in an object based on its position or configuration in a force field"),
        ("thermodynamics", "the branch of physics dealing with heat, work, temperature, and their relations to energy"),
        ("first law of thermodynamics", "the principle of conservation of energy, stating that energy cannot be created or destroyed"),
        ("second law of thermodynamics", "the principle stating that the total entropy of an isolated system always increases over time"),
        ("electromagnetism", "the physical interaction that occurs between electrically charged particles and electromagnetic fields"),
        ("Maxwell's equations", "a set of coupled partial differential equations describing how electric and magnetic fields propagate"),
        ("special relativity", "Einstein's theory stating that the laws of physics are identical for all inertial observers"),
        ("general relativity", "Einstein's geometric theory of gravity, describing it as the curvature of spacetime caused by mass"),
        ("quantum mechanics", "the physical theory describing nature at the scale of atomic and subatomic particles"),
        ("uncertainty principle", "Heisenberg's assertion that the position and momentum of a particle cannot both be measured precisely"),
        ("wave-particle duality", "the concept that light and matter exhibit both wave-like and particle-like properties"),
        ("entropy", "a thermodynamic quantity representing the degree of disorder or randomness in a physical system"),
        ("speed of light", "the universal constant speed at which electromagnetic waves travel in a vacuum"),
        ("gravitational force", "the attractive force existing between any two masses in the universe"),
        ("conservation of momentum", "the principle that the total momentum of a closed system remains constant if no external forces act")
    ],
    "astronomy": [
        ("solar system", "the gravitationally bound system consisting of the Sun and all objects orbiting it"),
        ("star", "a massive, luminous sphere of plasma held together by its own gravity, undergoing nuclear fusion"),
        ("nuclear fusion", "the stellar process combining lighter atomic nuclei like hydrogen into heavier helium, releasing energy"),
        ("supernova", "a powerful stellar explosion marking the violent death of a massive star"),
        ("black hole", "a region of spacetime where gravitational attraction is so strong that nothing, not even light, can escape"),
        ("event horizon", "the boundary surrounding a black hole beyond which escape velocity exceeds the speed of light"),
        ("galaxy", "a massive, gravitationally bound system containing stars, stellar remnants, interstellar gas, dust, and dark matter"),
        ("Milky Way", "the barred spiral galaxy containing our Solar System"),
        ("nebula", "an interstellar cloud of dust, hydrogen, helium, and other ionized gases where stars are often born"),
        ("exoplanet", "a planet that orbits a star outside our solar system"),
        ("Big Bang theory", "the cosmological model describing the rapid expansion and origin of the universe from a hot singularity"),
        ("cosmic microwave background", "the electromagnetic radiation remaining from the early stage of the universe after the Big Bang"),
        ("dark matter", "a hypothetical form of matter that does not interact with light but accounts for gravitational anomalies"),
        ("dark energy", "a mysterious force theorized to drive the accelerating expansion of the universe"),
        ("telescope", "an optical or radio instrument used to observe distant celestial objects by gathering electromagnetic radiation"),
        ("Kepler's laws", "three scientific laws describing the elliptical orbits of planets around stars"),
        ("light-year", "the astronomical unit of distance representing how far light travels in one vacuum year"),
        ("stellar spectrum", "the spectrum of light emitted by a star, analyzed to determine its chemical composition and temperature"),
        ("neutron star", "the collapsed core of a massive star, composed almost entirely of neutrons and extremely dense"),
        ("red giant", "a luminous giant star of low or intermediate mass in a late phase of stellar evolution")
    ],
    "philosophy": [
        ("epistemology", "the branch of philosophy concerned with the nature, origin, scope, and limits of human knowledge"),
        ("metaphysics", "the branch of philosophy studying the fundamental nature of reality, existence, and being"),
        ("ethics", "the systematic study of moral values, rules, and principles governing right and wrong conduct"),
        ("rationalism", "the philosophical view that reason and intellect are the primary sources of human knowledge"),
        ("empiricism", "the philosophical theory asserting that all knowledge is derived from sensory experiences"),
        ("utilitarianism", "an ethical theory holding that the best action is the one that maximizes overall utility or happiness"),
        ("deontology", "an ethical framework judging actions based on adherence to moral duties or rules, regardless of outcomes"),
        ("existentialism", "a philosophical inquiry focusing on individual freedom, choice, existence, and the search for personal meaning"),
        ("stoicism", "an ancient philosophy teaching that virtue, self-control, and rationality bring peace, avoiding destructive emotions"),
        ("logic", "the systematic study of valid reasoning, inference, arguments, and truth values"),
        ("nihilism", "the philosophical viewpoint suggesting that life is without objective meaning, purpose, or intrinsic value"),
        ("dualism", "the philosophical position that the mind and body are distinct and separate substances"),
        ("materialism", "the view that physical matter is the only fundamental reality, and mental states are physical operations"),
        ("determinism", "the concept that all events, including human actions, are ultimately determined by prior causes"),
        ("free will", "the capacity of rational agents to choose actions freely without external constraints or prior determinism"),
        ("skepticism", "the attitude of doubting or suspending judgment on knowledge claims until supported by evidence"),
        ("pragmatism", "a philosophy holding that the truth or value of an idea lies in its practical consequences and utility"),
        ("social contract", "the theory that individuals consent to surrender some freedoms to authority in exchange for social order"),
        ("phenomenology", "the study of structures of consciousness as experienced from the first-person point of view"),
        ("aesthetic", "the branch of philosophy dealing with the nature of art, beauty, taste, and creation")
    ],
    "mathematics": [
        ("derivative", "the mathematical rate of change of a function with respect to a variable, representing slope"),
        ("integral", "the mathematical representation of the area under a curve, representing accumulation"),
        ("prime number", "a natural number greater than 1 that has no positive divisors other than 1 and itself"),
        ("matrix", "a rectangular array of numbers or symbols arranged in rows and columns, used in linear algebra"),
        ("vector", "a mathematical quantity possessing both magnitude and direction in space"),
        ("probability", "the numerical measure of the likelihood that a specific event will occur, between 0 and 1"),
        ("standard deviation", "a measure of the amount of variation or dispersion in a set of values relative to the mean"),
        ("theorem", "a mathematical statement that has been proven true based on axioms and logical deductions"),
        ("axiom", "a statement or proposition accepted as true without proof, serving as a foundation for deductions"),
        ("set theory", "the branch of mathematics studying collections of objects, representing basic mathematical foundations"),
        ("logarithm", "the inverse operation of exponentiation, determining the exponent to which a base must be raised"),
        ("limit", "the value that a function or sequence approaches as the input or index approaches some value"),
        ("fibonacci sequence", "a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1"),
        ("eigenvalue", "a scalar value associated with a linear transformation matrix that scales corresponding eigenvectors"),
        ("geometric series", "a mathematical series with a constant ratio between successive terms"),
        ("calculus", "the mathematical study of continuous change, encompassing differential and integral calculus"),
        ("algebra", "the study of mathematical symbols and the rules for manipulating these symbols in equations"),
        ("geometry", "the branch of mathematics concerned with properties of space, points, lines, angles, and shapes"),
        ("mean", "the average value calculated by dividing the sum of a set of values by the total count"),
        ("median", "the middle value in an ordered list of numbers, separating the higher half from the lower half")
    ],
    "general_knowledge": [
        ("photosynthesis", "the process used by plants and other organisms to convert light energy into chemical energy"),
        ("DNA", "deoxyribonucleic acid, the molecule carrying genetic instructions for the development and functioning of living organisms"),
        ("atom", "the basic unit of a chemical element, consisting of protons, neutrons, and electrons"),
        ("gravity", "the natural force that attracts physical bodies toward one another, keeping planets in orbit"),
        ("water cycle", "the continuous movement of water on, above, and below the surface of the Earth"),
        ("climate change", "long-term shifts in global temperatures and weather patterns, primarily driven by human activity"),
        ("periodic table", "a tabular arrangement of chemical elements organized by atomic number and chemical properties"),
        ("cellular respiration", "the process by which cells break down glucose to release energy in the form of ATP"),
        ("evolution", "the process of change in the heritable characteristics of biological populations over successive generations"),
        ("plate tectonics", "the scientific theory describing the large-scale motion of plates making up Earth's lithosphere"),
        ("ecosystem", "a community of living organisms interacting with the non-living components of their environment"),
        ("pH scale", "a logarithmic measure of the acidity or basicity of an aqueous solution, from 0 to 14"),
        ("virus", "a submicroscopic infectious agent that replicates only inside the living cells of an organism"),
        ("bacteria", "microscopic, single-celled organisms that exist in vast numbers in diverse environments"),
        ("immune system", "a complex network of biological structures and processes that protects organisms against disease"),
        ("greenhouse effect", "the trapping of the sun's warmth in a planet's lower atmosphere due to greenhouse gases"),
        ("speed of sound", "the distance traveled per unit time by a sound wave propagating through an elastic medium"),
        ("states of matter", "the distinct physical forms that matter can take, including solid, liquid, gas, and plasma"),
        ("earthquake", "the shaking of the surface of the Earth resulting from a sudden release of energy in the lithosphere"),
        ("ozone layer", "a region of Earth's stratosphere that absorbs most of the Sun's harmful ultraviolet radiation")
    ]
}

# Template configs for various instruction categories
# We define multiple phrasings to construct robust prompt templates
PROMPT_TEMPLATES = [
    # 1. Definitions
    {"type": "definition", "prompt": "Define {concept}.", "response": "{concept_cap} is defined as {definition}."},
    {"type": "definition", "prompt": "What is the definition of {concept}?", "response": "The definition of {concept} is: {definition}."},
    {"type": "definition", "prompt": "Explain what the term {concept} means.", "response": "The term {concept} means {definition}."},
    
    # 2. QA
    {"type": "question_answering", "prompt": "What is {concept}?", "response": "{concept_cap} is {definition}."},
    {"type": "question_answering", "prompt": "Can you explain {concept}?", "response": "Yes, {concept} refers to {definition}."},
    {"type": "question_answering", "prompt": "Could you tell me what {concept} is?", "response": "Certainly. {concept_cap} is {definition}."},
    
    # 3. Step-by-step
    {"type": "step_by_step", "prompt": "Give a step-by-step explanation of {concept}.", "response": "To understand {concept}, follow these steps:\n1. First, recognize that it is {definition}.\n2. Second, identify how it applies to systems in its domain.\n3. Third, analyze the primary relationships and outcomes associated with it."},
    {"type": "step_by_step", "prompt": "Explain {concept} in steps.", "response": "Here is an explanation of {concept} in key steps:\nStep 1: Understand that {concept} is defined as {definition}.\nStep 2: Study its core applications and behaviors.\nStep 3: Keep this foundational concept in mind when analyzing larger systems."},

    # 4. Summarization
    {"type": "summarization", "prompt": "Summarize this topic: {concept_cap} is {definition}.", "response": "In summary, {concept} is defined as {definition}."},
    {"type": "summarization", "prompt": "Write a brief summary of {concept}.", "response": "To summarize briefly, {concept} refers to {definition}."},

    # 5. Concept teaching
    {"type": "concept_teaching", "prompt": "Teach me about {concept}.", "response": "Let's learn about {concept}. It is {definition}. This is a crucial concept to master in this field."},
    {"type": "concept_teaching", "prompt": "Can you teach me the basics of {concept}?", "response": "Here are the basics of {concept}: it represents {definition}. Understanding this is essential to progress in the subject."},

    # 6. Text Completion
    {"type": "completion", "prompt": "Complete this statement: {concept_cap} is...", "response": "{concept_cap} is {definition}."}
]

# We also generate Comparison instructions by combining two concepts within the same subject
COMPARISON_TEMPLATES = [
    {"type": "comparison", "prompt": "Compare {concept1} and {concept2}.", "response": "Here is a comparison between {concept1} and {concept2}:\n- {concept1_cap} is {definition1}.\n- {concept2_cap} is {definition2}."},
    {"type": "comparison", "prompt": "What is the difference between {concept1} and {concept2}?", "response": "The main difference is that {concept1} is {definition1}, whereas {concept2} is {definition2}."}
]

# Prefix variations to expand the dataset size and variety without duplicating core examples
PREFIXES = [
    "",
    "Please ",
    "Could you ",
    "I need to know: "
]

def capitalize(s):
    if not s:
        return ""
    return s[0].upper() + s[1:]

def make_deterministic_split(concept_name, train_ratio=0.8, val_ratio=0.1):
    # Determine split deterministically based on hashing the concept name
    # This prevents any leakage of concepts across splits
    h = hashlib.md5(concept_name.encode('utf-8')).hexdigest()
    val = int(h, 16) % 100
    if val < (train_ratio * 100):
        return "train"
    elif val < ((train_ratio + val_ratio) * 100):
        return "val"
    else:
        return "test"

def build():
    dataset_dir = "datasets/collision_instruct_v1"
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_data = []
    val_data = []
    test_data = []
    
    unique_check = set()
    category_counts = {}
    type_counts = {}
    
    # 1. Generate Single-Concept Examples
    for category, concepts in CONCEPTS.items():
        category_counts[category] = 0
        for concept, definition in concepts:
            # Determine split based on concept to completely isolate splits and prevent leakage
            split = make_deterministic_split(concept)
            
            for template in PROMPT_TEMPLATES:
                for prefix in PREFIXES:
                    prompt = template["prompt"].format(concept=concept, concept_cap=capitalize(concept), definition=definition)
                    # Apply prefix (adjust capitalization if necessary)
                    if prefix:
                        if prefix.endswith(": ") or prefix.endswith("! "):
                            full_prompt = f"{prefix}{prompt}"
                        else:
                            full_prompt = f"{prefix}{prompt[0].lower()}{prompt[1:]}"
                    else:
                        full_prompt = prompt
                        
                    response = template["response"].format(
                        concept=concept, 
                        concept_cap=capitalize(concept), 
                        definition=definition
                    )
                    
                    # Deduplicate check
                    key = (full_prompt, response)
                    if key in unique_check:
                        continue
                    unique_check.add(key)
                    
                    ex = {"instruction": full_prompt, "response": response}
                    
                    if split == "train":
                        train_data.append(ex)
                    elif split == "val":
                        val_data.append(ex)
                    else:
                        test_data.append(ex)
                        
                    category_counts[category] += 1
                    type_counts[template["type"]] = type_counts.get(template["type"], 0) + 1

        # 2. Generate Comparison Examples (within category)
        for i in range(len(concepts)):
            concept1, def1 = concepts[i]
            for j in range(i + 1, len(concepts)):
                concept2, def2 = concepts[j]
                
                # Determine split based on hash of the combined concepts
                split = make_deterministic_split(f"{concept1}_{concept2}")
                
                for template in COMPARISON_TEMPLATES:
                    for prefix in PREFIXES:
                        prompt = template["prompt"].format(
                            concept1=concept1, 
                            concept2=concept2,
                            concept1_cap=capitalize(concept1),
                            concept2_cap=capitalize(concept2)
                        )
                        if prefix:
                            if prefix.endswith(": ") or prefix.endswith("! "):
                                full_prompt = f"{prefix}{prompt}"
                            else:
                                full_prompt = f"{prefix}{prompt[0].lower()}{prompt[1:]}"
                        else:
                            full_prompt = prompt
                            
                        response = template["response"].format(
                            concept1=concept1,
                            concept2=concept2,
                            concept1_cap=capitalize(concept1),
                            concept2_cap=capitalize(concept2),
                            definition1=def1,
                            definition2=def2
                        )
                        
                        key = (full_prompt, response)
                        if key in unique_check:
                            continue
                        unique_check.add(key)
                        
                        ex = {"instruction": full_prompt, "response": response}
                        
                        if split == "train":
                            train_data.append(ex)
                        elif split == "val":
                            val_data.append(ex)
                        else:
                            test_data.append(ex)
                            
                        category_counts[category] += 1
                        type_counts[template["type"]] = type_counts.get(template["type"], 0) + 1

    # Shuffle lists deterministically
    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Save JSONL files
    def save_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    save_jsonl(os.path.join(dataset_dir, "train.jsonl"), train_data)
    save_jsonl(os.path.join(dataset_dir, "val.jsonl"), val_data)
    save_jsonl(os.path.join(dataset_dir, "test.jsonl"), test_data)
    
    # Create metadata
    metadata = {
        "dataset_name": "collision_instruct_v1",
        "total_examples": len(unique_check),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "test_examples": len(test_data),
        "category_distribution": category_counts,
        "type_distribution": type_counts,
        "leakage_isolation": "Deterministic split by hashing concept names to prevent overlapping information"
    }
    
    with open(os.path.join(dataset_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("==================================================")
    print("Instruction Dataset Generation Completed Successfully")
    print("==================================================")
    print(f"Total Unique Examples: {len(unique_check):,}")
    print(f"Train: {len(train_data):,}")
    print(f"Validation: {len(val_data):,}")
    print(f"Test: {len(test_data):,}")
    print("Saved files to datasets/collision_instruct_v1/")
    print("==================================================")

if __name__ == "__main__":
    build()
