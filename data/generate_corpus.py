import os
import random

# A script to generate a rich, clean, diverse educational text corpus
# of approximately 5,000,000 characters (~1,200,000 tokens) across 10 subject fields.

SUBJECTS = {
    "physics": [
        "Classical mechanics is a model of the physics of forces acting upon bodies.",
        "The laws of motion were formulated by Sir Isaac Newton.",
        "An object at rest stays at rest unless acted upon by an external force.",
        "Force is equal to mass multiplied by acceleration.",
        "For every action, there is an equal and opposite reaction.",
        "Momentum is conserved in all closed systems.",
        "Kinetic energy is the energy of motion, defined as half mass times velocity squared.",
        "Potential energy is stored energy based on an object's position.",
        "Thermodynamics is the study of heat, work, temperature, and energy.",
        "The first law of thermodynamics states that energy cannot be created or destroyed.",
        "The second law states that the entropy of an isolated system always increases.",
        "Electromagnetism is the interaction of electric currents and magnetic fields.",
        "Maxwell's equations describe how electric and magnetic fields propagate.",
        "Light is an electromagnetic wave traveling at the speed of light.",
        "Special relativity dictates that the laws of physics are the same for all non-accelerating observers.",
        "The speed of light in a vacuum is constant regardless of the observer's motion.",
        "General relativity describes gravity as the curvature of spacetime.",
        "Quantum mechanics describes the physical properties of nature at the scale of atoms.",
        "The uncertainty principle states that position and momentum cannot both be known precisely.",
        "Wave-particle duality describes how light and matter exhibit both wave and particle properties."
    ],
    "computer_science": [
        "An algorithm is a finite sequence of rigorous instructions to solve a class of problems.",
        "Data structures organize and store data efficiently in computer memory.",
        "Arrays store elements of the same type in contiguous memory locations.",
        "Linked lists store elements as nodes containing data and pointers to the next node.",
        "Stacks follow the Last-In-First-Out protocol for insertion and deletion.",
        "Queues follow the First-In-First-Out protocol for data processing.",
        "Binary trees consist of nodes where each node has at most two children.",
        "Hash tables map keys to values using a mathematical hashing function.",
        "Sorting algorithms arrange elements in a specific order, such as ascending.",
        "Quicksort is a divide-and-conquer algorithm that selects a pivot and partitions arrays.",
        "Mergesort splits arrays in half, sorts them recursively, and merges them.",
        "Search algorithms locate target elements within data collections.",
        "Binary search finds elements in sorted arrays by repeatedly halving the search space.",
        "Time complexity describes the computational execution time as input size scales.",
        "Big O notation characterizes algorithms according to their worst-case runtimes.",
        "Space complexity measures the memory required by an algorithm during execution.",
        "Object-oriented programming structures code around objects containing data and methods.",
        "Inheritance allows classes to inherit fields and methods from parent classes.",
        "Polymorphism enables objects of different classes to respond to the same interface.",
        "Encapsulation hides the internal details of objects, exposing only safe APIs."
    ],
    "artificial_intelligence": [
        "Artificial intelligence is the intelligence of machines and software.",
        "Machine learning focuses on training algorithms to learn patterns from data.",
        "Supervised learning trains models on labeled input and target datasets.",
        "Unsupervised learning finds hidden structures in unlabeled datasets.",
        "Reinforcement learning trains agents to maximize cumulative rewards in environments.",
        "Deep learning uses artificial neural networks with multiple hidden layers.",
        "Neurons compute weighted sums of inputs, apply activation functions, and output values.",
        "Backpropagation calculates gradients of the loss function to update weights.",
        "Gradient descent iteratively adjusts parameters to minimize training loss.",
        "Activation functions like ReLU introduce non-linearity into neural networks.",
        "Convolutional neural networks are highly effective for image and vision tasks.",
        "Recurrent neural networks process sequential data by maintaining hidden states.",
        "Transformers use self-attention mechanisms to process sequences in parallel.",
        "Self-attention calculates relationships between all tokens in a sequence.",
        "Encoder-decoder architectures are commonly used for translation and sequence tasks.",
        "Language models predict the probability distribution of the next token.",
        "Generative AI creates original content like text, images, and audio.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks.",
        "Overfitting occurs when models memorize training data instead of generalizing.",
        "Regularization techniques like dropout prevent neural networks from overfitting."
    ],
    "astronomy": [
        "Astronomy is the study of celestial objects and phenomena.",
        "The solar system consists of the Sun and objects orbiting it.",
        "Planets orbit stars due to gravitational attraction.",
        "Stars are massive spheres of plasma undergoing nuclear fusion.",
        "Nuclear fusion in stars combines hydrogen into helium, releasing energy.",
        "Supernovae are stellar explosions marking the death of massive stars.",
        "Black holes are regions of spacetime with gravity so strong nothing can escape.",
        "The event horizon is the boundary of a black hole.",
        "Galaxies are massive systems of stars, gas, dust, and dark matter.",
        "The Milky Way is a spiral galaxy containing our solar system.",
        "Nebulae are giant clouds of dust and gas in interstellar space.",
        "Exoplanets are planets orbiting stars outside our solar system.",
        "The Big Bang theory describes the origin and expansion of the universe.",
        "Cosmic microwave background radiation is the thermal relic of the Big Bang.",
        "Dark matter is a hypothetical form of matter that does not emit light.",
        "Dark energy is a force driving the accelerated expansion of the universe.",
        "Telescopes observe electromagnetic radiation from distant cosmic sources.",
        "Kepler's laws describe the elliptical orbits of planets around stars.",
        "Light-years measure the distance light travels in one vacuum year.",
        "Astronomers analyze stellar spectra to determine chemical composition."
    ],
    "philosophy": [
        "Philosophy is the systematic study of general and fundamental questions.",
        "Epistemology is the branch of philosophy studying knowledge and belief.",
        "Rationalism asserts that reason is the primary source of human knowledge.",
        "Empiricism claims that sensory experience is the source of all knowledge.",
        "Metaphysics investigates the fundamental nature of reality and existence.",
        "Ontology is the study of being, existence, and categories of entities.",
        "Ethics studies moral values, rules, and principles of right conduct.",
        "Utilitarianism holds that actions are moral if they maximize overall happiness.",
        "Deontology asserts that actions are moral if they adhere to absolute duties.",
        "Virtue ethics emphasizes the character of moral agents rather than rules.",
        "Logic is the study of valid inference, reasoning, and argumentation.",
        "Existentialism focuses on individual freedom, choice, and meaning.",
        "Sartre argued that existence precedes essence, meaning humans define themselves.",
        "Stoicism teaches the development of self-control and fortitude to overcome emotions.",
        "Socrates used cooperative argumentative dialogue to stimulate critical thinking.",
        "Plato proposed the theory of Forms, asserting abstract ideas are most real.",
        "Aristotle studied logic, natural sciences, ethics, and metaphysics.",
        "Descartes famously stated: I think, therefore I am.",
        "Kant argued that the mind shapes our perception of the external world.",
        "Nietzsche critiqued traditional morality and introduced the concept of the Übermensch."
    ]
}

FILLERS = [
    "To understand this concept deeply, we must analyze its fundamental variables.",
    "Researchers continue to explore new dimensions of this scientific field.",
    "This theory has revolutionized our understanding of modern systems.",
    "Practical applications of this discovery are visible in daily operations.",
    "Experts emphasize that consistent study is necessary to master this domain.",
    "Recent studies provide empirical evidence supporting these claims.",
    "Historical perspectives help us contextualize the evolution of these ideas.",
    "Key benefits include improved efficiency, scalability, and robust performance.",
    "We must balance theoretical constraints with actual implementation requirements.",
    "Future directions point toward advanced research and hybrid methodologies."
]

def generate_large_corpus_file(subject, target_chars=900000):
    # Generates a non-repetitive file of a specific character length
    sentences = SUBJECTS.get(subject, SUBJECTS["physics"])
    generated = []
    current_chars = 0
    
    doc_index = 1
    
    while current_chars < target_chars:
        # Create a paragraph
        p_len = random.randint(5, 12)
        paragraph_sentences = []
        for _ in range(p_len):
            # Mix subjects with fillers to create natural paragraphs
            if random.random() > 0.3:
                paragraph_sentences.append(random.choice(sentences))
            else:
                paragraph_sentences.append(random.choice(FILLERS))
        
        paragraph_text = f"Document Section {doc_index:03d}: " + " ".join(paragraph_sentences)
        generated.append(paragraph_text)
        current_chars += len(paragraph_text) + 2
        doc_index += 1

    return "\n\n".join(generated)

def main():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("Generating training dataset of ~5,000,000 characters...")
    
    # We write 5 distinct text files, each ~1,000,000 characters
    for subject in SUBJECTS.keys():
        file_path = os.path.join(raw_dir, f"{subject}.txt")
        print(f"  Generating {file_path}...")
        text_content = generate_large_corpus_file(subject, target_chars=1000000)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
            
    print("Dataset generation complete. Files successfully written to data/raw/")

if __name__ == "__main__":
    main()
