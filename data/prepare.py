import os
import argparse

def generate_sample_text():
    # A synthetic dataset designed to be educational, repetitive, and contain rich text patterns
    # for testing the model.
    text_blocks = [
        "COLLISION-1M is a small decoder-only Transformer language model.",
        "We are training COLLISION-1M from scratch on a computer CPU.",
        "A language model learns to predict the next token in a sequence.",
        "The model uses causal self-attention to read previous tokens.",
        "Each token is represented by an embedding vector of numbers.",
        "Attention heads compute relationships between words and sentences.",
        "The neural network updates its weights using gradient descent and backpropagation.",
        "AdamW is the optimizer used to train the language model weights.",
        "A checkpoint saves the model weights and optimizer state so we can resume training.",
        "Streamlit displays the COLLISION LAB dashboard for training progress.",
        "The physics of a collision involves conservation of momentum and energy.",
        "An elastic collision conserves both kinetic energy and momentum.",
        "An inelastic collision converts kinetic energy into other forms of energy like heat.",
        "When particles collide, they exchange energy and momentum.",
        "The collider accelerates particles to near the speed of light.",
        "High-energy collisions help physicists discover new particles and forces.",
        "In deep learning, we stack multiple layers of attention and feed-forward networks.",
        "Positional embeddings allow the model to understand the order of tokens in a sequence.",
        "A tokenizer splits text into integers called token IDs.",
        "Evaluating the model gives us the training loss, validation loss, and perplexity.",
        "Perplexity measures how well the model predicts sample text.",
        "We can scale the model: COLLISION-1M to COLLISION-10M and then COLLISION-50M.",
        "The future of learning is built on feedback loops and model alignment."
    ]
    # Replicate text blocks to make a reasonably-sized text file for the tokenizer and model training
    replicated_text = "\n".join(text_blocks * 50)
    return replicated_text

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset directories and sample data")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data directory")
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    sample_path = os.path.join(args.raw_dir, "sample.txt")
    
    # Do not overwrite raw data if it already exists
    if not os.path.exists(sample_path):
        sample_text = generate_sample_text()
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"Created demo raw dataset at {sample_path}")
    else:
        print(f"Sample file already exists at {sample_path}. Skipping creation.")

if __name__ == "__main__":
    main()
