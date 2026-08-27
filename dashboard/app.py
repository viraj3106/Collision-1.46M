import os
import time
import json
import pandas as pd
import streamlit as st
import torch

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import generate
from collision.config import CHECKPOINT_DIR, TOKENIZER_DIR, EXPERIMENT_DIR
from data.stats import get_latest_version_dir

# Page Config
st.set_page_config(page_title="COLLISION LAB", page_icon="💥", layout="wide")

st.title("💥 COLLISION LAB Dashboard")
st.write("---")

# Discover files and metadata
latest_dataset_dir = get_latest_version_dir()
log_path = os.path.join(EXPERIMENT_DIR, "training_log.csv")
latest_cp_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
model_exists = os.path.exists(latest_cp_path)

# 1. Load active dataset version metadata
dataset_meta = {}
if latest_dataset_dir and os.path.exists(os.path.join(latest_dataset_dir, "metadata.json")):
    try:
        with open(os.path.join(latest_dataset_dir, "metadata.json"), "r") as f:
            dataset_meta = json.load(f)
    except Exception:
        pass

# 2. Load active model configuration from latest checkpoint
model_cfg_dict = {}
best_val_loss = float('inf')
best_checkpoint_name = "N/A"
checkpoint_history = []

if os.path.exists(CHECKPOINT_DIR):
    checkpoint_history = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    checkpoint_history = sorted(checkpoint_history)

if model_exists:
    try:
        checkpoint = torch.load(latest_cp_path, map_location="cpu")
        model_cfg_dict = checkpoint.get("config", {})
        # Find best checkpoint (lowest validation loss)
        for cp_name in checkpoint_history:
            cp_path = os.path.join(CHECKPOINT_DIR, cp_name)
            try:
                cp = torch.load(cp_path, map_location="cpu")
                cp_val_loss = cp.get("val_loss", float('inf'))
                if cp_val_loss < best_val_loss:
                    best_val_loss = cp_val_loss
                    best_checkpoint_name = cp_name
            except Exception:
                pass
    except Exception:
        pass

# UI Columns Layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("📋 Dataset & Model Specs")
    
    # Dataset section
    st.subheader("Dataset Info")
    if dataset_meta:
        st.write(f"- **Dataset Version**: {dataset_meta.get('dataset_version', 'N/A')}")
        st.write(f"- **Files**: {', '.join(dataset_meta.get('source_files', []))}")
        st.write(f"- **Total Tokens**: {dataset_meta.get('token_count', 0):,}")
        st.write(f"- **Training Tokens**: {dataset_meta.get('train_tokens', 0):,}")
        st.write(f"- **Validation Tokens**: {dataset_meta.get('validation_tokens', 0):,}")
    else:
        st.info("No active dataset metadata found. Run data.prepare & data.tokenize.")

    st.write("---")

    # Model section
    st.subheader("Model Config")
    if model_cfg_dict:
        model_cfg = ModelConfig(**model_cfg_dict)
        st.write(f"- **Model Parameter Count**: {model_cfg.calculate_parameter_count():,}")
        st.write(f"- **Transformer Layers**: {model_cfg.n_layer}")
        st.write(f"- **Embedding Size**: {model_cfg.d_model}")
        st.write(f"- **Vocabulary Size**: {model_cfg.vocab_size}")
        st.write(f"- **Context Length**: {model_cfg.max_seq_len}")
    else:
        st.info("No trained model loaded yet.")

    st.write("---")

    # Checkpoint log section
    st.subheader("Checkpoints Management")
    if model_exists:
        st.write(f"- **Latest Checkpoint**: `latest.pt` (`{sorted(checkpoint_history)[-1] if checkpoint_history else 'N/A'}`)")
        st.write(f"- **Best Validation Checkpoint**: `{best_checkpoint_name}` (Val Loss: {best_val_loss:.4f})")
        st.write("**Checkpoint History Log**:")
        for cp in checkpoint_history:
            st.write(f"  - `{cp}`")
    else:
        st.info("No checkpoints found.")

with col_right:
    st.header("⚡ Training & Inference")

    # Training stats section
    st.subheader("Active Training Status")
    if os.path.exists(log_path) and os.path.getsize(log_path) > 50:
        try:
            df = pd.read_csv(log_path)
            latest_row = df.iloc[-1]
            
            t_col1, t_col2, t_col3 = st.columns(3)
            t_col1.metric("Current Step", int(latest_row['step']))
            t_col2.metric("Validation Loss", f"{latest_row['val_loss']:.4f}")
            t_col3.metric("Perplexity", f"{latest_row['perplexity']:.2f}")

            # Loss plot
            st.line_chart(df.set_index('step')[['train_loss', 'val_loss']])
        except Exception as e:
            st.error(f"Error reading logs: {e}")
    else:
        st.info("Training statistics empty. Start training to log real-time stats.")

    st.write("---")

    # Generation UI
    st.subheader("Interactive Text Generation")
    if model_exists:
        prompt = st.text_input("Prompt Box", "COLLISION-1M is")
        
        g_col1, g_col2, g_col3 = st.columns(3)
        temp = g_col1.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        top_k = g_col2.slider("Top-K", 1, 100, 50, 1)
        top_p = g_col3.slider("Top-P", 0.0, 1.0, 0.9, 0.05)
        
        max_toks = st.slider("Max Tokens", 10, 300, 50, 10)
        
        if st.button("GENERATE"):
            with st.spinner("Decoding output..."):
                try:
                    checkpoint = torch.load(latest_cp_path, map_location="cpu")
                    model_cfg = ModelConfig(**checkpoint["config"])
                    model = CollisionTransformer(model_cfg)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    
                    tokenizer = BPETokenizer()
                    tokenizer.load(TOKENIZER_DIR)
                    
                    generated_text = generate(
                        model, tokenizer, prompt=prompt, max_tokens=max_toks,
                        temperature=temp, top_k=top_k, top_p=top_p, device="cpu"
                    )
                    st.success("Generation Complete:")
                    st.write(f"_{generated_text}_")
                except Exception as e:
                    st.error(f"Generation error: {e}")
    else:
        st.warning("Please train the model first to enable text generation.")
