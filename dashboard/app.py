import os
import time
import pandas as pd
import streamlit as st
import torch

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import generate
from collision.config import CHECKPOINT_DIR, TOKENIZER_DIR, EXPERIMENT_DIR

# Set page config
st.set_page_config(page_title="COLLISION LAB", page_icon="💥", layout="wide")

st.title("💥 COLLISION LAB")
st.write("---")

# Load log file
log_path = os.path.join(EXPERIMENT_DIR, "training_log.csv")
latest_cp_path = os.path.join(CHECKPOINT_DIR, "latest.pt")

# Determine status
status = "STOPPED"
if os.path.exists(latest_cp_path):
    status = "COMPLETED" # or STOPPED if training is not actively running, let's say STOPPED/COMPLETED

# Check if model checkpoint exists
model_exists = os.path.exists(latest_cp_path)

# Sidebar with status info
st.sidebar.header("Model Specifications")
device = "CUDA" if torch.cuda.is_available() else "CPU"
st.sidebar.metric("Model Architecture", "Decoder Transformer")
st.sidebar.metric("Target Parameters", "~1 Million")
st.sidebar.metric("Compute Device", device)
st.sidebar.metric("Training Status", status)

# Main layout cols
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Training Progress & Metrics")
    
    if os.path.exists(log_path) and os.path.getsize(log_path) > 50:
        try:
            df = pd.read_csv(log_path)
            
            # Show latest metrics
            latest_row = df.iloc[-1]
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Current Step", int(latest_row['step']))
            m_col2.metric("Training Loss", f"{latest_row['train_loss']:.4f}")
            m_col3.metric("Validation Loss", f"{latest_row['val_loss']:.4f}")
            m_col4.metric("Perplexity", f"{latest_row['perplexity']:.2f}")
            
            m_col5, m_col6, m_col7 = st.columns(3)
            m_col5.metric("Tokens Processed", f"{int(latest_row['tokens_processed']):,}")
            m_col6.metric("CPU Utilization", f"{latest_row['cpu_util']:.1f}%")
            m_col7.metric("Latest Checkpoint", f"latest.pt")
            
            # Loss graph
            st.subheader("Training and Validation Loss Curve")
            st.line_chart(df.set_index('step')[['train_loss', 'val_loss']])
            
        except Exception as e:
            st.error(f"Error reading training logs: {e}")
    else:
        st.info("No training logs found. The model is currently NOT TRAINED.")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Current Step", "0")
        m_col2.metric("Training Loss", "N/A")
        m_col3.metric("Validation Loss", "N/A")

with col2:
    st.header("Interactive Generation")
    
    if model_exists:
        st.success("COLLISION-1M model checkpoint loaded.")
        
        prompt = st.text_input("Enter a prompt", "COLLISION-1M is")
        
        # Generation sliders
        temp = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        top_k = st.slider("Top-K", 1, 100, 50, 1)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05)
        max_toks = st.slider("Max Generated Tokens", 10, 200, 50, 10)
        
        if st.button("GENERATE"):
            with st.spinner("Generating text..."):
                try:
                    # Load model from checkpoint
                    checkpoint = torch.load(latest_cp_path, map_location="cpu")
                    model_cfg = ModelConfig(**checkpoint["config"])
                    model = CollisionTransformer(model_cfg)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    
                    tokenizer = BPETokenizer()
                    tokenizer.load(TOKENIZER_DIR)
                    
                    output_text = generate(
                        model, 
                        tokenizer, 
                        prompt=prompt, 
                        max_tokens=max_toks, 
                        temperature=temp, 
                        top_k=top_k, 
                        top_p=top_p, 
                        device="cpu"
                    )
                    
                    st.subheader("Model Output")
                    st.write(f"_{output_text}_")
                except Exception as e:
                    st.error(f"Generation error: {e}")
    else:
        st.warning("Model checkpoint not found. Please train COLLISION-1M first to enable text generation.")
        st.button("GENERATE", disabled=True)
