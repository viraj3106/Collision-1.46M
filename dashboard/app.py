import os
import sys
import time
import torch
import torch.nn.functional as F
import streamlit as st

# Resolve project root path and insert into Python path to prevent import errors
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import top_k_top_p_filtering
from collision.config import TOKENIZER_DIR

# Set Page Config with Dark Mode & Lab Aesthetic
st.set_page_config(
    page_title="COLLISION LAB",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Premium Styling
st.markdown(
    """
    <style>
    /* Dark Slate / Zinc Aesthetic */
    .stApp {
        background-color: #09090b;
        color: #f4f4f5;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers & Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    /* Title Accent */
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        margin-bottom: 0.25rem;
        background: linear-gradient(to right, #ffffff, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background-color: #18181b;
        border: 1px solid #27272a;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #a1a1aa;
        margin-bottom: 1.5rem;
    }
    
    .status-dot {
        height: 6px;
        width: 6px;
        background-color: #22c55e;
        border-radius: 50%;
        margin-right: 6px;
        display: inline-block;
        box-shadow: 0 0 8px #22c55e;
    }
    
    /* Hero stats styling */
    .hero-stats-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #27272a;
        padding-bottom: 1.5rem;
    }
    
    .hero-stat-card {
        background-color: #09090b;
        border: 1px solid #27272a;
        border-radius: 6px;
        padding: 10px 16px;
        min-width: 140px;
    }
    
    .hero-stat-value {
        font-size: 13px;
        font-weight: 700;
        color: #2563eb;
        letter-spacing: 0.05em;
    }
    
    .hero-stat-label {
        font-size: 10px;
        color: #71717a;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 2px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 0.5rem 1.75rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stButton>button:hover {
        background-color: #1d4ed8 !important;
        border-color: #2563eb !important;
        transform: translateY(-1px);
    }
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Action Buttons (Clear, Copy, Regenerate) */
    div.stButton > button[key^="action_"] {
        background-color: #18181b !important;
        color: #e4e4e7 !important;
        border: 1px solid #27272a !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        padding: 4px 10px !important;
        font-weight: 500 !important;
    }
    div.stButton > button[key^="action_"]:hover {
        background-color: #27272a !important;
        border-color: #3f3f46 !important;
        color: #ffffff !important;
    }
    
    /* Sample Prompt Button Pills */
    div.stButton > button[key^="pill_"] {
        background-color: #09090b !important;
        color: #a1a1aa !important;
        border: 1px solid #27272a !important;
        border-radius: 9999px !important;
        padding: 4px 12px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        margin: 2px !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[key^="pill_"]:hover {
        border-color: #2563eb !important;
        color: #3b82f6 !important;
        background-color: #172554 !important;
    }
    
    /* Card Panel */
    .panel-card {
        background-color: #09090b;
        border: 1px solid #27272a;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 1.5rem;
    }
    
    /* Info Spec Item */
    .spec-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #18181b;
        font-size: 13px;
    }
    .spec-item:last-child {
        border-bottom: none;
    }
    .spec-label {
        color: #a1a1aa;
    }
    .spec-value {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Timeline style */
    .timeline-container {
        padding: 6px 12px;
        border-left: 1px solid #27272a;
        margin-left: 8px;
    }
    .timeline-node {
        position: relative;
        padding-left: 16px;
        margin-bottom: 12px;
    }
    .timeline-node::before {
        content: '';
        position: absolute;
        left: -21px;
        top: 5px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #27272a;
    }
    .timeline-node.active::before {
        background-color: #2563eb;
        box-shadow: 0 0 6px #2563eb;
    }
    .timeline-node-title {
        font-weight: 600;
        font-size: 12px;
        color: #f4f4f5;
    }
    .timeline-node-desc {
        font-size: 11px;
        color: #71717a;
    }
    
    /* Metric widget custom overrides */
    div[data-testid="stMetric"] {
        background-color: #09090b !important;
        border: 1px solid #18181b !important;
        padding: 10px 14px !important;
        border-radius: 4px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #71717a !important;
        font-size: 11px !important;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Disclaimer */
    .subtle-disclaimer {
        font-size: 11px;
        color: #71717a;
        line-height: 1.5;
        border-top: 1px solid #27272a;
        padding-top: 1rem;
        margin-top: 1.5rem;
    }
    
    /* Comparison Container */
    .comparison-box {
        display: flex;
        align-items: center;
        justify-content: space-around;
        background-color: #09090b;
        border: 1px solid #27272a;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 0.75rem;
    }
    .comparison-item {
        text-align: center;
    }
    .comparison-title {
        font-size: 10px;
        color: #71717a;
        font-weight: 600;
        text-transform: uppercase;
    }
    .comparison-value {
        font-size: 16px;
        color: #ffffff;
        font-weight: 700;
        margin-top: 2px;
    }
    .comparison-arrow {
        color: #2563eb;
        font-weight: bold;
        font-size: 16px;
    }
    
    /* Customize input and textarea inputs */
    textarea {
        background-color: #09090b !important;
        color: #ffffff !important;
        border: 1px solid #27272a !important;
        border-radius: 4px !important;
    }
    textarea:focus {
        border-color: #2563eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 1. Discover Checkpoints (Cached)
@st.cache_data(show_spinner=False)
def load_checkpoint_metadata(path):
    try:
        if not os.path.exists(path):
            return {"step": "Missing", "val_loss": None, "config": {}, "error": True}
        cp = torch.load(path, map_location="cpu")
        return {
            "step": cp.get("step", "N/A"),
            "val_loss": cp.get("val_loss", None),
            "config": cp.get("config", {}),
            "error": False
        }
    except Exception as e:
        return {"step": "Corrupted", "val_loss": None, "config": {}, "error": str(e)}

def discover_checkpoints():
    checkpoint_options = {}
    
    # We prioritize phase6, then phase5, then root checkpoints.
    dirs_to_check = [
        ("Phase 6", os.path.join("checkpoints", "phase6")),
        ("Phase 5", os.path.join("checkpoints", "phase5")),
        ("Root", "checkpoints")
    ]
    
    for phase_name, folder in dirs_to_check:
        if os.path.exists(folder):
            for file in sorted(os.listdir(folder)):
                if file.endswith(".pt"):
                    path = os.path.join(folder, file)
                    meta = load_checkpoint_metadata(path)
                    
                    if meta["error"] and meta["step"] == "Corrupted":
                        display_name = f"❌ {phase_name} — {file} (CORRUPTED)"
                    else:
                        step = meta.get("step", "N/A")
                        val_loss = meta.get("val_loss", None)
                        loss_str = f" | Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                        
                        if "best" in file:
                            display_name = f"🏆 {phase_name} — Best (Step: {step}{loss_str})"
                        elif "initial" in file:
                            display_name = f"🎬 {phase_name} — Initial"
                        else:
                            display_name = f"⚙️ {phase_name} — Step {step}{loss_str}"
                            
                    checkpoint_options[display_name] = (path, meta)
                    
    return checkpoint_options

discovered = discover_checkpoints()

# Initialize Session State
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = "What is artificial intelligence?"
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""
if "gen_stats" not in st.session_state:
    st.session_state.gen_stats = None

# Callback helpers
def select_prompt(selected_prompt):
    st.session_state.prompt_input = selected_prompt

def clear_workspace():
    st.session_state.prompt_input = ""
    st.session_state.generated_text = ""
    st.session_state.gen_stats = None

# Header/Hero section
st.markdown('<div class="hero-title">COLLISION-1.46M</div>', unsafe_allow_html=True)
st.write("An experimental language model trained from scratch on CPU.")

st.markdown(
    """
    <div class="status-badge">
        <span class="status-dot"></span>ONLINE &nbsp;|&nbsp; CPU INFERENCE
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-stats-container">
        <div class="hero-stat-card">
            <div class="hero-stat-value">1,462,464</div>
            <div class="hero-stat-label">Parameters</div>
        </div>
        <div class="hero-stat-card">
            <div class="hero-stat-value">6.93</div>
            <div class="hero-stat-label">Best Perplexity</div>
        </div>
        <div class="hero-stat-card">
            <div class="hero-stat-value">CPU TRAINED</div>
            <div class="hero-stat-label">Hardware</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Main layout grid
col_left, col_right = st.columns([7, 5])

# Left column: Workspace & Output
with col_left:
    st.subheader("Generation Workspace")
    
    # Checkpoint selector (Compact selector)
    if not discovered:
        st.error("No checkpoints found under `checkpoints/` directory.")
        selected_cp_path = None
        cp_meta = {}
    else:
        # Default to Phase 6 Best Checkpoint
        default_index = 0
        cp_names = list(discovered.keys())
        for idx, name in enumerate(cp_names):
            if "Phase 6 — Best" in name:
                default_index = idx
                break
                
        selected_cp_name = st.selectbox("Model Checkpoint Selection", cp_names, index=default_index)
        selected_cp_path, cp_meta = discovered[selected_cp_name]

    # Compact settings panel under expander
    with st.expander("Generation Parameters", expanded=False):
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
            top_k = st.slider("Top-K", min_value=1, max_value=100, value=50, step=1)
        with p_col2:
            top_p = st.slider("Top-P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
            max_tokens = st.slider("Max New Tokens", min_value=1, max_value=256, value=100, step=1)

        # Validation checks
        if temp < 0.0 or temp > 2.0:
            st.error("Temperature must be between 0.0 and 2.0")
        if top_k < 1 or top_k > 100:
            st.error("Top-K must be between 1 and 100")
        if top_p < 0.0 or top_p > 1.0:
            st.error("Top-P must be between 0.0 and 1.0")
        if max_tokens < 1 or max_tokens > 256:
            st.error("Max tokens must be between 1 and 256")

    # Example Prompts
    st.write("💡 Example prompts:")
    example_prompts = [
        "What is artificial intelligence?",
        "Explain how an algorithm works.",
        "What is the future of technology?",
        "Why does the Earth orbit the Sun?",
        "How does machine learning work?"
    ]
    
    # Dynamic buttons
    p_cols = st.columns(len(example_prompts))
    for idx, prompt_text in enumerate(example_prompts):
        with p_cols[idx]:
            # Simple pill layout using session state callback
            st.button(prompt_text[:18] + "...", key=f"pill_{idx}", on_click=select_prompt, args=(prompt_text,))

    # Prompt editor input
    prompt = st.text_area("Prompt Editor", value=st.session_state.prompt_input, height=120, placeholder="Ask COLLISION something...", label_visibility="collapsed")
    st.session_state.prompt_input = prompt

    # Controls Row
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 1, 1, 1])
    
    trigger_generate = False
    
    with btn_col1:
        if st.button("Generate Output", use_container_width=True):
            trigger_generate = True
    with btn_col2:
        if st.button("Regenerate", key="action_regen", use_container_width=True):
            if st.session_state.prompt_input:
                trigger_generate = True
    with btn_col3:
        st.button("Clear", key="action_clear", on_click=clear_workspace, use_container_width=True)
    with btn_col4:
        # Streamlit standard code block handles client-side copying. 
        # We will show the generated text in st.code to allow copying, but we can also display a note.
        pass

    # Process Generation
    if trigger_generate:
        if not prompt.strip():
            st.warning("Please enter a non-empty prompt.")
        elif not selected_cp_path:
            st.error("No checkpoint loaded. Cannot generate.")
        elif cp_meta.get("error", False) and cp_meta.get("step") == "Corrupted":
            st.error("Selected checkpoint is corrupted.")
        else:
            with st.spinner("Executing CPU inference..."):
                try:
                    # Load model & config safely
                    checkpoint = torch.load(selected_cp_path, map_location="cpu")
                    model_cfg = ModelConfig(**checkpoint["config"])
                    model = CollisionTransformer(model_cfg)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    
                    # Load tokenizer safely
                    if not os.path.exists(TOKENIZER_DIR):
                        raise FileNotFoundError(f"Tokenizer directory not found at: {TOKENIZER_DIR}")
                    tokenizer = BPETokenizer()
                    tokenizer.load(TOKENIZER_DIR)
                    
                    # Autoregressive generation
                    model.eval()
                    ids = tokenizer.encode(prompt, bos=True)
                    prompt_len = len(ids)
                    
                    if prompt_len > 256:
                        raise ValueError(f"Prompt length of {prompt_len} tokens exceeds context length of 256 tokens.")
                        
                    x = torch.tensor([ids], dtype=torch.long, device="cpu")
                    tokens_generated = 0
                    
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        for _ in range(max_tokens):
                            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
                            logits, _ = model(x_cond)
                            next_token_logits = logits[0, -1, :]
                            
                            if temp > 0.0:
                                next_token_logits = next_token_logits / temp
                                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                                probs = F.softmax(filtered_logits, dim=-1)
                                next_token = torch.multinomial(probs, num_samples=1)
                            else:
                                next_token = torch.argmax(next_token_logits).unsqueeze(0)
                                
                            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
                            tokens_generated += 1
                            
                            if next_token.item() == tokenizer.special_tokens["[EOS]"]:
                                break
                                
                    end_time = time.perf_counter()
                    
                    gen_time = end_time - start_time
                    tok_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
                    decoded_text = tokenizer.decode(x[0].tolist())
                    
                    # Update session state
                    st.session_state.generated_text = decoded_text
                    st.session_state.gen_stats = {
                        "tokens_generated": tokens_generated,
                        "generation_time": gen_time,
                        "tokens_per_second": tok_per_sec
                    }
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")

    # Output Card
    st.write("")
    st.write("### COLLISION OUTPUT")
    
    if st.session_state.generated_text:
        # Use st.code to provide an elegant, copyable display box
        st.code(st.session_state.generated_text, language=None)
        
        # Display performance stats underneath the text
        if st.session_state.gen_stats:
            stats = st.session_state.gen_stats
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Tokens Generated", f"{stats['tokens_generated']}")
            s_col2.metric("Generation Time", f"{stats['generation_time']:.4f}s")
            s_col3.metric("Tokens/Second", f"{stats['tokens_per_second']:.2f}")
    else:
        st.markdown(
            """
            <div style="background-color: #09090b; border: 1px dashed #27272a; padding: 2rem; border-radius: 4px; text-align: center; color: #71717a; font-size: 13px;">
                Enter a prompt and click "Generate Output" to view model results.
            </div>
            """,
            unsafe_allow_html=True
        )

# Right column: Context, Story, specs
with col_right:
    # Why COLLISION?
    st.subheader("Why COLLISION?")
    st.markdown(
        """
        <div class="panel-card" style="font-size: 13px; line-height: 1.5; color: #a1a1aa; margin-bottom: 1.25rem;">
            <strong>COLLISION</strong> is an experiment in understanding what happens when a small Transformer is trained completely from scratch on limited hardware.
            <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; color: #e4e4e7; font-weight: 500;">
                <div>• No pretrained weights</div>
                <div>• No external LLM API</div>
                <div>• CPU-only training</div>
                <div>• Custom tokenizer</div>
                <div>• Reproducible configs</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Experiment Result comparison
    st.subheader("Experiment Result")
    st.markdown(
        """
        <div class="comparison-box">
            <div class="comparison-item">
                <div class="comparison-title">PHASE 5</div>
                <div class="comparison-value">62.86 PPX</div>
            </div>
            <div class="comparison-arrow">↓</div>
            <div class="comparison-item">
                <div class="comparison-title">PHASE 6</div>
                <div class="comparison-value" style="color: #2563eb;">6.93 PPX</div>
            </div>
        </div>
        <div style="font-size: 11px; color: #71717a; line-height: 1.4; margin-bottom: 1.5rem; text-align: center;">
            "Improvement came from dataset auditing and better train/validation separation—not from increasing model size."
        </div>
        """,
        unsafe_allow_html=True
    )

    # Experiment Timeline
    st.subheader("Experiment Story")
    st.markdown(
        """
        <div class="timeline-container">
            <div class="timeline-node">
                <div class="timeline-node-title">PHASE 1</div>
                <div class="timeline-node-desc">Prototype</div>
            </div>
            <div class="timeline-node">
                <div class="timeline-node-title">PHASE 2</div>
                <div class="timeline-node-desc">Training Framework</div>
            </div>
            <div class="timeline-node">
                <div class="timeline-node-title">PHASE 3</div>
                <div class="timeline-node-desc">Dataset Pipeline</div>
            </div>
            <div class="timeline-node">
                <div class="timeline-node-title">PHASE 4</div>
                <div class="timeline-node-desc">Readiness</div>
            </div>
            <div class="timeline-node">
                <div class="timeline-node-title">PHASE 5</div>
                <div class="timeline-node-desc">First Training</div>
            </div>
            <div class="timeline-node">
                <div class="timeline-node-title">PHASE 6</div>
                <div class="timeline-node-desc">Generalization</div>
            </div>
            <div class="timeline-node active">
                <div class="timeline-node-title" style="color: #2563eb;">PHASE 7</div>
                <div class="timeline-node-desc" style="color: #ffffff;">Public Playground</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Model Details collapsible
    with st.expander("Model Specifications & Architecture", expanded=False):
        st.markdown(
            """
            <div class="spec-item">
                <span class="spec-label">Architecture</span>
                <span class="spec-value">Decoder-only Transformer</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Parameters</span>
                <span class="spec-value">1,462,464</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Layers</span>
                <span class="spec-value">3</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Embedding</span>
                <span class="spec-value">128</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Attention heads</span>
                <span class="spec-value">4</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Context length</span>
                <span class="spec-value">256</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Tokenizer</span>
                <span class="spec-value">Custom BPE</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Dataset</span>
                <span class="spec-value">collision_dataset_v4</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Training device</span>
                <span class="spec-value">CPU</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Disclaimer
    st.markdown(
        """
        <div class="subtle-disclaimer">
            <strong>Disclaimer:</strong> COLLISION-1.46M is an experimental small language model. Outputs may be incomplete, repetitive, incorrect, or nonsensical.
        </div>
        """,
        unsafe_allow_html=True
    )
