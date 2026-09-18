import streamlit as st
import os
import sys
import time
import json
from datetime import datetime

# Insert project root into sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from playground.api_client import CollisionAPIClient

# Page configuration
st.set_page_config(
    page_title="COLLISION Developer Portal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling CSS
st.markdown("""
<style>
    /* Premium Minimal Dark/Light Geometric brand styling */
    :root {
        --primary: #8a63d2;
        --bg-card: #faf9fd;
        --border-color: #efeafb;
    }
    .stApp {
        background-color: #ffffff;
    }
    .portal-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a1a;
        margin-bottom: 0px;
        letter-spacing: -0.05rem;
    }
    .portal-subheader {
        font-family: 'Inter', sans-serif;
        font-size: 1.0rem;
        color: #666666;
        margin-top: 0px;
        margin-bottom: 25px;
    }
    .stat-card {
        background-color: #faf9fd;
        border: 1px solid #efeafb;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(138, 99, 210, 0.02);
    }
    .stat-val {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #8a63d2;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #555555;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        margin-bottom: 5px;
    }
    .output-box {
        background-color: #faf9fd;
        border: 1px solid #efeafb;
        border-radius: 8px;
        padding: 20px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.15rem;
        color: #1a1a1a;
        white-space: pre-wrap;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "session_token" not in st.session_state:
    st.session_state.session_token = None
if "developer_id" not in st.session_state:
    st.session_state.developer_id = None
if "email" not in st.session_state:
    st.session_state.email = None
if "new_api_key" not in st.session_state:
    st.session_state.new_api_key = None
if "playground_output" not in st.session_state:
    st.session_state.playground_output = ""
if "playground_metrics" not in st.session_state:
    st.session_state.playground_metrics = None
if "playground_request" not in st.session_state:
    st.session_state.playground_request = None
if "playground_response" not in st.session_state:
    st.session_state.playground_response = None
if "playground_history" not in st.session_state:
    st.session_state.playground_history = []

# Sidebar API Endpoint Configuration
st.sidebar.title("API Configuration")
api_url_input = st.sidebar.text_input(
    "COLLISION API Base URL",
    value=os.environ.get("COLLISION_API_URL", "http://localhost:8000"),
    help="FastAPI server URL."
)

client = CollisionAPIClient(
    base_url=api_url_input,
    session_token=st.session_state.session_token
)

# Connection Health Check
health_info = client.get_health()
status_indicator = "●"
status_text = "Disconnected"
status_color = "#dc3545"
device = "N/A"
model_name = "N/A"

if health_info["status"] == "ok":
    status_text = "Connected / Operational"
    status_color = "#28a745"
    device = health_info["data"].get("device", "cpu")
    model_name = health_info["data"].get("model", "collision-10m")

st.sidebar.markdown(f"""
<div style="background-color: #f8f9fa; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef; margin-bottom: 20px;">
    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.9rem;">
        <span style="font-weight: 600; color: #333333;">Service Status</span>
        <span style="color: {status_color}; font-weight: bold; display: flex; align-items: center; gap: 4px;">
            <span>{status_indicator}</span> <span>{status_text}</span>
        </span>
    </div>
    <hr style="margin: 8px 0; border: none; border-top: 1px solid #e9ecef;">
    <div style="font-size: 0.85rem; color: #666666; line-height: 1.4;">
        <b>Model:</b> {model_name}<br>
        <b>Execution:</b> local CPU ({device})
    </div>
</div>
""", unsafe_allow_html=True)

def get_auth_error_message(resp, default_msg):
    if isinstance(resp, dict):
        if "detail" in resp:
            detail = resp["detail"]
            if isinstance(detail, dict):
                return detail.get("message", str(detail))
            elif isinstance(detail, list) and len(detail) > 0:
                first = detail[0]
                if isinstance(first, dict) and "msg" in first:
                    return first["msg"]
                return str(detail)
            return str(detail)
        elif "error" in resp:
            err = resp["error"]
            if isinstance(err, dict):
                return err.get("message", str(err))
            return str(err)
    return default_msg

# ----------------- LOGIN / SIGNUP SCREEN -----------------
if not st.session_state.session_token:
    st.markdown('<div class="portal-header">COLLISION Developer Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="portal-subheader">Lightweight language model API developer platform</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Access Portal")
        login_email = st.text_input("Email Address", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Log In", type="primary", use_container_width=True):
            if not login_email or not login_password:
                st.error("Please fill in email and password.")
            else:
                code, resp = client.login(login_email.strip(), login_password)
                if code == 200:
                    st.session_state.session_token = resp["session_token"]
                    st.session_state.developer_id = resp["developer_id"]
                    st.session_state.email = resp["email"]
                    st.success("Successfully logged in!")
                    st.rerun()
                else:
                    err_msg = get_auth_error_message(resp, "Invalid email or password.")
                    st.error(f"Login failed: {err_msg}")
                    
    with col_right:
        st.markdown("### Create Developer Account")
        signup_email = st.text_input("Email Address", key="signup_email")
        signup_password = st.text_input("Password (min 8 chars)", type="password", key="signup_password")
        
        if st.button("Register Account", use_container_width=True):
            if not signup_email or not signup_password:
                st.error("Please fill in email and password.")
            elif len(signup_password) < 8:
                st.error("Password must be at least 8 characters long.")
            else:
                code, resp = client.signup(signup_email.strip(), signup_password)
                if code == 200 or code == 201:
                    # Auto-login after successful registration
                    login_code, login_resp = client.login(signup_email.strip(), signup_password)
                    if login_code == 200:
                        st.session_state.session_token = login_resp["session_token"]
                        st.session_state.developer_id = login_resp["developer_id"]
                        st.session_state.email = login_resp["email"]
                        st.success("Registration successful! Logging you in...")
                        st.rerun()
                    else:
                        st.success("Registration successful! Please log in on the left.")
                else:
                    err_msg = get_auth_error_message(resp, "Registration failed.")
                    st.error(f"Registration failed: {err_msg}")
    st.stop()

# ----------------- LOGGED IN PORTAL SCREEN -----------------

# Set active session on client object
client.session_token = st.session_state.session_token

# User Profile Header
st.markdown('<div class="portal-header">COLLISION Developer Portal</div>', unsafe_allow_html=True)
st.markdown(f'<div class="portal-subheader">Logged in as: <b>{st.session_state.email}</b> (Developer ID: {st.session_state.developer_id})</div>', unsafe_allow_html=True)

# Add logout button in sidebar
if st.sidebar.button("Log Out Developer Profile", use_container_width=True):
    client.logout()
    st.session_state.session_token = None
    st.session_state.developer_id = None
    st.session_state.email = None
    st.session_state.new_api_key = None
    st.session_state.playground_output = ""
    st.session_state.playground_metrics = None
    st.session_state.playground_request = None
    st.session_state.playground_response = None
    st.session_state.playground_history = []
    st.rerun()

# Fetch developer stats and key lists securely (IDOR protected in API)
status_code, stats = client.get_usage_stats(st.session_state.developer_id)
if status_code != 200:
    st.error("Session verification failed. Log out and try logging in again.")
    st.stop()

# Navigation tabs
tab_overview, tab_keys, tab_usage, tab_models, tab_playground, tab_brain, tab_docs = st.tabs([
    "Overview", "API Keys", "Usage Analytics", "Model Info", "Playground Client", "🧠 Brain Studio", "Documentation"
])

# 1. OVERVIEW TAB
with tab_overview:
    st.markdown("### Platform Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Model Status</div>
            <div class="stat-val" style="color: {status_color}; font-size: 1.4rem;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Requests</div>
            <div class="stat-val">{stats['total_requests']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Tokens</div>
            <div class="stat-val">{stats['total_tokens']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Avg Query Latency</div>
            <div class="stat-val">{stats['avg_latency_ms']:.0f} ms</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Check if they have keys to determine onboarding status
    code, keys_list = client.list_api_keys(st.session_state.developer_id)
    has_active_keys = any(k["status"] == "active" for k in keys_list) if code == 200 else False
    
    if not has_active_keys:
        st.info("💡 **Getting Started**: To make your first request, navigate to the **API Keys** tab and generate your authentication credentials.")
    else:
        st.markdown("### 🚀 Quickstart Integration Guide")
        st.caption("Copy this code snippet to execute completions requests directly against the REST API.")
        
        tab_curl, tab_py, tab_js = st.tabs(["cURL Command", "Python SDK", "Node.js Request"])
        
        with tab_curl:
            st.code(f"""curl -X POST {api_url_input}/v1/generate \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{{
    "model": "collision-10m",
    "prompt": "Artificial intelligence is",
    "max_tokens": 50
  }}'""", language="bash")
            
        with tab_py:
            st.code(f"""import requests

url = "{api_url_input}/v1/generate"
headers = {{
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}}
payload = {{
    "model": "collision-10m",
    "prompt": "Artificial intelligence is",
    "max_tokens": 50
}}

response = requests.post(url, json=payload, headers=headers)
print(response.json()["text"])""", language="python")
            
        with tab_js:
            st.code(f"""fetch("{api_url_input}/v1/generate", {{
  method: "POST",
  headers: {{
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
  }},
  body: JSON.stringify({{
    model: "collision-10m",
    prompt: "Artificial intelligence is",
    max_tokens: 50
  }})
}})
.then(res => res.json())
.then(data => console.log(data.text));""", language="javascript")

    st.markdown("---")
    st.markdown("### Flagship Model Details")
    col_details_l, col_details_r = st.columns(2)
    with col_details_l:
        st.markdown(f"""
        - **Model Name**: `collision-1.0b` (Official Flagship)
        - **Parameter Count**: `999,376,128` (~1.00B)
        - **Context limit**: `1,024 tokens`
        - **Architecture**: Decoder-only custom Transformer (24 layers, 16 heads, 2048 embedding dim)
        - **Edge Variant**: `collision-10m` (10,282,304 parameters, 256 context)
        """)
    with col_details_r:
        st.markdown("""
        - **Intended Use**: Causal text completion, reasoning synthesis, web grounding, industrial NLP tasks.
        - **Deployment**: CPU native and Multi-GPU scalable architecture.
        """)

# 2. API KEYS TAB
with tab_keys:
    st.markdown("### API Key Management")
    st.caption("API keys authorize requests to `/v1/generate`. Authorize using standard Bearer authorization headers.")
    
    col_btn, col_blank = st.columns([1, 3])
    with col_btn:
        if st.button("Generate API Key", type="primary", use_container_width=True):
            code, resp = client.generate_api_key(st.session_state.developer_id)
            if code == 200:
                st.session_state.new_api_key = resp["api_key"]
                st.rerun()
            else:
                st.error("Failed to generate key.")
                
    if st.session_state.new_api_key:
        st.warning("""
        ⚠️ **Store this key securely. You will not be able to view it again.**
        
        Copy it immediately to execute requests.
        """)
        st.code(st.session_state.new_api_key, language="text")
        if st.button("I have copied the key"):
            st.session_state.new_api_key = None
            st.rerun()
            
    st.markdown("---")
    st.markdown("#### Stored Keys")
    
    code, keys_list = client.list_api_keys(st.session_state.developer_id)
    if code == 200:
        if keys_list:
            for key in keys_list:
                col_key_prefix, col_key_created, col_key_status, col_key_action = st.columns([3, 2, 1, 1])
                with col_key_prefix:
                    st.markdown(f"Prefix: `Bearer {key['prefix']}••••••••`")
                with col_key_created:
                    st.caption(f"Created: {key['created_at'][:19]}")
                    if key['last_used_at']:
                        st.caption(f"Last Used: {key['last_used_at'][:19]}")
                with col_key_status:
                    if key["status"] == "active":
                        st.success("Active")
                    else:
                        st.error("Revoked")
                with col_key_action:
                    if key["status"] == "active":
                        if st.button("Revoke", key=f"rev_portal_{key['id']}", use_container_width=True):
                            client.revoke_api_key(key["id"])
                            st.success("Key revoked.")
                            st.rerun()
                    else:
                        st.button("Revoked", key=f"rev_portal_{key['id']}", disabled=True, use_container_width=True)
        else:
            st.caption("No API keys generated. Click the button above to create one.")
    else:
        st.error("Failed to retrieve key listings.")

# 3. USAGE ANALYTICS TAB
with tab_usage:
    st.markdown("### Usage Analytics")
    
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        st.metric("Total Requests Executed", f"{stats['total_requests']:,}")
    with col_u2:
        st.metric("Total Completed Tokens", f"{stats['total_completion_tokens']:,}")
    with col_u3:
        st.metric("Average Latency per Query", f"{stats['avg_latency_ms']:.1f} ms")
        
    st.markdown("---")
    st.markdown("#### Historic Token Utilization Stats")
    st.caption("Note: Storage records are hosted in your production PostgreSQL database environment.")
    
    # Render table of values
    st.json({
        "requests": stats["total_requests"],
        "prompt_tokens": stats["total_prompt_tokens"],
        "completion_tokens": stats["total_completion_tokens"],
        "total_tokens": stats["total_tokens"],
        "average_latency_ms": stats["avg_latency_ms"]
    })

# 4. MODEL INFO TAB
with tab_models:
    st.markdown("### COLLISION-10M Specifications")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        #### Network Properties
        - **Embedding Dim (d_model)**: `384`
        - **Attention Heads**: `8`
        - **Layers (n_layer)**: `6`
        - **Feedforward Dim (d_ff)**: `768`
        - **Context limit size**: `256 tokens`
        - **Initialization state**: Random initialization (from scratch)
        """)
    with col_m2:
        st.markdown("""
        #### Empirical Benchmark Metrics
        - **Validation Loss**: `0.7454`
        - **Validation Perplexity (PPL)**: `2.11`
        - **Test Split Loss**: `0.5805`
        - **Test Split Perplexity (PPL)**: `1.79`
        - **CPU Generation Throughput**: `42.38 tokens/second` (API average benchmark)
        - **Average Generation latency**: `2317.6 ms` (for ~97 tokens completed)
        """)
        
    st.success("""
    **Hierarchical Hybrid Intelligence Architecture**:
    COLLISION-10M is integrated with an intelligent multi-tier pipeline:
    1. **Conversational Intent Engine**: Instant natural handling for greetings, courtesies, and identity.
    2. **Safe AST Math Evaluator**: Deterministic computation for arithmetic, percentages, and formulas.
    3. **Multi-Source Hybrid RAG**: Real-time Wikipedia OpenSearch + DuckDuckGo knowledge extraction.
    4. **Local Vector Knowledge**: Embedded cosine similarity search across local architecture specs.
    """)

# 5. PLAYGROUND CLIENT TAB
with tab_playground:
    st.markdown("### Hierarchical Hybrid Intelligence Playground")
    st.caption("Ask anything: conversations, exact math, or open-domain factual questions. Powered by COLLISION-10M + Multi-Tier Grounding.")
    
    # Allow entering an API key to test generation
    playground_key = st.text_input("API Key for Playground Queries (col_...)", type="password", help="Requires an active API key to call completions.")
    
    col_play_l, col_play_r = st.columns([5, 3])
    
    with col_play_r:
        st.markdown("#### Generation Configs")
        p_web_search = st.selectbox("Knowledge & Web Mode", ["AUTO", "ON", "OFF"], index=0, help="AUTO intelligently answers greetings, math, and retrieves verified Wikipedia/Web facts for open-domain queries.")
        p_temp = st.slider("Temperature (Playground)", min_value=0.01, max_value=2.0, value=0.7, step=0.05)
        p_top_p = st.slider("Top P (Playground)", min_value=0.01, max_value=1.0, value=0.9, step=0.05)
        p_top_k = st.number_input("Top K (Playground)", min_value=0, max_value=200, value=50)
        p_max_tokens = st.number_input("Max Tokens (Playground)", min_value=1, max_value=256, value=100)

    with col_play_l:
        st.markdown("#### Domain Exploration Examples")
        sample_prompts = {
            "-- Select a domain example --": "",
            "Conversation & Identity": "hi, who are you and what can you do?",
            "Exact Math Calculation": "what is 45 * 128 + 15% of 850?",
            "Entity Fact": "who was Albert Einstein?",
            "Science & Technology": "what is quantum computing?",
            "Geography": "what is the capital of France?",
            "General Knowledge": "Explain how solar eclipses occur.",
            "Programming": "Write a Python function to check for palindromes.",
            "AI/ML": "Explain the difference between supervised and unsupervised learning.",
            "Reasoning": "If all A are B and all B are C, are all A necessarily C?",
            "Writing": "Draft a concise product announcement for a new API."
        }
        selected_domain = st.selectbox("Explore Sample Prompts", list(sample_prompts.keys()), key="domain_sample_select")
        initial_prompt_val = sample_prompts[selected_domain] if selected_domain != "-- Select a domain example --" else ""
        
        p_prompt = st.text_area("Prompt (Playground)", value=initial_prompt_val if initial_prompt_val else "", placeholder="Ask COLLISION anything...", height=120)

        
        col_pbtn_gen, col_pbtn_clear = st.columns(2)
        
        with col_pbtn_clear:
            if st.button("Clear Playground Output"):
                st.session_state.playground_output = ""
                st.session_state.playground_metrics = None
                st.session_state.playground_request = None
                st.session_state.playground_response = None
                st.rerun()
                
        with col_pbtn_gen:
            if st.button("Generate Completion", key="play_gen_btn", type="primary"):
                if not playground_key.strip():
                    st.error("Authentication required. Please enter an active API key generated from the API Keys tab.")
                elif not p_prompt.strip():
                    st.error("Prompt must not be empty.")
                else:
                    # Update client key
                    client.api_key = playground_key.strip()
                    with st.spinner("COLLISION-10M is executing completion..."):
                        response = client.generate(
                            prompt=p_prompt,
                            model="collision-10m",
                            max_tokens=p_max_tokens,
                            temp=p_temp,
                            top_k=p_top_k,
                            top_p=p_top_p,
                            web_search=p_web_search.lower()
                        )
                        
                        if response["success"]:
                            data = response["data"]
                            st.session_state.playground_output = data["text"]
                            st.session_state.playground_metrics = data["performance"]
                            st.session_state.playground_metrics["usage"] = data["usage"]
                            st.session_state.playground_request = response["request_json"]
                            st.session_state.playground_response = data
                            
                            # Add to playground history
                            st.session_state.playground_history.append({
                                "prompt": p_prompt,
                                "tokens": data["usage"]["completion_tokens"],
                                "latency_ms": data["performance"]["latency_ms"],
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                        else:
                            st.error(f"Failed to generate completion: {response['error']}")
                            st.session_state.playground_request = response["request_json"]
                            st.session_state.playground_response = {"error": response["error"]}

        # Display output
        if st.session_state.playground_output:
            st.markdown("#### Output Completion")
            st.markdown(f"""
            <div class="output-box">
            <span style="color: #6c757d; font-style: italic;">{p_prompt}</span><b>{st.session_state.playground_output}</b>
            </div>
            """, unsafe_allow_html=True)

            resp_data = st.session_state.playground_response
            if resp_data and isinstance(resp_data, dict) and resp_data.get("web_search_used"):
                st.markdown("##### 🌐 Web Sources")
                sources = resp_data.get("sources", [])
                if sources:
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"**{idx}. [{src['title']}]({src['url']})**")
                        if src.get("snippet"):
                            st.caption(f"\"{src['snippet']}\"")
                else:
                    st.caption("Web search was executed but no external page sources were retrieved.")
            
            metrics = st.session_state.playground_metrics
            if metrics:
                st.markdown(f"""
                <div style="display: flex; gap: 15px; align-items: center; margin-top: 10px; font-size: 0.85rem; background-color: #faf9fd; border: 1px solid #efeafb; border-radius: 6px; padding: 10px;">
                    <div>⚡ <b>Speed:</b> {metrics['tokens_per_second']:.1f} tok/s</div>
                    <div>|</div>
                    <div>⏱️ <b>Latency:</b> {metrics['latency_ms']/1000:.2f} s</div>
                    <div>|</div>
                    <div>🏷️ <b>Completion Tokens:</b> {metrics['usage']['completion_tokens']}</div>
                    <div>|</div>
                    <div>📄 <b>Prompt Tokens:</b> {metrics['usage']['prompt_tokens']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            st.markdown("##### 💬 Public Beta Feedback")
            col_fb1, col_fb2 = st.columns([1, 2])
            with col_fb1:
                rating_choice = st.radio("Was this completion helpful?", ["👍 Helpful", "👎 Not helpful"], key="fb_rating_choice")
                fb_category = st.selectbox(
                    "Category / Domain",
                    [
                        "general", "Programming", "AI/ML", "Science", "Mathematics",
                        "Reasoning", "Writing", "Summarization", "Troubleshooting",
                        "Conversation", "Instructions", "explanatory", "how-to", "planning"
                    ],
                    key="fb_cat_choice"
                )
            with col_fb2:
                fb_comment = st.text_input("Optional comment / details", placeholder="Add feedback comments...", key="fb_comment_choice")
                fb_consent = st.checkbox("Allow this interaction to be used to improve COLLISION", value=True, key="fb_consent_choice")
                fb_model_tag = st.caption("Model Version: `J52` (Collision-10M baseline)")


            if st.button("Submit Feedback", key="fb_submit_btn", type="secondary"):
                rating_val = "thumbs_up" if "Helpful" in rating_choice and "Not" not in rating_choice else "thumbs_down"
                code, resp = client.submit_feedback(
                    prompt=p_prompt,
                    response=st.session_state.playground_output,
                    rating=rating_val,
                    category=fb_category,
                    feedback=fb_comment,
                    consent=fb_consent,
                    model="J52"
                )
                if code == 200:
                    st.success("Thank you! Feedback recorded successfully.")
                else:
                    st.error(f"Failed to submit feedback: {resp}")

                
    # Introspection
    if st.session_state.playground_output or p_prompt.strip():
        st.markdown("---")
        st.markdown("#### Introspection Console")
        
        esc_prompt = p_prompt.replace('"', '\\"')
        curl_cmd = f"""curl -X POST {api_url_input}/v1/generate \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <API_KEY>" \\
  -d '{{
    "model": "collision-10m",
    "prompt": "{esc_prompt}",
    "max_tokens": {p_max_tokens},
    "temperature": {p_temp},
    "top_k": {p_top_k},
    "top_p": {p_top_p}
  }}'"""
        st.code(curl_cmd, language="bash")
        
        col_jreq, col_jres = st.columns(2)
        with col_jreq:
            st.markdown("##### Request JSON")
            if st.session_state.playground_request:
                st.json(st.session_state.playground_request)
        with col_jres:
            st.markdown("##### Response JSON")
            if st.session_state.playground_response:
                st.json(st.session_state.playground_response)

# 6. BRAIN STUDIO TAB
with tab_brain:
    st.markdown("### 🧠 COLLISION Synaptic Cognitive Brain Studio")
    st.caption("Inspect non-linear Graph-of-Thoughts (GoT 2.0), Hegelian Dialectic convergence, Hebbian working memory, and multi-scale information entropy.")

    from collision.brain import get_collision_brain
    brain_instance = get_collision_brain()

    sample_brain_inquiries = {
        "-- Select a multi-disciplinary inquiry --": "",
        "Distributed Systems & CAP": "Should high-throughput distributed databases prioritize strong consistency or partition availability?",
        "AI Epistemology & Chinese Room": "Does the Chinese Room argument by John Searle disprove semantic understanding in deep neural networks?",
        "Quantum & Information Physics": "Compare Heisenberg Uncertainty Principle with Shannon Information Entropy and Landauer's computational limit.",
        "Game Theory & Optimization": "How do Nash Equilibrium and Pareto Optimality diverge in non-cooperative multi-agent mechanism design?",
        "Cognitive Science & Dual-Process": "What are the cognitive trade-offs between System 1 reflexive intuition and System 2 deliberative synthesis?",
        "Byzantine Faults & Consensus": "Explain why Byzantine Fault Tolerance requires 3f + 1 nodes while crash fault tolerance requires only 2f + 1."
    }

    col_b_top_l, col_b_top_r = st.columns([5, 3])
    with col_b_top_r:
        st.markdown("#### Brain Parameters")
        b_domain = st.selectbox(
            "Primary Knowledge Domain",
            ["General", "Distributed Systems", "Artificial Intelligence", "Quantum Physics", "Economics & Game Theory", "Cognitive Science & Philosophy"]
        )
        b_dialectic = st.checkbox("Enable Hegelian Dialectic (Thesis + Antithesis + Synthesis)", value=True)
        b_probe = st.text_input("Optional Counterfactual Probe", placeholder="e.g. If network latency is zero...")

    with col_b_top_l:
        st.markdown("#### Deliberative Inquiry")
        selected_brain_sample = st.selectbox("Explore Deep Cognitive Inquiries", list(sample_brain_inquiries.keys()), key="brain_sample_select")
        init_brain_q = sample_brain_inquiries[selected_brain_sample] if selected_brain_sample != "-- Select a multi-disciplinary inquiry --" else ""
        b_query = st.text_area("Cognitive Query for Brain Deliberation", value=init_brain_q if init_brain_q else "", placeholder="Enter a complex philosophical, scientific, or systems question...", height=110)

        if st.button("Execute Cognitive Deliberation ⚡", key="brain_think_btn", type="primary"):
            if not b_query.strip():
                st.error("Please provide an inquiry for the brain to deliberate.")
            else:
                with st.spinner("Collision Brain is constructing Graph-of-Thoughts & Hegelian synthesis..."):
                    t_b0 = time.perf_counter()
                    b_res = brain_instance.think(
                        query=b_query.strip(),
                        domain=b_domain,
                        enable_dialectic=b_dialectic,
                        counterfactual=b_probe.strip() if b_probe.strip() else None
                    )
                    st.session_state["last_brain_response"] = b_res

    # Render Brain Results if available
    if "last_brain_response" in st.session_state and st.session_state["last_brain_response"]:
        b_res = st.session_state["last_brain_response"]
        st.markdown("---")

        # Top Metric Cards
        b_c1, b_c2, b_c3, b_c4, b_c5, b_c6 = st.columns(6)
        with b_c1:
            st.metric("Modality", b_res.modality.value.replace("DIALECTICAL_", ""))
        with b_c2:
            st.metric("Confidence", f"{b_res.confidence*100:.1f}%")
        with b_c3:
            st.metric("Certainty", f"{b_res.epistemic_certainty*100:.1f}%")
        with b_c4:
            st.metric("Resilience", f"{b_res.adversarial_resilience*100:.1f}%")
        with b_c5:
            st.metric("Merit Score", f"{b_res.merit_score*100:.1f}%")
        with b_c6:
            st.metric("Total Latency", f"{b_res.trace.total_brain_latency_ms:.1f} ms" if b_res.trace else "N/A")

        # Sub-tabs for deep inspection
        st_delib, st_graph, st_entropy, st_critic, st_memory, st_triples = st.tabs([
            "📖 Dialectical Synthesis",
            "🕸️ Graph of Thoughts (GoT)",
            "📊 Information Entropy",
            "🛡️ Metacognitive Critic",
            "💾 Synaptic Working Memory",
            "🧩 Semantic Triples"
        ])

        with st_delib:
            st.markdown("#### Dialectical Synthesis & Reconciled Truth")
            st.markdown(f"""
            <div class="output-box" style="font-family: 'Inter', sans-serif; font-size: 1.05rem;">
            {b_res.answer}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 💡 Key Cognitive Insights")
            for ins in b_res.key_insights:
                st.markdown(f"- {ins}")

            st.markdown("##### 🔮 Follow-up Hypotheses")
            for h in b_res.followup_hypotheses:
                st.markdown(f"- *{h}*")

        with st_graph:
            st.markdown("#### Interactive Reasoning DAG")
            if b_res.graph_mermaid:
                st.markdown(f"""
```mermaid
{b_res.graph_mermaid}
```
""")
            if b_res.graph_ascii:
                with st.expander("View Terminal ASCII Tree"):
                    st.code(b_res.graph_ascii, language="text")

            if b_res.trace and b_res.trace.graph_of_thoughts:
                st.markdown("##### Thought Nodes Breakdown")
                for nid, node in b_res.trace.graph_of_thoughts.nodes.items():
                    with st.expander(f"[{node.thought_type.value}] {nid} (Confidence: {node.confidence*100:.0f}%)"):
                        st.write(f"**Content:** {node.content}")
                        st.write(f"**Rationale:** {node.rationale}")
                        st.write(f"**Validation Status:** `{node.validation_status}`")
                        if node.dependencies:
                            st.write(f"**Dependencies:** `{', '.join(node.dependencies)}`")

        with st_entropy:
            st.markdown("#### Information-Theoretic Density & Complexity")
            if b_res.trace:
                ent_prof = b_res.metadata
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.metric("Shannon Entropy H(X)", f"{ent_prof.get('entropy_bits', 0.0):.2f} bits")
                with e2:
                    st.metric("Renyi Order-2 Collision Entropy", f"{ent_prof.get('renyi_bits', 0.0):.2f} bits")
                with e3:
                    st.metric("Nodes in Reasoning DAG", f"{ent_prof.get('graph_nodes_count', 0)}")

            if b_res.trace and b_res.trace.stage_latencies:
                st.markdown("##### Stage Latency Breakdown (ms)")
                st.json(b_res.trace.stage_latencies)

        with st_critic:
            st.markdown("#### Metacognitive Fallacy & Bias Audit Scorecard")
            if b_res.bias_audits:
                for audit in b_res.bias_audits:
                    sev_color = "#dc3545" if audit.severity.value == "CRITICAL" else "#ffc107" if audit.severity.value == "WARNING" else "#17a2b8"
                    st.markdown(f"""
                    <div style="border-left: 4px solid {sev_color}; padding: 10px; background-color: #faf9fd; margin-bottom: 10px; border-radius: 4px;">
                        <span style="font-weight: bold; color: {sev_color};">[{audit.severity.value}] {audit.bias_type}</span><br>
                        <b>Detected:</b> <code>"{audit.detected_snippet}"</code><br>
                        <b>Explanation:</b> {audit.explanation}<br>
                        <b>Remediation:</b> {audit.remediation_suggestion}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ **Epistemic Validation Passed**: Zero cognitive bias violations or formal logical fallacies detected in this reasoning chain.")

            with st.expander("Full Validation Notes"):
                for note in (b_res.trace.bias_check_notes if b_res.trace else []):
                    st.write(f"• {note}")

        with st_memory:
            st.markdown("#### Global Workspace & Synaptic Working Memory (GWT-SWM 2.0)")
            snap = brain_instance.get_workspace_snapshot()
            if snap:
                st.markdown("##### Active Short-Term Memory Items (with Hebbian Plasticity)")
                st.json(snap)
            else:
                st.info("Working memory is currently in resting baseline state.")

            if b_res.trace and b_res.trace.broadcast_messages:
                st.markdown("##### Conscious Global Workspace Broadcast Timeline")
                for msg in b_res.trace.broadcast_messages:
                    st.markdown(f"**Cycle {msg.broadcast_cycle}** [`{msg.sender_module}`] *(Salience: {msg.salience_weight:.2f})*: {msg.content_summary}")

        with st_triples:
            st.markdown("#### Extracted Semantic Knowledge Triples")
            if b_res.triples:
                for t in b_res.triples:
                    st.markdown(f"- **({t.subject})** ──`[{t.predicate}]`──> **({t.object})** *(Confidence: {t.confidence*100:.0f}%)*")
            else:
                st.caption("No explicit structured triples extracted from query.")

# 7. DOCUMENTATION TAB
with tab_docs:
    st.markdown("### API Integration Guide")
    st.caption("Extracted directly from the project API reference manual.")
    
    docs_path = os.path.join(PROJECT_ROOT, "docs", "api", "quickstart.md")
    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            docs_content = f.read()
        st.markdown(docs_content)
    else:
        # Fallback to readme
        docs_path = os.path.join(PROJECT_ROOT, "docs", "api", "README.md")
        if os.path.exists(docs_path):
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_content = f.read()
            st.markdown(docs_content)
        else:
            st.info("API Reference manual not found.")
