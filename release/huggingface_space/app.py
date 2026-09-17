"""
COLLISION AI & NLP Lab — Hugging Face Space Application
"""

import os
import sys
import time
import streamlit as st

# Setup sys.path to allow root imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.nlp import (
    CollisionNLPEngine,
    CollisionNLPProcessor,
    TextRankKeyphraseExtractor,
    TopicClassifier,
    ToneAnalyzer,
    SemanticSimilarityCalculator,
    GrammarProofreader,
    ReadingComprehensionEngine,
    TextTransformer
)

st.set_page_config(
    page_title="COLLISION AI & NLP Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        color: #1a1a1a;
        margin-bottom: 2px;
    }
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fc;
        border: 1px solid #e3e6f0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-title'>⚡ COLLISION AI & NLP Lab</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Industrial-Strength Natural Language Processing & CPU-First 10.28M Parameter Neural Intelligence</div>", unsafe_allow_html=True)

tabs = st.tabs([
    "💬 Conversational Assistant",
    "🏷️ Keyphrases & NER",
    "📊 Topic & Tone Classifier",
    "✍️ Grammar & Spell Proofreader",
    "📈 Readability & Complexity",
    "🔍 Reading Comprehension (Context QA)",
    "🔢 Deterministic Math & Conversions",
    "🔄 Text Transformers & Similarity"
])

# 1. Conversational Assistant Tab
with tabs[0]:
    st.subheader("💬 Conversational Dialogue & Grounded QA")
    st.caption("Zero-latency informal chat, mathematical answers, and grounded general knowledge.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am **COLLISION**, your neural AI assistant. How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question, enter math, or say hello...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            t0 = time.perf_counter()
            nlp_res = CollisionNLPEngine.process_query(user_input)
            if nlp_res:
                ans = nlp_res.answer
            else:
                ans = "I have received your query. In the full deployment, open-domain questions are synthesized using the Grounded Multi-Source Evidence Engine."
            elapsed_ms = (time.perf_counter() - t0) * 1000
            
            st.markdown(ans)
            st.caption(f"⚡ Latency: {elapsed_ms:.2f}ms")
            st.session_state.messages.append({"role": "assistant", "content": ans})

# 2. Keyphrase & NER Tab
with tabs[1]:
    st.subheader("🏷️ TextRank Keyphrase & Named Entity Extraction")
    kp_input = st.text_area(
        "Enter text to extract keyphrases and entities:",
        value="Quantum computing relies on qubits, superposition, and entanglement to execute quantum algorithms on specialized hardware developed by Google and IBM.",
        height=100
    )
    if st.button("Extract Keyphrases & Entities", key="btn_kp"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Keyphrases (TextRank)")
            kp = CollisionNLPEngine.extract_keywords(kp_input)
            st.write("**Top Keyphrases:**")
            for p in kp.keyphrases:
                st.markdown(f"• **{p}**")
            st.write("**Salient Words & Scores:**")
            for w, s in kp.top_keywords:
                st.markdown(f"• `{w}`: `{s:.2f}`")

        with c2:
            st.markdown("#### Named Entities (NER)")
            ent = CollisionNLPEngine.extract_entities(kp_input)
            if ent.persons: st.markdown(f"• **Persons**: {', '.join(ent.persons)}")
            if ent.organizations: st.markdown(f"• **Organizations**: {', '.join(ent.organizations)}")
            if ent.locations: st.markdown(f"• **Locations**: {', '.join(ent.locations)}")
            if ent.dates_years: st.markdown(f"• **Dates & Years**: {', '.join(ent.dates_years)}")
            if ent.quantities: st.markdown(f"• **Quantities**: {', '.join(ent.quantities)}")

# 3. Topic & Tone Tab
with tabs[2]:
    st.subheader("📊 Multi-Domain Topic & Tone Analysis")
    topic_input = st.text_area(
        "Enter text to classify topic and analyze tone:",
        value="The patient underwent cardiac bypass surgery at the hospital following clinical diagnosis and antibody therapy. Furthermore, empirical evidence demonstrates substantial quantitative correlation.",
        height=100
    )
    if st.button("Analyze Topic & Tone", key="btn_topic"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Topic Classification")
            top = CollisionNLPEngine.classify_topic(topic_input)
            st.success(f"**Primary Topic**: {top.primary_topic} (Confidence: {top.confidence*100:.0f}%)")
            st.write("**Domain Distribution:**")
            for t, s in top.topic_distribution:
                st.progress(float(s), text=f"{t} ({s*100:.0f}%)")
            if top.keywords_matched:
                st.write(f"**Key Markers**: {', '.join(top.keywords_matched)}")

        with col2:
            st.markdown("#### Tone & Formality")
            tone = CollisionNLPEngine.analyze_tone(topic_input)
            st.info(f"**Primary Tone**: {tone.primary_tone}")
            st.metric("Formality Score", f"{tone.formality_score:.0f}/100", tone.formality_label)
            st.write(f"**Objectivity**: {'Objective / Factual' if tone.is_objective else 'Subjective / Opinionated'}")
            st.write(f"**Subjectivity Score**: `{tone.subjectivity_score*100:.0f}%`")
            st.write(f"**Assessment**: {tone.summary}")

# 4. Grammar & Proofreader Tab
with tabs[3]:
    st.subheader("✍️ Rule-Based Grammar & Spell Proofreader")
    proof_input = st.text_area(
        "Enter text to proofread and correct:",
        value="I ate a apple on the the kitchen table . It were very delicious .",
        height=100
    )
    if st.button("Proofread Text", key="btn_proof"):
        proof = CollisionNLPEngine.proofread(proof_input)
        if proof.is_clean:
            st.success("✅ No issues detected. Text is clean.")
        else:
            st.markdown("#### Corrected Version")
            st.info(f"> {proof.corrected_text}")
            st.markdown(f"#### Issues Identified ({proof.issues_found})")
            for issue in proof.issues:
                st.markdown(f"• **{issue.issue_type}**: `\"{issue.original}\"` -> `\"{issue.replacement}\"` (*{issue.explanation}*)")

# 5. Readability Tab
with tabs[4]:
    st.subheader("📈 Readability & Complexity Calculator")
    read_input = st.text_area(
        "Enter text to compute readability indices:",
        value="Artificial intelligence and deep neural networks have fundamentally transformed computational linguistics by enabling scalable sequence modeling.",
        height=100
    )
    if st.button("Compute Readability", key="btn_read"):
        metrics = CollisionNLPEngine.compute_readability(read_input)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flesch Reading Ease", f"{metrics.flesch_reading_ease}/100")
        c2.metric("Flesch-Kincaid Grade", f"Grade {metrics.flesch_kincaid_grade:.1f}")
        c3.metric("Gunning Fog Index", f"{metrics.gunning_fog_index:.1f}")
        c4.metric("Lexical Richness (TTR)", f"{metrics.lexical_diversity_ttr*100:.1f}%")
        st.write(f"**Reading Ease Assessment**: **{metrics.reading_ease_description}**")
        st.write(f"**Statistics**: {metrics.word_count} words across {metrics.sentence_count} sentences ({metrics.avg_words_per_sentence:.1f} words/sentence, {metrics.avg_syllables_per_word:.2f} syllables/word)")

# 6. Reading Comprehension Tab
with tabs[5]:
    st.subheader("🔍 SQuAD-Style Context Reading Comprehension")
    ctx = st.text_area(
        "Context Passage:",
        value="Alan Turing designed the ACE (Automatic Computing Engine) in 1945 at the National Physical Laboratory in London. The ACE was one of the earliest stored-program computer designs.",
        height=120
    )
    q = st.text_input("Question regarding the context:", value="Where was the ACE computer designed?")
    if st.button("Answer Question from Context", key="btn_ctx"):
        ans = CollisionNLPEngine.answer_from_context(ctx, q)
        if ans.context_found:
            st.success(f"**Answer**: {ans.answer}")
            st.info(f"**Confidence**: `{ans.confidence*100:.0f}%` | **Source Sentence**: *\"{ans.sentence}\"*")
        else:
            st.warning("Could not conclusively find answer in context.")

# 7. Math Tab
with tabs[6]:
    st.subheader("🔢 Deterministic Math, Geometry & Statistics Solver")
    math_q = st.text_input("Enter math expression, geometry formula, statistics, or unit conversion:", value="area of circle with radius 7")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    if col_ex1.button("Example: Geometry (Circle)"): math_q = "area of circle with radius 7"
    if col_ex2.button("Example: Statistics (Mean)"): math_q = "mean of 15, 25, 35, 45, 55"
    if col_ex3.button("Example: Temp Conversion"): math_q = "100 celsius to fahrenheit"

    if st.button("Solve Query", key="btn_math"):
        res = CollisionNLPEngine.solve_math(math_q)
        if res:
            st.success(res)
        else:
            st.error("Could not parse mathematical expression.")

# 8. Transformers & Similarity Tab
with tabs[7]:
    st.subheader("🔄 Text Style Transformations & Semantic Similarity")
    t_input = st.text_area("Text to transform:", value="hlo bro this is gonna be super cool and awesome gotta check it out", height=80)
    c1, c2, c3 = st.columns(3)
    if c1.button("Formalize"):
        st.write(TextTransformer.formalize(t_input).transformed_text)
    if c2.button("Simplify"):
        st.write(TextTransformer.simplify("We utilize this methodology to elucidate the aforementioned paradigm.").transformed_text)
    if c3.button("Bulletize"):
        st.write(TextTransformer.bulletize("First key achievement. Second major breakthrough. Third strategic priority.").transformed_text)
