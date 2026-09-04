# BETA TESTER GUIDE — COLLISION PUBLIC BETA DATA COLLECTION

Welcome to the COLLISION Public Beta. This guide explains how human testers can interact with the public beta interface to contribute high-quality, consented real-world data to the COLLISION research project.

---

## STEP-BY-STEP INSTRUCTIONS

### Step 1: Open the Public Beta
- Launch the COLLISION Playground UI or connect via API client to the `/v1/generate` and `/v1/feedback` endpoints.

### Step 2: Formulate Genuine Questions & Tasks
- Ask real, non-fabricated questions across diverse subject areas.

### Step 3: Explore Varied Categories
Select the appropriate category when evaluating the response:
- **Programming**: Debugging, Python/Java snippets, SQL queries, algorithms.
- **AI/ML**: Concept explanations, loss functions, attention mechanisms.
- **Mathematics**: Algebra, probability, step-by-step calculations.
- **Science**: Physics principles, chemical reactions, astronomy questions.
- **Reasoning**: Logic puzzles, trade-off comparisons, multi-step planning.
- **Writing**: Rewriting, summarizing text, email drafting, grammar.
- **Troubleshooting**: Environment setup, dependency issues, API error interpretation.
- **Everyday Tasks**: Planning itineraries, recipe modifications, practical advice.
- **General Knowledge**: General facts and Q&A.

### Step 4: Try Natural Follow-up Questions
- Engage in multi-turn conversations where appropriate (e.g., initial query $\rightarrow$ answer $\rightarrow$ clarification request $\rightarrow$ follow-up).

### Step 5: Rate Useful Responses
- Provide positive feedback (`thumbs_up`) ONLY if the response is helpful, accurate, and high-quality.

### Step 6: Grant Explicit Consent
- Check the consent box (`consent = True`) to grant permission for your interaction to be included in the consented research dataset.

### Step 7: DO NOT Submit Sensitive Data
**NEVER submit**:
- Passwords or credentials
- Secret API keys (`col_`, `sk-`, `bearer`)
- Personal email addresses
- Phone numbers or IP addresses
- Private personal identifiers or confidential data

---

## SUCCESS CRITERION & TARGET

- Target: Move from 7 clean records to 20+ clean records.
- Note: Synthetic data, automated test traffic, and manufactured LLM conversations are strictly prohibited. Only genuine human interactions will be accepted.
