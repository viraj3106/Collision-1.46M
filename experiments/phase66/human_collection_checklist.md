# Phase 66 — Human Collection Checklist

This checklist is provided for beta testers participating in the COLLISION Public Beta. Beta testers should follow this step-by-step procedure to ensure their interactions are valid, consented, and properly ingested by the pipeline.

---

## BETA TESTER STEP-BY-STEP CHECKLIST

1. **Access the Public Beta Interface**:
   - Open the COLLISION Playground UI or submit requests via API key to `/v1/generate`.

2. **Formulate a Genuine Query / Task**:
   - Ask a real question or task (e.g. debugging code, explaining ML concepts, solving math problems, summarizing text, troubleshooting setup issues).

3. **Select the Appropriate Category**:
   - Select the matching category in the feedback dropdown:
     - `Programming`
     - `AI/ML`
     - `Mathematics`
     - `Science`
     - `Writing`
     - `Reasoning`
     - `Troubleshooting`
     - `Everyday Tasks`
     - `Summarization`

4. **Test Multi-turn & Follow-up Interactions (Where Natural)**:
   - Ask a follow-up question referencing the previous answer (e.g. "Can you rewrite that Python snippet using list comprehensions?").

5. **Evaluate the Model Output**:
   - Provide a positive rating (`thumbs_up` / positive signal) ONLY if the response is helpful and accurate.

6. **Grant Explicit Consent**:
   - Ensure the consent checkbox is checked (`consent = True`). Unconsented feedback will be automatically quarantined.

7. **Avoid Sensitive & Private Information**:
   - DO NOT include real email addresses, phone numbers, IP addresses, passwords, API keys (`col_`, `sk-`), or private personal data in prompts or feedback comments.

8. **Submit Feedback**:
   - Submit feedback through the UI or POST `/v1/feedback`.

---

## RECOMMENDED CATEGORY TARGETS FOR NEXT 13+ RECORDS

| Target Category | Recommended New Records | Focus Areas |
| :--- | :---: | :--- |
| **Programming** | 2+ | Python, Java, SQL, debugging, algorithms |
| **AI/ML** | 2+ | Model architecture, loss functions, prompt logic |
| **Mathematics** | 2+ | Algebra, probability, step-by-step calculations |
| **Science** | 1+ | Physics, chemistry, astronomy |
| **Writing** | 1+ | Grammar, rewriting, email drafting |
| **Reasoning** | 2+ | Logical comparison, trade-off analysis, multi-step planning |
| **Troubleshooting**| 1+ | Environment setup, API errors, dependency issues |
| **Everyday Tasks** | 1+ | Practical planning, recommendations |
| **Multi-turn / Follow-up** | 2+ | Conversational context, clarification requests |
