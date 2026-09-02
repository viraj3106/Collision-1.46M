# COLLISION Developer Platform Guide

COLLISION is a small developer-focused language model platform designed to run lightweight AI tasks on standard local systems.

## 1. What is COLLISION?
COLLISION-10M is a lightweight, decoder-only base transformer model containing exactly `10,282,304` parameters. It provides efficient causal text continuation on standard CPU devices.

## 2. Who is it For?
- **Prototyping**: Developers who want to test basic language generation workflows locally without API fees.
- **Academic Research**: Students studying attention characteristics and parameter scaling bounds.
- **Embedded Computing**: Deploying tiny, localized autocompletion helpers on low-power devices.

## 3. How API Keys Work
API authorization requires a Bearer key.
- Flat file keys are prefix-tagged with `col_`.
- Plaintext credentials are shown **only once** upon generation. The SQLite database stores only the secure SHA256 cryptographic hash of the key.
- If a key is compromised, it can be revoked instantly via the Developer Dashboard, immediately disabling any further generation queries.

## 4. Usage Tracking
Developer usage statistics are logged directly to the local SQLite database.
- Total requests executed
- Prompt token and completion token counts
- Generation latency (ms)
Stats are displayed in the dashboard tab of the portal interface.

## 5. Capabilities and Limitations
- **Not a Chatbot**: This is a base model. It has not undergone conversational alignment (RLHF or SFT). It will autocomplete patterns instead of answering questions dialogically.
- **Strict Context Limit**: The context window is limited to **256 tokens** (combined prompt and completion).
- **CPU Execution**: Latency scales based on local CPU clock cycles and core counts.

## 6. Product Positioning
> **Positioning**: "Build small AI features with a lightweight language model API."
> COLLISION is not a ChatGPT replacement, GPT killer, or an AGI system. It is an honest, local base language model optimized for resource efficiency.
