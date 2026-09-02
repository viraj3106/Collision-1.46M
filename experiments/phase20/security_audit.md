# COLLISION Developer API Security Audit

This document audits the security characteristics of the Phase 20 developer-facing REST API.

## 1. Credentials and Secret Exposure

- **API Keys Logging**: API keys are never logged. The logs only register prefixes (e.g. `col_xxxx`) for tracking and request IDs (e.g. `req_xxxx`).
- **Secrets in Commits**: There are no hardcoded secrets, API signing tokens, or database passwords in the codebase.
- **Environment Exclusions**: The `.env` file and local test sqlite databases are explicitly excluded from tracking in `.gitignore`.
- **Database Storage**: API keys are generated using cryptographically secure random values and stored as SHA256 hashes. Plaintext keys are never stored.

## 2. Authentication Integrity

- **Invalid Keys**: Tested and rejected with `401 Unauthorized`.
- **Revoked Keys**: Keys marked as `revoked` in the SQLite database are verified and rejected immediately.
- **Header Parsing**: Malformed, empty, or non-Bearer headers fail authorization gracefully without raising server-side traceback errors.

## 3. Input Validation & Filesystem Access

- **Arbitrary Model Loading**: The model parameter is validated to match only `"collision-10m"`. Arbitrary path strings or model weights are blocked from loading, eliminating local file inclusion (LFI) or path traversal vulnerabilities.
- **Context Boundaries**: Prompt and completion length checks prevent memory overflow or out-of-boundary indexing in PyTorch. Prompts exceeding 256 tokens are rejected with a 400 validation error.
- **Prompts Filesystem Traversal**: Since prompt strings are strictly tokenized by BPE and fed into the transformer embeddings, they cannot execute shell commands or access the file system.

## 4. Error Handling

- **Tracebacks & Stack Traces**: Catch-all exception handlers in FastAPI suppress python stack traces, formatting all server errors into structured JSON response errors (e.g. `type: server_error`).
- **Request IDs**: Request IDs (`X-Request-ID`) are generated dynamically using UUIDs containing no system metadata or secret indicators.
