# COLLISION Production Deployment Guide

This guide details the steps to deploy the **COLLISION** grounded answering system in production environments.

---

## 1. System Requirements

### Hardware Requirements
* **CPU:** 4+ Cores (x86_64 or ARM64)
* **RAM:** 4 GB minimum (8 GB recommended for concurrent workloads)
* **Storage:** 2 GB available SSD storage
* **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ for accelerated batch inference (CPU inference is fully supported)

### Software Requirements
* **Operating System:** Linux (Ubuntu 22.04 LTS+, Debian 12+, RHEL 9+) or Windows Server 2022+
* **Python:** 3.10, 3.11, 3.12, or 3.13
* **Container Runtime (Optional):** Docker 24.0+ & Docker Compose v2+

---

## 2. Environment Configuration

Copy the production environment template:

```bash
cp .env.example .env
```

### Key Configuration Variables (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `COLLISION_ENVIRONMENT` | `production` | Environment mode (`development`, `staging`, `production`) |
| `COLLISION_HOST` | `0.0.0.0` | API binding address |
| `COLLISION_PORT` | `8000` | API service port |
| `COLLISION_LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `COLLISION_CORS_ORIGINS` | `http://localhost:3000,http://localhost:8000` | Allowed CORS origins (comma-separated, no wildcards in prod) |
| `COLLISION_MAX_INPUT_LENGTH` | `4096` | Max query string length (characters) |
| `COLLISION_MAX_CONTEXT_LENGTH`| `16384` | Max combined context size (characters) |
| `COLLISION_REQUEST_TIMEOUT_SECONDS` | `30.0` | Global query execution timeout |
| `COLLISION_WEB_TIMEOUT_SECONDS` | `5.0` | HTTP fetch and search timeout |
| `COLLISION_RATE_LIMIT` | `60/minute` | Per-client IP rate limit |
| `COLLISION_RATE_LIMIT_ENABLED` | `true` | Enable/disable rate limiter middleware |
| `COLLISION_DB_PATH` | `collision_production.db`| SQLite metadata & sessions storage |
| `COLLISION_LOCAL_INDEX_DIR` | `data/rag_index` | Local vector/BM25 knowledge base index |

---

## 3. Deployment Methods

### Option A: Docker Container Deployment (Recommended)

COLLISION provides a hardened, non-root multi-stage Dockerfile:

1. **Build the container image:**
   ```bash
   docker build -f Dockerfile.api -t collision-api:latest .
   ```

2. **Run with Docker Compose:**
   ```bash
   docker compose up -d
   ```

3. **Verify running containers:**
   ```bash
   docker compose ps
   ```

4. **View logs:**
   ```bash
   docker compose logs -f api
   ```

---

### Option B: Bare-Metal / Systemd Deployment

1. **Create virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Verify checkpoint integrity:**
   ```bash
   python -c "from collision.config import validate_checkpoints; print(validate_checkpoints())"
   ```

3. **Configure systemd service (`/etc/systemd/system/collision.service`):**
   ```ini
   [Unit]
   Description=COLLISION Grounded Answering API
   After=network.target

   [Service]
   Type=simple
   User=collision
   Group=collision
   WorkingDirectory=/opt/collision
   EnvironmentFile=/opt/collision/.env
   ExecStart=/opt/collision/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
   Restart=always
   RestartSec=5s
   LimitNOFILE=65536

   [Install]
   WantedBy=multi-user.target
   ```

4. **Enable and start the service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable collision
   sudo systemctl start collision
   sudo systemctl status collision
   ```

---

## 4. Health & Readiness Probes

COLLISION exposes standard orchestration probes for Kubernetes and load balancers:

### Liveness Probe (`GET /health`)
Verifies process execution and basic server health:
```bash
curl -f http://localhost:8000/health
```
Response:
```json
{
  "status": "healthy",
  "service": "COLLISION API",
  "version": "1.0.0",
  "model": "collision-10m",
  "timestamp": "2026-09-16T11:00:00.000000Z"
}
```

### Readiness Probe (`GET /ready`)
Performs deep subsystem verification (model weights, SHA-256 integrity, tokenizer, database, local RAG index):
```bash
curl -f http://localhost:8000/ready
```
Response:
```json
{
  "status": "ready",
  "checks": {
    "database": true,
    "model_initialized": true,
    "model_checkpoint": true,
    "tokenizer": true,
    "local_index": true
  },
  "timestamp": "2026-09-16T11:00:00.000000Z"
}
```

---

## 5. Observability & Monitoring

COLLISION provides real-time JSON metrics endpoint at `GET /v1/metrics`:
```bash
curl http://localhost:8000/v1/metrics
```

Metrics tracked include:
* `total_requests`: Total number of queries processed
* `total_errors`: Count of 4xx and 5xx responses
* `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`: Response time percentiles
* `mode_breakdown`: Queries served per mode (`AUTO`, `LOCAL`, `WEB`, `HYBRID`, `REFUSAL`)
* `status_breakdown`: Answering statuses (`ANSWERED`, `INSUFFICIENT_INFORMATION`, `CONFLICT_DETECTED`, `PROMPT_INJECTION_BLOCKED`)
* `rate_limit_exceeded_count`: Throttled requests count

---

## 6. Security & Hardening Checklist

- [x] **Zero Model Training**: Checkpoints locked via SHA-256 verification on startup.
- [x] **SSRF Protection**: Outbound web grounding blocks private IP spaces (127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, metadata servers).
- [x] **Input Sanitization**: Request bodies exceeding `COLLISION_MAX_INPUT_LENGTH` (4096 chars) rejected with HTTP 413.
- [x] **Rate Limiting**: Sliding window rate limiting enabled per client IP.
- [x] **Prompt Injection Defense**: Multi-tier extraction filtering and adversarial pattern rejection.
- [x] **Non-Root Container**: Container executes under unprivileged UID 10001.
