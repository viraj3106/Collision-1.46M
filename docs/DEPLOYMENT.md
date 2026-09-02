# COLLISION-10M Deployment Guide

This guide provides step-by-step instructions to configure, build, and deploy the **COLLISION-10M** Public Beta application.

---

## 1. System Architecture

```
                               ┌─────────────────────────────┐
                               │  Firebase Auth & Firestore  │
                               └──────────────┬──────────────┘
                                              │ (Client Auth & Feedback)
                                              ▼
┌─────────────────────────┐          ┌─────────────────────────┐          ┌─────────────────────────┐
│   Vite / React Website  ├─────────►│  FastAPI COLLISION API  ├─────────►│  COLLISION-10M Model    │
│   (Console & Playground)│          │  (Port 8000 REST API)   │          │  (10,282,304 Params)    │
└─────────────────────────┘          └────────────┬────────────┘          └─────────────────────────┘
                                                  │
                                                  ▼
                                     ┌─────────────────────────┐
                                     │ SQLite / PostgreSQL DB  │
                                     └─────────────────────────┘
```

---

## 2. Step-by-Step Deployment Instructions

### Step 1: Configure Environment Variables
Copy `.env.example` to `.env` and fill in configuration values:

```bash
cp .env.example .env
```

Set frontend Firebase values and backend secret tokens:
- `VITE_FIREBASE_API_KEY`: Your Firebase project API key
- `VITE_FIREBASE_PROJECT_ID`: Your Firebase project ID
- `ADMIN_SECRET`: Strong secret for admin management endpoints
- `MODEL_PATH`: `models/collision-10m`
- `MODEL_SHA256`: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`

---

### Step 2: Build Frontend Website
Navigate to `website/` and install dependencies & build static bundle:

```bash
cd website
npm install
npm run build
```

Static production assets will be output to `website/dist/`.

---

### Step 3: Build Backend Docker Image
Build the lightweight container image:

```bash
docker build -t collision-api -f Dockerfile.api .
```

---

### Step 4: Start Backend Service
Run the Docker container or local FastAPI instance:

```bash
# Using Docker:
docker run -d -p 8000:8000 --env-file .env --name collision-api-service collision-api

# Or directly with Python / Uvicorn:
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

### Step 5: Verify Health & Readiness
Check service endpoints to confirm API status, database connection, and model checksum integrity:

```bash
# Process health
curl http://localhost:8000/health

# Readiness probe (verifies database & model SHA256)
curl http://localhost:8000/ready
```

Expected output for `/ready`:
```json
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "model": "ok"
  }
}
```

---

### Step 6: Create Developer Account
Register a developer account via API or developer portal:

```bash
curl -X POST http://localhost:8000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "developer@example.com", "password": "securepassword123"}'
```

---

### Step 7: Create API Key
Authenticate to receive a session token, then generate an API key:

```bash
# 1. Login
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "developer@example.com", "password": "securepassword123"}'

# 2. Generate API Key (use returned session_token)
curl -X POST http://localhost:8000/v1/keys \
  -H "Authorization: Bearer <session_token>" \
  -H "Content-Type: application/json" \
  -d '{"developer_id": 1}'
```

---

### Step 8: Generate Text
Submit completion requests to `COLLISION-10M`:

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Authorization: Bearer col_<your_api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "collision-10m",
    "prompt": "Artificial intelligence is",
    "max_tokens": 20,
    "temperature": 0.7
  }'
```

---

### Step 9: Submit Feedback
Submit user feedback telemetry for quality data pipeline processing:

```bash
curl -X POST http://localhost:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "prompt": "Artificial intelligence is",
    "model": "collision-10m",
    "response": "a branch of computer science.",
    "rating": "thumbs_up",
    "feedback": "Clear explanation",
    "category": "general",
    "consent": true
  }'
```
