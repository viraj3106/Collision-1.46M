# COLLISION Containerization Deployment Guide

This guide details how to build and run the containerized COLLISION Developer Platform locally.

## 1. Prerequisites
- Docker (version 20.10 or higher)
- Docker Compose v2

## 2. Model Placement
The containerized API does not bake model weights inside images. You must mount the local weights folder:
1. Verify the `models/` directory structure exists in your workspace root:
   ```
   models/
   └── collision-10m/
       ├── model.pt
       ├── config.json
       └── tokenizer/
   ```
2. Check that `models/collision-10m/model.pt` exists and matches the frozen checksum:
   - **Expected SHA256**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`

## 3. Environment Setup
Copy the template `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Ensure the DB passwords and allowed origins are correctly set.

## 4. Docker Compose Startup
Build and launch all services (PostgreSQL, FastAPI API, and Streamlit Portal) in the background:
```bash
docker compose up --build -d
```

## 5. Port Allocations & Access
Once started, the following services are accessible:
- **Developer Portal**: [http://localhost:8501](http://localhost:8501)
- **FastAPI API**: [http://localhost:8000](http://localhost:8000)
- **Postgres DB**: Hidden from host public port maps for safety; only accessible internally within the isolated `collision-net` network.

## 6. Access Workflow
1. Navigate to the Developer Portal ([http://localhost:8501](http://localhost:8501)).
2. Under **Create Developer Account**, sign up with your email and password.
3. Access your account, click **Generate API Key**, and copy your `col_...` token.
4. Go to the **Playground Client** tab, enter the copied token, configure decoding settings, and execute a completion query.

## 7. First API Request (cURL)
Test completions from your terminal:
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer col_your_actual_copied_key_token" \
  -d '{
    "model": "collision-10m",
    "prompt": "Artificial intelligence is",
    "max_tokens": 50
  }'
```

## 8. Shutdown
Stop and remove all running containers and volumes:
```bash
docker compose down -v
```
