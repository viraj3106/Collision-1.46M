# COLLISION Production Deployment Documentation

This directory contains the production topology and configuration settings for hosting the COLLISION Developer Platform.

## 1. Production Topology
COLLISION runs a single-instance containerized deployment:
- **Nginx Reverse Proxy**: Gateway exposed on port 80/443 mapping SSL termination.
- **FastAPI API**: Upstream completions backend.
- **Streamlit Portal**: Upstream developer console web dashboard.
- **PostgreSQL**: Internal relational database.
- **Redis**: Internal rate limiting cache store.

```
Public Traffic (80/443)
        ↓
   Nginx (SSL)
   ├── /v1, /health, /ready  →  collision-api:8000
   └── / (Web portal UI)     →  collision-portal:8501
```

## 2. TLS Certificates Setup
Do not commit private certificates to version control.
1. Place your certificates in `/etc/ssl/certs/` on your host machine:
   - Certificate file: `collision.crt`
   - Private key file: `collision.key`
2. Update the docker-compose volumes in your production file to mount these certificates:
   ```yaml
   volumes:
     - /etc/ssl/certs/collision.crt:/etc/ssl/certs/collision.crt:ro
     - /etc/ssl/private/collision.key:/etc/ssl/private/collision.key:ro
   ```

### Creating Self-Signed Certs for Testing
For staging or local test runs:
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/nginx/collision.key \
  -out deployment/nginx/collision.crt \
  -subj "/CN=localhost"
```

## 3. Database & Cache Security
PostgreSQL and Redis run within the internal `collision-net` network bridge. They are not exposed to the public internet, securing data storage from outer ports scanning.
