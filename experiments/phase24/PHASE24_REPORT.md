# COLLISION Phase 24 — Production Cloud Deployment Report

This report presents the system architecture, Nginx configurations, backup scripts, and security details for deploying the COLLISION developer platform.

## 1. System Architecture

The platform runs a five-service stack:
- **`nginx`**: Exposes port 80/443 mapping SSL reverse proxy rules and HSTS.
- **`collision-portal`**: Dashboard dashboard running on port 8501.
- **`collision-api`**: Causal completions engine running on port 8000.
- **`postgres`**: Relational database storing credentials.
- **`redis`**: Key-value cache store mapping rate limits.

## 2. Docker Services Configuration

All containers run inside an isolated network bridge:
- **Exposed ports**: Only Nginx (ports 80/443) is mapped to the host interface.
- **Privileges**: API and Portal containers run as `appuser` (non-root).
- **Resource Constraints**: API is capped at `2.0 CPUs` and `2G Memory`.

## 3. HTTPS Gateway

- **TLS Mappings**: Configured to mount Let's Encrypt certificates (`fullchain.pem` and `privkey.pem`) from host `/etc/letsencrypt/`.
- **Headers**: Enforces `X-Frame-Options: DENY`, `Strict-Transport-Security`, and custom CSP restrictions.

## 4. Backups and Recovery

Disaster recovery configurations are located in `deployment/backup/`:
- **`backup.sh`**: Dumps database state using `pg_dump` and gz-compresses the files, retaining the last 7 days of backups.
- **`restore.sh`**: Safely drops and restores database state from a gzip backup.

## 5. Security Gates & Model Verification

- **Admin Routes**: `/v1/developers` and `/v1/developers/{email}` endpoints are secured by checking `X-Admin-Token` against `ADMIN_SECRET`.
- **Model Checksum**: Startup checks hash `model.pt` and compare it against `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`, halting execution if it fails.

## 6. Smoke Test Results

- All 14 unittest test cases pass successfully.
- **E2E Smoke Test**: `tests/test_production_deployment.py` confirms successful `/health` and `/ready` states, administrative header checks, parameter counters, and checksum verifications.

## 7. Public Cloud Readiness Summary

Since the actual domain DNS mappings and TLS certificates are missing from our sandbox workspace, the deployment state is classified as:

```
INFRASTRUCTURE READY / DEPLOYMENT BLOCKED
```

The stack is structurally complete, verified syntactically, and fully prepared for EC2 orchestration.
