# COLLISION Phase 26 — Public Cloud Deployment Report

This report presents the system architecture, DNS layout, gateway parameters, and verification checklists for the first public deployment of the COLLISION platform.

## 1. System Architecture

The platform deployment architecture is organized as follows:
- **`nginx`**: Gateway service serving the static React landing page at `/` from `/usr/share/nginx/html`, proxying the FastAPI server under `/v1`, `/health`, `/ready`, and routing the Streamlit developer portal under `/portal`.
- **`collision-portal`**: Dashboard dashboard running under `--server.baseUrlPath /portal`.
- **`collision-api`**: completions service with concurrent semaphore limits.
- **`postgres` / `redis`**: private internal services with no public host ports exposed.

## 2. DNS Configuration
To route public traffic to the AWS EC2 instance:
- **A Record**: `YOUR_DOMAIN` → `EC2_ELASTIC_IP`
- **A Record Subdomain**: `console.YOUR_DOMAIN` (optional subdomain, currently mapped under `/portal` path).

## 3. AWS Security Group Rules
Only the minimal required ports are exposed publicly:
- **Port 80 (HTTP)**: Open globally (redirects to port 443).
- **Port 443 (HTTPS)**: Open globally (TLS reverse proxy gateway).
- **Port 22 (SSH)**: Restricted strictly to the administrator's IP address.
- **Internal Ports** (`5432`, `6379`, `8000`, `8501`): completely private.

## 4. HTTPS & TLS Configuration
- **Certificates**: Mapped from host `/etc/letsencrypt/` directories outside git version controls.
- **TLS Version support**: locked to TLSv1.2 and TLSv1.3 protocols.
- **Redirection**: HTTP -> HTTPS redirects enforced via Nginx.

## 5. Acceptance Verification Checklists

| Metric / Check | Status | Verification Detail |
| :--- | :--- | :--- |
| **Domain Resolves** | **PENDING** | Requires registrar DNS propagation to Elastic IP. |
| **HTTPS Gateway** | **PENDING** | Requires Certbot certificate generation on target host. |
| **Nginx Path Mappings** | **PASS** | Serves website dist on `/`, routes `/portal` to Streamlit. |
| **Model checksum** | **PASS** | verified match: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97` |
| **Model Parameters** | **PASS** | Verified param count is `10,282,304`. |
| **PostgreSQL / Redis** | **PASS** | Local unittests pass cleanly. Private Compose network isolation confirmed. |
| **Onboarding signup** | **PASS** | Integration tests check credential hashing and session keys. |
| **Rate Limit 429** | **PASS** | Redis Sliding-Window limiter verified with correct headers. |
| **Admin protection** | **PASS** | Secured `/v1/developers` behind `verify_admin_token` filters. |
| **No secrets in git** | **PASS** | Sensitive values mapped in `.env.example` templates. |

---

## 6. Public Deployment Classification

Since the target AWS EC2 host Elastic IP address and registrar DNS routing configurations are pending, this phase is classified as:

```
PHASE_26_INFRASTRUCTURE_READY_DEPLOYMENT_BLOCKED
```

All docker compose layers, static landing pages, base-path routers, and smoke verification tests are verified, packaged, and fully prepared for EC2 instantiation once public endpoints are supplied.
