# COLLISION API Admin Boundary Audit

This document audits the administrative endpoints and access controls implemented in Phase 20 and Phase 21.

## 1. Current Administrative Endpoints

The following endpoints are defined in `api/routes.py` for developer account management and API key creation:

- `POST /v1/developers`: Legacy developer registration (passwordless).
- `GET /v1/developers/{email}`: Legacy developer detail retrieval.
- `POST /v1/keys`: Generates a new API key.
- `GET /v1/developers/{id}/keys`: Lists keys for a developer.
- `POST /v1/keys/{id}/revoke`: Revokes a key.
- `GET /v1/developers/{id}/usage`: Retrieves usage statistics.

## 2. Admin Protection Status and Weaknesses

- **Weakness 1: Legacy Endpoints are Publicly Accessible**: The routes `POST /v1/developers` and `GET /v1/developers/{email}` are unprotected and do not require session authentication. A malicious actor can register arbitrary emails or scan details of registered developer accounts.
- **Weakness 2: Lack of Role-Based Access Controls (RBAC)**: Currently, there is no separate "admin" flag or role. All active developer accounts can access normal portal operations, but the system relies purely on IDOR checks inside session filters rather than checking for admin privileges on creation routes.
- **Why it is safe locally**: Because these routes are bound to `127.0.0.1` and are only queried inside local sandbox environments, there is no external hazard. However, they should **never** be exposed publicly without admin key filters.

## 3. Required Production Security Changes

1. **Admin Secret Header Validation**: Create a secret admin key parameter (e.g. `COLLISION_ADMIN_SECRET`) in `.env` and validate it via header checks (e.g. `X-Admin-Token: <secret>`) on legacy developer creation routes.
2. **Deprecate Passwordless Registrations**: Restrict developer registration strictly to the `POST /v1/auth/signup` endpoint using password credentials, or integrate an external OIDC/OAuth2 service provider (like Auth0 or Firebase Auth) which handles user verification securely.
3. **Audit Log Trail**: Write dedicated audit logs for administrative events (e.g. key creation, user suspension) to track administrator operations.
