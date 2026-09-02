#!/bin/bash
# -----------------------------------------------------------------------------
# COLLISION Production Deployment Script
# -----------------------------------------------------------------------------
set -e

echo "=== Starting COLLISION Cloud Deployment ==="

# 1. Validate Environment File
if [ ! -f .env ]; then
    echo "Error: .env configuration file not found. Copy .env.example and populate secrets."
    exit 1
fi

# Load variables
export $(grep -v '^#' .env | xargs)

# 2. Validate Secrets present
if [ -z "${POSTGRES_USER}" ] || [ -z "${POSTGRES_PASSWORD}" ] || [ -z "${POSTGRES_DB}" ] || [ -z "${ADMIN_SECRET}" ]; then
    echo "Error: Critical database or administrator secrets are missing in .env."
    exit 1
fi

# 3. Validate Docker Installation
if ! command -v docker &> /dev/null; then
    echo "Error: docker command not found. Install Docker before deployment."
    exit 1
fi
if ! docker compose version &> /dev/null; then
    echo "Error: docker compose version not found."
    exit 1
fi

# 4. Validate Model Availability & Checksum
MODEL_FILE="models/collision-10m/model.pt"
if [ ! -f "${MODEL_FILE}" ]; then
    echo "Error: Production model weights file not found at ${MODEL_FILE}."
    exit 1
fi

EXPECTED_SHA="d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
echo "Validating model checksum..."
# Check OS platform to select checksum command
if command -v sha256sum &> /dev/null; then
    ACTUAL_SHA=$(sha256sum "${MODEL_FILE}" | awk '{print $1}')
elif command -v shasum &> /dev/null; then
    ACTUAL_SHA=$(shasum -a 256 "${MODEL_FILE}" | awk '{print $1}')
else
    # Fallback to python hashing to remain cross-platform compatible
    ACTUAL_SHA=$(python3 -c "import hashlib; print(hashlib.sha256(open('${MODEL_FILE}', 'rb').read()).hexdigest())")
fi

if [ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]; then
    echo "Error: Model checksum validation failed! Mismatch."
    echo "Expected: ${EXPECTED_SHA}"
    echo "Actual:   ${ACTUAL_SHA}"
    exit 1
fi
echo "Model checksum matched."

# 5. Build and Deploy stack
echo "Building and launching containers..."
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

# 6. Monitor container health status
echo "Waiting for services to report healthy..."
sleep 10

# Print status reports
docker compose -f docker-compose.prod.yml ps

echo "Checking backend health endpoint..."
if curl -s -f http://localhost:8000/health &> /dev/null; then
    echo "FastAPI backend health check passed."
else
    echo "Warning: FastAPI backend health check failed. View logs via: docker compose logs api"
fi

echo "=== COLLISION Deployment Status: SUCCESS ==="
