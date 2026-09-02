#!/bin/bash
set -e

# Load environment variables
BACKUP_DIR="/var/backups/collision"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/collision_db_${TIMESTAMP}.sql"

mkdir -p "${BACKUP_DIR}"

echo "Starting database backup at $(date)..."

# Exec pg_dump inside the docker container
docker exec collision-db pg_dump -U collision_admin collision_api > "${BACKUP_FILE}"

# Compress the backup file
gzip "${BACKUP_FILE}"
echo "Backup completed successfully: ${BACKUP_FILE}.gz"

# Clean up backups older than 7 days
find "${BACKUP_DIR}" -name "collision_db_*.sql.gz" -mtime +7 -delete
echo "Old backups cleanup completed."
