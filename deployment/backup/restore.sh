#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/backup.sql.gz"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "${BACKUP_FILE}" ]; then
    echo "Error: Backup file not found at ${BACKUP_FILE}"
    exit 1
fi

TEMP_SQL="/tmp/collision_restore.sql"

echo "Extracting backup to temporary space..."
gunzip -c "${BACKUP_FILE}" > "${TEMP_SQL}"

echo "Restoring database from ${TEMP_SQL}..."
# Drop existing connections and restore
docker exec -i collision-db psql -U collision_admin -d postgres -c "DROP DATABASE IF EXISTS collision_api;"
docker exec -i collision-db psql -U collision_admin -d postgres -c "CREATE DATABASE collision_api;"
docker exec -i collision-db psql -U collision_admin -d collision_api < "${TEMP_SQL}"

rm -f "${TEMP_SQL}"
echo "Database restore completed successfully."
