# COLLISION PostgreSQL Backup & Disaster Recovery Guide

This directory contains utility scripts to back up and restore the production PostgreSQL database.

## 1. Automated Backups (Cron Job)
To set up daily database backups at 02:00 AM, register the backup script in the host crontab:
1. Open the crontab editor:
   ```bash
   crontab -e
   ```
2. Append the cron job execution rule:
   ```text
   0 2 * * * /bin/bash /path/to/collision/deployment/backup/backup.sh >> /var/log/collision_backup.log 2>&1
   ```

## 2. Manual Backup
Execute the backup script directly:
```bash
bash deployment/backup/backup.sh
```
Backups are compressed with gzip and stored in `/var/backups/collision/` (retaining the last 7 days of backups).

## 3. Disaster Recovery (Restore)
To restore a specific backup file to the database container:
```bash
bash deployment/backup/restore.sh /var/backups/collision/collision_db_20260830_020000.sql.gz
```
> ⚠️ **Warning**: Restoring drops the existing database and replaces all tables. Verify your target backup timestamp before running.
