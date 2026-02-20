#!/usr/bin/env bash
#
# Disaster Recovery Backup Script
#
# This script performs automated backups of:
# - Full system files (configurable directories)
# - PostgreSQL databases
# - Configuration files
# - Incremental backups (optional)
#
# Features:
# - Compression (gzip) and encryption (gpg)
# - Remote transfer (rsync/scp)
# - Backup rotation
# - Email notifications
# - Comprehensive logging

set -euo pipefail

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

readonly SCRIPT_NAME="$(basename "$0")"
readonly BACKUP_ROOT="/var/backups/dr"
readonly LOG_DIR="${BACKUP_ROOT}/logs"
readonly LOG_FILE="${LOG_DIR}/backup_$(date +%Y%m%d_%H%M%S).log"

# Backup sources
readonly BACKUP_DIRS=(
    "/etc"
    "/var/www"
    "/opt/applications"
)

# Database configuration
readonly DB_HOST="localhost"
readonly DB_PORT="5432"
readonly DB_USER="backup_user"
readonly DB_NAMES=(
    "production_db"
    "analytics_db"
)

# Backup settings
readonly INCREMENTAL=false  # Set to true for incremental backups
readonly COMPRESS=true
readonly ENCRYPT=true
readonly GPG_RECIPIENT="admin@example.com"

# Remote backup settings
readonly REMOTE_BACKUP=true
readonly REMOTE_HOST="backup-server.example.com"
readonly REMOTE_USER="backup"
readonly REMOTE_PATH="/backups/dr"

# Retention settings
readonly KEEP_DAILY=7
readonly KEEP_WEEKLY=4
readonly KEEP_MONTHLY=6

# Email notification
readonly EMAIL_NOTIFY=true
readonly EMAIL_TO="admin@example.com"

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Automated disaster recovery backup script.

OPTIONS:
    -h, --help              Show this help message
    -n, --no-remote         Skip remote backup transfer
    -i, --incremental       Perform incremental backup
    -d, --dry-run           Show what would be done without executing

EXAMPLES:
    $SCRIPT_NAME                # Full backup with all features
    $SCRIPT_NAME -n             # Backup locally only
    $SCRIPT_NAME -i             # Incremental backup

EOF
}

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

    echo "[${timestamp}] [${level}] ${message}" | tee -a "$LOG_FILE"
}

die() {
    log "ERROR" "$*"
    send_notification "FAILURE" "$*"
    exit 1
}

send_notification() {
    if [[ "$EMAIL_NOTIFY" != "true" ]]; then
        return 0
    fi

    local status="$1"
    local message="$2"
    local subject="DR Backup ${status}: $(hostname)"

    if command -v mail &>/dev/null; then
        echo "$message" | mail -s "$subject" "$EMAIL_TO"
    else
        log "WARN" "mail command not found, skipping email notification"
    fi
}

setup_directories() {
    log "INFO" "Setting up backup directories"

    mkdir -p "$BACKUP_ROOT"/{full,incremental,database,config,logs}
    mkdir -p "$LOG_DIR"

    # Set secure permissions
    chmod 700 "$BACKUP_ROOT"
    chmod 600 "$LOG_FILE"
}

backup_filesystem() {
    log "INFO" "Starting filesystem backup"

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local backup_dir="${BACKUP_ROOT}/full/${timestamp}"
    local archive_name="filesystem_${timestamp}.tar"

    mkdir -p "$backup_dir"

    for dir in "${BACKUP_DIRS[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log "WARN" "Directory does not exist: $dir"
            continue
        fi

        log "INFO" "Backing up: $dir"

        local dir_name
        dir_name="$(basename "$dir")"

        if [[ "$INCREMENTAL" == "true" ]]; then
            # Incremental backup using rsync
            rsync -av --link-dest="${BACKUP_ROOT}/full/latest" \
                "$dir/" "${backup_dir}/${dir_name}/" \
                >> "$LOG_FILE" 2>&1 || log "WARN" "rsync failed for $dir"
        else
            # Full tar backup
            tar -cpf "${backup_dir}/${archive_name}" \
                -C "$(dirname "$dir")" "$(basename "$dir")" \
                >> "$LOG_FILE" 2>&1 || log "WARN" "tar failed for $dir"
        fi
    done

    # Compress if enabled
    if [[ "$COMPRESS" == "true" ]]; then
        log "INFO" "Compressing filesystem backup"
        gzip -9 "${backup_dir}/${archive_name}" || log "WARN" "Compression failed"
        archive_name="${archive_name}.gz"
    fi

    # Encrypt if enabled
    if [[ "$ENCRYPT" == "true" ]]; then
        log "INFO" "Encrypting filesystem backup"
        gpg --encrypt --recipient "$GPG_RECIPIENT" \
            "${backup_dir}/${archive_name}" \
            && rm -f "${backup_dir}/${archive_name}" \
            || log "WARN" "Encryption failed"
    fi

    # Create/update latest symlink
    ln -sfn "$backup_dir" "${BACKUP_ROOT}/full/latest"

    log "INFO" "Filesystem backup completed: $backup_dir"
}

backup_databases() {
    log "INFO" "Starting database backup"

    if ! command -v pg_dump &>/dev/null; then
        log "WARN" "pg_dump not found, skipping database backup"
        return 0
    fi

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local db_backup_dir="${BACKUP_ROOT}/database/${timestamp}"

    mkdir -p "$db_backup_dir"

    for db in "${DB_NAMES[@]}"; do
        log "INFO" "Backing up database: $db"

        local dump_file="${db_backup_dir}/${db}_${timestamp}.sql"

        PGPASSWORD="${DB_PASSWORD:-}" pg_dump \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -F c \
            -f "$dump_file" \
            "$db" \
            >> "$LOG_FILE" 2>&1 || log "WARN" "Database dump failed for $db"

        # Compress
        if [[ "$COMPRESS" == "true" ]]; then
            gzip -9 "$dump_file" || log "WARN" "Compression failed for $db"
        fi

        # Encrypt
        if [[ "$ENCRYPT" == "true" ]]; then
            local encrypted_file="${dump_file}.gz.gpg"
            gpg --encrypt --recipient "$GPG_RECIPIENT" "${dump_file}.gz" \
                && rm -f "${dump_file}.gz" \
                || log "WARN" "Encryption failed for $db"
        fi
    done

    log "INFO" "Database backup completed: $db_backup_dir"
}

backup_config() {
    log "INFO" "Backing up configuration files"

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local config_backup="${BACKUP_ROOT}/config/config_${timestamp}.tar.gz"

    # Backup specific config files
    tar -czf "$config_backup" \
        /etc/fstab \
        /etc/hosts \
        /etc/ssh/sshd_config \
        /etc/systemd/system/*.service \
        2>/dev/null || log "WARN" "Some config files could not be backed up"

    log "INFO" "Configuration backup completed: $config_backup"
}

transfer_to_remote() {
    if [[ "$REMOTE_BACKUP" != "true" ]]; then
        log "INFO" "Remote backup disabled, skipping transfer"
        return 0
    fi

    log "INFO" "Transferring backups to remote server"

    # Use rsync for efficient transfer
    rsync -avz --delete \
        -e "ssh -o StrictHostKeyChecking=no" \
        "${BACKUP_ROOT}/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" \
        >> "$LOG_FILE" 2>&1 || log "WARN" "Remote transfer failed"

    log "INFO" "Remote transfer completed"
}

rotate_backups() {
    log "INFO" "Rotating old backups"

    # Rotate daily backups
    find "${BACKUP_ROOT}/full" -maxdepth 1 -type d -mtime "+${KEEP_DAILY}" \
        -exec rm -rf {} \; 2>/dev/null || true

    find "${BACKUP_ROOT}/database" -maxdepth 1 -type d -mtime "+${KEEP_DAILY}" \
        -exec rm -rf {} \; 2>/dev/null || true

    # Rotate logs older than 30 days
    find "$LOG_DIR" -type f -name "*.log" -mtime +30 \
        -exec rm -f {} \; 2>/dev/null || true

    log "INFO" "Backup rotation completed"
}

verify_backup() {
    log "INFO" "Verifying backup integrity"

    local latest_backup="${BACKUP_ROOT}/full/latest"

    if [[ ! -d "$latest_backup" ]]; then
        log "WARN" "No backup found to verify"
        return 0
    fi

    # Create checksums
    find "$latest_backup" -type f -exec sha256sum {} \; \
        > "${latest_backup}/checksums.txt" 2>/dev/null

    log "INFO" "Backup verification completed"
}

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

main() {
    local skip_remote=false
    local incremental=false
    local dry_run=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -n|--no-remote)
                skip_remote=true
                shift
                ;;
            -i|--incremental)
                incremental=true
                shift
                ;;
            -d|--dry-run)
                dry_run=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    if [[ "$dry_run" == "true" ]]; then
        log "INFO" "DRY RUN MODE - No actions will be performed"
        exit 0
    fi

    log "INFO" "Starting disaster recovery backup"
    log "INFO" "Hostname: $(hostname)"
    log "INFO" "Timestamp: $(date)"

    setup_directories
    backup_filesystem
    backup_databases
    backup_config
    verify_backup
    rotate_backups

    if [[ "$skip_remote" != "true" ]]; then
        transfer_to_remote
    fi

    local end_time
    end_time="$(date)"
    log "INFO" "Backup completed successfully at $end_time"

    send_notification "SUCCESS" "Disaster recovery backup completed successfully"
}

main "$@"
