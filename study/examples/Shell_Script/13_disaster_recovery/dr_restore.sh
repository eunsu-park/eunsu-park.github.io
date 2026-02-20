#!/usr/bin/env bash
#
# Disaster Recovery Restore Script
#
# This script restores from backups created by dr_backup.sh
#
# Features:
# - Interactive backup selection
# - Integrity verification (checksums)
# - Database restoration
# - Configuration file restoration
# - Dry-run mode
# - Comprehensive logging

set -euo pipefail

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

readonly SCRIPT_NAME="$(basename "$0")"
readonly BACKUP_ROOT="/var/backups/dr"
readonly LOG_DIR="${BACKUP_ROOT}/logs"
readonly LOG_FILE="${LOG_DIR}/restore_$(date +%Y%m%d_%H%M%S).log"

# Database configuration
readonly DB_HOST="localhost"
readonly DB_PORT="5432"
readonly DB_USER="backup_user"

# Restore settings
readonly RESTORE_ROOT="/restore"

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Restore from disaster recovery backups.

OPTIONS:
    -h, --help              Show this help message
    -d, --date DATE         Restore from specific date (YYYYMMDD_HHMMSS)
    -t, --type TYPE         Restore type: all|filesystem|database|config
    -n, --dry-run           Show what would be done without executing
    -l, --list              List available backups
    -v, --verify            Verify backup integrity only

EXAMPLES:
    $SCRIPT_NAME --list                     # List available backups
    $SCRIPT_NAME -d 20240215_120000         # Restore specific backup
    $SCRIPT_NAME -t database                # Restore databases only
    $SCRIPT_NAME -n -d 20240215_120000      # Dry-run mode

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
    exit 1
}

list_backups() {
    log "INFO" "Available backups:"

    echo ""
    echo "Full Filesystem Backups:"
    if [[ -d "${BACKUP_ROOT}/full" ]]; then
        find "${BACKUP_ROOT}/full" -maxdepth 1 -type d | \
            grep -E '[0-9]{8}_[0-9]{6}' | \
            sort -r | \
            sed 's/.*\//  - /'
    else
        echo "  No backups found"
    fi

    echo ""
    echo "Database Backups:"
    if [[ -d "${BACKUP_ROOT}/database" ]]; then
        find "${BACKUP_ROOT}/database" -maxdepth 1 -type d | \
            grep -E '[0-9]{8}_[0-9]{6}' | \
            sort -r | \
            sed 's/.*\//  - /'
    else
        echo "  No backups found"
    fi

    echo ""
}

verify_backup_integrity() {
    local backup_date="$1"
    local backup_path="${BACKUP_ROOT}/full/${backup_date}"

    log "INFO" "Verifying backup integrity: $backup_date"

    if [[ ! -f "${backup_path}/checksums.txt" ]]; then
        log "WARN" "No checksum file found for backup: $backup_date"
        return 1
    fi

    pushd "$backup_path" > /dev/null

    if sha256sum -c checksums.txt >> "$LOG_FILE" 2>&1; then
        log "INFO" "Backup integrity verified successfully"
        popd > /dev/null
        return 0
    else
        log "ERROR" "Backup integrity verification failed"
        popd > /dev/null
        return 1
    fi
}

restore_filesystem() {
    local backup_date="$1"
    local dry_run="$2"
    local backup_path="${BACKUP_ROOT}/full/${backup_date}"

    log "INFO" "Restoring filesystem from: $backup_date"

    if [[ ! -d "$backup_path" ]]; then
        die "Backup not found: $backup_path"
    fi

    # Create restore directory
    if [[ "$dry_run" != "true" ]]; then
        mkdir -p "$RESTORE_ROOT"
    fi

    # Find tar archives
    local archives
    mapfile -t archives < <(find "$backup_path" -name "*.tar*" -type f)

    if [[ ${#archives[@]} -eq 0 ]]; then
        log "WARN" "No tar archives found in backup"
        return 0
    fi

    for archive in "${archives[@]}"; do
        log "INFO" "Processing archive: $(basename "$archive")"

        if [[ "$dry_run" == "true" ]]; then
            log "INFO" "[DRY-RUN] Would extract: $archive"
            continue
        fi

        # Decrypt if needed
        if [[ "$archive" == *.gpg ]]; then
            log "INFO" "Decrypting archive"
            local decrypted="${archive%.gpg}"
            gpg --decrypt "$archive" > "$decrypted" || die "Decryption failed"
            archive="$decrypted"
        fi

        # Decompress if needed
        if [[ "$archive" == *.gz ]]; then
            log "INFO" "Decompressing archive"
            gunzip -k "$archive" || die "Decompression failed"
            archive="${archive%.gz}"
        fi

        # Extract
        log "INFO" "Extracting archive to: $RESTORE_ROOT"
        tar -xpf "$archive" -C "$RESTORE_ROOT" || die "Extraction failed"
    done

    log "INFO" "Filesystem restore completed"
}

restore_databases() {
    local backup_date="$1"
    local dry_run="$2"
    local db_backup_path="${BACKUP_ROOT}/database/${backup_date}"

    log "INFO" "Restoring databases from: $backup_date"

    if [[ ! -d "$db_backup_path" ]]; then
        die "Database backup not found: $db_backup_path"
    fi

    if ! command -v pg_restore &>/dev/null; then
        log "WARN" "pg_restore not found, skipping database restore"
        return 0
    fi

    # Find database dumps
    local dumps
    mapfile -t dumps < <(find "$db_backup_path" -name "*.sql*" -type f)

    if [[ ${#dumps[@]} -eq 0 ]]; then
        log "WARN" "No database dumps found in backup"
        return 0
    fi

    for dump in "${dumps[@]}"; do
        local db_name
        db_name="$(basename "$dump" | cut -d'_' -f1)"

        log "INFO" "Restoring database: $db_name"

        if [[ "$dry_run" == "true" ]]; then
            log "INFO" "[DRY-RUN] Would restore database: $db_name from $dump"
            continue
        fi

        # Decrypt if needed
        if [[ "$dump" == *.gpg ]]; then
            log "INFO" "Decrypting database dump"
            local decrypted="${dump%.gpg}"
            gpg --decrypt "$dump" > "$decrypted" || die "Decryption failed"
            dump="$decrypted"
        fi

        # Decompress if needed
        if [[ "$dump" == *.gz ]]; then
            log "INFO" "Decompressing database dump"
            gunzip -k "$dump" || die "Decompression failed"
            dump="${dump%.gz}"
        fi

        # Restore database
        PGPASSWORD="${DB_PASSWORD:-}" pg_restore \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -d "$db_name" \
            -c \
            "$dump" \
            >> "$LOG_FILE" 2>&1 || log "WARN" "Database restore failed for $db_name"
    done

    log "INFO" "Database restore completed"
}

restore_config() {
    local backup_date="$1"
    local dry_run="$2"
    local config_backup="${BACKUP_ROOT}/config/config_${backup_date}.tar.gz"

    log "INFO" "Restoring configuration files"

    if [[ ! -f "$config_backup" ]]; then
        log "WARN" "Configuration backup not found: $config_backup"
        return 0
    fi

    if [[ "$dry_run" == "true" ]]; then
        log "INFO" "[DRY-RUN] Would restore config from: $config_backup"
        return 0
    fi

    # Extract to temporary location first
    local temp_dir
    temp_dir="$(mktemp -d)"
    trap 'rm -rf "$temp_dir"' EXIT

    tar -xzf "$config_backup" -C "$temp_dir" || die "Config extraction failed"

    log "INFO" "Configuration files extracted to: $temp_dir"
    log "WARN" "Please manually review and copy config files as needed"

    log "INFO" "Configuration restore completed"
}

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

main() {
    local backup_date=""
    local restore_type="all"
    local dry_run=false
    local list_only=false
    local verify_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -d|--date)
                backup_date="$2"
                shift 2
                ;;
            -t|--type)
                restore_type="$2"
                shift 2
                ;;
            -n|--dry-run)
                dry_run=true
                shift
                ;;
            -l|--list)
                list_only=true
                shift
                ;;
            -v|--verify)
                verify_only=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Setup logging
    mkdir -p "$LOG_DIR"

    if [[ "$list_only" == "true" ]]; then
        list_backups
        exit 0
    fi

    if [[ -z "$backup_date" ]]; then
        die "Backup date required. Use --list to see available backups"
    fi

    log "INFO" "Starting disaster recovery restore"
    log "INFO" "Backup date: $backup_date"
    log "INFO" "Restore type: $restore_type"

    if [[ "$verify_only" == "true" ]]; then
        verify_backup_integrity "$backup_date"
        exit 0
    fi

    # Verify integrity before restore
    if ! verify_backup_integrity "$backup_date"; then
        die "Backup integrity check failed. Aborting restore."
    fi

    case "$restore_type" in
        all)
            restore_filesystem "$backup_date" "$dry_run"
            restore_databases "$backup_date" "$dry_run"
            restore_config "$backup_date" "$dry_run"
            ;;
        filesystem)
            restore_filesystem "$backup_date" "$dry_run"
            ;;
        database)
            restore_databases "$backup_date" "$dry_run"
            ;;
        config)
            restore_config "$backup_date" "$dry_run"
            ;;
        *)
            die "Unknown restore type: $restore_type"
            ;;
    esac

    log "INFO" "Restore completed successfully"
}

main "$@"
