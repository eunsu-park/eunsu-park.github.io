#!/usr/bin/env bash
# =============================================================================
# disaster_recovery.sh - Disaster Recovery Planning and Simulation Script
#
# PURPOSE: Demonstrates DR concepts including backup verification, service
#          health checks, RTO/RPO estimation, and recovery procedure documentation.
#
# USAGE:
#   ./disaster_recovery.sh [--check|--simulate|--report]
#
# MODES:
#   --check     Run all health and backup checks (default)
#   --simulate  Simulate a recovery scenario step-by-step
#   --report    Generate a full DR readiness report
#
# CONCEPTS COVERED:
#   - RTO (Recovery Time Objective): max acceptable downtime
#   - RPO (Recovery Point Objective): max acceptable data loss window
#   - Backup integrity verification
#   - Service dependency mapping
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Color codes for visual severity feedback
# Using ANSI escape sequences that work on most terminals
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# ---------------------------------------------------------------------------
# DR Configuration — in a real environment these would be in a config file
# ---------------------------------------------------------------------------
BACKUP_DIR="${BACKUP_DIR:-/var/backups}"          # Where backups are stored
MAX_BACKUP_AGE_HOURS=24                            # RPO threshold in hours
MIN_BACKUP_SIZE_KB=100                             # Sanity-check minimum size
CRITICAL_SERVICES=("ssh" "cron" "syslog")         # Services that must be running
DB_BACKUP_PATH="/tmp/dr_sim_db_backup.sql"        # Simulated DB backup path
ESTIMATED_RTO_MINUTES=60                           # Target recovery time

PASS=0
WARN=0
FAIL=0

# ---------------------------------------------------------------------------
# Helper: print a labeled result line
# ---------------------------------------------------------------------------
result() {
    local status="$1"  # OK | WARN | FAIL
    local message="$2"
    case "$status" in
        OK)   echo -e "  [${GREEN}OK${RESET}]   $message"; ((PASS++)) ;;
        WARN) echo -e "  [${YELLOW}WARN${RESET}] $message"; ((WARN++)) ;;
        FAIL) echo -e "  [${RED}FAIL${RESET}] $message"; ((FAIL++)) ;;
    esac
}

section() {
    echo ""
    echo -e "${BOLD}${BLUE}=== $1 ===${RESET}"
}

# ---------------------------------------------------------------------------
# CHECK 1: Backup Directory and File Integrity
# DR concept: Backups are useless if they cannot be restored. Verify that
# backup files exist, are recent enough to meet RPO, and are non-empty.
# ---------------------------------------------------------------------------
check_backups() {
    section "Backup Verification (RPO: ${MAX_BACKUP_AGE_HOURS}h)"

    if [[ ! -d "$BACKUP_DIR" ]]; then
        result WARN "Backup directory $BACKUP_DIR not found — using /tmp for simulation"
        BACKUP_DIR="/tmp"
    fi

    # Find the most recent file in the backup directory
    local newest
    newest=$(find "$BACKUP_DIR" -maxdepth 1 -type f -newer /tmp 2>/dev/null | head -1 || true)

    if [[ -z "$newest" ]]; then
        result WARN "No backup files found in $BACKUP_DIR (simulation mode)"
    else
        local age_hours=$(( ( $(date +%s) - $(stat -c%Y "$newest" 2>/dev/null || stat -f%m "$newest") ) / 3600 ))
        local size_kb=$(du -k "$newest" 2>/dev/null | cut -f1)

        [[ $age_hours -le $MAX_BACKUP_AGE_HOURS ]] \
            && result OK "Most recent backup is ${age_hours}h old (within RPO)" \
            || result FAIL "Most recent backup is ${age_hours}h old (exceeds ${MAX_BACKUP_AGE_HOURS}h RPO)"

        [[ ${size_kb:-0} -ge $MIN_BACKUP_SIZE_KB ]] \
            && result OK "Backup file size ${size_kb} KB meets minimum threshold" \
            || result WARN "Backup file size ${size_kb} KB is below minimum (${MIN_BACKUP_SIZE_KB} KB)"
    fi
}

# ---------------------------------------------------------------------------
# CHECK 2: Critical Service Health
# DR concept: Know which services are essential before a disaster; document
# their startup order and dependencies (the "service dependency map").
# ---------------------------------------------------------------------------
check_services() {
    section "Critical Service Health Checks"

    for svc in "${CRITICAL_SERVICES[@]}"; do
        # Use 'pgrep' as a portable fallback when 'systemctl' is unavailable
        if command -v systemctl &>/dev/null; then
            systemctl is-active --quiet "$svc" 2>/dev/null \
                && result OK "Service '$svc' is active" \
                || result WARN "Service '$svc' is not active (may not exist in this environment)"
        else
            pgrep -x "$svc" &>/dev/null \
                && result OK "Process '$svc' is running" \
                || result WARN "Process '$svc' not found (may not apply to this OS)"
        fi
    done
}

# ---------------------------------------------------------------------------
# CHECK 3: Database Backup Verification
# DR concept: Database backups require special handling — a file that exists
# does not mean data is recoverable. Validate the dump format.
# ---------------------------------------------------------------------------
check_database_backup() {
    section "Database Backup Verification (PostgreSQL simulation)"

    # Simulate creating a pg_dump; real usage: pg_dump -Fc mydb > backup.dump
    echo "-- PostgreSQL DR test dump $(date)" > "$DB_BACKUP_PATH"
    echo "-- Tables: users, orders, inventory" >> "$DB_BACKUP_PATH"

    if [[ -f "$DB_BACKUP_PATH" ]]; then
        result OK "Database backup file exists at $DB_BACKUP_PATH"
        # Check that the file begins with a recognizable header
        if head -1 "$DB_BACKUP_PATH" | grep -q "PostgreSQL"; then
            result OK "Backup file header matches expected format"
        else
            result WARN "Backup header format unexpected — verify with pg_restore --list"
        fi
    else
        result FAIL "Database backup not found — RPO violation risk"
    fi
}

# ---------------------------------------------------------------------------
# CHECK 4: Network Connectivity
# DR concept: Recovery often requires network access to pull backups from
# remote storage or reach a standby site.
# ---------------------------------------------------------------------------
check_network() {
    section "Network Connectivity Checks"

    local targets=("8.8.8.8" "1.1.1.1")
    for host in "${targets[@]}"; do
        ping -c 1 -W 2 "$host" &>/dev/null \
            && result OK "Reachable: $host" \
            || result WARN "Unreachable: $host (check firewall / DNS)"
    done

    # Verify DNS resolution is functional
    if host google.com &>/dev/null 2>&1 || nslookup google.com &>/dev/null 2>&1; then
        result OK "DNS resolution is working"
    else
        result WARN "DNS resolution failed — recovery from remote storage may be impacted"
    fi
}

# ---------------------------------------------------------------------------
# CHECK 5: Disk Space
# DR concept: Running out of disk space during recovery is a common failure
# mode. Ensure the recovery target volume has sufficient headroom.
# ---------------------------------------------------------------------------
check_disk_space() {
    section "Disk Space Monitoring"

    while IFS= read -r line; do
        local use pct mount
        use=$(echo "$line" | awk '{print $5}' | tr -d '%')
        mount=$(echo "$line" | awk '{print $6}')
        [[ -z "$use" || ! "$use" =~ ^[0-9]+$ ]] && continue

        if   [[ $use -ge 90 ]]; then result FAIL "Disk $mount at ${use}% — critical, recovery may fail"
        elif [[ $use -ge 75 ]]; then result WARN "Disk $mount at ${use}% — low space, monitor closely"
        else                         result OK   "Disk $mount at ${use}% — sufficient free space"
        fi
    done < <(df -h | tail -n +2)
}

# ---------------------------------------------------------------------------
# SIMULATE: Walk through recovery steps interactively
# DR concept: A DR plan that has never been rehearsed is not a plan.
# Tabletop exercises and simulated failovers validate procedures.
# ---------------------------------------------------------------------------
simulate_recovery() {
    section "Recovery Simulation (Tabletop Exercise)"
    echo -e "${YELLOW}Simulating disaster scenario: Primary database host failure${RESET}"
    echo ""

    local steps=(
        "DETECT  | Monitor alerts fire; on-call engineer paged via PagerDuty"
        "ASSESS  | Confirm outage scope — DB host unreachable, app layer affected"
        "DECLARE | Incident declared; DR team assembled; stakeholders notified"
        "FAILOVER| Promote read-replica to primary (pg_promote / Route 53 update)"
        "RESTORE | If no replica: restore latest pg_dump to standby host"
        "VERIFY  | Run smoke tests; confirm application connectivity"
        "CUTOVER | Update load balancer; route traffic to recovered instance"
        "MONITOR | Watch error rates and latency for 30 min post-recovery"
        "REVIEW  | Schedule post-mortem within 48 h; update runbook"
    )

    local step_num=1
    for step in "${steps[@]}"; do
        printf "  ${BOLD}Step %d${RESET}: %s\n" "$step_num" "$step"
        ((step_num++))
        sleep 0.2
    done

    echo ""
    echo -e "  ${GREEN}Estimated RTO for this scenario: ${ESTIMATED_RTO_MINUTES} minutes${RESET}"
    echo -e "  ${YELLOW}Actual RTO depends on backup restore speed (~1 GB/min typical)${RESET}"
}

# ---------------------------------------------------------------------------
# REPORT: Summarize all checks and DR readiness
# ---------------------------------------------------------------------------
generate_report() {
    PASS=0; WARN=0; FAIL=0
    check_backups
    check_services
    check_database_backup
    check_network
    check_disk_space

    section "DR Readiness Summary"
    echo -e "  ${GREEN}PASS: $PASS${RESET}  ${YELLOW}WARN: $WARN${RESET}  ${RED}FAIL: $FAIL${RESET}"
    echo ""

    if [[ $FAIL -gt 0 ]]; then
        echo -e "  ${RED}${BOLD}DR READINESS: NOT READY — $FAIL critical issue(s) must be resolved${RESET}"
        exit 1
    elif [[ $WARN -gt 0 ]]; then
        echo -e "  ${YELLOW}${BOLD}DR READINESS: PARTIAL — review $WARN warning(s) before next drill${RESET}"
    else
        echo -e "  ${GREEN}${BOLD}DR READINESS: GOOD — all checks passed${RESET}"
    fi
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
main() {
    echo -e "${BOLD}Disaster Recovery Script — $(date '+%Y-%m-%d %H:%M:%S')${RESET}"

    case "${1:---check}" in
        --check)    check_backups; check_services; check_database_backup; check_network; check_disk_space ;;
        --simulate) simulate_recovery ;;
        --report)   generate_report ;;
        *)
            echo "Usage: $0 [--check|--simulate|--report]"
            exit 1
            ;;
    esac
}

main "$@"
