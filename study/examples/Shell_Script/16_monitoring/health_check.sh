#!/usr/bin/env bash
set -euo pipefail

# Health Check and Alerting Script
# Monitors system resources and services, sends alerts on failures
# Designed to run from cron for continuous monitoring

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-/var/log/health_check.log}"
STATE_FILE="${STATE_FILE:-/tmp/health_check.state}"

# Alert thresholds
CPU_THRESHOLD="${CPU_THRESHOLD:-85}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-90}"
DISK_THRESHOLD="${DISK_THRESHOLD:-90}"

# Services to check (process names)
REQUIRED_PROCESSES="${REQUIRED_PROCESSES:-sshd}"

# HTTP endpoints to check (format: name,url,timeout)
HTTP_ENDPOINTS="${HTTP_ENDPOINTS:-}"
# Example: "API,http://localhost:8080/health,5;Frontend,http://localhost:3000,3"

# Webhook URL for alerts (Slack-compatible)
WEBHOOK_URL="${WEBHOOK_URL:-}"

# Platform detection
OS_TYPE=$(uname -s)

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "$@"
}

log_warning() {
    log "WARNING" "$@"
}

log_error() {
    log "ERROR" "$@"
}

log_success() {
    log "SUCCESS" "$@"
}

# ============================================================================
# State Management
# ============================================================================

# Track if we've already alerted for this issue
has_alerted() {
    local key=$1

    if [[ ! -f "$STATE_FILE" ]]; then
        return 1
    fi

    grep -q "^${key}$" "$STATE_FILE"
}

mark_alerted() {
    local key=$1

    mkdir -p "$(dirname "$STATE_FILE")"
    echo "$key" >> "$STATE_FILE"
}

clear_alert() {
    local key=$1

    if [[ -f "$STATE_FILE" ]]; then
        grep -v "^${key}$" "$STATE_FILE" > "${STATE_FILE}.tmp" || true
        mv "${STATE_FILE}.tmp" "$STATE_FILE"
    fi
}

# ============================================================================
# Metric Collection
# ============================================================================

get_cpu_usage() {
    if [[ "$OS_TYPE" == "Linux" ]]; then
        if command -v top &> /dev/null; then
            top -bn2 -d 0.1 | grep '^%Cpu' | tail -n1 | awk '{print int(100 - $8)}'
        else
            echo "0"
        fi
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        top -l 2 -n 0 -F | grep 'CPU usage' | tail -n1 | awk '{print int($3 + $5)}'
    else
        echo "0"
    fi
}

get_memory_usage() {
    if [[ "$OS_TYPE" == "Linux" ]]; then
        if [[ -f /proc/meminfo ]]; then
            local total
            local available
            total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
            available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
            echo $(( (total - available) * 100 / total ))
        else
            echo "0"
        fi
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        local page_size
        local free_pages
        local active_pages
        local inactive_pages
        local wired_pages

        page_size=$(vm_stat | grep 'page size' | awk '{print $8}')
        free_pages=$(vm_stat | grep 'Pages free' | awk '{print $3}' | tr -d '.')
        active_pages=$(vm_stat | grep 'Pages active' | awk '{print $3}' | tr -d '.')
        inactive_pages=$(vm_stat | grep 'Pages inactive' | awk '{print $3}' | tr -d '.')
        wired_pages=$(vm_stat | grep 'Pages wired' | awk '{print $4}' | tr -d '.')

        local used=$((active_pages + inactive_pages + wired_pages))
        local total=$((used + free_pages))

        echo $((used * 100 / total))
    else
        echo "0"
    fi
}

get_disk_usage() {
    df -h / | awk 'NR==2 {print int($5)}'
}

# ============================================================================
# Health Checks
# ============================================================================

check_cpu() {
    local usage
    usage=$(get_cpu_usage)
    local alert_key="cpu_high"

    if [[ $usage -ge $CPU_THRESHOLD ]]; then
        if ! has_alerted "$alert_key"; then
            log_error "CPU usage is high: ${usage}%"
            send_alert "CPU Alert" "CPU usage is ${usage}% (threshold: ${CPU_THRESHOLD}%)" "danger"
            mark_alerted "$alert_key"
        fi
        return 1
    else
        log_info "CPU usage is normal: ${usage}%"
        clear_alert "$alert_key"
        return 0
    fi
}

check_memory() {
    local usage
    usage=$(get_memory_usage)
    local alert_key="memory_high"

    if [[ $usage -ge $MEMORY_THRESHOLD ]]; then
        if ! has_alerted "$alert_key"; then
            log_error "Memory usage is high: ${usage}%"
            send_alert "Memory Alert" "Memory usage is ${usage}% (threshold: ${MEMORY_THRESHOLD}%)" "danger"
            mark_alerted "$alert_key"
        fi
        return 1
    else
        log_info "Memory usage is normal: ${usage}%"
        clear_alert "$alert_key"
        return 0
    fi
}

check_disk() {
    local usage
    usage=$(get_disk_usage)
    local alert_key="disk_full"

    if [[ $usage -ge $DISK_THRESHOLD ]]; then
        if ! has_alerted "$alert_key"; then
            log_error "Disk usage is high: ${usage}%"
            send_alert "Disk Alert" "Disk usage is ${usage}% (threshold: ${DISK_THRESHOLD}%)" "warning"
            mark_alerted "$alert_key"
        fi
        return 1
    else
        log_info "Disk usage is normal: ${usage}%"
        clear_alert "$alert_key"
        return 0
    fi
}

check_process() {
    local process_name=$1
    local alert_key="process_${process_name}"

    if pgrep -x "$process_name" > /dev/null; then
        log_info "Process is running: $process_name"
        clear_alert "$alert_key"
        return 0
    else
        if ! has_alerted "$alert_key"; then
            log_error "Process is not running: $process_name"
            send_alert "Process Alert" "Required process '$process_name' is not running" "danger"
            mark_alerted "$alert_key"
        fi
        return 1
    fi
}

check_http_endpoint() {
    local name=$1
    local url=$2
    local timeout=$3
    local alert_key="http_${name}"

    if command -v curl &> /dev/null; then
        if curl --silent --fail --max-time "$timeout" "$url" > /dev/null 2>&1; then
            log_info "HTTP endpoint is healthy: $name ($url)"
            clear_alert "$alert_key"
            return 0
        else
            if ! has_alerted "$alert_key"; then
                log_error "HTTP endpoint is unhealthy: $name ($url)"
                send_alert "HTTP Alert" "Endpoint '$name' at $url is not responding" "danger"
                mark_alerted "$alert_key"
            fi
            return 1
        fi
    else
        log_warning "curl not available, skipping HTTP check for $name"
        return 0
    fi
}

# ============================================================================
# Alerting
# ============================================================================

send_alert() {
    local title=$1
    local message=$2
    local color=${3:-warning}  # good, warning, danger

    if [[ -z "$WEBHOOK_URL" ]]; then
        log_warning "WEBHOOK_URL not set, skipping alert notification"
        return 0
    fi

    # Slack-compatible JSON payload
    local payload
    payload=$(cat <<EOF
{
  "attachments": [
    {
      "color": "$color",
      "title": "$title",
      "text": "$message",
      "footer": "Health Check on $(hostname)",
      "ts": $(date +%s)
    }
  ]
}
EOF
)

    if command -v curl &> /dev/null; then
        if curl --silent --fail \
                --max-time 10 \
                -X POST \
                -H 'Content-Type: application/json' \
                -d "$payload" \
                "$WEBHOOK_URL" > /dev/null 2>&1; then
            log_info "Alert sent successfully"
        else
            log_error "Failed to send alert via webhook"
        fi
    else
        log_warning "curl not available, cannot send webhook alert"
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log_info "========== Health Check Started =========="

    local checks_passed=0
    local checks_failed=0

    # System resource checks
    if check_cpu; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi

    if check_memory; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi

    if check_disk; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi

    # Process checks
    if [[ -n "$REQUIRED_PROCESSES" ]]; then
        IFS=',' read -ra processes <<< "$REQUIRED_PROCESSES"
        for process in "${processes[@]}"; do
            if check_process "$process"; then
                ((checks_passed++))
            else
                ((checks_failed++))
            fi
        done
    fi

    # HTTP endpoint checks
    if [[ -n "$HTTP_ENDPOINTS" ]]; then
        IFS=';' read -ra endpoints <<< "$HTTP_ENDPOINTS"
        for endpoint in "${endpoints[@]}"; do
            IFS=',' read -r name url timeout <<< "$endpoint"
            if check_http_endpoint "$name" "$url" "$timeout"; then
                ((checks_passed++))
            else
                ((checks_failed++))
            fi
        done
    fi

    # Summary
    log_info "========== Health Check Complete =========="
    log_info "Checks passed: $checks_passed"
    log_info "Checks failed: $checks_failed"

    if [[ $checks_failed -eq 0 ]]; then
        log_success "All health checks passed"
        exit 0
    else
        log_error "Some health checks failed"
        exit 1
    fi
}

main "$@"
