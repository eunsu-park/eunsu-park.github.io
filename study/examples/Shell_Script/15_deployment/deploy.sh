#!/usr/bin/env bash
set -euo pipefail

# Deployment Automation Script
# Supports rolling deployments, rollback, and health checks
# Can deploy to staging or production environments

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration (can be overridden by deploy.conf or environment)
APP_NAME="${APP_NAME:-myapp}"
DEPLOY_USER="${DEPLOY_USER:-deploy}"
DEPLOY_PATH="${DEPLOY_PATH:-/var/www/apps}"
KEEP_RELEASES="${KEEP_RELEASES:-5}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-30}"
HEALTH_CHECK_PATH="${HEALTH_CHECK_PATH:-/health}"

# Environment-specific server lists
STAGING_SERVERS="${STAGING_SERVERS:-staging1.example.com}"
PRODUCTION_SERVERS="${PRODUCTION_SERVERS:-prod1.example.com,prod2.example.com,prod3.example.com}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ✓ $*"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ✗ $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ⚠ $*"
}

log_section() {
    echo
    echo -e "${CYAN}${BOLD}==> $*${NC}"
}

# ============================================================================
# Configuration Loading
# ============================================================================

load_config() {
    local config_file="$SCRIPT_DIR/deploy.conf"

    if [[ -f "$config_file" ]]; then
        log_info "Loading configuration from $config_file"
        # shellcheck disable=SC1090
        source "$config_file"
    fi
}

get_servers() {
    local env=$1

    case "$env" in
        staging)
            echo "$STAGING_SERVERS"
            ;;
        production)
            echo "$PRODUCTION_SERVERS"
            ;;
        *)
            log_error "Invalid environment: $env"
            exit 1
            ;;
    esac
}

# ============================================================================
# Remote Execution Functions
# ============================================================================

remote_exec() {
    local server=$1
    shift
    local command="$*"

    ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        "${DEPLOY_USER}@${server}" \
        "$command"
}

sync_files() {
    local server=$1
    local source=$2
    local dest=$3

    rsync -avz --delete \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        "$source" \
        "${DEPLOY_USER}@${server}:${dest}"
}

# ============================================================================
# Deployment Functions
# ============================================================================

prepare_release() {
    local release_name
    release_name=$(date +%Y%m%d-%H%M%S)
    echo "$release_name"
}

create_release_structure() {
    local server=$1
    local release=$2

    log_info "Creating release structure on $server"

    remote_exec "$server" "
        mkdir -p ${DEPLOY_PATH}/${APP_NAME}/{releases,shared}
        mkdir -p ${DEPLOY_PATH}/${APP_NAME}/releases/${release}
    "
}

upload_release() {
    local server=$1
    local release=$2
    local source_dir=$3

    log_info "Uploading release to $server"

    # In real scenario, this would upload actual built artifacts
    # For demo, we'll create a simple structure
    remote_exec "$server" "
        cat > ${DEPLOY_PATH}/${APP_NAME}/releases/${release}/app.sh << 'EOF'
#!/bin/bash
echo 'Application version: ${release}'
echo 'Server: \$(hostname)'
echo 'Status: Running'
EOF
        chmod +x ${DEPLOY_PATH}/${APP_NAME}/releases/${release}/app.sh
    "
}

link_shared_resources() {
    local server=$1
    local release=$2

    log_info "Linking shared resources on $server"

    remote_exec "$server" "
        # Create shared directories if they don't exist
        mkdir -p ${DEPLOY_PATH}/${APP_NAME}/shared/{config,logs,data}

        # Link shared resources into release
        ln -sf ${DEPLOY_PATH}/${APP_NAME}/shared/config \
               ${DEPLOY_PATH}/${APP_NAME}/releases/${release}/config
        ln -sf ${DEPLOY_PATH}/${APP_NAME}/shared/logs \
               ${DEPLOY_PATH}/${APP_NAME}/releases/${release}/logs
        ln -sf ${DEPLOY_PATH}/${APP_NAME}/shared/data \
               ${DEPLOY_PATH}/${APP_NAME}/releases/${release}/data
    "
}

switch_current_release() {
    local server=$1
    local release=$2

    log_info "Switching current release on $server"

    remote_exec "$server" "
        ln -sfn ${DEPLOY_PATH}/${APP_NAME}/releases/${release} \
                ${DEPLOY_PATH}/${APP_NAME}/current
    "
}

health_check() {
    local server=$1

    log_info "Running health check on $server"

    # In real scenario, this would check HTTP endpoint
    # For demo, we'll check if the symlink exists and app runs
    if remote_exec "$server" "
        [[ -L ${DEPLOY_PATH}/${APP_NAME}/current ]] && \
        ${DEPLOY_PATH}/${APP_NAME}/current/app.sh > /dev/null 2>&1
    "; then
        log_success "Health check passed on $server"
        return 0
    else
        log_error "Health check failed on $server"
        return 1
    fi
}

cleanup_old_releases() {
    local server=$1

    log_info "Cleaning up old releases on $server"

    remote_exec "$server" "
        cd ${DEPLOY_PATH}/${APP_NAME}/releases
        ls -t | tail -n +$((KEEP_RELEASES + 1)) | xargs -r rm -rf
    "
}

# ============================================================================
# Deployment Commands
# ============================================================================

deploy() {
    local env=$1
    local servers

    IFS=',' read -ra servers <<< "$(get_servers "$env")"

    log_section "Starting deployment to $env environment"
    log_info "Servers: ${servers[*]}"

    local release
    release=$(prepare_release)
    log_info "Release: $release"

    # Deploy to each server in rolling fashion
    for server in "${servers[@]}"; do
        log_section "Deploying to $server"

        # Create structure
        create_release_structure "$server" "$release"

        # Upload files
        upload_release "$server" "$release" "."

        # Link shared resources
        link_shared_resources "$server" "$release"

        # Switch to new release
        switch_current_release "$server" "$release"

        # Health check
        if ! health_check "$server"; then
            log_error "Deployment failed on $server"
            log_warning "Consider rolling back"
            exit 1
        fi

        # Cleanup old releases
        cleanup_old_releases "$server"

        log_success "Deployment successful on $server"

        # Wait between servers in production for rolling deployment
        if [[ "$env" == "production" ]] && [[ "$server" != "${servers[-1]}" ]]; then
            log_info "Waiting 10 seconds before next server..."
            sleep 10
        fi
    done

    log_section "Deployment Complete"
    log_success "Successfully deployed $release to $env"
}

rollback() {
    local env=$1
    local servers

    IFS=',' read -ra servers <<< "$(get_servers "$env")"

    log_section "Starting rollback on $env environment"

    for server in "${servers[@]}"; do
        log_info "Rolling back on $server"

        # Get previous release
        local previous_release
        previous_release=$(remote_exec "$server" "
            cd ${DEPLOY_PATH}/${APP_NAME}/releases
            current=\$(readlink ${DEPLOY_PATH}/${APP_NAME}/current | xargs basename)
            ls -t | grep -v \"\$current\" | head -n1
        ")

        if [[ -z "$previous_release" ]]; then
            log_error "No previous release found on $server"
            continue
        fi

        log_info "Previous release: $previous_release"

        # Switch to previous release
        switch_current_release "$server" "$previous_release"

        # Health check
        if health_check "$server"; then
            log_success "Rollback successful on $server"
        else
            log_error "Rollback failed on $server"
        fi
    done

    log_section "Rollback Complete"
}

status() {
    local env=$1
    local servers

    IFS=',' read -ra servers <<< "$(get_servers "$env")"

    log_section "Status for $env environment"

    for server in "${servers[@]}"; do
        echo
        echo -e "${BOLD}Server: $server${NC}"

        # Get current release
        local current
        current=$(remote_exec "$server" "
            if [[ -L ${DEPLOY_PATH}/${APP_NAME}/current ]]; then
                readlink ${DEPLOY_PATH}/${APP_NAME}/current | xargs basename
            else
                echo 'not deployed'
            fi
        ")

        echo "  Current release: $current"

        # List available releases
        echo "  Available releases:"
        remote_exec "$server" "
            cd ${DEPLOY_PATH}/${APP_NAME}/releases 2>/dev/null && ls -t | head -n5 || echo '    (none)'
        " | sed 's/^/    /'

        # Health check
        if health_check "$server" > /dev/null 2>&1; then
            echo -e "  Health: ${GREEN}OK${NC}"
        else
            echo -e "  Health: ${RED}FAILED${NC}"
        fi
    done

    echo
}

# ============================================================================
# Usage
# ============================================================================

show_usage() {
    cat << EOF
${BOLD}Deployment Automation Script${NC}

${BOLD}Usage:${NC}
  $0 <command> [options]

${BOLD}Commands:${NC}
  deploy   --env <staging|production>    Deploy new release
  rollback --env <staging|production>    Rollback to previous release
  status   --env <staging|production>    Show deployment status

${BOLD}Examples:${NC}
  $0 deploy --env staging
  $0 deploy --env production
  $0 rollback --env production
  $0 status --env staging

EOF
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    load_config

    local command="${1:-}"
    local env="staging"

    # Parse arguments
    shift || true
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --env)
                env="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Validate environment
    if [[ "$env" != "staging" ]] && [[ "$env" != "production" ]]; then
        log_error "Invalid environment: $env"
        show_usage
        exit 1
    fi

    # Execute command
    case "$command" in
        deploy)
            deploy "$env"
            ;;
        rollback)
            rollback "$env"
            ;;
        status)
            status "$env"
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
