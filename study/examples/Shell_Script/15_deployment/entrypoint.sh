#!/usr/bin/env bash
set -euo pipefail

# Docker Entrypoint Script
# Handles initialization, configuration, and graceful shutdown for containerized applications

# ============================================================================
# Configuration
# ============================================================================

APP_NAME="${APP_NAME:-myapp}"
APP_ENV="${APP_ENV:-production}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
DATA_DIR="${DATA_DIR:-/app/data}"

# Colors (for logs)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Logging Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $*"
}

# ============================================================================
# Environment Variable Validation
# ============================================================================

validate_required_vars() {
    log_info "Validating required environment variables..."

    local required_vars=(
        "APP_NAME"
        "APP_ENV"
    )

    local missing_vars=()

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        else
            log_info "  ✓ $var=${!var}"
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        exit 1
    fi

    log_success "All required variables present"
}

validate_optional_vars() {
    log_info "Optional configuration:"

    local optional_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "LOG_LEVEL"
        "MAX_WORKERS"
    )

    for var in "${optional_vars[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            # Mask sensitive values
            local display_value="${!var}"
            if [[ "$var" =~ URL ]]; then
                display_value=$(echo "${!var}" | sed 's/:[^@]*@/:***@/')
            fi
            log_info "  ✓ $var=$display_value"
        fi
    done
}

# ============================================================================
# Configuration Template Processing
# ============================================================================

process_templates() {
    log_info "Processing configuration templates..."

    local template_dir="/app/templates"

    if [[ ! -d "$template_dir" ]]; then
        log_warning "Template directory not found: $template_dir"
        return 0
    fi

    # Create config directory if it doesn't exist
    mkdir -p "$CONFIG_DIR"

    # Process all .template files
    find "$template_dir" -name "*.template" -type f | while read -r template_file; do
        local config_file
        config_file="$CONFIG_DIR/$(basename "${template_file%.template}")"

        log_info "  Processing: $(basename "$template_file") -> $(basename "$config_file")"

        # Use envsubst to replace environment variables
        envsubst < "$template_file" > "$config_file"

        # Set appropriate permissions
        chmod 644 "$config_file"
    done

    log_success "Configuration templates processed"
}

# ============================================================================
# Wait for Dependencies
# ============================================================================

wait_for_service() {
    local host=$1
    local port=$2
    local timeout=${3:-30}

    log_info "Waiting for $host:$port (timeout: ${timeout}s)..."

    local start_time
    start_time=$(date +%s)

    while ! nc -z "$host" "$port" 2>/dev/null; do
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [[ $elapsed -ge $timeout ]]; then
            log_error "Timeout waiting for $host:$port"
            return 1
        fi

        sleep 1
    done

    log_success "$host:$port is available"
}

wait_for_dependencies() {
    log_info "Waiting for service dependencies..."

    # Parse DATABASE_URL if present
    if [[ -n "${DATABASE_URL:-}" ]]; then
        # Extract host and port from URL (simplified)
        # Example: postgresql://user:pass@host:5432/db
        local db_host
        local db_port

        db_host=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
        db_port=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

        if [[ -n "$db_host" ]] && [[ -n "$db_port" ]]; then
            wait_for_service "$db_host" "$db_port" 60
        fi
    fi

    # Parse REDIS_URL if present
    if [[ -n "${REDIS_URL:-}" ]]; then
        local redis_host
        local redis_port

        redis_host=$(echo "$REDIS_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
        redis_port=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\).*/\1/p')

        # Default Redis port if not specified
        redis_port=${redis_port:-6379}

        if [[ -n "$redis_host" ]]; then
            wait_for_service "$redis_host" "$redis_port" 30
        fi
    fi

    log_success "All dependencies are ready"
}

# ============================================================================
# Signal Handling for Graceful Shutdown
# ============================================================================

cleanup() {
    log_warning "Received shutdown signal, cleaning up..."

    # If a child process is running, send it SIGTERM
    if [[ -n "${APP_PID:-}" ]]; then
        log_info "Sending SIGTERM to application (PID: $APP_PID)"
        kill -TERM "$APP_PID" 2>/dev/null || true

        # Wait for graceful shutdown (max 30 seconds)
        local timeout=30
        local elapsed=0

        while kill -0 "$APP_PID" 2>/dev/null && [[ $elapsed -lt $timeout ]]; do
            sleep 1
            elapsed=$((elapsed + 1))
        done

        # Force kill if still running
        if kill -0 "$APP_PID" 2>/dev/null; then
            log_warning "Application didn't stop gracefully, forcing shutdown"
            kill -KILL "$APP_PID" 2>/dev/null || true
        fi
    fi

    log_success "Cleanup complete"
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT SIGQUIT

# ============================================================================
# Application Initialization
# ============================================================================

initialize_app() {
    log_info "Initializing application..."

    # Create necessary directories
    mkdir -p "$DATA_DIR" "$CONFIG_DIR"

    # Set permissions
    chmod 755 "$DATA_DIR" "$CONFIG_DIR"

    # Run any database migrations (if needed)
    if [[ -x "/app/bin/migrate" ]]; then
        log_info "Running database migrations..."
        /app/bin/migrate
    fi

    log_success "Application initialized"
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    log_info "=== Starting $APP_NAME ($APP_ENV) ==="
    echo

    # Step 1: Validate environment
    validate_required_vars
    validate_optional_vars
    echo

    # Step 2: Process configuration templates
    process_templates
    echo

    # Step 3: Wait for dependencies
    wait_for_dependencies
    echo

    # Step 4: Initialize application
    initialize_app
    echo

    # Step 5: Start the application
    log_info "Starting application..."

    # Default command if none provided
    if [[ $# -eq 0 ]]; then
        set -- "/app/bin/server"
    fi

    log_info "Command: $*"
    echo

    # Execute the application
    # Use exec to replace the shell with the application
    # This ensures signals are properly forwarded
    exec "$@" &
    APP_PID=$!

    # Wait for the application process
    wait $APP_PID
}

# Run main with all arguments
main "$@"
