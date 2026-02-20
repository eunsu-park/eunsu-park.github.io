#!/usr/bin/env bash
set -euo pipefail

# Task Runner - Modern build automation for bash
# Discovers and runs tasks defined as task::* functions
# Supports dependencies, help generation, and colored output

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_PREFIX="task::"
EXECUTED_TASKS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} ✓ $*"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')]${NC} ✗ $*" >&2
}

log_task() {
    echo -e "\n${CYAN}${BOLD}==>${NC} ${BOLD}$*${NC}"
}

# ============================================================================
# Task Discovery and Help
# ============================================================================

# Get all defined tasks
list_tasks() {
    declare -F | awk '{print $3}' | grep "^${TASK_PREFIX}" | sed "s/^${TASK_PREFIX}//"
}

# Extract help comment from task function
get_task_help() {
    local task_name=$1
    local func_name="${TASK_PREFIX}${task_name}"

    # Look for ## comment before the function
    local help_text
    help_text=$(awk "/^## /{comment=\$0; sub(/^## /, \"\", comment)}
                     /^${func_name}\(\)/{if(comment) print comment; comment=\"\"}" "${BASH_SOURCE[0]}")

    echo "${help_text:-No description available}"
}

# Show usage information
show_usage() {
    cat << EOF
${BOLD}Task Runner${NC} - Build automation for bash projects

${BOLD}Usage:${NC}
  ./task.sh [options] <task> [<task>...]

${BOLD}Options:${NC}
  -h, --help     Show this help message
  -l, --list     List all available tasks

${BOLD}Available Tasks:${NC}
EOF

    local tasks
    tasks=$(list_tasks)

    if [[ -z "$tasks" ]]; then
        echo "  (no tasks defined)"
        return
    fi

    while IFS= read -r task; do
        local help
        help=$(get_task_help "$task")
        printf "  ${GREEN}%-15s${NC} %s\n" "$task" "$help"
    done <<< "$tasks"

    echo
    echo "${BOLD}Examples:${NC}"
    echo "  ./task.sh build        # Run the build task"
    echo "  ./task.sh clean build  # Run multiple tasks"
    echo "  ./task.sh deploy       # Run task with dependencies"
}

# ============================================================================
# Dependency Management
# ============================================================================

# Declare task dependencies
depends_on() {
    for dep in "$@"; do
        if ! task_exists "$dep"; then
            log_error "Dependency not found: $dep"
            exit 1
        fi

        if ! has_executed "$dep"; then
            log_info "Running dependency: $dep"
            run_task "$dep"
        fi
    done
}

# Check if task exists
task_exists() {
    local task_name=$1
    declare -f "${TASK_PREFIX}${task_name}" > /dev/null
}

# Check if task has been executed
has_executed() {
    local task_name=$1
    for executed in "${EXECUTED_TASKS[@]}"; do
        if [[ "$executed" == "$task_name" ]]; then
            return 0
        fi
    done
    return 1
}

# Mark task as executed
mark_executed() {
    local task_name=$1
    EXECUTED_TASKS+=("$task_name")
}

# ============================================================================
# Task Execution
# ============================================================================

# Run a single task
run_task() {
    local task_name=$1
    local func_name="${TASK_PREFIX}${task_name}"

    if ! task_exists "$task_name"; then
        log_error "Task not found: $task_name"
        log_info "Run './task.sh --list' to see available tasks"
        exit 1
    fi

    if has_executed "$task_name"; then
        log_info "Task already executed: $task_name (skipping)"
        return 0
    fi

    log_task "Running task: $task_name"

    local start_time
    start_time=$(date +%s)

    # Execute the task
    if "$func_name"; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))

        mark_executed "$task_name"
        log_success "Task completed: $task_name (${duration}s)"
    else
        log_error "Task failed: $task_name"
        exit 1
    fi
}

# ============================================================================
# Task Definitions
# ============================================================================

## Clean build artifacts and temporary files
task::clean() {
    log_info "Removing build artifacts..."

    # Simulate cleaning
    rm -rf "$SCRIPT_DIR/dist" "$SCRIPT_DIR/build" "$SCRIPT_DIR"/*.log || true
    mkdir -p "$SCRIPT_DIR/dist" "$SCRIPT_DIR/build"

    log_info "Clean complete"
}

## Install project dependencies
task::deps() {
    log_info "Installing dependencies..."

    # Simulate dependency installation
    local deps=("shellcheck" "bats" "jq")

    for dep in "${deps[@]}"; do
        if command -v "$dep" &> /dev/null; then
            log_info "✓ $dep already installed"
        else
            log_info "✗ $dep not found (would install)"
        fi
    done

    log_info "Dependencies checked"
}

## Run linting checks (ShellCheck)
task::lint() {
    depends_on deps

    log_info "Running ShellCheck..."

    if command -v shellcheck &> /dev/null; then
        if shellcheck "$0"; then
            log_info "Lint passed"
        else
            log_error "Lint failed"
            return 1
        fi
    else
        log_info "ShellCheck not installed, skipping"
    fi
}

## Run unit tests
task::test() {
    depends_on deps lint

    log_info "Running tests..."

    # Simulate test execution
    local test_files=("utils" "config" "main")
    local passed=0
    local total=${#test_files[@]}

    for test in "${test_files[@]}"; do
        log_info "Testing $test..."
        sleep 0.2

        # Simulate random test result
        if [[ $((RANDOM % 10)) -gt 1 ]]; then
            ((passed++))
        fi
    done

    log_info "Tests: $passed/$total passed"

    if [[ $passed -eq $total ]]; then
        return 0
    else
        log_error "Some tests failed"
        return 1
    fi
}

## Build the project
task::build() {
    depends_on clean test

    log_info "Building project..."

    # Simulate build steps
    log_info "Compiling sources..."
    sleep 0.3
    echo "#!/bin/bash" > "$SCRIPT_DIR/build/app"
    echo "echo 'Hello from built app'" >> "$SCRIPT_DIR/build/app"
    chmod +x "$SCRIPT_DIR/build/app"

    log_info "Generating documentation..."
    sleep 0.2
    echo "# Project Documentation" > "$SCRIPT_DIR/build/README.md"

    log_info "Creating archives..."
    sleep 0.2
    tar -czf "$SCRIPT_DIR/build/app.tar.gz" -C "$SCRIPT_DIR/build" app README.md

    log_info "Build complete"
}

## Create distribution package
task::package() {
    depends_on build

    log_info "Creating distribution package..."

    # Create package structure
    local pkg_dir="$SCRIPT_DIR/dist/myapp-1.0.0"
    mkdir -p "$pkg_dir"/{bin,lib,doc}

    # Copy files
    cp "$SCRIPT_DIR/build/app" "$pkg_dir/bin/"
    cp "$SCRIPT_DIR/build/README.md" "$pkg_dir/doc/"

    # Create installer script
    cat > "$pkg_dir/install.sh" << 'EOF'
#!/bin/bash
echo "Installing myapp..."
mkdir -p /usr/local/bin
cp bin/app /usr/local/bin/myapp
echo "Installation complete"
EOF
    chmod +x "$pkg_dir/install.sh"

    # Create tarball
    tar -czf "$SCRIPT_DIR/dist/myapp-1.0.0.tar.gz" -C "$SCRIPT_DIR/dist" myapp-1.0.0

    log_info "Package created: dist/myapp-1.0.0.tar.gz"
}

## Deploy to production
task::deploy() {
    depends_on package

    log_info "Deploying to production..."

    # Simulate deployment steps
    log_info "Uploading package..."
    sleep 0.5

    log_info "Running remote install..."
    sleep 0.3

    log_info "Verifying deployment..."
    sleep 0.2

    log_info "Deployment complete"
    log_success "Application deployed successfully!"
}

## Run development server (no dependencies)
task::dev() {
    log_info "Starting development server..."
    log_info "Server running at http://localhost:8000"
    log_info "Press Ctrl+C to stop"

    # Simulate server (would normally run indefinitely)
    sleep 2
    log_info "Development server stopped"
}

## Show project status and info
task::status() {
    log_info "Project Status"
    echo
    echo "  Directory:  $SCRIPT_DIR"
    echo "  Git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'not a git repo')"
    echo "  Files:      $(find "$SCRIPT_DIR" -type f | wc -l | tr -d ' ')"
    echo

    if [[ -d "$SCRIPT_DIR/dist" ]]; then
        echo "  Build artifacts:"
        ls -lh "$SCRIPT_DIR/dist" | tail -n +2 | awk '{print "    " $9 " (" $5 ")"}'
    fi
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    # Handle options
    case "${1:-}" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--list)
            echo "${BOLD}Available tasks:${NC}"
            list_tasks | while read -r task; do
                echo "  - $task"
            done
            exit 0
            ;;
        "")
            log_error "No task specified"
            echo
            show_usage
            exit 1
            ;;
    esac

    # Run all specified tasks
    local overall_success=true

    for task_name in "$@"; do
        if ! run_task "$task_name"; then
            overall_success=false
            break
        fi
    done

    echo
    if [[ "$overall_success" == true ]]; then
        log_success "All tasks completed successfully"
        exit 0
    else
        log_error "Task execution failed"
        exit 1
    fi
}

main "$@"
