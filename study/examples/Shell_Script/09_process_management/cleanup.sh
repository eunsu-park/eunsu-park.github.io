#!/usr/bin/env bash
set -euo pipefail

# Cleanup and Signal Handling Patterns
# Demonstrates proper cleanup of resources and graceful shutdown

# ============================================================================
# Color definitions for output
# ============================================================================

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# ============================================================================
# Global Variables for Cleanup Tracking
# ============================================================================

# Track temporary files and directories
declare -a TEMP_FILES=()
declare -a TEMP_DIRS=()

# Lock file location
LOCK_FILE=""

# Background process PID
BG_PROCESS_PID=""

# ============================================================================
# Cleanup Functions
# ============================================================================

# Cleanup function called on script exit
cleanup_on_exit() {
    local exit_code=$?

    echo
    echo -e "${CYAN}=== Cleanup Handler (EXIT) ===${NC}"
    echo "Exit code: $exit_code"

    # Clean up temporary files
    if [[ ${#TEMP_FILES[@]} -gt 0 ]]; then
        echo "Removing temporary files..."
        for file in "${TEMP_FILES[@]}"; do
            if [[ -f "$file" ]]; then
                rm -f "$file"
                echo -e "  ${GREEN}✓${NC} Removed: $file"
            fi
        done
    fi

    # Clean up temporary directories
    if [[ ${#TEMP_DIRS[@]} -gt 0 ]]; then
        echo "Removing temporary directories..."
        for dir in "${TEMP_DIRS[@]}"; do
            if [[ -d "$dir" ]]; then
                rm -rf "$dir"
                echo -e "  ${GREEN}✓${NC} Removed: $dir"
            fi
        done
    fi

    # Remove lock file
    if [[ -n "$LOCK_FILE" && -f "$LOCK_FILE" ]]; then
        rm -f "$LOCK_FILE"
        echo -e "  ${GREEN}✓${NC} Released lock: $LOCK_FILE"
    fi

    # Kill background processes
    if [[ -n "$BG_PROCESS_PID" ]] && kill -0 "$BG_PROCESS_PID" 2>/dev/null; then
        echo "Terminating background process (PID $BG_PROCESS_PID)..."
        kill "$BG_PROCESS_PID" 2>/dev/null || true
        wait "$BG_PROCESS_PID" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Background process terminated"
    fi

    echo -e "${GREEN}Cleanup complete${NC}"
}

# Handle interrupt signal (Ctrl+C)
cleanup_on_interrupt() {
    echo
    echo -e "${YELLOW}=== Cleanup Handler (SIGINT) ===${NC}"
    echo "Caught interrupt signal (Ctrl+C)"
    echo "Performing graceful shutdown..."

    # Exit will trigger cleanup_on_exit
    exit 130
}

# Handle termination signal
cleanup_on_terminate() {
    echo
    echo -e "${YELLOW}=== Cleanup Handler (SIGTERM) ===${NC}"
    echo "Caught termination signal"
    echo "Performing graceful shutdown..."

    exit 143
}

# Handle hangup signal
cleanup_on_hangup() {
    echo
    echo -e "${YELLOW}=== Cleanup Handler (SIGHUP) ===${NC}"
    echo "Caught hangup signal"
    echo "Performing graceful shutdown..."

    exit 129
}

# ============================================================================
# Setup Signal Handlers
# ============================================================================

setup_traps() {
    echo -e "${CYAN}Setting up signal handlers...${NC}"

    # EXIT: Always called when script exits (success or failure)
    trap cleanup_on_exit EXIT

    # SIGINT: Interrupt from keyboard (Ctrl+C)
    trap cleanup_on_interrupt INT

    # SIGTERM: Termination signal
    trap cleanup_on_terminate TERM

    # SIGHUP: Hangup detected on controlling terminal
    trap cleanup_on_hangup HUP

    echo -e "${GREEN}✓${NC} Signal handlers installed: EXIT, INT, TERM, HUP"
    echo
}

# ============================================================================
# Resource Management Functions
# ============================================================================

# Create a temporary file and register it for cleanup
create_temp_file() {
    local prefix="${1:-temp}"
    local temp_file

    temp_file=$(mktemp "/tmp/${prefix}.XXXXXX")
    TEMP_FILES+=("$temp_file")

    echo -e "${GREEN}✓${NC} Created temp file: $temp_file"
    echo "$temp_file"
}

# Create a temporary directory and register it for cleanup
create_temp_dir() {
    local prefix="${1:-tempdir}"
    local temp_dir

    temp_dir=$(mktemp -d "/tmp/${prefix}.XXXXXX")
    TEMP_DIRS+=("$temp_dir")

    echo -e "${GREEN}✓${NC} Created temp directory: $temp_dir"
    echo "$temp_dir"
}

# Acquire a lock file
acquire_lock() {
    local lock_name="${1:-script}"
    LOCK_FILE="/tmp/${lock_name}.lock"

    if [[ -f "$LOCK_FILE" ]]; then
        local lock_pid
        lock_pid=$(cat "$LOCK_FILE")

        if kill -0 "$lock_pid" 2>/dev/null; then
            echo -e "${RED}✗${NC} Lock already held by process $lock_pid"
            return 1
        else
            echo -e "${YELLOW}⚠${NC} Stale lock file found, removing..."
            rm -f "$LOCK_FILE"
        fi
    fi

    echo $$ > "$LOCK_FILE"
    echo -e "${GREEN}✓${NC} Lock acquired: $LOCK_FILE"
    return 0
}

# Start a background process that will be cleaned up
start_background_process() {
    echo "Starting background process..."

    (
        echo "Background process started (PID $$)"
        for i in {1..30}; do
            echo "  [BG] Working... ($i/30)"
            sleep 1
        done
        echo "Background process completed"
    ) &

    BG_PROCESS_PID=$!
    echo -e "${GREEN}✓${NC} Background process started (PID $BG_PROCESS_PID)"
}

# ============================================================================
# Demo Functions
# ============================================================================

demo_temp_file_cleanup() {
    echo -e "${BLUE}=== Temporary File Management Demo ===${NC}\n"

    # Create several temp files
    local file1 file2 file3
    file1=$(create_temp_file "demo1")
    file2=$(create_temp_file "demo2")
    file3=$(create_temp_file "demo3")

    # Write some data
    echo "Data in file 1" > "$file1"
    echo "Data in file 2" > "$file2"
    echo "Data in file 3" > "$file3"

    echo
    echo "Temporary files created and will be cleaned up on exit"
    echo "Total temp files: ${#TEMP_FILES[@]}"
    echo
}

demo_temp_dir_cleanup() {
    echo -e "${BLUE}=== Temporary Directory Management Demo ===${NC}\n"

    # Create temp directory
    local temp_dir
    temp_dir=$(create_temp_dir "workspace")

    # Create some files in it
    touch "$temp_dir/file1.txt"
    touch "$temp_dir/file2.txt"
    mkdir -p "$temp_dir/subdir"
    touch "$temp_dir/subdir/file3.txt"

    echo "Created directory structure:"
    ls -R "$temp_dir" | sed 's/^/  /'
    echo
}

demo_lock_file() {
    echo -e "${BLUE}=== Lock File Demo ===${NC}\n"

    if acquire_lock "cleanup_demo"; then
        echo "Performing work while holding lock..."
        sleep 1
        echo -e "${GREEN}✓${NC} Work completed"
    else
        echo -e "${RED}✗${NC} Could not acquire lock"
        return 1
    fi
    echo
}

demo_background_process_cleanup() {
    echo -e "${BLUE}=== Background Process Cleanup Demo ===${NC}\n"

    start_background_process

    echo
    echo "Background process running..."
    echo "Script will clean it up on exit"
    sleep 2
    echo
}

demo_graceful_shutdown() {
    echo -e "${BLUE}=== Graceful Shutdown Demo ===${NC}\n"

    echo "This demo shows cleanup on interrupt"
    echo -e "${YELLOW}Press Ctrl+C to trigger graceful shutdown${NC}"
    echo "Or wait 5 seconds for normal exit..."
    echo

    # Simulate long-running work
    for i in {1..5}; do
        echo "Working... ($i/5)"
        sleep 1
    done

    echo -e "${GREEN}✓${NC} Work completed normally"
    echo
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     Cleanup & Signal Handling Demo        ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
    echo

    # Setup cleanup handlers
    setup_traps

    # Run demos
    demo_temp_file_cleanup
    demo_temp_dir_cleanup
    demo_lock_file
    demo_background_process_cleanup
    demo_graceful_shutdown

    echo -e "${GREEN}=== Demo Complete ===${NC}"
    echo "Cleanup will run automatically on exit..."
    echo

    # EXIT trap will handle cleanup
}

main "$@"
