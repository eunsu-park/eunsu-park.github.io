#!/usr/bin/env bash
set -euo pipefail

# Safe Operations Framework
# Demonstrates safe file and command operations with proper error handling

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
# Safe Operation Functions
# ============================================================================

# Safe cd: Change directory with error checking
safe_cd() {
    local target_dir="$1"

    if [[ ! -d "$target_dir" ]]; then
        echo -e "${RED}✗${NC} Directory does not exist: $target_dir" >&2
        return 1
    fi

    if ! cd "$target_dir"; then
        echo -e "${RED}✗${NC} Failed to change directory to: $target_dir" >&2
        return 1
    fi

    echo -e "${GREEN}✓${NC} Changed directory to: $target_dir"
    return 0
}

# Safe rm: Remove with confirmation and safeguards
safe_rm() {
    local target="$1"
    local force="${2:-false}"

    # Check if target exists
    if [[ ! -e "$target" ]]; then
        echo -e "${YELLOW}⚠${NC} Target does not exist: $target"
        return 1
    fi

    # Prevent dangerous operations
    if [[ "$target" == "/" ]] || [[ "$target" == "/home" ]] || [[ "$target" == "/usr" ]]; then
        echo -e "${RED}✗${NC} Refusing to remove protected directory: $target" >&2
        return 1
    fi

    # Ask for confirmation unless force mode
    if [[ "$force" != "true" ]]; then
        local file_type="file"
        [[ -d "$target" ]] && file_type="directory"

        echo -n "Remove $file_type '$target'? [y/N] "
        read -r response

        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}⚠${NC} Operation cancelled"
            return 1
        fi
    fi

    # Perform removal
    if [[ -d "$target" ]]; then
        rm -rf "$target"
    else
        rm -f "$target"
    fi

    echo -e "${GREEN}✓${NC} Removed: $target"
    return 0
}

# Safe write: Atomic write operation (write to temp, then move)
safe_write() {
    local target_file="$1"
    local content="$2"

    # Create temp file in same directory as target
    local target_dir
    target_dir=$(dirname "$target_file")

    if [[ ! -d "$target_dir" ]]; then
        echo -e "${RED}✗${NC} Target directory does not exist: $target_dir" >&2
        return 1
    fi

    local temp_file
    temp_file=$(mktemp "${target_dir}/.tmp.XXXXXX")

    # Write to temp file
    if ! echo "$content" > "$temp_file"; then
        echo -e "${RED}✗${NC} Failed to write to temp file" >&2
        rm -f "$temp_file"
        return 1
    fi

    # Atomic move (overwrites target)
    if ! mv "$temp_file" "$target_file"; then
        echo -e "${RED}✗${NC} Failed to move temp file to target" >&2
        rm -f "$temp_file"
        return 1
    fi

    echo -e "${GREEN}✓${NC} Safely wrote to: $target_file"
    return 0
}

# Require command: Check if required command exists
require_cmd() {
    local cmd="$1"
    local install_hint="${2:-}"

    if ! command -v "$cmd" &> /dev/null; then
        echo -e "${RED}✗${NC} Required command not found: $cmd" >&2

        if [[ -n "$install_hint" ]]; then
            echo -e "${CYAN}Hint:${NC} $install_hint" >&2
        fi

        return 1
    fi

    echo -e "${GREEN}✓${NC} Command available: $cmd"
    return 0
}

# Retry: Retry a command with exponential backoff
retry() {
    local max_attempts="${1:-3}"
    local initial_delay="${2:-1}"
    shift 2
    local command=("$@")

    local attempt=1
    local delay=$initial_delay

    while [[ $attempt -le $max_attempts ]]; do
        echo -e "${CYAN}Attempt $attempt/$max_attempts:${NC} ${command[*]}"

        if "${command[@]}"; then
            echo -e "${GREEN}✓${NC} Command succeeded on attempt $attempt"
            return 0
        fi

        if [[ $attempt -lt $max_attempts ]]; then
            echo -e "${YELLOW}⚠${NC} Command failed, retrying in ${delay}s..."
            sleep "$delay"

            # Exponential backoff
            delay=$((delay * 2))
        fi

        ((attempt++))
    done

    echo -e "${RED}✗${NC} Command failed after $max_attempts attempts" >&2
    return 1
}

# Safe copy: Copy with verification
safe_cp() {
    local source="$1"
    local dest="$2"

    # Check source exists
    if [[ ! -e "$source" ]]; then
        echo -e "${RED}✗${NC} Source does not exist: $source" >&2
        return 1
    fi

    # Check destination directory exists
    local dest_dir
    if [[ -d "$dest" ]]; then
        dest_dir="$dest"
    else
        dest_dir=$(dirname "$dest")
    fi

    if [[ ! -d "$dest_dir" ]]; then
        echo -e "${RED}✗${NC} Destination directory does not exist: $dest_dir" >&2
        return 1
    fi

    # Perform copy
    if ! cp -r "$source" "$dest"; then
        echo -e "${RED}✗${NC} Failed to copy: $source → $dest" >&2
        return 1
    fi

    echo -e "${GREEN}✓${NC} Copied: $source → $dest"
    return 0
}

# Create directory with parents and error checking
safe_mkdir() {
    local dir_path="$1"

    if [[ -e "$dir_path" ]]; then
        if [[ -d "$dir_path" ]]; then
            echo -e "${YELLOW}⚠${NC} Directory already exists: $dir_path"
            return 0
        else
            echo -e "${RED}✗${NC} Path exists but is not a directory: $dir_path" >&2
            return 1
        fi
    fi

    if ! mkdir -p "$dir_path"; then
        echo -e "${RED}✗${NC} Failed to create directory: $dir_path" >&2
        return 1
    fi

    echo -e "${GREEN}✓${NC} Created directory: $dir_path"
    return 0
}

# ============================================================================
# Demo Functions
# ============================================================================

demo_safe_cd() {
    echo -e "\n${BLUE}=== Safe CD Demo ===${NC}\n"

    local original_dir=$PWD

    # Test 1: Valid directory
    echo "Test 1: Change to /tmp (should succeed)"
    safe_cd /tmp
    echo "  Current directory: $PWD"
    echo

    # Test 2: Invalid directory
    echo "Test 2: Change to /nonexistent (should fail)"
    safe_cd /nonexistent || echo -e "  ${CYAN}Handled error gracefully${NC}"
    echo "  Current directory: $PWD"
    echo

    # Restore original directory
    cd "$original_dir"
}

demo_safe_rm() {
    echo -e "${BLUE}=== Safe RM Demo ===${NC}\n"

    local test_dir="/tmp/safe_rm_test"
    safe_mkdir "$test_dir"

    # Create test files
    echo "Creating test files..."
    touch "$test_dir/file1.txt"
    touch "$test_dir/file2.txt"
    echo

    # Test 1: Remove with confirmation (force mode for demo)
    echo "Test 1: Remove file (force mode)"
    safe_rm "$test_dir/file1.txt" true
    echo

    # Test 2: Try to remove non-existent
    echo "Test 2: Remove non-existent file"
    safe_rm "$test_dir/nonexistent.txt" true || echo -e "  ${CYAN}Handled error gracefully${NC}"
    echo

    # Test 3: Try to remove protected directory
    echo "Test 3: Try to remove protected directory"
    safe_rm "/" true || echo -e "  ${CYAN}Protection worked!${NC}"
    echo

    # Cleanup
    rm -rf "$test_dir"
}

demo_safe_write() {
    echo -e "${BLUE}=== Safe Write Demo ===${NC}\n"

    local test_file="/tmp/safe_write_test.txt"

    # Test 1: Normal write
    echo "Test 1: Write to file"
    safe_write "$test_file" "Hello, World!"
    echo "  Content: $(cat "$test_file")"
    echo

    # Test 2: Overwrite existing file
    echo "Test 2: Overwrite existing file"
    safe_write "$test_file" "Updated content"
    echo "  Content: $(cat "$test_file")"
    echo

    # Test 3: Write to non-existent directory
    echo "Test 3: Write to non-existent directory"
    safe_write "/nonexistent/dir/file.txt" "Test" || echo -e "  ${CYAN}Handled error gracefully${NC}"
    echo

    # Cleanup
    rm -f "$test_file"
}

demo_require_cmd() {
    echo -e "${BLUE}=== Require Command Demo ===${NC}\n"

    # Test with existing commands
    echo "Test 1: Check for 'bash'"
    require_cmd bash
    echo

    echo "Test 2: Check for 'ls'"
    require_cmd ls
    echo

    # Test with non-existent command
    echo "Test 3: Check for non-existent command"
    require_cmd nonexistent_command_xyz "Install with: apt-get install xyz" || \
        echo -e "  ${CYAN}Handled missing dependency${NC}"
    echo
}

demo_retry() {
    echo -e "${BLUE}=== Retry Demo ===${NC}\n"

    # Create a command that fails 2 times then succeeds
    local attempt_file="/tmp/retry_attempt.txt"
    echo "0" > "$attempt_file"

    failing_command() {
        local attempts
        attempts=$(cat "$attempt_file")
        attempts=$((attempts + 1))
        echo "$attempts" > "$attempt_file"

        if [[ $attempts -lt 3 ]]; then
            echo "  Simulating failure..."
            return 1
        else
            echo "  Success!"
            return 0
        fi
    }

    echo "Test: Command that fails twice then succeeds"
    retry 5 1 failing_command
    echo

    # Cleanup
    rm -f "$attempt_file"

    # Test command that always fails
    echo "Test: Command that always fails"
    retry 3 1 false || echo -e "  ${CYAN}All retries exhausted${NC}"
    echo
}

demo_safe_operations_combined() {
    echo -e "${BLUE}=== Combined Safe Operations Demo ===${NC}\n"

    local work_dir="/tmp/safe_ops_demo"

    echo "Creating workspace..."
    safe_mkdir "$work_dir"

    echo "Writing configuration file..."
    safe_write "$work_dir/config.txt" "app_name=demo\nversion=1.0"

    echo "Creating subdirectory..."
    safe_mkdir "$work_dir/data"

    echo "Copying file..."
    safe_cp "$work_dir/config.txt" "$work_dir/data/"

    echo "Listing workspace:"
    ls -R "$work_dir" | sed 's/^/  /'

    echo
    echo "Cleaning up..."
    safe_rm "$work_dir" true

    echo
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║        Safe Operations Demo                ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

    demo_safe_cd
    demo_safe_rm
    demo_safe_write
    demo_require_cmd
    demo_retry
    demo_safe_operations_combined

    echo -e "${GREEN}=== All Safe Operations Demos Complete ===${NC}\n"
}

main "$@"
