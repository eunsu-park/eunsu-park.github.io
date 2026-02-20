#!/usr/bin/env bash
set -euo pipefail

# Error Handling Framework
# Demonstrates reusable error handling patterns for robust shell scripts

# ============================================================================
# Color definitions for output
# ============================================================================

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly NC='\033[0m' # No Color

# ============================================================================
# Error Codes
# ============================================================================

readonly E_SUCCESS=0
readonly E_GENERIC=1
readonly E_INVALID_ARG=2
readonly E_FILE_NOT_FOUND=3
readonly E_PERMISSION_DENIED=4
readonly E_NETWORK_ERROR=5
readonly E_TIMEOUT=6
readonly E_DEPENDENCY_MISSING=7

# ============================================================================
# Error Handling Functions
# ============================================================================

# Print error message and exit with code
die() {
    local message="$1"
    local exit_code="${2:-$E_GENERIC}"

    echo -e "${RED}ERROR:${NC} $message" >&2
    exit "$exit_code"
}

# Print warning message without exiting
warn() {
    local message="$1"
    echo -e "${YELLOW}WARNING:${NC} $message" >&2
}

# Print info message
info() {
    local message="$1"
    echo -e "${CYAN}INFO:${NC} $message"
}

# Print success message
success() {
    local message="$1"
    echo -e "${GREEN}SUCCESS:${NC} $message"
}

# ============================================================================
# Stack Trace on Error
# ============================================================================

# Print stack trace
print_stack_trace() {
    local frame=0
    local line_no
    local func_name
    local source_file

    echo -e "${MAGENTA}Stack trace:${NC}" >&2

    # Skip the first frame (this function itself)
    for ((frame=1; frame < ${#FUNCNAME[@]}; frame++)); do
        func_name="${FUNCNAME[$frame]}"
        source_file="${BASH_SOURCE[$frame]}"
        line_no="${BASH_LINENO[$((frame-1))]}"

        echo -e "  ${CYAN}[$frame]${NC} $func_name at ${source_file}:${line_no}" >&2
    done
}

# Error handler that shows context
error_handler() {
    local line_no="$1"
    local bash_lineno="$2"
    local func_name="${3:-main}"
    local command="$4"
    local error_code="${5:-1}"

    echo >&2
    echo -e "${RED}╔════════════════════════════════════════════╗${NC}" >&2
    echo -e "${RED}║            ERROR OCCURRED                  ║${NC}" >&2
    echo -e "${RED}╚════════════════════════════════════════════╝${NC}" >&2
    echo >&2
    echo -e "${RED}Error Code:${NC} $error_code" >&2
    echo -e "${RED}Function:${NC} $func_name" >&2
    echo -e "${RED}Line:${NC} $bash_lineno" >&2
    echo -e "${RED}Command:${NC} $command" >&2
    echo >&2

    print_stack_trace

    echo >&2
}

# Setup ERR trap for automatic error context
setup_error_trap() {
    set -eE  # Inherit ERR trap in functions

    # Note: BASH_COMMAND contains the command that triggered the error
    trap 'error_handler ${LINENO} ${BASH_LINENO[0]} "${FUNCNAME[1]}" "$BASH_COMMAND" $?' ERR
}

# ============================================================================
# Try/Catch Simulation Using Subshells
# ============================================================================

# Try to execute a command, catch errors
# Returns: 0 if success, error code if failure
try() {
    local error_code=0

    # Execute in subshell to isolate errors
    (
        set -e
        "$@"
    ) || error_code=$?

    return $error_code
}

# Execute catch block if try failed
catch() {
    local error_code=$?

    if [[ $error_code -ne 0 ]]; then
        "$@" "$error_code"
    fi

    return 0
}

# ============================================================================
# Demo Functions
# ============================================================================

# Function that simulates an error
function_that_fails() {
    local depth="${1:-1}"

    if [[ $depth -gt 1 ]]; then
        nested_function_call $((depth - 1))
    else
        info "About to trigger an error..."
        # This will fail and trigger error handler
        false
    fi
}

# Nested function to show stack trace
nested_function_call() {
    local depth="$1"

    if [[ $depth -gt 0 ]]; then
        nested_function_call $((depth - 1))
    else
        # Trigger error
        false
    fi
}

# Demo: Basic error handling
demo_die_and_warn() {
    echo -e "\n${BLUE}=== Die and Warn Demo ===${NC}\n"

    warn "This is a warning message - script continues"
    info "Processing continues after warning"
    success "This step succeeded"

    echo
    echo "Example: die() would exit the script"
    echo "  die \"Critical error occurred\" $E_GENERIC"
    echo
}

# Demo: Try/catch pattern
demo_try_catch() {
    echo -e "${BLUE}=== Try/Catch Pattern Demo ===${NC}\n"

    # Example 1: Successful command
    echo "Example 1: Successful command"
    if try ls /tmp > /dev/null 2>&1; then
        success "Command succeeded"
    else
        warn "Command failed (unexpected)"
    fi

    echo

    # Example 2: Failing command
    echo "Example 2: Failing command (caught)"
    if try ls /nonexistent/path > /dev/null 2>&1; then
        success "Command succeeded (unexpected)"
    else
        local error_code=$?
        warn "Command failed with exit code $error_code (expected)"
        info "Error was caught and handled gracefully"
    fi

    echo
}

# Demo: Try/catch with error handler
demo_try_catch_with_handler() {
    echo -e "${BLUE}=== Try/Catch with Error Handler ===${NC}\n"

    # Define error handler function
    handle_error() {
        local error_code="$1"
        echo -e "${RED}Caught error with code: $error_code${NC}"
        echo "Performing recovery actions..."
        echo -e "${GREEN}Recovery complete${NC}"
    }

    # Try a command that will fail
    echo "Attempting risky operation..."
    try some_nonexistent_command 2>/dev/null || catch handle_error

    echo "Script continues after error handling"
    echo
}

# Demo: Error codes
demo_error_codes() {
    echo -e "${BLUE}=== Error Code Demo ===${NC}\n"

    echo "Defined error codes:"
    echo "  E_SUCCESS=$E_SUCCESS (Success)"
    echo "  E_GENERIC=$E_GENERIC (Generic error)"
    echo "  E_INVALID_ARG=$E_INVALID_ARG (Invalid argument)"
    echo "  E_FILE_NOT_FOUND=$E_FILE_NOT_FOUND (File not found)"
    echo "  E_PERMISSION_DENIED=$E_PERMISSION_DENIED (Permission denied)"
    echo "  E_NETWORK_ERROR=$E_NETWORK_ERROR (Network error)"
    echo "  E_TIMEOUT=$E_TIMEOUT (Timeout)"
    echo "  E_DEPENDENCY_MISSING=$E_DEPENDENCY_MISSING (Missing dependency)"

    echo
    echo "Example usage:"
    echo '  [[ ! -f "$file" ]] && die "File not found: $file" $E_FILE_NOT_FOUND'
    echo
}

# Demo: Input validation with error codes
demo_input_validation() {
    echo -e "${BLUE}=== Input Validation Demo ===${NC}\n"

    validate_number() {
        local input="$1"
        local name="${2:-value}"

        if [[ ! "$input" =~ ^[0-9]+$ ]]; then
            die "Invalid $name: must be a number" $E_INVALID_ARG
        fi

        success "Valid number: $input"
    }

    validate_file() {
        local filepath="$1"

        if [[ ! -f "$filepath" ]]; then
            die "File not found: $filepath" $E_FILE_NOT_FOUND
        fi

        if [[ ! -r "$filepath" ]]; then
            die "Permission denied: $filepath" $E_PERMISSION_DENIED
        fi

        success "Valid file: $filepath"
    }

    # Test validation
    echo "Testing number validation:"
    if try validate_number "123" "test_number"; then
        info "Validation passed"
    else
        warn "Validation failed (expected)"
    fi

    echo
    echo "Testing with invalid number:"
    if try validate_number "abc" "test_number"; then
        warn "Validation passed (unexpected)"
    else
        info "Validation failed as expected"
    fi

    echo
}

# Demo: Stack trace (disabled by default to avoid script exit)
demo_stack_trace() {
    echo -e "${BLUE}=== Stack Trace Demo ===${NC}\n"

    echo "To see a stack trace, uncomment the following line in the script:"
    echo "  # function_that_fails 3"
    echo
    echo "This would call a nested function that fails,"
    echo "triggering the error handler which prints:"
    echo "  - Error code"
    echo "  - Function name"
    echo "  - Line number"
    echo "  - Failed command"
    echo "  - Full stack trace"
    echo

    # Uncomment to actually trigger error (will exit script):
    # function_that_fails 3
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       Error Handling Framework Demo       ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

    # Setup error trap (commented out to avoid script exit during demos)
    # Uncomment to enable automatic error context on failures:
    # setup_error_trap

    demo_die_and_warn
    demo_try_catch
    demo_try_catch_with_handler
    demo_error_codes
    demo_input_validation
    demo_stack_trace

    echo -e "${GREEN}=== All Error Handling Demos Complete ===${NC}\n"
}

main "$@"
