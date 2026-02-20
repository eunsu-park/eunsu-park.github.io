#!/usr/bin/env bash
set -euo pipefail

# Test Runner Script
# Runs Bats tests and ShellCheck validation with summary reporting

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SHELLCHECK_ERRORS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================================
# Check Dependencies
# ============================================================================

check_dependencies() {
    print_header "Checking Dependencies"

    local all_deps_ok=true

    # Check for bats
    if command -v bats &> /dev/null; then
        local bats_version
        bats_version=$(bats --version | head -n1)
        print_success "Bats found: $bats_version"
    else
        print_error "Bats not found"
        echo "  Install: npm install -g bats"
        echo "  Or: brew install bats-core (macOS)"
        echo "  Or: apt install bats (Ubuntu/Debian)"
        all_deps_ok=false
    fi

    # Check for shellcheck
    if command -v shellcheck &> /dev/null; then
        local shellcheck_version
        shellcheck_version=$(shellcheck --version | grep version: | awk '{print $2}')
        print_success "ShellCheck found: version $shellcheck_version"
    else
        print_warning "ShellCheck not found (optional)"
        echo "  Install: brew install shellcheck (macOS)"
        echo "  Or: apt install shellcheck (Ubuntu/Debian)"
    fi

    if [[ "$all_deps_ok" == false ]]; then
        echo
        print_error "Missing required dependencies"
        exit 1
    fi

    echo
}

# ============================================================================
# Run Bats Tests
# ============================================================================

run_bats_tests() {
    print_header "Running Bats Tests"

    local test_files=("$SCRIPT_DIR"/*.bats)

    if [[ ${#test_files[@]} -eq 0 ]] || [[ ! -f "${test_files[0]}" ]]; then
        print_warning "No .bats test files found"
        return
    fi

    # Run bats with TAP output
    local bats_output
    local bats_exit_code=0

    for test_file in "${test_files[@]}"; do
        echo -e "${BLUE}Running: $(basename "$test_file")${NC}"
        echo

        # Capture output and exit code
        if bats_output=$(bats --tap "$test_file" 2>&1); then
            echo "$bats_output"
        else
            bats_exit_code=$?
            echo "$bats_output"
        fi

        # Parse TAP output for statistics
        local tests_in_file
        local passed_in_file
        local failed_in_file

        tests_in_file=$(echo "$bats_output" | grep -c "^ok\|^not ok" || true)
        passed_in_file=$(echo "$bats_output" | grep -c "^ok" || true)
        failed_in_file=$(echo "$bats_output" | grep -c "^not ok" || true)

        TOTAL_TESTS=$((TOTAL_TESTS + tests_in_file))
        PASSED_TESTS=$((PASSED_TESTS + passed_in_file))
        FAILED_TESTS=$((FAILED_TESTS + failed_in_file))

        echo
    done

    # Print summary
    if [[ $FAILED_TESTS -eq 0 ]]; then
        print_success "All Bats tests passed: $PASSED_TESTS/$TOTAL_TESTS"
    else
        print_error "Some Bats tests failed: $PASSED_TESTS passed, $FAILED_TESTS failed"
    fi

    echo
}

# ============================================================================
# Run ShellCheck
# ============================================================================

run_shellcheck() {
    print_header "Running ShellCheck"

    if ! command -v shellcheck &> /dev/null; then
        print_warning "ShellCheck not installed, skipping"
        return
    fi

    local shell_files
    shell_files=$(find "$SCRIPT_DIR" -name "*.sh" -type f)

    if [[ -z "$shell_files" ]]; then
        print_warning "No .sh files found"
        return
    fi

    local has_errors=false

    while IFS= read -r file; do
        echo -e "${BLUE}Checking: $(basename "$file")${NC}"

        if shellcheck_output=$(shellcheck "$file" 2>&1); then
            print_success "No issues found"
        else
            has_errors=true
            SHELLCHECK_ERRORS=$((SHELLCHECK_ERRORS + 1))
            print_error "Issues found:"
            echo "$shellcheck_output" | sed 's/^/  /'
        fi

        echo
    done <<< "$shell_files"

    # Summary
    if [[ "$has_errors" == false ]]; then
        print_success "All ShellCheck tests passed"
    else
        print_error "ShellCheck found issues in $SHELLCHECK_ERRORS file(s)"
    fi

    echo
}

# ============================================================================
# Print Final Summary
# ============================================================================

print_summary() {
    print_header "Test Summary"

    echo "Bats Tests:"
    echo "  Total:  $TOTAL_TESTS"
    echo "  Passed: $PASSED_TESTS"
    echo "  Failed: $FAILED_TESTS"
    echo

    if command -v shellcheck &> /dev/null; then
        echo "ShellCheck:"
        if [[ $SHELLCHECK_ERRORS -eq 0 ]]; then
            echo "  Status: All files passed"
        else
            echo "  Status: $SHELLCHECK_ERRORS file(s) with issues"
        fi
        echo
    fi

    # Overall status
    if [[ $FAILED_TESTS -eq 0 ]] && [[ $SHELLCHECK_ERRORS -eq 0 ]]; then
        print_success "All tests passed!"
        return 0
    else
        print_error "Some tests failed"
        return 1
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo "Test Runner for Math Library"
    echo "============================="

    check_dependencies
    run_bats_tests
    run_shellcheck
    print_summary
}

# Run main and exit with appropriate code
if main; then
    exit 0
else
    exit 1
fi
