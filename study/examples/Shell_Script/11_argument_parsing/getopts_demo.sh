#!/usr/bin/env bash
set -euo pipefail

# Getopts Demonstration
# Shows POSIX-compliant argument parsing with getopts

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
# Default Values
# ============================================================================

VERBOSE=false
OUTPUT_FILE=""
COUNT=1
SHOW_HELP=false

# ============================================================================
# Usage Function
# ============================================================================

usage() {
    cat << EOF
${BLUE}USAGE:${NC}
    $(basename "$0") [OPTIONS] [ARGUMENTS...]

${BLUE}DESCRIPTION:${NC}
    Demonstrates POSIX getopts argument parsing.
    Processes options and remaining positional arguments.

${BLUE}OPTIONS:${NC}
    -v              Enable verbose output
    -o FILE         Specify output file
    -n COUNT        Set count value (default: 1)
    -h              Show this help message

${BLUE}EXAMPLES:${NC}
    $(basename "$0") -v file1 file2
    $(basename "$0") -o output.txt -n 5 input.txt
    $(basename "$0") -vn 3 -o result.log data/*

${BLUE}EXIT CODES:${NC}
    0   Success
    1   Invalid option or missing argument
    2   Invalid argument value

EOF
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_arguments() {
    local OPTIND OPTARG opt

    # getopts format: "vho:n:"
    # - Letters without colon: flag options (no argument)
    # - Letters with colon: options that require an argument
    while getopts "vho:n:" opt; do
        case "$opt" in
            v)
                VERBOSE=true
                ;;
            h)
                SHOW_HELP=true
                ;;
            o)
                OUTPUT_FILE="$OPTARG"
                ;;
            n)
                if [[ ! "$OPTARG" =~ ^[0-9]+$ ]]; then
                    echo -e "${RED}Error:${NC} -n requires a numeric argument" >&2
                    usage
                    exit 2
                fi
                COUNT="$OPTARG"
                ;;
            \?)
                echo -e "${RED}Error:${NC} Invalid option: -$OPTARG" >&2
                usage
                exit 1
                ;;
            :)
                echo -e "${RED}Error:${NC} Option -$OPTARG requires an argument" >&2
                usage
                exit 1
                ;;
        esac
    done

    # Shift processed options
    shift $((OPTIND - 1))

    # Remaining arguments are positional
    POSITIONAL_ARGS=("$@")
}

# ============================================================================
# Display Configuration
# ============================================================================

display_config() {
    echo -e "${CYAN}=== Configuration ===${NC}"
    echo

    echo "Options parsed:"
    echo "  Verbose:     $VERBOSE"
    echo "  Output file: ${OUTPUT_FILE:-<not specified>}"
    echo "  Count:       $COUNT"
    echo

    if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
        echo "Positional arguments (${#POSITIONAL_ARGS[@]}):"
        for i in "${!POSITIONAL_ARGS[@]}"; do
            echo "  [$((i+1))] ${POSITIONAL_ARGS[$i]}"
        done
    else
        echo "No positional arguments provided"
    fi

    echo
}

# ============================================================================
# Processing Function
# ============================================================================

process_data() {
    echo -e "${CYAN}=== Processing ===${NC}"
    echo

    # Verbose logging
    verbose_log() {
        if [[ "$VERBOSE" == true ]]; then
            echo -e "${YELLOW}[VERBOSE]${NC} $*"
        fi
    }

    verbose_log "Starting processing with count=$COUNT"

    # Process each positional argument
    if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
        for arg in "${POSITIONAL_ARGS[@]}"; do
            verbose_log "Processing: $arg"

            for ((i=1; i<=COUNT; i++)); do
                echo "  Processing '$arg' (iteration $i/$COUNT)"

                if [[ "$VERBOSE" == true ]]; then
                    sleep 0.1  # Small delay in verbose mode to show progress
                fi
            done
        done
    else
        echo "  No input files to process"
    fi

    echo

    # Handle output file
    if [[ -n "$OUTPUT_FILE" ]]; then
        verbose_log "Writing results to: $OUTPUT_FILE"

        {
            echo "# Processing Results"
            echo "# Generated: $(date)"
            echo "# Count: $COUNT"
            echo
            echo "Processed items:"
            for arg in "${POSITIONAL_ARGS[@]}"; do
                echo "  - $arg"
            done
        } > "$OUTPUT_FILE"

        echo -e "${GREEN}✓${NC} Results written to: $OUTPUT_FILE"
    else
        verbose_log "No output file specified, results not saved"
    fi

    echo
}

# ============================================================================
# Demo Examples
# ============================================================================

run_demo_examples() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║         Getopts Demo Examples              ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
    echo

    local script_name
    script_name=$(basename "$0")

    echo "This script demonstrates getopts argument parsing."
    echo
    echo -e "${CYAN}Try these examples:${NC}"
    echo
    echo "1. Basic usage with verbose flag:"
    echo "   ./$script_name -v file1.txt file2.txt"
    echo
    echo "2. Specify output file and count:"
    echo "   ./$script_name -o output.log -n 3 input.txt"
    echo
    echo "3. Combined short options:"
    echo "   ./$script_name -vn 5 -o result.txt data/*"
    echo
    echo "4. Show help:"
    echo "   ./$script_name -h"
    echo
    echo "5. Test error handling (invalid option):"
    echo "   ./$script_name -x"
    echo
    echo "6. Test error handling (missing argument):"
    echo "   ./$script_name -o"
    echo
    echo "7. Test error handling (invalid count):"
    echo "   ./$script_name -n abc"
    echo

    echo -e "${YELLOW}Note:${NC} Running demo with default behavior..."
    echo
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Parse command-line arguments
    declare -a POSITIONAL_ARGS=()
    parse_arguments "$@"

    # Show help if requested
    if [[ "$SHOW_HELP" == true ]]; then
        usage
        exit 0
    fi

    # If no arguments provided, show demo examples
    if [[ $# -eq 0 ]]; then
        run_demo_examples
        echo "Continuing with demo execution..."
        echo

        # Set demo values
        VERBOSE=true
        COUNT=2
        POSITIONAL_ARGS=("demo_file1.txt" "demo_file2.txt" "demo_file3.txt")
    fi

    # Display parsed configuration
    display_config

    # Process data based on arguments
    process_data

    echo -e "${GREEN}=== Processing Complete ===${NC}"
    echo
}

main "$@"
