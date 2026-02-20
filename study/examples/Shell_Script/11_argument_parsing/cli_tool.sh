#!/usr/bin/env bash
set -euo pipefail

# Professional CLI Tool
# Demonstrates a complete command-line interface with progress indicators,
# colored output, and both short/long option parsing

# ============================================================================
# Script Metadata
# ============================================================================

readonly SCRIPT_NAME=$(basename "$0")
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_AUTHOR="Demo Author"

# ============================================================================
# Color definitions for output
# ============================================================================

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly BOLD='\033[1m'
readonly DIM='\033[2m'
readonly NC='\033[0m' # No Color

# ============================================================================
# Configuration Variables
# ============================================================================

VERBOSE=false
QUIET=false
OUTPUT_FILE=""
INPUT_DIR="."
DRY_RUN=false
FORCE=false
SHOW_PROGRESS=true

# ============================================================================
# Logging Functions
# ============================================================================

log_info() {
    [[ "$QUIET" == true ]] && return
    echo -e "${CYAN}ℹ${NC} $*"
}

log_success() {
    [[ "$QUIET" == true ]] && return
    echo -e "${GREEN}✓${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $*" >&2
}

log_error() {
    echo -e "${RED}✗${NC} $*" >&2
}

log_verbose() {
    [[ "$VERBOSE" == true ]] || return
    echo -e "${DIM}[DEBUG]${NC} $*"
}

# ============================================================================
# Progress Indicators
# ============================================================================

# Show a progress bar
# Usage: show_progress <current> <total> <label>
show_progress() {
    local current=$1
    local total=$2
    local label="${3:-Processing}"

    [[ "$SHOW_PROGRESS" == false ]] && return
    [[ "$QUIET" == true ]] && return

    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))

    # Build progress bar
    local bar=""
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=0; i<empty; i++)); do bar+="░"; done

    # Print progress (use \r to overwrite previous line)
    printf "\r${CYAN}%s:${NC} [%s] %3d%% (%d/%d)" \
        "$label" "$bar" "$percent" "$current" "$total"

    # Add newline on completion
    [[ $current -eq $total ]] && echo
}

# Show a spinner for long operations
# Usage: run_with_spinner <command> [args...]
run_with_spinner() {
    [[ "$SHOW_PROGRESS" == false ]] && {
        "$@"
        return
    }

    [[ "$QUIET" == true ]] && {
        "$@" 2>&1 | cat > /dev/null
        return
    }

    local pid
    local spin='-\|/'
    local i=0

    # Run command in background
    "$@" &> /dev/null &
    pid=$!

    # Show spinner while command runs
    while kill -0 "$pid" 2>/dev/null; do
        i=$(((i+1) % 4))
        printf "\r${CYAN}${spin:$i:1}${NC} Working..."
        sleep 0.1
    done

    # Clear spinner
    printf "\r"

    # Check exit code
    wait "$pid"
    return $?
}

# ============================================================================
# Usage and Help
# ============================================================================

usage() {
    cat << EOF
${BOLD}${BLUE}$SCRIPT_NAME${NC} - Professional CLI Tool Demo

${BOLD}USAGE:${NC}
    $SCRIPT_NAME [OPTIONS] [FILES...]

${BOLD}DESCRIPTION:${NC}
    A demonstration of a professional command-line tool with:
    - Short and long option support
    - Colored output and progress indicators
    - Verbose/quiet modes
    - Dry-run capability

${BOLD}OPTIONS:${NC}
    ${GREEN}-h, --help${NC}              Show this help message
    ${GREEN}-v, --verbose${NC}           Enable verbose output
    ${GREEN}-q, --quiet${NC}             Suppress all output except errors
    ${GREEN}-V, --version${NC}           Show version information
    ${GREEN}-o, --output FILE${NC}       Write output to FILE
    ${GREEN}-i, --input-dir DIR${NC}     Set input directory (default: .)
    ${GREEN}-n, --dry-run${NC}           Show what would be done without doing it
    ${GREEN}-f, --force${NC}             Force operation without confirmation
    ${GREEN}--no-progress${NC}           Disable progress indicators

${BOLD}EXAMPLES:${NC}
    ${DIM}# Process files with verbose output${NC}
    $SCRIPT_NAME -v file1.txt file2.txt

    ${DIM}# Dry-run with output file${NC}
    $SCRIPT_NAME --dry-run --output result.log *.txt

    ${DIM}# Quiet mode with custom input directory${NC}
    $SCRIPT_NAME -q -i /path/to/data

    ${DIM}# Force operation without confirmation${NC}
    $SCRIPT_NAME --force --output result.txt input/*

${BOLD}EXIT CODES:${NC}
    ${GREEN}0${NC}   Success
    ${RED}1${NC}   General error
    ${RED}2${NC}   Invalid arguments

EOF
}

show_version() {
    cat << EOF
${BOLD}$SCRIPT_NAME${NC} version ${GREEN}$SCRIPT_VERSION${NC}
Written by $SCRIPT_AUTHOR
EOF
}

# ============================================================================
# Argument Parsing (Long and Short Options)
# ============================================================================

parse_arguments() {
    local positional_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -V|--version)
                show_version
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                QUIET=false
                shift
                ;;
            -q|--quiet)
                QUIET=true
                VERBOSE=false
                SHOW_PROGRESS=false
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            --no-progress)
                SHOW_PROGRESS=false
                shift
                ;;
            -o|--output)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option $1 requires an argument"
                    exit 2
                fi
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -i|--input-dir)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option $1 requires an argument"
                    exit 2
                fi
                INPUT_DIR="$2"
                shift 2
                ;;
            --)
                shift
                positional_args+=("$@")
                break
                ;;
            -*)
                log_error "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 2
                ;;
            *)
                positional_args+=("$1")
                shift
                ;;
        esac
    done

    # Set global positional arguments
    POSITIONAL_ARGS=("${positional_args[@]}")
}

# ============================================================================
# Business Logic
# ============================================================================

validate_config() {
    log_verbose "Validating configuration..."

    # Check input directory
    if [[ ! -d "$INPUT_DIR" ]]; then
        log_error "Input directory does not exist: $INPUT_DIR"
        return 1
    fi

    # Check output file parent directory
    if [[ -n "$OUTPUT_FILE" ]]; then
        local output_dir
        output_dir=$(dirname "$OUTPUT_FILE")

        if [[ ! -d "$output_dir" ]]; then
            log_error "Output directory does not exist: $output_dir"
            return 1
        fi
    fi

    log_verbose "Configuration is valid"
    return 0
}

display_config() {
    log_verbose "=== Configuration ==="
    log_verbose "Verbose:     $VERBOSE"
    log_verbose "Quiet:       $QUIET"
    log_verbose "Dry-run:     $DRY_RUN"
    log_verbose "Force:       $FORCE"
    log_verbose "Input dir:   $INPUT_DIR"
    log_verbose "Output file: ${OUTPUT_FILE:-<none>}"
    log_verbose "Files:       ${#POSITIONAL_ARGS[@]}"
}

process_files() {
    local -a files=("${POSITIONAL_ARGS[@]}")

    # If no files specified, use demo files
    if [[ ${#files[@]} -eq 0 ]]; then
        log_info "No files specified, using demo mode"
        files=("file1.txt" "file2.txt" "file3.txt" "file4.txt" "file5.txt")
    fi

    local total=${#files[@]}
    local current=0
    local results=()

    log_info "Processing $total file(s)..."

    [[ "$DRY_RUN" == true ]] && log_warning "DRY-RUN MODE: No actual changes will be made"

    for file in "${files[@]}"; do
        ((current++))

        log_verbose "Processing file $current/$total: $file"

        # Show progress bar
        show_progress "$current" "$total" "Progress"

        # Simulate processing
        if [[ "$DRY_RUN" == false ]]; then
            sleep 0.2  # Simulate work
        fi

        results+=("Processed: $file")
    done

    echo  # Newline after progress bar

    # Save results if output file specified
    if [[ -n "$OUTPUT_FILE" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            log_info "Would write results to: $OUTPUT_FILE"
        else
            log_info "Writing results to: $OUTPUT_FILE"

            {
                echo "# Processing Results"
                echo "# Generated: $(date)"
                echo "# Total files: $total"
                echo
                for result in "${results[@]}"; do
                    echo "$result"
                done
            } > "$OUTPUT_FILE"

            log_success "Results saved to: $OUTPUT_FILE"
        fi
    fi

    log_success "Processing complete! ($total file(s))"
}

# Simulate a long-running operation with spinner
simulate_long_operation() {
    log_info "Running long operation..."

    if run_with_spinner sleep 2; then
        log_success "Long operation completed"
    else
        log_error "Long operation failed"
        return 1
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Parse command-line arguments
    declare -a POSITIONAL_ARGS=()
    parse_arguments "$@"

    # Show banner
    if [[ "$QUIET" == false ]]; then
        echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════╗${NC}"
        echo -e "${BOLD}${BLUE}║      Professional CLI Tool Demo           ║${NC}"
        echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════╝${NC}"
        echo
    fi

    # Display configuration
    display_config

    # Validate configuration
    if ! validate_config; then
        log_error "Configuration validation failed"
        exit 1
    fi

    # Run long operation demo
    simulate_long_operation

    # Process files
    process_files

    echo
    log_success "All operations completed successfully!"
}

main "$@"
